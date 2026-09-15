# -*- coding: utf-8 -*-
"""
VLM 裁判：候选缺陷合规性审核（管线第三级"脑"）
==============================================
角色：YOLO-World 定位出的候选框，交由云端多模态大模型结合 RAG 规则
逐个审核 —— 判断"该区域是否为规则文档定义的真缺陷"，并纠正缺陷类型、
过滤误报（如污渍 vs 规则定义的裂纹）。

未配置视觉大模型 Key（离线模式）时，自动降级为启发式裁判：
利用框内区域与外围邻域的对比度显著性做真伪过滤（方向无关，
兼容亮缺陷/暗缺陷），不调用任何 API。

单次调用失败（网络/限流）不致命：保留该候选并沿用定位器给出的类型。
"""
from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from app.config import settings
from app.services.llm.base import BaseLLM, parse_json_from_text

_JUDGE_SYSTEM = (
    "你是严谨的工业表面缺陷质检专家。只依据给定的检测规则判定，"
    "不臆造规则中不存在的结论。"
)


async def judge_candidates(
    llm: BaseLLM,
    image: Image.Image,
    candidates: List[Dict[str, Any]],
    rule_context: str,
    known_defect_types: List[str],
) -> List[Dict[str, Any]]:
    """
    用视觉大模型逐个审核候选框（原地更新候选字段）。

    :param candidates: [{"defect_type","confidence","bbox","description",...}]
                       （bbox 为原图坐标）
    :return: 审核通过的候选列表（含 VLM 纠正后的类型/置信度/描述）
    """
    width, height = image.size
    kept: List[Dict[str, Any]] = []
    for cand in candidates[: settings.judge_max_candidates]:
        crop_b64 = _crop_to_jpeg_b64(image, cand["bbox"], width, height)
        verdict = await _judge_one(
            llm, crop_b64, cand, rule_context, known_defect_types
        )
        if verdict is None:
            # 模型输出无法解析：保守保留定位器结果
            kept.append(cand)
            continue
        if verdict.get("is_defect"):
            cand["defect_type"] = str(verdict.get("defect_type") or cand["defect_type"])[:20]
            try:
                vconf = float(verdict.get("confidence", cand["confidence"]))
            except (TypeError, ValueError):
                vconf = cand["confidence"]
            # 定位置信度与判定置信度加权融合，防止单方虚高
            cand["confidence"] = round(
                min(1.0, 0.5 * float(cand["confidence"]) + 0.5 * max(0.0, min(1.0, vconf))),
                3,
            )
            desc = str(verdict.get("description") or "").strip()
            if desc:
                cand["description"] = desc[:200]
            kept.append(cand)
    return kept


async def _judge_one(
    llm: BaseLLM,
    crop_b64: str,
    cand: Dict[str, Any],
    rule_context: str,
    known_defect_types: List[str],
) -> Optional[Dict[str, Any]]:
    """对单个候选区域发起审核，返回解析后的判定 dict 或 None。"""
    prompt = (
        "质检系统已用定位模型在工业品表面框出一处疑似缺陷（见图片中央区域）。\n"
        f"定位器初步判断类型：{cand['defect_type']}。\n"
        f"已知缺陷类型：{', '.join(known_defect_types) or '气孔、划痕、起层、缺肉、裂纹'}。\n"
        "请依据下方检测规则审核：该区域是真实缺陷，还是光影/水渍/磨削痕迹/"
        "正常纹理等干扰？若为真缺陷，请给出符合规则表述的缺陷类型。\n"
        "仅输出 JSON 对象，字段：\n"
        '- is_defect: true/false（是否真实缺陷）\n'
        '- defect_type: 缺陷类型（is_defect 为 true 时必填）\n'
        "- confidence: 0~1 置信度\n"
        "- description: 一句话外观描述\n\n"
        f"检测规则依据：\n{rule_context}"
    )
    try:
        raw = await llm.vision_chat(prompt, crop_b64, temperature=0.05)
        data = parse_json_from_text(raw)
    except Exception:  # noqa: BLE001  网络/限流/解析失败：交由上层保守处理
        return None
    return data if isinstance(data, dict) else None


def heuristic_filter(
    image_rgb: np.ndarray,
    candidates: List[Dict[str, Any]],
    dist_median: Optional[float] = None,
    region_count: int = 0,
) -> List[Dict[str, Any]]:
    """
    离线启发式裁判：双证据判真伪（方向无关，兼容亮/暗缺陷）。

    证据一【局部对比度显著性】：
      取候选框中心 1/2×1/2 区域为内部信号（避免小缺陷被大框均值稀释），
      与外围环形邻域比较三种统计量（均值差 / P90差 / P10差取最大），
      超过阈值即存在显著局部异常。
    证据二【哨兵异常幅度】（仅候选携带 sentinel_score 时可用）：
      margin = 区域哨兵分 − 全图距离中位数。真缺陷图上异常 patch
      显著抬高中位数之上的尾部；纹理噪声图（长条磨削纹）P99 区域
      成片且 margin 平坦。区域数稀少（≤ judge_sparse_regions）说明
      P99 尾部未被噪声填满，阈值取宽（sparse）；区域饱和说明长图
      纹理噪声占满尾部，只信强异常（dense）。

    两证据任一成立即保留候选。
    """
    H, W = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    margin_sparse = settings.judge_sentinel_margin_sparse
    margin_dense = settings.judge_sentinel_margin_dense
    m_thr = margin_sparse if region_count <= settings.judge_sparse_regions \
        else margin_dense
    kept: List[Dict[str, Any]] = []
    for cand in candidates:
        x, y, w, h = cand["bbox"]
        pad = max(6, int(0.6 * max(w, h)))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        ring = gray[y0:y1, x0:x1]
        # 中心 1/2×1/2 内部窗口
        ix0, iy0 = x + w // 4, y + h // 4
        inner = gray[iy0 : iy0 + max(2, h // 2), ix0 : ix0 + max(2, w // 2)]
        if inner.size == 0 or ring.size == 0:
            continue
        ring_med = float(np.median(ring))
        p10, p90 = np.quantile(inner, [0.10, 0.90])
        contrast = max(
            abs(float(inner.mean()) - float(ring.mean())),
            abs(float(p90) - ring_med),
            abs(ring_med - float(p10)),
        )
        if contrast >= settings.judge_heuristic_contrast:
            kept.append(cand)
            continue
        # 证据二：哨兵异常幅度（证据一不足时的补判）
        score = cand.get("sentinel_score")
        if dist_median is not None and score is not None:
            if float(score) - float(dist_median) >= m_thr:
                kept.append(cand)
    return kept


def _crop_to_jpeg_b64(
    image: Image.Image, bbox: List[int],
    width: int, height: int,
) -> str:
    """按候选框裁剪（适度外扩上下文），必要时缩放后编码 JPEG base64。"""
    x, y, w, h = bbox
    margin = max(
        settings.judge_crop_margin_px,
        int(settings.judge_crop_margin_scale * max(w, h)),
    )
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(width, x + w + margin), min(height, y + h + margin)
    crop = image.convert("RGB").crop((x0, y0, x1, y1))
    # 长边超过限制时缩放，控制请求体大小
    long_side = max(crop.size)
    limit = settings.judge_crop_max_side
    if long_side > limit:
        s = limit / float(long_side)
        crop = crop.resize(
            (max(1, int(crop.size[0] * s)), max(1, int(crop.size[1] * s))),
            Image.BILINEAR,
        )
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=settings.tile_jpeg_quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")
