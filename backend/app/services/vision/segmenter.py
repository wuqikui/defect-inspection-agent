# -*- coding: utf-8 -*-
"""
缺陷部位精确分割（轻量局部 ROI 轮廓抠图）
==========================================
角色：检测管线最后一级"像素级落地"。仅在缺陷 bbox 的 ROI 内做
对比度增强 + 大津法二值化 + 形态学闭运算 + 轮廓提取（耗时毫秒级），
输出多边形点序列（原图坐标），供前端半透明高亮渲染。

自适应极性：分别尝试"暗于背景"与"亮于背景"两个二值化方向，
以 ROI 内外对比度更大者为准 —— 兼容暗底亮缺陷与亮底暗缺陷。

历史说明：SAM 分割方案因本地推理内存开销过大被正式弃用，
SAMSegmenter 类仅作保留参考，工厂函数不再构建。
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from app.config import settings


class OpenCVSegmenter:
    """OpenCV ROI 轮廓分割器（始终可用）。"""

    name = "opencv"

    def segment(
        self, image_rgb: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[List[List[int]]]:
        """
        在 bbox ROI 内提取缺陷轮廓多边形。

        :param image_rgb: 原图 RGB ndarray
        :param bbox: (x, y, w, h)
        :return: 多边形点序列或 None
        """
        x, y, w, h = bbox
        H, W = image_rgb.shape[:2]
        # 适度外扩上下文，帮助阈值在缺陷边缘处收敛
        pad = max(4, int(0.08 * max(w, h)))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        roi = image_rgb[y0:y1, x0:x1]
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        # CLAHE 局部对比度增强，让浅淡缺陷边界更清晰
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # Otsu 自动阈值 + 形态学闭运算填补孔洞
        otsu_thr, _ = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        def _pick(inverse: bool):
            binary = (
                enhanced < otsu_thr if inverse else enhanced > otsu_thr
            ).astype(np.uint8) * 255
            binary = cv2.morphologyEx(
                binary, cv2.MORPH_CLOSE, kernel, iterations=2
            )
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                return None
            cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(cnt) < 8:
                return None
            epsilon = 0.01 * cv2.arcLength(cnt, closed=True)
            approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
            if len(approx) < 3:
                return None
            polygon = [
                [int(p[0][0]) + x0, int(p[0][1]) + y0] for p in approx
            ]
            # 极性打分：轮廓内部与外围邻域的灰度差（绝对值越大越可信）
            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            inner_mean = float(gray[mask > 0].mean())
            ring = gray[mask == 0]
            outer_mean = float(ring.mean()) if ring.size else inner_mean
            return polygon, abs(inner_mean - outer_mean)

        # 两个极性择优：兼容暗底亮缺陷与亮底暗缺陷
        best: Optional[List[List[int]]] = None
        best_score = -1.0
        for inverse in (True, False):
            picked = _pick(inverse)
            if picked is not None and picked[1] > best_score:
                best, best_score = picked[0], picked[1]
        return best


class SAMSegmenter:
    """SAM 分割器（惰性加载，仅在依赖与权重齐备时创建成功）。"""

    name = "sam"

    def __init__(self, checkpoint: str, model_type: str, device: str):
        # 依赖在工厂函数中预先验证，这里直接导入
        import torch  # noqa: F401
        from segment_anything import sam_model_registry, SamPredictor

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=device)
        self._predictor = SamPredictor(sam)
        self._lock = threading.Lock()
        self._image_set = False

    @property
    def device(self) -> str:
        return self._device

    def set_image(self, image_rgb: np.ndarray) -> None:
        """整图只编码一次，后续多个框复用 image embedding（省算力）。"""
        with self._lock:
            self._predictor.set_image(image_rgb)
            self._image_set = True

    def segment(
        self, image_rgb: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[List[List[int]]]:
        x, y, w, h = bbox
        box = np.array([x, y, x + w, y + h], dtype=np.float32)
        with self._lock:
            masks, scores, _ = self._predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box[None, :],
                multimask_output=False,
            )
        if masks is None or masks.size == 0:
            return None
        mask = masks[0].astype(np.uint8)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        epsilon = 0.005 * cv2.arcLength(cnt, closed=True)
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
        polygon = [[int(p[0][0]), int(p[0][1])] for p in approx]
        return polygon if len(polygon) >= 3 else None


# 模块级单例
_opencv_instance = OpenCVSegmenter()


def get_segmenter():
    """
    获取分割器：SAM 已正式弃用（本地内存开销过大），恒为 OpenCV 实现。
    返回 (segmenter, is_sam=False)。保留返回二元组以兼容既有调用方。
    """
    return _opencv_instance, False


def segment_defect(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Dict:
    """
    对单个缺陷框执行分割的统一入口。

    :return: {"available": bool, "engine": "sam"/"opencv",
              "polygon": [[x,y],...] 或 None}
    """
    segmenter, is_sam = get_segmenter()
    image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    try:
        polygon = segmenter.segment(image_rgb, bbox)
    except Exception:  # noqa: BLE001
        polygon = None
    return {
        "available": polygon is not None,
        "engine": segmenter.name,
        "polygon": polygon,
        "is_sam": is_sam,
    }
