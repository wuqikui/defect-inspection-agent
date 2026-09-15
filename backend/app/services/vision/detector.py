# -*- coding: utf-8 -*-
"""
缺陷检测器门面（深度模型管线 v2）
==================================
主管线（CPU 友好，模型随项目分发于 backend/models/）：

    ① PatchCore 哨兵全图扫描 —— 异常热力图过滤 100% 正常区域，
       仅输出高危异常区域（CPU，秒级）
        ↓ 仅在高危区域附近
    ② 智能动态切片 —— 抛弃盲目全图滑窗，按高危区域自适应开窗
        ↓
    ③ YOLO-World 空间定位 —— 零样本开放词汇检测，输出高精度 bbox
        ↓
    ④ 云端 VLM 裁决（配置 Key 时）—— 结合 RAG 规则审核候选框，
       纠正类型 / 过滤误报；离线自动降级启发式显著性裁判
        ↓（审核通过的真缺陷 bbox）
    ⑤ 轻量局部 OpenCV 轮廓抠图 —— 仅在 bbox ROI 内二值化 +
       大津法 + 形态学，像素级掩膜落地并还原至原图坐标
        ↓
    ⑥ 整理结构化结果（绘制与持久化由上层完成）

依赖缺失（模型文件 / onnxruntime）时自动退回旧版盲滑窗管线
（_legacy_detect：逐窗 VLM/CV → NMS → 分割），保证系统始终可用。
"""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from app.config import settings
from app.core.exceptions import ModelCallError
from app.services.llm import get_vision_llm
from app.services.vision.nms import non_max_suppression
from app.services.vision.patchcore_sentinel import get_sentinel
from app.services.vision.segmenter import get_segmenter
from app.services.vision.sliding_window import (
    Tile,
    crop_tile,
    plan_region_tiles,
    plan_tiles,
    restore_bbox,
)
from app.services.vision.vlm_judge import heuristic_filter, judge_candidates
from app.services.vision.yolo_world import get_yolo_world

# 进度回调：(百分比 0-100, 阶段描述) -> awaitable
ProgressFn = Callable[[int, str], Awaitable[None]]


class DefectDetector:
    """无状态检测器（模型客户端在 LLM 工厂内复用）。"""

    async def detect(
        self,
        image: Image.Image,
        rule_context: str,
        known_defect_types: List[str],
        progress: Optional[ProgressFn] = None,
    ) -> Dict:
        """
        对一张任意尺寸的工业图片执行完整检测。

        :param image: PIL 原图（调用方负责解压炸弹限制与格式校验）
        :param rule_context: 送入裁决模型的规则文本（结构化规则 + RAG 片段）
        :param known_defect_types: 规则库中已知的缺陷类型
        :param progress: 异步进度回调
        :return: {"defects", "tile_count", "resized", "engine",
                  "segmenter", "sentinel"(可选调试信息)}
        """

        async def report(p: int, stage: str) -> None:
            if progress is not None:
                await progress(p, stage)

        sentinel = get_sentinel() if settings.dnn_pipeline_enabled else None
        yolo = get_yolo_world() if settings.dnn_pipeline_enabled else None
        if sentinel is not None and yolo is not None:
            try:
                return await self._detect_dnn(
                    image, rule_context, known_defect_types,
                    sentinel, yolo, report,
                )
            except ModelCallError:
                # VLM 裁决鉴权/限流等问题不应导致整单失败：
                # 转入纯本地深度管线（启发式裁判）重试一次
                return await self._detect_dnn(
                    image, rule_context, known_defect_types,
                    sentinel, yolo, report, use_vlm=False,
                )
        # 旧版盲滑窗管线（模型缺失 / 深度管线关闭时的降级路径）
        return await self._legacy_detect(
            image, rule_context, known_defect_types, report
        )

    # ==================================================================
    # 主管线：哨兵 → 动态切片 → YOLO-World → VLM 裁决 → ROI 轮廓
    # ==================================================================
    async def _detect_dnn(
        self,
        image: Image.Image,
        rule_context: str,
        known_defect_types: List[str],
        sentinel,
        yolo,
        report: Callable[[int, str], Awaitable[None]],
        use_vlm: bool = True,
    ) -> Dict:
        width, height = image.size

        # ① 哨兵全图异常扫描
        await report(3, "PatchCore 哨兵全图异常扫描…")
        image_rgb = await asyncio.to_thread(
            lambda: np.asarray(image.convert("RGB"), dtype=np.uint8)
        )
        result = await asyncio.to_thread(sentinel.scan, image_rgb)
        regions = result.regions
        stats = result.stats
        await report(
            8,
            f"哨兵扫描完成：{len(regions)} 处高危区域"
            f"（{stats.get('elapsed_s', '?')}s）",
        )

        # ② 智能动态切片（无高危区域 → 大图直接放行）
        tiles, resized = plan_region_tiles(regions, width, height)
        if not tiles:
            await report(98, "未发现高危异常区域，判定合格")
            return {
                "defects": [],
                "tile_count": 0,
                "resized": resized,
                "engine": "yolo",
                "segmenter": "opencv",
                "sentinel": stats,
            }
        total = len(tiles)
        await report(
            10, f"动态切片：{total} 个疑似缺陷局部窗，开始定位…"
        )

        # ③ YOLO-World 空间定位
        raw_detections: List[Dict] = []
        for tile in tiles:
            tile_img = await asyncio.to_thread(crop_tile, image, tile)
            tile_np = np.asarray(tile_img, dtype=np.uint8)
            dets = await asyncio.to_thread(yolo.detect_tile, tile_np)
            for d in dets:
                x, y, w, h = restore_bbox(d["bbox"], tile, width, height)
                raw_detections.append(
                    {
                        "defect_type": d["defect_type"],
                        "confidence": d["confidence"],
                        "bbox": [x, y, w, h],
                        "description": d.get("description", ""),
                        "tile_index": tile.index,
                    }
                )
            done = tile.index + 1
            await report(
                10 + int(40 * done / total),
                f"YOLO-World 定位 {done}/{total} 窗",
            )

        # ③' 候选整理：区域关联 + 哨兵兜底
        # YOLO-World 零样本分值整体偏低，但候选已约束在哨兵高危区域内；
        # 对定位未命中的高危区域，以区域框作为兜底候选交由第④级裁决
        candidates = self._associate_candidates(raw_detections, regions)
        await report(55, f"合并候选（{len(candidates)} 处）…")
        candidates = non_max_suppression(
            candidates,
            iou_threshold=settings.nms_iou_threshold,
            same_type_only=True,
        )

        # ④ VLM 裁决（离线降级启发式显著性裁判）
        engine = "yolo"
        if candidates:
            vision_llm = get_vision_llm() if use_vlm else None
            if vision_llm is not None:
                await report(
                    60, f"VLM 裁决 {len(candidates)} 处候选缺陷…"
                )
                candidates = await judge_candidates(
                    vision_llm, image, candidates,
                    rule_context, known_defect_types,
                )
                engine = "vlm"
            else:
                await report(
                    60, f"启发式裁判过滤 {len(candidates)} 处候选…"
                )
                candidates = heuristic_filter(
                    image_rgb,
                    candidates,
                    dist_median=stats.get("dist_median"),
                    region_count=len(regions),
                )

        # ⑤ 轻量局部 OpenCV 轮廓抠图（bbox ROI 内）
        await report(90, "缺陷部位像素级轮廓提取…")
        segmenter, _ = get_segmenter()
        for defect in candidates:
            seg = await asyncio.to_thread(
                self._safe_segment, segmenter, image_rgb,
                tuple(defect["bbox"]),
            )
            defect["segmentation_available"] = seg is not None
            defect["mask_polygon"] = seg
            defect.setdefault("description", "")
            defect.setdefault("tile_index", -1)

        await report(98, "检测完成，整理结果…")
        return {
            "defects": candidates,
            "tile_count": total,
            "resized": resized,
            "engine": engine,
            "segmenter": "opencv",
            "sentinel": stats,
        }

    # ==================================================================
    # 旧版盲滑窗管线（降级路径）
    # ==================================================================
    async def _legacy_detect(
        self,
        image: Image.Image,
        rule_context: str,
        known_defect_types: List[str],
        report: Callable[[int, str], Awaitable[None]],
    ) -> Dict:
        from app.services.vision.cv_detector import detect_tile_cv
        from app.services.vision.vlm_detector import detect_tile_vlm

        width, height = image.size
        tiles, resized = plan_tiles(width, height)
        total = len(tiles)
        await report(5, f"共规划 {total} 个检测窗，开始检测…")

        vision_llm = get_vision_llm()
        engine = "vlm" if vision_llm is not None else "cv"
        raw_detections: List[Dict] = []
        vlm_failed = False

        for tile in tiles:
            tile_img = await asyncio.to_thread(crop_tile, image, tile)

            # ---- 逐窗检测：视觉模型；首窗失败则整单降级为 CV ----
            if engine == "vlm" and not vlm_failed:
                try:
                    dets = await detect_tile_vlm(
                        vision_llm, tile_img, tile, width, height,
                        rule_context, known_defect_types,
                    )
                except ModelCallError:
                    # 鉴权/限流/网络等问题：后续窗不再请求模型，全部走 CV
                    vlm_failed = True
                    engine = "cv"
                    dets = await asyncio.to_thread(detect_tile_cv, tile_img)
                    dets = self._restore_cv(dets, tile, width, height)
            else:
                dets = await asyncio.to_thread(detect_tile_cv, tile_img)
                dets = self._restore_cv(dets, tile, width, height)

            raw_detections.extend(dets)
            done = tile.index + 1
            percent = 5 + int(80 * done / total)
            await report(
                percent,
                f"第 {done}/{total} 个检测窗"
                f"{('（已切换经典CV兜底）' if engine == 'cv' else '')}",
            )

        # ---- 置信度过滤（CV 路径再保险过滤一次） ----
        kept = [
            d for d in raw_detections
            if d["confidence"] >= settings.det_confidence_threshold
        ]

        # ---- 跨滑窗 NMS：同一缺陷在相邻窗重复检出时只保留最高分框 ----
        await report(87, "合并跨窗重复检测结果…")
        merged = non_max_suppression(
            kept,
            iou_threshold=settings.nms_iou_threshold,
            same_type_only=True,
        )

        # ---- 逐缺陷精确分割（SAM 已弃用，恒为 OpenCV ROI 轮廓） ----
        await report(92, "缺陷部位精确分割…")
        segmenter, _ = get_segmenter()
        image_rgb = await asyncio.to_thread(
            lambda: np.asarray(image.convert("RGB"), dtype=np.uint8)
        )
        for defect in merged:
            seg = await asyncio.to_thread(
                self._safe_segment, segmenter, image_rgb,
                tuple(defect["bbox"]),
            )
            defect["segmentation_available"] = seg is not None
            defect["mask_polygon"] = seg
            defect.setdefault("description", "")
            defect.setdefault("tile_index", -1)

        await report(98, "检测完成，整理结果…")
        return {
            "defects": merged,
            "tile_count": total,
            "resized": resized,
            "engine": engine,
            "segmenter": "opencv",
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _associate_candidates(
        dets: List[Dict], regions: List[Dict]
    ) -> List[Dict]:
        """
        候选整理：
        * 哨兵无高危区域（小图整检模式）→ 保留全部定位候选；
        * 有高危区域 → 仅保留【定位合理】的候选：中心落入区域内，
          且框边长 ≤ 2×区域长边（近乎全图的大框不携带定位信息，
          不能算作命中，否则会错误抑制哨兵兜底候选）；
        * 未被定位命中的区域以区域框生成"未知缺陷"兜底候选
          （类型与真伪交由 VLM/启发式裁决）。
        """
        if not regions:
            return list(dets)

        out: List[Dict] = []
        covered: set = set()
        for d in dets:
            x, y, w, h = d["bbox"]
            cx, cy = x + w / 2.0, y + h / 2.0
            for i, r in enumerate(regions):
                rx, ry, rw, rh = r["bbox"]
                if rx <= cx <= rx + rw and ry <= cy <= ry + rh:
                    # 尺寸合理性门槛：大框不是对当前区域的定位
                    if w > 2.0 * max(rw, rh) or h > 2.0 * max(rw, rh):
                        continue
                    # 哨兵分数随候选下传，供离线裁判做幅度判据
                    d["sentinel_score"] = r["score"]
                    out.append(d)
                    covered.add(i)
                    break
        for i, r in enumerate(regions):
            if i in covered:
                continue
            rx, ry, rw, rh = r["bbox"]
            out.append(
                {
                    "defect_type": "未知缺陷",
                    "confidence": 0.5,
                    "bbox": [rx, ry, rw, rh],
                    "description": "哨兵标记的异常区域（开放词汇定位未命中，类型待裁决）",
                    "tile_index": -1,
                    "sentinel_score": r["score"],
                }
            )
        return out

    @staticmethod
    def _restore_cv(dets: List[Dict], tile: Tile,
                    width: int, height: int) -> List[Dict]:
        """CV 检测器输出的是切片坐标，这里统一复原为原图坐标。"""
        out = []
        for d in dets:
            x, y, w, h = restore_bbox(d["bbox"], tile, width, height)
            out.append(
                {
                    "defect_type": d["defect_type"],
                    "confidence": d["confidence"],
                    "bbox": [x, y, w, h],
                    "description": d.get("description", ""),
                    "tile_index": tile.index,
                }
            )
        return out

    @staticmethod
    def _safe_segment(segmenter, image_rgb: np.ndarray,
                      bbox: Tuple[int, int, int, int]):
        """分割异常不影响整体结果产出。"""
        try:
            return segmenter.segment(image_rgb, bbox)
        except Exception:  # noqa: BLE001
            return None


# 全局单例
defect_detector = DefectDetector()
