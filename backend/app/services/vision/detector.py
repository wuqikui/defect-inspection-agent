# -*- coding: utf-8 -*-
"""
缺陷检测器门面
==============
串联视觉检测全链路（对应需求“滑窗检测 → 结果复原 → NMS 合并 → 精确分割”）：

    规划滑窗 → 逐窗检测（多模态模型，失败自动降级经典 CV）
             → 检测框复原到原图坐标 → 置信度过滤 → 跨窗 NMS 去重
             → SAM/OpenCV 逐缺陷分割 → 返回结构化结果

逐窗处理而非整图缩放：每个时刻内存中只有原图 + 一个 1024×1024 切片，
从而对大尺寸图像友好，避免内存溢出。
"""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from app.config import settings
from app.core.exceptions import ModelCallError
from app.services.llm import get_vision_llm
from app.services.vision.cv_detector import detect_tile_cv
from app.services.vision.nms import non_max_suppression
from app.services.vision.segmenter import get_segmenter
from app.services.vision.sliding_window import (
    Tile,
    crop_tile,
    plan_tiles,
    restore_bbox,
)
from app.services.vision.vlm_detector import detect_tile_vlm

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
        :param rule_context: 送入多模态模型的规则文本（结构化规则 + RAG 片段）
        :param known_defect_types: 规则库中已知的缺陷类型
        :param progress: 异步进度回调
        :return: {"defects": [...], "tile_count", "resized", "engine"}
        """

        async def report(p: int, stage: str) -> None:
            if progress is not None:
                await progress(p, stage)

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
                f"第 {done}/{total} 个检测窗{('（已切换经典CV兜底）' if engine == 'cv' else '')}",
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

        # ---- 逐缺陷精确分割 ----
        await report(92, "缺陷部位精确分割…")
        segmenter, is_sam = get_segmenter()
        image_rgb = await asyncio.to_thread(
            lambda: np.asarray(image.convert("RGB"), dtype=np.uint8)
        )
        if is_sam:
            # SAM 整图只编码一次，多框复用 embedding
            await asyncio.to_thread(segmenter.set_image, image_rgb)

        for defect in merged:
            seg = await asyncio.to_thread(
                self._safe_segment, segmenter, image_rgb,
                tuple(defect["bbox"])
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
            "segmenter": "sam" if is_sam else "opencv",
        }

    # ------------------------------------------------------------------
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
