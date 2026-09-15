# -*- coding: utf-8 -*-
"""
多模态大模型滑窗检测适配
========================
把单个 window×window 切片连同 RAG 规则上下文交给视觉 LLM，
解析其返回的 JSON 检测结果，并做严格的坐标合法性校验
（防止模型输出越界 / 缺字段 / 非数值导致后续复原出错）。
"""
from __future__ import annotations

from typing import Any, Dict, List

from app.config import settings
from app.services.llm.base import BaseLLM
from app.services.vision.sliding_window import (
    Tile,
    restore_bbox,
    tile_to_jpeg_base64,
)
from PIL import Image


async def detect_tile_vlm(
    llm: BaseLLM,
    tile_img: Image.Image,
    tile: Tile,
    orig_width: int,
    orig_height: int,
    rule_context: str,
    known_defect_types: List[str],
) -> List[Dict[str, Any]]:
    """
    用视觉模型检测一个切片，并把结果直接复原到原图坐标系。

    :return: [{"defect_type","confidence","bbox"(原图坐标),
               "description","tile_index"}]
    """
    b64 = tile_to_jpeg_base64(tile_img)
    raw_items = await llm.judge_tile(rule_context, b64, known_defect_types)

    results: List[Dict[str, Any]] = []
    for item in raw_items:
        parsed = _coerce_item(item, tile, orig_width, orig_height)
        if parsed is not None:
            results.append(parsed)
    return results


def _coerce_item(
    item: Any,
    tile: Tile,
    orig_width: int,
    orig_height: int,
) -> Dict[str, Any] | None:
    """校验并转换单条模型输出；非法项返回 None 丢弃。"""
    if not isinstance(item, dict):
        return None
    raw_box = item.get("bbox")
    if not (isinstance(raw_box, (list, tuple)) and len(raw_box) == 4):
        return None
    try:
        box = [float(v) for v in raw_box]
    except (TypeError, ValueError):
        return None
    if any(v < 0 or v != v for v in box):  # 负数 / NaN 一律丢弃
        return None

    try:
        confidence = float(item.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    if confidence < settings.det_confidence_threshold:
        return None

    defect_type = str(item.get("defect_type", "缺陷")).strip()[:20] or "缺陷"
    # 先在切片坐标系内钳制（resize 模式下钳制到 window；网格模式同理）
    box[2] = max(1.0, min(box[2], tile.size - box[0]))
    box[3] = max(1.0, min(box[3], tile.size - box[1]))
    if box[0] >= tile.size or box[1] >= tile.size:
        return None

    x, y, w, h = restore_bbox(box, tile, orig_width, orig_height)
    return {
        "defect_type": defect_type,
        "confidence": round(confidence, 3),
        "bbox": [x, y, w, h],
        "description": str(item.get("description", ""))[:200],
        "tile_index": tile.index,
    }
