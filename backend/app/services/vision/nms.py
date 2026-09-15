# -*- coding: utf-8 -*-
"""
非极大值抑制（Non-Maximum Suppression）
======================================
滑窗之间有重叠，同一个真实缺陷可能被相邻窗各检出一次。
所有窗的检测框汇总到原图坐标后，用 NMS 去除重复框：

1. 按置信度降序排序；
2. 取最高分框，删除与其 IoU 超过阈值的其它框；
3. 重复直至队列为空。

可选择“仅在同缺陷类型间抑制”，避免把气孔与划痕误合并。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    计算两个 [x, y, w, h] 框的交并比。
    """
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = ax1 + box_a[2], ay1 + box_a[3]
    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = bx1 + box_b[2], by1 + box_b[3]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = box_a[2] * box_a[3] + box_b[2] * box_b[3] - inter
    return float(inter / union) if union > 0 else 0.0


def non_max_suppression(
    detections: List[Dict],
    iou_threshold: float,
    same_type_only: bool = True,
) -> List[Dict]:
    """
    对检测结果做 NMS。

    :param detections: [{"bbox":[x,y,w,h], "confidence", "defect_type", ...}]
    :param iou_threshold: IoU 超过该值视为重复
    :param same_type_only: True 时只在相同缺陷类型之间抑制
    :return: 去重后的检测列表
    """
    if not detections:
        return []
    order = sorted(
        range(len(detections)),
        key=lambda i: detections[i]["confidence"],
        reverse=True,
    )
    boxes = np.array([d["bbox"] for d in detections], dtype=np.float64)
    alive = [True] * len(detections)
    kept: List[int] = []

    for i in order:
        if not alive[i]:
            continue
        kept.append(i)
        for j in order:
            if not alive[j] or j == i:
                continue
            if same_type_only and (
                detections[i]["defect_type"] != detections[j]["defect_type"]
            ):
                continue
            if iou(boxes[i], boxes[j]) >= iou_threshold:
                alive[j] = False
    return [detections[i] for i in kept]
