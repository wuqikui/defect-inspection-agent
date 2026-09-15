# -*- coding: utf-8 -*-
"""
YOLO-World 空间定位器（ONNX Runtime，CPU 友好）
==============================================
角色：检测管线第二级"定位器"。对 PatchCore 哨兵切出的疑似缺陷局部
切片做零样本开放词汇检测，输出像素级精度的高置信 [bbox]。

模型：backend/models/yolov8s-worldv2.onnx —— 由 scripts/prepare_models.py
导出，缺陷类别文本嵌入在导出时已【内联】进检测头权重：
  * 运行时只需 onnxruntime + numpy，无需 torch / CLIP；
  * 类别索引顺序 = app/services/vision/yolo_labels.py 的 LABELS 顺序。

输出坐标为切片内 [x, y, w, h]（左上角原点），由调用方映射回原图。
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import onnxruntime as ort

from app.config import settings
from app.services.vision.nms import non_max_suppression
from app.services.vision.yolo_labels import LABELS, zh_name

# letterbox 填充灰（ultralytics 惯例值）
_PAD_GRAY = 114.0


class YoloWorldDetector:
    """YOLO-World ONNX 推理器（惰性单例，见 get_yolo_world）。"""

    def __init__(self, model_path: Path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(model_path), opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._imgsz = self._session.get_inputs()[0].shape[-1] or 640
        self._prompts = [en for en, _ in LABELS]
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def detect_tile(self, tile_rgb: np.ndarray) -> List[Dict[str, Any]]:
        """
        对一个切片执行检测。

        :param tile_rgb: 切片 RGB uint8 ndarray（任意尺寸）
        :return: 切片坐标系检测结果
                 [{"defect_type","confidence","bbox":[x,y,w,h],"description"}]
        """
        canvas, r, pad_x, pad_y = self._letterbox(tile_rgb)
        pred = self._forward(canvas)                 # (4+nc, N)

        boxes = pred[:4].T                           # (N,4) cxcywh
        scores = pred[4:].T                          # (N,nc)
        if scores.size and scores.max() > 1.5:       # 兼容未加 sigmoid 的导出
            scores = 1.0 / (1.0 + np.exp(-scores))

        cls_idx = scores.argmax(axis=1)
        conf = scores.max(axis=1)

        keep = conf >= settings.yolo_conf_threshold
        if not keep.any():
            return []
        boxes, conf, cls_idx = boxes[keep], conf[keep], cls_idx[keep]

        # cxcywh → xywh（映射回切片坐标需先去 letterbox）
        results: List[Dict[str, Any]] = []
        H, W = tile_rgb.shape[:2]
        for (cx, cy, w, h), cf, ci in zip(boxes, conf, cls_idx):
            x0 = max(0.0, (float(cx) - float(w) / 2.0 - pad_x) / r)
            y0 = max(0.0, (float(cy) - float(h) / 2.0 - pad_y) / r)
            x1 = min(float(W), (float(cx) + float(w) / 2.0 - pad_x) / r)
            y1 = min(float(H), (float(cy) + float(h) / 2.0 - pad_y) / r)
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            # 退化框过滤：长宽均接近整窗的大框不携带定位信息
            # （哨兵高危区域面积占比上限 25%，合法缺陷不可能占满整窗；
            #   零样本模型对整图常给出这种"全图框"，必须丢弃）
            if x1 - x0 > 0.85 * W and y1 - y0 > 0.85 * H:
                continue
            prompt = self._prompts[int(ci)] if int(ci) < len(self._prompts) else ""
            zh = zh_name(prompt) if prompt else "缺陷"
            results.append(
                {
                    "defect_type": zh,
                    "confidence": round(float(cf), 3),
                    "bbox": [int(round(x0)), int(round(y0)),
                             int(round(x1 - x0)), int(round(y1 - y0))],
                    "description": f"开放词汇定位命中类别「{zh}」",
                }
            )

        # 切片内 NMS（复用既有实现，同类抑制）
        results = non_max_suppression(
            results,
            iou_threshold=settings.yolo_iou_threshold,
            same_type_only=True,
        )
        results.sort(key=lambda d: d["confidence"], reverse=True)
        return results[: settings.yolo_max_det_per_tile]

    # ------------------------------------------------------------------
    @staticmethod
    def _letterbox(img: np.ndarray):
        """
        等比缩放 + 灰边填充到 imgsz×imgsz。

        :return: (画布图, 缩放比 r, x向补边 pad_x, y向补边 pad_y)
                 反变换：原图坐标 = (画布坐标 - pad) / r
        """
        h, w = img.shape[:2]
        r = min(float(settings.yolo_input_size) / h,
                float(settings.yolo_input_size) / w)
        nh, nw = max(1, int(round(h * r))), max(1, int(round(w * r)))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full(
            (settings.yolo_input_size, settings.yolo_input_size, 3),
            _PAD_GRAY, np.float32,
        )
        pad_x = (settings.yolo_input_size - nw) / 2.0
        pad_y = (settings.yolo_input_size - nh) / 2.0
        x0, y0 = int(pad_x), int(pad_y)
        canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
        return canvas, r, pad_x, pad_y

    def _forward(self, blob: np.ndarray) -> np.ndarray:
        """前向：返回 (4+nc, N) 预测矩阵。"""
        x = blob.transpose(2, 0, 1)[None] / 255.0    # (1,3,s,s) RGB 0~1
        x = np.ascontiguousarray(x, dtype=np.float32)
        with self._lock:
            out = self._session.run(None, {self._input_name: x})[0]
        return out[0] if out.ndim == 3 else out


# ----------------------------------------------------------------------
# 惰性单例
# ----------------------------------------------------------------------
_yolo: Optional[YoloWorldDetector] = None
_yolo_lock = threading.Lock()


def yolo_model_path() -> Path:
    return settings.models_dir / "yolov8s-worldv2.onnx"


def get_yolo_world() -> Optional[YoloWorldDetector]:
    """模型文件存在则返回定位器单例，否则 None。"""
    global _yolo
    if _yolo is not None:
        return _yolo
    path = yolo_model_path()
    if not path.exists():
        return None
    with _yolo_lock:
        if _yolo is None:
            try:
                _yolo = YoloWorldDetector(path)
            except Exception:  # noqa: BLE001
                return None
    return _yolo
