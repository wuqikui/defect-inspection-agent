# -*- coding: utf-8 -*-
"""
缺陷部位精确分割
================
两级实现，接口完全一致：

1. SAM（Segment Anything Model，Meta）—— 首选：
   用检测框作为 box prompt，由 SAM 预测像素级掩膜，精度最高。
   仅当 (a) 已安装 torch + segment_anything，(b) 权重文件存在时启用。
2. OpenCV 轮廓分割 —— 自动兜底：
   在检测框 ROI 内做 CLAHE 对比度增强 + Otsu 阈值，取最大轮廓。

输出统一为多边形点序列 ``[[x, y], ...]``（原图坐标），
前端可用该多边形做半透明高亮填充。

SAM 为可选重依赖（torch 体积大），默认不安装，系统自动降级，
符合“模型接口标准化、便于替换升级”的要求。
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
    """OpenCV 兜底分割器（始终可用）。"""

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
        _, binary = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        # 取面积最大的轮廓作为缺陷外形
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 8:
            return None
        epsilon = 0.01 * cv2.arcLength(cnt, closed=True)
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
        # 坐标加回 ROI 偏移，回到原图坐标系
        polygon = [[int(p[0][0]) + x0, int(p[0][1]) + y0] for p in approx]
        return polygon if len(polygon) >= 3 else None


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


def _try_build_sam() -> Optional[SAMSegmenter]:
    """检测依赖与权重，齐备则构造 SAM；否则返回 None。"""
    ckpt = settings.sam_checkpoint_path.strip()
    if not ckpt:
        # 也接受放到默认目录的 vit_b 权重
        default = settings.sam_dir / f"sam_{settings.sam_model_type}_*.pth"
        candidates = list(settings.sam_dir.glob("sam_*.pth"))
        if not candidates:
            return None
        ckpt = str(candidates[0])
    if not Path(ckpt).exists():
        return None
    try:
        import torch  # noqa: F401
        import segment_anything  # noqa: F401
    except Exception:  # noqa: BLE001
        return None
    try:
        return SAMSegmenter(ckpt, settings.sam_model_type, settings.sam_device)
    except Exception:  # noqa: BLE001
        return None


# 模块级单例（惰性）
_sam_instance: Optional[SAMSegmenter] = None
_opencv_instance = OpenCVSegmenter()
_probed = False
_probe_lock = threading.Lock()


def get_segmenter():
    """
    获取当前最佳分割器：
    优先 SAM（惰性探测一次），失败则恒为 OpenCV 实现。
    返回 (segmenter, is_sam)。
    """
    global _sam_instance, _probed
    with _probe_lock:
        if not _probed:
            _sam_instance = _try_build_sam()
            _probed = True
    if _sam_instance is not None:
        return _sam_instance, True
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
