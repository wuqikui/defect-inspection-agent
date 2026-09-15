# -*- coding: utf-8 -*-
"""
离线经典 CV 缺陷检测（兜底检测器）
=================================
当系统未配置视觉大模型（智谱 GLM-4V / 千问 VL）时启用，
基于 OpenCV 的传统图像处理链路给出“可用但有限”的检测结果，
保证无网络 / 无 Key 环境下检测流程仍可完整跑通：

    中值滤波去噪
        → 自适应高斯阈值（提取比局部背景明显更暗的区域，抗光照不均）
        → 形态学开运算去除椒盐碎点
        → 轮廓发现 + 面积/圆度过滤
        → 形状分类：近圆形暗斑→气孔/孔洞；细长暗痕→划痕
        → 置信度 = 局部对比度 + 面积显著性 的归一化组合

注意：该方法无法理解规则语义，复杂纹理下误报率高于多模态模型，
结果中会标注 detector="cv" 以与模型结果区分。
"""
from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np
from PIL import Image


def detect_tile_cv(tile_img: Image.Image) -> List[Dict]:
    """
    对单个 window×window 切片执行经典 CV 检测。

    :param tile_img: PIL RGB 切片
    :return: 切片坐标系下的检测列表
             [{"defect_type","confidence","bbox":[x,y,w,h],"description"}]
    """
    gray = np.asarray(tile_img.convert("L"), dtype=np.uint8)
    h_img, w_img = gray.shape

    # 1) 中值滤波抑制工业相机常见的椒盐噪声
    denoised = cv2.medianBlur(gray, 5)

    # 2) 自适应阈值：blockSize 随窗尺寸取奇数，C 控制灵敏度
    block = max(51, (min(h_img, w_img) // 12) | 1)  # 保证为奇数
    dark_mask = cv2.adaptiveThreshold(
        denoised,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,  # 暗于邻域的像素置白
        blockSize=block,
        C=9,
    )

    # 3) 形态学开运算：先腐蚀后膨胀，去掉 1~2 像素的碎点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 4) 轮廓提取
    contours, _ = cv2.findContours(
        dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    tile_area = h_img * w_img
    min_area = max(16.0, tile_area * 0.00002)   # 约 1024 窗下 20 像素
    max_area = tile_area * 0.15                 # 过大区域通常是阴影/遮挡
    detections: List[Dict] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < 3 or h < 3:
            continue

        # 圆度：4πA / P²，圆=1，细长形状趋近 0
        perimeter = cv2.arcLength(cnt, closed=True)
        circularity = (
            4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        )
        extent = area / float(w * h)  # 轮廓填充外接矩形的比例
        if extent < 0.25:            # 过于松散的噪点团
            continue

        # 局部对比度：轮廓内灰度均值 vs 外围环形邻域均值（越暗差异越大）
        contrast = _local_contrast(denoised, x, y, w, h)
        if contrast < 12:  # 灰度差小于 12 视为光影/磨削痕迹，不报
            continue

        # 5) 形状分类
        aspect = max(w, h) / float(max(min(w, h), 1))
        if aspect >= 3.0 and circularity < 0.35:
            defect_type = "划痕"
            desc = "局部细长深色线性痕迹（疑似划痕/划伤）"
        else:
            defect_type = "气孔"
            desc = "局部近圆形深色斑点（疑似气孔/孔洞）"

        # 6) 置信度：对比度主导，面积/圆度微调，限制在 [0.35, 0.92]
        confidence = 0.40 + min(contrast / 90.0, 0.45)
        confidence += min(area / (tile_area * 0.01), 1.0) * 0.05
        confidence += max(circularity - 0.5, 0) * 0.05
        confidence = float(np.clip(confidence, 0.35, 0.92))

        detections.append(
            {
                "defect_type": defect_type,
                "confidence": round(confidence, 3),
                "bbox": [int(x), int(y), int(w), int(h)],
                "description": desc,
            }
        )

    return detections


def _local_contrast(gray: np.ndarray, x: int, y: int,
                    w: int, h: int, pad: int = 6) -> float:
    """
    计算框内区域相对其外围环形邻域的平均暗度差。

    :return: 外围均值 - 内部均值（正值表示内部更暗）
    """
    H, W = gray.shape
    inner = gray[y : y + h, x : x + w]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    outer = gray[y0:y1, x0:x1].astype(np.float32)
    if outer.size == 0 or inner.size == 0:
        return 0.0
    outer_mean = outer.mean()
    inner_mean = inner.astype(np.float32).mean()
    return float(outer_mean - inner_mean)
