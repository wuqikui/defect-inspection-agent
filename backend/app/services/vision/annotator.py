# -*- coding: utf-8 -*-
"""
检测结果可视化（标注图绘制）
==========================
在原图副本上绘制：
1. SAM/OpenCV 分割多边形 → 半透明填充 + 轮廓线；
2. 检测边界框 → 按缺陷类型着色的矩形；
3. 标签 → “类型 置信度%”（中文使用 PIL 绘制，避免 OpenCV 中文乱码）。

输出 JPEG 到 results 目录。为避免超大图标注文件过大，
当最长边超过 6000 像素时等比缩小保存（前端主要依据 JSON 坐标自行叠加，
标注图仅作快速预览，因此不影响框选精度）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 每种缺陷类型固定配色（RGB），未知类型使用黄色
_TYPE_COLORS = {
    "气孔": (239, 68, 68),
    "孔洞": (239, 68, 68),
    "针孔": (249, 115, 22),
    "划痕": (59, 130, 246),
    "划伤": (59, 130, 246),
    "刮伤": (59, 130, 246),
    "起层": (168, 85, 247),
    "分层": (168, 85, 247),
    "缺肉": (234, 179, 8),
    "裂纹": (220, 38, 38),
}
_DEFAULT_COLOR = (234, 179, 8)
_MAX_SAVE_SIDE = 6000


def _color_for(defect_type: str) -> tuple:
    for key, color in _TYPE_COLORS.items():
        if key in defect_type:
            return color
    return _DEFAULT_COLOR


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """优先加载 Windows 常见中文字体；失败退化为位图字体（英文仍可读）。"""
    for candidate in (
        "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",    # 黑体
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def draw_annotations(image: Image.Image, defects: List[Dict]) -> Image.Image:
    """
    在原图副本上绘制全部缺陷标注。

    :param image: 原始 PIL 图像
    :param defects: 已完成坐标复原与分割的缺陷列表
    """
    annotated = image.convert("RGB").copy()
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(annotated)
    overlay_draw = ImageDraw.Draw(overlay)

    # 字号随图像尺寸自适应（1024 基准下约 28px）
    base = max(annotated.size)
    font_size = max(18, int(base / 1024 * 26))
    line_width = max(2, int(base / 1200))
    font = _load_font(font_size)

    for defect in defects:
        color = _color_for(defect["defect_type"])
        x, y, w, h = defect["bbox"]

        # 1) 分割掩膜半透明高亮
        polygon = defect.get("mask_polygon")
        if polygon and len(polygon) >= 3:
            flat = [(p[0], p[1]) for p in polygon]
            overlay_draw.polygon(flat, fill=color + (70,), outline=color + (200,))

        # 2) 边界框
        draw.rectangle([x, y, x + w, y + h], outline=color, width=line_width)

        # 3) 标签（带与框同色的底条，保证可读性）
        label = f"{defect['defect_type']} {defect['confidence'] * 100:.0f}%"
        try:
            tb = draw.textbbox((x, y), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except AttributeError:  # 极旧 Pillow 无 textbbox
            tw, th = font_size * len(label) // 2, font_size
        label_y = y - th - 6 if y - th - 6 >= 0 else y + 2
        draw.rectangle(
            [x - 1, label_y - 2, x + tw + 8, label_y + th + 2],
            fill=color,
        )
        draw.text((x + 3, label_y), label, fill=(255, 255, 255), font=font)

    return Image.alpha_composite(annotated.convert("RGBA"), overlay).convert("RGB")


def save_annotated(image: Image.Image, defects: List[Dict],
                   out_path: Path) -> Path:
    """绘制并保存标注 JPEG（大图等比缩小），返回输出路径。"""
    annotated = draw_annotations(image, defects)
    longest = max(annotated.size)
    if longest > _MAX_SAVE_SIDE:
        ratio = _MAX_SAVE_SIDE / float(longest)
        new_size = (
            int(annotated.size[0] * ratio),
            int(annotated.size[1] * ratio),
        )
        annotated = annotated.resize(new_size, Image.BILINEAR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(out_path, format="JPEG", quality=82, optimize=True)
    return out_path
