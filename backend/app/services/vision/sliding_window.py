# -*- coding: utf-8 -*-
"""
固定方形滑窗：规划 / 切片 / 坐标复原
====================================
核心规则（严格对应需求）：

场景一 —— 图片可被滑窗“完全覆盖”（宽、高均 ≤ window）：
    将整张图 resize 成 window×window，只检测一次，检测框按
    宽/高各自的缩放比还原到原图坐标。

场景二 —— 图片任一维度大于 window：
    在每个维度上计算网格数 n = ceil(dim / window)，
    再反推均匀步长 step = (dim - window) / (n - 1)（n>1 时）。
    最后一个窗的起点恰好为 dim - window，因此：
      * 所有窗都是完整的 window×window，无需 padding；
      * 窗的并集 = 整张图（不漏）；
      * 相邻窗之间等距重叠，边缘与图像边界严格对齐（不溢出、不漏检）。

坐标复原：切片内检测框 + 切片左上角偏移；单次 resize 模式则按缩放比还原。
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image

from app.config import settings


@dataclass
class Tile:
    """一个检测窗（坐标均相对于原图）。"""

    index: int
    x: int                 # 左上角 x
    y: int                 # 左上角 y
    size: int              # 边长（固定等于 window）
    resized: bool = False  # 是否为“整图 resize 单检”模式


def plan_tiles(width: int, height: int,
               window: int | None = None) -> Tuple[List[Tile], bool]:
    """
    规划滑窗布局。

    :param width: 原图宽（像素）
    :param height: 原图高（像素）
    :param window: 方形窗边长，默认取配置 window_size
    :return: (窗列表, 是否为整图resize单检模式)
    """
    window = window or settings.window_size

    # 场景一：一张窗即可完整覆盖 → resize 单检
    if max(width, height) <= window:
        return [Tile(index=0, x=0, y=0, size=window, resized=True)], True

    # 场景二：网格 + 均匀步长
    nx = max(2, math.ceil(width / window))
    ny = max(2, math.ceil(height / window))
    xs = _axis_positions(width, window, nx)
    ys = _axis_positions(height, window, ny)

    tiles: List[Tile] = []
    idx = 0
    for y in ys:
        for x in xs:
            tiles.append(Tile(index=idx, x=x, y=y, size=window, resized=False))
            idx += 1
    return tiles, False


def _axis_positions(dim: int, window: int, n: int) -> List[int]:
    """
    计算单轴上的窗起点序列：首窗贴 0、末窗贴 dim-window、中间均匀分布。

    :param dim: 该轴图像长度
    :param window: 窗长
    :param n: 窗数量（≥2）
    """
    if n == 1:
        return [0]
    step = (dim - window) / (n - 1)
    # 末点强制对齐边界，消除浮点取整误差
    return [int(round(i * step)) for i in range(n - 1)] + [dim - window]


def crop_tile(image: Image.Image, tile: Tile) -> Image.Image:
    """
    从原图取出该窗对应的 window×window RGB 切片。

    * 网格模式：直接按坐标裁剪（窗都在图内，尺寸恒为 window）；
    * resize 单检模式：整图缩放到 window×window。
    """
    if tile.resized:
        return image.convert("RGB").resize(
            (tile.size, tile.size), Image.BILINEAR
        )
    box = (tile.x, tile.y, tile.x + tile.size, tile.y + tile.size)
    return image.convert("RGB").crop(box)


def tile_to_jpeg_base64(tile_img: Image.Image) -> str:
    """把切片编码为 JPEG base64（供多模态模型输入）。"""
    import base64

    buf = io.BytesIO()
    tile_img.save(buf, format="JPEG", quality=settings.tile_jpeg_quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def restore_bbox(bbox: List[float], tile: Tile,
                 orig_width: int, orig_height: int) -> Tuple[int, int, int, int]:
    """
    把“切片内坐标”的检测框复原为“原图坐标”。

    :param bbox: [x, y, w, h]（切片像素坐标，约定 x,y 为左上角）
    :param tile: 所属窗
    :param orig_width / orig_height: 原图尺寸（resize 模式反缩放用）
    :return: 原图坐标整数框 (x, y, w, h)，已做边界钳制
    """
    x, y, w, h = bbox
    if tile.resized:
        # resize 模式：宽高分别缩放回原图
        sx = orig_width / tile.size
        sy = orig_height / tile.size
        x, w = x * sx, w * sx
        y, h = y * sy, h * sy
    else:
        # 网格模式：叠加窗偏移
        x += tile.x
        y += tile.y

    x, y = int(round(x)), int(round(y))
    w, h = int(round(w)), int(round(h))
    # 钳制到图像范围内，过滤退化框
    x = max(0, min(x, orig_width - 1))
    y = max(0, min(y, orig_height - 1))
    w = max(1, min(w, orig_width - x))
    h = max(1, min(h, orig_height - y))
    return x, y, w, h


# ----------------------------------------------------------------------
# 动态切片（滑窗升级版）：仅在哨兵标记的高危区域附近开窗
# ----------------------------------------------------------------------
def plan_region_tiles(
    regions: List[Dict],
    width: int, height: int,
) -> Tuple[List[Tile], bool]:
    """
    依据 PatchCore 哨兵输出的高危区域规划【动态检测窗】。

    规则：
    * 小图（≤ tile_max_size）：整图 resize 单检，成本极低，不依赖哨兵；
    * 大图：只为高危区域开窗 —— 窗边长随区域尺寸自适应
      （clip(区域长边×tile_scale, tile_min_size, tile_max_size)），
      中心对齐区域中心并钳制到图内；区域中心已落在既有窗内的合并跳过；
      无高危区域 → 返回空窗列表（100% 正常区域直接放行，不再盲扫）。

    :param regions: [{"bbox":[x,y,w,h], "score":float}]（哨兵输出，按分降序）
    :return: (窗列表, 是否为整图resize单检模式)
    """
    max_size = settings.tile_max_size
    if max(width, height) <= max_size:
        return [Tile(index=0, x=0, y=0, size=max_size, resized=True)], True

    tiles: List[Tile] = []
    for region in regions:
        if len(tiles) >= settings.tile_max_count:
            break
        x, y, w, h = region["bbox"]
        cx, cy = x + w / 2.0, y + h / 2.0
        side = int(np.clip(
            max(w, h) * settings.tile_scale,
            min(settings.tile_min_size, min(width, height)),
            min(max_size, min(width, height)),
        ))
        tx = int(np.clip(cx - side / 2.0, 0, width - side))
        ty = int(np.clip(cy - side / 2.0, 0, height - side))
        # 合并：区域中心已覆盖于既有窗内（含 20% 余量）则不再开新窗
        covered = any(
            t.x - 0.2 * t.size <= cx <= t.x + t.size * 1.2
            and t.y - 0.2 * t.size <= cy <= t.y + t.size * 1.2
            for t in tiles
        )
        if covered:
            continue
        tiles.append(
            Tile(index=len(tiles), x=tx, y=ty, size=side, resized=False)
        )
    return tiles, False
