# -*- coding: utf-8 -*-
"""
管线分级诊断：定位漏检发生在哪一级
==================================
对指定图片逐级打印：
  ① 哨兵高危区域（数量/框/分数/阈值）
  ② 动态切片规划结果
  ③ YOLO-World 原始检测（含被阈值丢弃的低分框 Top10）
  ④ 候选关联结果（含哨兵兜底框）
  ⑤ 启发式裁判对比度值（对照阈值 50）

运行：cd backend && .conda-env/python.exe scripts/debug_pipeline.py [图名...]
不传图名则默认诊断当前 4 张漏检图 + 1 张退化框图。
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import cv2
import numpy as np
from PIL import Image

from app.config import settings
from app.services.vision.detector import DefectDetector
from app.services.vision.patchcore_sentinel import get_sentinel
from app.services.vision.sliding_window import crop_tile, plan_region_tiles
from app.services.vision.yolo_world import get_yolo_world

IMG_DIR = BACKEND / "data" / "_selftest_imgs"
DEFAULT = [
    "def_1745417514.jpg", "hard_0.jpg", "hard_1.jpg", "hard_3.jpg",
    "def_579802251.jpg",
]


def contrast_value(gray: np.ndarray, bbox) -> float:
    """复现 heuristic_filter 的对比度计算（仅诊断用）。"""
    H, W = gray.shape[:2]
    x, y, w, h = bbox
    pad = max(6, int(0.6 * max(w, h)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return -1.0
    ring = gray[y0:y1, x0:x1]
    ix0, iy0 = x + w // 4, y + h // 4
    inner = gray[iy0 : iy0 + max(2, h // 2), ix0 : ix0 + max(2, w // 2)]
    if inner.size == 0 or ring.size == 0:
        return -1.0
    ring_med = float(np.median(ring))
    p10, p90 = np.quantile(inner, [0.10, 0.90])
    return max(
        abs(float(inner.mean()) - float(ring.mean())),
        abs(float(p90) - ring_med),
        abs(ring_med - float(p10)),
    )


def main(names: list[str]) -> None:
    sentinel = get_sentinel()
    yolo = get_yolo_world()
    if sentinel is None or yolo is None:
        print("模型缺失")
        return
    assoc = DefectDetector._associate_candidates

    for name in names:
        path = IMG_DIR / name
        if not path.exists():
            print(f"== {name}: 文件不存在")
            continue
        image = Image.open(path).convert("RGB")
        W, H = image.size
        rgb = np.asarray(image, dtype=np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        print(f"\n======== {name} ({W}x{H}) ========")

        res = sentinel.scan(rgb)
        st = res.stats
        print(f"① 哨兵：thr={st['threshold']} patches={st['patches']} "
              f"bank={st['bank']} 区域数={len(res.regions)}")
        for r in res.regions:
            print(f"   region bbox={r['bbox']} score={r['score']}")

        tiles, resized = plan_region_tiles(res.regions, W, H)
        print(f"② 切片：resized={resized} 数量={len(tiles)}")
        for t in tiles:
            print(f"   tile idx={t.index} xy=({t.x},{t.y}) size={t.size}")

        dets = []
        for t in tiles:
            tile_img = crop_tile(image, t)
            tile_np = np.asarray(tile_img, dtype=np.uint8)
            for d in yolo.detect_tile(tile_np):
                dets.append((t, d))
        print(f"③ YOLO 原始检出：{len(dets)}")
        for t, d in dets:
            x, y, w, h = d["bbox"]
            ox = x + t.x if not t.resized else x * W / t.size
            oy = y + t.y if not t.resized else y * H / t.size
            print(f"   {d['defect_type']}({d['confidence']}) 切片内{x},{y},{w},{h}"
                  f" → 原图≈({int(ox)},{int(oy)})")

        # 低分框分布（阈值 0.01 之上的原始 top10，不看类型）
        canvas, r, px, py = yolo._letterbox(np.asarray(crop_tile(image, tiles[0]), np.uint8)) if tiles else (None, 0, 0, 0)
        if canvas is not None:
            pred = yolo._forward(canvas)
            scores = pred[4:].T
            if scores.size and scores.max() > 1.5:
                scores = 1.0 / (1.0 + np.exp(-scores))
            top = np.sort(scores.max(axis=1))[::-1][:10] if scores.size else []
            print(f"   全图单窗原始分 Top10：{np.round(top, 4).tolist()}")

        cands = assoc([{"defect_type": d["defect_type"],
                        "confidence": d["confidence"],
                        "bbox": [int(round(x * W / t.size)) if t.resized else x + t.x,
                                 int(round(y * H / t.size)) if t.resized else y + t.y,
                                 int(round(w * W / t.size)) if t.resized else w,
                                 int(round(h * H / t.size)) if t.resized else h]}
                       for t, d in dets], res.regions)
        print(f"④ 关联后候选：{len(cands)}")
        for c in cands:
            cv_ = contrast_value(gray, c["bbox"])
            keep = cv_ >= settings.judge_heuristic_contrast
            print(f"   {c['defect_type']}({c['confidence']}) bbox={c['bbox']}"
                  f" contrast={cv_:.1f} {'保留' if keep else '被裁判过滤'}")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT)
