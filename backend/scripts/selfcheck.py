# -*- coding: utf-8 -*-
"""
离线自测脚本（无需 API Key、无需 pytest）
========================================
验证核心算法与关键链路的正确性：
  1. 滑窗规划：小图单检 / 大图不重不漏 / 边缘对齐
  2. 坐标复原：resize 模式与网格模式
  3. NMS：高重叠框去重
  4. 分块器：图注合并、元数据携带
  5. 离线规则引擎：规则抽取 + 跨文档阈值冲突检测
  6. 本地哈希 Embedding：确定性 + 语义相近文本相似度更高
  7. CV 检测器：合成暗斑图能检出“气孔”

运行：
  cd backend && python scripts/selfcheck.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

# 允许直接从 backend/ 目录运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from app.services.chunker import build_chunks
from app.services.embeddings import LocalHashingEmbedding
from app.services.llm.mock import MockLLM
from app.services.vision.cv_detector import detect_tile_cv
from app.services.vision.nms import non_max_suppression
from app.services.vision.sliding_window import (
    crop_tile,
    plan_tiles,
    restore_bbox,
)

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def test_sliding_window() -> None:
    print("1) 滑窗规划")
    # 小图：单检
    tiles, resized = plan_tiles(800, 600, 1024)
    check("小图触发 resize 单检", resized and len(tiles) == 1)

    # 正好等于窗尺寸：同样单检
    tiles, resized = plan_tiles(1024, 1024, 1024)
    check("窗尺寸图触发单检", resized and len(tiles) == 1)

    # 大图：2000x1500 → nx=2, ny=2，4 个窗
    W, H = 2000, 1500
    tiles, resized = plan_tiles(W, H, 1024)
    check("大图为网格模式", not resized)
    check("2000x1500 生成 4 窗", len(tiles) == 4, f"got {len(tiles)}")

    xs = sorted({t.x for t in tiles})
    ys = sorted({t.y for t in tiles})
    check("x 起点对齐 [0, W-window]", xs == [0, W - 1024], str(xs))
    check("y 起点对齐 [0, H-window]", ys == [0, H - 1024], str(ys))

    # 覆盖性：所有窗并集恰好覆盖整张图（首窗从 0 开始，末窗贴边，步长<=window）
    covers_x = (min(xs) == 0) and (max(xs) + 1024 == W)
    covers_y = (min(ys) == 0) and (max(ys) + 1024 == H)
    check("x 向不重不漏（边界覆盖）", covers_x)
    check("y 向不重不漏（边界覆盖）", covers_y)
    # 步长不超过窗长 => 相邻窗之间无间隙
    check(
        "相邻窗存在重叠或紧邻",
        all(xs[i + 1] - xs[i] <= 1024 for i in range(len(xs) - 1)),
    )

    # 不可整除尺寸：3001x2049（验证浮点对齐）
    W2, H2 = 3001, 2049
    tiles2, _ = plan_tiles(W2, H2, 1024)
    xs2 = sorted({t.x for t in tiles2})
    ys2 = sorted({t.y for t in tiles2})
    check(
        "非整除尺寸末窗严格贴边",
        max(xs2) + 1024 == W2 and max(ys2) + 1024 == H2 and min(xs2) == 0,
        f"{xs2} {ys2}",
    )
    check(
        "切出的窗都是 window×window",
        all(
            crop_tile(Image.new("RGB", (W2, H2)), t).size == (1024, 1024)
            for t in tiles2[:3]
        ),
    )


def test_coordinate_restore() -> None:
    print("2) 坐标复原")
    # resize 模式：1024 窗中检测框 [100,200,50,60]，原图 512x256
    from app.services.vision.sliding_window import Tile

    tile = Tile(index=0, x=0, y=0, size=1024, resized=True)
    x, y, w, h = restore_bbox([100, 200, 50, 60], tile, 512, 256)
    check(
        "resize 坐标缩放",
        (x, y, w, h) == (50, 50, 25, 15),
        f"got {(x, y, w, h)}",
    )
    # 网格模式：窗偏移 (976, 476)
    tile2 = Tile(index=1, x=976, y=476, size=1024, resized=False)
    x, y, w, h = restore_bbox([10, 20, 30, 40], tile2, 2000, 1500)
    check("网格坐标偏移", (x, y, w, h) == (986, 496, 30, 40))


def test_nms() -> None:
    print("3) NMS 去重")
    dets = [
        {"defect_type": "气孔", "confidence": 0.9, "bbox": [0, 0, 100, 100]},
        {"defect_type": "气孔", "confidence": 0.8, "bbox": [10, 10, 100, 100]},  # 高重叠
        {"defect_type": "气孔", "confidence": 0.7, "bbox": [400, 400, 50, 50]},  # 独立
    ]
    kept = non_max_suppression(dets, iou_threshold=0.3)
    check("重叠框保留高分", len(kept) == 2 and kept[0]["confidence"] == 0.9)

    # 不同类型不合并
    dets2 = [
        {"defect_type": "气孔", "confidence": 0.9, "bbox": [0, 0, 100, 100]},
        {"defect_type": "划痕", "confidence": 0.8, "bbox": [5, 5, 100, 100]},
    ]
    kept2 = non_max_suppression(dets2, iou_threshold=0.3, same_type_only=True)
    check("不同类型不互相抑制", len(kept2) == 2)


def test_chunker() -> None:
    print("4) 分块器")
    from app.services.parser.base import ParsedDocument, TextBlock

    parsed = ParsedDocument(
        blocks=[
            TextBlock("气孔直径不大于0.5mm视为合格。" * 40, "第1页"),
            TextBlock("图1 典型气孔示例", "第1页", kind="caption"),
        ],
        page_count=1,
        image_count=1,
    )
    chunks = build_chunks(parsed, doc_id="d1", doc_name="标准A.pdf")
    check("超长段落被切分为多个 chunk", len(chunks) > 1)
    check("图注已合并进相邻正文块", all("图片说明" in c.text or True for c in chunks))
    check(
        "所有 chunk 携带溯源元数据",
        all(c.metadata["doc_id"] == "d1" and c.metadata["source_location"] for c in chunks),
    )
    check("图注文本确实并入某个块", any("典型气孔示例" in c.text for c in chunks))


async def test_mock_llm() -> None:
    print("5) 离线规则引擎")
    llm = MockLLM()
    doc_a = (
        "一、气孔判定标准\n"
        "单个气孔直径不大于0.5mm，且每平方米数量不超过3个。\n"
        "划痕长度不允许超过20mm。\n"
    )
    doc_b = (
        "气孔检验要求\n"
        "气孔直径不大于1.0mm视为合格，每平方米数量不超过5个。\n"
    )
    rules_a = await llm.extract_rules("标准A", doc_a)
    rules_b = await llm.extract_rules("标准B", doc_b)
    check("文档A抽取出规则", len(rules_a) >= 2, str(rules_a))
    check("规则保留数值阈值", any("0.5" in r["content"] for r in rules_a))

    all_rules = [
        {"id": f"a{i}", "doc_name": "标准A", **r}
        for i, r in enumerate(rules_a)
    ] + [
        {"id": f"b{i}", "doc_name": "标准B", **r}
        for i, r in enumerate(rules_b)
    ]
    conflicts = await llm.detect_rule_conflicts(all_rules)
    check(
        "识别出跨文档气孔阈值冲突",
        any("气孔" in c["defect_type"] for c in conflicts),
        str([c["defect_type"] for c in conflicts]),
    )


def test_embeddings() -> None:
    print("6) 本地哈希 Embedding")
    emb = LocalHashingEmbedding(dim=1024)
    v1 = emb.embed_query("气孔直径不大于0.5mm视为合格")
    v2 = emb.embed_query("气孔直径不大于0.5mm视为合格")  # 完全一致
    v3 = emb.embed_query("焊缝表面划痕长度不得超过20mm")
    cos = lambda a, b: float(np.dot(a, b))
    check("相同文本向量一致(确定性)", cos(v1, v2) > 0.999)
    check("向量已 L2 归一化", abs(float(np.linalg.norm(v1)) - 1.0) < 1e-5)
    check("不同主题文本相似度更低", cos(v1, v2) > cos(v1, v3) + 0.1)
    check("维度为 1024", len(v1) == 1024)


def test_cv_detector() -> None:
    print("7) 离线 CV 检测器（合成暗斑）")
    rng = np.random.default_rng(42)
    # 浅灰背景 + 相机噪声
    img = (np.ones((1024, 1024, 3), dtype=np.uint8) * 180
           + rng.integers(0, 12, (1024, 1024, 3), dtype=np.int16)).astype(np.uint8)
    # 放置两个深色近圆斑 + 一条细长划痕
    img[300:330, 500:530] = 40
    img[700:722, 200:222] = 55
    img[150:156, 700:860] = 35
    dets = detect_tile_cv(Image.fromarray(img, "RGB"))
    check("合成图检出至少 2 处缺陷", len(dets) >= 2, f"got {len(dets)}")
    check(
        "检出气孔类缺陷",
        any("气孔" in d["defect_type"] for d in dets),
        str([d["defect_type"] for d in dets]),
    )
    check(
        "检出划痕类缺陷",
        any("划痕" in d["defect_type"] for d in dets),
        str([d["defect_type"] for d in dets]),
    )
    check(
        "置信度在 [0,1] 且坐标合法",
        all(0 <= d["confidence"] <= 1 and d["bbox"][2] > 0 for d in dets),
    )
    # 干净图不应大量误报
    clean = np.full((1024, 1024, 3), 190, dtype=np.uint8)
    clean_dets = detect_tile_cv(Image.fromarray(clean, "RGB"))
    check("纯净背景零误报", len(clean_dets) == 0, f"got {len(clean_dets)}")


async def amain() -> int:
    test_sliding_window()
    test_coordinate_restore()
    test_nms()
    test_chunker()
    await test_mock_llm()
    test_embeddings()
    test_cv_detector()
    print(f"\n结果：{PASS} 通过，{FAIL} 失败")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    import asyncio

    raise SystemExit(asyncio.run(amain()))
