# -*- coding: utf-8 -*-
"""
深度模型管线端到端自测（离线、无 API Key）
==========================================
对 data/_selftest_imgs 真实测试图逐张执行【生产检测器】完整管线：

    DefectDetector.detect()
      = PatchCore 哨兵扫描 → 动态切片 → YOLO-World 定位
        → 候选关联+哨兵兜底 → 离线启发式裁判 → OpenCV ROI 轮廓抠图

预期效果：
  * def_*（含缺陷图）：检出 ≥1 处缺陷；
  * miss_0（历史漏检长图）：能标出缺陷位置。

运行：
  cd backend && .conda-env/python.exe scripts/e2e_dnn_test.py
输出：
  backend/data/results/_e2e/<图名>_annotated.jpg + 控制台统计表
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from PIL import Image

from app.services.vision.annotator import save_annotated
from app.services.vision.detector import defect_detector

IMG_DIR = BACKEND / "data" / "_selftest_imgs"
OUT_DIR = BACKEND / "data" / "results" / "_e2e"

RULE_CTX = "检测规则（测试用）：气孔、划痕、夹杂、污渍、裂纹等表面缺陷均判不合格。"
KNOWN_TYPES = ["气孔", "划痕", "夹杂", "污渍", "裂纹", "亮点", "未知缺陷"]


async def run_one(path: Path) -> dict:
    t_all = time.perf_counter()
    image = Image.open(path).convert("RGB")
    W, H = image.size

    stages: list[str] = []

    async def progress(pct: int, stage: str) -> None:
        stages.append(f"{pct}%:{stage}")

    result = await defect_detector.detect(
        image, RULE_CTX, KNOWN_TYPES, progress=progress
    )
    judged = result["defects"]

    # 标注落盘
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{path.stem}_annotated.jpg"
    await asyncio.to_thread(save_annotated, image, judged, out_path)

    return {
        "name": path.name,
        "size": f"{W}x{H}",
        "tiles": result.get("tile_count", 0),
        "engine": result.get("engine", "?"),
        "sentinel": result.get("sentinel", {}),
        "defects": len(judged),
        "types": [
            f'{d["defect_type"]}({d["confidence"]:.2f})@{d["bbox"]}'
            for d in judged
        ],
        "total_s": round(time.perf_counter() - t_all, 2),
        "out": out_path.name,
    }


async def amain() -> int:
    files = sorted(IMG_DIR.glob("*.jpg"))
    if not files:
        print(f"无测试图：{IMG_DIR}")
        return 1
    rows = []
    for p in files:
        try:
            rows.append(await run_one(p))
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            rows.append({"name": p.name, "error": repr(exc)})
        r = rows[-1]
        if "error" in r:
            print(f"{r['name']}:  ERROR {r['error']}")
        else:
            st = r.get("sentinel") or {}
            print(
                f"{r['name']:<22} {r['size']:<12}"
                f" 切片{r['tiles']:>2} 引擎{r['engine']:<4}"
                f" 哨兵[{st.get('bands', '-')}带/{st.get('patches', '-')}patch"
                f"/库{st.get('bank', '-')}/thr{st.get('threshold', '-')}"
                f"/{st.get('elapsed_s', '-')}s]"
                f" 缺陷{r['defects']:>2} 总{r['total_s']:>6}s"
            )
            for t in r["types"]:
                print(f"    - {t}")

    print("\n==== 统计 ====")
    defect_imgs = [r for r in rows if "error" not in r and r["name"].startswith(("def_", "hard_", "miss"))]
    hit = sum(1 for r in defect_imgs if r["defects"] > 0)
    print(f"缺陷图命中：{hit}/{len(defect_imgs)}")
    print(f"标注图输出：{OUT_DIR}")
    return 0 if hit == len(defect_imgs) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
