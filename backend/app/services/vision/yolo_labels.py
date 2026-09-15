# -*- coding: utf-8 -*-
"""
YOLO-World 内联标签表（运行时与导出脚本共用的唯一标签定义源）
============================================================
YOLO-World 是"先提示后检测"的开放词汇检测模型：导出 ONNX 时通过
``set_classes`` 把类别文本嵌入固化进检测头权重（由
``scripts/prepare_models.py`` 完成内联导出），运行时只需
onnxruntime + numpy，无需 torch / CLIP。

* PROMPTS 顺序即 ONNX 输出的类别索引顺序，**不可随意调换**；
* 如需扩展新缺陷类型：在 LABELS 追加条目后重跑
  ``scripts/prepare_models.py`` 重新导出模型，运行时代码零改动。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# (英文文本提示, 中文名)。英文提示供 CLIP 文本空间编码，中文与
# 规则库 / 前端展示对齐
LABELS: List[Tuple[str, str]] = [
    ("scratch on metal surface", "划痕"),
    ("crack line on metal surface", "裂纹"),
    ("small round pore hole on metal surface", "气孔"),
    ("dent pit depression on metal surface", "凹坑"),
    ("peeling flaking coating layer", "起层"),
    ("chipped missing material defect", "缺肉"),
    ("stain smudge fingerprint on surface", "污渍"),
    ("rust corrosion on metal surface", "锈蚀"),
    ("bubble blister on surface", "气泡"),
    ("foreign particle inclusion on surface", "夹杂"),
    ("foreign object debris on surface", "异物"),
    ("bright spot blob on dark surface", "亮点"),
    ("oil grease mark on surface", "油污"),
    ("burn mark discoloration on surface", "烧痕"),
    ("burr sharp protruding edge", "毛刺"),
    ("surface defect damage", "缺陷"),
]

# 类别名 → 中文名（运行时把 ONNX 输出的类别索引映射回中文）
ZH_NAME_MAP: Dict[str, str] = {en: zh for en, zh in LABELS}


def prompt_list() -> List[str]:
    """导出脚本用：YOLO-World ``set_classes`` 需要的英文提示列表。"""
    return [en for en, _ in LABELS]


def zh_name(class_name: str) -> str:
    """把内联类别名映射为中文缺陷类型（未知类别原样返回）。"""
    return ZH_NAME_MAP.get(class_name, class_name)
