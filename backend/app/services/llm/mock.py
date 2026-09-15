# -*- coding: utf-8 -*-
"""
离线兜底“模型”（MockLLM）
=========================
当未配置任何在线 API Key 时启用，保证系统在完全离线环境仍可跑通
文档理解 → 规则抽取 → 冲突检测主链路（视觉检测由经典 CV 算法兜底）。

它不是“假装的大模型”，而是一套确定性的工业规则文本启发式：
* 摘要：取包含缺陷关键词的首句；
* 规则抽取：按行扫描，命中缺陷类型词且包含判定性表述（阈值数字 /
  不允许 / 判废 等）的行收为规则；
* 冲突检测：跨文档、同缺陷类型的规则对中，若出现“同单位不同数值”的
  量化阈值，则判定为逻辑冲突。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from app.services.llm.base import BaseLLM, Message

# 常见工业表面缺陷类型关键词（命中即作为候选缺陷类型）
_DEFECT_KEYWORDS = [
    "气孔", "气穴", "针孔", "孔洞", "凹坑", "划痕", "划伤", "刮伤",
    "起层", "分层", "缺肉", "裂纹", "裂缝", "开裂", "夹渣", "夹杂",
    "麻点", "斑点", "凸起", "鼓包", "变形", "色差", "锈蚀", "毛刺",
]

# 判定性表述：出现这些词说明该行是一条“可执行规则”
_JUDGE_WORDS = [
    "不大于", "不小于", "不超过", "不允许", "不得", "不能", "禁止",
    "大于", "小于", "超过", "允许", "判废", "不合格", "合格", "判定",
    "以内", "以下", "以上", "直径", "深度", "长度", "面积", "数量",
]

# 阈值表达式：数字 + 可选小数 + 单位
_THRESHOLD_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mm|MM|μm|um|µm|个|条|处|%|％)"
)


class MockLLM(BaseLLM):
    """离线启发式实现。"""

    name = "mock"
    label = "离线规则引擎"
    supports_vision = False  # 视觉走 CV 兜底，见 vision_detector.py

    async def chat(
        self,
        messages: List[Message],
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        # 纯对话能力在业务中只用于摘要/规则/冲突，均已覆写；
        # 若被直接调用，返回最后一条用户消息的截断，保证不抛异常。
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return str(msg.get("content", ""))[:200]
        return ""

    async def vision_chat(self, prompt: str, image_jpeg_b64: str,
                          temperature: float = 0.1) -> str:
        raise NotImplementedError("离线模式不使用多模态模型，改由 CV 算法检测")

    # ------------------------------------------------------------------
    async def summarize_document(self, doc_name: str, text: str) -> str:
        for line in re.split(r"[\n。；;]", text):
            line = line.strip()
            if line and any(k in line for k in _DEFECT_KEYWORDS):
                return f"本文档规定了{doc_name}中{line[:80]}"
        return f"《{doc_name}》为缺陷检测规则文档，共{len(text)}字。"

    async def extract_rules(self, doc_name: str, text: str) -> List[Dict[str, Any]]:
        """按行扫描，抽取“缺陷类型 + 判定表述”的规则。"""
        rules: List[Dict[str, Any]] = []
        seen = set()
        # 按换行 / 分号 / 句号切分，兼顾表格行（| 分隔先替换为顿号）
        lines = re.split(r"[\n；;]", text.replace("|", "，"))
        for raw_line in lines:
            line = raw_line.strip(" 　\t。.")
            if len(line) < 4:
                continue
            matched_type = next((k for k in _DEFECT_KEYWORDS if k in line), None)
            if not matched_type:
                continue
            is_judgable = any(w in line for w in _JUDGE_WORDS)
            has_number = bool(_THRESHOLD_RE.search(line))
            if not (is_judgable or has_number):
                continue
            key = (matched_type, line[:40])
            if key in seen:
                continue
            seen.add(key)
            rules.append(
                {
                    "defect_type": matched_type,
                    "title": line[:20],
                    "content": line[:500],
                    "severity": "",
                    "source_location": "",
                }
            )
        return rules

    async def detect_rule_conflicts(
        self, rules: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        冲突启发式：不同文档、同缺陷类型、且阈值“同单位不同数值”的规则对。
        """
        conflicts: List[Dict[str, Any]] = []
        seen_pairs = set()

        for i, a in enumerate(rules):
            for b in rules[i + 1 :]:
                if a.get("doc_name") == b.get("doc_name"):
                    continue  # 只检测跨文档冲突
                if a.get("defect_type") != b.get("defect_type"):
                    continue
                ta = _normalize_thresholds(a.get("content", ""))
                tb = _normalize_thresholds(b.get("content", ""))
                shared = sorted(set(ta) & set(tb))
                # 同一单位出现不同数值 => 阈值矛盾
                clash = [u for u in shared if ta[u] != tb[u]]
                if not clash:
                    continue
                pair_key = tuple(sorted([a["id"], b["id"]]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                units = "、".join(clash)
                conflicts.append(
                    {
                        "defect_type": a.get("defect_type", ""),
                        "description": (
                            f"两份文档对“{a.get('defect_type')}”的合格阈值"
                            f"（单位：{units}）规定不一致，需要人工确认以哪一份为准。"
                        ),
                        "rule_a_id": a["id"],
                        "rule_b_id": b["id"],
                        "doc_a_name": a.get("doc_name", ""),
                        "doc_b_name": b.get("doc_name", ""),
                        "content_a": a.get("content", ""),
                        "content_b": b.get("content", ""),
                        "source_a": a.get("source_location", ""),
                        "source_b": b.get("source_location", ""),
                    }
                )
        return conflicts


def _normalize_thresholds(content: str) -> Dict[str, float]:
    """
    提取规则文本中的“单位 → 数值”映射。
    同一单位出现多个数值时取最小值（通常是最严阈值），用于比较矛盾。
    μm/um/µm 统一归一为 mm 数值（/1000）。
    """
    found: Dict[str, List[float]] = {}
    for value, unit in _THRESHOLD_RE.findall(content):
        num = float(value)
        unit = unit.lower()
        if unit in {"um", "μm", "µm"}:
            num, unit = num / 1000.0, "mm"
        if unit == "％":
            unit = "%"
        found.setdefault(unit, []).append(num)
    return {u: min(vs) for u, vs in found.items()}
