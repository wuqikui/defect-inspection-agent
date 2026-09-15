# -*- coding: utf-8 -*-
"""
LLM 适配基类与通用工具
======================
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# 对话消息结构：{"role": "system"/"user"/"assistant", "content": ...}
Message = Dict[str, Any]

# 规则抽取 / 冲突检测 / 图像判定共用的系统提示前缀
_SYSTEM_COMMON = (
    "你是一名资深的工业表面缺陷检测专家，熟悉气孔、夹渣、划痕、起层、缺肉、"
    "裂纹等缺陷的判定标准。请严格依据用户给出的规则文档作答，"
    "不要臆造文档中不存在的阈值或结论。"
)


def parse_json_from_text(text: str) -> Any:
    """
    从模型输出中容错地解析 JSON：
    1. 去除 ```json ... ``` 代码块围栏；
    2. 截取第一个 '[' / '{' 到最后一个 ']' / '}' 的子串；
    3. 仍失败则尝试把尾随逗号等常见小瑕疵修复后再解析。

    :raises ValueError: 无法解析时抛出（由上层决定回退策略）
    """
    if not text:
        raise ValueError("模型返回为空")

    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()

    # 优先整体解析
    for candidate in (cleaned, _slice_json(cleaned, "{"), _slice_json(cleaned, "[")):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # 修复尾逗号： {"a":1,} / [1,2,]
            fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                continue
    raise ValueError(f"无法从模型输出解析 JSON：{text[:200]}")


def _slice_json(text: str, opener: str) -> str:
    """截取从 opener 到对应闭合符号之间的子串。"""
    closer = "}" if opener == "{" else "]"
    start = text.find(opener)
    end = text.rfind(closer)
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return ""


class BaseLLM(ABC):
    """所有 LLM 适配器的抽象基类。"""

    name: str = "base"
    label: str = "基础模型"
    supports_vision: bool = False

    # ------------------------------------------------------------------
    # 原子能力
    # ------------------------------------------------------------------
    @abstractmethod
    async def chat(
        self,
        messages: List[Message],
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        """
        纯文本对话。

        :param messages: 消息列表
        :param temperature: 采样温度（规则抽取等任务用低温保证稳定）
        :param json_mode: 是否要求模型以 JSON 输出（支持则下发 response_format）
        :return: 助手文本
        """
        raise NotImplementedError

    async def vision_chat(
        self,
        prompt: str,
        image_jpeg_b64: str,
        temperature: float = 0.1,
    ) -> str:
        """多模态对话（输入 JPEG base64）。不支持视觉的模型不应被调用。"""
        raise NotImplementedError(f"{self.label} 不支持图像理解")

    # ------------------------------------------------------------------
    # 业务级能力：默认实现走通用 Prompt + JSON 解析；离线模型可覆写
    # ------------------------------------------------------------------
    async def summarize_document(self, doc_name: str, text: str) -> str:
        """生成 100 字以内的文档内容摘要。"""
        messages = [
            {"role": "system", "content": _SYSTEM_COMMON},
            {
                "role": "user",
                "content": (
                    f"以下是规则文档《{doc_name}》的部分内容，请用不超过100字概括"
                    f"其涵盖的缺陷类型与核心判定标准：\n\n{text[:6000]}"
                ),
            },
        ]
        return (await self.chat(messages, temperature=0.2)).strip()

    async def extract_rules(self, doc_name: str, text: str) -> List[Dict[str, Any]]:
        """
        从文档文本中抽取结构化检测规则。

        :return: [{"defect_type","title","content","severity","source_location"}]
        """
        prompt = (
            f"文档《{doc_name}》内容如下。请抽取其中所有“可执行的缺陷判定规则”，"
            "每条规则对应一个明确的缺陷类型与判定标准（含尺寸/数量/形态阈值）。\n"
            "仅输出 JSON 数组，每个元素字段：\n"
            '- defect_type: 缺陷类型（如 "气孔"、"划痕"）\n'
            '- title: 不超过20字的规则标题\n'
            "- content: 规则完整表述，保留所有数值与单位\n"
            '- severity: 严重等级，没有则填 ""\n'
            '- source_location: 尽量标注页码/章节，没有则填 ""\n'
            "若文档没有任何规则，输出 []。\n\n"
            f"文档内容：\n{text[:12000]}"
        )
        raw = await self.chat(
            [
                {"role": "system", "content": _SYSTEM_COMMON},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            json_mode=True,
        )
        data = parse_json_from_text(raw)
        if not isinstance(data, list):
            raise ValueError("规则抽取结果不是 JSON 数组")
        return _normalize_rules(data)

    async def detect_rule_conflicts(
        self, rules: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        在跨文档规则集合中识别逻辑冲突。

        :param rules: [{"id","doc_name","defect_type","content","source_location"}]
        :return: [{"defect_type","description","rule_a_id","rule_b_id",
                   "content_a","content_b","doc_a_name","doc_b_name",
                   "source_a","source_b"}]
        """
        compact = [
            {
                "id": r["id"],
                "doc": r.get("doc_name", ""),
                "defect_type": r.get("defect_type", ""),
                "rule": r.get("content", ""),
                "source": r.get("source_location", ""),
            }
            for r in rules
        ]
        prompt = (
            "下面是来自多份规则文档的检测规则（JSON）。\n"
            "请识别“同一缺陷类型上相互矛盾、无法同时成立”的规则对，"
            "例如同一缺陷的合格阈值一个是 ≤0.5mm 另一个是 ≤1.0mm、"
            "或一个判合格另一个判不合格。表述不同但含义一致的不算冲突。\n"
            "只输出 JSON 数组，每个冲突包含字段：\n"
            "defect_type, description(冲突点说明), rule_a_id, rule_b_id, "
            "content_a, content_b, doc_a_name, doc_b_name, source_a, source_b。\n"
            "没有冲突时输出 []。\n\n规则集合：\n"
            + json.dumps(compact, ensure_ascii=False)
        )
        raw = await self.chat(
            [
                {"role": "system", "content": _SYSTEM_COMMON},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            json_mode=True,
        )
        data = parse_json_from_text(raw)
        return data if isinstance(data, list) else []

    async def judge_tile(
        self,
        rule_context: str,
        image_jpeg_b64: str,
        known_defect_types: List[str],
    ) -> List[Dict[str, Any]]:
        """
        判定单个滑窗切片。返回 JSON：
        [{"defect_type","confidence","bbox":[x,y,w,h](切片内像素坐标),
          "description"}]
        """
        prompt = (
            "这是一张工业产品表面照片的方形局部切片。请依据以下检测规则，"
            "找出切片中的缺陷。注意区分真实缺陷与光影、水渍、磨削痕迹、停光点等干扰。\n"
            f"已知缺陷类型：{', '.join(known_defect_types) or '气孔、划痕、起层、缺肉、裂纹'}。\n"
            "仅输出 JSON 数组，每个缺陷字段：\n"
            '- defect_type: 缺陷类型\n'
            "- confidence: 0~1 置信度\n"
            "- bbox: 缺陷在本切片中的边界框 [x, y, w, h]（像素坐标，"
            "左上角为原点，坐标必须落在当前方形切片范围内）\n"
            "- description: 一句话外观描述\n"
            "没有缺陷时输出 []。\n\n"
            f"检测规则依据：\n{rule_context}"
        )
        raw = await self.vision_chat(prompt, image_jpeg_b64, temperature=0.05)
        data = parse_json_from_text(raw)
        return data if isinstance(data, list) else []


def _normalize_rules(items: List[Any]) -> List[Dict[str, Any]]:
    """清洗 LLM 抽取出的规则字段，丢弃无法识别的脏项。"""
    normalized: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        dtype = str(item.get("defect_type", "")).strip()
        if not content or not dtype:
            continue
        normalized.append(
            {
                "defect_type": dtype[:50],
                "title": str(item.get("title", ""))[:100],
                "content": content[:2000],
                "severity": str(item.get("severity", ""))[:50],
                "source_location": str(item.get("source_location", ""))[:100],
            }
        )
    return normalized
