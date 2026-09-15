# -*- coding: utf-8 -*-
"""
解析器公共接口与数据结构
========================
无论 PDF 还是 DOCX，最终都被规整为统一的 ``ParsedDocument``：
* blocks：有序内容块（正文段落 + 图片说明），携带页码 / 段落定位；
* image_count：文档内嵌图片数量（用于“图文文档”识别与提示）；
* page_count：PDF 为页数，DOCX 为内容块数。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class TextBlock:
    """
    单个内容块。

    :property text:     文本内容（正文或图片说明文字）
    :property location: 溯源定位，如 "第3页" / "段落12"
    :property kind:     text=正文段落；caption=图片说明
    """

    text: str
    location: str
    kind: str = "text"


@dataclass
class ParsedDocument:
    """解析器统一输出。"""

    blocks: List[TextBlock] = field(default_factory=list)
    page_count: int = 0
    image_count: int = 0

    @property
    def full_text(self) -> str:
        """拼接全部文本（供摘要 / 规则抽取的兜底输入）。"""
        return "\n".join(b.text for b in self.blocks if b.text.strip())


class BaseParser(ABC):
    """解析器抽象基类：所有格式解析器必须实现 parse()。"""

    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocument:
        """
        解析磁盘上的文档。

        :param file_path: 文档绝对路径
        :return: ParsedDocument
        """
        raise NotImplementedError
