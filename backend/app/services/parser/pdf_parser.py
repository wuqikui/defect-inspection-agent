# -*- coding: utf-8 -*-
"""
PDF 解析器（pypdf）
===================
能力：
1. 逐页提取正文文字，并按“空行 / 换行”切分为段落块，每块记录所在页码；
2. 统计每页内嵌图片数量（XObject /Image），得到整份文档的图片数；
3. 识别“图片说明”：包含 “图1 / 图 1-2 / Fig. / Figure”等编号模式、
   且长度较短（通常 <120 字）的段落，标记为 caption 块，
   从而满足“准确识别文档中的文字描述与图片说明”。

注意：扫描件（纯图片 PDF）提取不到文字时给出明确提示，由上层决定是否拒绝。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from pypdf import PdfReader

from app.core.exceptions import ValidationError
from app.services.parser.base import BaseParser, ParsedDocument, TextBlock

# 图注编号正则：图1 / 图 1-2 / 图3.4 / Fig.1 / Figure 2-3
_CAPTION_RE = re.compile(
    r"^\s*(?:图\s*\d+|图\s*[一二三四五六七八九十]+|Fig(?:ure)?\.?\s*\d+)",
    re.IGNORECASE,
)
# 段落切分：连续换行
_PARA_SPLIT_RE = re.compile(r"\n{1,}")
# 合并行内被 PDF 断开的中文/英文行（单个换行通常只是排版折行）
def _join_wrapped_lines(text: str) -> str:
    # 两个及以上换行视为段落边界；单个换行直接拼接
    return re.sub(r"(?<!\n)\n(?!\n)", "", text)


class PdfParser(BaseParser):
    """PDF 文档解析器。"""

    def parse(self, file_path: Path) -> ParsedDocument:
        try:
            reader = PdfReader(str(file_path))
        except Exception as exc:  # noqa: BLE001
            raise ValidationError(f"PDF 文件已损坏或已加密，无法解析：{exc}") from exc

        blocks: List[TextBlock] = []
        image_count = 0

        for page_idx, page in enumerate(reader.pages, start=1):
            location = f"第{page_idx}页"

            # ---- 1) 图片统计 ----
            try:
                image_count += len(list(page.images))
            except Exception:  # noqa: BLE001
                # 某些 PDF 的 XObject 表不标准，统计失败不应阻断文字提取
                pass

            # ---- 2) 文字提取与段落切分 ----
            try:
                raw = page.extract_text() or ""
            except Exception:  # noqa: BLE001
                raw = ""
            raw = _join_wrapped_lines(raw)

            for para in _PARA_SPLIT_RE.split(raw):
                text = para.strip()
                if not text:
                    continue
                kind = "caption" if self._is_caption(text) else "text"
                blocks.append(TextBlock(text=text, location=location, kind=kind))

        if not blocks:
            raise ValidationError(
                "未能从 PDF 中提取到任何文字，可能是扫描件（纯图片 PDF），"
                "请上传带文字层的规则文档。"
            )

        return ParsedDocument(
            blocks=blocks,
            page_count=len(reader.pages),
            image_count=image_count,
        )

    @staticmethod
    def _is_caption(text: str) -> bool:
        """判断一个段落是否为图片说明：以图编号开头且整体较短。"""
        return bool(_CAPTION_RE.match(text)) and len(text) <= 120
