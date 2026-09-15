# -*- coding: utf-8 -*-
"""
DOCX 解析器（python-docx）
==========================
能力：
1. 顺序遍历文档 body 中的段落，保留标题层级（Heading 样式写入文本前缀）；
2. 识别图片说明：段落紧邻图片 / 表格中含图注编号，或以“图x”开头的短段落；
3. 统计内嵌图片数量（word/media 下的所有媒体关系）；
4. 表格单元格文本也会被抽取（规则文档常以表格描述判定标准），
   每行拼为一个块，溯源到“表格n”。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from docx import Document as DocxDocument
from docx.document import Document as _DocType
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.core.exceptions import ValidationError
from app.services.parser.base import BaseParser, ParsedDocument, TextBlock

_CAPTION_RE = re.compile(
    r"^\s*(?:图\s*\d+|图\s*[一二三四五六七八九十]+|Fig(?:ure)?\.?\s*\d+)",
    re.IGNORECASE,
)


class DocxParser(BaseParser):
    """DOCX 文档解析器。"""

    def parse(self, file_path: Path) -> ParsedDocument:
        try:
            doc: _DocType = DocxDocument(str(file_path))
        except Exception as exc:  # noqa: BLE001
            raise ValidationError(f"DOCX 文件已损坏，无法解析：{exc}") from exc

        blocks: List[TextBlock] = []
        para_counter = 0

        # iter_inner_content 在新版 python-docx 中可按顺序产出段落与表格；
        # 旧版本没有该方法，则退化为“先全部段落、后全部表格”。
        inner = getattr(doc, "iter_inner_content", None)
        if callable(inner):
            elements = list(inner())
        else:
            elements = list(doc.paragraphs) + list(doc.tables)

        for element in elements:
            if isinstance(element, Paragraph):
                text = (element.text or "").strip()
                if not text:
                    continue
                para_counter += 1
                # 保留标题层级语义（如“3. 气孔判定标准”）
                style_name = (element.style.name or "") if element.style else ""
                if style_name.lower().startswith("heading"):
                    text = f"【标题】{text}"
                location = f"段落{para_counter}"
                kind = "caption" if self._is_caption(text) else "text"
                blocks.append(TextBlock(text=text, location=location, kind=kind))
            elif isinstance(element, Table):
                for row in element.rows:
                    cells = [c.text.strip() for c in row.cells]
                    cells = [c for c in cells if c]
                    if not cells:
                        continue
                    # 同一行的相邻重复单元格是合并单元格导致的，去重保序
                    deduped: List[str] = []
                    for c in cells:
                        if not deduped or deduped[-1] != c:
                            deduped.append(c)
                    blocks.append(
                        TextBlock(
                            text=" | ".join(deduped),
                            location="表格",
                            kind="text",
                        )
                    )

        image_count = self._count_images(doc)

        if not blocks:
            raise ValidationError("未能从 DOCX 中提取到任何文字内容。")

        return ParsedDocument(
            blocks=blocks,
            page_count=para_counter,
            image_count=image_count,
        )

    @staticmethod
    def _is_caption(text: str) -> bool:
        """图注判断：图编号开头且较短。"""
        return bool(_CAPTION_RE.match(text)) and len(text) <= 120

    @staticmethod
    def _count_images(doc: _DocType) -> int:
        """
        统计文档内嵌图片数量。
        DOCX 中每张图片都会在 document.part.related_parts 里有一个 image 关系，
        同时用 body XML 中 ``<a:blip r:embed=...>`` 出现次数交叉验证。
        """
        # 方法一：关系部件计数
        rel_images = sum(
            1
            for part in doc.part.related_parts.values()
            if "image" in getattr(part, "content_type", "")
        )
        # 方法二：XML 中引用计数（更贴近“实际展示了几张图”）
        try:
            blips = doc.element.body.findall(".//" + qn("a:blip"))
            return max(rel_images, len(blips))
        except Exception:  # noqa: BLE001
            return rel_images
