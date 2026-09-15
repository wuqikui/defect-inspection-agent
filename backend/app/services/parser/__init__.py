# -*- coding: utf-8 -*-
"""
文档解析器子包
==============
* base.BaseParser      —— 统一解析接口与解析结果数据结构
* pdf_parser.PdfParser —— 基于 pypdf 的 PDF 解析
* docx_parser.DocxParser —— 基于 python-docx 的 DOCX 解析
* parse_document()     —— 工厂方法：按扩展名分派
"""
from __future__ import annotations

from pathlib import Path

from app.core.exceptions import UnsupportedFileError
from app.services.parser.base import ParsedDocument
from app.services.parser.docx_parser import DocxParser
from app.services.parser.pdf_parser import PdfParser


def parse_document(file_path: Path, ext: str) -> ParsedDocument:
    """
    根据扩展名选择解析器。

    :param file_path: 文档在服务器上的绝对路径
    :param ext: 小写扩展名（.pdf / .docx）
    :return: ParsedDocument 统一解析结果
    """
    if ext == ".pdf":
        return PdfParser().parse(file_path)
    if ext == ".docx":
        return DocxParser().parse(file_path)
    raise UnsupportedFileError(f"不支持的文档类型：{ext}")


__all__ = ["parse_document", "ParsedDocument", "PdfParser", "DocxParser"]
