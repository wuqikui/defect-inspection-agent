# -*- coding: utf-8 -*-
"""文档管理相关数据模型。"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    """文档列表 / 上传成功后返回给前端的文档信息。"""

    id: str
    original_name: str = Field(description="原始文件名")
    ext: str
    size_bytes: int
    status: str = Field(description="parsed=解析成功 / failed=解析失败")
    page_count: int = Field(description="页数（PDF 为页数，DOCX 为段落块数）")
    chunk_count: int = Field(description="入库的文本片段数量")
    summary: str = Field(description="文档内容摘要")
    uploaded_at: str


class DocumentListResponse(BaseModel):
    """文档列表响应（同时回传数量上限，供前端控制上传按钮）。"""

    items: List[DocumentOut]
    total: int
    max_documents: int


class RuleOut(BaseModel):
    """从文档中抽取的结构化检测规则。"""

    id: str
    doc_id: str
    doc_name: str = Field(default="", description="来源文档名（联表补充）")
    defect_type: str = Field(description="缺陷类型")
    title: str
    content: str
    severity: str
    source_location: str = Field(description="溯源位置：PDF 页码 / DOCX 段落")
    status: str = Field(description="active=生效中 / superseded=冲突中被淘汰")


class RuleListResponse(BaseModel):
    items: List[RuleOut]
    total: int
