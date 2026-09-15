# -*- coding: utf-8 -*-
"""规则冲突检测与解决相关数据模型。"""
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class ConflictOut(BaseModel):
    """单条规则冲突：前端逐条呈现 A / B 两种说法供用户裁决。"""

    id: str
    defect_type: str = Field(description="冲突主题（通常是缺陷类型）")
    description: str = Field(description="冲突点的自然语言描述")
    doc_a_name: str
    doc_b_name: str
    content_a: str = Field(description="文档 A 中的规则表述")
    content_b: str = Field(description="文档 B 中的规则表述")
    source_a: str = Field(description="规则 A 的页码/段落溯源")
    source_b: str = Field(description="规则 B 的页码/段落溯源")
    status: Literal["pending", "resolved"]
    resolution_choice: str = Field(description="A / B / CUSTOM")
    resolution_text: str
    created_at: str
    resolved_at: str = ""


class ConflictListResponse(BaseModel):
    items: List[ConflictOut]
    pending_count: int = Field(description="未解决冲突数；为 0 时才允许发起检测")
    resolved_count: int


class ConflictResolveRequest(BaseModel):
    """用户对单条冲突的裁决。"""

    choice: Literal["A", "B", "CUSTOM"] = Field(
        description="A=采纳文档A规则；B=采纳文档B规则；CUSTOM=采用用户自定义表述"
    )
    custom_text: str = Field(
        default="",
        max_length=2000,
        description="choice=CUSTOM 时必填：用户确认后的统一规则表述",
    )
