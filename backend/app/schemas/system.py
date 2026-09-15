# -*- coding: utf-8 -*-
"""系统 / 模型状态相关数据模型。"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class ProviderStatus(BaseModel):
    """单个模型提供方的配置状态。"""

    name: str
    label: str = Field(description="中文名")
    configured: bool = Field(description="是否已配置 API Key")
    supports_vision: bool = Field(description="是否具备图像理解能力")
    chat_model: str = ""
    selected: bool = Field(description="是否为当前生效的文本模型")


class ModelStatusResponse(BaseModel):
    """模型配置总览：给前端“设置”页使用。"""

    active_text_provider: str
    active_vision_provider: str
    offline_mode: bool = Field(
        description="是否运行在离线兜底模式（未配置任何可用 API Key）"
    )
    sam_available: bool = Field(description="SAM 是否可用（已安装依赖且权重存在）")
    sam_device: str = ""
    providers: List[ProviderStatus]
