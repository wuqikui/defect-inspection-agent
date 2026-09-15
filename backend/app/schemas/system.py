# -*- coding: utf-8 -*-
"""系统 / 模型状态相关数据模型。"""
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class ProviderStatus(BaseModel):
    """单个模型提供方的配置状态。"""

    name: str
    label: str = Field(description="中文名")
    configured: bool = Field(description="是否已配置 API Key")
    supports_vision: bool = Field(description="是否具备图像理解能力")
    chat_model: str = ""
    key_hint: str = Field(default="", description="脱敏 Key 提示（如 sk-1****abcd）")
    selected: bool = Field(description="是否为当前生效的文本模型")
    selected_vision: bool = Field(default=False, description="是否为当前生效的视觉模型")


class ModelStatusResponse(BaseModel):
    """模型配置总览：给前端“设置”页使用。"""

    active_text_provider: str
    active_vision_provider: str
    offline_mode: bool = Field(
        description="是否运行在离线兜底模式（未配置任何可用 API Key）"
    )
    dnn_pipeline_enabled: bool = Field(
        description="本地深度视觉管线总开关是否开启"
    )
    sentinel_ready: bool = Field(
        description="PatchCore 哨兵（ResNet18 ONNX）是否就绪"
    )
    locator_ready: bool = Field(
        description="YOLO-World-S 定位器（ONNX）是否就绪"
    )
    providers: List[ProviderStatus]


class ApiKeyUpdateRequest(BaseModel):
    """保存提供方 API Key 请求体。"""

    api_key: str = Field(min_length=8, max_length=256, description="明文 API Key")


class ProviderSelectRequest(BaseModel):
    """设为默认提供方请求体。"""

    role: Literal["text", "vision"] = Field(description="角色：文本模型 / 视觉模型")
