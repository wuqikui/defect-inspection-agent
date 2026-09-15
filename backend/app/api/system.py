# -*- coding: utf-8 -*-
"""
系统 / 模型状态路由
===================
* GET /api/health  存活探针
* GET /api/models  当前模型配置（在线/离线、视觉能力、SAM 可用性）
"""
from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.services.llm import list_provider_status
from app.services.vision.segmenter import get_segmenter

router = APIRouter(tags=["系统"])


@router.get("/health", summary="健康检查")
async def health():
    return {"status": "ok", "version": __version__}


@router.get("/models", summary="模型配置状态")
async def model_status():
    status = list_provider_status()
    _, is_sam = get_segmenter()
    from app.config import settings

    status["sam_available"] = is_sam
    status["sam_device"] = settings.sam_device if is_sam else ""
    return status
