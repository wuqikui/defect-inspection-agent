# -*- coding: utf-8 -*-
"""
规则冲突路由
============
* GET  /api/conflicts          冲突列表（含待确认数量）
* POST /api/conflicts/{id}/resolve  逐条裁决（采纳A / 采纳B / 自定义）
"""
from __future__ import annotations

from fastapi import APIRouter

from app.schemas.conflict import ConflictListResponse, ConflictResolveRequest
from app.services.conflict_service import conflict_service

router = APIRouter(prefix="/conflicts", tags=["规则冲突"])


@router.get("", response_model=ConflictListResponse, summary="冲突列表")
async def list_conflicts():
    """返回全部冲突；pending 在前，前端逐条弹出让用户确认。"""
    items = conflict_service.list_conflicts()
    pending = sum(1 for c in items if c["status"] == "pending")
    return {
        "items": items,
        "pending_count": pending,
        "resolved_count": len(items) - pending,
    }


@router.post("/{conflict_id}/resolve", summary="裁决单条冲突")
async def resolve_conflict(conflict_id: str, body: ConflictResolveRequest):
    """
    对一条冲突做出最终确认：
    A=以文档A规则为准；B=以文档B规则为准；CUSTOM=采用用户输入的统一表述。
    """
    row = conflict_service.resolve(
        conflict_id, choice=body.choice, custom_text=body.custom_text
    )
    return row
