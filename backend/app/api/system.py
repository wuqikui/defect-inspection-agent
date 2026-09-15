# -*- coding: utf-8 -*-
"""
系统 / 模型状态路由
===================
* GET /api/health  存活探针
* GET /api/models  当前模型配置（在线/离线、视觉能力、本地视觉管线就绪状态）
* PUT/DELETE /api/providers/{name}/key  前端一键保存 / 清除 API Key（保存即生效）
* PUT /api/providers/{name}/select      设为默认文本 / 视觉模型
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter

from app import __version__
from app.config import settings
from app.core.exceptions import (
    ConflictStateError,
    NotFoundError,
    ValidationError,
)
from app.schemas.system import ApiKeyUpdateRequest, ProviderSelectRequest
from app.services import key_store
from app.services.document_service import document_service
from app.services.llm import list_provider_status, reload_llm_runtime
from app.services.vector_store import vector_store
from app.services.vision.patchcore_sentinel import get_sentinel
from app.services.vision.yolo_world import get_yolo_world

router = APIRouter(tags=["系统"])

# 各角色允许被设为默认的提供方
_SELECTABLE = {
    "text": ("zhipu", "qwen", "deepseek", "mock"),
    "vision": ("zhipu", "qwen"),  # DeepSeek 无视觉能力
}


@router.get("/health", summary="健康检查")
async def health():
    return {"status": "ok", "version": __version__}


@router.get("/models", summary="模型配置状态")
async def model_status():
    status = list_provider_status()
    status["dnn_pipeline_enabled"] = settings.dnn_pipeline_enabled
    status["sentinel_ready"] = get_sentinel() is not None
    status["locator_ready"] = get_yolo_world() is not None
    return status


# ----------------------------------------------------------------------
# 前端一键配置：API Key 管理
# ----------------------------------------------------------------------
def _refresh_runtime() -> bool:
    """
    Key 变更后的运行时热刷新：
    1. 使 LLM 客户端缓存失效（下次调用按新 Key 重建）；
    2. 重查 embedding 提供方；若语义空间切换（如 local → zhipu），
       后台自动重建 RAG 向量索引。返回是否发生了提供方切换。
    """
    reload_llm_runtime()
    switched = vector_store.refresh_provider()
    if switched:
        task = asyncio.create_task(document_service.reindex_if_needed())
        task.add_done_callback(
            lambda t: t.exception()
            and print(f"[providers] 向量索引后台重建失败：{t.exception()}")
        )
    return switched


def _provider_or_404(name: str) -> None:
    if name not in key_store.KEY_PROVIDERS:
        raise NotFoundError(f"未知的模型提供方：{name}")


@router.put("/providers/{name}/key", summary="保存提供方 API Key（保存即生效）")
async def save_provider_key(name: str, body: ApiKeyUpdateRequest):
    _provider_or_404(name)
    api_key = body.api_key.strip()
    if len(api_key) < 8:
        raise ValidationError("API Key 格式不正确（长度不足 8 位）。")
    key_store.set_stored_key(name, api_key)
    switched = _refresh_runtime()
    return {
        "saved": name,
        "key_hint": key_store.mask_key(api_key),
        "embedding_switched": switched,
        "message": (
            "已保存并立即生效；RAG 向量索引正在按新模型后台重建。"
            if switched
            else "已保存并立即生效，无需重启。"
        ),
    }


@router.delete("/providers/{name}/key", summary="清除已保存的 API Key")
async def clear_provider_key(name: str):
    _provider_or_404(name)
    key_store.delete_stored_key(name)
    switched = _refresh_runtime()
    return {
        "cleared": name,
        "embedding_switched": switched,
        "message": "已清除（.env 中配置的 Key 仍然有效）。",
    }


@router.put("/providers/{name}/select", summary="设为默认文本 / 视觉模型")
async def select_provider(name: str, body: ProviderSelectRequest):
    valid = _SELECTABLE[body.role]
    if name not in valid:
        raise ValidationError(
            f"{name} 不能设为默认{'文本' if body.role == 'text' else '视觉'}模型。"
        )
    if name != "mock" and not key_store.resolve_api_key(name):
        raise ConflictStateError("该提供方尚未配置 API Key，请先在设置页保存 Key。")
    key_store.set_preference(body.role, name)
    reload_llm_runtime()
    role_label = "文本模型" if body.role == "text" else "视觉模型（VLM 裁决）"
    return {"selected": name, "role": body.role, "message": f"已将 {name} 设为默认{role_label}。"}
