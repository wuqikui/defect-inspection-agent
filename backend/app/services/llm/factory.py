# -*- coding: utf-8 -*-
"""
LLM 工厂
========
选择逻辑（文本模型）：
    用户固定偏好（前端设置）> 显式配置 llm_provider > 按 Key 自动
    （zhipu → qwen → deepseek）> 离线 mock
视觉模型：
    用户固定偏好 > zhipu/qwen（按 Key）> None（由本地视觉管线启发式裁判兜底）

API Key 来源：前端保存（SQLite，见 key_store）> .env 环境变量。
模型实例在进程内缓存（无状态客户端，复用 httpx 连接池）；
Key / 偏好变更后调用 reload_llm_runtime() 使缓存失效，下次调用即按新配置重建。
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

from app.config import settings
from app.services import key_store
from app.services.llm.base import BaseLLM
from app.services.llm.mock import MockLLM
from app.services.llm.online import (
    OpenAICompatibleLLM,
    make_deepseek,
    make_qwen,
    make_zhipu,
)

# 自动选择的固定优先级
_PRIORITY = ("zhipu", "qwen", "deepseek")
_FACTORIES = {
    "zhipu": make_zhipu,
    "qwen": make_qwen,
    "deepseek": make_deepseek,
}


@lru_cache(maxsize=None)
def _build(provider: str) -> BaseLLM:
    """根据名称构造（并缓存）模型实例。"""
    if provider in _FACTORIES:
        return _FACTORIES[provider]()
    return MockLLM()


def reload_llm_runtime() -> None:
    """Key / 提供方偏好变更后调用：使缓存实例失效，实现配置热切换。"""
    _build.cache_clear()


def _is_configured(provider: str) -> bool:
    """该提供方是否已配置 API Key（数据库保存值或 .env 环境变量）。"""
    return bool(key_store.resolve_api_key(provider))


def resolve_text_provider() -> str:
    """解析当前实际生效的文本模型提供方名。"""
    pref = key_store.get_preference("text")
    if pref == "mock":
        return "mock"
    if pref in _FACTORIES and _is_configured(pref):
        return pref
    explicit = settings.llm_provider.strip().lower()
    if explicit in {"zhipu", "qwen", "deepseek"}:
        return explicit
    if explicit == "mock":
        return "mock"
    for name in _PRIORITY:
        if _is_configured(name):
            return name
    return "mock"


def resolve_vision_provider() -> Optional[str]:
    """解析当前实际生效的视觉模型提供方；无可用视觉模型时返回 None。"""
    pref = key_store.get_preference("vision")
    if pref in ("zhipu", "qwen") and _is_configured(pref):
        return pref
    explicit = settings.vision_provider.strip().lower()
    if explicit in {"zhipu", "qwen"} and _is_configured(explicit):
        return explicit
    if explicit == "mock":
        return None
    for name in ("zhipu", "qwen"):  # DeepSeek 无视觉能力，不参与
        if _is_configured(name):
            return name
    return None


def get_text_llm() -> BaseLLM:
    """获取文本 LLM（业务规则抽取 / 冲突检测 / 摘要统一入口）。"""
    return _build(resolve_text_provider())


def get_vision_llm() -> Optional[BaseLLM]:
    """获取视觉 LLM；未配置时返回 None，检测服务自动切换本地启发式裁判兜底。"""
    provider = resolve_vision_provider()
    return _build(provider) if provider else None


def list_provider_status() -> Dict:
    """供 /api/models 状态接口使用的提供方配置快照。"""
    active_text = resolve_text_provider()
    active_vision = resolve_vision_provider() or ""
    providers: List[Dict] = []
    meta = {
        "zhipu": ("智谱 GLM", True, settings.zhipu_chat_model),
        "qwen": ("阿里千问", True, settings.qwen_chat_model),
        "deepseek": ("DeepSeek", False, settings.deepseek_chat_model),
        "mock": ("离线规则引擎（兜底）", False, "heuristic"),
    }
    for name in ("zhipu", "qwen", "deepseek", "mock"):
        label, vision, model = meta[name]
        configured = True if name == "mock" else _is_configured(name)
        providers.append(
            {
                "name": name,
                "label": label,
                "configured": configured,
                "supports_vision": vision,
                "chat_model": model,
                "key_hint": key_store.key_hint(name),
                "selected": name == active_text,
                "selected_vision": bool(active_vision) and name == active_vision,
            }
        )
    return {
        "active_text_provider": active_text,
        "active_vision_provider": active_vision,
        "offline_mode": active_text == "mock",
        "providers": providers,
    }
