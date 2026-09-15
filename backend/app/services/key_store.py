# -*- coding: utf-8 -*-
"""
API Key / 提供方偏好存储
========================
把用户在前端填写的模型 API Key 与「当前生效提供方」偏好持久化到 SQLite
的 app_meta 表，实现前端一键配置：保存即生效，无需修改 .env、无需重启。

* Key 优先级：前端保存的 Key > .env 环境变量；
* Key 以明文存放在本地 data/app.db（单机部署，密钥本就归属本机用户），
  所有对外接口只返回脱敏提示（key_hint），不回传完整 Key；
* 「偏好」允许用户把某家已配置的提供方固定为默认文本 / 视觉模型；
  偏好失效（如对应 Key 被清除）时工厂自动回退到自动选择逻辑。
"""
from __future__ import annotations

from typing import Optional

from app.config import settings
from app.db.database import db

# 允许保存 Key 的提供方（mock 为内置兜底，不可配置 Key）
KEY_PROVIDERS = ("zhipu", "qwen", "deepseek")

# .env 中各提供方 Key 的字段名（前端保存的 Key 优先于环境变量）
_ENV_KEY_FIELDS = {
    "zhipu": "zhipu_api_key",
    "qwen": "qwen_api_key",
    "deepseek": "deepseek_api_key",
}

_KEY_PREFIX = "api_key_"
_PREF_PREFIX = "pref_"


# ----------------------------------------------------------------------
# API Key 读写
# ----------------------------------------------------------------------
def get_stored_key(provider: str) -> str:
    """读取前端保存在数据库中的 Key（未保存返回空串）。"""
    row = db.query_one(
        "SELECT value FROM app_meta WHERE key=?", (_KEY_PREFIX + provider,)
    )
    return row["value"] if row else ""


def set_stored_key(provider: str, api_key: str) -> None:
    """保存 / 覆盖某提供方的 API Key。"""
    db.execute(
        """
        INSERT INTO app_meta(key, value) VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (_KEY_PREFIX + provider, api_key.strip()),
    )


def delete_stored_key(provider: str) -> None:
    """清除数据库中保存的 Key（.env 中的 Key 不受影响）。"""
    db.execute("DELETE FROM app_meta WHERE key=?", (_KEY_PREFIX + provider,))


def resolve_api_key(provider: str) -> str:
    """解析某提供方当前实际生效的 Key：数据库保存值 > .env 环境变量。"""
    stored = get_stored_key(provider)
    if stored:
        return stored
    field = _ENV_KEY_FIELDS.get(provider, "")
    return str(getattr(settings, field, "") or "") if field else ""


# ----------------------------------------------------------------------
# 提供方偏好（默认文本 / 视觉模型）
# ----------------------------------------------------------------------
def get_preference(role: str) -> str:
    """读取用户固定选择的提供方名（未设置返回空串）。role: text / vision。"""
    row = db.query_one(
        "SELECT value FROM app_meta WHERE key=?", (_PREF_PREFIX + role + "_provider",)
    )
    return row["value"] if row else ""


def set_preference(role: str, provider: str) -> None:
    """固定某角色（text / vision）的默认提供方。"""
    db.execute(
        """
        INSERT INTO app_meta(key, value) VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (_PREF_PREFIX + role + "_provider", provider),
    )


# ----------------------------------------------------------------------
# 脱敏
# ----------------------------------------------------------------------
def mask_key(api_key: str) -> str:
    """Key 脱敏展示：保留前 4 位与后 4 位，中间用 **** 遮蔽。"""
    api_key = (api_key or "").strip()
    if not api_key:
        return ""
    if len(api_key) <= 8:
        return "****"
    return f"{api_key[:4]}****{api_key[-4:]}"


def key_hint(provider: str) -> str:
    """对外接口使用的脱敏提示（未配置返回空串）。"""
    return mask_key(resolve_api_key(provider)) if provider in _ENV_KEY_FIELDS else ""
