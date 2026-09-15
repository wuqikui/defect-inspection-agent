# -*- coding: utf-8 -*-
"""
在线大模型适配器（OpenAI 兼容协议）
===================================
智谱 GLM、阿里千问（DashScope compatible-mode）、DeepSeek 均提供
OpenAI 风格的 ``POST /chat/completions`` 接口，请求/响应结构一致，
因此用同一个客户端类 + 不同配置（endpoint / model / 能力位）实现，
避免重复代码，也让“模型可替换”落到实处。

失败处理：连接错误 / 5xx / 429 按指数退避重试，超过次数抛出
ModelCallError（由全局异常处理器转友好提示）。
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import httpx

from app.config import settings
from app.core.exceptions import ModelCallError
from app.services import key_store
from app.services.llm.base import BaseLLM, Message


class OpenAICompatibleLLM(BaseLLM):
    """OpenAI /chat/completions 兼容客户端。"""

    def __init__(
        self,
        name: str,
        label: str,
        base_url: str,
        api_key: str,
        chat_model: str,
        vision_model: str = "",
        supports_vision: bool = False,
    ):
        self.name = name
        self.label = label
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._chat_model = chat_model
        self._vision_model = vision_model or chat_model
        self.supports_vision = supports_vision

    # ------------------------------------------------------------------
    # HTTP 通信
    # ------------------------------------------------------------------
    async def _post_chat(
        self,
        model: str,
        messages: List[Message],
        temperature: float,
        json_mode: bool,
    ) -> str:
        """带重试的 chat/completions 请求，返回助手文本。"""
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        # json_mode 在三家协议里均可用 response_format 表达；
        # 个别小模型不支持时忽略 400 重试一次（去掉该参数）
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._base_url}/chat/completions"
        last_exc: Optional[Exception] = None

        async with httpx.AsyncClient(
            timeout=settings.llm_timeout_seconds
        ) as client:
            for attempt in range(settings.llm_max_retries + 1):
                try:
                    resp = await client.post(url, json=payload, headers=headers)
                    # 400 + json_object 不被支持 → 去掉参数重试
                    if (
                        resp.status_code == 400
                        and json_mode
                        and "response_format" in payload
                    ):
                        payload.pop("response_format", None)
                        json_mode = False
                        continue
                    resp.raise_for_status()
                    body = resp.json()
                    return body["choices"][0]["message"]["content"]
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_exc = exc  # 网络类错误：退避后重试
                except httpx.HTTPStatusError as exc:
                    last_exc = exc
                    # 429 限流 / 5xx 服务端错误才重试；401/403 等立即失败
                    if exc.response.status_code not in (429, 500, 502, 503, 504):
                        break
                if attempt < settings.llm_max_retries:
                    await asyncio.sleep(0.8 * (2**attempt))  # 0.8s, 1.6s...

        raise ModelCallError(
            f"{self.label} 模型调用失败（已重试 {settings.llm_max_retries} 次）：{last_exc}"
        )

    # ------------------------------------------------------------------
    # BaseLLM 实现
    # ------------------------------------------------------------------
    async def chat(
        self,
        messages: List[Message],
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        return await self._post_chat(
            self._chat_model, messages, temperature, json_mode
        )

    async def vision_chat(
        self,
        prompt: str,
        image_jpeg_b64: str,
        temperature: float = 0.1,
    ) -> str:
        if not self.supports_vision:
            raise NotImplementedError(f"{self.label} 不支持图像理解")
        # 多模态消息：text + image_url(data URL)，三家视觉模型统一格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_jpeg_b64}"
                        },
                    },
                ],
            }
        ]
        return await self._post_chat(
            self._vision_model, messages, temperature, json_mode=False
        )


# =====================================================================
# 三家厂商的快捷构造函数（endpoint / 默认模型集中在此维护）
# =====================================================================
def make_zhipu() -> OpenAICompatibleLLM:
    """智谱 GLM：文本 glm-4-flash，视觉 glm-4v-flash。"""
    return OpenAICompatibleLLM(
        name="zhipu",
        label="智谱 GLM",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key=key_store.resolve_api_key("zhipu"),
        chat_model=settings.zhipu_chat_model,
        vision_model=settings.zhipu_vision_model,
        supports_vision=bool(settings.zhipu_vision_model),
    )


def make_qwen() -> OpenAICompatibleLLM:
    """阿里千问：文本 qwen-plus，视觉 qwen-vl-max。"""
    return OpenAICompatibleLLM(
        name="qwen",
        label="阿里千问",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=key_store.resolve_api_key("qwen"),
        chat_model=settings.qwen_chat_model,
        vision_model=settings.qwen_vision_model,
        supports_vision=bool(settings.qwen_vision_model),
    )


def make_deepseek() -> OpenAICompatibleLLM:
    """DeepSeek：仅文本（deepseek-chat），无视觉能力。"""
    return OpenAICompatibleLLM(
        name="deepseek",
        label="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        api_key=key_store.resolve_api_key("deepseek"),
        chat_model=settings.deepseek_chat_model,
        vision_model="",
        supports_vision=False,
    )
