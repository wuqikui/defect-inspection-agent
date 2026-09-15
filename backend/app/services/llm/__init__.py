# -*- coding: utf-8 -*-
"""
大语言模型（LLM）标准化适配层
=============================
* ``base.BaseLLM`` 定义统一接口：对话、文档摘要、规则抽取、冲突检测、
  滑窗图像判定（多模态），并内置 JSON 容错解析与重试；
* ``online.OpenAICompatibleLLM`` 覆盖智谱 GLM / 阿里千问 / DeepSeek
  （三者均提供 OpenAI 兼容的 /chat/completions 协议）；
* ``mock.MockLLM`` 为离线兜底实现（关键词 / 正则启发式）；
* 工厂方法 ``get_text_llm`` / ``get_vision_llm`` 按配置自动选择。

未来接入新模型时，只需新增一个 BaseLLM 子类并在工厂注册，
业务层（规则 / 冲突 / 检测服务）无需任何改动。
"""
from app.services.llm.factory import (
    get_text_llm,
    get_vision_llm,
    list_provider_status,
    reload_llm_runtime,
)

__all__ = [
    "get_text_llm",
    "get_vision_llm",
    "list_provider_status",
    "reload_llm_runtime",
]
