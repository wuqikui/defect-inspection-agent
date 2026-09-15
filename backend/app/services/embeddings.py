# -*- coding: utf-8 -*-
"""
文本向量化（Embedding）抽象层
=============================
设计目标：模型可替换、可离线运行。

* ``EmbeddingClient``：统一接口（embed_documents / embed_query）；
* ``ZhipuEmbedding`` / ``QwenEmbedding``：在线 API 实现；
* ``LocalHashingEmbedding``：零依赖离线兜底实现——
  基于中文“字 / 二元字组(bigram)”特征哈希的确定性向量，L2 归一化后
  用余弦距离衡量规则文本相似度，适合规则文档中大量专有名词/阈值数字的匹配；
* ``get_embedding_client()``：工厂方法，按配置的 Key 自动选择。

统一维度 EMBED_DIM=1024：智谱 embedding-3 与千问 text-embedding-v3
均支持 1024 维输出，离线实现也取 1024 维，保证向量库集合维度不变；
但不同模型的语义空间不可混用，切换提供方时服务层会触发全量重建索引。
"""
from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import List, Sequence

import httpx
import numpy as np

from app.config import settings
from app.core.exceptions import ModelCallError
from app.services import key_store

# 全局统一向量维度
EMBED_DIM = 1024
# API 单次批量请求的最大文本条数（保守值，兼顾各家限制）
_BATCH_SIZE = 16


class EmbeddingClient(ABC):
    """向量化客户端抽象基类。"""

    name: str = "base"

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """批量文本 → 向量列表。"""
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        """单条查询文本 → 向量（默认复用批量接口）。"""
        return self.embed_documents([text])[0]


# =====================================================================
# 智谱 GLM embedding-3
# =====================================================================
class ZhipuEmbedding(EmbeddingClient):
    """智谱开放平台 Embedding：POST /api/paas/v4/embeddings。"""

    name = "zhipu"
    _URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"

    def __init__(self, api_key: str, model: str, dim: int = EMBED_DIM):
        self._api_key = api_key
        self._model = model
        self._dim = dim
        self._client = httpx.Client(timeout=settings.llm_timeout_seconds)

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        results: List[List[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = list(texts[start : start + _BATCH_SIZE])
            payload = {"model": self._model, "input": batch}
            # embedding-3 支持自定义维度；老模型忽略该参数也不会报错（按模型而定）
            if self._model.startswith("embedding-3"):
                payload["dimensions"] = self._dim
            try:
                resp = self._client.post(
                    self._URL,
                    json=payload,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise ModelCallError(f"智谱 Embedding 调用失败：{exc}") from exc
            data = resp.json()["data"]
            # API 返回顺序与输入一致，按 index 排序以防万一
            data.sort(key=lambda d: d["index"])
            results.extend([d["embedding"] for d in data])
        return results


# =====================================================================
# 阿里千问 text-embedding-v3（DashScope OpenAI 兼容模式）
# =====================================================================
class QwenEmbedding(EmbeddingClient):
    """千问 Embedding：DashScope compatible-mode /v1/embeddings。"""

    name = "qwen"
    _URL = (
        "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
    )

    def __init__(self, api_key: str, model: str, dim: int = EMBED_DIM):
        self._api_key = api_key
        self._model = model
        self._dim = dim
        self._client = httpx.Client(timeout=settings.llm_timeout_seconds)

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        results: List[List[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = list(texts[start : start + _BATCH_SIZE])
            # DashScope OpenAI 兼容模式：input 为字符串数组，
            # dimensions 为顶层参数
            payload = {
                "model": self._model,
                "input": batch,
                "dimensions": self._dim,
                "encoding_format": "float",
            }
            try:
                resp = self._client.post(
                    self._URL,
                    json=payload,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise ModelCallError(f"千问 Embedding 调用失败：{exc}") from exc
            data = resp.json()["data"]
            data.sort(key=lambda d: d["index"])
            results.extend([d["embedding"] for d in data])
        return results


# =====================================================================
# 离线兜底：特征哈希 Embedding（确定性、零网络、零模型下载）
# =====================================================================
class LocalHashingEmbedding(EmbeddingClient):
    """
    字 / 二元字组特征哈希向量（Hashing Trick）。

    算法：
    1. 抽取两类特征：
       - 单字（中文/数字/字母），保留“气孔、0.5、mm”等关键 token；
       - 相邻二元字组，捕获“气孔直径、单个缺陷”等局部搭配；
    2. 每个特征用 MD5 哈希到 [0, dim) 槽位，带符号散列（signed hashing）
       减少哈希碰撞造成的偏差；
    3. 对数字（阈值）额外加权——规则文档中数值是核心判定信息；
    4. L2 归一化，使内积等价于余弦相似度，可直接与 Chroma cosine 配合。
    """

    name = "local"

    # 中文字符 / 字母 / 数字（含小数点）
    _TOKEN_RE = re.compile(r"[\u4e00-\u9fa5]|[A-Za-z]+|\d+(?:\.\d+)?%?")

    def __init__(self, dim: int = EMBED_DIM):
        self._dim = dim

    @classmethod
    def _features(cls, text: str) -> List[tuple]:
        text = text.lower()
        unigrams = cls._TOKEN_RE.findall(text)
        features: List[tuple] = [(t, 1.0) for t in unigrams]
        # bigram：相邻 token 拼接（中文上即相邻汉字，英文上即相邻单词）
        for a, b in zip(unigrams, unigrams[1:]):
            features.append((f"{a}{b}", 1.0))
        # 数字类特征（尺寸阈值/数量阈值）加权 2 倍
        weighted = [
            (tok, 2.0 if re.fullmatch(r"\d+(?:\.\d+)?%?", tok) else w)
            for tok, w in features
        ]
        return weighted

    def _hash(self, feature: str) -> tuple:
        digest = hashlib.md5(feature.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % self._dim
        # 用高位字节决定符号，降低碰撞偏差
        sign = 1.0 if digest[4] & 1 else -1.0
        return idx, sign

    def embed_one(self, text: str) -> List[float]:
        vec = np.zeros(self._dim, dtype=np.float32)
        for tok, weight in self._features(text):
            idx, sign = self._hash(tok)
            vec[idx] += sign * weight
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.embed_one(t) for t in texts]


# =====================================================================
# 工厂
# =====================================================================
def available_embedding_providers() -> List[str]:
    """按固定优先级返回“已配置可用”的在线提供方（Key：前端保存 > .env）。"""
    available = []
    if key_store.resolve_api_key("zhipu"):
        available.append("zhipu")
    if key_store.resolve_api_key("qwen"):
        available.append("qwen")
    return available


def get_embedding_client(prefer: str = "") -> EmbeddingClient:
    """
    按优先级构造 Embedding 客户端：
    显式指定 > 已配置 Key 的提供方 > 离线哈希兜底。

    :param prefer: 期望提供方（zhipu/qwen/local）；与文本 LLM 选择保持联动
    """
    zhipu_key = key_store.resolve_api_key("zhipu")
    qwen_key = key_store.resolve_api_key("qwen")

    if prefer == "local":
        return LocalHashingEmbedding()
    if prefer == "zhipu" and zhipu_key:
        return ZhipuEmbedding(zhipu_key, settings.zhipu_embedding_model)
    if prefer == "qwen" and qwen_key:
        return QwenEmbedding(qwen_key, settings.qwen_embedding_model)

    # 自动选择：跟随配置好的 Key
    if zhipu_key:
        return ZhipuEmbedding(zhipu_key, settings.zhipu_embedding_model)
    if qwen_key:
        return QwenEmbedding(qwen_key, settings.qwen_embedding_model)

    # 无任何 Key：离线模式，系统仍可完整跑通
    return LocalHashingEmbedding()
