# -*- coding: utf-8 -*-
"""
向量数据库封装（Chroma）
========================
职责：
1. 持久化存储规则文档 chunk 向量（data/vector_db）；
2. 提供“带相似度阈值”的检索：配置 retrieval_similarity_threshold
   必须真正生效——distance→similarity 转换后显式过滤，低分片段不进入上下文；
3. 按文档删除（文档删除时同步清除向量数据）；
4. Embedding 提供方变更检测：语义空间不兼容时自动清空集合，
   由服务层从 SQLite 中的原始 chunk 重新建索引。

Chroma 距离空间：cosine（distance ∈ [0,2]，similarity = 1 - distance）。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import chromadb

from app.config import settings
from app.db.database import db
from app.services.chunker import Chunk
from app.services.embeddings import EmbeddingClient, get_embedding_client

_COLLECTION = "rule_chunks"
_META_KEY = "embedding_provider"


@dataclass
class RetrievedChunk:
    """单条命中片段（相似度与溯源信息齐全）。"""

    text: str
    similarity: float
    metadata: Dict[str, str]


class VectorStore:
    """
    Chroma 门面（Facade）：服务层只与本类打交道，不直接依赖 chromadb API，
    便于未来替换为 Milvus / Qdrant 等实现。
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._client = chromadb.PersistentClient(path=str(settings.vector_dir))
        self._embedder: Optional[EmbeddingClient] = None
        self._collection = None
        self._provider = ""

    # ------------------------------------------------------------------
    # 初始化 / 提供方一致性
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """
        应用启动时调用：
        1. 选定 embedding 提供方并读取上次使用的提供方；
        2. 不一致则删除旧集合并重建空集合（服务层随后触发重索引）。
        """
        with self._lock:
            self._embedder = get_embedding_client(self._prefer_provider())
            self._provider = self._embedder.name
            last = db.query_one(
                "SELECT value FROM app_meta WHERE key=?", (_META_KEY,)
            )
            collection = self._client.get_or_create_collection(
                name=_COLLECTION,
                metadata={"hnsw:space": "cosine"},  # HNSW + 余弦距离
            )
            if last is not None and last["value"] != self._provider:
                # 语义空间变化：清空旧向量（原始文本仍在 SQLite，可重建）
                self._client.delete_collection(_COLLECTION)
                collection = self._client.get_or_create_collection(
                    name=_COLLECTION,
                    metadata={"hnsw:space": "cosine"},
                )
            self._collection = collection
            db.execute(
                """
                INSERT INTO app_meta(key, value) VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (_META_KEY, self._provider),
            )

    @staticmethod
    def _prefer_provider() -> str:
        """根据文本 LLM 选择推断优先的 embedding 提供方。"""
        p = settings.llm_provider.strip().lower()
        return p if p in {"zhipu", "qwen"} else ""

    @property
    def provider(self) -> str:
        return self._provider

    def needs_reindex(self) -> bool:
        """集合为空但 SQLite 中存在 chunk 时，说明需要重建索引。"""
        with self._lock:
            row = db.query_one("SELECT COUNT(*) AS c FROM document_chunks")
            return (self._collection.count() == 0) and (row["c"] > 0)

    def reindex_all(self) -> int:
        """从 SQLite 读取全部 chunk 重新写入向量库，返回写入条数。"""
        rows = db.query_all(
            "SELECT id, text, metadata_json FROM document_chunks ORDER BY doc_id, ord"
        )
        import json

        chunks = [
            Chunk(text=r["text"], metadata=json.loads(r["metadata_json"]))
            for r in rows
        ]
        ids = [r["id"] for r in rows]
        if chunks:
            self.add(chunks, ids=ids)
        return len(chunks)

    # ------------------------------------------------------------------
    # 写入 / 删除
    # ------------------------------------------------------------------
    def add(self, chunks: List[Chunk], ids: Optional[List[str]] = None) -> None:
        """
        将 chunk 批量向量化并写入集合。

        :param chunks: 待入库片段
        :param ids: 可选外部 id（重建索引时复用 SQLite chunk 主键）
        """
        if not chunks:
            return
        with self._lock:
            texts = [c.text for c in chunks]
            embeddings = self._embedder.embed_documents(texts)
            metadatas = [_stringify_metadata(c.metadata) for c in chunks]
            if ids is None:
                import uuid

                ids = [uuid.uuid4().hex for _ in chunks]
            # Chroma 单批建议不超过数百条，这里按 256 切片
            for start in range(0, len(ids), 256):
                end = start + 256
                self._collection.add(
                    ids=ids[start:end],
                    documents=texts[start:end],
                    embeddings=embeddings[start:end],
                    metadatas=metadatas[start:end],
                )

    def delete_by_document(self, doc_id: str) -> None:
        """删除某文档的全部向量（where 过滤 metadata.doc_id）。"""
        with self._lock:
            try:
                self._collection.delete(where={"doc_id": doc_id})
            except Exception:  # noqa: BLE001
                # 集合不存在 / 无命中均视为已删除
                pass

    def reset(self) -> None:
        """危险操作：清空整个规则向量集合（保留 SQLite 原文）。"""
        with self._lock:
            self._client.delete_collection(_COLLECTION)
            self._collection = self._client.get_or_create_collection(
                name=_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def query(
        self,
        text: str,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """
        语义检索（相似度阈值显式生效）。

        算法：
        1. 查询文本 → embedding；
        2. Chroma HNSW 召回 top_k 个最近邻（返回 cosine distance）；
        3. similarity = 1 - distance，**显式按阈值过滤**；
        4. 同文档同位置的相邻片段做去重，避免 4 条结果全来自同一页。

        :param text: 查询文本（如“气孔缺陷判定”）
        :param top_k: 召回数量，默认取配置 retrieval_top_k
        :param min_similarity: 相似度下限，默认取配置 retrieval_similarity_threshold
        :return: 按相似度降序的命中片段列表
        """
        top_k = top_k or settings.retrieval_top_k
        min_similarity = (
            settings.retrieval_similarity_threshold
            if min_similarity is None
            else min_similarity
        )
        with self._lock:
            # 多召回一些，供阈值过滤与去重
            fetch_k = max(top_k * 3, 8)
            query_emb = self._embedder.embed_query(text)
            res = self._collection.query(
                query_embeddings=[query_emb],
                n_results=min(fetch_k, max(self._collection.count(), 1)),
                include=["documents", "metadatas", "distances"],
            )

        documents = (res.get("documents") or [[]])[0]
        metadatas = (res.get("metadatas") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]

        hits: List[RetrievedChunk] = []
        seen_keys = set()
        for doc, meta, dist in zip(documents, metadatas, distances):
            similarity = 1.0 - float(dist)  # cosine distance → similarity
            if similarity < min_similarity:
                continue  # 阈值过滤：低相关片段绝不进入 RAG 上下文
            meta = meta or {}
            dedup_key = (meta.get("doc_id"), meta.get("source_location"), doc[:32])
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            hits.append(
                RetrievedChunk(
                    text=doc,
                    similarity=round(similarity, 4),
                    metadata={k: str(v) for k, v in meta.items()},
                )
            )
            if len(hits) >= top_k:
                break
        return hits


def _stringify_metadata(meta: Dict) -> Dict[str, str]:
    """Chroma metadata 只接受标量基本类型，统一转字符串。"""
    return {k: ("" if v is None else str(v)) for k, v in meta.items()}


# 全局单例（门面模式：服务层通过此实例访问向量库，不直接操作模块/客户端）
vector_store = VectorStore()
