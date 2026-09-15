# -*- coding: utf-8 -*-
"""
文档管理服务
============
编排规则文档的完整入库流水线：
    安全校验 → 数量控制 → 落盘 → 解析(PDF/DOCX) → 分块
    → SQLite 持久化(原文 chunks) → 向量化入 Chroma
    → LLM 抽取结构化规则 → 触发跨文档冲突检测

门面模式：API 层只调用本服务，不直接操作向量库 / 解析器 / LLM。
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, List

from app.config import settings
from app.core.exceptions import DocumentLimitError, NotFoundError
from app.core.security import (
    sanitize_filename,
    safe_join,
    validate_document,
)
from app.db.database import db, utcnow_iso
from app.services.chunker import build_chunks
from app.services.conflict_service import conflict_service
from app.services.llm import get_text_llm
from app.services.parser import parse_document
from app.services.vector_store import vector_store


class DocumentService:
    """规则文档上传 / 列表 / 删除 / 重索引。"""

    # ------------------------------------------------------------------
    # 上传
    # ------------------------------------------------------------------
    async def upload(self, filename: str, content: bytes) -> Dict[str, Any]:
        """
        上传并处理一份规则文档（PDF/DOCX）。

        :param filename: 原始文件名
        :param content: 文件二进制内容
        :return: 新建文档的可序列化信息（含解析出的页数 / 片段数）
        :raises DocumentLimitError: 文档数量已达上限
        """
        # 1) 数量控制：落盘前检查，避免无效写入
        if db.count_documents() >= settings.max_documents:
            raise DocumentLimitError(settings.max_documents)

        # 2) 安全校验（扩展名白名单 + 魔数 + 大小）
        ext = validate_document(filename, head=content[:16], size_bytes=len(content))
        stored_name = sanitize_filename(filename)
        file_path = safe_join(settings.upload_dir, stored_name)
        file_path.write_bytes(content)

        # 提前生成文档主键，使 chunk 元数据在分块阶段即可携带 doc_id
        doc_id = uuid.uuid4().hex

        # 3) 解析与分块（CPU 操作，放到线程池避免阻塞事件循环）
        try:
            parsed, chunks = await asyncio.to_thread(
                self._parse_and_chunk, file_path, ext, doc_id, filename
            )
        except Exception:
            # 解析失败时清理已落盘文件，再向上抛出业务异常
            file_path.unlink(missing_ok=True)
            raise

        # 4) 持久化文档与 chunk 原文（chunk 表是向量库可重建的依据）
        db.execute(
            """INSERT INTO documents
               (id, stored_name, original_name, ext, size_bytes, status,
                page_count, chunk_count, summary, uploaded_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                doc_id,
                stored_name,
                filename,
                ext,
                len(content),
                "parsed",
                parsed.page_count,
                len(chunks),
                "",
                utcnow_iso(),
            ),
        )
        chunk_ids = [uuid.uuid4().hex for _ in chunks]
        chunk_rows = [
            (chunk_ids[idx], doc_id, idx, c.text,
             json.dumps(c.metadata, ensure_ascii=False))
            for idx, c in enumerate(chunks)
        ]
        self._insert_chunks(chunk_rows)

        # 5) 向量化入库（网络 / 计算密集，放线程池）；失败则回滚已落数据
        try:
            await asyncio.to_thread(vector_store.add, chunks, chunk_ids)
        except Exception:
            file_path.unlink(missing_ok=True)
            db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
            await asyncio.to_thread(vector_store.delete_by_document, doc_id)
            raise

        # 6) LLM 抽取结构化规则 + 摘要（失败不阻断文档入库）
        summary, extracted_rules = await self._understand_document(
            filename, parsed.full_text
        )
        db.execute(
            "UPDATE documents SET summary=? WHERE id=?",
            (summary, doc_id),
        )
        self._save_rules(doc_id, extracted_rules)

        # 7) 重新计算跨文档冲突（仅基于 active 规则，已解决的冲突不复活）
        await conflict_service.recompute()

        return self.get_document(doc_id)

    # ------------------------------------------------------------------
    # 查询 / 删除
    # ------------------------------------------------------------------
    def list_documents(self) -> List[Dict[str, Any]]:
        return db.query_all(
            "SELECT * FROM documents ORDER BY uploaded_at DESC"
        )

    def get_document(self, doc_id: str) -> Dict[str, Any]:
        row = db.query_one("SELECT * FROM documents WHERE id=?", (doc_id,))
        if not row:
            raise NotFoundError("文档不存在或已被删除。")
        return row

    def list_rules(self) -> List[Dict[str, Any]]:
        """联表返回规则（附文档名），供前端“规则库”页面展示。"""
        return db.query_all(
            """
            SELECT r.*, d.original_name AS doc_name
              FROM rules r JOIN documents d ON r.doc_id = d.id
             ORDER BY r.defect_type, r.created_at
            """
        )

    async def delete_document(self, doc_id: str) -> None:
        """
        删除文档：向量 → 磁盘文件 → 数据库（外键级联删 chunks/rules/conflicts）。
        删除后：把其余规则全部恢复 active 并重算冲突。
        """
        doc = self.get_document(doc_id)
        await asyncio.to_thread(vector_store.delete_by_document, doc_id)
        safe_join(settings.upload_dir, doc["stored_name"]).unlink(missing_ok=True)
        db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        # 此前因冲突被淘汰的规则恢复生效，再重新检测现存文档间冲突
        db.execute("UPDATE rules SET status='active' WHERE status='superseded'")
        await conflict_service.recompute()

    async def reindex_if_needed(self) -> int:
        """启动时若发现向量库为空（如切换了 embedding 提供方），用原文重建。"""
        if vector_store.needs_reindex():
            return await asyncio.to_thread(vector_store.reindex_all)
        return 0

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_and_chunk(file_path, ext, doc_id: str, original_name: str):
        """同步解析 + 分块（在线程池中执行）。"""
        parsed = parse_document(file_path, ext)
        chunks = build_chunks(parsed, doc_id=doc_id, doc_name=original_name)
        return parsed, chunks

    @staticmethod
    def _insert_chunks(rows: List[tuple]) -> None:
        with db._lock:  # noqa: SLF001
            db._conn.executemany(
                "INSERT INTO document_chunks(id, doc_id, ord, text, metadata_json)"
                " VALUES (?,?,?,?,?)",
                rows,
            )

    @staticmethod
    async def _understand_document(filename: str, full_text: str):
        """调用 LLM 做摘要与规则抽取；任何异常降级为空结果，不阻断上传。"""
        llm = get_text_llm()
        try:
            summary = await llm.summarize_document(filename, full_text)
        except Exception:  # noqa: BLE001
            summary = full_text[:100]
        try:
            rules = await llm.extract_rules(filename, full_text)
        except Exception:  # noqa: BLE001
            rules = []
        return summary[:300], rules

    @staticmethod
    def _save_rules(doc_id: str, rules: List[Dict[str, Any]]) -> None:
        for rule in rules:
            db.execute(
                """INSERT INTO rules
                   (id, doc_id, defect_type, title, content, severity,
                    source_location, status, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    uuid.uuid4().hex,
                    doc_id,
                    rule["defect_type"],
                    rule.get("title", ""),
                    rule["content"],
                    rule.get("severity", ""),
                    rule.get("source_location", ""),
                    "active",
                    utcnow_iso(),
                ),
            )


# 全局单例
document_service = DocumentService()
