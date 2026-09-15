# -*- coding: utf-8 -*-
"""
SQLite 持久化层
===============
为什么使用 SQLite：
* Python 标准库内置，零外部服务依赖，Windows/Linux/macOS 行为一致；
* 单文件数据库（data/app.db），便于备份与随容器卷迁移；
* 本系统读写并发极低（本地/内网智能体），配合互斥锁完全够用。

表结构：
    documents        规则文档元数据（上传即写一行）
    rules            LLM 从文档中抽取出的结构化检测规则
    conflicts        跨文档规则冲突及用户解决结果
    inspection_jobs  检测任务 / 历史记录（结果 JSON 内联存储）
    app_meta         系统元信息（schema 版本等）
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.config import settings

# 建表 DDL：IF NOT EXISTS 保证重复启动幂等
_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,          -- UUID
    stored_name   TEXT NOT NULL,             -- 落盘文件名（带随机前缀）
    original_name TEXT NOT NULL,             -- 用户所见原始文件名
    ext           TEXT NOT NULL,             -- .pdf / .docx
    size_bytes    INTEGER NOT NULL,
    status        TEXT NOT NULL DEFAULT 'parsed',  -- parsed / failed
    page_count    INTEGER NOT NULL DEFAULT 0,
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    summary       TEXT NOT NULL DEFAULT '',
    uploaded_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id            TEXT PRIMARY KEY,          -- 与向量库中的 id 一致
    doc_id        TEXT NOT NULL,             -- 所属文档
    ord           INTEGER NOT NULL,          -- 文档内顺序号
    text          TEXT NOT NULL,             -- chunk 原文（重建索引的依据）
    metadata_json TEXT NOT NULL,             -- 溯源等元数据 JSON
    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS rules (
    id              TEXT PRIMARY KEY,        -- UUID
    doc_id          TEXT NOT NULL,           -- 来源文档
    defect_type     TEXT NOT NULL,           -- 缺陷类型：气孔/划痕/起层……
    title           TEXT NOT NULL DEFAULT '',-- 规则标题
    content         TEXT NOT NULL,           -- 规则正文（判定标准）
    severity        TEXT NOT NULL DEFAULT '',-- 严重等级（若文档中有）
    source_location TEXT NOT NULL DEFAULT '',-- 页码/段落等溯源定位
    status          TEXT NOT NULL DEFAULT 'active', -- active / superseded
    created_at      TEXT NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS conflicts (
    id               TEXT PRIMARY KEY,
    defect_type      TEXT NOT NULL DEFAULT '',       -- 冲突围绕的缺陷主题
    description      TEXT NOT NULL,                  -- 冲突说明
    rule_a_id        TEXT NOT NULL,
    rule_b_id        TEXT NOT NULL,
    doc_a_id         TEXT NOT NULL,
    doc_b_id         TEXT NOT NULL,
    doc_a_name       TEXT NOT NULL,
    doc_b_name       TEXT NOT NULL,
    content_a        TEXT NOT NULL,
    content_b        TEXT NOT NULL,
    source_a         TEXT NOT NULL DEFAULT '',
    source_b         TEXT NOT NULL DEFAULT '',
    status           TEXT NOT NULL DEFAULT 'pending', -- pending / resolved
    resolution_choice TEXT NOT NULL DEFAULT '',       -- A / B / CUSTOM
    resolution_text  TEXT NOT NULL DEFAULT '',
    created_at       TEXT NOT NULL,
    resolved_at      TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (doc_a_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY (doc_b_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS inspection_jobs (
    id             TEXT PRIMARY KEY,
    image_name     TEXT NOT NULL,               -- 原始图片名
    stored_image   TEXT NOT NULL,               -- 原图副本路径
    annotated_path TEXT NOT NULL DEFAULT '',    -- 标注结果图路径
    width          INTEGER NOT NULL DEFAULT 0,
    height         INTEGER NOT NULL DEFAULT 0,
    status         TEXT NOT NULL DEFAULT 'processing', -- processing/completed/failed
    progress       INTEGER NOT NULL DEFAULT 0,  -- 0-100
    stage          TEXT NOT NULL DEFAULT '',    -- 当前阶段（供前端展示）
    has_defect     INTEGER NOT NULL DEFAULT 0,  -- 0/1
    summary        TEXT NOT NULL DEFAULT '',
    result_json    TEXT NOT NULL DEFAULT '[]',  -- 缺陷框 / 依据等结构化结果
    error_message  TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL,
    completed_at   TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS app_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc      ON document_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_rules_doc       ON rules(doc_id);
CREATE INDEX IF NOT EXISTS idx_conflicts_status ON conflicts(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created    ON inspection_jobs(created_at);
"""


def utcnow_iso() -> str:
    """统一的 UTC ISO8601 时间戳（带时区），便于跨平台排序与展示。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    """
    极薄的 SQLite 访问封装：
    * 单连接 + 互斥锁（SQLite 默认禁止跨线程使用同一连接）；
    * ``rows`` 方法把结果转成 dict 列表，服务层无需关心游标；
    * JSON 字段由服务层自行序列化（见 inspection_jobs.result_json）。
    """

    def __init__(self, db_path: Optional[str] = None):
        settings.ensure_dirs()
        # check_same_thread=False：FastAPI 线程池中的任意线程均可使用
        self._conn = sqlite3.connect(
            db_path or str(settings.db_path),
            check_same_thread=False,
            isolation_level=None,  # 自动提交；写事务用 BEGIN/COMMIT 显式包裹
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON;")  # 开启外键级联删除
        # WAL 模式：读写互不阻塞，崩溃恢复更稳
        self._conn.execute("PRAGMA journal_mode = WAL;")
        self._lock = threading.RLock()

    def init_schema(self) -> None:
        """初始化所有表（幂等）。"""
        with self._lock:
            self._conn.executescript(_SCHEMA)

    # --------------------------------------------------------------
    # 基础读写
    # --------------------------------------------------------------
    def execute(self, sql: str, params: tuple = ()) -> str:
        """执行写操作，返回 lastrowid（UUID 主键场景下业务层自行生成 id）。"""
        with self._lock:
            cur = self._conn.execute(sql, params)
            return cur.lastrowid

    def query_one(self, sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        """查询单行，返回 dict 或 None。"""
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
            return dict(row) if row else None

    def query_all(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """查询多行，返回 dict 列表。"""
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --------------------------------------------------------------
    # 业务快捷方法
    # --------------------------------------------------------------
    def count_documents(self) -> int:
        return self.query_one("SELECT COUNT(*) AS c FROM documents")["c"]

    def insert_job_result(self, job: Dict[str, Any]) -> None:
        """检测完成后更新任务行（结果 / 状态 / 标注图 / 时间戳）。"""
        self.execute(
            """
            UPDATE inspection_jobs
               SET status=?, progress=?, stage=?, has_defect=?, summary=?,
                   result_json=?, annotated_path=?, width=?, height=?,
                   error_message=?, completed_at=?
             WHERE id=?
            """,
            (
                job["status"],
                job.get("progress", 100),
                job.get("stage", ""),
                1 if job.get("has_defect") else 0,
                job.get("summary", ""),
                json.dumps(job.get("defects", []), ensure_ascii=False),
                job.get("annotated_path", ""),
                job.get("width", 0),
                job.get("height", 0),
                job.get("error_message", ""),
                utcnow_iso() if job["status"] != "processing" else "",
                job["id"],
            ),
        )


# 全局单例：init_db 在应用启动时调用
db = Database()
