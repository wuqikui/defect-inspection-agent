# -*- coding: utf-8 -*-
"""
FastAPI 应用入口
================
启动流程（lifespan）：
    创建数据目录 → 初始化 SQLite 表 → 初始化向量库
    → 必要时用 SQLite 中的 chunk 原文重建向量索引（如更换 embedding 提供方）

运行：
    uvicorn app.main:app --host 0.0.0.0 --port 8000
文档：
    http://localhost:8000/docs （Swagger UI）
"""
from __future__ import annotations

import asyncio
import contextlib
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api import conflicts as conflicts_router
from app.api import documents as documents_router
from app.api import inspection as inspection_router
from app.api import system as system_router
from app.config import settings
from app.core.exceptions import register_exception_handlers
from app.db.database import db
from app.services.document_service import document_service
from app.services.vector_store import vector_store


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动 / 关闭钩子。"""
    # ---------- 启动 ----------
    settings.ensure_dirs()
    db.init_schema()
    # 向量库初始化（含 embedding 提供方变更检测）在线程中执行，避免阻塞事件循环
    await asyncio.to_thread(vector_store.initialize)
    # 若向量集合被清空（换模型 / 首次持久化缺失），用 SQLite 原文重建索引
    reindexed = await document_service.reindex_if_needed()
    if reindexed:
        print(f"[startup] 向量库已重建，共写入 {reindexed} 个片段")
    print(f"[startup] {settings.app_name} v{__version__} 就绪")
    print(f"[startup] embedding provider = {vector_store.provider}")
    yield
    # ---------- 关闭 ----------
    db.close()


app = FastAPI(
    title="多模态缺陷检测智能体 API",
    description=(
        "规则文档（PDF/DOCX）上传 → RAG 知识库 → 跨文档冲突确认 → "
        "工业图片滑窗缺陷检测（多模态模型 / SAM / CV 兜底）→ 历史回溯"
    ),
    version=__version__,
    lifespan=lifespan,
)

# ---- CORS：默认放行本地前端（Vite 5173），生产可在 .env 收紧 ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- 统一异常输出 ----
register_exception_handlers(app)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """为每个响应附加处理耗时头，便于前端 / 运维观察性能。"""
    start = time.perf_counter()
    response: JSONResponse = await call_next(request)
    response.headers["X-Process-Time-ms"] = str(
        int((time.perf_counter() - start) * 1000)
    )
    return response


# ---- 路由挂载 ----
app.include_router(system_router.router, prefix=settings.api_prefix)
app.include_router(documents_router.router, prefix=settings.api_prefix)
app.include_router(conflicts_router.router, prefix=settings.api_prefix)
app.include_router(inspection_router.router, prefix=settings.api_prefix)


@app.get("/", include_in_schema=False)
async def root():
    """根路径快捷信息。"""
    return {
        "name": settings.app_name,
        "version": __version__,
        "docs": "/docs",
        "api_prefix": settings.api_prefix,
    }
