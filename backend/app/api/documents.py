# -*- coding: utf-8 -*-
"""
规则文档管理路由
================
* POST   /api/documents/upload   上传 PDF/DOCX（数量上限 10）
* GET    /api/documents          文档列表（含数量上限）
* GET    /api/documents/rules    已抽取的结构化规则列表
* DELETE /api/documents/{id}     删除文档（向量/规则/冲突联动清理）
"""
from __future__ import annotations

from fastapi import APIRouter, File, UploadFile

from app.config import settings
from app.schemas.document import (
    DocumentListResponse,
    DocumentOut,
    RuleListResponse,
    RuleOut,
)
from app.services.document_service import document_service

router = APIRouter(prefix="/documents", tags=["规则文档"])


@router.post("/upload", response_model=DocumentOut, summary="上传规则文档")
async def upload_document(file: UploadFile = File(...)):
    """
    上传单份 PDF/DOCX 规则图文文档：
    解析 → 分块 → 向量化 → LLM 规则抽取 → 自动冲突检测。
    """
    content = await file.read()
    doc = await document_service.upload(
        filename=file.filename or "unnamed", content=content
    )
    return doc


@router.get("", response_model=DocumentListResponse, summary="文档列表")
async def list_documents():
    items = document_service.list_documents()
    return {
        "items": items,
        "total": len(items),
        "max_documents": settings.max_documents,
    }


@router.get("/rules", response_model=RuleListResponse, summary="结构化规则列表")
async def list_rules():
    items = document_service.list_rules()
    return {"items": items, "total": len(items)}


@router.delete("/{doc_id}", summary="删除文档")
async def delete_document(doc_id: str):
    await document_service.delete_document(doc_id)
    return {"deleted": doc_id}
