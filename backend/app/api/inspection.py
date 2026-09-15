# -*- coding: utf-8 -*-
"""
缺陷检测路由
============
* POST /api/inspection/detect     上传单张图片并启动检测（返回 job_id）
* GET  /api/inspection/jobs/{id}  轮询任务进度 / 获取检测结果
* GET  /api/inspection/history    检测历史
* GET  /api/inspection/image/{id} 读取任务原图
* GET  /api/inspection/results/{name} 读取标注结果图
"""
from __future__ import annotations

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse

from app.config import settings
from app.core.exceptions import NotFoundError
from app.core.security import safe_join
from app.schemas.inspection import (
    HistoryListItem,
    JobCreatedResponse,
    JobOut,
)
from app.services.conflict_service import conflict_service
from app.services.inspection_service import inspection_service

router = APIRouter(prefix="/inspection", tags=["缺陷检测"])


@router.post(
    "/detect",
    response_model=JobCreatedResponse,
    summary="上传图片并启动检测",
)
async def detect(file: UploadFile = File(...)):
    """
    接收单张任意尺寸工业图片；若存在未解决的规则冲突会被服务端拒绝（409）。
    """
    # 冲突门禁：创建任务前立即拒绝，前端可收到明确的 409 提示
    conflict_service.ensure_no_pending()
    content = await file.read()
    job_id = inspection_service.create_job(
        filename=file.filename or "unnamed.jpg", content=content
    )
    inspection_service.start_job(job_id)
    return {"job_id": job_id, "status": "processing"}


@router.get("/jobs/{job_id}", response_model=JobOut, summary="任务进度/结果")
async def get_job(job_id: str):
    """前端以 1~2s 间隔轮询，status=completed/failed 时停止。"""
    return inspection_service.get_job(job_id)


@router.get(
    "/history",
    response_model=list[HistoryListItem],
    summary="检测历史",
)
async def history(limit: int = 50):
    return inspection_service.list_history(limit=limit)


@router.get("/image/{job_id}", summary="原始图片")
async def get_original_image(job_id: str):
    """按任务 id 返回原始待测图（FileResponse 自动处理缓存头）。"""
    job = inspection_service.get_job(job_id)
    # job 中的 stored_image 不直接暴露，反查数据库行获取安全文件名
    row = inspection_service._get_job_row(job_id)  # noqa: SLF001
    path = safe_join(settings.result_dir, row["stored_image"])
    if not path.exists():
        raise NotFoundError("图片不存在。")
    return FileResponse(path)


@router.get("/results/{name}", summary="标注结果图")
async def get_result_image(name: str):
    """按安全文件名返回标注图（safe_join 防止路径穿越）。"""
    path = safe_join(settings.result_dir, name)
    if not path.exists():
        raise NotFoundError("结果图片不存在。")
    return FileResponse(path)
