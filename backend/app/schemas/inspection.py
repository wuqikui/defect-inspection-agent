# -*- coding: utf-8 -*-
"""缺陷检测任务 / 结果 / 历史相关数据模型。"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """缺陷边界框，坐标体系为“原始待测图片像素坐标”（已由滑窗坐标复原）。"""

    x: int = Field(description="框左上角 x")
    y: int = Field(description="框左上角 y")
    width: int
    height: int


class RuleEvidence(BaseModel):
    """判定依据：RAG 检索命中的规则片段（带来源与相似度，可解释 / 可溯源）。"""

    doc_name: str
    source_location: str = Field(description="页码 / 段落")
    content: str
    similarity: float = Field(description="相似度分数 0~1，越高越相关")


class DefectResult(BaseModel):
    """单个缺陷的完整检测结果。"""

    defect_type: str = Field(description="缺陷类型：气孔/划痕/起层/缺肉……")
    confidence: float = Field(ge=0, le=1, description="模型置信度")
    bbox: BoundingBox
    segmentation_available: bool = Field(
        default=False, description="是否已生成像素级轮廓（OpenCV ROI 提取）"
    )
    mask_polygon: Optional[List[List[int]]] = Field(
        default=None, description="掩膜轮廓点序列 [[x,y],...]，供前端高亮填充"
    )
    description: str = Field(default="", description="缺陷外观描述")
    evidences: List[RuleEvidence] = Field(
        default_factory=list, description="支撑判定的规则依据"
    )
    tile_index: int = Field(default=-1, description="来源滑窗编号（调试用）")


class JobOut(BaseModel):
    """检测任务状态 / 结果（轮询接口返回）。"""

    id: str
    image_name: str
    width: int
    height: int
    status: Literal["processing", "completed", "failed"]
    progress: int = Field(description="0-100 总进度百分比")
    stage: str = Field(description="当前阶段中文提示，如“第 3/12 滑窗检测中”")
    has_defect: bool = False
    summary: str = ""
    defects: List[DefectResult] = Field(default_factory=list)
    error_message: str = ""
    created_at: str
    completed_at: str = ""
    # 前端可直接访问的图片 URL（原图 / 标注图）
    image_url: str = ""
    annotated_url: str = ""


class JobCreatedResponse(BaseModel):
    """上传图片后立即返回任务 id，前端据此轮询进度。"""

    job_id: str
    status: str = "processing"


class HistoryListItem(BaseModel):
    """历史记录列表项（缩略图使用 annotated_url）。"""

    id: str
    image_name: str
    width: int
    height: int
    has_defect: bool
    defect_count: int
    summary: str
    created_at: str
    annotated_url: str
