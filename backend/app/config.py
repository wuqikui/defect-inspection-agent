# -*- coding: utf-8 -*-
"""
全局配置模块
============
所有可调参数集中在此，通过环境变量或项目根目录下的 ``.env`` 文件注入，
保证同一份代码可以在本地开发环境与 Docker 容器中运行，无需修改源码。

优先级：环境变量 > .env 文件 > 代码默认值
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ 目录：本文件位于 backend/app/config.py，向上两级即 backend/
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """系统配置（字段名大小写不敏感，环境变量名与字段名一致）。"""

    # ---------- 基础 ----------
    app_name: str = "多模态缺陷检测智能体 / Multimodal Defect Inspection Agent"
    api_prefix: str = "/api"
    # 允许跨域访问的前端来源（逗号分隔），默认开放给 Vite 开发服务器
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # ---------- 数据存储目录 ----------
    # 文档原件、标注结果图、向量库、SQLite 数据库均落在 DATA_DIR 下
    data_dir: Path = Field(default=BASE_DIR / "data")

    # ---------- 文档上传限制 ----------
    max_documents: int = 10                 # 规则文档数量上限（需求：不超过 10 个）
    max_doc_size_mb: int = 50               # 单个文档大小上限（MB）
    allowed_doc_exts: str = ".pdf,.docx"    # 允许的文档格式

    # ---------- 图片上传限制 ----------
    max_image_size_mb: int = 100            # 单张工业图像大小上限（MB）
    allowed_image_exts: str = ".jpg,.jpeg,.png,.bmp,.tif,.tiff,.webp"
    # Pillow 解压像素安全阀：工业大图可能远超默认 1.7 亿像素，这里放宽
    # 设为 0 表示不限制（生产环境可按设备内存调低）
    image_max_pixels: int = 800_000_000

    # ---------- 滑窗检测参数 ----------
    window_size: int = 1024                 # 固定方形滑窗边长（像素）
    tile_jpeg_quality: int = 85             # 滑窗切片送多模态模型时的 JPEG 质量
    det_confidence_threshold: float = 0.35  # 缺陷置信度过滤阈值
    nms_iou_threshold: float = 0.30         # 跨滑窗重复检测框 NMS 的 IoU 阈值

    # ---------- RAG 检索参数 ----------
    retrieval_top_k: int = 4                # 每次规则检索返回的最相关片段数
    # 相似度阈值：Chroma 返回余弦距离 distance ∈ [0,2]，similarity = 1-distance
    # 仅保留 similarity >= 阈值的片段，避免低相关规则干扰判定（配置必须真正生效）
    retrieval_similarity_threshold: float = 0.30

    # ---------- LLM 提供方 ----------
    # 可选：zhipu（智谱 GLM）/ qwen（阿里千问）/ deepseek / mock（离线规则兜底）
    # 留空(空字符串)表示“自动选择”：按已配置 Key 的顺序 zhipu -> qwen -> deepseek -> mock
    llm_provider: str = ""
    # 视觉理解模型可单独指定（deepseek 目前无视觉能力，会自动回退到 mock 视觉检测）
    vision_provider: str = ""
    # 各家 API Key（建议只通过环境变量注入，不要写进代码）
    zhipu_api_key: str = ""
    qwen_api_key: str = ""
    deepseek_api_key: str = ""
    # 各家模型名（可随官方升级在 .env 中替换，实现“模型可替换”）
    zhipu_chat_model: str = "glm-4-flash"
    zhipu_vision_model: str = "glm-4v-flash"
    zhipu_embedding_model: str = "embedding-3"
    qwen_chat_model: str = "qwen-plus"
    qwen_vision_model: str = "qwen-vl-max"
    qwen_embedding_model: str = "text-embedding-v3"
    deepseek_chat_model: str = "deepseek-chat"
    llm_timeout_seconds: int = 60           # LLM 请求超时
    llm_max_retries: int = 2                # 失败重试次数

    # ---------- SAM 分割（可选） ----------
    # 未安装 segment-anything / 未配置权重时，自动降级为 OpenCV 轮廓分割
    sam_checkpoint_path: str = ""
    sam_model_type: str = "vit_b"           # vit_b / vit_l / vit_h
    sam_device: str = "auto"                # auto / cpu / cuda

    # ---------- 任务进度 ----------
    job_result_ttl_hours: int = 24 * 7      # 检测任务临时状态保留时长

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # 派生路径与工具方法
    # ------------------------------------------------------------------
    @property
    def upload_dir(self) -> Path:
        """规则文档原件存放目录。"""
        return self.data_dir / "uploads"

    @property
    def result_dir(self) -> Path:
        """检测原图副本 / 标注结果图存放目录。"""
        return self.data_dir / "results"

    @property
    def vector_dir(self) -> Path:
        """Chroma 持久化向量库目录。"""
        return self.data_dir / "vector_db"

    @property
    def db_path(self) -> Path:
        """SQLite 数据库文件路径。"""
        return self.data_dir / "app.db"

    @property
    def sam_dir(self) -> Path:
        """SAM 权重文件目录。"""
        return self.data_dir / "sam_checkpoints"

    @property
    def cors_origin_list(self) -> List[str]:
        """把逗号分隔的 CORS 配置解析为列表。"""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowed_doc_ext_list(self) -> List[str]:
        return [e.strip().lower() for e in self.allowed_doc_exts.split(",") if e.strip()]

    @property
    def allowed_image_ext_list(self) -> List[str]:
        return [e.strip().lower() for e in self.allowed_image_exts.split(",") if e.strip()]

    def ensure_dirs(self) -> None:
        """启动时创建所有必需目录，避免首次写文件时报错（跨平台）。"""
        for d in (
            self.data_dir,
            self.upload_dir,
            self.result_dir,
            self.vector_dir,
            self.sam_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


# 全局单例：其它模块统一 `from app.config import settings` 导入使用
settings = Settings()
