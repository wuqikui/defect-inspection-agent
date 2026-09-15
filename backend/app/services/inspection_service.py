# -*- coding: utf-8 -*-
"""
检测任务编排服务
================
职责：
1. 接收单张任意尺寸工业图片（安全校验、解码验证、大图像素安全阀）；
2. 创建检测任务（SQLite 行 + 进度字段），后台异步执行；
3. 执行前做“冲突门禁”（仍有未解决规则冲突则拒绝检测）；
4. 组装检测规则上下文（active 结构化规则 + RAG 检索片段）；
5. 调用视觉检测器（滑窗 / 模型 / CV / 分割），实时回写进度；
6. 为每个缺陷附加 RAG 规则依据（相似度 + 文档名 + 页码，可解释）；
7. 生成文字结论与标注图，完成后落库；支持历史查询与详情轮询。
"""
from __future__ import annotations

import asyncio
import io
import json
import uuid
from typing import Any, Dict, List, Set

from PIL import Image

from app.config import settings
from app.core.exceptions import (
    ImageProcessError,
    NotFoundError,
    ValidationError,
)
from app.core.security import safe_join, validate_image
from app.db.database import db, utcnow_iso
from app.services.conflict_service import conflict_service
from app.services.llm import get_text_llm
from app.services.vector_store import vector_store
from app.services.vision.annotator import save_annotated
from app.services.vision.detector import defect_detector

# 防止后台 asyncio 任务被垃圾回收（官方推荐做法：保存强引用集合）
_BACKGROUND_TASKS: Set[asyncio.Task] = set()

# 支持的图像格式 → PIL format 映射（解码验证用）
_PIL_FORMATS = {"JPEG", "PNG", "BMP", "TIFF", "WEBP"}


class InspectionService:
    """检测任务创建 / 执行 / 查询 / 历史。"""

    # ------------------------------------------------------------------
    # 创建任务
    # ------------------------------------------------------------------
    def create_job(self, filename: str, content: bytes) -> str:
        """
        校验并落盘待测图片，创建 processing 状态的检测任务。

        :return: job_id
        """
        ext = validate_image(filename, content[:16], len(content))
        image = self._decode_image(content)  # 解码失败会在此抛出

        job_id = uuid.uuid4().hex
        stored_image = f"{job_id}{ext}"
        img_path = safe_join(settings.result_dir, stored_image)
        img_path.write_bytes(content)

        db.execute(
            """INSERT INTO inspection_jobs
               (id, image_name, stored_image, width, height, status,
                progress, stage, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                job_id,
                filename,
                stored_image,
                image.size[0],
                image.size[1],
                "processing",
                0,
                "等待开始…",
                utcnow_iso(),
            ),
        )
        return job_id

    def start_job(self, job_id: str) -> None:
        """以 asyncio 后台任务方式启动检测（不阻塞上传响应）。"""
        task = asyncio.create_task(self.run_job(job_id))
        _BACKGROUND_TASKS.add(task)
        task.add_done_callback(_BACKGROUND_TASKS.discard)

    # ------------------------------------------------------------------
    # 执行检测
    # ------------------------------------------------------------------
    async def run_job(self, job_id: str) -> None:
        """完整检测流水线（任何异常都会落为 failed 状态并写友好提示）。"""
        job = self._get_job_row(job_id)
        try:
            # 0) 门禁：必须有规则文档且无未解决冲突
            conflict_service.ensure_no_pending()
            if db.count_documents() == 0:
                raise ValidationError("请先上传至少一份缺陷检测规则文档。")

            image = self._load_job_image(job)

            async def progress(percent: int, stage: str) -> None:
                db.execute(
                    "UPDATE inspection_jobs SET progress=?, stage=? WHERE id=?",
                    (int(percent), stage, job_id),
                )

            # 1) 组装规则上下文
            await progress(2, "检索规则知识库…")
            rule_context, known_types = self._build_rule_context()

            # 2) 滑窗检测（模型/CV + NMS + 分割）
            result = await defect_detector.detect(
                image=image,
                rule_context=rule_context,
                known_defect_types=known_types,
                progress=progress,
            )
            defects = result["defects"]

            # 3) 逐缺陷附加 RAG 规则依据
            await progress(95, "匹配规则依据…")
            for defect in defects:
                defect["evidences"] = self._collect_evidences(
                    defect["defect_type"]
                )

            # 4) 生成结论
            summary = await self._compose_summary(defects, result["engine"])

            # 5) 绘制并保存标注图
            annotated_name = f"{job_id}_annotated.jpg"
            annotated_path = safe_join(settings.result_dir, annotated_name)
            await asyncio.to_thread(
                save_annotated, image, defects, annotated_path
            )

            # 6) 落库
            db.insert_job_result(
                {
                    "id": job_id,
                    "status": "completed",
                    "progress": 100,
                    "stage": "检测完成",
                    "has_defect": bool(defects),
                    "summary": summary,
                    "defects": defects,
                    "annotated_path": annotated_name,
                    "width": image.size[0],
                    "height": image.size[1],
                }
            )
        except Exception as exc:  # noqa: BLE001
            from app.core.exceptions import AppException

            message = (
                exc.message
                if isinstance(exc, AppException)
                else f"检测失败：{exc}"
            )
            db.insert_job_result(
                {
                    "id": job_id,
                    "status": "failed",
                    "progress": 100,
                    "stage": "检测失败",
                    "has_defect": False,
                    "summary": "",
                    "defects": [],
                    "error_message": message,
                }
            )

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def get_job(self, job_id: str) -> Dict[str, Any]:
        """获取任务详情（含结构化检测结果，供前端轮询与结果页使用）。"""
        row = self._get_job_row(job_id)
        return self._hydrate(row)

    def list_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """检测历史（仅已完成任务，倒序）。"""
        rows = db.query_all(
            """SELECT * FROM inspection_jobs
                WHERE status='completed'
                ORDER BY created_at DESC LIMIT ?""",
            (max(1, min(limit, 200)),),
        )
        items = []
        for row in rows:
            defects = json.loads(row["result_json"] or "[]")
            items.append(
                {
                    "id": row["id"],
                    "image_name": row["image_name"],
                    "width": row["width"],
                    "height": row["height"],
                    "has_defect": bool(row["has_defect"]),
                    "defect_count": len(defects),
                    "summary": row["summary"],
                    "created_at": row["created_at"],
                    "annotated_url": f"/api/inspection/results/{row['annotated_path']}"
                    if row["annotated_path"]
                    else "",
                }
            )
        return items

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _decode_image(content: bytes) -> Image.Image:
        """解码并验证上传图片（同时承担大图像素安全阀）。"""
        Image.MAX_IMAGE_PIXELS = (
            None if settings.image_max_pixels == 0 else settings.image_max_pixels
        )
        try:
            image = Image.open(io.BytesIO(content))
            image.load()  # 强制解码，立即暴露损坏文件
        except Image.DecompressionBombError as exc:
            raise ImageProcessError(
                "图像像素总量超过安全上限，为避免内存溢出已拒绝处理，"
                "请压缩后再上传或在服务端配置中调高 image_max_pixels。"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise ImageProcessError(f"图像无法解码，可能已损坏：{exc}") from exc
        if image.format not in _PIL_FORMATS:
            raise ValidationError(f"暂不支持的图像格式：{image.format}")
        if min(image.size) < 16:
            raise ValidationError("图像尺寸过小（短边至少 16 像素）。")
        return image

    @staticmethod
    def _load_job_image(job: Dict) -> Image.Image:
        """从 results 目录读取任务原图。"""
        path = safe_join(settings.result_dir, job["stored_image"])
        if not path.exists():
            raise NotFoundError("待测图片文件已丢失。")
        Image.MAX_IMAGE_PIXELS = (
            None if settings.image_max_pixels == 0 else settings.image_max_pixels
        )
        return Image.open(path)

    @staticmethod
    def _build_rule_context() -> tuple:
        """
        汇总当前生效规则：
        * 主体：rules 表中所有 active 规则（冲突裁决后的无矛盾规则集）；
        * 补充：对每条缺陷类型做一次 RAG 检索，丰富细则表述。

        优化点：
        1. 跨 dtype 去重改用文本前缀比较——原整文本比较对 RAG chunk
           与结构化规则几乎不命中，导致同一文档同一段落切片被多个
           缺陷类型关键词重复检索、堆叠进上下文，稀释了 VLM 可见
           的规则覆盖面；
        2. 上下文组装按预算分配——结构化规则为主体必保留，RAG 补充
           按剩余预算顺序填入，避免规则多时硬截断掉关键判定阈值。

        :return: (送给视觉模型的规则文本, 缺陷类型列表)
        """
        rules = db.query_all(
            "SELECT defect_type, content FROM rules WHERE status='active'"
        )
        if not rules:
            # 没有结构化规则时（LLM 抽取为空）退化为纯 RAG 片段
            chunks = vector_store.query("工业产品表面缺陷 判定标准 气孔 划痕", top_k=6)
            context = "\n".join(
                f"- [{c.metadata.get('doc_name', '')}] {c.text}" for c in chunks
            )
            return context or "（未检索到明确规则，请按通用工业表面缺陷经验判断）", []

        types = sorted({r["defect_type"] for r in rules})
        lines = [f"- 【{r['defect_type']}】{r['content']}" for r in rules]
        # 对每种类型补充最相关的原文片段
        extras: List[str] = []
        # 去重 key：文本前 64 字符。既过滤与结构化规则高度重叠的片段，
        # 也过滤跨 dtype 检索命中的同一原文切片（vector_store 内部
        # 已用 doc[:32] 去重，这里覆盖跨 query 场景）
        seen_keys: Set[str] = {r["content"][:64] for r in rules}
        for dtype in types:
            for hit in vector_store.query(f"{dtype} 判定标准", top_k=2):
                dedup_key = hit.text[:64]
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)
                extras.append(
                    f"- 【{dtype}·补充·{hit.metadata.get('doc_name', '')}"
                    f"{hit.metadata.get('source_location', '')}】{hit.text}"
                )
        # 上下文组装：结构化规则主体必保留，RAG 补充按剩余预算顺序填入
        budget = 12000
        context = "检测规则（必须严格遵守）：\n" + "\n".join(lines)
        if extras:
            supplement = "\n\n规则原文补充：\n"
            remaining = budget - len(context) - len(supplement)
            kept: List[str] = []
            for line in extras:
                if len(line) + 1 > remaining:  # +1 给换行符
                    break
                kept.append(line)
                remaining -= len(line) + 1
            if kept:
                context += supplement + "\n".join(kept)
        return context[:budget], types

    @staticmethod
    def _collect_evidences(defect_type: str) -> List[Dict[str, Any]]:
        """为缺陷结论检索规则依据（相似度阈值在向量库内已生效）。"""
        hits = vector_store.query(
            f"{defect_type} 缺陷 判定标准 阈值", top_k=3
        )
        return [
            {
                "doc_name": h.metadata.get("doc_name", "未知文档"),
                "source_location": h.metadata.get("source_location", ""),
                "content": h.text[:500],
                "similarity": h.similarity,
            }
            for h in hits
        ]

    # 检测引擎展示名（engine 标识 → 人类可读描述）
    _ENGINE_LABELS = {
        "vlm": "本地定位+多模态大模型裁决",
        "yolo": "本地深度模型（PatchCore 哨兵 + YOLO-World 定位）",
        "cv": "经典视觉算法（离线兜底）",
    }

    async def _compose_summary(self, defects: List[Dict], engine: str) -> str:
        """生成结论文本：优先让文本 LLM 组织，失败用模板兜底。"""
        if not defects:
            return "未检出缺陷，该产品表面符合当前规则文档的判定标准。"
        # 类型聚合统计
        counter: Dict[str, int] = {}
        for d in defects:
            counter[d["defect_type"]] = counter.get(d["defect_type"], 0) + 1
        type_text = "、".join(f"{k}×{v}" for k, v in counter.items())
        engine_label = self._ENGINE_LABELS.get(engine, engine)
        template = (
            f"共检出 {len(defects)} 处缺陷：{type_text}。"
            f"检测引擎：{engine_label}，请结合标注框与规则依据复核。"
        )
        try:
            llm = get_text_llm()
            brief = [
                f"{d['defect_type']}(置信度{d['confidence']:.2f},"
                f"位置{d['bbox']})"
                for d in defects[:10]
            ]
            messages = [
                {
                    "role": "system",
                    "content": "你是工业缺陷检测报告助手，用一句话中文总结检测结论。",
                },
                {
                    "role": "user",
                    "content": f"检测到：{';'.join(brief)}。请给现场质检人员一句话结论。",
                },
            ]
            text = (await llm.chat(messages, temperature=0.3)).strip()
            return text or template
        except Exception:  # noqa: BLE001
            return template

    @staticmethod
    def _get_job_row(job_id: str) -> Dict[str, Any]:
        row = db.query_one(
            "SELECT * FROM inspection_jobs WHERE id=?", (job_id,)
        )
        if not row:
            raise NotFoundError("检测任务不存在。")
        return row

    @staticmethod
    def _serialize_defect(defect: Dict[str, Any]) -> Dict[str, Any]:
        """
        把检测引擎内部缺陷结构转换为 API 约定结构：
        bbox 由内部的 [x, y, w, h] 列表转换为 {x, y, width, height} 对象。
        """
        bbox = defect.get("bbox") or [0, 0, 0, 0]
        x, y, w, h = (int(v) for v in bbox[:4])
        polygon = defect.get("mask_polygon")
        if polygon:
            # OpenCV / numpy 元素统一转成 Python int，保证 JSON / Pydantic 兼容
            polygon = [[int(p[0]), int(p[1])] for p in polygon]
        return {
            "defect_type": defect.get("defect_type", "未知缺陷"),
            "confidence": float(defect.get("confidence", 0.0)),
            "bbox": {"x": x, "y": y, "width": w, "height": h},
            "segmentation_available": bool(
                defect.get("segmentation_available", polygon)
            ),
            "mask_polygon": polygon,
            "description": defect.get("description", ""),
            "evidences": defect.get("evidences") or [],
            "tile_index": int(defect.get("tile_index", -1)),
        }

    @staticmethod
    def _hydrate(row: Dict[str, Any]) -> Dict[str, Any]:
        """把数据库行转换为 API 返回结构（解析结果 JSON、补图片 URL）。"""
        raw_defects = json.loads(row.get("result_json") or "[]")
        defects = [
            InspectionService._serialize_defect(d) for d in raw_defects
        ]
        return {
            "id": row["id"],
            "image_name": row["image_name"],
            "width": row["width"],
            "height": row["height"],
            "status": row["status"],
            "progress": row["progress"],
            "stage": row["stage"],
            "has_defect": bool(row["has_defect"]),
            "summary": row["summary"],
            "defects": defects,
            "error_message": row["error_message"],
            "created_at": row["created_at"],
            "completed_at": row.get("completed_at", ""),
            "image_url": f"/api/inspection/image/{row['id']}",
            "annotated_url": f"/api/inspection/results/{row['annotated_path']}"
            if row["annotated_path"]
            else "",
        }


# 全局单例
inspection_service = InspectionService()
