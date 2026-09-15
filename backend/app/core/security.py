# -*- coding: utf-8 -*-
"""
文件安全校验工具
================
集中处理上传链路的安全问题：
1. 扩展名白名单（区分文档 / 图片）；
2. 文件头魔数（magic bytes）校验，防止把可执行文件改后缀上传；
3. 文件名消毒（path traversal 防护，拒绝 ``..``、绝对路径、盘符）；
4. ``safe_join`` 保证最终写入路径一定位于允许的基目录之内。
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Tuple

from app.config import settings
from app.core.exceptions import (
    FileTooLargeError,
    UnsupportedFileError,
    ValidationError,
)

# 常见格式的魔数特征：(起始偏移, 字节前缀元组)
# PDF: %PDF ; PNG / JPEG / BMP / TIFF / ZIP(docx 本质是 zip)
_MAGIC_TABLE = {
    ".pdf": (0, (b"%PDF",)),
    ".docx": (0, (b"PK\x03\x04",)),
    ".png": (0, (b"\x89PNG\r\n\x1a\n",)),
    ".jpg": (0, (b"\xff\xd8\xff",)),
    ".jpeg": (0, (b"\xff\xd8\xff",)),
    ".bmp": (0, (b"BM",)),
    ".tif": (0, (b"II*\x00", b"MM\x00*")),
    ".tiff": (0, (b"II*\x00", b"MM\x00*")),
    # webp: RIFF????WEBP
    ".webp": (0, (b"RIFF",)),
}

# 文件名只保留中英文、数字、下划线、短横线、点；其余字符一律替换
_SAFE_NAME_RE = re.compile(r"[^\w\u4e00-\u9fa5.\-]+", re.UNICODE)


def sanitize_filename(filename: str) -> str:
    """
    消毒用户上传文件名：
    * 去除任何路径分隔符与 ``..``；
    * 非法字符替换为下划线；
    * 拼上随机前缀，避免同名覆盖与枚举攻击。

    :param filename: 原始文件名
    :return: 安全的存储文件名（保留原始扩展名）
    """
    if not filename:
        raise ValidationError("文件名不能为空。")

    # 只取 basename，剥离客户端可能伪造的路径
    name = Path(filename.replace("\\", "/")).name
    name = name.replace("..", "_").strip()
    if not name or name in {".", "_"}:
        raise ValidationError("非法文件名。")

    cleaned = _SAFE_NAME_RE.sub("_", name)
    # 8 位随机前缀防冲突 / 防枚举
    return f"{uuid.uuid4().hex[:8]}_{cleaned}"


def safe_join(base_dir: Path, filename: str) -> Path:
    """
    把文件名安全地拼接到基目录下，并确保结果不会逃逸出基目录。

    :param base_dir: 允许写入的基目录（必须为绝对路径）
    :param filename: 已消毒的文件名（不得包含路径分隔符）
    :raises ValidationError: 当解析结果逃逸出基目录时
    """
    if "/" in filename or "\\" in filename or filename in {"", ".", ".."}:
        raise ValidationError("非法的存储路径。")
    base = base_dir.resolve()
    target = (base / filename).resolve()
    if base != target and base not in target.parents:
        raise ValidationError("文件路径越界，拒绝写入。")
    return target


def check_extension(filename: str, allowed: Tuple[str, ...]) -> str:
    """
    校验扩展名白名单，返回小写扩展名（含点，如 ``.pdf``）。
    """
    ext = Path(filename).suffix.lower()
    if ext not in allowed:
        raise UnsupportedFileError(
            f"不支持的文件格式 {ext or '(无扩展名)'}，允许：{', '.join(allowed)}"
        )
    return ext


def check_size(size_bytes: int, limit_mb: int, kind: str) -> None:
    """校验文件大小。:param kind: 用于错误提示的文件类型中文名。"""
    if size_bytes <= 0:
        raise ValidationError(f"{kind}内容为空。")
    max_bytes = limit_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise FileTooLargeError(
            f"{kind}大小 {size_bytes / 1024 / 1024:.1f}MB 超过上限 {limit_mb}MB。"
        )


def sniff_extension(head: bytes, claimed_ext: str) -> str:
    """
    用文件头魔数校验“真实格式”。

    :param head: 文件起始字节（建议至少 16 字节）
    :param claimed_ext: 上传时声明的扩展名
    :return: 校验通过后返回真实扩展名；webp 需额外检查 RIFF....WEBP
    :raises UnsupportedFileError: 魔数与声明类型不符
    """
    if claimed_ext == ".webp":
        if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
            return claimed_ext
        raise UnsupportedFileError("文件内容不是有效的 WebP 图像。")

    offset, prefixes = _MAGIC_TABLE.get(claimed_ext, (0, ()))
    if not prefixes:
        # 未登记魔数的类型直接放行，交由后续解析器验证
        return claimed_ext
    if any(head[offset : offset + len(p)] == p for p in prefixes):
        return claimed_ext
    raise UnsupportedFileError(
        "文件内容与扩展名不匹配，可能是伪造文件，已拒绝上传。"
    )


def validate_document(filename: str, head: bytes, size_bytes: int) -> str:
    """文档上传的完整校验入口，返回小写扩展名。"""
    ext = check_extension(filename, tuple(settings.allowed_doc_ext_list))
    check_size(size_bytes, settings.max_doc_size_mb, "规则文档")
    sniff_extension(head, ext)
    return ext


def validate_image(filename: str, head: bytes, size_bytes: int) -> str:
    """图片上传的完整校验入口，返回小写扩展名。"""
    ext = check_extension(filename, tuple(settings.allowed_image_ext_list))
    check_size(size_bytes, settings.max_image_size_mb, "待测图片")
    sniff_extension(head, ext)
    return ext
