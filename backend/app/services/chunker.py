# -*- coding: utf-8 -*-
"""
文本分块器（Chunking）
======================
RAG 检索质量的关键环节。针对“规则图文文档”的特点采用策略：

1. 以解析得到的段落块（TextBlock）为最小单位，优先保留段落语义完整；
2. 超长段落按字符长度二次切分，切分时带 overlap（重叠窗口），
   避免判定阈值等关键数字被拦腰截断；
3. 每个 chunk 携带元数据：文档 id / 文件名 / 页码或段落（溯源）/
   是否图注 / chunk 序号，满足结果可解释、可溯源；
4. 图注块（caption）不单独入向量库，而是拼接到它前面最近的正文块上，
   使“图1：直径大于0.5mm的气孔”这类图示规则与文字描述一起被检索到。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from app.services.parser.base import ParsedDocument, TextBlock

# 单块目标字符数与重叠字符数（中文按字符计，约等于 token 数的 0.7~1 倍）
DEFAULT_CHUNK_SIZE = 480
DEFAULT_CHUNK_OVERLAP = 80


@dataclass
class Chunk:
    """向量库最小检索单元。"""

    text: str
    metadata: Dict[str, str] = field(default_factory=dict)


def _split_long_text(text: str, size: int, overlap: int) -> List[str]:
    """
    按字符长度滑窗切分超长文本，相邻切片有 ``overlap`` 字符重叠。

    :param text: 原始长文本
    :param size: 每片最大字符数
    :param overlap: 相邻片重叠字符数（必须 < size）
    """
    if len(text) <= size:
        return [text]
    step = size - overlap
    pieces: List[str] = []
    start = 0
    while start < len(text):
        piece = text[start : start + size]
        pieces.append(piece)
        if start + size >= len(text):
            break
        start += step
    return pieces


def build_chunks(
    parsed: ParsedDocument,
    doc_id: str,
    doc_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Chunk]:
    """
    将解析结果构建为可入库的 chunk 列表。

    :param parsed:   解析器输出
    :param doc_id:   文档主键（写入 metadata，用于按文档过滤 / 删除）
    :param doc_name: 文档原始名（溯源展示）
    """
    # 第一步：把图注合并到前一个正文块
    merged_units: List[TextBlock] = []
    for block in parsed.blocks:
        if block.kind == "caption" and merged_units:
            prev = merged_units[-1]
            merged_units[-1] = TextBlock(
                text=f"{prev.text}\n【图片说明】{block.text}",
                location=prev.location,
                kind="text",
            )
        else:
            merged_units.append(block)

    # 第二步：逐块切分并附加元数据
    chunks: List[Chunk] = []
    for idx, unit in enumerate(merged_units):
        pieces = _split_long_text(unit.text, chunk_size, chunk_overlap)
        for sub_idx, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    text=piece,
                    metadata={
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "source_location": unit.location,
                        "block_index": str(idx),
                        "sub_index": str(sub_idx),
                        "kind": unit.kind,
                    },
                )
            )
    return chunks
