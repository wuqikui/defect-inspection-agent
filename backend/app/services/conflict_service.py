# -*- coding: utf-8 -*-
"""
规则冲突检测与解决服务
======================
工作流（对应需求“自动识别冲突 → 逐条呈现 → 用户确认 → 消除全部冲突”）：

1. ``recompute()``：取所有 active 规则（联表带文档名）交 LLM/离线引擎
   做两两矛盾分析；对新发现的规则对写入 conflicts 表（pending）。
   已存在（pending 或 resolved）的规则对不重复入库——用户已裁决的冲突
   不会因再次扫描而“复活”。
2. 前端通过列表接口逐条获取 pending 冲突，每条给出 A/B 两种原文表述。
3. ``resolve()``：用户逐条裁决：
   * 选 A：规则 B 置 superseded（淘汰，不进入检测依据）；
   * 选 B：规则 A 置 superseded；
   * CUSTOM：A/B 均淘汰，写入一条“用户确认”的合成规则作为最终标准。
4. 系统在仍有 pending 冲突时拒绝发起检测（ConflictStateError），
   确保检测时依据的是一套无矛盾的规则集。
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

from app.core.exceptions import ConflictStateError, NotFoundError
from app.db.database import db, utcnow_iso
from app.services.llm import get_text_llm


class ConflictService:
    """冲突的检测、列出、裁决。"""

    # ------------------------------------------------------------------
    # 检测
    # ------------------------------------------------------------------
    async def recompute(self) -> int:
        """
        基于当前 active 规则重新扫描冲突，写入新出现的 pending 冲突。

        :return: 本次新发现的冲突条数
        """
        rules = self._active_rules_with_docs()
        # 只有一份文档 / 规则太少时不可能有跨文档冲突
        doc_count = len({r["doc_id"] for r in rules})
        if doc_count < 2 or len(rules) < 2:
            return 0

        llm = get_text_llm()
        try:
            raw_conflicts = await llm.detect_rule_conflicts(rules)
        except Exception:  # noqa: BLE001
            # 冲突检测失败不应阻断文档上传主流程
            return 0

        # 已存在的规则对（含已解决），避免重复提示与裁决复活
        existing = {
            self._pair_key(r["rule_a_id"], r["rule_b_id"])
            for r in db.query_all("SELECT rule_a_id, rule_b_id FROM conflicts")
        }
        rule_index = {r["id"]: r for r in rules}
        inserted = 0
        for item in raw_conflicts:
            pair = self._validate_pair(item, rule_index)
            if pair is None:
                continue
            key = self._pair_key(pair[0], pair[1])
            if key in existing:
                continue
            existing.add(key)
            self._insert_conflict(item, pair[0], pair[1], rule_index)
            inserted += 1
        return inserted

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def list_conflicts(self) -> List[Dict[str, Any]]:
        """按状态返回全部冲突（pending 排在最前，便于用户逐条处理）。"""
        return db.query_all(
            "SELECT * FROM conflicts ORDER BY "
            "CASE status WHEN 'pending' THEN 0 ELSE 1 END, created_at"
        )

    def pending_count(self) -> int:
        return db.query_one(
            "SELECT COUNT(*) AS c FROM conflicts WHERE status='pending'"
        )["c"]

    def ensure_no_pending(self) -> None:
        """发起检测前的门禁：存在未解决冲突时拒绝检测。"""
        count = self.pending_count()
        if count > 0:
            raise ConflictStateError(
                f"当前仍有 {count} 条规则冲突未确认，请先在“冲突中心”逐条解决后再检测。"
            )

    # ------------------------------------------------------------------
    # 裁决
    # ------------------------------------------------------------------
    def resolve(self, conflict_id: str, choice: str,
                custom_text: str = "") -> Dict[str, Any]:
        """
        解决单条冲突。

        :param conflict_id: 冲突 id
        :param choice: A / B / CUSTOM
        :param custom_text: CUSTOM 时用户确认的统一规则表述
        """
        conflict = db.query_one(
            "SELECT * FROM conflicts WHERE id=?", (conflict_id,)
        )
        if not conflict:
            raise NotFoundError("冲突不存在或已被删除。")
        if conflict["status"] == "resolved":
            raise ConflictStateError("该冲突已解决，不能重复裁决。")

        if choice == "CUSTOM":
            text = custom_text.strip()
            if not text:
                raise ConflictStateError("选择自定义方案时必须填写统一规则表述。")
            # 双方规则都淘汰，写入用户确认的合成规则（挂在文档 A 下满足外键）
            self._set_status(conflict["rule_a_id"], "superseded")
            self._set_status(conflict["rule_b_id"], "superseded")
            self._insert_custom_rule(conflict, text)
            resolution_text = text
        elif choice == "A":
            self._set_status(conflict["rule_b_id"], "superseded")
            resolution_text = conflict["content_a"]
        elif choice == "B":
            self._set_status(conflict["rule_a_id"], "superseded")
            resolution_text = conflict["content_b"]
        else:
            raise ConflictStateError(f"无效的裁决选项：{choice}")

        db.execute(
            """UPDATE conflicts
                  SET status='resolved', resolution_choice=?,
                      resolution_text=?, resolved_at=?
                WHERE id=?""",
            (choice, resolution_text, utcnow_iso(), conflict_id),
        )
        return db.query_one("SELECT * FROM conflicts WHERE id=?", (conflict_id,))

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _active_rules_with_docs() -> List[Dict[str, Any]]:
        rows = db.query_all(
            """
            SELECT r.id, r.doc_id, d.original_name AS doc_name,
                   r.defect_type, r.content, r.source_location
              FROM rules r JOIN documents d ON r.doc_id = d.id
             WHERE r.status='active'
            """
        )
        return rows

    @staticmethod
    def _pair_key(a: str, b: str) -> str:
        """规则对的无序唯一键（A-B 与 B-A 视为同一对）。"""
        return "|".join(sorted([a, b]))

    @staticmethod
    def _validate_pair(item: Dict[str, Any],
                       rule_index: Dict[str, Dict]) -> Optional[Tuple[str, str]]:
        """
        校验 LLM 返回的冲突：双方规则必须存在、当前为 active、且来自不同文档。
        """
        if not isinstance(item, dict):
            return None
        a_id = str(item.get("rule_a_id", ""))
        b_id = str(item.get("rule_b_id", ""))
        a, b = rule_index.get(a_id), rule_index.get(b_id)
        if not a or not b or a["doc_id"] == b["doc_id"]:
            return None
        return a_id, b_id

    @staticmethod
    def _insert_conflict(item: Dict, a_id: str, b_id: str,
                         rule_index: Dict[str, Dict]) -> None:
        a, b = rule_index[a_id], rule_index[b_id]
        db.execute(
            """INSERT INTO conflicts
               (id, defect_type, description, rule_a_id, rule_b_id,
                doc_a_id, doc_b_id, doc_a_name, doc_b_name,
                content_a, content_b, source_a, source_b,
                status, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                uuid.uuid4().hex,
                str(item.get("defect_type") or a["defect_type"])[:50],
                str(item.get("description", "规则表述存在矛盾，请确认以哪份文档为准。"))
                [:500],
                a_id,
                b_id,
                a["doc_id"],
                b["doc_id"],
                a["doc_name"],
                b["doc_name"],
                str(item.get("content_a") or a["content"]),
                str(item.get("content_b") or b["content"]),
                str(item.get("source_a") or a.get("source_location", "")),
                str(item.get("source_b") or b.get("source_location", "")),
                "pending",
                utcnow_iso(),
            ),
        )

    @staticmethod
    def _set_status(rule_id: str, status: str) -> None:
        db.execute("UPDATE rules SET status=? WHERE id=?", (status, rule_id))

    @staticmethod
    def _insert_custom_rule(conflict: Dict, text: str) -> None:
        """CUSTOM 裁决：落一条用户确认的合成规则（作为后续检测依据）。"""
        db.execute(
            """INSERT INTO rules
               (id, doc_id, defect_type, title, content, severity,
                source_location, status, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                uuid.uuid4().hex,
                conflict["doc_a_id"],
                conflict["defect_type"],
                "用户确认规则",
                text,
                "",
                "冲突人工裁决",
                "active",
                utcnow_iso(),
            ),
        )


# 全局单例
conflict_service = ConflictService()
