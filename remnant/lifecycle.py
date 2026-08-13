"""Atomic high-level lifecycle operations for Remnant projections."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .claims import _claim_parts
from .config import RemnantConfig
from .db import RemnantDB
from .embed import Embedder


class MemoryLifecycle:
    def __init__(self, db: RemnantDB, config: RemnantConfig, embedder: Embedder | None):
        self.db = db
        self.config = config
        self.embedder = embedder

    def replace(
        self,
        *,
        original_ids: list[str],
        content: str,
        actor: str,
        agent_id: str | None,
        visibility: str | None = None,
        session_id: str | None = None,
        action: str = "update",
    ) -> dict[str, Any]:
        """Prepare remote embedding, then atomically commit all local projections."""
        originals = [self.db.get_memory(memory_id) for memory_id in original_ids]
        if any(memory is None for memory in originals):
            raise KeyError("one or more original memories do not exist")
        rows = [memory for memory in originals if memory is not None]
        if agent_id is not None and any(row.get("agent") != agent_id for row in rows):
            raise PermissionError("memory is owned by another agent")
        first = rows[0]
        embedding = self.embedder.embed(content) if self.embedder is not None else None
        tags: list[str] = []
        for row in rows:
            if isinstance(row.get("tags"), list):
                tags.extend(row["tags"])
        metadata = {
            "replaces": original_ids,
            **({"session_id": session_id} if session_id else {}),
        }
        claims = [self.db.get_claim_for_memory(str(row["id"])) for row in rows]
        existing_claims = [claim for claim in claims if claim is not None]
        claim_projection: dict[str, Any] | None = None
        if existing_claims and all(
            claim["subject"].casefold() == existing_claims[0]["subject"].casefold()
            and claim["predicate"].casefold() == existing_claims[0]["predicate"].casefold()
            for claim in existing_claims
        ):
            predecessor = existing_claims[0]
            _, object_value = _claim_parts(str(predecessor["subject"]), content)
            claim_projection = {
                "subject": predecessor["subject"],
                "predicate": predecessor["predicate"],
                "object": object_value or content,
                "qualifiers": predecessor.get("qualifiers"),
                "confidence": predecessor.get("confidence"),
                "scope_type": predecessor.get("scope_type"),
                "scope_value": predecessor.get("scope_value"),
                "modality": predecessor.get("modality"),
                "source_turn_id": predecessor.get("source_turn_id"),
            }
        return self.db.replace_memories_atomic(
            original_ids=original_ids,
            content=content,
            source="manual",
            agent=agent_id or first.get("agent") or self.config.agent_id,
            visibility=visibility or first.get("visibility") or self.config.default_visibility,
            type="fact" if len(rows) > 1 else str(first.get("type") or "fact"),
            tags=list(dict.fromkeys(tags)) or None,
            metadata=metadata,
            confidence=float(first.get("confidence") or 0.5),
            trust_score=float(first.get("trust_score") or 0.5),
            embedding=embedding,
            embed_model=getattr(self.embedder, "_model", None),
            claim_projection=claim_projection,
            actor=actor,
            action=action,
        )

    def forget(self, memory_id: str, *, actor: str, agent_id: str | None) -> int:
        memory = self.db.get_memory(memory_id)
        if memory is None:
            raise KeyError(memory_id)
        if agent_id is not None and memory.get("agent") != agent_id:
            raise PermissionError("memory is owned by another agent")
        return self.db.transition_memory_atomic(
            memory_id, status="forgotten", actor=actor, action="forget"
        )

    def visibility(
        self,
        memory_id: str,
        *,
        value: str,
        actor: str,
        agent_id: str | None,
        action: str,
    ) -> int:
        memory = self.db.get_memory(memory_id)
        if memory is None:
            raise KeyError(memory_id)
        if agent_id is not None and memory.get("agent") != agent_id:
            raise PermissionError("memory is owned by another agent")
        return self.db.transition_memory_atomic(
            memory_id, visibility=value, actor=actor, action=action
        )


def backfill_relation_evidence(
    db: RemnantDB, *, dry_run: bool = True, limit: int = 1000
) -> dict[str, int | bool]:
    """Derive evidence rows from legacy relation source IDs, idempotently."""
    with db.read() as cur:
        cur.execute(
            "SELECT entity_a, entity_b, relation_type, strength, source_memory_id "
            "FROM relations WHERE source_memory_id IS NOT NULL ORDER BY created_at LIMIT ?",
            (max(1, int(limit)),),
        )
        rows = [dict(row) for row in cur.fetchall()]
    if dry_run:
        return {"dry_run": True, "eligible": len(rows), "written": 0}
    written = 0
    now = datetime.now(timezone.utc).isoformat()
    with db.transaction() as cur:
        for row in rows:
            cur.execute(
                "SELECT id FROM claims WHERE memory_id=?",
                (row["source_memory_id"],),
            )
            claim = cur.fetchone()
            cur.execute(
                "INSERT OR IGNORE INTO relation_evidence(entity_a, entity_b, relation_type, "
                "memory_id, claim_id, strength, active, created_at, updated_at) "
                "VALUES(?,?,?,?,?,?,1,?,?)",
                (
                    row["entity_a"], row["entity_b"], row["relation_type"],
                    row["source_memory_id"], claim["id"] if claim else None,
                    float(row.get("strength") or 0.5), now, now,
                ),
            )
            written += max(0, cur.rowcount)
    return {"dry_run": False, "eligible": len(rows), "written": written}
__all__ = ["MemoryLifecycle", "backfill_relation_evidence"]
