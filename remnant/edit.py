"""`memory_edit` actions: update, merge, forget, feedback, share, unshare.

Every action writes to `audit_log` with a before/after snapshot. Nothing is
deleted: ``forget`` sets ``status='forgotten'``, ``update``/``merge`` mark old
memories as ``status='superseded'``. The returned dict always carries the
resulting memory id(s) and the audit log id(s).
"""

from __future__ import annotations

import logging
from typing import Any

from .config import RemnantConfig
from .db import RemnantDB
from .embed import Embedder

log = logging.getLogger("remnant.edit")

# Trust-score adjustment deltas for the `feedback` action.
_FEEDBACK_DELTAS = {
    "useful": +0.1,
    "wrong": -0.2,
}

# Visibility transitions for `share` / `unshare`.
_VISIBILITY_TRANSITIONS = {
    "share": "shared",
    "unshare": "private",
}


def memory_edit(
    db: RemnantDB,
    config: RemnantConfig,
    embedder: Embedder,
    *,
    action: str,
    actor: str,
    memory_id: str | None = None,
    memory_ids: list[str] | None = None,
    content: str | None = None,
    visibility: str | None = None,
    feedback: str | None = None,
    agent_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Dispatch a `memory_edit` action. Returns a result dict.

    Actions:
      - ``update``: create a new version of `memory_id` with `content`, mark
        the old one superseded. Returns the new memory id.
      - ``merge``: combine `memory_ids` into one new memory with `content`,
        supersede all originals. Returns the new memory id.
      - ``forget``: mark `memory_id` status='forgotten' (row preserved).
      - ``feedback``: adjust `trust_score` of `memory_id` by `feedback`
        ('useful' raises, 'wrong' lowers).
      - ``share``: promote `memory_id` visibility to 'shared'.
      - ``unshare``: revert `memory_id` visibility to 'private'.
    """
    action = (action or "").strip().lower()
    if action not in _ACTIONS:
        return {"error": f"unknown action: {action}"}
    return _ACTIONS[action](
        db,
        config,
        embedder,
        actor=actor,
        memory_id=memory_id,
        memory_ids=memory_ids,
        content=content,
        visibility=visibility,
        feedback=feedback,
        agent_id=agent_id,
        session_id=session_id,
    )


def _do_update(
    db: RemnantDB,
    config: RemnantConfig,
    embedder: Embedder,
    *,
    actor: str,
    memory_id: str | None,
    memory_ids: list[str] | None,
    content: str | None,
    visibility: str | None,
    feedback: str | None,
    agent_id: str | None,
    session_id: str | None,
) -> dict[str, Any]:
    if not memory_id:
        return {"error": "memory_id is required for update"}
    new_content = (content or "").strip()
    if not new_content:
        return {"error": "content is required for update"}
    before = db.get_memory(memory_id)
    if before is None:
        return {"error": f"memory not found: {memory_id}"}
    # Reuse the old memory's scope defaults.
    vis = visibility or before.get("visibility") or config.default_visibility
    aid = agent_id or before.get("agent") or config.agent_id
    # Re-embed the new content if an embedder is available.
    embedding = embedder.embed(new_content) if embedder else None
    new_mid = db.insert_memory(
        content=new_content,
        source="manual",
        agent=aid,
        visibility=vis,
        source_id=str(before.get("source_id")) if before.get("source_id") else None,
        type=before.get("type") or "fact",
        tags=before.get("tags") if isinstance(before.get("tags"), list) else None,
        metadata={
            **(before.get("metadata") or {}),
            "updated_from": memory_id,
            **({"session_id": session_id} if session_id else {}),
        },
        confidence=before.get("confidence") or 0.5,
        embedding=embedding or None,
        embed_model=getattr(embedder, "_model", None) if embedder else None,
    )
    # Supersede the old memory and point it at the new one.
    db.supersede(memory_id, new_mid, actor=actor)
    # Re-link the same entities to the new memory so the graph stays connected.
    _carry_over_entity_links(db, old_id=memory_id, new_id=new_mid, agent_id=aid)
    audit_id = db.write_audit(
        actor=actor,
        action="update",
        memory_id=new_mid,
        details={"before_id": memory_id, "before": _snapshot(before), "after_id": new_mid},
    )
    return {"memory_id": new_mid, "superseded_id": memory_id, "audit_id": audit_id}


def _do_merge(
    db: RemnantDB,
    config: RemnantConfig,
    embedder: Embedder,
    *,
    actor: str,
    memory_id: str | None,
    memory_ids: list[str] | None,
    content: str | None,
    visibility: str | None,
    feedback: str | None,
    agent_id: str | None,
    session_id: str | None,
) -> dict[str, Any]:
    ids = list(memory_ids or [])
    if memory_id and memory_id not in ids:
        ids.append(memory_id)
    if len(ids) < 2:
        return {"error": "merge requires at least two memory_ids"}
    new_content = (content or "").strip()
    if not new_content:
        return {"error": "content is required for merge"}
    # Pull all originals to inherit scope.
    originals = []
    for mid in ids:
        m = db.get_memory(mid)
        if m is None:
            return {"error": f"memory not found: {mid}"}
        originals.append(m)
    # Inherit agent + visibility from the first original.
    first = originals[0]
    aid = agent_id or first.get("agent") or config.agent_id
    vis = visibility or first.get("visibility") or config.default_visibility
    embedding = embedder.embed(new_content) if embedder else None
    merged_tags: list[str] = []
    for o in originals:
        tags = o.get("tags")
        if isinstance(tags, list):
            merged_tags.extend(tags)
    merged_tags = list(dict.fromkeys(merged_tags))
    new_mid = db.insert_memory(
        content=new_content,
        source="manual",
        agent=aid,
        visibility=vis,
        type="fact",
        tags=merged_tags or None,
        metadata={
            "merged_from": ids,
            **({"session_id": session_id} if session_id else {}),
        },
        embedding=embedding or None,
        embed_model=getattr(embedder, "_model", None) if embedder else None,
    )
    # Supersede every original and re-link its entities to the merged memory.
    for o in originals:
        db.supersede(o["id"], new_mid, actor=actor)
        _carry_over_entity_links(db, old_id=o["id"], new_id=new_mid, agent_id=aid)
    audit_id = db.write_audit(
        actor=actor,
        action="merge",
        memory_id=new_mid,
        details={"merged_from": ids, "after_id": new_mid},
    )
    return {"memory_id": new_mid, "superseded_ids": ids, "audit_id": audit_id}


def _do_forget(
    db: RemnantDB,
    config: RemnantConfig,
    embedder: Embedder,
    *,
    actor: str,
    memory_id: str | None,
    memory_ids: list[str] | None,
    content: str | None,
    visibility: str | None,
    feedback: str | None,
    agent_id: str | None,
    session_id: str | None,
) -> dict[str, Any]:
    if not memory_id:
        return {"error": "memory_id is required for forget"}
    before = db.get_memory(memory_id)
    if before is None:
        return {"error": f"memory not found: {memory_id}"}
    audit_id = db.set_memory_field(
        memory_id,
        "status",
        "forgotten",
        actor=actor,
        action="forget",
        details={"before": _snapshot(before), "after_status": "forgotten"},
    )["audit_id"]
    return {"memory_id": memory_id, "status": "forgotten", "audit_id": audit_id}


def _do_feedback(
    db: RemnantDB,
    config: RemnantConfig,
    embedder: Embedder,
    *,
    actor: str,
    memory_id: str | None,
    memory_ids: list[str] | None,
    content: str | None,
    visibility: str | None,
    feedback: str | None,
    agent_id: str | None,
    session_id: str | None,
) -> dict[str, Any]:
    if not memory_id:
        return {"error": "memory_id is required for feedback"}
    fb = (feedback or "").strip().lower()
    if fb not in _FEEDBACK_DELTAS:
        return {"error": f"feedback must be one of: {sorted(_FEEDBACK_DELTAS)}"}
    before = db.get_memory(memory_id)
    if before is None:
        return {"error": f"memory not found: {memory_id}"}
    delta = _FEEDBACK_DELTAS[fb]
    current = before.get("trust_score")
    base = float(current) if current is not None else 0.5
    new_score = max(0.0, min(1.0, base + delta))
    res = db.set_memory_field(
        memory_id,
        "trust_score",
        new_score,
        actor=actor,
        action="feedback",
        details={
            "feedback": fb,
            "before_score": before.get("trust_score"),
            "after_score": new_score,
        },
    )
    return {
        "memory_id": memory_id,
        "trust_score": new_score,
        "audit_id": res["audit_id"],
    }


def _do_visibility(
    db: RemnantDB,
    config: RemnantConfig,
    embedder: Embedder,
    *,
    action: str,
    actor: str,
    memory_id: str | None,
    agent_id: str | None,
    session_id: str | None,
) -> dict[str, Any]:
    if not memory_id:
        return {"error": f"memory_id is required for {action}"}
    target = _VISIBILITY_TRANSITIONS[action]
    before = db.get_memory(memory_id)
    if before is None:
        return {"error": f"memory not found: {memory_id}"}
    res = db.set_memory_field(
        memory_id,
        "visibility",
        target,
        actor=actor,
        action=action,
        details={"before": before.get("visibility"), "after": target},
    )
    return {"memory_id": memory_id, "visibility": target, "audit_id": res["audit_id"]}


def _do_share(
    db: RemnantDB,
    config: RemnantConfig,
    embedder: Embedder,
    *,
    actor: str,
    memory_id: str | None,
    memory_ids: list[str] | None,
    content: str | None,
    visibility: str | None,
    feedback: str | None,
    agent_id: str | None,
    session_id: str | None,
) -> dict[str, Any]:
    return _do_visibility(
        db, config, embedder,
        action="share", actor=actor, memory_id=memory_id,
        agent_id=agent_id, session_id=session_id,
    )


def _do_unshare(
    db: RemnantDB,
    config: RemnantConfig,
    embedder: Embedder,
    *,
    actor: str,
    memory_id: str | None,
    memory_ids: list[str] | None,
    content: str | None,
    visibility: str | None,
    feedback: str | None,
    agent_id: str | None,
    session_id: str | None,
) -> dict[str, Any]:
    return _do_visibility(
        db, config, embedder,
        action="unshare", actor=actor, memory_id=memory_id,
        agent_id=agent_id, session_id=session_id,
    )


def _carry_over_links_to_new(
    db: RemnantDB, *, old_id: str, new_id: str, agent_id: str | None
) -> None:
    """Deprecated alias; kept for clarity. See `_carry_over_entity_links`."""
    _carry_over_entity_links(db, old_id=old_id, new_id=new_id, agent_id=agent_id)


def _carry_over_entity_links(
    db: RemnantDB, *, old_id: str, new_id: str, agent_id: str | None
) -> None:
    """Copy memory_entities links from an old memory to a new one.

    Called after update/merge so the new version stays connected in the graph.
    """
    with db.read() as cur:
        cur.execute(
            "SELECT entity_id, relation_role FROM memory_entities WHERE memory_id=?",
            (old_id,),
        )
        rows = cur.fetchall()
    for r in rows:
        db.link_entity(
            memory_id=new_id,
            entity_id=r["entity_id"],
            agent_id=agent_id,
            relation_role=r["relation_role"],
        )


def _snapshot(mem: dict[str, Any]) -> dict[str, Any]:
    """Compact before/after snapshot for the audit log."""
    if not mem:
        return {}
    return {
        "id": mem.get("id"),
        "content": mem.get("content"),
        "visibility": mem.get("visibility"),
        "status": mem.get("status"),
        "trust_score": mem.get("trust_score"),
        "tags": mem.get("tags"),
    }


_ACTIONS = {
    "update": _do_update,
    "merge": _do_merge,
    "forget": _do_forget,
    "feedback": _do_feedback,
    "share": _do_share,
    "unshare": _do_unshare,
}


__all__ = ["memory_edit"]
