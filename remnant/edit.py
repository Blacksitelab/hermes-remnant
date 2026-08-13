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
from .lifecycle import MemoryLifecycle

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
    if not _can_mutate(before, agent_id):
        return {"error": "memory is owned by another agent"}
    try:
        result = MemoryLifecycle(db, config, embedder).replace(
            original_ids=[memory_id],
            content=new_content,
            actor=actor,
            agent_id=agent_id,
            visibility=visibility,
            session_id=session_id,
            action="update",
        )
    except (KeyError, PermissionError, ValueError) as exc:
        return {"error": str(exc)}
    return {**result, "superseded_id": memory_id}


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
        if not _can_mutate(m, agent_id):
            return {"error": "memory is owned by another agent"}
        originals.append(m)
    try:
        result = MemoryLifecycle(db, config, embedder).replace(
            original_ids=ids,
            content=new_content,
            actor=actor,
            agent_id=agent_id,
            visibility=visibility,
            session_id=session_id,
            action="merge",
        )
    except (KeyError, PermissionError, ValueError) as exc:
        return {"error": str(exc)}
    return {**result, "superseded_ids": ids}


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
    if not _can_mutate(before, agent_id):
        return {"error": "memory is owned by another agent"}
    audit_id = MemoryLifecycle(db, config, embedder).forget(
        memory_id, actor=actor, agent_id=agent_id
    )
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
    if not _can_mutate(before, agent_id):
        return {"error": "memory is owned by another agent"}
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
    if not _can_mutate(before, agent_id):
        return {"error": "memory is owned by another agent"}
    audit_id = MemoryLifecycle(db, config, embedder).visibility(
        memory_id,
        value=target,
        actor=actor,
        agent_id=agent_id,
        action=action,
    )
    return {"memory_id": memory_id, "visibility": target, "audit_id": audit_id}


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


def _can_mutate(memory: dict[str, Any], agent_id: str | None) -> bool:
    """Provider edits are owner-only, including records shared for recall.

    ``memory_edit`` is also a public low-level helper used by maintenance
    scripts. Callers without an agent context retain the historical privileged
    behavior; the Hermes provider always supplies one.
    """
    if agent_id is None:
        return True
    owner = memory.get("agent")
    return bool(agent_id) and owner == agent_id


_ACTIONS = {
    "update": _do_update,
    "merge": _do_merge,
    "forget": _do_forget,
    "feedback": _do_feedback,
    "share": _do_share,
    "unshare": _do_unshare,
}


__all__ = ["memory_edit"]
