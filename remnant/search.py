"""Memory search: BM25 keyword, cosine semantic, RRF hybrid fusion, and graph.

Strategies:
- ``keyword``: BM25 over the FTS5 index. The Phase 1 behavior.
- ``semantic``: cosine similarity over a bounded, authorization-scoped vector
  corpus. This keeps lexical mismatch from hiding semantically relevant facts.
- ``auto`` (default): Reciprocal Rank Fusion (RRF, k=60) of BM25 + semantic.
  Results below ``min_semantic_score`` (top semantic score) are dropped to
  avoid returning keyword-dominated noise for semantically unrelated queries.
- ``graph``: pure-SQLite entity-graph traversal. Extracts entity names from
  the query, resolves them, BFS over `relations` up to N hops, and returns
  linked active memories. No LLM, no embeddings.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from .config import RRF_K, RemnantConfig
from .db import RemnantDB
from .embed import Embedder, cosine
from .resolve import retrieval_query
from .scope import (
    effective_profile_scope,
    normalize_profile_scope,
    path_in_profile_scope,
    visibility_allows,
)


def _scope_filter(results: list[dict[str, Any]], visibility: str | None) -> list[dict[str, Any]]:
    return [r for r in results if visibility_allows(r.get("visibility"), visibility)]


def _profile_scope_filter(
    results: list[dict[str, Any]], profile_scope: list[str] | None
) -> list[dict[str, Any]]:
    """Restrict vault/document memories to allowed path prefixes.

    Non-document memories are never filtered by profile_scope. A vault memory
    is identified by ``source='vault'``; legacy/imported document rows are
    also identified by ``type='document'`` so their paths cannot bypass the
    provider scope merely because their source label differs. ``None`` means
    no scope; an empty effective scope excludes document rows while retaining
    ordinary facts.
    A scoped document is included if its ``source_id`` starts with one of the
    allowed prefixes (after normalizing separators to ``/``).
    """
    if profile_scope is None:
        return results
    prefixes = normalize_profile_scope(profile_scope)
    if not prefixes:
        # An empty effective scope means no vault document is allowed. This is
        # distinct from ``None`` (no configured scope).
        return [
            r for r in results
            if r.get("source") != "vault" and r.get("type") != "document"
        ]
    out: list[dict[str, Any]] = []
    for r in results:
        # Legacy search rows may omit both fields; _attach_source() enriches
        # normal lanes before this function runs. Treat only explicit vault or
        # document rows as path-scoped so ordinary facts remain searchable.
        is_document = r.get("source") == "vault" or r.get("type") == "document"
        if not is_document:
            out.append(r)
            continue
        sid = r.get("source_id") or ""
        if path_in_profile_scope(sid, prefixes):
            out.append(r)
    return out


def _mask_locked(
    results: list[dict[str, Any]], *, viewer_agent: str | None
) -> list[dict[str, Any]]:
    """Hide content of locked vault documents from non-owner agents.

    A result is masked when its metadata carries ``locked=True`` and the viewer
    is not the memory's own owner agent (``r['agent']``). Masked rows keep
    id/title/path/metadata but have ``content`` replaced with a placeholder so
    other agents see that a note exists and its metadata, but not its body.
    Owner identity is per-row (the memory's ``agent`` column), not a single
    global owner, so a single search spanning multiple owners masks each row
    against its own author.
    """
    out: list[dict[str, Any]] = []
    for r in results:
        meta = r.get("metadata")
        if isinstance(meta, str):
            import json

            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = None
        owner = r.get("agent") or r.get("agent_id")
        is_locked = isinstance(meta, dict) and meta.get("locked") is True
        # Mask when the viewer is not the row's owner. A None viewer (anonymous
        # search) is treated as a non-owner: locked content is masked.
        is_owner = viewer_agent is not None and viewer_agent == owner
        if (
            r.get("source") == "vault"
            and is_locked
            and not is_owner
        ):
            d = dict(r)
            d["content"] = "[locked note: content hidden]"
            d["locked"] = True
            out.append(d)
        else:
            out.append(r)
    return out


def search(
    db: RemnantDB,
    config: RemnantConfig,
    query: str,
    *,
    agent_id: str | None = None,
    visibility: str | None = None,
    limit: int | None = None,
    strategy: str | None = None,
    embedder: Embedder | None = None,
    profile_scope: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run a memory search with the given strategy and scope filtering.

    - ``keyword``: BM25 only.
    - ``semantic``: cosine over embeddings, BM25-pre-filtered candidates.
    - ``auto`` (default): RRF fusion of keyword + semantic.

    ``profile_scope`` (Phase 4) restricts ``document``/``vault`` memories to
    those whose ``source_id`` starts with one of the allowed prefixes. ``None``
    means no configured scope; an empty effective scope excludes document rows
    while retaining ordinary facts. Locked vault notes have their content
    masked for any viewer that is not the memory's owner agent.

    For ``semantic`` and ``auto``, if the top semantic cosine score is below
    ``config.min_semantic_score`` (default 0.3), the semantic signal is weak.
    In ``semantic`` mode this means no results; in ``auto`` mode we fall back to
    the BM25 keyword results rather than returning nothing.
    """
    if limit is None:
        limit = config.search_limit
    if strategy is None:
        strategy = config.default_search_strategy
    if strategy not in ("keyword", "semantic", "auto", "graph"):
        strategy = config.default_search_strategy

    # A model-provided scope can narrow, but never broaden or disable, the
    # provider-configured scope.
    scope = effective_profile_scope(config.profile_scope, profile_scope)
    if not config.profile_scope and not profile_scope:
        scope = None
    # The "viewer" for locked-note masking is the agent performing the search:
    # an explicit agent_id wins, else the provider-configured agent_id. A None
    # viewer is treated as a non-owner anonymous search (locked content masked).
    viewer = agent_id if agent_id is not None else config.agent_id
    lexical_query = retrieval_query(query)
    history_intent = bool(
        re.search(
            r"\b(when|then|before|previously|used to|historical|history|at that time)\b",
            query,
            re.I,
        )
        or re.search(r"\b20\d{2}-[01]\d-[0-3]\d\b", query)
    )

    if strategy == "graph":
        from .graph import graph_search

        results = graph_search(
            db,
            lexical_query,
            agent_id=agent_id,
            limit=limit,
            profile_scope=scope,
            evidence_only=bool(getattr(config, "relation_evidence_enabled", False)),
        )
        results = _attach_source(db, results)
        results = _profile_scope_filter(results, scope)
        results = _scope_filter(results, visibility)
        results = _mask_locked(results, viewer_agent=viewer)
        results = results[:limit]
        return results

    if strategy == "keyword":
        results = db.search_bm25(
            lexical_query,
            agent_id=agent_id,
            profile_scope=scope,
            include_historical=history_intent and config.claim_aware_ranking_enabled,
            limit=limit * 3 if (visibility or scope) else limit,
        )
        results = _attach_source(db, results)
        results = _profile_scope_filter(results, scope)
        results = _scope_filter(results, visibility)
        results = _mask_locked(results, viewer_agent=viewer)
        results = results[:limit]
        return results

    if strategy == "semantic":
        ranked = _semantic_rank(
            db, config, query, agent_id=agent_id, embedder=embedder, profile_scope=scope
        )
        # Drop noise: if the top semantic cosine score is below the configured
        # minimum, there are no strong matches.
        if ranked and ranked[0].get("score", 0.0) < config.min_semantic_score:
            return []
        ranked = _attach_source(db, ranked)
        ranked = _profile_scope_filter(ranked, scope)
        ranked = _scope_filter(ranked, visibility)
        ranked = _mask_locked(ranked, viewer_agent=viewer)
        ranked = ranked[:limit]
        return ranked

    # auto: RRF fusion. If the top semantic score is below threshold, the
    # semantic signal is too weak for fusion, so we fall back to BM25-only
    # results instead of returning an empty list.
    kw = db.search_bm25(
        lexical_query,
        agent_id=agent_id,
        profile_scope=scope,
        include_historical=history_intent and config.claim_aware_ranking_enabled,
        limit=max(limit * 3, 100),
    )
    sem = _semantic_rank(
        db, config, query, agent_id=agent_id, embedder=embedder, profile_scope=scope
    )
    if sem and sem[0].get("score", 0.0) < config.min_semantic_score:
        results = _attach_source(db, kw)
        results = _profile_scope_filter(results, scope)
        results = _scope_filter(results, visibility)
        results = _mask_locked(results, viewer_agent=viewer)
        results = results[:limit]
        return results
    fused = _rrf_fuse(kw, sem)
    fused = _attach_source(db, fused)
    fused = _apply_source_weights(fused)
    fused = _profile_scope_filter(fused, scope)
    fused = _scope_filter(fused, visibility)
    fused = _mask_locked(fused, viewer_agent=viewer)
    fused = fused[:limit]
    return fused


def _apply_decay(
    current: float,
    updated_at: str | None,
    now: float,
    config: RemnantConfig,
) -> float:
    """Apply exponential time decay to a trust score.

    Half-life is ``config.trust_decay_half_life_days``. Decay never drops a
    score below ``config.trust_decay_floor``. Disabled if
    ``config.trust_decay_enabled`` is False.
    """
    if not getattr(config, "trust_decay_enabled", True):
        return current
    if not updated_at:
        return current
    try:
        dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        age_days = (now - dt.timestamp()) / 86400.0
    except (ValueError, TypeError):
        return current
    if age_days <= 0:
        return current
    half_life = float(
        getattr(config, "trust_decay_half_life_days", 30.0) or 30.0
    )
    if half_life <= 0:
        return current
    decayed = current * (0.5 ** (age_days / half_life))
    floor = float(getattr(config, "trust_decay_floor", 0.3))
    return max(decayed, floor)


def _utc_now() -> float:
    import time as _time

    return _time.time()


def decay_trust_scores(
    db: RemnantDB,
    config: RemnantConfig,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Batch decay every active memory's trust_score (issue #24).

    Applies the same exponential half-life decay used on query, but to the whole
    active corpus in one pass. Intended for cron/systemd timer jobs and for the
    dream pipeline's nightly run.

    Returns a summary dict: ``{"updated": int, "skipped": int, "dry_run": bool}``.
    """
    rows = db.list_active_memories_for_decay()
    now = _utc_now()
    updated = 0
    skipped = 0
    for r in rows:
        mid = r.get("id")
        if not mid:
            continue
        current = float(r.get("trust_score") or 0.5)
        decayed = _apply_decay(current, r.get("updated_at"), now, config)
        if abs(decayed - current) > 1e-9:
            if not dry_run:
                db.set_memory_field(
                    mid,
                    "trust_score",
                    decayed,
                    actor="system",
                    action="trust_decay_batch",
                    details={"old_score": current, "new_score": decayed},
                )
            updated += 1
        else:
            skipped += 1
    return {"updated": updated, "skipped": skipped, "dry_run": dry_run}


def _attach_source(db: RemnantDB, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enrich result rows with `source`, `source_id`, and `metadata` so that
    profile_scope filtering and locked masking can be applied in-process.

    search_bm25 / search_by_embedding don't JOIN source/source_id/metadata; we
    fetch them in one batched read to keep this cheap (bounded by the result
    set size, never the whole table).
    """
    if not results:
        return results
    ids = [r["id"] for r in results if r.get("id")]
    if not ids:
        return results
    placeholders = ",".join("?" for _ in ids)
    with db.read() as cur:
        cur.execute(
            f"SELECT id, agent, source, type, source_id, metadata, tags FROM memories "
            f"WHERE id IN ({placeholders})",
            ids,
        )
        rows = {r["id"]: dict(r) for r in cur.fetchall()}
    import json

    out: list[dict[str, Any]] = []
    for r in results:
        d = dict(r)
        meta = rows.get(d.get("id"), {})
        # Overwrite source/source_id even if the incoming row has a None value,
        # so that fusion paths that pre-populate these keys don't block
        # enrichment.
        if "agent" not in d or d.get("agent") is None:
            d["agent"] = meta.get("agent")
        if "agent_id" not in d or d.get("agent_id") is None:
            d["agent_id"] = meta.get("agent")
        if "source" not in d or d.get("source") is None:
            d["source"] = meta.get("source")
        if "type" not in d or d.get("type") is None:
            d["type"] = meta.get("type")
        if "source_id" not in d or d.get("source_id") is None:
            d["source_id"] = meta.get("source_id")
        raw_meta = meta.get("metadata")
        if isinstance(raw_meta, str) and raw_meta:
            try:
                d["metadata"] = json.loads(raw_meta)
            except (json.JSONDecodeError, TypeError):
                d["metadata"] = None
        elif isinstance(raw_meta, dict):
            d["metadata"] = raw_meta
        out.append(d)
    return out


def _semantic_rank(
    db: RemnantDB,
    config: RemnantConfig,
    query: str,
    *,
    agent_id: str | None = None,
    embedder: Embedder | None = None,
    profile_scope: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Cosine-rank the bounded, authorization-scoped vector corpus."""
    if embedder is None:
        return []
    qvec = embedder.embed(query)
    if not qvec:
        # No query embedding available (remote failure / None). Skip semantic
        # comparison entirely; treat as no match rather than scoring against a
        # zero vector.
        return []

    rows = db.search_all_embeddings(
        agent_id=agent_id,
        profile_scope=profile_scope,
        limit=config.semantic_scan_limit,
    )
    scored: list[dict[str, Any]] = []
    for r in rows:
        vec = r.get("embedding") or []
        if not vec:
            continue
        sim = cosine(qvec, vec)
        d = {
            "id": r["id"],
            "content": r["content"],
            "visibility": r["visibility"],
            "agent_id": r["agent_id"],
            "created_at": r.get("created_at"),
            "updated_at": r.get("updated_at"),
            "score": sim,
        }
        scored.append(d)
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def _rrf_fuse(
    kw: list[dict[str, Any]],
    sem: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion: score = sum(1 / (k + rank)) over ranked lists."""
    scores: dict[str, float] = {}
    meta: dict[str, dict[str, Any]] = {}

    def _rrf_meta(r: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": r["id"],
            "content": r.get("content", ""),
            "visibility": r.get("visibility", "private"),
            "agent_id": r.get("agent_id"),
            "created_at": r.get("created_at"),
            "updated_at": r.get("updated_at"),
        }

    for rank, r in enumerate(kw):
        mid = r["id"]
        scores[mid] = scores.get(mid, 0.0) + 1.0 / (RRF_K + rank + 1)
        if mid not in meta:
            meta[mid] = _rrf_meta(r)

    for rank, r in enumerate(sem):
        mid = r["id"]
        scores[mid] = scores.get(mid, 0.0) + 1.0 / (RRF_K + rank + 1)
        if mid not in meta:
            meta[mid] = _rrf_meta(r)

    fused: list[dict[str, Any]] = []
    for mid, score in scores.items():
        d = dict(meta[mid])
        d["score"] = score
        fused.append(d)
    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused


def _apply_source_weights(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-balance RRF scores so small precise facts compete with long vault notes.

    Multipliers are intentionally mild (not a hard filter) so a directly
    matching vault note can still rank first when it is genuinely the best
    match.
    """
    weights = {
        "conversation": 1.3,
        "manual": 1.3,
        "import": 1.3,
        "hindsight": 1.3,
        "dream": 1.1,
        "cron": 1.1,
        "sensor": 1.1,
        "email": 1.1,
        "vault": 0.7,
    }
    default_weight = 1.0
    out: list[dict[str, Any]] = []
    for r in results:
        d = dict(r)
        base = float(d.get("score", 0.0))
        src = d.get("source") or "unknown"
        weight = weights.get(src, default_weight)
        d["score"] = base * weight
        d["_rrf_base"] = base
        d["_source_weight"] = weight
        out.append(d)
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


__all__ = ["search"]
