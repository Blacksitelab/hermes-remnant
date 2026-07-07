"""Memory search: BM25 keyword, cosine semantic, RRF hybrid fusion, and graph.

Strategies:
- ``keyword``: BM25 over the FTS5 index. The Phase 1 behavior.
- ``semantic``: cosine similarity over stored embeddings. To avoid scanning
  the whole database, embeddings are only loaded for a BM25-pre-filtered
  candidate set (capped by ``SEMANTIC_CANDIDATE_LIMIT``).
- ``auto`` (default): Reciprocal Rank Fusion (RRF, k=60) of BM25 + semantic.
  Results below ``min_semantic_score`` (top semantic score) are dropped to
  avoid returning keyword-dominated noise for semantically unrelated queries.
- ``graph``: pure-SQLite entity-graph traversal. Extracts entity names from
  the query, resolves them, BFS over `relations` up to N hops, and returns
  linked active memories. No LLM, no embeddings.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .config import RRF_K, SEMANTIC_CANDIDATE_LIMIT, RemnantConfig
from .db import RemnantDB
from .embed import Embedder, cosine

# Visibility precedence: private < shared < fleet. A search scoped to a higher
# tier can see lower tiers; a lower-tier search cannot see higher tiers.
_VIS_ORDER = {"private": 0, "shared": 1, "fleet": 2}


def _scope_filter(results: list[dict[str, Any]], visibility: str | None) -> list[dict[str, Any]]:
    if visibility and _VIS_ORDER.get(visibility) is not None:
        cap = _VIS_ORDER[visibility]
        results = [r for r in results if _VIS_ORDER.get(r.get("visibility", "private"), 0) <= cap]
    return results


def _profile_scope_filter(
    results: list[dict[str, Any]], profile_scope: list[str] | None
) -> list[dict[str, Any]]:
    """Restrict document memories (source='vault') to allowed path prefixes.

    Non-document memories are never filtered by profile_scope. When
    `profile_scope` is None or empty, no additional filtering is applied.
    A document memory is included if its ``source_id`` starts with one of the
    allowed prefixes (after normalizing separators to ``/``).
    """
    if not profile_scope:
        return results
    prefixes = [p.strip().rstrip("/") for p in profile_scope if p and p.strip()]
    if not prefixes:
        return results
    out: list[dict[str, Any]] = []
    for r in results:
        # Only document/vault memories are scope-restricted. The `source` may
        # not be attached to every result dict (older search paths don't carry
        # it); when absent we treat the row as non-document and let it through.
        src = r.get("source")
        if src != "vault":
            out.append(r)
            continue
        sid = r.get("source_id") or ""
        sid = sid.replace("\\", "/")
        if any(sid == p or sid.startswith(p + "/") or sid.startswith(p) for p in prefixes):
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
    those whose ``source_id`` starts with one of the allowed prefixes. When
    None/empty, no additional filtering is applied. Locked vault notes have
    their content masked for any viewer that is not the memory's owner agent.

    For ``semantic`` and ``auto``, if the top semantic cosine score is below
    ``config.min_semantic_score`` (default 0.5), an empty list is returned
    rather than keyword-dominated noise.
    """
    if limit is None:
        limit = config.search_limit
    if strategy is None:
        strategy = config.default_search_strategy
    if strategy not in ("keyword", "semantic", "auto", "graph"):
        strategy = config.default_search_strategy

    # Resolve the effective scope: explicit arg overrides config.profile_scope.
    scope = profile_scope if profile_scope is not None else (config.profile_scope or [])
    # The "viewer" for locked-note masking is the agent performing the search:
    # an explicit agent_id wins, else the provider-configured agent_id. A None
    # viewer is treated as a non-owner anonymous search (locked content masked).
    viewer = agent_id if agent_id is not None else config.agent_id

    if strategy == "graph":
        from .graph import graph_search

        results = graph_search(db, query, agent_id=agent_id, limit=limit)
        results = _attach_source(db, results)
        results = _profile_scope_filter(results, scope)
        results = _scope_filter(results, visibility)
        results = _mask_locked(results, viewer_agent=viewer)
        results = results[:limit]
        _reinforce(db, config, results, query=query, strategy=strategy)
        return results

    if strategy == "keyword":
        results = db.search_bm25(
            query, agent_id=agent_id, limit=limit * 3 if (visibility or scope) else limit
        )
        results = _attach_source(db, results)
        results = _profile_scope_filter(results, scope)
        results = _scope_filter(results, visibility)
        results = _mask_locked(results, viewer_agent=viewer)
        results = results[:limit]
        _reinforce(db, config, results, query=query, strategy=strategy)
        return results

    if strategy == "semantic":
        ranked = _semantic_rank(db, config, query, agent_id=agent_id, embedder=embedder)
        # Drop noise: if the top semantic cosine score is below the configured
        # minimum, there are no strong matches.
        if ranked and ranked[0].get("score", 0.0) < config.min_semantic_score:
            return []
        ranked = _attach_source(db, ranked)
        ranked = _profile_scope_filter(ranked, scope)
        ranked = _scope_filter(ranked, visibility)
        ranked = _mask_locked(ranked, viewer_agent=viewer)
        ranked = ranked[:limit]
        _reinforce(db, config, ranked, query=query, strategy=strategy)
        return ranked

    # auto: RRF fusion
    kw = db.search_bm25(query, agent_id=agent_id, limit=SEMANTIC_CANDIDATE_LIMIT)
    sem = _semantic_rank(db, config, query, agent_id=agent_id, embedder=embedder)
    # Drop noise: if the top semantic cosine score is below the configured
    # minimum, the query is not semantically related to any memory. Return an
    # empty list instead of keyword-dominated BM25 noise.
    if sem and sem[0].get("score", 0.0) < config.min_semantic_score:
        return []
    fused = _rrf_fuse(kw, sem)
    fused = _attach_source(db, fused)
    fused = _profile_scope_filter(fused, scope)
    fused = _scope_filter(fused, visibility)
    fused = _mask_locked(fused, viewer_agent=viewer)
    fused = fused[:limit]
    _reinforce(db, config, fused, query=query, strategy=strategy)
    return fused


def _reinforce(
    db: RemnantDB,
    config: RemnantConfig,
    results: list[dict[str, Any]],
    *,
    query: str,
    strategy: str,
) -> None:
    """Retrieval reinforcement (issue #11 + #16).

    For each distinct memory id in ``results``:
      1. Decay the current trust_score by age since last update.
      2. Bump seen_count.
      3. Add +0.02 trust_score (capped at 0.95).

    Time decay uses a configurable half-life; scores cannot fall below
    ``config.trust_decay_floor``. Decay is skipped when disabled.
    """
    seen: set[str] = set()
    now = _utc_now()
    for r in results:
        mid = r.get("id")
        if not mid or mid in seen:
            continue
        seen.add(mid)
        mem = db.get_memory(mid)
        if mem is None:
            continue
        updated_at = mem.get("updated_at") or mem.get("created_at")
        current = float(mem.get("trust_score") or 0.5)
        decayed = _apply_decay(current, updated_at, now, config)
        db.increment_seen_count(mid)
        new_score = min(decayed + 0.02, 0.95)
        db.set_memory_field(
            mid,
            "trust_score",
            new_score,
            actor="system",
            action="trust_reinforce",
            details={"query": query, "strategy": strategy, "decayed_from": current},
        )


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
            f"SELECT id, agent, source, source_id, metadata, tags FROM memories "
            f"WHERE id IN ({placeholders})",
            ids,
        )
        rows = {r["id"]: dict(r) for r in cur.fetchall()}
    import json

    out: list[dict[str, Any]] = []
    for r in results:
        d = dict(r)
        meta = rows.get(d.get("id"), {})
        d.setdefault("agent", meta.get("agent"))
        d.setdefault("agent_id", meta.get("agent"))
        d.setdefault("source", meta.get("source"))
        d.setdefault("source_id", meta.get("source_id"))
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
) -> list[dict[str, Any]]:
    """Cosine-rank memories over a BM25-pre-filtered candidate set.

    Pre-filter with BM25 (cap ~100), load embeddings only for those ids, then
    compute cosine against the query embedding. Falls back to recency-ordered
    active memories if BM25 yields nothing (so semantic still works for terms
    not in the FTS index but present as embeddings).
    """
    if embedder is None:
        return []
    qvec = embedder.embed(query)
    if not qvec:
        # No query embedding available (remote failure / None). Skip semantic
        # comparison entirely; treat as no match rather than scoring against a
        # zero vector.
        return []

    candidates = _bm25_candidates(db, query, agent_id=agent_id, limit=SEMANTIC_CANDIDATE_LIMIT)
    if not candidates:
        # Fall back to recent active memories (still bounded) so semantic search
        # works even when the query terms are not in the FTS index.
        recent = db.search_all_active(agent_id=agent_id, limit=SEMANTIC_CANDIDATE_LIMIT)
        candidates = [{"id": r["id"]} for r in recent]
    if not candidates:
        return []

    ids = [c["id"] for c in candidates]
    rows = db.search_by_embedding(ids, agent_id=agent_id)
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


def _bm25_candidates(
    db: RemnantDB, query: str, *, agent_id: str | None = None, limit: int
) -> list[dict[str, Any]]:
    """Pre-filter candidates via BM25 before computing cosine. Cheap + local."""
    rows = db.search_bm25(query, agent_id=agent_id, limit=limit)
    return [{"id": r["id"]} for r in rows]


def _rrf_fuse(
    kw: list[dict[str, Any]], sem: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion: score = sum(1 / (k + rank)) over ranked lists."""
    scores: dict[str, float] = {}
    meta: dict[str, dict[str, Any]] = {}

    for rank, r in enumerate(kw):
        mid = r["id"]
        scores[mid] = scores.get(mid, 0.0) + 1.0 / (RRF_K + rank + 1)
        if mid not in meta:
            meta[mid] = {
                "id": mid,
                "content": r.get("content", ""),
                "visibility": r.get("visibility", "private"),
                "agent_id": r.get("agent_id"),
                "created_at": r.get("created_at"),
                "updated_at": r.get("updated_at"),
            }

    for rank, r in enumerate(sem):
        mid = r["id"]
        scores[mid] = scores.get(mid, 0.0) + 1.0 / (RRF_K + rank + 1)
        if mid not in meta:
            meta[mid] = {
                "id": mid,
                "content": r.get("content", ""),
                "visibility": r.get("visibility", "private"),
                "agent_id": r.get("agent_id"),
                "created_at": r.get("created_at"),
                "updated_at": r.get("updated_at"),
            }

    fused: list[dict[str, Any]] = []
    for mid, score in scores.items():
        d = dict(meta[mid])
        d["score"] = score
        fused.append(d)
    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused


__all__ = ["search"]
