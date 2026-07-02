"""Memory search: BM25 keyword, cosine semantic, and RRF hybrid fusion.

Strategies:
- ``keyword`` (default): BM25 over the FTS5 index. The Phase 1 behavior.
- ``semantic``: cosine similarity over stored embeddings. To avoid scanning
  the whole database, embeddings are only loaded for a BM25-pre-filtered
  candidate set (capped by ``SEMANTIC_CANDIDATE_LIMIT``).
- ``auto``: Reciprocal Rank Fusion (RRF, k=60) of BM25 + semantic.
"""

from __future__ import annotations

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


def search(
    db: RemnantDB,
    config: RemnantConfig,
    query: str,
    *,
    agent_id: str | None = None,
    visibility: str | None = None,
    limit: int | None = None,
    strategy: str = "keyword",
    embedder: Embedder | None = None,
) -> list[dict[str, Any]]:
    """Run a memory search with the given strategy and scope filtering.

    - ``keyword``: BM25 only.
    - ``semantic``: cosine over embeddings, BM25-pre-filtered candidates.
    - ``auto``: RRF fusion of keyword + semantic.
    """
    if limit is None:
        limit = config.search_limit
    if strategy not in ("keyword", "semantic", "auto"):
        strategy = "keyword"

    if strategy == "keyword":
        results = db.search_bm25(query, agent_id=agent_id, limit=limit * 3 if visibility else limit)
        results = _scope_filter(results, visibility)
        return results[:limit]

    if strategy == "semantic":
        ranked = _semantic_rank(db, config, query, agent_id=agent_id, embedder=embedder)
        ranked = _scope_filter(ranked, visibility)
        return ranked[:limit]

    # auto: RRF fusion
    kw = db.search_bm25(query, agent_id=agent_id, limit=SEMANTIC_CANDIDATE_LIMIT)
    sem = _semantic_rank(db, config, query, agent_id=agent_id, embedder=embedder)
    fused = _rrf_fuse(kw, sem)
    fused = _scope_filter(fused, visibility)
    return fused[:limit]


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
