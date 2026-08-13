"""Pure-SQLite graph traversal helpers.

No LLM, no embeddings. All traversal is bounded BFS over the `relations`
table followed by a `memory_entities` join. These wrap `RemnantDB` methods so
callers (search, tools) get a single clean entry point.
"""

from __future__ import annotations

from typing import Any

from .db import RemnantDB


def graph_search(
    db: RemnantDB,
    query: str,
    *,
    agent_id: str | None = None,
    depth: int = 2,
    limit: int = 20,
    profile_scope: list[str] | None = None,
    evidence_only: bool = False,
) -> list[dict[str, Any]]:
    """Extract entity names from `query`, resolve them, traverse the graph,
    and return linked active memories. Pure SQLite.

    Falls back to proper-noun extraction when no direct entity name match is
    found, so a fresh query that mentions a known entity by display name still
    resolves via the alias index.
    """
    from .entity import extract_entities

    names = [e["name"] for e in extract_entities(query)]
    if not names:
        # Fall back to whitespace tokens as candidate entity names.
        names = [t for t in (query or "").split() if t]
    if not names:
        return []
    return db.search_graph(
        names,
        agent_id=agent_id,
        depth=depth,
        limit=limit,
        profile_scope=profile_scope,
        evidence_only=evidence_only,
    )


def graph_traverse(
    db: RemnantDB,
    entity_name: str,
    *,
    agent_id: str | None = None,
    depth: int = 2,
    profile_scope: list[str] | None = None,
    evidence_only: bool = False,
) -> dict[str, Any]:
    """Resolve `entity_name` to its canonical id and traverse the graph.

    Returns ``{"entity": {...}|None, "entities": [...], "memories": [...]}``.
    ``entity`` is the seed (or None if unresolvable); ``entities`` are visited
    nodes (including the seed at depth 0); ``memories`` are deduped active
    memories linked to any visited entity.
    """
    if not entity_name:
        return {"entity": None, "entities": [], "memories": []}
    eid = db.find_entity_by_name(entity_name, agent_id=agent_id)
    if not eid:
        return {"entity": None, "entities": [], "memories": []}
    result = db.traverse_graph(
        eid,
        depth=depth,
        agent_id=agent_id,
        profile_scope=profile_scope,
        evidence_only=evidence_only,
    )
    seed = db.get_entity(eid)
    return {
        "entity": seed,
        "entities": result["entities"],
        "memories": result["memories"],
    }


__all__ = ["graph_search", "graph_traverse"]
