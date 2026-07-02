"""BM25 keyword search over active memories with visibility + agent filtering."""

from __future__ import annotations

from typing import Any

from .config import RemnantConfig
from .db import RemnantDB

# Visibility precedence: private < shared < fleet. A search scoped to a higher
# tier can see lower tiers; a lower-tier search cannot see higher tiers.
_VIS_ORDER = {"private": 0, "shared": 1, "fleet": 2}


def search(
    db: RemnantDB,
    config: RemnantConfig,
    query: str,
    *,
    agent_id: str | None = None,
    visibility: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Run BM25 keyword search and filter results by visibility scope.

    - When `agent_id` is given, only that agent's memories are returned.
    - When `visibility` is given, only memories at or below that visibility
      tier are returned (e.g. a `private` search sees only private; `fleet`
      sees private+shared+fleet).
    """
    if limit is None:
        limit = config.search_limit

    results = db.search_bm25(query, agent_id=agent_id, limit=limit * 3 if visibility else limit)
    if visibility and _VIS_ORDER.get(visibility) is not None:
        cap = _VIS_ORDER[visibility]
        results = [r for r in results if _VIS_ORDER.get(r.get("visibility", "private"), 0) <= cap]
    return results[:limit]


__all__ = ["search"]
