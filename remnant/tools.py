"""Tool schemas and dispatch for the Remnant memory provider.

Two tools are exposed to the agent:
- `memory_search`: BM25 keyword search over active memories.
- `memory_store`: manual fact storage with dedup + transient filter.
"""

from __future__ import annotations

import logging
from typing import Any

from .config import RemnantConfig
from .db import RemnantDB
from .embed import Embedder
from .ingest import store_memory
from .search import search

log = logging.getLogger("remnant.tools")

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": (
                "Search the Remnant memory store for durable facts using keyword (BM25) "
                "ranking. Results are scoped to the current agent and visibility."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords or phrase to search for.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 10).",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_store",
            "description": (
                "Manually store a durable fact in Remnant memory. Duplicates and transient "
                "facts (percentages, current status) are automatically filtered out."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "A complete declarative sentence describing a durable fact.",
                    },
                    "entity": {
                        "type": "string",
                        "description": "The subject of the fact (person, project, device).",
                        "default": "general",
                    },
                    "visibility": {
                        "type": "string",
                        "enum": ["private", "shared", "fleet"],
                        "description": "Who can see this memory.",
                        "default": "private",
                    },
                },
                "required": ["fact"],
            },
        },
    },
]


def handle_tool_call(
    tool_name: str,
    args: dict[str, Any],
    *,
    db: RemnantDB,
    config: RemnantConfig,
    embedder: Embedder,
    session_id: str,
    agent_id: str | None = None,
) -> dict[str, Any]:
    """Dispatch a tool call. Returns a tool-result dict for the agent."""
    aid = agent_id or config.agent_id
    if tool_name == "memory_search":
        query = str(args.get("query", "")).strip()
        limit = int(args.get("limit", config.search_limit))
        if not query:
            return {"error": "query is required"}
        results = search(db, config, query, agent_id=aid, limit=limit)
        return {
            "results": [
                {
                    "id": r["id"],
                    "entity": "",
                    "fact": r["content"],
                    "visibility": r["visibility"],
                    "score": round(r.get("score", 0.0), 4),
                }
                for r in results
            ],
            "count": len(results),
        }
    if tool_name == "memory_store":
        fact = str(args.get("fact", "")).strip()
        entity = str(args.get("entity", "general")).strip() or "general"
        visibility = str(args.get("visibility", config.default_visibility)).strip()
        if visibility not in ("private", "shared", "fleet"):
            visibility = config.default_visibility
        if not fact:
            return {"error": "fact is required"}
        canonical = db.resolve_entity(entity, aid)
        mid = store_memory(
            db,
            embedder,
            config,
            fact=fact,
            entity=canonical,
            session_id=session_id,
            agent_id=aid,
            visibility=visibility,
        )
        if mid is None:
            return {"stored": False, "reason": "duplicate or transient fact rejected"}
        return {"stored": True, "memory_id": mid, "entity": canonical, "visibility": visibility}
    return {"error": f"unknown tool: {tool_name}"}


__all__ = ["TOOL_SCHEMAS", "handle_tool_call"]
