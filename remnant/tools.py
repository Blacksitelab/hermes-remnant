"""Tool schemas and dispatch for the Remnant memory provider.

Tools exposed to the agent:
- `memory_search`: keyword (BM25), semantic (cosine), auto (RRF hybrid), or
  graph (pure-SQLite entity traversal) search over active memories.
- `memory_store`: manual fact storage with dedup + transient filter + entity
  graph linking.
- `memory_reflect`: synthesize an answer across the top relevant memories.
- `memory_graph`: traverse the entity graph around a named entity.
- `memory_edit`: update / merge / forget / feedback / share / unshare memories.
  Every edit is audit-logged; nothing is ever deleted.
"""

from __future__ import annotations

import logging
from typing import Any

from .config import RemnantConfig
from .db import RemnantDB
from .edit import memory_edit
from .embed import Embedder
from .graph import graph_traverse
from .ingest import store_memory
from .reflect import memory_reflect
from .search import search

log = logging.getLogger("remnant.tools")

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": (
                "Search the Remnant memory store for durable facts. Supports "
                "keyword (BM25), semantic (cosine over embeddings), auto "
                "(RRF hybrid), and graph (entity-graph traversal) strategies. "
                "Results are scoped to the current agent and visibility."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords or phrase to search for.",
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["keyword", "semantic", "auto", "graph"],
                        "description": "Search strategy (default keyword).",
                        "default": "keyword",
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
                "facts (percentages, current status) are automatically filtered out. "
                "The fact's entity is linked into the entity graph."
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
    {
        "type": "function",
        "function": {
            "name": "memory_reflect",
            "description": (
                "Synthesize an answer across the top relevant memories for a "
                "question using the local reflection LLM. Returns a grounded "
                "synthesis and the source memory ids used."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to reflect on across stored memories.",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_graph",
            "description": (
                "Traverse the entity graph around a named entity and return "
                "connected entities (within N hops) plus the active memories "
                "linked to them. Pure local traversal; no LLM."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "The entity name to traverse from.",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Max hop distance (default 2).",
                        "default": 2,
                    },
                },
                "required": ["entity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_edit",
            "description": (
                "Edit a stored memory. Actions: update (create new version, "
                "supersede old), merge (combine several into one), forget "
                "(hide from search, row preserved), feedback (adjust "
                "trust_score), share / unshare (change visibility). Every "
                "edit is audit-logged; nothing is deleted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "update", "merge", "forget",
                            "feedback", "share", "unshare",
                        ],
                        "description": "The edit action to perform.",
                    },
                    "memory_id": {
                        "type": "string",
                        "description": "Target memory id (for all actions except merge).",
                    },
                    "memory_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Memory ids to merge (merge action only).",
                    },
                    "content": {
                        "type": "string",
                        "description": "New content for update / merge.",
                    },
                    "visibility": {
                        "type": "string",
                        "enum": ["private", "shared", "fleet"],
                        "description": "Visibility for update / merge (defaults to original).",
                    },
                    "feedback": {
                        "type": "string",
                        "enum": ["useful", "wrong"],
                        "description": "Feedback signal for the feedback action.",
                    },
                },
                "required": ["action"],
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
        strategy = str(args.get("strategy", "keyword")).strip() or "keyword"
        if not query:
            return {"error": "query is required"}
        results = search(
            db, config, query, agent_id=aid, limit=limit,
            strategy=strategy, embedder=embedder,
        )
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
        mid = store_memory(
            db,
            embedder,
            config,
            fact=fact,
            entity=entity,
            session_id=session_id,
            agent_id=aid,
            visibility=visibility,
        )
        if mid is None:
            return {"stored": False, "reason": "duplicate or transient fact rejected"}
        return {"stored": True, "memory_id": mid, "entity": entity, "visibility": visibility}
    if tool_name == "memory_reflect":
        question = str(args.get("question", "")).strip()
        if not question:
            return {"error": "question is required"}
        return memory_reflect(question, db, config, embedder, aid)
    if tool_name == "memory_graph":
        entity = str(args.get("entity", "")).strip()
        depth = int(args.get("depth", 2))
        if not entity:
            return {"error": "entity is required"}
        res = graph_traverse(db, entity, agent_id=aid, depth=depth)
        return {
            "entity": res["entity"],
            "entities": [
                {
                    "id": e["id"],
                    "name": e.get("name"),
                    "type": e.get("type"),
                    "depth": e.get("depth"),
                }
                for e in res["entities"]
            ],
            "memories": [
                {
                    "id": m["id"],
                    "content": m["content"],
                    "visibility": m["visibility"],
                }
                for m in res["memories"]
            ],
            "count": len(res["memories"]),
        }
    if tool_name == "memory_edit":
        action = str(args.get("action", "")).strip()
        if not action:
            return {"error": "action is required"}
        return memory_edit(
            db,
            config,
            embedder,
            action=action,
            actor=aid,
            memory_id=args.get("memory_id"),
            memory_ids=args.get("memory_ids"),
            content=args.get("content"),
            visibility=args.get("visibility"),
            feedback=args.get("feedback"),
            agent_id=aid,
            session_id=session_id,
        )
    return {"error": f"unknown tool: {tool_name}"}


__all__ = ["TOOL_SCHEMAS", "handle_tool_call"]
