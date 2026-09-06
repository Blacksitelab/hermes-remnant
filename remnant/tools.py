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
- `memory_import`: bulk-import a memory source. Phase 4 supports
  ``source='vault'`` (Obsidian vault re-index). Phase 6 adds
  ``source='memory_store'`` (the current profile's MEMORY.md / USER.md) and
  ``source='hindsight'`` (bounded broad-query recall from the Hindsight store).
  Both new sources support ``dry_run`` and ``shadow`` modes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import RemnantConfig
from .db import RemnantDB
from .edit import memory_edit
from .embed import Embedder
from .graph import graph_traverse
from .import_sources import import_hindsight, import_memory_store
from .ingest import store_memory
from .recall import RecallRequest, RecallService
from .reflect import memory_reflect
from .threads import (
    create_thread,
    resolve_thread,
    update_thread,
)
from .vault import index_vault

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
                        "description": (
                            "Search strategy (default auto): keyword (BM25), "
                            "semantic (cosine over embeddings), auto (RRF "
                            "hybrid fusion), or graph (entity-graph traversal)."
                        ),
                        "default": "auto",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 10).",
                        "default": 10,
                    },
                    "profile_scope": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional list of allowed vault path prefixes. When "
                            "set, document memories (source='vault') are only "
                            "returned if their source_id starts with one of these "
                            "prefixes. Ignored for non-document memories. "
                            "Defaults to the provider's configured profile_scope."
                        ),
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
                        "description": (
                            "Legacy label within this profile; never grants cross-profile access."
                        ),
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
    {
        "type": "function",
        "function": {
            "name": "memory_import",
            "description": (
                "Import memories from an existing store into Remnant. "
                "source='vault' re-indexes the Obsidian vault (new/changed "
                "notes become document memories; deleted notes are forgotten). "
                "source='memory_store' parses the current profile's MEMORY.md / USER.md "
                "into facts with "
                "visibility heuristics (fleet/shared/private). "
                "source='hindsight' issues a bounded set of broad recall "
                "queries to the Hindsight store and dedups by content hash. "
                "Use dry_run to preview counts without writing. Use shadow=True "
                "to log what would be imported to ~/.hermes/remnant/shadow.log "
                "instead of touching the DB."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["memory_store", "hindsight", "vault"],
                        "description": "The memory source to import.",
                    },
                    "profile": {
                        "type": "string",
                        "description": (
                            "memory_store only: limit import to a single "
                            "profile name. Ignored for other sources."
                        ),
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": (
                            "Preview counts without writing to the DB or "
                            "shadow log (default false)."
                        ),
                        "default": False,
                    },
                    "shadow": {
                        "type": "boolean",
                        "description": (
                            "Log proposed actions to ~/.hermes/remnant/shadow.log "
                            "instead of importing (default false)."
                        ),
                        "default": False,
                    },
                    "force": {
                        "type": "boolean",
                        "description": (
                            "vault only: re-index every non-excluded file even "
                            "when its hash is unchanged (default false)."
                        ),
                        "default": False,
                    },
                },
                "required": ["source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_thread",
            "description": (
                "Manage topic threads. Actions: create (start a new thread), "
                "update (edit title/status/importance), resolve (mark a thread "
                "done), list (list threads, optionally filtered by status), "
                "stale (sweep threads inactive for 14 days to status='stale'). "
                "Threads are never deleted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "update", "resolve", "list", "stale"],
                        "description": "The thread action to perform.",
                    },
                    "thread_id": {
                        "type": "string",
                        "description": "Target thread id (update/resolve).",
                    },
                    "title": {
                        "type": "string",
                        "description": "Thread title (create) or new title (update).",
                    },
                    "topic": {
                        "type": "string",
                        "description": "Short topic key (create).",
                    },
                    "importance": {
                        "type": "number",
                        "description": "Importance 0.0-1.0 (create/update).",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["active", "stale", "resolved"],
                        "description": "New status (update).",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for the thread (create/update).",
                    },
                    "related_entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Entity ids/names linked to the thread.",
                    },
                    "status_filter": {
                        "type": "string",
                        "enum": ["active", "stale", "resolved"],
                        "description": "Filter for the list action.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max threads to return (list, default 50).",
                        "default": 50,
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
    hermes_home: str | None = None,
    echo: Any | None = None,
) -> dict[str, Any]:
    """Dispatch a tool call. Returns a tool-result dict for the agent."""
    aid = config.agent_id
    if tool_name == "memory_search":
        query = str(args.get("query", "")).strip()
        try:
            limit = max(1, min(int(args.get("limit", config.search_limit)), 100))
        except (TypeError, ValueError):
            limit = max(1, min(int(config.search_limit), 100))
        strategy = str(args.get("strategy") or config.default_search_strategy).strip()
        if strategy not in ("keyword", "semantic", "auto", "graph"):
            strategy = config.default_search_strategy
        # profile_scope may narrow the provider-configured scope, but search()
        # caps it so an explicit empty list cannot disable configured policy.
        raw_scope = args.get("profile_scope")
        profile_scope: list[str] | None
        if raw_scope is None:
            profile_scope = None
        elif isinstance(raw_scope, list):
            profile_scope = [str(p) for p in raw_scope if str(p).strip()]
        else:
            profile_scope = None
        if not query:
            return {"error": "query is required"}
        response = RecallService(db, config).recall(
            RecallRequest(
                query=query,
                agent_id=aid,
                session_id=session_id,
                strategy=strategy,
                limit=limit,
                profile_scope=profile_scope,
                include_pending=bool(getattr(config, "recent_turn_overlay_enabled", False)),
                echo_service=echo,
                echo_viewer_key=(getattr(echo, "viewer_key", None) if echo else aid),
            ),
            embedder=embedder,
        )
        results = response.results
        return {
            "results": [
                {
                    "id": r["id"],
                    "entity": "",
                    "fact": r["content"],
                    "visibility": r["visibility"],
                    "source": r.get("source"),
                    "source_id": r.get("source_id"),
                    "locked": r.get("locked", False),
                    "score": round(r.get("score", 0.0), 4),
                    "ranking": r.get("ranking"),
                    "claim_status": r.get("claim_status"),
                    **({
                        "claim": {
                            "subject": r["claim"]["subject"],
                            "predicate": r["claim"]["predicate"],
                            "object": r["claim"]["object"],
                            "status": r["claim"]["status"],
                            "valid_from": r["claim"].get("valid_from"),
                            "valid_to": r["claim"].get("valid_to"),
                        }
                    } if r.get("claim") else {}),
                    **({"claim_group": r["claim_group"]} if r.get("claim_group") else {}),
                }
                for r in results
            ],
            "count": len(results),
            "diagnostics": response.diagnostics,
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
        return memory_reflect(question, db, config, embedder, aid, session_id)
    if tool_name == "memory_graph":
        entity = str(args.get("entity", "")).strip()
        depth = int(args.get("depth", 2))
        if not entity:
            return {"error": "entity is required"}
        res = graph_traverse(
            db,
            entity,
            agent_id=aid,
            depth=depth,
            evidence_only=bool(getattr(config, "relation_evidence_enabled", False)),
        )
        graph_response = RecallService(db, config).recall(
            RecallRequest(
                query=entity,
                agent_id=aid,
                strategy="graph",
                limit=100,
            ),
            candidates=res["memories"],
        )
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
                    "claim_status": m.get("claim_status"),
                    "ranking": m.get("ranking"),
                }
                for m in graph_response.results
            ],
            "count": len(graph_response.results),
            "diagnostics": graph_response.diagnostics,
        }
    if tool_name == "memory_edit":
        action = str(args.get("action", "")).strip()
        if not action:
            return {"error": "action is required"}
        result = memory_edit(
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
        if (not result.get("error") and echo is not None
            and action == "feedback" and args.get("memory_id")):
            try:
                echo.record_feedback(
                    memory_id=str(args["memory_id"]),
                    feedback=str(args.get("feedback") or ""),
                    agent_id=aid,
                    viewer_key=getattr(echo, "viewer_key", aid),
                    query=str(args.get("query") or "") or None,
                )
                echo.aggregate(limit=20)
            except Exception:
                pass
        return result
    if tool_name == "memory_import":
        source = str(args.get("source", "")).strip().lower()
        if source not in ("vault", "hindsight", "memory_store"):
            return {"error": f"unknown import source: {source}"}
        dry_run = bool(args.get("dry_run", False))
        shadow = bool(args.get("shadow", False))
        profile = args.get("profile")
        if profile is not None:
            profile = str(profile).strip() or None
        if profile is not None and profile != aid:
            return {"error": "imports are restricted to the current profile"}
        profile = aid
        if source == "vault":
            force = bool(args.get("force", False))
            stats = index_vault(db, config, embedder, force=force)
            return {
                "source": source,
                "indexed": stats["indexed"],
                "skipped": stats["skipped"],
                "forgotten": stats["forgotten"],
            }
        # memory_store / hindsight need a hermes_home to discover profiles and
        # to write the shadow log. The provider passes it through; fall back to
        # the standard location for non-provider callers.
        home = hermes_home or str(Path.home() / ".hermes")
        if source == "memory_store":
            stats = import_memory_store(
                db, config, embedder, home,
                dry_run=dry_run, shadow=shadow, profile=profile,
            )
        else:  # hindsight
            stats = import_hindsight(
                db, config, embedder,
                dry_run=dry_run, shadow=shadow, hermes_home=home,
            )
        return {"source": source, "stats": stats}
    if tool_name == "memory_thread":
        action = str(args.get("action", "")).strip().lower()
        if not action:
            return {"error": "action is required"}
        if action in {"update", "resolve"}:
            thread = db.get_thread(str(args.get("thread_id") or ""))
            if thread is None or thread.get("added_by") != aid:
                return {"error": "thread not found"}
        if action == "create":
            title = str(args.get("title", "")).strip()
            topic = str(args.get("topic", "")).strip()
            if not title or not topic:
                return {"error": "title and topic are required for create"}
            try:
                importance = float(args.get("importance", 0.5))
            except (TypeError, ValueError):
                importance = 0.5
            tags = args.get("tags")
            related = args.get("related_entities")
            tid = create_thread(
                db,
                title=title,
                topic=topic,
                importance=importance,
                tags=[str(t) for t in tags] if isinstance(tags, list) else None,
                related_entities=(
                    [str(e) for e in related] if isinstance(related, list) else None
                ),
                source="agent",
                added_by=aid,
            )
            return {"thread_id": tid, "title": title, "topic": topic}
        if action == "update":
            tid = str(args.get("thread_id", "")).strip()
            if not tid:
                return {"error": "thread_id is required for update"}
            try:
                importance = (
                    float(args.get("importance")) if args.get("importance") is not None
                    else None
                )
            except (TypeError, ValueError):
                importance = None
            tags = args.get("tags")
            related = args.get("related_entities")
            res = update_thread(
                db,
                tid,
                title=args.get("title"),
                status=args.get("status"),
                importance=importance,
                tags=([str(t) for t in tags] if isinstance(tags, list) else None),
                related_entities=(
                    [str(e) for e in related] if isinstance(related, list) else None
                ),
            )
            if res is None:
                return {"error": f"thread not found: {tid}"}
            return {"thread": res}
        if action == "resolve":
            tid = str(args.get("thread_id", "")).strip()
            if not tid:
                return {"error": "thread_id is required for resolve"}
            res = resolve_thread(db, tid)
            if res is None:
                return {"error": f"thread not found: {tid}"}
            return {"thread": res}
        if action == "list":
            status_filter = args.get("status_filter")
            try:
                limit = int(args.get("limit", 50))
            except (TypeError, ValueError):
                limit = 50
            threads = db.list_threads(status=status_filter, limit=limit, agent_id=aid)
            return {"threads": threads, "count": len(threads)}
        if action == "stale":
            marked = db.sweep_stale_threads(agent_id=aid)
            return {"marked_stale": marked, "count": len(marked)}
        return {"error": f"unknown thread action: {action}"}
    return {"error": f"unknown tool: {tool_name}"}


__all__ = ["TOOL_SCHEMAS", "handle_tool_call"]
