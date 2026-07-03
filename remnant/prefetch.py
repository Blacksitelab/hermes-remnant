"""Proactive memory injection: intent classification, query expansion, hybrid
retrieval, dedup against the conversation, token-budget enforcement, and
diff-based suppression of unchanged context.

All classification and candidate filtering use only local, cheap operations
(regex/BM25). The only network call is the single query embedding via the
Embedder, which is cached per session by the provider.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Any

from .config import RemnantConfig
from .db import RemnantDB
from .embed import Embedder
from .search import search as hybrid_search

# Intent classifier: keywords that signal the user is asking about something
# durable worth recalling from memory. Kept deliberately small + local.
_NEEDS_MEMORY_RE = re.compile(
    r"\b(remember|recall|did we|do we|what did|what do|decide|decided|"
    r"status of|last time|previously|before|earlier|who is|what is|where is|"
    r"how many|which|configure|setup|preference|prefers|policy|policies)\b",
    re.IGNORECASE,
)

# Greetings / small talk / filler that never needs memory. If the query is
# dominated by these and contains no needs-memory signal, skip injection.
_GREETING_RE = re.compile(
    r"^(hey|hi|hello|yo|sup|howdy|thanks|thank you|ok|okay|sure|yes|no|"
    r"cool|nice|got it|bye|good night|how are you|how's it going|"
    r"what's up|whats up)\b[!?.]*$",
    re.IGNORECASE,
)

# Proper nouns / capitalized multi-word phrases for query expansion.
_PROPER_RE = re.compile(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\b")
# Stopwords stripped when deriving expansion terms.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "of", "to", "in", "on",
    "for", "and", "or", "what", "who", "where", "how", "why", "did", "do", "does",
    "i", "you", "we", "they", "my", "your", "our", "me", "us",
}


def _needs_memory(query: str) -> bool:
    """Lightweight local intent classifier. No network, no LLM."""
    q = (query or "").strip()
    if not q:
        logging.debug("Remnant prefetch skip: empty query")
        return False
    if _NEEDS_MEMORY_RE.search(q):
        return True
    if _GREETING_RE.match(q):
        logging.debug("Remnant prefetch skip: greeting/small-talk: %r", q)
        return False
    logging.debug("Remnant prefetch pass: %r", q)
    return True


def _expand_queries(query: str) -> list[str]:
    """Derive 2-3 related search terms from the query. Purely local."""
    q = (query or "").strip()
    if not q:
        return []
    terms: list[str] = []
    # 1. Proper nouns as-is (entity lookups).
    for m in _PROPER_RE.findall(q):
        term = m.strip()
        if term and term.lower() not in _STOPWORDS and term not in terms:
            terms.append(term)
    # 2. Content tokens (lowercased, stopwords removed).
    content = [t for t in re.split(r"[\s,.;:!?]+", q) if t and t.lower() not in _STOPWORDS]
    if content:
        phrase = " ".join(content)
        if phrase and phrase not in terms:
            terms.append(phrase)
    # 3. Cap at 3 expansions.
    return terms[:3]


def _approx_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token). No tokenizer dependency."""
    return max(1, len(text or "") // 4)


def _format_context(memories: list[dict[str, Any]]) -> str:
    """Compact one-line-per-memory context block."""
    lines = ["# Recalled memory (Remnant)"]
    for m in memories:
        vis = m.get("visibility", "private")
        line = f"- [{vis}] {m.get('content', '').strip()}"
        lines.append(line)
    return "\n".join(lines)


def _dedup_against_messages(
    memories: list[dict[str, Any]], messages: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Drop memories whose text already appears in the current conversation."""
    if not messages:
        return memories
    seen: set[str] = set()
    for msg in messages:
        content = (msg.get("content") or "") if isinstance(msg, dict) else str(msg)
        seen.add(_normalize(content))
    out: list[dict[str, Any]] = []
    for m in memories:
        if _normalize(m.get("content", "")) not in seen:
            out.append(m)
    return out


def _normalize(s: str) -> str:
    s = (s or "").lower()
    s = s.strip().strip(".,!?;:")
    return re.sub(r"\s+", " ", s).strip()


def prefetch(
    provider: Any,
    query: str,
    session_id: str,
    messages: list[dict[str, Any]] | None = None,
    deadline_ms: int | None = None,
) -> dict[str, Any]:
    """Run hybrid retrieval, dedup, budget, and diff-based suppression.

    Returns ``{}`` when memory isn't needed, the deadline is exceeded, the
    token budget would be blown, or the injected context is unchanged since
    the last call in this session. Otherwise returns::

        {"context": <compact block str>, "memories": [...], "token_estimate": int,
         "hash": <sha256 hex>, "session_id": <sid>}
    """
    cfg: RemnantConfig = provider._config
    db: RemnantDB = provider._db
    embedder: Embedder = provider._embedder

    if not cfg.prefetch_enabled:
        return {}
    if not _needs_memory(query):
        return {}
    if deadline_ms is None:
        deadline_ms = cfg.injection_prefetch_deadline_ms
    deadline_s = deadline_ms / 1000.0
    t0 = time.monotonic()

    # Single network call for the query embedding; cached per session by the
    # provider. We pass the session-scoped embedder wrapper if present.
    session_embedder = provider._session_embedder(session_id, query) or embedder

    agent_id = cfg.agent_id
    limit = cfg.search_limit

    # Gather candidate memories across expanded queries via hybrid (auto) search.
    seen_ids: set[str] = set()
    merged: list[dict[str, Any]] = []
    expansions = _expand_queries(query) or [query]
    for term in expansions:
        if time.monotonic() - t0 > deadline_s:
            return {}
        try:
            results = hybrid_search(
                db, cfg, term,
                agent_id=agent_id, limit=limit, strategy="auto",
                embedder=session_embedder,
            )
        except Exception:
            results = []
        for r in results:
            mid = r.get("id")
            if mid and mid not in seen_ids:
                seen_ids.add(mid)
                merged.append(r)

    if not merged:
        return {}

    # Dedup against current conversation messages.
    merged = _dedup_against_messages(merged, messages)
    if not merged:
        return {}

    # Token budget enforcement: greedy add until budget exhausted.
    budget = cfg.injection_token_budget
    selected: list[dict[str, Any]] = []
    running = _approx_tokens("# Recalled memory (Remnant)")
    for m in merged:
        line_tokens = _approx_tokens(f"- [{m.get('visibility','private')}] {m.get('content','')}")
        if running + line_tokens > budget:
            break
        selected.append(m)
        running += line_tokens
    if not selected:
        return {}

    context = _format_context(selected)
    if _approx_tokens(context) > budget:
        return {}

    if time.monotonic() - t0 > deadline_s:
        return {}

    ctx_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()
    # Diff-based suppression: same context as last turn in this session.
    if provider._last_injected_hash.get(session_id) == ctx_hash:
        return {}
    provider._last_injected_hash[session_id] = ctx_hash

    return {
        "context": context,
        "memories": [
            {"id": m.get("id"), "content": m.get("content", ""), "visibility": m.get("visibility")}
            for m in selected
        ],
        "token_estimate": _approx_tokens(context),
        "hash": ctx_hash,
        "session_id": session_id,
    }


__all__ = ["prefetch", "_needs_memory", "_expand_queries"]
