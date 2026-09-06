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
from .context import compile_context_details
from .db import RemnantDB
from .recall import RecallRequest, RecallService
from .search import search as hybrid_search

log = logging.getLogger("remnant.prefetch")

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


# -- graph-based query expansion (Issue #28) ---------------------------------
#
# The core problem: when a user says "the printer", _expand_queries strips
# "the" as a stopword and produces ["printer"]. BM25/semantic search for
# "printer" is weak — it doesn't find the memory about the "Elegoo Centauri
# Carbon V1" because the entity name doesn't contain the word "printer".
#
# But the entity graph already knows: alias "the printer" → entity "elegoo
# centauri carbon v1". This function resolves entity mentions (including
# multi-word phrases with stopwords like "the printer") against the entity
# graph, traverses 1 hop to find related entity names, and returns the
# canonical names as additional search terms. Pure SQLite, <10ms.

# Phrases to try as entity lookups. We generate n-grams (1-3 words) from the
# query, preserving stopwords so "the printer" survives. Dedup + cap.
def _entity_lookup_phrases(query: str, max_phrases: int = 8) -> list[str]:
    """Generate candidate entity names from the query, preserving stopwords.

    Unlike _expand_queries which strips stopwords, this keeps phrases like
    "the printer" intact so they can match aliases in the entity graph.
    """
    q = (query or "").strip()
    if not q:
        return []
    tokens = re.split(r"[\s,.;:!?]+", q)
    tokens = [t for t in tokens if t]
    if not tokens:
        return []
    phrases: list[str] = []
    seen: set[str] = set()
    # Unigrams + bigrams + trigrams.
    for n in (3, 2, 1):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i : i + n]).strip()
            key = phrase.lower()
            if key and key not in seen:
                seen.add(key)
                phrases.append(phrase)
            if len(phrases) >= max_phrases:
                break
        if len(phrases) >= max_phrases:
            break
    return phrases


def _graph_expand(
    db: Any,
    query: str,
    agent_id: str | None = None,
    max_terms: int = 5,
    evidence_only: bool = False,
) -> list[str]:
    """Resolve entity mentions in the query against the graph and return
    canonical entity names (resolved + 1-hop neighbours) as search terms.

    Pure SQLite — no embeddings, no LLM. Returns [] if nothing resolves.
    """
    phrases = _entity_lookup_phrases(query)
    if not phrases:
        return []
    seed_ids: list[str] = []
    resolved_names: list[str] = []
    seen_names: set[str] = set()
    for phrase in phrases:
        eid = db.find_entity_by_name(phrase, agent_id=agent_id)
        if eid:
            name = db.entity_name_for(eid)
            if name and name.lower() not in seen_names:
                seen_names.add(name.lower())
                resolved_names.append(name)
            if eid not in seed_ids:
                seed_ids.append(eid)
    if not seed_ids:
        return []
    # 1-hop traversal to find related entity names.
    for eid in seed_ids:
        result = db.traverse_graph(
            eid, depth=1, agent_id=agent_id, evidence_only=evidence_only
        )
        for ent in result.get("entities", []):
            name = ent.get("name")
            if name and name.lower() not in seen_names:
                seen_names.add(name.lower())
                resolved_names.append(name)
    return resolved_names[:max_terms]


def prefetch(
    provider: Any,
    query: str,
    session_id: str,
    messages: list[dict[str, Any]] | None = None,
    deadline_ms: int | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run hybrid retrieval, dedup, budget, and diff-based suppression.

    Returns ``{}`` when memory isn't needed, no usable result is found, the
    token budget cannot fit any result, or the injected context is unchanged
    since the last call in this session. Otherwise returns::

        {"context": <compact block str>, "memories": [...], "token_estimate": int,
         "hash": <sha256 hex>, "session_id": <sid>}
    """
    cfg: RemnantConfig = provider._config
    db: RemnantDB = provider._db
    agent_id = cfg.agent_id

    def _record(outcome: str, reason: str | None, t0: float,
                count: int = 0, tokens: int = 0) -> None:
        """Best-effort stats recording — never breaks prefetch."""
        if diagnostics is not None:
            diagnostics.update(outcome=outcome, reason=reason)
            return
        try:
            elapsed = (time.monotonic() - t0) * 1000.0
            db.record_prefetch(
                session_id=session_id, outcome=outcome, reason=reason,
                elapsed_ms=elapsed, result_count=count, token_estimate=tokens,
                query=query, agent_id=agent_id,
            )
        except Exception:
            pass

    _t_init = time.monotonic()

    if not cfg.prefetch_enabled:
        _record("empty", "disabled", _t_init)
        return {}
    if not _needs_memory(query):
        _record("empty", "not_needed", _t_init)
        return {}
    if deadline_ms is None:
        deadline_ms = cfg.injection_prefetch_deadline_ms
    deadline_s = deadline_ms / 1000.0
    t0 = time.monotonic()

    limit = cfg.search_limit

    # Build a local keyword baseline first.  This is intentionally before any
    # network call: if Ollama is busy or unavailable, BM25 results remain
    # usable and can still be injected within the prefetch deadline.
    seen_ids: set[str] = set()
    merged: list[dict[str, Any]] = []
    expansions = [query]
    for term in _expand_queries(query):
        if term and term not in expansions:
            expansions.append(term)

    # Graph-based query expansion (Issue #28): resolve entity mentions
    # (including stopword-bearing phrases like "the printer") against the
    # entity graph and add canonical entity names as search terms. Pure
    # SQLite, <10ms — stays well within the prefetch deadline.
    try:
        graph_terms = _graph_expand(
            db,
            query,
            agent_id=agent_id,
            evidence_only=bool(getattr(cfg, "relation_evidence_enabled", False)),
        )
        for term in graph_terms:
            if term and term not in expansions:
                expansions.append(term)
    except Exception:
        pass  # Never let graph expansion break prefetch.

    def _merge(results: list[dict[str, Any]]) -> None:
        for r in results:
            mid = r.get("id")
            if mid and mid not in seen_ids:
                seen_ids.add(mid)
                merged.append(r)

    for term in expansions:
        if time.monotonic() - t0 >= deadline_s:
            break
        try:
            results = hybrid_search(
                db, cfg, term,
                agent_id=agent_id,
                limit=limit,
                strategy="keyword",
                embedder=None,
            )
        except Exception:
            results = []
        _merge(results)

    # Try one bounded semantic pass after the lexical baseline exists.  The
    # request timeout is shorter than the overall deadline and leaves room for
    # formatting/SQLite work.  A failed attempt is a normal degraded path, not
    # a reason to throw away the keyword baseline.
    semantic_ready = False
    semantic_used = False
    embedding_budget_ms = int(getattr(cfg, "prefetch_embedding_timeout_ms", 250) or 0)
    remaining_ms = max(0.0, deadline_s * 1000.0 - (time.monotonic() - t0) * 1000.0)
    embedding_budget_ms = min(embedding_budget_ms, max(0, int(remaining_ms - 50)))
    session_embedder = None
    if embedding_budget_ms > 0:
        session_embedder = provider._session_embedder(
            session_id,
            query,
            timeout_s=embedding_budget_ms / 1000.0,
        )
        semantic_ready = bool(session_embedder and session_embedder.embed(query))
        semantic_remaining_ms = (
            deadline_s * 1000.0 - (time.monotonic() - t0) * 1000.0
        )
        # Exact-vector ranking is local but can still scan thousands of BLOBs;
        # do not start it when there is no room left for ranking and formatting.
        if semantic_ready and semantic_remaining_ms >= 100:
            try:
                with db.deadline(t0 + deadline_s - 0.05):
                    semantic_results = hybrid_search(
                        db,
                        cfg,
                        query,
                        agent_id=agent_id,
                        limit=limit,
                        strategy="auto",
                        embedder=session_embedder,
                    )
                semantic_used = True
            except Exception:
                semantic_results = []
            # Put the hybrid ranking ahead of expansion-only keyword results,
            # while still retaining every lexical result not already present.
            original_merged = list(merged)
            merged.clear()
            seen_ids.clear()
            _merge(semantic_results)
            _merge(original_merged)

    budget = cfg.injection_token_budget
    response = RecallService(db, cfg).recall(
        RecallRequest(
            query=query,
            agent_id=agent_id,
            session_id=session_id,
            strategy="auto",
            limit=limit,
            messages=messages,
            include_pending=cfg.recent_turn_overlay_enabled,
            token_budget=budget,
            output_mode="context",
            token_counter=getattr(provider, "_token_counter", None),
            echo_service=getattr(provider, "_echo", None),
            echo_viewer_key=(
                provider._effective_identity.viewer_key
                if getattr(provider, "_effective_identity", None) is not None
                else agent_id
            ),
        ),
        candidates=merged,
    )
    selected = response.results
    context = response.context
    if not selected or not context:
        reason = response.diagnostics.get("reason", "budget_exhausted")
        if reason == "no_results" and time.monotonic() - t0 >= deadline_s:
            reason = "deadline"
        _record("empty", reason, t0)
        return {}

    # The remote portion is strictly bounded above.  If local formatting runs
    # a few milliseconds over, preserve the already-safe lexical fallback
    # rather than converting a useful result into an empty injection.

    ctx_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()
    injection_reason = None if semantic_used else "semantic_timeout_keyword_fallback"
    _record("injected", injection_reason, t0, count=len(selected),
            tokens=response.diagnostics["token_estimate"])
    result = {
        "context": context,
        "memories": [
            {"id": m.get("id"), "content": m.get("content", ""), "visibility": m.get("visibility")}
            for m in selected
        ],
        "token_estimate": response.diagnostics["token_estimate"],
        "hash": ctx_hash,
        "session_id": session_id,
    }
    echo = getattr(provider, "_echo", None)
    if echo is not None:
        try:
            compiled_context = response.compiled_context or compile_context_details(
                selected,
                token_budget=budget,
                token_counter=getattr(provider, "_token_counter", None),
            )
            result["_echo_draft"] = echo.build_receipt_draft(
                query=query,
                session_id=session_id,
                agent_id=agent_id,
                viewer_key=(
                    provider._effective_identity.viewer_key
                    if getattr(provider, "_effective_identity", None) is not None
                    else agent_id
                ),
                profile_scope=getattr(cfg, "profile_scope", None),
                memory_generation=getattr(provider, "_memory_generation", 0),
                context=compiled_context,
            )
        except Exception:
            log.debug("Echo receipt draft failed", exc_info=True)
    return result


__all__ = [
    "prefetch",
    "_needs_memory",
    "_expand_queries",
    "_graph_expand",
    "_entity_lookup_phrases",
]
