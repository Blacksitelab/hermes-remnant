# Remnant — Phase 2 Spec: Semantic Search + RRF Fusion + Proactive Injection

## Goal

Add semantic vector search, hybrid RRF ranking, and proactive memory injection before LLM calls. Introduce a reflection tool that synthesizes across memories.

## Hermes API

- `prefetch(query, *, session_id="")` — called before each API call; return recalled context.
- `queue_prefetch(query)` — optional, called after each turn to pre-warm next turn.
- Tool schemas and `handle_tool_call` as before.

## In scope

- Semantic vector search (`strategy="semantic"`) using stored embeddings
- Hybrid auto search (`strategy="auto"`) with Reciprocal Rank Fusion (RRF) of BM25 + vector
- `memory_search` gains `strategy` parameter
- `prefetch()` proactive injection:
  - Lightweight local intent classifier (regex/keyword) to decide if memory is needed
  - Skip casual chat ("hey", "how are you")
  - Entity extraction from query + 2-3 derived search terms
  - Run hybrid search with deadline (default 500ms)
  - Enforce token budget (default 2000 tokens)
  - Deduplicate against current conversation messages
  - Diff-based injection: hash the injected context; if unchanged since last turn, return {}
  - Return compact context block in `prefetch()` result
- `memory_reflect` tool: synthesize answer across top-N memories via local LLM
- Per-session injection cache (last injected hash) in provider
- Config options: `injection_token_budget`, `injection_prefetch_deadline_ms`, `prefetch_enabled`

## Out of scope

- Entity graph traversal (Phase 3)
- Self-editing tools (Phase 3)
- Vault indexing (Phase 4)
- Dream loop / threads (Phase 5)

## File targets

| File | Changes |
|------|---------|
| `remnant/__init__.py` | Implement `prefetch()`, `queue_prefetch()`; add injection state |
| `remnant/config.py` | Add `injection_token_budget`, `injection_prefetch_deadline_ms`, `prefetch_enabled`, `reflect_model`, `reflect_endpoint` |
| `remnant/db.py` | Add `search_all_active()` and `search_by_embedding()` helpers; load embeddings for candidate set |
| `remnant/search.py` | Add `search(strategy="semantic")` and `search(strategy="auto")` with RRF |
| `remnant/prefetch.py` | New file: intent classifier, query expansion, token budget, dedup, formatting |
| `remnant/reflect.py` | New file: `memory_reflect` synthesis via LLM |
| `remnant/tools.py` | Add `memory_reflect` schema + dispatch |
| `tests/test_phase2.py` | New tests: semantic search, RRF, prefetch skips casual, prefetch injects relevant, token budget, diff-based, memory_reflect |

## Acceptance criteria

- [ ] `memory_search(strategy="semantic")` returns semantically similar memories
- [ ] `memory_search(strategy="auto")` merges BM25 + vector via RRF
- [ ] `prefetch()` returns empty for casual chat
- [ ] `prefetch()` returns relevant context for factual questions within 500ms
- [ ] Injected context stays under 2000 tokens
- [ ] Diff-based injection prevents repeating same context within a session
- [ ] Injected memories are deduplicated against conversation history
- [ ] `memory_reflect` synthesizes across top memories with source attribution
- [ ] All Phase 1 tests still pass
- [ ] All Phase 2 tests pass

## Token / speed constraints

- `prefetch()` must complete within `injection_prefetch_deadline_ms` (default 500ms). Use a monotonic timer; if exceeded, return empty.
- Semantic search loads embeddings only for candidate memories returned by BM25 pre-filter (limit ~100), not the entire database.
- Brute-force cosine over candidates is fine; at <100K scale this is milliseconds.
- RRF constant k=60.
- Injected context format must be compact: one line per memory.
- Reflection uses `gemma4:12b` with max_tokens 512 and bounded input.

## Design notes

- Keep embedding model consistent with storage (`nomic-embed-text`, 768-dim).
- Query embedding should be cached per session to avoid duplicate Ollama calls.
- RRF formula: score = sum(1 / (k + rank)) across each ranked list.
- Intent classifier: keywords like "remember", "did we", "what did", "decide", "status of", "last time" → need memory. Greetings and small talk → skip.
- Query expansion: extract capitalized/proper nouns and noun phrases from the query; issue multiple searches and merge.
