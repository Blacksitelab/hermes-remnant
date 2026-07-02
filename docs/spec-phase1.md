# Remnant — Phase 1 Spec: Core Storage + Basic Retrieval

## Goal

Implement a Hermes memory provider plugin that stores conversation turns, extracts durable facts asynchronously, deduplicates memories, and answers keyword searches via tool calls.

## Hermes MemoryProvider API Reference

Read https://hermes-agent.nousresearch.com/docs/developer-guide/memory-provider-plugin before implementing.

Key points:
- Plugin directory format: `plugins/memory/remnant/` (or local repo root with symlink)
- Entry point: `register(ctx)` calls `ctx.register_memory_provider(RemnantMemoryProvider())`
- ABC: `from agent.memory_provider import MemoryProvider`
- Required methods:
  - `name` property
  - `is_available()` — no network calls
  - `initialize(session_id, **kwargs)` — `kwargs` includes `hermes_home`; use it for all storage paths
  - `get_tool_schemas()`
  - `handle_tool_call(tool_name, args, **kwargs)`
  - `get_config_schema()`
  - `save_config(values, hermes_home)`
- Optional hooks implemented now:
  - `system_prompt_block()` — static, byte-stable for conversation lifetime
  - `sync_turn(user, assistant, *, session_id="", messages=None)` — non-blocking, write raw turn and enqueue async extraction
  - `shutdown()`
- **No `backup_paths()` method exists.** The SQLite DB is included in workspace backup because it lives under `hermes_home`.

## In scope

- SQLite schema with all Phase 1 tables
- Hermes `MemoryProvider` implementation (`RemnantMemoryProvider`)
- `memory_search` tool (BM25 / FTS5 only)
- `memory_store` tool (manual storage with dedup)
- `sync_turn()` writes raw turns and enqueues async extraction
- Async extraction via `gemma4:12b` on BSL1
- Entity extraction and resolution
- Deduplication (BM25 + cosine similarity)
- Transient-state filter
- Ollama embedding via `nomic-embed-text`
- Static, byte-stable `system_prompt_block()`
- `plugin.yaml` manifest
- `get_config_schema()` / `save_config()` for basic Ollama endpoints

## Out of scope

- Semantic / vector search (Phase 2)
- Proactive `prefetch()` injection (Phase 2)
- Graph traversal (Phase 3)
- Self-editing tools (Phase 3)
- Vault indexing (Phase 4)
- Dream loop / threads (Phase 5)
- Migration from Hindsight (cross-phase)

## File targets

| File | Responsibility |
|------|----------------|
| `remnant/__init__.py` | `RemnantMemoryProvider` + `register()` entry point |
| `remnant/plugin.yaml` | Manifest |
| `remnant/config.py` | Load config from `hermes_home/remnant.json` |
| `remnant/db.py` | SQLite schema, migrations, CRUD, FTS5 setup |
| `remnant/embed.py` | Ollama embedding client + cosine helper |
| `remnant/extract.py` | Async extraction worker, LLM prompts |
| `remnant/ingest.py` | `sync_turn()` pipeline, dedup, transient filter |
| `remnant/search.py` | BM25 keyword search |
| `remnant/tools.py` | Tool schemas + dispatch |
| `tests/test_phase1.py` | Phase 1 integration smoke tests |

## Acceptance criteria

- [ ] Plugin loads in Hermes without errors when installed/symlinked into `plugins/memory/remnant/`
- [ ] `sync_turn()` writes conversation turns in <10ms and returns immediately
- [ ] Async extraction produces facts + entities in background
- [ ] `memory_search` returns BM25-ranked active memories
- [ ] `memory_store` deduplicates identical or near-identical facts
- [ ] Transient facts like "printer is at 32%" are rejected
- [ ] `system_prompt_block()` does not change during a conversation
- [ ] All storage paths are profile-scoped under `hermes_home`
- [ ] `shutdown()` cleanly stops the extraction worker
- [ ] All tests pass

## Configuration

Default config in `hermes_home/remnant.json`:
```json
{
  "embedding_model": "nomic-embed-text",
  "embedding_endpoint": "http://192.168.0.11:11434/api/embeddings",
  "extraction_model": "gemma4:12b",
  "extraction_endpoint": "http://192.168.0.11:11434/v1/chat/completions"
}
```

`get_config_schema()` should expose these as optional fields with defaults, so setup is painless.

## Implementation notes

- Use `ThreadPoolExecutor` for the extraction worker
- Serialize queue items to SQLite (`extraction_queue` table) so restarts don't lose turns
- Keep embeddings as float32 blobs
- BM25 via FTS5 `bm25()` rank; fallback if no results
- Static system prompt block only describes tools, never includes live data
- Profile isolation: everything under `hermes_home/remnant/`
