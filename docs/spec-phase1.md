# Remnant — Phase 1 Spec: Core Storage + Basic Retrieval

## Goal

Implement a Hermes plugin that can store conversation turns, extract durable facts asynchronously, deduplicate memories, and answer keyword searches via tool calls.

## In scope

- SQLite schema with all Phase 1 tables
- Hermes memory provider API (`RemnantMemoryProvider`)
- `memory_search` tool (BM25 / FTS5 only)
- `memory_store` tool (manual storage with dedup)
- `sync_turn()` writes raw turns and enqueues async extraction
- Async extraction via `gemma4:12b` on BSL1
- Entity extraction and resolution
- Deduplication (BM25 + cosine similarity)
- Transient-state filter
- Ollama embedding via `nomic-embed-text`
- Static, byte-stable `system_prompt_block()`
- `backup_paths()` exposes SQLite file

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
| `remnant/__init__.py` | Plugin entry point, `RemnantMemoryProvider` |
| `remnant/config.py` | Load `config.yaml` `memory.remnant.*` |
| `remnant/db.py` | SQLite schema, migrations, CRUD, FTS5 setup |
| `remnant/embed.py` | Ollama embedding client + cosine helper |
| `remnant/extract.py` | Async extraction worker, LLM prompts |
| `remnant/ingest.py` | `sync_turn()` pipeline, dedup, transient filter |
| `remnant/search.py` | BM25 keyword search |
| `remnant/tools.py` | Tool schemas + dispatch |
| `tests/test_phase1.py` | Phase 1 integration smoke tests |

## Acceptance criteria

- [ ] Plugin loads in Hermes without errors
- [ ] `sync_turn()` writes conversation turns in <10ms
- [ ] Async extraction produces facts + entities in background
- [ ] `memory_search` returns BM25-ranked active memories
- [ ] `memory_store` deduplicates identical or near-identical facts
- [ ] Transient facts like "printer is at 32%" are rejected
- [ ] `system_prompt_block()` does not change during a conversation
- [ ] `backup_paths()` returns the SQLite file path
- [ ] All tests pass

## Open questions resolved

- Embedding model: `nomic-embed-text` via BSL1 Ollama (768-dim)
- Extraction queue persistence: write pending items to `extraction_queue` table
- Plugin location in dev: repo at `/mnt/data/dev/hermes-remnant`; install to `~/.hermes/plugins/remnant/` via symlink or pip install
- Visibility: implement `private`/`shared`/`fleet` fields even though Phase 1 only uses `private` and `fleet`

## Implementation notes

- Use `ThreadPoolExecutor` for the extraction worker
- Serialize queue items to SQLite so restarts don't lose turns
- Keep embeddings as float32 blobs
- BM25 via FTS5 `bm25()` rank; fallback if no results
- Static system prompt block only describes tools, never includes live data
