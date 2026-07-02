# Hermes Remnant

Remnant is the memory engine plugin for the Hermes Agent fleet.

It replaces Hindsight with a deduplicating, trust-scored, self-editing memory system that proactively injects relevant context before LLM calls.

## Status

Under active development. Phase 1 (core storage + retrieval) in progress.

## Architecture

- Storage: SQLite + FTS5
- Embeddings: `nomic-embed-text` via BSL1 Ollama (default)
- Extraction: `gemma4:12b` via BSL1 Ollama
- Dream loop: cloud model (overnight)
- Hermes plugin API: `sync_turn()`, `prefetch()`, `system_prompt_block()`, tools

## Repository layout

```
hermes-remnant/
├── remnant/              # Python package
│   ├── __init__.py       # Hermes plugin entry point
│   ├── db.py             # SQLite schema + queries
│   ├── embed.py          # Embedding client (Ollama)
│   ├── extract.py        # Async extraction worker
│   ├── ingest.py         # Conversation turn ingestion
│   ├── search.py         # BM25 / vector / graph search
│   ├── tools.py          # memory_search, memory_store, etc.
│   └── config.py         # Configuration loader
├── tests/                # pytest suite
├── docs/                 # Architecture + phase specs
├── pyproject.toml
└── README.md
```

## Development

Local repo: `/mnt/data/dev/hermes-remnant`
Live plugin path: `~/.hermes/plugins/remnant/`
