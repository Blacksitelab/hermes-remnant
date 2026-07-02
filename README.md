# Hermes Remnant

Remnant is the long-term memory engine for the [Hermes Agent](https://hermes-agent.nousresearch.com) fleet. It is implemented as a Hermes memory-provider plugin that stores durable facts, observations, documents, and threads in a local SQLite database, retrieves them with hybrid keyword + vector + graph search, and proactively injects relevant context before each LLM call.

This repo contains both the **Hermes plugin** (`remnant/`) and the **test suite** that exercises every phase of the system.

**Repository:** [https://github.com/Blacksitelab/hermes-remnant](https://github.com/Blacksitelab/hermes-remnant)  
**Local path:** `/mnt/data/dev/hermes-remnant`  
**Author / commits:** Sven (`sven@blacksitelab.com`)

---

## What Remnant does

- **Stores durable facts** extracted from conversation turns, vault notes, and imports.
- **Dedupes aggressively** — the same fact is never stored twice; duplicate hits increment a `seen_count`.
- **Filters transient state** — percentages, current timestamps, and words like *currently* / *now* are rejected.
- **Scores trust** — every memory has a `trust_score` that is raised or lowered by feedback.
- **Self-edits** — agents can update, merge, forget, share, unshare, and score memories through tools.
- **Searches three ways** — BM25 keyword, cosine vector similarity, entity-graph traversal, plus a hybrid RRF fusion.
- **Proactively injects context** via Hermes' `prefetch()` hook, with a hard deadline, token budget, and diff-based suppression.
- **Indexes the Obsidian vault** as `document` memories, respecting workspace exclusions, frontmatter, locked notes, and per-agent profile scopes.
- **Runs a bounded dream loop** that finds non-obvious connections across memories using a cloud model, writes a first-person diary, and promotes real insights to threads.
- **Imports existing memory stores** — Hermes MEMORY.md / USER.md files and Hindsight memories — with `dry_run` and `shadow` modes.

---

## Design principles

1. **Injection must not measurably increase response latency.**  
   `prefetch()` runs in a background thread and returns within 500 ms. If retrieval is not done in time, it returns empty. Better no context than late context.

2. **Injection must not bloat context.**  
   Hard 2000-token budget. Memories are compact, deduplicated against the current conversation, and only included when genuinely relevant.

3. **Extraction is always async.**  
   `sync_turn()` writes the raw turn to SQLite in a single transaction (<10 ms) and enqueues extraction. The conversation never waits for LLM extraction, embedding, or entity resolution.

4. **Embeddings are cached.**  
   Every memory embedding is stored in SQLite. Query embeddings are cached per session. Never re-embed the same content.

5. **The dream loop is bounded.**  
   Only a pre-filtered candidate list (≤30 pairs) is sent to the cloud model. Local cosine similarity does the heavy lifting.

---

## Tech stack

| Component | Choice | Reason |
|-----------|--------|--------|
| Storage | SQLite + FTS5 | Zero dependency, single-file, WAL mode, fast |
| Embeddings | `nomic-embed-text` via BSL1 Ollama (768-dim) | Already loaded, CPU/GPU-capable, simple HTTP API |
| Extraction / rerank / reflect | `gemma4:12b` on BSL1 via Ollama OpenAI-compatible API | Proven for extraction, already running |
| Dream loop | Cloud model (`deepseek-v4-flash:cloud` by default) | Overnight quality, latency irrelevant |
| Framework | Hermes memory-provider plugin | Registers `sync_turn`, `prefetch`, tool schemas |
| Concurrency | `ThreadPoolExecutor` | Proper shutdown semantics |

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/Blacksitelab/hermes-remnant.git /mnt/data/dev/hermes-remnant
cd /mnt/data/dev/hermes-remnant
```

### 2. Create a virtual environment and install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. Register as a Hermes plugin

Hermes discovers plugins from `~/.hermes/plugins/`. Link the package:

```bash
mkdir -p ~/.hermes/plugins
ln -s /mnt/data/dev/hermes-remnant/remnant ~/.hermes/plugins/remnant
```

The plugin manifest is `remnant/plugin.yaml`.

### 4. Configure Hermes to use Remnant

In your Hermes `config.yaml`:

```yaml
memory:
  provider: remnant
```

Run `hermes memory setup` to configure endpoints and agent id.

---

## Quick start

Run the test suite:

```bash
python -m pytest tests/ -q
```

Run a local smoke test:

```python
from pathlib import Path
from remnant import RemnantMemoryProvider

provider = RemnantMemoryProvider()
home = Path("/tmp/remnant-smoke")
provider.initialize("session-1", hermes_home=str(home))
provider.sync_turn("Sven prefers dark mode.", "Noted.", session_id="session-1")
result = provider.handle_tool_call("memory_search", {"query": "Sven preference"}, session_id="session-1")
print(result)
provider.shutdown()
```

---

## Project structure

```text
remnant/
├── __init__.py          # RemnantMemoryProvider + Hermes register entry point
├── plugin.yaml          # Hermes plugin manifest
├── config.py            # Configuration model, defaults, load/save
├── db.py                # SQLite schema, migrations, CRUD, search helpers
├── embed.py             # Embedding cache + Ollama embedder
├── extract.py           # Async fact/entity extraction worker
├── ingest.py            # Turn ingestion, transient filter, contradiction detection
├── entity.py            # Entity extraction, resolution, alias normalization
├── graph.py             # Pure-SQLite graph traversal
├── search.py            # BM25, vector, RRF, graph, profile-scope search
├── tools.py             # Tool schemas and dispatch (search/store/edit/graph/reflect/import/thread)
├── edit.py              # memory_edit actions + audit logging
├── prefetch.py          # Proactive prefetch with deadline/budget/dedup
├── reflect.py           # memory_reflect synthesis
├── vault.py             # Obsidian vault indexer
├── threads.py           # Thread CRUD + stale sweep
├── dream.py             # Day/night dream loop
└── import_sources.py    # MEMORY.md / USER.md / Hindsight / shadow import

tests/
├── test_phase1.py       # Core storage, retrieval, dedup, transient filter
├── test_phase2.py       # Semantic search, RRF, prefetch, reflection
├── test_phase3.py       # Entity graph, self-editing, audit log, contradictions
├── test_phase4.py       # Vault indexing, profile scope, locked notes
├── test_phase5.py       # Threads, dream loop, budget, diary
└── test_migration.py    # Memory-store + Hindsight import, dry_run, shadow

docs/
├── spec-phase1.md
├── spec-phase2.md
├── spec-phase3.md
├── spec-phase4.md
├── spec-phase5.md
└── spec-migration.md
```

---

## Hermes MemoryProvider API

The provider implements the Hermes `MemoryProvider` ABC:

| Method | Purpose |
|--------|---------|
| `name` | Returns `"remnant"` |
| `is_available()` | File-system check, no network calls |
| `initialize(session_id, **kwargs)` | Receives `hermes_home`; opens DB, starts extraction worker |
| `get_config_schema()` | Returns config keys for `hermes memory setup` |
| `save_config(values, hermes_home)` | Persists config to YAML |
| `system_prompt_block()` | Static, byte-stable tool description |
| `sync_turn(...)` | Persist turn, enqueue extraction, non-blocking |
| `prefetch(query, ...)` | Return relevant memories before LLM call |
| `queue_prefetch(query)` | Best-effort prefetch warming |
| `get_tool_schemas()` | Exposes all memory tools |
| `handle_tool_call(tool_name, args, ...)` | Dispatches to internal tools |
| `shutdown()` | Stops worker, closes DB |

Entry point:

```python
def register(ctx):
    ctx.register_memory_provider(RemnantMemoryProvider())
```

---

## Tools exposed to agents

| Tool | What it does |
|------|--------------|
| `memory_search` | Keyword, semantic, graph, or hybrid (RRF) search over active memories |
| `memory_store` | Explicitly store a durable fact, with dedup |
| `memory_edit` | Update, merge, forget, feedback, share, unshare a memory |
| `memory_graph` | Traverse entity graph around a named entity |
| `memory_reflect` | Synthesize an answer across top memories |
| `memory_thread` | Create, update, resolve, list, or sweep stale threads |
| `memory_import` | Import from `vault`, `memory_store`, or `hindsight` |

---

## Configuration

Default config lives at `~/.hermes/remnant/config.yaml` and can be edited or set through `hermes memory setup`.

```yaml
agent_id: default
embed_url: http://192.168.0.11:11434/api/embeddings
embed_model: nomic-embed-text
extract_url: http://192.168.0.11:11434/v1/chat/completions
extract_model: gemma4:12b
extract_enabled: true
default_visibility: private
vault_path: /home/jd/obsidian-vaults/BlacksiteLabVault
vault_exclude:
  - "90_*"
  - "91_*"
  - "92_*"
  - "93_*"
  - "94_*"
  - "95_*"
  - "99_ARCHIVE/"
dream_day_model: deepseek-v4-flash:cloud
dream_night_model: deepseek-v4-flash:cloud
dream_cooldown_minutes: 120
injection_token_budget: 2000
injection_prefetch_deadline_ms: 500
```

---

## Phases

Remnant was built in five implementation phases plus a migration phase.

| Phase | Focus | Commit | Tests |
|-------|-------|--------|-------|
| 1 | Core storage, async extraction, BM25, dedup, transient filter, visibility | `77a5680` | 29 |
| 2 | Semantic search, RRF fusion, proactive `prefetch()`, `memory_reflect` | `a39e8e1` | 58 |
| 3 | Entity graph, `memory_edit`, audit log, contradiction detection | `2f08eca` | 108 |
| 4 | Vault indexing, frontmatter, profile-scoped search, locked notes | `20b4793` | 153 |
| 5 | Threads, bounded day/night dream loop, diary | `064dd86` | 185 |
| Migration | Import from Hindsight + MEMORY.md, dry_run, shadow mode | `6824b65` | 223 |

---

## Development

### Run tests

```bash
python -m pytest tests/ -v
```

### Lint

```bash
ruff check remnant tests
ruff check remnant tests --fix
```

### Type check (optional)

```bash
mypy remnant
```

---

## Migration from Hindsight

Remnant can import existing facts without disrupting current operation.

### Dry-run preview

```python
provider.handle_tool_call("memory_import", {
    "source": "memory_store",
    "dry_run": True,
})
```

### Shadow mode

```python
provider.handle_tool_call("memory_import", {
    "source": "hindsight",
    "shadow": True,
})
```

Shadow entries are appended to `~/.hermes/remnant/shadow.log` as JSON lines for comparison against Hindsight's actual injections. Once Remnant is consistently better, switch `memory.provider` to `remnant` in Hermes config.

---

## Backup

The SQLite database at `~/.hermes/remnant/remnant.db` is returned by the provider's backup path list. Include it in normal Hermes workspace backups.

---

## Roadmap / not in scope

- BSL Hub integration as a future data source.
- Entity community detection (deferred until graph traversal needs it).
- Email / feed / sensor indexing.
- Web dashboard.

---

## License

Internal BlacksiteLab project. Not licensed for public distribution.
