# Hermes Remnant

Remnant is the long-term memory engine for the [Hermes Agent](https://hermes-agent.nousresearch.com) fleet. It is implemented as a Hermes memory-provider plugin that stores durable facts, observations, documents, and threads in a local SQLite database, retrieves them with hybrid keyword + vector + graph search, and proactively injects relevant context before each LLM call.

This repo contains both the **Hermes plugin** (`remnant/`) and the **test suite** that exercises every phase of the system.

**Repository:** [https://github.com/Blacksitelab/hermes-remnant](https://github.com/Blacksitelab/hermes-remnant)  
**Local path:** `/mnt/data/dev/hermes-remnant`  


---

## What Remnant does

- **Stores durable facts** extracted from conversation turns, vault notes, and imports.
- **Dedupes aggressively** — the same fact is never stored twice; duplicate hits increment a `seen_count`.
- **Filters transient state** — percentages, current timestamps, and words like *currently* / *now* are rejected.
- **Scores trust** — every memory has a `trust_score` calibrated by source quality, confidence, verification status, and engagement. Trust scores influence search ranking and decay over time.
- **Self-edits** — agents can update, merge, forget, share, unshare, and score memories through tools.
- **Searches three ways** — BM25 keyword, cosine vector similarity, entity-graph traversal, plus a hybrid RRF fusion.
- **Proactively injects context** via Hermes' `prefetch()` hook, with a hard deadline, token budget, and diff-based suppression.
- **Indexes the Obsidian vault** as `document` memories, respecting workspace exclusions, frontmatter, locked notes, and per-agent profile scopes.
- **Extracts entities with GLiNER** — a lightweight NER model (`urchade/gliner_small_v2`) extracts named entities with typed labels (person, tool, service, project, place, organization, concept). Falls back to regex if GLiNER is unavailable.
- **Classifies typed relations** — co-occurrence edges are classified into semantic types (owns, uses, created, depends_on, monitors, manages, interacts_with, references, part_of) using entity-type heuristics.
- **Expands queries via the entity graph** — when a user says "the printer", the prefetch pipeline generates n-gram phrases, resolves them against the entity graph (including aliases), traverses 1 hop to related entities, and adds canonical entity names as additional search terms. This bridges the gap between colloquial references and canonical entity names without adding latency.
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

6. **Entity extraction quality matters.**  
   GLiNER (a transformer-based NER model) is the primary entity extractor. Regex patterns are the fallback. The difference in quality is significant — GLiNER correctly identifies entities like `Qwen3-TTS` as a single tool rather than splitting it into `Qwen3` and `TTS`.

7. **The entity graph is a query expansion surface, not just a traversal tool.**  
   When a user says "the printer", `_graph_expand()` resolves the phrase against entity aliases ("the printer" → `elegoo centauri carbon v1`), traverses 1 hop to related entities, and adds canonical names as additional search terms. This runs in pure SQLite (<10 ms) and bridges the gap between how people talk and how entities are stored.

---

## Tech stack

| Component | Choice | Reason |
|-----------|--------|--------|
| Storage | SQLite + FTS5 | Zero dependency, single-file, WAL mode, fast |
| Embeddings | `nomic-embed-text` via BSL1 Ollama (768-dim) | Already loaded, CPU/GPU-capable, simple HTTP API |
| Entity extraction | `urchade/gliner_small_v2` via GLiNER (CPU) | Purpose-built NER, ~400 MB, millisecond inference, typed entities |
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
pip install gliner  # for GLiNER-based entity extraction
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

Remnant stores its SQLite database at a **shared** location — `~/.hermes/remnant/remnant.db` — used by every Hermes profile and agent, so cross-agent features (shared vault search, dream-loop cross-agent dedup, entity-graph traversal across agents) work without merging per-profile databases. The location can be overridden with the `REMNANT_DB_HOME` env var. Per-profile **config** stays under each profile's `hermes_home` at `hermes_home/remnant.json`; only the DB is shared.

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
├── __init__.py              # RemnantMemoryProvider + Hermes register entry point
├── plugin.yaml              # Hermes plugin manifest
├── config.py                # Configuration model, defaults, load/save
├── db.py                    # SQLite schema, migrations, CRUD, search helpers
├── embed.py                 # Embedding cache + Ollama embedder
├── extract.py               # Async fact/entity extraction worker
├── ingest.py                # Turn ingestion, transient filter, contradiction detection
├── entity.py               # Entity extraction (GLiNER + regex fallback), resolution, alias normalization
├── graph.py                 # Pure-SQLite graph traversal
├── search.py                # BM25, vector, RRF, graph, profile-scope search
├── tools.py                 # Tool schemas and dispatch (search/store/edit/graph/reflect/import/thread)
├── edit.py                  # memory_edit actions + audit logging
├── prefetch.py              # Proactive prefetch with deadline/budget/dedup + entity-graph query expansion
├── reflect.py               # memory_reflect synthesis
├── vault.py                 # Obsidian vault indexer
├── threads.py               # Thread CRUD + stale sweep
├── dream.py                 # Day/night dream loop
├── import_sources.py        # MEMORY.md / USER.md / Hindsight / shadow import
├── reextract.py             # Batch entity re-extraction (GLiNER) + orphan cleanup
├── classify_relations.py    # Typed relation classifier (entity-type heuristics)
└── calibrate_trust.py       # Trust score calibration (source quality, verification, engagement)

tests/
├── test_phase1.py           # Core storage, retrieval, dedup, transient filter
├── test_phase2.py           # Semantic search, RRF, prefetch, reflection
├── test_phase3.py           # Entity graph, self-editing, audit log, contradictions
├── test_phase4.py           # Vault indexing, profile scope, locked notes
├── test_phase5.py           # Threads, dream loop, budget, diary
└── test_migration.py         # Memory-store + Hindsight import, dry_run, shadow

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
| `prefetch(query, ...)` | Return relevant memory context text before an LLM call |
| `get_tool_schemas()` | Exposes all memory tools |
| `handle_tool_call(tool_name, args, ...)` | Dispatches to internal tools |
| `on_session_switch()` | Clears per-session recall state when Hermes rotates sessions |
| `backup_paths()` | Declares the shared database for Hermes backups |
| `shutdown()` | Stops worker, closes DB |

Entry point:

```python
def register(ctx):
    ctx.register_memory_provider(RemnantMemoryProvider())
```

Install from GitHub and select Remnant as the single active external memory provider:

```bash
hermes plugins install Blacksitelab/hermes-remnant
hermes config set memory.provider remnant
hermes memory status
```

Hermes also exposes provider selection through `hermes plugins` and
`hermes memory setup`. Remnant continues to support keyword-only recall when
the configured embedding or extraction service is unavailable.

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

## Entity extraction

Remnant uses a two-tier entity extraction strategy:

### Primary: GLiNER (transformer-based NER)

[GLiNER](https://github.com/urchade/GLiNER) is a lightweight NER model that identifies named entities in text and assigns them typed labels. Remnant uses `urchade/gliner_small_v2` (~400 MB), which runs on CPU in milliseconds.

**Supported entity types:** person, organization, project, tool, service, place, concept

**How it works:**
1. Text is passed to the GLiNER model with the label set above
2. Model returns entities with confidence scores (threshold: 0.5)
3. Entities are deduplicated, filtered against a stoplist, and resolved against the existing entity graph
4. If GLiNER is not installed or returns no entities, the regex extractor runs as a fallback

**Wiring:** `extract_entities_gliner()` in `entity.py` is called by `extract_high_signal_entities()` when `use_gliner=True` (default). The model is loaded lazily as a module-level singleton — first call loads the model, subsequent calls reuse it.

### Fallback: Regex patterns

If GLiNER is unavailable, a regex-based extractor identifies entities using capitalization patterns, CamelCase detection, known-project matching, and a stoplist. This is the original Phase 3 extractor, preserved for environments without GLiNER.

### Batch re-extraction

`reextract.py` re-extracts entities for all memories in the DB:

```bash
python -m remnant.reextract --dry-run    # preview counts
python -m remnant.reextract --batch 100   # run with progress every 100 memories
```

This clears existing entity links and relations, re-extracts with GLiNER, cleans up orphaned entities, and VACUUMs the database. Always back up the DB first.

---

## Typed relations

Relations between entities are classified into semantic types using entity-type heuristics:

| Relation type | Example | How it's detected |
|---------------|---------|-------------------|
| `owns` | kris → remnant | person owns project/tool |
| `uses` | sven → ollama | person uses tool/service |
| `created` | sven → skill | person created project/service |
| `depends_on` | remnant → sqlite | project/service depends on tool |
| `monitors` | claire → fleet | person monitors project/service |
| `manages` | kris → bsl1 | person manages place/organization |
| `interacts_with` | claire → sven | person interacts with person |
| `references` | vault → remnant | project/service references project |
| `part_of` | hub → blacksitelab | entity is part of organization |
| `co_occurs` | docker → ollama | entities co-occur in memories but no typed relation |
| `related_to` | (fallback) | no heuristic matched |

```bash
python -m remnant.classify_relations --dry-run  # preview
python -m remnant.classify_relations --yes       # apply
```

---

## Trust scoring

Every memory has a `trust_score` (0.0–1.0) that influences search ranking. Trust is calibrated by:

| Factor | Adjustment |
|--------|------------|
| Source: vault document | +0.15 |
| Source: import (memory_store, hindsight) | +0.05 |
| Source: manual entry | +0.05 |
| Source: conversation | 0 (baseline) |
| Source: hindsight | −0.05 |
| Verified by agent | +0.10 |
| Engagement (seen_count > 1) | +0.05 |
| Cap | 0.95 |

Trust scores decay over time via `decay_trust_scores()` in `search.py` — unverified memories drift toward 0.5, verified memories hold their floor.

```bash
python -m remnant.calibrate_trust  # recalibrate all trust scores
```

---

## Configuration

Per-profile config lives at `hermes_home/remnant.json` (where `hermes_home` is the active Hermes profile directory, e.g. `~/.hermes/profiles/<profile>`). Edit it directly or set values through `hermes memory setup`. Each profile keeps its own config — `agent_id`, endpoints, vault path, visibility defaults — so multiple agents can share the single DB while remaining independently configured.

The SQLite database is **shared** across all profiles at `~/.hermes/remnant/remnant.db` (override with the `REMNANT_DB_HOME` env var). Config is profile-scoped; storage is shared.

The default vault path can be overridden with the `REMNANT_VAULT_PATH` env var before constructing a `RemnantConfig`.

```yaml
agent_id: default
embed_url: http://your-ollama-host.local:11434/api/embeddings
embed_model: nomic-embed-text
embed_keep_alive: 10m
extract_url: http://your-ollama-host.local:11434/v1/chat/completions
extract_model: gemma4:12b
extract_keep_alive: 2m
extract_enabled: true
extract_num_ctx: 8192
extract_max_input_tokens: 5500
extract_max_output_tokens: 1536
extract_max_facts: 8
extract_think: false
extract_structured_output: true
default_visibility: private
vault_path: /path/to/your/obsidian-vault
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
prefetch_embedding_timeout_ms: 250
runtime_identity_enabled: false  # enable only with a stable gateway user identity
structured_claim_extraction_v2: true
claim_reconciliation_enabled: true
claim_aware_ranking_enabled: true
ranking_profile: claims-v1
resolved_context_enabled: true
recent_turn_overlay_enabled: true
relation_evidence_enabled: true
```

The 2,000-token injection budget is the recommended ceiling. Resolved context
allocates it deterministically across current claims (60%), uncertainty and
conditional evidence (20%), supporting document/provenance passages (15%),
and recent unprocessed turns (5%), redistributing unused capacity while
preferring complete compact claims. Integrations that expose the Hermes
deployment tokenizer can pass it to the recall service; standalone operation
uses a conservative offline counter.

Prefetch always establishes a local BM25 baseline before attempting the remote
query embedding. If Ollama is busy or unavailable, that keyword context is
injected instead of blocking or dropping recall. Keep-alive values are finite by
default because extraction and embedding commonly share one Ollama host.

The claim-aware correctness stack is the recommended default for new
configurations. Existing explicit values are preserved, and every flag remains
independently reversible. Keep runtime identity disabled unless Hermes supplies
a stable platform user identity; its fail-closed anonymous fallback is scoped to
one session and would otherwise prevent cross-session recall.

## Operations and safe upgrades

```bash
# Bounded local health report; performs no network requests
python -m remnant.maintenance health

# Create and integrity-check a new backup (never overwrites)
python -m remnant.maintenance backup --output /safe/path/remnant-before-0.2.db

# Preview and then apply derived relation-evidence backfill
python -m remnant.maintenance backfill-relation-evidence
python -m remnant.maintenance backfill-relation-evidence --yes

# Restore into a new path for validation; never overwrite the live database
python -m remnant.maintenance restore \
  --backup /safe/path/remnant-before-0.2.db \
  --output /safe/path/remnant-restored.db
```

The health report includes schema/integrity, queue and dead-letter state,
claim coverage and unresolved age, embedding model/dimension coverage,
prefetch outcomes and latency, entity/relation evidence counts, and bounded
operation counters. See [the provider comparison](docs/provider-comparison.md)
for the current evidence-based positioning against Hermes' popular providers.

**GLiNER entity extraction** is enabled by default when the `gliner` package is installed. No configuration needed — the model (`urchade/gliner_small_v2`) is downloaded automatically on first use from HuggingFace (no token required). If `gliner` is not installed, the regex extractor runs automatically.

---

## Phases

Remnant was built in five implementation phases plus a migration phase and a post-launch improvement phase.

| Phase | Focus | Tests |
|-------|-------|-------|
| 1 | Core storage, async extraction, BM25, dedup, transient filter, visibility | 29 |
| 2 | Semantic search, RRF fusion, proactive `prefetch()`, `memory_reflect` | 58 |
| 3 | Entity graph, `memory_edit`, audit log, contradiction detection | 108 |
| 4 | Vault indexing, frontmatter, profile-scoped search, locked notes | 153 |
| 5 | Threads, bounded day/night dream loop, diary | 185 |
| Migration | Import from Hindsight + MEMORY.md, dry_run, shadow mode | 223 |
| 6 | GLiNER NER, typed relation classifier, trust calibration, embedding backfill | 308 |
| 7 | Entity-graph query expansion in prefetch (alias resolution + 1-hop traversal) | 315 |

---

## Production stats

The BlacksiteLab production database (as of July 2026):

| Metric | Value |
|--------|-------|
| Active memories | 1,574 |
| Entities | 2,971 |
| Memory-entity links | 5,886 |
| Relations | 6,399 |
| Embeddings | 1,576 (100% coverage) |
| DB size | 30.4 MB |
| Tests | 315 passing |

Relation type distribution:

| Type | Count |
|------|-------|
| `related_to` | 3,208 |
| `co_occurs` | 2,408 |
| `owns` | 345 |
| `uses` | 175 |
| `created` | 118 |
| `depends_on` | 49 |
| `monitors` | 30 |
| `references` | 30 |
| `manages` | 17 |
| `interacts_with` | 16 |
| `part_of` | 3 |

Entity type distribution:

| Type | Count |
|------|-------|
| `concept` | 952 |
| `tool` | 696 |
| `service` | 553 |
| `project` | 362 |
| `place` | 185 |
| `person` | 107 |
| `organization` | 100 |

Top entities by link count: `user` (510), `kris` (122), `sven` (92), `claire` (88), `project` (74), `assistant` (73), `system` (67), `yuki` (62), `remnant` (52), `klaus` (42), `sasha` (39), `margot` (36), `ai` (35), `bsl1` (31), `blacksitelab` (27).

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

### Evaluate retrieval and inspect health

Use a versioned JSON case file (`query`, `expected_ids`, optional strategy and
agent scope) to measure recall@k, MRR, and latency without mutating memories:

```bash
python -m remnant.evaluate --cases retrieval-cases.json
python -m remnant.maintenance health
python -m remnant.maintenance migrate-default-agent --agent claire  # dry run
python -m remnant.maintenance migrate-default-agent --agent claire --yes
```

Run the scale-envelope harness separately from unit CI before changing the
exact-vector ceiling:

```bash
python -m remnant.evaluation.scale --sizes 5000 --probes 5 --output scale-report.json
```

### Release-track claim resolution

Remnant's recommended profile enables temporal claims, conservative conflict
handling, provenance-aware prompt context, immediate recent-turn recall, and
evidence-backed graph traversal:

```json
{
  "structured_claim_extraction_v2": true,
  "claim_reconciliation_enabled": true,
  "claim_aware_ranking_enabled": true,
  "ranking_profile": "claims-v1",
  "resolved_context_enabled": true,
  "recent_turn_overlay_enabled": true,
  "relation_evidence_enabled": true,
  "runtime_identity_enabled": false
}
```

Enable runtime identity only for deployments whose Hermes gateway provides a
stable platform user identity. Leave it `false` for anonymous or session-only
gateways.

The flags are independent so an operator can roll back one behavior without
discarding stored evidence. Claim rows retain source-turn, validity, scope,
modality, extractor-version, and conflict metadata. Retrieval resolves those
rows before injection, while recent raw turns are labelled as unprocessed and
remain private to the active agent/session. The Hermes lifecycle hooks also
cover queued prefetch, built-in writes, context compression, delegation,
session end, backup paths, and session switching.

Explicit legacy overrides remain supported. Run the evaluation and health gates
in [`docs/remnant-v0.2.1-leadership-plan.md`](docs/remnant-v0.2.1-leadership-plan.md)
before changing a production deployment that has pinned any of these flags.

Named operational profiles are available for controlled rollout:

```bash
python -m remnant.maintenance config-profile --home ~/.hermes/profiles/default \
  --name claim_aware                 # preview
python -m remnant.maintenance config-profile --home ~/.hermes/profiles/default \
  --name claim_aware --yes           # apply
```

The other profiles are `claim_aware_shadow` and `legacy`. The command reports
every changed field and preserves unrelated settings.

### Type check (optional)

```bash
mypy remnant
```

### Batch re-extraction

After changing the entity extractor, re-extract all memories:

```bash
cp ~/.hermes/remnant/remnant.db ~/.hermes/remnant/remnant.db.backup
python -m remnant.reextract --dry-run      # preview
python -m remnant.reextract --batch 100     # run
python -m remnant.classify_relations --yes  # reclassify typed relations
python -m remnant.calibrate_trust           # recalibrate trust scores
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

The shared SQLite database at `~/.hermes/remnant/remnant.db` is returned by the provider's backup path list. Include it in normal Hermes workspace backups. Per-profile config files at `hermes_home/remnant.json` should also be backed up.

---

## Roadmap / not in scope

- BSL Hub integration as a future data source.
- Entity community detection (deferred until graph traversal needs it).
- Email / feed / sensor indexing.
- Web dashboard.
- GLiNER model fine-tuning on fleet-specific vocabulary (fleet agent names, homelab services).
- Self-tuning prefetch: use prefetch_stats data to adjust deadline/budget/expand depth based on observed hit rates. Deferred until sufficient stats are collected (~1 week of production data).
- Curation loop: surface never-seen memories proactively to keep the corpus fresh. Depends on a reward signal (implicit reuse detection).

---

## License

Internal BlacksiteLab project. Not licensed for public distribution.
