# Hermes Remnant

Remnant is the long-term memory engine for the [Hermes Agent](https://hermes-agent.nousresearch.com) fleet. It is implemented as a Hermes memory-provider plugin that stores durable facts, observations, documents, and threads in a local SQLite database, retrieves them with hybrid keyword + vector + graph search, and proactively injects relevant context before each LLM call.

This repo contains both the **Hermes plugin** (`remnant/`) and the **test suite** that exercises every phase of the system.

**Repository:** [https://github.com/Blacksitelab/hermes-remnant](https://github.com/Blacksitelab/hermes-remnant)  

## What you need to run it

- Python 3.10+ (httpx and pyyaml are installed automatically).
- An embedding model: any Ollama or OpenAI-compatible endpoint serving
  `nomic-embed-text` (768-dim) by default.
- An extraction model: a cheap local LLM (qwen3:8b-class, or any
  OpenAI-compatible endpoint). Remnant uses it to extract facts, reflect,
  and summarize conversation history. If it is down, Remnant keeps working —
  extraction just pauses and keyword search still functions.
- Optional: an Obsidian vault to index as documents. Without a vault,
  Remnant is fully functional for conversation memory.
- Optional: GLiNER (`urchade/gliner_small_v2`) for better entity
  extraction; a regex fallback is built in.


---

## What Remnant does

- **Stores durable facts** extracted from conversation turns, vault notes, and imports.
- **Dedupes aggressively** — the same fact is never stored twice; duplicate hits increment a `seen_count`.
- **Filters transient state** — percentages, current timestamps, and words like *currently* / *now* are rejected.
- **Scores trust** — every memory has a `trust_score` calibrated by source quality, confidence, verification status, and engagement. Trust scores influence search ranking and decay over time.
- **Self-edits** — agents can update, merge, forget, share, unshare, and score memories through tools.
- **Searches three ways** — BM25 keyword, cosine vector similarity, entity-graph traversal, plus a hybrid RRF fusion.
- **Proactively injects context** via Hermes' `prefetch()` hook, with a hard deadline, token budget, and diff-based suppression.
- **Recalls conversation history explicitly** with `memory_history`, using the profile-local Hermes archive for dated, relative, topic, decision, correction, and unresolved-work questions.
- **Indexes an Obsidian vault when configured** as `document` memories, respecting workspace exclusions, frontmatter, locked notes, and per-agent profile scopes.
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
| Storage | SQLite + FTS5 | Zero dependency, single-file, WAL on fixed engines, DELETE fallback, fast |
| Embeddings | `nomic-embed-text` via local Ollama (768-dim) | Small, CPU/GPU-capable, simple HTTP API |
| Entity extraction | `urchade/gliner_small_v2` via GLiNER (CPU) | Purpose-built NER, ~400 MB, millisecond inference, typed entities |
| Extraction / rerank / reflect | A local LLM such as `qwen3:8b` via an OpenAI-compatible API | Cheap extraction and summarization |
| Dream loop | Cloud model (`deepseek-v4-flash:cloud` by default) | Overnight quality, latency irrelevant |
| Framework | Hermes memory-provider plugin | Registers `sync_turn`, `prefetch`, tool schemas |
| Concurrency | `ThreadPoolExecutor` | Proper shutdown semantics |

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/Blacksitelab/hermes-remnant.git ~/hermes-remnant
cd ~/hermes-remnant
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
ln -s ~/hermes-remnant/remnant ~/.hermes/plugins/remnant
```

The plugin manifest is `remnant/plugin.yaml`.

### 4. Configure Hermes to use Remnant

In your Hermes `config.yaml`:

```yaml
memory:
  provider: remnant
```

Run `hermes memory setup` to configure endpoints and agent id.

Remnant uses one SQLite file at `~/.hermes/remnant/remnant.db` (override with
`REMNANT_DB_HOME`), with strict ownership enforced by the memory provider. Each
profile can retrieve and modify only its own memories, including vault notes.
`shared` and `fleet` remain legacy labels; they do not grant another profile
access. Named profiles use their directory name as their storage identity;
configurations under `hermes_home/remnant.json` remain independent.

For upgrades from split profile stores, use the explicit
[offline fleet recovery procedure](docs/profile-recovery.md). Schema migration
alone does not combine databases or change running service paths.

Version 0.3.0 migrates vault tracking to `(profile, path)` keys without changing
existing memory ownership. Runtime identity v2 also includes the profile. Older
runtime identity v1 records remain preserved but require an explicit operator
mapping before reuse. Configured-owner mode retains its existing keys.
Keep a SQLite backup before updating: rolling back across schema 16 requires
restoring that database backup along with the earlier code. Direct database
Direct database access and operator maintenance commands remain administrative capabilities.

### Hygiene report and reviewed apply

The hygiene workflow is report-first and never chooses a disposition. Point it
at an approved snapshot; it opens SQLite read-only, excludes locked notes, and
writes a redacted CSV (or JSON) plus a hashed companion manifest:

```bash
python -m remnant.hygiene report --db SNAPSHOT --agent OWNER --output REPORT.csv
```

An operator may construct an explicit version-1 approval manifest from that
persisted report and its companion manifest. Apply re-reads the report content
and hash, verifies operation membership, and binds the report content hash,
schema identity, store binding, owner, human approver, reason, and every
operation's expected row fingerprint. Apply only to the same store, and use
dry-run before applying. Operations stop at the first conflict; receipts
contain IDs/statuses only and are kept in a 0700 directory with 0600 files.
Locked, foreign, changed, or unknown rows are not mutated. A live WAL is read
through a disposable private capture; closed WAL snapshots with no sidecars
are opened immutably, and unsupported journal layouts fail without writes.

```bash
python -m remnant.hygiene apply --db APPROVED_DB --agent OWNER \
  --manifest APPROVAL.json --dry-run
python -m remnant.hygiene apply --db APPROVED_DB --agent OWNER \
  --manifest APPROVAL.json --apply
```

The classifier recognises a finite set of high-confidence literal credential
forms. Pointer prose is a review finding, not an automatic forget decision;
credential revocation and human disposition remain separate operator actions.
The historical private extractor mentioned in the clearance report is not part
of this checkout; `remnant.hygiene` is the small reviewable replacement.

### SQLite journal mode and the WAL-reset race

Upstream SQLite documents a WAL-reset race affecting engines 3.7.0 through
3.51.2, fixed in 3.51.3 and the backport lines 3.44.6 (3.44.x) and 3.50.7
(3.50.x): <https://sqlite.org/wal.html#walresetbug>. The race needs multiple
connections writing and checkpointing one WAL file, and a passing
`integrity_check` does not prove safety against it. Remnant has not reproduced
corruption.

Remnant therefore chooses the journal mode from the *linked SQLite engine
version* (`configure_sqlite_journal` in `remnant/db.py`, covering `RemnantDB`,
`reextract`, `calibrate_trust` and `classify_relations`):

- Recognized fixed upstream engines (>=3.51.3, 3.44.6+, 3.50.7+) keep
  WAL with `synchronous=NORMAL`.
- Affected or unverified engines use `DELETE` with `synchronous=FULL` for
  fresh/non-WAL files. Vendor builds that keep older version numbers are
  treated as unverified, not proven vulnerable: `DELETE` works without
  upgrading, and no distribution or package metadata is trusted. Recognizing a
  specific vendor build requires separately reviewed exact-build evidence; no
  force-WAL bypass exists.
- An existing WAL database on an affected or unverified engine is refused with
  an actionable `RuntimeError` *before* any write, migration, checkpoint or
  mode change. There is no automatic live WAL-to-DELETE conversion, because
  acquiring SQLite's conversion lock is not a substitute for a coordinated
  process shutdown while older clients may re-enable WAL.
- Backup, restore and recovery destinations are private files this process
  owns, so they convert to `DELETE` instead of refusing; shared writable
  sources are never checkpointed.

Operator transition guidance (documentation only, not automated): stop every
client that shares the database, prevent legacy writers from re-enabling WAL,
take a verified backup, then convert or switch to a patched runtime on a
disposable copy first. Never delete `-wal`/`-shm` sidecars as a workaround.
Expect `DELETE`/`FULL` to cost throughput compared with WAL/NORMAL; no specific
level is promised without measuring your workload. Conversion timing, runtime
choice and any cleanup of previously escaped vault data remain operator
decisions.

Vault indexing resolves each candidate against the canonical vault root before
hashing, reading, embedding or entity extraction. Escapes, broken links,
symlink loops and excluded-scope aliases are rejected, and a stable in-vault
symlink is indexed only when both its entry path and its canonical target pass
the exclusion and profile-scope checks — deduplicated under the canonical
relative path. This is not a race-free sandbox against an adversary
concurrently replacing canonical ancestor directories.

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
provider.sync_turn("Sam prefers dark mode.", "Noted.", session_id="session-1")
result = provider.handle_tool_call("memory_search", {"query": "Sam preference"}, session_id="session-1")
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
├── tools.py                 # Tool schemas and dispatch (search/history/store/edit/graph/reflect/import/thread)
├── edit.py                  # memory_edit actions + audit logging
├── prefetch.py              # Proactive prefetch with deadline/budget/dedup + entity-graph query expansion
├── history.py               # Read-only Hermes archive adapter and bounded historical recall
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
| `on_session_end()` | Coalesces an asynchronous historical-summary job for the ended session |
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
| `memory_history` | Recall source-linked historical conversation evidence by time, topic, or session |
| `memory_thread` | Create, update, resolve, list, or sweep stale threads |
| `memory_import` | Import from `vault`, `memory_store`, or `hindsight` |

---

## Historical conversation recall

`memory_history` is explicit rather than automatic: use it for questions about
what was discussed, a project decision, a correction, an abandoned proposal, or
unresolved alternatives. Examples:

```json
{"query":"the deployment rollback decision","synthesize":false}
{"start":"2024-06-14","timezone":"Pacific/Auckland"}
{"relative":"yesterday","timezone":"America/New_York"}
{"query":"printer project","cursor":"<next_cursor from the previous result>"}
```

`start` is inclusive and `end` is exclusive. A date-only `start` selects that
local calendar day; a datetime start needs an explicit end. `today`,
`yesterday`, and `last_week` use local civil midnights in the selected IANA
zone, while `6h`, `2d`, and `1w` are rolling elapsed durations. Dates must have
a four-digit year; naive datetimes and unknown zones are rejected. DST
short/long days are handled as calendar boundaries, not as 24-hour arithmetic.

The tool reads only the active profile's Hermes `state.db`. It returns native
session/message links, uncertainty, and coverage. Interactive sessions are
preferred; `cron` history is labelled as automated. Hidden sessions, kanban,
subagent/tool scaffolding, rewound inactive rows, hidden display rows, and
compressed-summary rows are excluded. Runtime-identity mode additionally
requires known ownership in Remnant turns; unknown historical ownership fails
closed. Parent/delegation links do not grant access to another session.

Coverage is deliberately bounded and may be partial: at most 200 date-session
candidates, 20 sessions, 100 topic hits, 80 raw messages, and 2,000 characters
per excerpt are considered in one call. Continue with `next_cursor` when
`has_more` is true. Search terms are lexical hints, not semantic guarantees;
an empty accessible result means no evidence was found in the searched
coverage, not that the topic was never discussed. Model synthesis is optional,
uses at most one call, and every paraphrase must cite supplied source records;
invalid output falls back to deterministic source-linked excerpts. Archived
text is untrusted data and never instructions.

Summaries run asynchronously after session boundaries and are only a bounded
cache, not a second transcript store: no summary embeddings, no new ordinary
turn model call, 1,000 cache rows, 64 queued jobs per owner, three attempts per
source version, and 20 summary attempts per owner per UTC day. A missing,
stale, unavailable, or dead-letter summary falls back to raw archive evidence.
The request path has a 3-second archive-query ceiling, 5,500 estimated model
input tokens including framing/provenance, 1,024 output tokens, and a 4,000
token serialized result ceiling. Disable new history work with
`history_enabled: false`; the Hermes archive is never modified. Preserve a
database backup before schema rollback, since older Remnant versions do not
accept a newer schema.

Configuration defaults are `history_enabled: true`,
`history_timezone: UTC`, and `history_summary_enabled: true`.

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
| `owns` | alex → remnant | person owns project/tool |
| `uses` | sam → ollama | person uses tool/service |
| `created` | sam → skill | person created project/service |
| `depends_on` | remnant → sqlite | project/service depends on tool |
| `monitors` | atlas → fleet | person monitors project/service |
| `manages` | alex → server01 | person manages place/organization |
| `interacts_with` | atlas → sam | person interacts with person |
| `references` | vault → remnant | project/service references project |
| `part_of` | hub → examplecorp | entity is part of organization |
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

The SQLite database is **shared** across all profiles at `~/.hermes/remnant/remnant.db` (override with the `REMNANT_DB_HOME` env var). Config and memory access are profile-scoped; only the database file is shared.

Vault indexing is optional: set `vault_path` per profile (or with the `REMNANT_VAULT_PATH` env var before constructing a `RemnantConfig`) to index an Obsidian vault. Without it Remnant runs as conversation memory and vault import no-ops cleanly.

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

### Model-backed historical claim backfill

The legacy `reextract_claims` command is a deterministic projection from stored
entity/tag metadata. It does not call a language model. For historical fact
memories that need structured subject, predicate, object, temporal, scope, and
modality fields, use the model-backed pass instead:

```bash
# Shadow mode, no database writes
python -m remnant.model_backfill --home ~/.hermes/profiles/myprofile --home ~/.hermes --limit 20

# Apply validated projections, preserving memories and writing audit entries
python -m remnant.model_backfill --home ~/.hermes/profiles/myprofile --home ~/.hermes --batch-size 8 --yes
```

The model pass updates the unique claim projection in place, preserves claim
status and reconciliation state, and records before/after claim rows under the
`claim_model_backfill` audit action. It defaults to the configured extraction
endpoint and model, so it can use the configured local model deployment without
changing Hermes configuration.

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
python -m remnant.maintenance migrate-default-agent --agent myprofile  # dry run
python -m remnant.maintenance migrate-default-agent --agent myprofile --yes
```

Run the scale-envelope harness separately from unit CI before changing the
retrieval implementation:

```bash
python -m remnant.evaluation.scale --sizes 5000 --probes 5 --output scale-report.json
```

### Retrieval budgets and cache retention

Semantic retrieval streams compact float32 vectors and keeps only the best
candidates. `semantic_scan_limit: 0` searches every eligible memory, including
old facts; a positive value explicitly caps the scan. Each semantic result must
meet `min_semantic_score` and match the configured embedding model/dimensions.
Foreground prefetch includes SQLite lock waits and scans in its time budget,
reserving time for a keyword fallback. Query vectors use a bounded RAM cache;
queued context is marked delivered only when Hermes consumes it.

Extraction maintenance retries missing or incompatible vectors independently of
vault file hashes. A changed note immediately loses its old vector if its new
embedding fails. Maintenance retains at most `embedding_cache_max_entries`
(default 10,000) document-cache entries for `embedding_cache_max_age_days`
(default 30), flushes bounded telemetry in batches, and prunes old diagnostics.
Echo retention also runs periodically. Raw turns and source memories are retained.
SQLite reuses freed pages; these limits do not promise an immediate file shrink.

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
described in [docs/evaluation.md](docs/evaluation.md) before changing a
deployment that has pinned any of these flags.

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

- Entity community detection (deferred until graph traversal needs it).
- Email / feed / sensor indexing.
- Web dashboard.
- GLiNER model fine-tuning on deployment-specific vocabulary (agent names, homelab services).
- Self-tuning prefetch: use prefetch_stats data to adjust deadline/budget/expand depth based on observed hit rates. Deferred until sufficient stats are collected (~1 week of real traffic).
- Curation loop: surface never-seen memories proactively to keep the corpus fresh. Depends on a reward signal (implicit reuse detection).

---

## License

MIT — see `pyproject.toml` (`license = MIT`).
