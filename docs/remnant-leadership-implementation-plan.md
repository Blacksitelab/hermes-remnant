# Remnant leadership implementation plan

Status: implementation handoff

Release branch note: the first integrated implementation is tracked in
`codex/remnant-release`; the feature flags remain opt-in until deployment
evaluation gates are run against a representative Hermes corpus.

Repository: `Blacksitelab/hermes-remnant` only

Baseline: `85926f74dfac454cde6ac65385e359365059344f`

Schema baseline: `SCHEMA_VERSION = 14` (release branch implementation)

Target: make Remnant the most correct, efficient, private, and operationally
reliable memory provider for Hermes Agent.

## How to use this document

This plan is written for an implementation model that should make narrow,
reviewable changes without redesigning the project opportunistically.

Rules for the implementation model:

1. Work only in `Blacksitelab/hermes-remnant`. Do not open issues, branches,
   commits, or pull requests in any external repository.
2. Implement one work package per branch and pull request. Do not combine work
   packages unless this document explicitly permits it.
3. Start every branch from the latest `main` and use the `codex/` branch prefix.
4. Preserve compatibility with existing databases and existing Remnant config.
5. Put behavior-changing retrieval work behind feature flags until it passes
   the evaluation gates in this document.
6. Never silently delete, re-scope, or reinterpret existing memories during a
   migration. Backfills must be explicit, restartable, and dry-run capable.
7. Keep `sync_turn()` non-blocking and keep useful prefetch within the configured
   deadline. Do not solve correctness by moving unbounded LLM work into either
   foreground path.
8. Treat memory text as untrusted data in every prompt and context block.
9. Preserve authorization before ranking: inaccessible candidates must never
   consume top-k positions or enter LLM prompts.
10. Add tests before enabling each new default. Run the full validation matrix
    at the end of every package.

## Current baseline

Already implemented and not to be rebuilt:

- transactional turn plus extraction-job creation;
- retry, stale-job recovery, and dead-letter extraction states;
- fail-closed vault scanning;
- normalized Ollama/OpenAI-compatible LLM requests;
- centralized visibility/profile-scope policy hardening;
- profile-scope enforcement for legacy/import document rows;
- BM25, exact cosine, RRF, and entity-graph candidate retrieval;
- bounded prefetch with BM25-first fallback when embeddings stall;
- prompt-injection fencing and context-tag sanitization;
- explicit memory edit/merge/forget/share/feedback tools;
- immutable backing memories, audit logs, preliminary claims, entity graph,
  vault indexing, threads, dreams, and imports.

Known baseline gaps this plan must close:

- Remnant has no end-to-end answer-quality benchmark and cannot substantiate a
  claim that it is the best provider.
- Claims are projected but do not resolve automatic prefetch results.
- Claim validity timestamps and conditions are mostly unused.
- Trust, verification, and confidence are stored but normal hybrid ranking uses
  fixed source multipliers instead.
- Temporal language is rejected too broadly during extraction.
- Contradiction detection cannot distinguish updates, false contradictions,
  conditional alternatives, compatible facts, and unresolved disputes.
- Prefetch renders flat bullets without validity, provenance, or uncertainty.
- Async extraction creates a read-after-write gap.
- The current Hermes lifecycle hooks `queue_prefetch`, `on_memory_write`,
  `on_pre_compress`, `on_delegation`, and `on_session_end` are not implemented.
- Runtime user/profile/workspace identity is not fully represented in Remnant's
  effective actor scope.
- Lifecycle writes do not atomically maintain every derived projection.
- There is no repository CI workflow and `is_available()` always returns true.

## Definition of leadership

Do not declare the project complete based on feature count. Completion requires
all of the following measured outcomes:

| Dimension | Required gate |
|---|---|
| Conflict correctness | Macro score at least `0.52` over dynamic, static, and conditional cases, with each category non-negative and no category below `0.35` |
| Wrong answers | Fewer than `10%` confidently wrong/contradictory answers on the conflict suite |
| Retrieval | Recall@5 at least `0.85`, MRR at least `0.70`, and no regression greater than `0.02` from the lexical/vault baseline |
| Read-after-write | `100%` of eligible immediately preceding facts available to the next turn through committed claims or the pending overlay |
| Isolation | Zero cross-user/private-scope leaks in the complete authorization matrix |
| Context efficiency | Median injected memory context below `1,000` tokens and p95 below the configured `2,000` token ceiling |
| Prefetch latency | Keyword/degraded p95 below `100 ms`; full hybrid p95 below `500 ms` on the supported production-sized corpus |
| Durability | No lost acknowledged turns across restart, timeout, dead-letter, migration, or bounded shutdown tests |
| Operability | CI green on supported Python versions; Ruff and build clean; migration and rollback tests green |

If the answer model or evaluation judge is nondeterministic, run at least three
repeats for any comparison within `0.03` and report the mean plus range.

## Target architecture

The final automatic recall path should be:

```text
runtime identity and authorization scope
    -> pending recent-turn overlay
    -> BM25/vector/graph candidate discovery
    -> claim and evidence enrichment
    -> temporal/conditional/conflict resolution
    -> relevance plus bounded quality ranking
    -> version grouping and diversity selection
    -> provenance-aware compact context
    -> token/deadline enforcement
```

Backing memories remain immutable evidence. Structured claims become the
authority for deciding which interpretation is currently applicable. No claim
resolver is allowed to erase evidence.

## Dependency order

```text
WP-01 evaluation foundation
|-- WP-02 Hermes contract and runtime identity
|-- WP-03 temporal extraction and schema
|   `-- WP-04 conservative claim reconciliation
|       `-- WP-05 claim-aware retrieval and ranking
|           `-- WP-06 conflict-aware context compiler
|-- WP-07 read-after-write overlay
|-- WP-08 transactional lifecycle and relation evidence
|-- WP-09 efficiency and scaling
`-- WP-10 release engineering and documentation
```

WP-01 is first. WP-03 through WP-06 must remain sequential. WP-02, WP-07,
WP-08, and the CI portion of WP-10 may proceed independently after WP-01, but
each still needs its own branch and review.

---

## WP-01: Build the Remnant quality laboratory

Suggested branch: `codex/remnant-evaluation-foundation`

Suggested PR title: `test: add Remnant answer-quality and conflict evaluation`

### Objective

Create reproducible evidence for memory correctness before changing extraction
or ranking. Keep all datasets, runners, and reports in this repository.

### Files

- extend `remnant/evaluate.py`;
- add `remnant/evaluation/` modules rather than allowing one large script;
- add `tests/fixtures/evaluation/` for small deterministic fixtures;
- add `evaluation/cases/` for versioned non-secret case files;
- add `evaluation/baselines/` for committed summary JSON, not raw private data;
- add `docs/evaluation.md`.

### Case format

Use JSONL with a schema version. Each scenario should contain:

```json
{
  "schema_version": 1,
  "case_id": "dynamic.preference.editor-theme.001",
  "category": "dynamic",
  "persona": "test-user-a",
  "sessions": [
    {
      "observed_at": "2026-01-01T12:00:00Z",
      "turns": [{"user": "I prefer light mode.", "assistant": "Noted."}]
    },
    {
      "observed_at": "2026-02-01T12:00:00Z",
      "turns": [{"user": "I switched to dark mode.", "assistant": "Understood."}]
    }
  ],
  "query": {"text": "Which editor theme do I prefer?", "at": "2026-02-02T12:00:00Z"},
  "expected": {
    "answer_contains": ["dark"],
    "answer_must_not_contain": ["light"],
    "supporting_memory_labels": ["new-preference"]
  }
}
```

### Required categories

Create at least 20 cases in each category before changing defaults:

- dynamic update;
- static false contradiction;
- conditional fact;
- stable fact over long time;
- historical query;
- unresolved conflict;
- duplicate/paraphrase;
- distractor entity;
- immediate next-turn recall;
- vault passage versus conversational fact;
- private/shared/fleet scope;
- profile and runtime-user isolation.

Include explicit adversarial cases where the newest statement is false, where
two conditions are both valid, and where the query asks about a past state.

### Runner layers

Implement three separately runnable layers:

1. `retrieval`: score returned memory/claim IDs without an answer LLM;
2. `context`: inspect the exact text Remnant would inject;
3. `answer`: optionally call a configured answer model and deterministic or
   judge-assisted scorer.

The default CI suite runs only deterministic retrieval/context fixtures. The
answer suite must be opt-in because it uses model tokens.

### Metrics

Write JSON summaries containing:

- recall@1/3/5;
- MRR and nDCG@5;
- context precision;
- answer correct/partial/blank/wrong;
- macro and per-category scores with wrong answer scored `-1`;
- stale claim exposure;
- unresolved conflict exposure;
- duplicate top-k occupancy;
- injected tokens;
- extraction and answer-model usage when reported;
- ingest, queue-drain, search, resolution, and formatting latency;
- configuration, commit SHA, schema version, model names, and random seed.

### Acceptance tests

- schema validation rejects unknown categories and malformed timestamps;
- the same deterministic run produces byte-stable summary JSON after removing
  explicitly variable timing fields;
- a deliberately stale-only retriever fails the dynamic cases;
- a newest-always-wins resolver fails static contradiction cases;
- an unscoped retriever fails isolation cases;
- evaluator commands never mutate the source production database;
- reports distinguish retrieval failure from evidence-utilization failure.

### Rollback

Evaluation code is additive. Revert the package if it affects provider runtime
imports or normal installation size. Runtime code must not import optional
evaluation dependencies.

---

## WP-02: Complete the Hermes contract and runtime identity model

Suggested branch: `codex/hermes-contract-parity`

Suggested PR title: `feat: complete Hermes lifecycle and runtime identity support`

### Objective

Use the current Hermes hooks while preserving compatibility with older Hermes
versions. Prevent a shared Remnant database from conflating users, profiles,
workspaces, sessions, primary agents, cron runs, and subagents.

### Files

- `remnant/__init__.py`;
- `remnant/config.py`;
- `remnant/scope.py`;
- `remnant/db.py` only if identity mappings need durable storage;
- new `remnant/identity.py`;
- new `tests/test_identity.py` and `tests/test_hermes_contract.py`.

### Effective identity

At `initialize()` capture, normalize, and retain:

- configured `agent_id`;
- `agent_identity`;
- `agent_workspace`;
- `platform`;
- `user_id` and `user_id_alt`;
- `agent_context`;
- session and parent-session IDs.

Create an `EffectiveIdentity` value object. Derive stable actor/storage keys from
explicit components; do not concatenate ambiguous raw strings. Hash external
user IDs before putting them in logs. Provide a configured alias map for users
who intentionally want multiple platform IDs to share one identity.

Default policy:

- primary agents may read/write their effective private scope;
- subagent and cron contexts do not automatically extract user-profile facts;
- delegation results arrive through the parent's `on_delegation()` hook;
- missing user identity never broadens access;
- legacy memories retain their current configured-agent behavior until an
  explicit migration is run.

### Hooks

Implement:

- `queue_prefetch()`: bounded per-session cached prefetch for the next turn;
- `on_memory_write()`: mirror successful built-in add/replace/remove events with
  `write_origin`, session, platform, and actor provenance;
- `on_pre_compress()`: enqueue extraction of high-signal discarded context and
  return a compact preservation hint;
- `on_delegation()`: store the task/result as an observation of the parent, with
  explicit child-session provenance;
- `on_session_end()`: flush eligible turn buffers and optionally trigger only
  bounded local maintenance;
- enhanced `on_session_switch()`: invalidate caches by old/new/parent identity
  and handle rewind without duplicating writes.

Use duck typing and default no-op behavior so import against older Hermes does
not fail.

### Prefetch cache

Cache keys must contain:

- normalized query hash;
- session ID;
- effective viewer identity and visibility scope;
- profile scope hash;
- embedding model;
- memory-generation counter.

The cache must be size bounded, TTL bounded, cancelled on shutdown, and
invalidated after relevant memory/lifecycle writes. Never reuse a cached result
across users or broader scopes.

### Acceptance tests

- two gateway users with identical queries cannot share private cache entries;
- unknown/missing user IDs cannot read another user's private memories;
- profile aliases merge only when explicitly configured;
- subagent and cron contexts do not pollute the primary user profile;
- built-in memory add/replace/remove mirrors exactly once;
- compression and delegation hooks retain provenance;
- a queued prefetch is invalidated after a write;
- all hooks remain safe when invoked before or after initialization;
- older Hermes base-class stubs still load the provider.

### Feature flags and rollback

Add `runtime_identity_enabled`, initially `false` for upgraded databases and
`true` for new installations after migration tests pass. Provide an identity
diagnostic command and dry-run mapping report. Roll back by disabling the flag;
do not delete newly stored scoped memories.

---

## WP-03: Introduce temporal and conditional claim extraction

Suggested branch: `codex/temporal-claim-extraction`

Suggested PR title: `feat: extract temporal and conditional claims`

### Objective

Preserve meaningful time and condition information instead of rejecting the
entire fact. This package changes extraction and schema, but does not yet make
claims authoritative in search.

### Schema migration

Increment `SCHEMA_VERSION`. Extend `claims` with additive columns:

- `observed_at TEXT NOT NULL`;
- `event_at TEXT`;
- `valid_from TEXT`;
- existing `valid_to TEXT` retained;
- `scope_type TEXT` such as `global`, `work`, `home`, `project`, `device`,
  `person`, `location`, or `custom`;
- `scope_value TEXT`;
- `modality TEXT` such as `asserted`, `inferred`, `hypothetical`, `negated`;
- `resolution_status TEXT` such as `active`, `superseded`, `contradicted`,
  `unresolved`, `historical`;
- `conflict_type TEXT` nullable: `update`, `contradiction`, `conditional`,
  `compatible`, `duplicate`, `unresolved`;
- `extractor_version TEXT`;
- `source_turn_id INTEGER` where available.

Prefer constrained text values enforced by service validation. Avoid a table
rewrite solely to add SQLite CHECK constraints to an existing production table.

### Timestamp semantics

- `observed_at`: when the source statement was observed by Remnant;
- `event_at`: explicit time described by the statement, if any;
- `valid_from`: earliest time the claim is known to apply;
- `valid_to`: exclusive end of applicability;
- no timestamp may default to model-invented precision;
- if only a date is known, retain date precision in qualifiers instead of
  fabricating a time.

Pass the source turn/session timestamp into extraction. Do not use wall-clock
time inside pure parsing tests.

### Extraction schema

Replace the flat fact response with versioned strict JSON containing:

- `fact` for human-readable immutable evidence;
- `subject`, `predicate`, `object`;
- `conditions`;
- temporal fields;
- modality and confidence;
- typed entities;
- `durability`: `durable`, `temporary_but_relevant`, or `discard`.

Keep one compatibility parser for the old response shape. Stamp every result
with an extractor version.

### Transient filtering

Replace the broad regex veto with field-aware rules:

- reject pure telemetry such as a fleeting percentage with no durable meaning;
- retain genuine state transitions and preference changes;
- retain explicit dates as claim metadata;
- retain time-bounded commitments when future recall is useful;
- keep hypothetical statements out of active truth while preserving them only
  when explicitly configured.

Examples:

- `CPU is currently at 37%` -> discard;
- `I now prefer dark mode` -> durable update candidate;
- `I am in Wellington until Friday` -> temporary but relevant, bounded validity;
- `If I am at work, use email` -> conditional claim;
- `Maybe I will move to Sydney` -> hypothetical, not active truth.

### Backfill

Do not automatically re-extract every existing memory during migration. Add a
separate command:

```text
python -m remnant.reextract_claims --dry-run --batch 100
python -m remnant.reextract_claims --yes --batch 100
```

It must be resumable, record extractor version, skip already-current claims,
and never alter backing memory content.

### Acceptance tests

- temporal update, historical date, conditional preference, hypothetical, and
  telemetry examples produce expected structured output;
- an invalid model timestamp becomes null plus a bounded diagnostic;
- source timestamps survive queue retry and restart;
- old extractor JSON remains readable;
- migration is idempotent and interruption safe;
- dry-run backfill writes nothing;
- no active claim is created from hypothetical text by default.

### Feature flag and rollback

Add `structured_claim_extraction_v2`, default `false` for upgraded installs.
Disabling it returns to the old extractor while retaining additive columns.

---

## WP-04: Reconcile competing claims conservatively

Suggested branch: `codex/conservative-claim-reconciliation`

Suggested PR title: `feat: reconcile updates, contradictions, and conditions`

### Objective

Replace newest-value supersession and antonym-only decisions with an explicit,
auditable reconciliation service.

### Files

- add `remnant/reconcile.py`;
- refactor `remnant/claims.py` into parsing/projection helpers only;
- call reconciliation from `remnant/ingest.py`;
- extend `remnant/db.py` with bounded claim-candidate queries;
- add `tests/test_reconcile.py`.

### Candidate generation

For each new claim, find active/historical competitors using:

- exact normalized subject plus predicate;
- entity identity and aliases;
- overlapping condition scope;
- bounded semantic predicate/object similarity only when necessary.

Candidate generation must be deterministic, authorized, and capped. Do not send
the full database or unrelated private memories to a model.

### Decision ladder

Use cheapest reliable rules first:

1. exact semantic duplicate -> `duplicate`;
2. disjoint condition scopes -> `conditional`, retain both;
3. explicit transition language plus same scope -> `update` candidate;
4. explicit correction with clear reference -> `update` or `contradiction`
   candidate depending on evidence;
5. compatible predicates/objects -> `compatible`;
6. clear negation/antonym -> contradiction candidate;
7. otherwise -> `unresolved`.

An optional LLM classifier may decide only among a supplied bounded candidate
set and must return strict JSON with rationale labels, not free-form lifecycle
instructions. Local validation owns the final state transition.

### Evidence rules

- never supersede established evidence solely because a conflicting statement
  is newer;
- a high-confidence explicit change may close the predecessor validity window;
- a false contradiction keeps both evidence records and marks the dispute;
- conditional claims remain simultaneously active in their scopes;
- unresolved conflicts remain visible as unresolved and must not be flattened
  into certainty;
- verification, corroboration, source type, explicit correction language, and
  repeated observations influence resolution but are stored separately from
  temporal validity.

### Audit

Every reconciliation writes an audit entry containing:

- new and candidate claim IDs;
- decision type;
- rule/classifier version;
- evidence features used;
- confidence;
- resulting validity/status transitions;
- actor and source turn.

Do not store full prompts or secrets in audit details.

### Acceptance tests

- `light -> explicitly switched to dark` closes light and activates dark;
- a later false statement does not automatically replace a repeatedly verified
  stable fact;
- work/home preferences remain active under separate conditions;
- identical paraphrases reinforce one claim without top-k duplication;
- ambiguous statements produce `unresolved`;
- historical evidence remains queryable after supersession;
- repeated reconciliation is idempotent;
- classifier timeout leaves claims unresolved and loses no evidence;
- an unauthorized candidate is never loaded or mentioned in classifier input.

### Feature flag and rollback

Add `claim_reconciliation_enabled`, default `false`. Shadow mode should record
proposed decisions without changing active status. Enable state transitions
only after shadow evaluation beats the baseline and manual review finds no
unsafe supersessions.

---

## WP-05: Make retrieval claim-aware and quality-aware

Suggested branch: `codex/claim-aware-ranking`

Suggested PR title: `feat: resolve and rank claims before context selection`

### Objective

Keep existing retrieval lanes for candidate discovery, then resolve applicability
and quality before top-k selection.

### Files

- add `remnant/resolve.py`;
- add `remnant/ranking.py`;
- modify `remnant/search.py`;
- add batched claim/evidence fetches in `remnant/db.py`;
- expose score components in `remnant/evaluate.py` and tool diagnostics;
- add `tests/test_claim_ranking.py`.

### Query interpretation

Derive, without an LLM where possible:

- query time: current, explicit historical date, or unknown;
- condition hints: work/home/project/device/person/location;
- requested subject/entities;
- current-state versus history intent.

An optional bounded query rewrite may be evaluated later. It must never be a
mandatory network call for basic recall.

### Applicability filtering

Before final top-k:

- discard memories forbidden by authorization;
- exclude forgotten backing memories;
- prefer claims valid at query time;
- exclude superseded claims from current-state queries unless needed to explain
  a transition;
- retain superseded claims for historical queries;
- require compatible condition scope when the query specifies one;
- group unresolved competitors rather than selecting one as truth.

### Ranking formula

Keep components separate and observable:

```text
final = relevance * applicability * bounded_quality * diversity_adjustment
```

Requirements:

- `relevance` remains BM25/vector/RRF based;
- `applicability` encodes time, condition, lifecycle, and query intent;
- `bounded_quality` combines trust, confidence, verification, corroboration,
  source prior, and evidence count in a narrow range such as `0.80..1.20`;
- `diversity_adjustment` prevents duplicate paraphrases and multiple versions of
  one claim from consuming the whole context;
- generic age must not lower stable identity/preference truth;
- trust decay must not run as a search side effect;
- strong lexical/semantic relevance cannot be erased by a minor quality gap.

Do not hard-code final weights without a committed evaluation comparison.
Store weights in a versioned ranking profile and stamp the profile into reports.

### Result model

Return an internal structured result containing:

- backing memory IDs;
- claim ID and grouped historical/conflict IDs;
- relevance components;
- applicability explanation;
- quality components;
- provenance;
- final score and ranking profile version.

Keep the public compatibility result shape while adding optional diagnostics.

### Acceptance tests

- verified evidence beats an equally relevant low-confidence duplicate;
- a highly relevant low-trust result is demoted but not erased;
- current queries return current claims and historical queries return the claim
  valid at the requested date;
- conditional queries select the matching condition;
- unresolved conflicts become one grouped result;
- fixed-source multipliers no longer contradict the trust model;
- search performs zero writes;
- all authorization filters occur before candidate limit;
- evaluation gates show no lexical/vault regression greater than `0.02`.

### Feature flag and rollback

Add `claim_aware_ranking_enabled` and `ranking_profile`, defaulting to legacy.
Run legacy and candidate ranking side by side in evaluation/shadow diagnostics.
Rollback is configuration-only.

---

## WP-06: Compile conflict-aware, provenance-aware context

Suggested branch: `codex/evidence-context-compiler`

Suggested PR title: `feat: compile compact resolved memory context`

### Objective

Give Hermes the smallest sufficient, correctly qualified memory context instead
of unrelated flat bullets.

### Files

- add `remnant/context.py`;
- simplify formatting responsibilities in `remnant/prefetch.py`;
- reuse the compiler in `remnant/reflect.py` where appropriate;
- add `tests/test_context_compiler.py`.

### Output rules

Examples:

```text
- Current preference [m:abc123]: Kris prefers dark mode.
  Updated 2026-06-12; prior value: light mode.
```

```text
- Conditional preference [m:def456]: At work, Kris prefers email.
  At home, Kris prefers Signal.
```

```text
- Unresolved conflict [m:ghi789,m:jkl012]: established evidence says X;
  one later unverified statement says Y. Do not assume either is certain.
```

Requirements:

- include compact observation/validity time only when relevant;
- include condition and uncertainty;
- include short opaque memory references that explicit tools can use;
- never expose private filesystem paths or internal database details;
- group versions and conflicts into one context item;
- preserve untrusted-data fencing and sanitize nested context tags;
- prioritize resolved facts before history and unresolved background;
- truncate individual items safely instead of allowing one vault passage to
  block all later facts;
- use a real tokenizer when available and retain the conservative estimator as
  fallback.

### Token allocation

Allocate the configured budget across categories rather than greedily allowing
one category to consume everything. Initial evaluated policy:

- 60% current resolved claims;
- 20% query-relevant provenance/history;
- 15% unresolved conflict/conditions;
- 5% header and safety framing.

Unused category budget may flow to resolved claims. Do not make this allocation
default unless evaluation improves answer correctness.

### Acceptance tests

- old/current versions occupy one item;
- two conditions occupy one item;
- unresolved conflicts are never rendered as settled truth;
- provenance references map back to authorized memories;
- locked notes and unauthorized provenance remain redacted;
- prompt-injection strings cannot close the memory fence;
- output stays inside the exact token budget;
- evaluation shows reduced duplicate occupancy and improved answer accuracy.

### Feature flag and rollback

Add `resolved_context_enabled`, default `false`. Preserve the legacy flat
formatter for immediate rollback.

---

## WP-07: Guarantee read-after-write recall

Suggested branch: `codex/read-after-write-overlay`

Suggested PR title: `fix: make recent turns recallable before extraction completes`

### Objective

Preserve asynchronous extraction while ensuring that the next turn can recall
eligible information from the immediately preceding turn.

### Design

Add a bounded pending-turn overlay sourced from durable `turns` rows whose
extraction state is pending/running/retry-wait.

On prefetch:

1. fetch only recent pending turns for the effective identity and session;
2. apply a cheap local memory-worthiness/relevance gate;
3. include relevant snippets as explicitly unprocessed recent context;
4. remove them from the overlay once extraction completes;
5. deduplicate overlay text against committed memories and current messages.

Do not synchronously call the extraction LLM. A very small configurable wait
for an already-running job is acceptable only when it remains inside the
prefetch deadline and is proven useful by measurement.

### Schema/API

Prefer existing durable turn/extraction state. Add indexes or helper methods,
not a second queue, unless profiling proves necessary. Add a queue watermark so
tests and health output can distinguish committed-through turn IDs.

### Safety

- overlay scope follows effective runtime identity;
- exclude assistant tool output and system scaffolding unless explicitly
  classified as a durable user-approved fact;
- label overlay content as recent/unprocessed and lower its authority;
- never let retry/dead-letter text remain in overlay indefinitely;
- cap turns, characters, age, and total tokens.

### Acceptance tests

- a fact from turn N is available on turn N+1 while extraction is blocked;
- the overlay disappears after successful extraction;
- the same fact is not injected twice during the handoff;
- cross-session and cross-user pending turns do not leak;
- dead-letter work ages out with a diagnostic;
- prefetch remains within deadline when hundreds of old jobs exist;
- crash/restart preserves correct overlay behavior.

### Feature flag and rollback

Add `recent_turn_overlay_enabled`, initially shadowed in diagnostics. Rollback
by disabling it; durable turn and queue state remain unchanged.

---

## WP-08: Make lifecycle projections transactional

Suggested branch: `codex/transactional-memory-lifecycle`

Suggested PR title: `fix: make memory lifecycle and relation evidence atomic`

### Objective

Ensure create, edit, merge, supersede, forget, and visibility changes leave
memories, claims, embeddings, entity links, relations, and audit state mutually
consistent.

### Schema migration

Add `relation_evidence`:

- `relation_id`;
- `memory_id`;
- `claim_id` nullable;
- `strength`;
- `active`;
- timestamps;
- composite primary key and foreign keys.

If needed, add a durable lifecycle-job table for remote prerequisites that must
complete before mutation. Do not hold a SQLite transaction open during remote
embedding or LLM calls.

### Service

Add `remnant/lifecycle.py` as the only high-level write service for:

- create memory plus claim and entity evidence;
- update by creating a replacement version;
- merge after replacement is fully prepared;
- supersede;
- forget;
- visibility changes;
- verification and feedback changes.

Remote embeddings are computed and validated before the database transaction.
Inside the transaction perform only bounded database work. Any error rolls back
the whole mutation.

### Compatibility migration

Backfill relation evidence from current memory/entity links using a restartable,
dry-run command. Do not delete existing relations until the evidence model is
validated. Run old and derived relation counts side by side first.

### Acceptance tests

- fault injection at every lifecycle stage leaves originals active and all
  projections consistent;
- merging never supersedes inputs before replacement completion;
- forgetting the last evidence removes a relation from active traversal;
- forgetting one of several evidence memories preserves the relation;
- editing creates a new claim version and closes the correct predecessor;
- visibility changes cannot broaden derived content automatically;
- audit entries and state transition commit atomically;
- migration is idempotent, resumable, and rollback-safe.

### Rollback

Keep legacy relation rows during one compatibility release. A flag selects
evidence-derived traversal. Roll back to legacy traversal without deleting
evidence rows.

---

## WP-09: Optimize only measured bottlenecks

Suggested branch: `codex/measured-memory-efficiency`

Suggested PR title: `perf: add measured retrieval and model-cost controls`

### Objective

Beat cloud-provider quality without copying their token consumption or adding
unbounded infrastructure.

### Instrumentation first

Measure and expose:

- extraction prompt/completion tokens;
- reflection/dream tokens;
- embedding requests, cache hit rate, and failures;
- keyword, vector, graph, resolution, and formatting latency;
- injected tokens and result counts;
- queue age and throughput;
- corpus size and vector scan cost;
- cache hit/miss/eviction by effective identity.

Do not record full private prompts in metrics.

### Efficiency work

In order:

1. batch embeddings during background extraction/import;
2. reuse cached query embeddings across safe normalized-query keys;
3. gate LLM conflict classification behind deterministic ambiguity detection;
4. batch bounded ambiguous claim decisions when supported;
5. add diversity grouping before expensive reranking;
6. evaluate a small local reranker only if retrieval metrics show a ranking,
   rather than candidate-generation, failure;
7. evaluate ANN only when exact-scan p95 breaches the documented corpus gate.

### ANN gate

Do not add ANN merely because competitors use it. Introduce it only when:

- exact semantic p95 exceeds the supported target at a measured corpus size;
- ANN improves p95 materially;
- recall@5 loses no more than `0.01` against exact search;
- index rebuild, backup, model-dimension change, and exact fallback are tested.

### Acceptance tests

- every remote call has bounded connect/read/write/pool timeout;
- a stalled embedding or classifier cannot erase BM25 context;
- cache keys cannot cross authorization scope;
- no model call occurs for deterministic duplicate/update cases;
- token and latency summaries are reproducible;
- performance tests use production-shaped corpus sizes and publish hardware
  assumptions;
- default context remains below the leadership gates.

### Rollback

Each optimization needs an independent flag. Legacy exact search remains the
oracle and fallback.

---

## WP-10: Productionize installation, CI, health, and documentation

Suggested branch: `codex/remnant-production-gates`

Suggested PR title: `chore: add CI, packaging, health, and release gates`

### Objective

Make Remnant as easy to trust and operate as a mature provider.

### CI

Add GitHub Actions for:

- supported Python versions from `pyproject.toml`;
- `python -m pytest -q`;
- `python -m ruff check remnant tests`;
- `python -m build` and wheel import smoke;
- fresh database creation;
- migration from checked-in representative old schemas;
- interrupted migration recovery;
- deterministic evaluation smoke;
- `git diff --check` equivalent whitespace checks where useful.

Pin actions to stable major versions and keep secrets out of pull-request CI.
Remote LLM tests use mocks. Optional live evaluation runs manually or on a
protected schedule.

### Packaging and setup

- add a documented one-command install path where Hermes can discover Remnant;
- validate plugin manifest and config schema in tests;
- expose optional extras for GLiNER and evaluation dependencies;
- never download a large model merely by importing Remnant;
- make local keyword-only operation an explicit supported degraded mode;
- validate endpoint protocol/model configuration during setup.

### Availability and health

Replace `is_available() -> True` with local, non-network checks:

- config parse validity;
- database parent writability;
- schema compatibility;
- optional dependency status;
- endpoint configuration shape.

Report `available`, `degraded`, and `unavailable` reasons separately through
maintenance health output. Health must include:

- DB integrity and schema version;
- queue counts, oldest age, retries, dead letters;
- embedding coverage by model/dimension;
- pending overlay count;
- unresolved conflict count and age;
- claim backfill coverage/version;
- prefetch injection/fallback/deadline rates;
- latency percentiles;
- last successful vault scan and dream run;
- active memory/entity/relation-evidence counts.

### Documentation truth pass

Update README claims from executable behavior and generated defaults. In
particular verify:

- whether trust actually affects ranking;
- whether duplicate observations increment evidence/seen count;
- config file format and location;
- exact Hermes hooks implemented;
- test count and latest evaluation report;
- optional versus required model infrastructure;
- privacy behavior for shared databases and gateway users.

Add a capability matrix against Hermes providers, but label external benchmark
numbers and do not claim superiority until the leadership gates pass.

### Release process

- use semantic versions;
- publish a changelog with migration and rollback notes;
- attach evaluation summary and CI commit SHA to releases;
- keep one compatibility release before removing legacy flags/formatters;
- document backup and restore before schema-changing upgrades.

### Acceptance tests

- clean install into an isolated Hermes home works from the built wheel;
- keyword-only degraded operation works without Ollama or GLiNER;
- health never performs an unbounded network request;
- CI catches a deliberately broken migration, Ruff error, and fixture regression;
- README defaults match `RemnantConfig` automatically or through a validation
  test;
- backup/import restores the shared DB and new derived tables.

---

## Cross-package migration policy

Every schema-changing package must follow this sequence:

1. acquire provider maintenance lock;
2. stop or drain background mutators;
3. run `PRAGMA integrity_check`;
4. create and verify a backup at a new explicit path;
5. record source schema version and application commit;
6. run additive migration inside one transaction;
7. run post-migration integrity and invariant checks;
8. commit only if every check passes;
9. preserve the failed candidate DB for diagnosis if validation fails;
10. restore the verified backup to a new path rather than destructively
    rewriting the failed database.

There is no automatic in-place downgrade. Feature flags provide behavioral
rollback while additive schema remains readable.

## Cross-package security checklist

Every PR must answer these questions in its description:

- Can this change load a memory before authorization is applied?
- Can a requested scope broaden configured scope?
- Can cache keys collide across users, agents, profiles, or workspaces?
- Can private text reach a cloud model, log, metric, error, or audit record?
- Can recalled text escape the memory-context fence?
- Can a model response directly choose IDs or lifecycle mutations without local
  validation?
- Can a retry duplicate a write?
- Can a missing identity default to broader access?
- Can a migration or partial scan interpret failure as deletion?

Any `yes` or uncertain answer blocks merge until addressed by a test.

## Standard validation matrix

Run after every work package:

```text
python -m pytest -q
python -m ruff check remnant tests
python -m build
git diff --check
```

Also run package-specific commands:

- schema packages: fresh DB, oldest supported fixture migration, interrupted
  migration, backup/restore, `PRAGMA integrity_check`;
- retrieval packages: deterministic evaluation baseline versus candidate;
- identity/security packages: full authorization matrix;
- performance packages: production-shaped corpus benchmark with p50/p95;
- context packages: exact token-budget and prompt-injection fixtures.

If Python is unavailable in the implementation environment, stop and report
that validation is blocked. Do not claim tests passed based on source review.

## Pull-request template for the implementation model

Each PR description should contain:

```text
Work package:
Problem:
Behavior before:
Behavior after:
Schema/config changes:
Feature flag and default:
Security/privacy impact:
Evaluation delta:
Latency/token delta:
Tests run and exact result:
Migration validation:
Rollback procedure:
Known limitations:
```

Reviewers should reject a PR that lacks measured before/after evidence for a
retrieval, ranking, extraction, or performance change.

## Final enablement sequence

After all packages are merged but before legacy removal:

1. deploy schema and hooks with all new behavioral flags off;
2. run claim extraction/reconciliation in shadow mode;
3. inspect unsafe supersession and unresolved-conflict samples;
4. enable recent-turn overlay for one profile;
5. enable v2 extraction for one profile;
6. enable claim reconciliation transitions;
7. enable claim-aware ranking while retaining legacy shadow scores;
8. enable resolved context formatting;
9. run the full evaluation suite and production health cycle;
10. expand profile-by-profile only after leadership gates hold;
11. retain rollback flags for at least one release;
12. remove legacy behavior only in a separately reviewed major/minor release
    with documented migration and restore instructions.

## Final definition of done

The program is complete only when:

- all ten work packages have merged independently;
- all new-database and upgraded-database tests pass;
- claim resolution is authoritative for automatic recall;
- temporal, conditional, historical, and unresolved cases behave correctly;
- immediate next-turn recall is guaranteed without blocking extraction;
- runtime identities and cache scopes are isolated;
- lifecycle projections are transactional;
- evaluation exceeds the leadership correctness and efficiency gates;
- CI, packaging, health, backup, migration, and rollback are operational;
- documentation matches current executable behavior;
- the release evidence supports, rather than merely asserts, that Remnant is the
  best memory provider for the target Hermes deployment.
