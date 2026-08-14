# Remnant Echo implementation plan

Status: implemented on `codex/echo-implementation` (shadow-first Echo v1)

Baseline: `hermes-remnant` 0.2.1, schema 14, `main` at `bf367a1`

Audience: an implementation model working one pull request at a time

## 1. Objective

Build **Remnant Echo**, a local, auditable memory-utility subsystem that learns
which recalled memories actually help or harm Hermes for a class of task.

Remnant already ranks evidence by retrieval relevance, claim applicability, and
evidence quality. Echo adds a separate, contextual utility signal:

- **truth** answers whether evidence is reliable and currently applicable;
- **utility** answers whether it helps with this kind of query;
- **risk** answers whether it has distracted or misled Hermes in this context.

Echo must not replace the existing `claims-v1` ranker, mutate truth, or add an
LLM call to the response path. It starts in shadow mode and may affect ranking
only after the release gates in this document pass.

The motivating research is utility-based retrieval rather than relevance-only
retrieval. Useful primary references are:

- [SCARLet: utility retrieval through perturbation attribution](https://aclanthology.org/2025.emnlp-main.33/)
- [Feedback Adaptation for Retrieval-Augmented Generation](https://aclanthology.org/2026.findings-acl.1419/)

The Remnant-specific contribution is the combination of contextual utility,
temporal claims, explicit evidence provenance, bounded counterfactual replay,
Hermes lifecycle integration, and fully local audit/rollback behavior.

## 2. Required outcomes

The completed system must:

1. Record exactly which memory items crossed the Hermes provider boundary.
2. Correlate the injected items with the next persisted Hermes turn.
3. Accept strong explicit feedback and conservative inferred outcome signals.
4. Run sampled counterfactual evaluation only in a background worker.
5. Maintain bounded per-memory and sparse per-pair utility aggregates.
6. Calculate a capped contextual rank adjustment without changing truth state.
7. Explain every adjustment from stored aggregate evidence.
8. Degrade immediately to the existing ranker on timeout, error, disablement,
   database contention, or evaluator unavailability.
9. Bound storage, model utilization, write load, and retention at steady state.
10. Demonstrate improvement through versioned evaluation before rank influence
    is enabled by default.

## 3. Non-goals

Do not implement any of the following as part of Echo:

- online training or fine-tuning of an embedding/reranking model;
- automatic changes to claim status, verification, provenance, or validity;
- a replacement for BM25, vector, graph, RRF, or `claims-v1` ranking;
- ANN/vector-index changes;
- an unbounded interaction log;
- arbitrary all-pairs memory attribution;
- inferred approval from user silence;
- a required cloud evaluator;
- a foreground counterfactual or evaluator model call;
- automatic cross-user transfer of utility;
- changes to other Hermes memory-provider repositories.

## 4. Architectural invariants

These rules are release blockers, not suggestions.

### 4.1 Truth and utility remain separate

Echo may attach `ranking.echo` diagnostics and add a bounded rank adjustment.
It must never update:

- `claims.status`, validity, confidence, or reconciliation decisions;
- `memories.verified`, `trust_score`, or `status`;
- provenance, visibility, ownership, or profile scope.

Existing explicit `memory_edit feedback=wrong` behavior may continue to update
trust through the existing edit path. Echo records a corresponding utility
signal, but it does not become the owner of truth state.

### 4.2 Authorization precedes utility

Recall must apply identity, visibility, locked-note masking, and profile scope
before Echo sees candidates. Echo must never promote a candidate that the base
recall path removed. Utility keys include the effective viewer/profile scope so
one user's behavior cannot affect another user's private ranking.

### 4.3 The hot path is local and bounded

The only allowed foreground work is:

- deterministic query-archetype classification;
- one indexed batch lookup for the candidate memory IDs;
- bounded arithmetic over at most the already-discovered candidates;
- enqueueing a small receipt payload without waiting for its commit.

The Echo foreground budget is 3 ms by default. If the budget is exceeded,
return the unchanged base ranking and record a bounded metric.

### 4.4 Only consumed context earns a receipt

`queue_prefetch()` may compute context that Hermes never consumes. Therefore:

- `_run_prefetch()` returns a receipt **draft** with its structured result;
- background prefetch must not persist or activate the draft;
- `RemnantMemoryProvider.prefetch()` activates the receipt immediately before
  it returns a non-empty context string, for both queued and live results;
- an empty, suppressed, expired, or unused queued result creates no receipt;
- repeated consumption of the same draft is idempotent.

This rule prevents learning from memories that were never shown to Hermes.

### 4.5 Raw content is not copied into long-lived Echo tables

The existing `turns` and `memories` rows already hold the source content. Echo
stores IDs, hashes, scores, counters, policy/model versions, and timestamps.
The worker reconstructs its bounded evaluation input from those authoritative
rows. Do not persist full prompts, assistant answers, or compiled context in
Echo tables.

### 4.6 Baseline behavior is always available

With `echo_enabled=false`, on any Echo exception, or after dropping/ignoring
Echo tables, recall output must be byte-for-byte equivalent to the non-Echo
path for the same database and configuration.

## 5. End-to-end lifecycle

```text
Hermes calls prefetch(query)
  -> RecallService discovers, resolves, and base-ranks authorized candidates
  -> context compiler selects exact rendered items
  -> Echo calculates a shadow or active utility ordering
  -> provider returns context
  -> only now: activate one injection receipt

Hermes generates an answer or performs work
  -> Hermes calls sync_turn(user, assistant)
  -> Remnant atomically persists the turn and extraction job
  -> Echo closes the matching open receipt with the returned turn_id
  -> deterministic signals are recorded
  -> an eligible counterfactual job may be queued

Background Echo worker
  -> enforces local-only, backlog, time, and daily resource budgets
  -> reconstructs query, answer, and memory set from authoritative rows
  -> performs a bounded single-item or pair ablation
  -> writes a versioned signal
  -> folds signals into utility aggregates
  -> compacts expired receipts, signals, jobs, and weak pair rows

Next recall
  -> base claims-v1 ranking remains authoritative
  -> Echo batch-loads contextual utility
  -> shadow mode records an alternative order but returns baseline
  -> active mode applies a capped adjustment before context allocation
```

## 6. Prerequisite: structured compiled-context output

Echo cannot learn correctly from `RecallResponse.results`, because the context
compiler can omit or truncate a different subset under its evidence-class token
allocation. Implement this prerequisite in the first pull request.

Add immutable types in `remnant/context.py` or `remnant/types.py`:

```python
@dataclass(frozen=True)
class RenderedMemory:
    memory_id: str
    ordinal: int
    evidence_class: str
    rendered_tokens: int
    rendered_hash: str
    truncated: bool
    item_kind: str  # memory | pending
    source_turn_id: int | None = None

@dataclass(frozen=True)
class CompiledContext:
    text: str
    items: tuple[RenderedMemory, ...]
    token_count: int
    omitted_ids: tuple[str, ...]
    per_class_tokens: dict[str, int]
```

Change `compile_context()` to return `CompiledContext`. Keep a temporary
compatibility wrapper if tests or public callers require a string. Migrate all
internal callers in the same pull request. `RecallResponse` gains
`rendered_ids` or the structured compiled object; do not infer rendered IDs by
searching for truncated UUIDs in context text.

Acceptance criteria:

- every returned rendered ID corresponds to a line present in `text`;
- omitted and truncated items are explicit;
- tokenizer-aware accounting is used for both selection and diagnostics;
- prefetch's `memories` metadata contains only actually rendered items;
- existing context budget and injection-safety tests remain green;
- evaluation adds `context_recall_at_k` using `CompiledContext.items`.

## 7. Database design

At implementation time, bump `SCHEMA_VERSION` from 14 to 15 only if 14 is
still current. Otherwise use the next free schema version. Add tables to the
base schema and idempotent creation/index logic to `_apply_migrations()`.
Opening an old database must create empty Echo tables without backfilling.

### 7.1 `echo_receipts`

One row represents context that actually crossed the provider boundary.

```sql
CREATE TABLE IF NOT EXISTS echo_receipts (
    id TEXT PRIMARY KEY,
    activation_key TEXT NOT NULL UNIQUE,
    session_id TEXT NOT NULL,
    turn_id INTEGER,
    agent_id TEXT NOT NULL,
    viewer_key_hash TEXT NOT NULL,
    profile_scope_hash TEXT NOT NULL,
    query_fingerprint TEXT NOT NULL,
    query_archetype TEXT NOT NULL,
    context_hash TEXT NOT NULL,
    memory_generation INTEGER NOT NULL,
    rendered_count INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    policy_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('open','closed','expired')),
    outcome TEXT,
    created_at REAL NOT NULL,
    closed_at REAL,
    FOREIGN KEY(turn_id) REFERENCES turns(id)
);

CREATE INDEX IF NOT EXISTS idx_echo_receipt_match
    ON echo_receipts(session_id, query_fingerprint, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_echo_receipt_created
    ON echo_receipts(created_at);
CREATE INDEX IF NOT EXISTS idx_echo_receipt_turn
    ON echo_receipts(turn_id);
```

Use SHA-256 for query, viewer, profile-scope, context, and rendered-text
fingerprints. Normalize queries exactly once in a shared helper. Never use
Python's process-randomized `hash()`.

### 7.2 `echo_receipt_items`

This is the exact ordered rendered set, not the pre-budget candidate set.

```sql
CREATE TABLE IF NOT EXISTS echo_receipt_items (
    receipt_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    item_kind TEXT NOT NULL CHECK(item_kind IN ('memory','pending')),
    source_turn_id INTEGER,
    evidence_class TEXT NOT NULL,
    score_lane TEXT,
    base_score REAL NOT NULL,
    base_rank INTEGER NOT NULL,
    rendered_tokens INTEGER NOT NULL,
    rendered_hash TEXT NOT NULL,
    claim_status TEXT,
    PRIMARY KEY(receipt_id, memory_id),
    FOREIGN KEY(receipt_id) REFERENCES echo_receipts(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_echo_item_memory
    ON echo_receipt_items(memory_id, receipt_id);
```

Do not add a foreign key from `memory_id` to `memories`: pending-turn IDs are
valid rendered items and are intentionally not memory rows.

### 7.3 `echo_signals`

Signals are bounded, explainable observations. They expire after aggregation.

```sql
CREATE TABLE IF NOT EXISTS echo_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    receipt_id TEXT,
    memory_id TEXT NOT NULL,
    paired_memory_id TEXT,
    agent_id TEXT NOT NULL,
    viewer_key_hash TEXT NOT NULL,
    query_archetype TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    direction INTEGER NOT NULL CHECK(direction IN (-1, 1)),
    weight REAL NOT NULL CHECK(weight > 0 AND weight <= 1),
    source TEXT NOT NULL,
    evaluator_version TEXT,
    created_at REAL NOT NULL,
    aggregated_at REAL,
    FOREIGN KEY(receipt_id) REFERENCES echo_receipts(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_echo_signal_pending
    ON echo_signals(aggregated_at, created_at);
CREATE INDEX IF NOT EXISTS idx_echo_signal_memory
    ON echo_signals(memory_id, query_archetype, created_at);
```

`signal_type` is application-validated rather than constrained by a schema
CHECK so future versions can add types without rebuilding the table.

### 7.4 `echo_utility`

This is the long-lived bounded aggregate. Rows are created only after an actual
signal; never materialize the Cartesian product of memories and archetypes.

```sql
CREATE TABLE IF NOT EXISTS echo_utility (
    agent_id TEXT NOT NULL,
    viewer_key_hash TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    query_archetype TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    explicit_positive_mass REAL NOT NULL DEFAULT 0,
    explicit_negative_mass REAL NOT NULL DEFAULT 0,
    inferred_positive_mass REAL NOT NULL DEFAULT 0,
    inferred_negative_mass REAL NOT NULL DEFAULT 0,
    explicit_positive INTEGER NOT NULL DEFAULT 0,
    explicit_negative INTEGER NOT NULL DEFAULT 0,
    evaluator_samples INTEGER NOT NULL DEFAULT 0,
    effective_observations REAL NOT NULL DEFAULT 0,
    utility_mean REAL NOT NULL DEFAULT 0.5,
    harm_risk REAL NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0,
    last_signal_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY(
        agent_id, viewer_key_hash, memory_id, query_archetype, policy_version
    )
);

CREATE INDEX IF NOT EXISTS idx_echo_utility_lookup
    ON echo_utility(
        agent_id, viewer_key_hash, query_archetype, memory_id
    );
CREATE INDEX IF NOT EXISTS idx_echo_utility_updated
    ON echo_utility(updated_at);
```

Policy version is part of the key so incompatible scoring/evaluator policies
are not silently mixed. Retrieval reads only the configured active policy.

### 7.5 `echo_pair_utility`

Pair rows represent sparse complementarity or conflict. Store memory IDs in
lexicographic canonical order.

```sql
CREATE TABLE IF NOT EXISTS echo_pair_utility (
    agent_id TEXT NOT NULL,
    viewer_key_hash TEXT NOT NULL,
    first_memory_id TEXT NOT NULL,
    second_memory_id TEXT NOT NULL,
    query_archetype TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    positive_mass REAL NOT NULL DEFAULT 0,
    negative_mass REAL NOT NULL DEFAULT 0,
    sample_count INTEGER NOT NULL DEFAULT 0,
    synergy_score REAL NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0,
    last_signal_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY(
        agent_id, viewer_key_hash, first_memory_id, second_memory_id,
        query_archetype, policy_version
    ),
    CHECK(first_memory_id < second_memory_id)
);

CREATE INDEX IF NOT EXISTS idx_echo_pair_first
    ON echo_pair_utility(agent_id, viewer_key_hash, first_memory_id, query_archetype);
CREATE INDEX IF NOT EXISTS idx_echo_pair_second
    ON echo_pair_utility(agent_id, viewer_key_hash, second_memory_id, query_archetype);
```

### 7.6 `echo_jobs`

Use a separate durable queue rather than overloading `extraction_queue`.

```sql
CREATE TABLE IF NOT EXISTS echo_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    receipt_id TEXT NOT NULL,
    job_type TEXT NOT NULL CHECK(job_type IN ('single','pair')),
    target_ids TEXT NOT NULL,
    priority REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK(status IN ('pending','running','done','failed','skipped')),
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at REAL NOT NULL DEFAULT 0,
    started_at REAL,
    last_error TEXT,
    evaluator_version TEXT NOT NULL,
    created_at REAL NOT NULL,
    completed_at REAL,
    FOREIGN KEY(receipt_id) REFERENCES echo_receipts(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_echo_job_ready
    ON echo_jobs(status, next_attempt_at, priority DESC, id);
```

`target_ids` is a JSON array of one or two canonical IDs. Validate its length
before insert and again when claiming a job.

### 7.7 `echo_daily_metrics`

Store bounded daily aggregates after detailed rows expire.

```sql
CREATE TABLE IF NOT EXISTS echo_daily_metrics (
    day TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    metric TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    total REAL NOT NULL DEFAULT 0,
    maximum REAL NOT NULL DEFAULT 0,
    PRIMARY KEY(day, agent_id, metric)
);
```

Also add retention pruning for existing `prefetch_stats`, which currently does
not have the 10,000-row cap used by `operation_metrics`.

## 8. Module boundaries

Do not add hundreds more lines to `db.py`. Introduce:

```text
remnant/echo.py
    EchoService orchestration
    receipt-draft activation and turn correlation
    shadow/active rank adjustment
    public explain/status methods

remnant/echo_store.py
    EchoRepository using RemnantDB.read()/transaction()
    table-specific reads/writes, aggregation, retention, job claiming

remnant/echo_policy.py
    query archetypes
    signal weights
    decay and Bayesian utility math
    pair canonicalization and pruning priority

remnant/echo_worker.py
    background lifecycle and resource budgets
    durable job recovery/retry

remnant/echo_evaluate.py
    local evaluator protocol, prompt, parsing, and result validation
    deterministic fake evaluator for tests

remnant/echo_types.py
    frozen dataclasses and enums shared by the modules above
```

`db.py` owns only base schema/migration creation. Echo CRUD belongs in
`EchoRepository`, which receives the existing `RemnantDB` and uses its locking
and transaction primitives.

## 9. Configuration

Add typed fields to `RemnantConfig`, plugin configuration metadata, README, and
configuration tests. These are the initial optimal defaults:

```yaml
echo_enabled: true
echo_shadow_mode: true
echo_policy_version: echo-v1
echo_rank_influence: 0.0

echo_receipt_retention_days: 30
echo_signal_retention_days: 30
echo_store_raw_content: false

echo_initial_sample_rate: 0.05
echo_mature_sample_rate: 0.005
echo_mature_observations: 20
echo_max_jobs_per_day: 20
echo_max_evaluator_seconds_per_day: 300
echo_worker_poll_interval_s: 5
echo_job_stale_after_s: 900
echo_job_max_attempts: 3

echo_min_observations: 10
echo_max_rank_adjustment: 0.10
echo_utility_half_life_days: 90
echo_explicit_feedback_half_life_days: 365

echo_pair_attribution_enabled: true
echo_max_pairs_per_receipt: 3
echo_max_pairs_per_memory_archetype: 20
echo_pair_half_life_days: 60

echo_hot_path_budget_ms: 3
echo_disable_on_budget_exceeded: true
echo_pause_when_model_busy: true
echo_allow_remote_evaluator: false
```

Configuration validation must reject or normalize:

- rates outside `[0, 1]`;
- non-positive retention, timeout, worker, and observation values;
- rank adjustment outside `[0, 0.25]`;
- rank influence outside `[0, 1]`;
- more than five pairs per receipt;
- a remote evaluator when `echo_allow_remote_evaluator=false`;
- `echo_shadow_mode=false` unless an explicit non-zero influence is desired.

Do not use `getattr()` fallbacks throughout Echo. Normalize configuration once
on load and pass a valid typed config to the subsystem.

## 10. Query archetypes

Version 1 uses deterministic, local classification. Do not call an LLM.

Allowed values:

```text
preference
historical
project_state
document_lookup
troubleshooting
person_or_entity
procedure
delegation
general_recall
unknown
global  # aggregate only; the query classifier never emits this value
```

Classification order matters. Recommended priority:

1. history/date intent;
2. document/file/note intent;
3. troubleshooting/error intent;
4. preference intent;
5. project/status intent;
6. delegation intent;
7. procedure/how-to intent;
8. person/entity intent using already-resolved query entities;
9. explicit recall language;
10. unknown.

Return both archetype and classifier version. Tests must pin ambiguous cases.
Do not create free-form archetype labels because they would create unbounded
utility-key cardinality. `global` is reserved for explicit feedback that is not
tied to a reliable current-query receipt. Retrieval may blend a global utility
aggregate at 25% weight with the current archetype; it must never treat
`global` as a classified query.

## 11. Receipt creation and correlation

### 11.1 Receipt draft

After context compilation, create an immutable in-memory draft containing:

- query fingerprint and archetype;
- effective agent/viewer/profile hashes;
- memory-generation counter;
- context hash and token count;
- exact `CompiledContext.items`;
- base scores, lanes, ranks, claim statuses, and evidence classes;
- policy version and creation monotonic/wall-clock timestamps.

The draft contains no assistant answer and performs no write.

### 11.2 Activation

`RemnantMemoryProvider.prefetch()` calls `EchoService.activate_receipt(draft)`
only immediately before returning a non-empty string. Activation:

- generates a UUID;
- inserts receipt and items in one short transaction;
- is idempotent by draft activation key;
- enqueues through a bounded in-memory writer when available;
- falls back to a direct best-effort short transaction;
- never causes prefetch to fail.

The activation key is a SHA-256 of:

```text
viewer_key_hash | session_id | query_fingerprint | context_hash |
memory_generation | policy_version
```

Use a unique index or `INSERT OR IGNORE` to prevent duplicate receipts.

### 11.3 Turn correlation

`ingest_turn()` already returns `turn_id`. Change `sync_turn()` to retain it,
then call:

```python
echo.close_receipt(
    session_id=sid,
    viewer_key_hash=effective_viewer_hash,
    query_fingerprint=fingerprint(user_content),
    turn_id=turn_id,
)
```

Close the newest open receipt matching session, viewer, and query fingerprint
within a bounded age, default five minutes. Never match on session alone.

If no receipt matches, do nothing. If several match, close only the newest and
expire the older duplicates. Closing failure must not roll back turn ingestion.

On startup, a recovery sweep may close an open receipt by matching a later
`turns` row with the same session/agent and query fingerprint. Cap the sweep to
the retention window and do not add raw query hashes to `turns`; compute them
while scanning a bounded recent set.

## 12. Outcome signals

Signal weights are policy-versioned constants. Start conservatively:

| Signal | Direction | Weight | Attribution |
|---|---:|---:|---|
| Explicit `feedback=useful` | + | 1.00 | named memory, global and current archetype |
| Explicit `feedback=wrong` | - | 1.00 | named memory, global and current archetype |
| User gives an explicit correction tied to evidence | - | 0.90 | implicated memory only |
| Tool/result directly contradicts evidence | - | 0.85 | implicated memory only |
| Successful delegated/tool outcome with cited support | + | 0.60 | cited/supporting items only |
| Counterfactual replay improves answer with item | + | 0.40 | evaluated item/pair |
| Counterfactual replay improves answer without item | - | 0.50 | evaluated item/pair |
| Local support assessor finds direct grounding | + | 0.20 | supporting item |
| Immediate repeated question | - | 0.10 | receipt-level diagnostic only in v1 |
| User silence | none | 0 | never recorded |

Version 1 must implement explicit feedback and counterfactual signals. User
correction parsing may be added only with conservative fixtures. Tool outcome
signals must wait for a reliable Hermes callback or explicit result linkage;
do not infer them from arbitrary assistant prose.

Never spread a negative receipt-level signal across all rendered memories when
the implicated evidence is unknown. Store it as a receipt metric, not a
per-memory utility update.

## 13. Utility calculation

Use a weighted Beta estimate with a neutral prior:

```python
PRIOR_POSITIVE = 2.0
PRIOR_NEGATIVE = 2.0

positive_mass = decayed_explicit_positive + decayed_inferred_positive
negative_mass = decayed_explicit_negative + decayed_inferred_negative
mean = (PRIOR_POSITIVE + positive_mass) / (
    PRIOR_POSITIVE + PRIOR_NEGATIVE + positive_mass + negative_mass
)
effective_observations = positive_mass + negative_mass
confidence = 1.0 - exp(-effective_observations / 10.0)
centered_utility = 2.0 * (mean - 0.5)  # [-1, +1]
raw_adjustment = max_rank_adjustment * confidence * centered_utility
```

Apply time decay lazily when reading or aggregating:

```python
decay = 0.5 ** (age_days / half_life_days)
```

Explicit masses use the longer explicit-feedback half-life. Evaluator masses
use the normal utility half-life. Store enough separate counts/masses to apply
those decay schedules independently; do not reconstruct them from `utility_mean`.

`harm_risk` is the decayed negative mass divided by total decayed mass. It is
diagnostic until minimum observations are reached.

### 13.1 Ranking integration

Integrate after base claim-aware ranking and before token-budget selection:

```python
echo_adjustment = 0.0
if observations >= echo_min_observations:
    echo_adjustment = clamp(raw_adjustment, -max_adjustment, max_adjustment)

adjusted_score = base_final_score + echo_adjustment
```

Before the addition, multiply `echo_adjustment` by `echo_rank_influence`.
Shadow/default configuration uses `0.0`; canary stages use `0.25`, `0.5`, then
`1.0`. `echo_shadow_mode=true` always forces effective influence to zero even
if a stale configuration contains a non-zero value.

The base score and adjustment must both remain in diagnostics:

```json
{
  "ranking": {
    "base_final": 0.73,
    "echo": {
      "archetype": "troubleshooting",
      "utility_mean": 0.68,
      "confidence": 0.74,
      "observations": 18.2,
      "adjustment": 0.027,
      "policy": "echo-v1"
    },
    "final": 0.757
  }
}
```

In shadow mode calculate both orders, but return the baseline order and context.
Record only aggregate rank-change metrics, not duplicate full contexts.

### 13.2 Exploration

Do not add random exploration in the first active release. Shadow mode supplies
observations without risking user-visible ranking. If later evaluation proves a
long-tail starvation problem, add deterministic exploration among equally safe
candidates using a stable query fingerprint seed. Never explore invalid,
stale-for-current-query, unauthorized, locked, or unresolved unsafe evidence.

## 14. Pair utility

Pair evaluation addresses complementarity and conflict. It must remain sparse.

Selection rules:

- only pairs where both items were actually rendered;
- no more than three pairs per receipt;
- prioritize different evidence classes or shared entities/claims;
- skip pairs already above mature confidence;
- persist a pair aggregate only after three observations;
- retain no more than twenty pairs per memory/archetype;
- prune by confidence multiplied by absolute synergy score;
- expire or ignore pairs containing non-active memory versions.

Compute pair synergy as the pair's observed utility minus the sum of bounded
individual utility contributions. In the first release, pair utility is
diagnostic only. A later release may add at most `0.03` when selecting a second
item after its paired first item is already selected. Pair utility must never
make an unsafe item eligible.

## 15. Counterfactual evaluator

### 15.1 Eligibility

A closed receipt is eligible when:

- Echo and counterfactual sampling are enabled;
- the receipt has at least two real memory items;
- an authoritative turn is attached;
- daily job/time budgets remain;
- no existing job covers the same receipt/targets/policy;
- the target is new, low-confidence, recently corrected, potentially harmful,
  or disagrees strongly with base relevance;
- the configured evaluator is local, unless remote evaluation is explicitly
  allowed.

Sampling transitions from `echo_initial_sample_rate` to
`echo_mature_sample_rate` after the target reaches mature observations.

### 15.2 Evaluation modes

Implement two explicit modes:

1. **support assessment**: score whether each item supports, contradicts, or is
   irrelevant to the recorded answer; one bounded evaluator call;
2. **counterfactual replay**: compare a bounded answer/assessment with and
   without one item or one pair; sampled more rarely.

Do not call the mechanism counterfactual if it only asks for support labels.
Store the mode in `signal_type` and evaluator version.

### 15.3 Input reconstruction

Reconstruct from:

- `turns.user_text` and `turns.assistant_text` through `receipt.turn_id`;
- `memories.content` for memory items;
- `turns.user_text` for pending-turn items using `source_turn_id`;
- receipt ordering and evidence metadata.

Memory rows are immutable evidence even when superseded, so old receipt input
remains reconstructable. If any required row is missing, mark the job skipped;
do not fabricate content.

### 15.4 Evaluator output

Require strict JSON:

```json
{
  "target_ids": ["memory-id"],
  "with_score": 0.0,
  "without_score": 0.0,
  "support": "supports|contradicts|irrelevant|uncertain",
  "confidence": 0.0,
  "reason_code": "bounded-enum-value"
}
```

Validate all fields, clamp scores to `[0,1]`, reject target mismatches, and map
free-form explanations to no persisted field. The evaluator prompt and model
identifier form `evaluator_version`. Temperature is zero.

Only produce a signal if evaluator confidence is at least 0.7 and absolute
score delta is at least 0.1. Cap evaluator-derived signal weight at 0.5.

### 15.5 Resource control

The worker must:

- use a durable LIFO/priority queue with stale-running recovery;
- attempt a job at most three times with bounded exponential backoff;
- stop after twenty jobs or 300 evaluator seconds per UTC day;
- pause when extraction backlog is non-empty if model resources are shared;
- pause while a foreground query embedding/model request is active when that
  state is available;
- perform no more than one evaluator request concurrently by default;
- record only bounded counters and error codes;
- wake on new eligible work and shut down within the provider timeout.

## 16. Retention and compaction

Run bounded compaction at startup and no more than once per day:

1. fold unaggregated signals into utility rows transactionally;
2. mark signals aggregated;
3. update daily metrics;
4. delete aggregated signals older than 30 days;
5. delete closed/expired receipt items and receipts older than 30 days;
6. expire open receipts older than five minutes;
7. delete completed/skipped jobs older than seven days;
8. prune failed jobs older than 30 days;
9. decay/prune utility rows for forgotten/non-active memories;
10. prune sparse pair rows to configured caps and half-life;
11. cap existing `prefetch_stats` by retention or maximum row count;
12. perform no `VACUUM` automatically on the foreground process.

Aggregation must be idempotent. Update utility and set `aggregated_at` in the
same transaction so a crash cannot double-apply a signal.

## 17. Provider integration

### Initialization

- Construct `EchoRepository` after `RemnantDB` is open.
- Construct `EchoService` with validated config and effective identity.
- Start `EchoWorker` only when enabled and an allowed evaluator exists.
- Recover stale running jobs and expire old open receipts asynchronously.

### Prefetch

- Get exact rendered IDs from `CompiledContext`.
- Batch-read utility for authorized candidate IDs.
- Enforce the 3 ms Echo budget.
- Produce shadow diagnostics or active adjusted ordering.
- Activate receipt only when non-empty context is returned to Hermes.
- Never let receipt failure erase safe recalled context.

### `sync_turn`

- Retain the `turn_id` returned by `ingest_turn()`.
- Close the exact matching receipt after turn persistence.
- Record deterministic signals and enqueue eligible jobs best-effort.
- Preserve the existing sub-10 ms target; Echo close/enqueue p95 must remain
  below 2 ms or be moved fully behind the writer queue.

### Session switch/reset/rewind

- clear in-memory receipt drafts for affected sessions;
- expire unmatched open receipts for reset/rewind sessions;
- never delete aggregate utility merely because a session changes.

### Shutdown

- stop accepting new jobs;
- give the writer a bounded flush window;
- return running jobs to pending if evaluator completion cannot be committed;
- stop the worker before closing the shared database.

## 18. Explainability and operations

Prefer extending existing surfaces over adding many tools.

### Search diagnostics

Add optional Echo diagnostics to `memory_search` results when available. Do not
expose viewer hashes or other users' aggregates.

### Explain operation

Add either `memory_explain(memory_id, query?)` or an `explain` action to an
existing memory tool. It returns:

- active query archetype;
- base and adjusted rank components;
- observation counts and decayed masses;
- utility mean, confidence, and harm risk;
- explicit versus evaluator signal counts;
- policy/evaluator versions;
- pair synergies relevant to the supplied query;
- no raw historic prompts or answers.

### Health report

Add:

- receipt rows by status and oldest open age;
- receipt writer backlog and dropped receipts;
- signal rows pending aggregation;
- utility/pair row counts and confident coverage;
- jobs by status, oldest pending age, retry/failure counts;
- daily evaluator calls and seconds;
- hot-path p50/p95 and budget bypass count;
- shadow rank-change and selected-set-change rates;
- estimated storage by Echo table;
- compaction last-success time.

Health must use the active profile configuration, not a fresh default config.

## 19. Test plan

Every pull request must add tests before production behavior is enabled.

### 19.1 Unit tests

Create focused tests for:

- every query archetype and priority collision;
- stable SHA-256 fingerprints;
- weighted Beta calculation;
- time decay at zero, one half-life, and multiple half-lives;
- minimum-observation and maximum-adjustment caps;
- explicit/evaluator half-life separation;
- canonical pair ordering;
- pair cap/pruning priority;
- evaluator JSON validation and rejection paths;
- configuration validation and serialization.

### 19.2 Migration and storage tests

- schema 14 opens and migrates idempotently;
- repeated open does not duplicate or lose rows;
- newer schema remains rejected/degraded as today;
- receipt plus items is atomic;
- signal aggregation is atomic and idempotent;
- a forced failure rolls back aggregation and leaves signal pending;
- concurrent reads do not observe partial receipt items;
- pruning respects retention and per-memory pair caps;
- foreign-key deletion behavior matches the schema;
- no raw query, context, or assistant answer appears in Echo tables.

### 19.3 Receipt lifecycle tests

- live non-empty prefetch activates exactly one receipt;
- queued prefetch not consumed creates no receipt;
- consumed queued prefetch creates exactly one receipt;
- empty, deduplicated, deadline, and diff-suppressed paths create none;
- receipt items exactly equal `CompiledContext.items`;
- omitted candidate IDs never appear in receipt items;
- `sync_turn` closes the matching query/session receipt;
- same-session different-query receipt is not closed;
- older duplicate receipts expire;
- restart recovery closes a safely matchable recent receipt;
- ambiguous recovery leaves receipts expired, not guessed;
- session reset/rewind clears drafts safely;
- identities/profile scopes cannot share utility accidentally.

### 19.4 Signal tests

- explicit useful/wrong affects only the named memory;
- user silence emits no signal;
- un-attributable repeated question emits no per-memory negative;
- evaluator mismatch/low confidence/small delta emits no signal;
- superseded memory can be explained but not actively promoted;
- negative signals do not mutate claims or memory trust through Echo.

### 19.5 Worker tests

- disabled/local-only configuration makes no model call;
- foreground recall and `sync_turn` never invoke the evaluator;
- worker obeys daily job and seconds budgets;
- worker pauses under extraction/model pressure;
- stale running jobs recover;
- retry backoff and maximum attempts are bounded;
- missing source rows skip safely;
- shutdown returns incomplete work to a recoverable state;
- fake-clock tests cover UTC-day budget reset.

### 19.6 Ranking tests

- disabled Echo is byte-identical to baseline;
- shadow Echo is byte-identical to baseline output;
- active adjustment never exceeds the configured cap;
- fewer than ten observations produce no adjustment;
- Echo cannot reintroduce an unauthorized/filtered candidate;
- invalid/stale evidence is not promoted over valid claim policy;
- an Echo read exception returns baseline ranking;
- hot-path timeout returns baseline ranking;
- diagnostics preserve base, Echo, and final scores.

### 19.7 Evaluation cases

Extend the versioned held-out corpus with cases where:

- two memories are topically relevant but only one answers the question;
- a highly similar memory distracts the answer;
- a true memory is useful for troubleshooting but irrelevant to preferences;
- a pair is useful only together;
- a correction must affect the next related query;
- a superseded memory retains historic utility but is not currently selected;
- an evaluator gives noisy or adversarial feedback;
- a new memory has no observations and remains neutral.

Report baseline versus shadow counterfactual ordering before active ranking.

### 19.8 Scale benchmarks

Extend the disposable scale harness; never use production data. Measure:

- 10k, 100k, and 1m receipts;
- 10k, 100k, and 1m utility rows;
- 100k and 1m bounded pair rows;
- single- and multi-agent/profile workloads;
- warm/cold utility batch lookup;
- receipt activation/close latency;
- signal aggregation throughput;
- compaction duration and database-size reduction;
- SQLite lock waits under concurrent extraction, recall, and aggregation;
- queue backlog under evaluator slowdown;
- peak memory and database size.

## 20. Performance and quality gates

Echo cannot become rank-active unless all gates pass:

| Area | Gate |
|---|---|
| Recall correctness | No regression in held-out retrieval/context recall |
| Temporal safety | No increase in stale-claim exposure |
| Duplicate control | No increase in duplicate top-k occupancy |
| Foreground latency | Prefetch p95 delta < 5 ms |
| Turn persistence | `sync_turn` p95 delta < 2 ms |
| Foreground model use | Exactly zero additional model calls |
| Failure behavior | Echo-disabled result equals baseline |
| Queue stability | Backlog returns to zero under expected load |
| Storage | Growth remains within configured retention envelope |
| Adaptation | Improved correction lag and post-correction correctness |
| Value | >=10% token reduction or statistically meaningful answer gain |
| Privacy | No raw duplicated content in Echo tables |
| Explainability | Every non-zero adjustment has an explanation |

Do not accept a retrieval-only metric as proof of utility. At least one
answer-level or task-outcome metric must improve.

## 21. Rollout

### Stage A: offline only

- run new deterministic and adversarial evaluation cases;
- populate synthetic utility signals;
- verify caps, decay, and counterfactual labels;
- publish the scale report and baseline comparison.

### Stage B: production shadow

- `echo_enabled=true`, `echo_shadow_mode=true`;
- collect at least 30 days or 5,000 receipts, whichever is later;
- no utility changes user-visible ranking;
- review health, evaluator disagreement, storage, lock contention, and shadow
  selected-set changes;
- audit a sample of positive and negative high-confidence utility rows.

### Stage C: canary influence

- enable only for an explicit canary profile;
- retain `echo_max_rank_adjustment=0.10`;
- start with an additional influence multiplier of 0.25, then 0.5, then 1.0;
- each step requires a complete evaluation window and rollback comparison;
- disable immediately on stale exposure, deadline regression, or correction
  failure.

### Stage D: default active

- only after all gates pass on the target Hermes deployment;
- keep one-switch rollback through `echo_enabled=false` or shadow mode;
- keep exact baseline diagnostics for continued comparison.

## 22. Ordered pull-request work packages

Each pull request must be independently reviewable, migration-safe, and green.
Do not combine all work into one large change.

### PR 1: Structured context contract

Primary files:

- `remnant/context.py`
- `remnant/recall.py`
- `remnant/prefetch.py`
- `remnant/evaluation/runner.py`
- context, prefetch, recall, and evaluation tests

Tasks:

1. Add `CompiledContext` and `RenderedMemory`.
2. Remove the conservative preselection/render mismatch.
3. Align `RecallResponse.results` or explicit rendered results with context.
4. Add context recall and exact rendered-ID diagnostics.
5. Preserve compatibility only where a public string return is required.

Exit: rendered results and injected context are provably identical sets.

### PR 2: Echo schema, repository, configuration, and retention

Primary files:

- `remnant/db.py` schema/migration only
- `remnant/echo_store.py`
- `remnant/echo_types.py`
- `remnant/config.py`
- `remnant/maintenance.py`
- migration/config/storage tests

Tasks:

1. Add all Echo tables and indexes.
2. Add validated optimal defaults.
3. Implement atomic receipt, signal, utility, pair, and job repository methods.
4. Implement idempotent aggregation and bounded compaction.
5. Bound existing `prefetch_stats` retention.
6. Add Echo health output.

Exit: schema migration and retention tests pass; no provider behavior changes.

### PR 3: Receipt lifecycle integration

Primary files:

- `remnant/echo.py`
- `remnant/__init__.py`
- `remnant/prefetch.py`
- `remnant/ingest.py` only if its return contract needs documentation
- receipt lifecycle tests

Tasks:

1. Implement archetype/fingerprint receipt drafts.
2. Activate only at actual provider-boundary consumption.
3. Retain `turn_id` in `sync_turn()` and close exact receipts.
4. Add restart recovery, expiration, session reset, and idempotency.
5. Add a bounded receipt writer and baseline fallback.

Exit: queued-unused context produces zero receipts; consumed context correlates
exactly with the next persisted turn.

### PR 4: Explicit signals, utility math, and shadow ranking

Primary files:

- `remnant/echo_policy.py`
- `remnant/echo.py`
- `remnant/edit.py`
- `remnant/recall.py` or ranking integration point
- `remnant/tools.py` explain surface
- signal, policy, ranking, and explain tests

Tasks:

1. Implement versioned archetypes, weighted Beta utility, and decay.
2. Mirror explicit useful/wrong feedback into Echo signals.
3. Batch-load utility and calculate a capped shadow adjustment.
4. Preserve byte-identical baseline output in shadow mode.
5. Add explanation and metrics.

Exit: shadow alternative ranking is observable; user-visible ranking unchanged.

### PR 5: Counterfactual worker

Primary files:

- `remnant/echo_worker.py`
- `remnant/echo_evaluate.py`
- `remnant/echo.py`
- provider initialization/shutdown
- worker/evaluator/privacy tests

Tasks:

1. Add eligible-job sampling and the durable worker.
2. Add strict local evaluator protocol and fake evaluator.
3. Enforce daily, concurrency, model-busy, retry, and shutdown budgets.
4. Implement single-item evaluation; keep pair evaluation disabled initially.
5. Aggregate evaluator signals and publish health metrics.

Exit: no foreground model calls; worker backlog and resource use are bounded.

### PR 6: Pair attribution, evaluation, and scale gates

Primary files:

- Echo policy/evaluator/worker modules
- `remnant/evaluation/`
- `evaluation/cases/`
- documentation and benchmark reports

Tasks:

1. Add bounded diagnostic pair attribution.
2. Add utility-specific held-out cases and correction-lag metrics.
3. Extend scale harness and publish reproducible commands/results.
4. Run at least the required shadow dataset/window.
5. Produce a go/no-go report for canary influence.

Exit: evidence supports canary activation or Echo remains shadow-only.

### PR 7: Canary activation, only if gates pass

Tasks:

1. Add an explicit canary profile/configuration.
2. Apply the capped adjustment at 0.25 influence.
3. Keep pair influence diagnostic-only unless separately proven.
4. Document rollback and compare canary with baseline.

This PR must not be opened merely because PRs 1-6 merged. It requires measured
go/no-go evidence.

## 23. Commands and definition of done

The implementer must discover and use the repository's configured Python
environment. At minimum, each PR runs:

```text
python -m pytest -q
python -m ruff check remnant tests
python -m build
git diff --check
```

Evaluation/scale PRs additionally run the documented held-out, leadership, and
scale commands and commit only stable reports intended for source control.

For every PR:

- start from current `main` and re-check `SCHEMA_VERSION`;
- use a `codex/` branch unless directed otherwise;
- do not modify other organizations' repositories;
- keep migrations idempotent and backward-open compatible;
- add tests for success, failure, timeout, disablement, and privacy;
- update README/configuration/change log for user-visible behavior;
- preserve the existing lexical fallback and Hermes provider contract;
- resolve review feedback and CI before merging;
- delete the temporary branch after merge;
- re-check clean synchronized `main` and post-merge CI.

Echo is complete only when:

1. PRs 1-6 are merged and all tests/evaluations are green;
2. production shadow evidence satisfies the stated window;
3. a written go/no-go decision exists;
4. canary activation is merged only if its gates pass;
5. disabling Echo demonstrably restores baseline behavior;
6. storage and evaluator work remain bounded after simulated six-month load;
7. no required work remains on an unmerged implementation branch.

## 24. Decisions the implementer must not make implicitly

Stop and request review rather than guessing if any of these arise:

- Hermes changes the `MemoryProvider.prefetch()` or `sync_turn()` contract;
- the actual context-consumption point cannot be identified reliably;
- evaluator calls can only use a remote endpoint while remote use is disabled;
- a signal cannot be attributed to a specific memory without guessing;
- schema version conflicts with another in-flight migration;
- utility gains require weakening claim validity, authorization, or provenance;
- active ranking fails any safety, latency, or correction gate;
- pair rows cannot remain within the configured sparse cap;
- the target deployment cannot provide a stable viewer identity.

The safe resolution is always to preserve baseline recall and keep Echo in
shadow mode.
