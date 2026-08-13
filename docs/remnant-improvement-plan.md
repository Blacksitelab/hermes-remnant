# Remnant hardening and retrieval improvement plan

Status: superseded for new implementation work

Packages 1 through 5 were substantially delivered by PRs #35 and #37, with
the subsequent prefetch-responsiveness work delivered by PR #38. For the
current baseline and the remaining competitive roadmap, use
`docs/remnant-leadership-implementation-plan.md`. Keep this document as the
audit record and detailed rationale for the completed hardening work.

Audit baseline: `hermes-remnant@0b6cab6` and `hermes-agent@794d6c4`

Validation baseline: 333 tests passing

## Objective

Make Remnant a safe, predictable, and efficient default memory provider for
Hermes Agent. The work is ordered so privacy and durable-state correctness land
before ranking or performance changes.

This document is deliberately split into small, independently reviewable work
packages. An implementation agent should complete one package at a time and
must not combine packages 1 through 4 into a single pull request.

## Current integration model

Hermes initializes Remnant through the `MemoryProvider` interface. Before a
model call Hermes invokes `prefetch(query, session_id=...)`; after a completed
turn it invokes `sync_turn(...)` on a serialized background worker. Remnant
persists turns and memories in a shared SQLite database, extracts durable facts
with a configured LLM, and retrieves context through FTS5 BM25, exact cosine
similarity, reciprocal-rank fusion, and entity-graph expansion.

The architecture is a good small-to-medium-corpus baseline. The main risks are
inconsistent authorization predicates, ambiguous durable job state, and
lifecycle operations that update related tables independently.

## Non-negotiable invariants

Every implementation package must preserve these invariants:

1. A private memory is visible only to its owning agent.
2. No automatic process may broaden a memory's visibility.
3. Authorization and profile scope are applied before candidate ranking and
   limiting.
4. A failed or incomplete external scan never implies that source data was
   deleted.
5. A durable job has an explicit terminal state, including successful jobs that
   produce zero facts.
6. A memory lifecycle change updates its claims, graph evidence, embeddings,
   and audit record atomically or leaves all of them unchanged.
7. Retrieval quality changes are evaluated against a versioned baseline before
   becoming defaults.

## Package 1: enforce memory privacy boundaries

Priority: P0

Suggested title: `fix: enforce memory privacy boundaries`

### Problem

The night dream corpus is loaded without a visibility restriction. Private
memories owned by different agents can therefore be paired, included verbatim
in a cloud request, and merged into a new shared memory. Separately, an explicit
empty `profile_scope` supplied through `memory_search` overrides a configured
scope and disables the restriction.

Visibility behavior also differs among keyword, semantic, and graph retrieval.

### Implementation

- Add one central scope-policy module that accepts viewer agent, owner agent,
  memory visibility, source, configured profile scope, and requested profile
  scope.
- Define private/shared/fleet/vault behavior in that module and reuse the same
  predicate in all retrieval and consolidation paths.
- Treat configured profile scope as the maximum allowed scope. A requested
  scope may narrow it but must never broaden or disable it.
- Remove `profile_scope` from the model tool schema if callers have no valid
  reason to narrow it. Otherwise compute an intersection with path-boundary
  matching.
- Restrict cross-agent dream candidates to explicitly shareable visibility.
- Compute merged visibility from the least permissive input. Reject a merge
  when there is no valid shared visibility.
- Add a configuration allowlist controlling which visibility/source classes
  may be sent to a cloud dream endpoint. Default to shared/fleet only.
- Delimit memory content as untrusted data in dream prompts and validate all
  returned IDs against the candidate set before performing an action.

### Acceptance tests

- Two identical private memories owned by different agents are never paired.
- Private memory text is never present in the mocked cloud request body.
- A dream merge never produces broader visibility than either input.
- `profile_scope=[]` cannot bypass a non-empty configured scope.
- `Projects/a.md` is allowed by `Projects`; `Projects-Secret/a.md` is not.
- The same agent/visibility matrix passes for keyword, semantic, graph,
  reflection, and prefetch retrieval.

## Package 2: make extraction jobs durable

Priority: P0

Suggested title: `fix: make extraction queue durable`

### Problem

Extraction HTTP and parsing failures currently return an empty fact list. The
worker interprets that as success and deletes the queue row. A legitimate
zero-fact turn has no durable completion marker, so the startup sweep discovers
and requeues it on every restart.

### Schema migration

Add extraction lifecycle fields, either directly to `turns` or to a durable job
table:

- `status`: pending, running, completed, retry_wait, dead_letter
- `attempts`
- `next_attempt_at`
- `started_at`
- `completed_at`
- `last_error_class`
- `last_error_message` with a bounded length
- optional `fact_count`

### Migration rollback and recovery

Schema migrations must be forward-only and transactional: each migration runs
inside a transaction, commits only after its post-migration checks pass, and
must not drop or rewrite user data in place. Before applying a migration,
create a verified SQLite backup and record the source schema version. Run
`PRAGMA integrity_check` on both the source database and the backup before
proceeding.

There is no automatic in-place downgrade. If a migration fails or the upgraded
database fails integrity or startup checks, stop the provider, preserve the
failed database for diagnosis, and restore the verified backup to a new
database path. Re-run the previous provider version against that restored copy,
then retry the migration only after the failure is understood. The migration
test must cover interruption before commit and confirm that the original
database and queued extraction work remain recoverable.

Use a new schema version and make migration idempotent.

### Implementation

- Return a typed extraction result that distinguishes success from failure.
- Mark a successful zero-fact extraction `completed` with `fact_count=0`.
- Retry transient network, timeout, 429, and 5xx failures with exponential
  backoff and jitter.
- Dead-letter permanent response-shape or validation failures after a bounded
  number of attempts.
- On startup, return stale `running` jobs to a retryable state.
- Keep job diagnostics in the health report: pending count, oldest age,
  retries, dead letters, and last failure.
- Do not store full conversation text in error messages or logs.
- Make shutdown bounded. Preserve an in-flight job as retryable if the request
  cannot finish before shutdown.

### Acceptance tests

- Successful zero-fact turns are not requeued after restart.
- A transient HTTP failure is not marked complete.
- Retry timing and maximum attempts are deterministic under a fake clock.
- A stale running job is recovered after simulated process death.
- Provider shutdown completes within the configured bound while a mocked HTTP
  call is blocked.
- Existing databases migrate without losing queued work.

## Package 3: make vault reconciliation fail closed

Priority: P0

Suggested title: `fix: make vault reconciliation fail closed`

### Problem

When the configured vault root does not exist or cannot be read, the indexer
reconciles against an empty path set and marks every indexed vault memory
forgotten. A mount outage or configuration typo therefore looks like mass
deletion.

### Implementation

- Separate enumeration from reconciliation.
- Create a scan record with a generated ID and states `running`, `complete`, or
  `failed`.
- Enumerate readable files and index changed content under the scan ID.
- Reconcile missing paths only after a complete traversal succeeds.
- If the root is missing, unreadable, changes identity, or traversal has an
  unhandled error, mark the scan failed and do not forget anything.
- Return structured status including indexed, skipped, forgotten, failed,
  incomplete, and error category.
- Optionally require explicit confirmation for unusually large deletion ratios.
  Keep this as a safety valve, not a substitute for scan completeness.

### Acceptance tests

- A missing or temporarily unmounted root forgets zero memories.
- A traversal error after reading some files forgets zero memories.
- A successful complete scan forgets genuinely deleted files.
- An interrupted scan can be restarted without duplicate memories.
- Existing vault hashes and stable passage IDs remain unchanged when content is
  unchanged.

## Package 4: normalize LLM endpoint behavior

Priority: P0/P1

Suggested title: `refactor: normalize LLM endpoint behavior`

### Problem

Extraction sends and parses Ollama-native `/api/chat` messages, while setup
examples also advertise OpenAI-compatible `/v1/chat/completions`. Reflection
defaults to the extraction endpoint but parses only the OpenAI-compatible
response shape. Endpoint configuration can therefore appear valid while
silently producing no facts or failed reflections.

### Implementation

- Add a shared `LLMClient` used by extraction, reflection, and dreams.
- Support explicit `ollama_native` and `openai_compatible` protocols.
- Normalize responses to a typed object containing content, model, usage when
  available, and finish reason.
- Validate endpoint/protocol combinations during provider initialization.
- Use separate connect, read, write, and pool timeouts.
- Make retries explicit and restricted to idempotent calls.
- Never convert a transport or schema error into a successful empty result.
- Preserve backward compatibility by inferring the protocol from known endpoint
  paths once, while logging a migration warning. Persist an explicit protocol
  on the next config save.

### Acceptance tests

- Native Ollama and OpenAI-compatible request bodies match their APIs.
- Both response shapes produce the same normalized content.
- Invalid JSON, missing content, timeout, and 5xx responses return typed errors.
- Extraction and reflection work with the documented defaults.
- Secrets and full prompts do not appear in error logs.

## Package 5: unify scoped retrieval before ranking

Priority: P1

Suggested title: `fix: unify scoped retrieval`

### Problem

Candidate authorization differs by search strategy, and some profile and
visibility filters run after top-k selection. Relevant allowed results can be
displaced by inaccessible candidates before the filter runs.

### Implementation

- Express the package 1 scope policy as reusable SQL fragments with bound
  parameters.
- Apply status, viewer, visibility, source, locked-note, and profile predicates
  in SQL before `ORDER BY` and `LIMIT`.
- Reuse the predicate in BM25, embedding corpus selection, graph traversal,
  reflection, prefetch, and tool searches.
- Add composite indexes based on measured query plans rather than individual
  columns alone.
- Clamp public tool inputs: search limit, graph depth, thread limit, import
  batch size, and reflection result count.
- Return a stable, redacted representation for locked notes instead of loading
  content and masking it later where practical.

### Acceptance tests

- An allowed result outside the unfiltered top 100 is still returned after
  scoped ranking.
- No strategy returns a row rejected by the central policy.
- Query-plan tests or documented `EXPLAIN QUERY PLAN` fixtures demonstrate use
  of the intended indexes for representative corpus sizes.
- Tool input limits cannot allocate or return unbounded data.

## Package 6: introduce evaluated quality-aware ranking

Priority: P1

Suggested title: `feat: add evaluated memory quality ranking`

### Problem

Trust feedback and time decay update `trust_score`, but retrieval ranking does
not use that field. Current ranking combines BM25, cosine, reciprocal-rank
fusion, and source multipliers. A memory can be distrusted without its retrieval
position changing.

### Implementation

- Stop mutating trust scores as a side effect of retrieval.
- Define a bounded quality factor using trust, extraction confidence, verified
  status, seen count, source prior, and freshness.
- Keep relevance and quality as separate score components and expose them in
  debug/evaluation output.
- Avoid a hard freshness penalty for durable preferences and identity facts.
  Apply type-aware freshness only where evidence supports it.
- Add a versioned evaluation corpus covering lexical matches, paraphrases,
  contradictions, outdated facts, vault passages, and cross-agent visibility.
- Record recall@k, MRR, nDCG, empty-result rate, and p50/p95 latency.
- Ship the new ranking behind a feature flag. Make it default only if it meets
  documented quality and latency thresholds.

### Acceptance tests

- Negative feedback lowers rank when relevance is otherwise equal.
- Verified corroborated facts beat low-confidence duplicates.
- Strong relevance cannot be erased by a small quality difference.
- Search performs no database writes.
- Baseline and candidate evaluation reports are reproducible.

## Package 7: make memory lifecycle operations transactional

Priority: P1

Suggested title: `fix: make memory lifecycle updates transactional`

### Problem

Dream merges supersede original memories before a replacement is safely stored.
Relations retain one source memory even when several memories provide evidence,
and relations can survive after their evidence is forgotten. Structured claims
are not consistently transitioned when memories are edited, merged,
superseded, or forgotten.

### Schema and service changes

- Introduce a memory lifecycle service as the only write path for create,
  update, forget, supersede, and merge.
- Add `relation_evidence` keyed by relation and memory, with strength and active
  state. Derive relation support from active evidence.
- Transition claims alongside their source memory. Create a new claim version
  when edited content changes the projection.
- Create replacement memory, claim, embedding, entity links, and audit rows in
  one transaction before superseding originals.
- If embedding generation is remote, perform it before the transaction and
  validate the result; keep database mutation atomic.

### Remote embedding failure semantics

Remote embedding calls are completed and validated before the lifecycle
transaction opens. A timeout, transport error, rate limit, or invalid vector
therefore produces no lifecycle writes: originals remain active and the caller
retries with bounded exponential backoff. Once the transaction begins, it
performs database-only work; any SQLite or validation failure rolls back the
entire replacement, claim, relation-evidence, entity-link, embedding, and
audit set. SQLite crash recovery provides the same rollback guarantee after a
process exit. After the retry budget is exhausted, retain the original memory,
record a bounded failure diagnostic, and leave the operation retryable or
dead-lettered according to the package's job policy.

### Acceptance tests

- Forced failure at every merge stage leaves original memories active.
- Forgetting the last evidence removes a relation from traversal.
- Forgetting one of several evidence memories preserves the relation and
  decrements its support.
- Editing, merging, superseding, and forgetting produce consistent claim state.
- Audit entries identify actor, source IDs, replacement ID, and visibility
  decision.

## Package 8: scope threads and complete the Hermes contract

Priority: P1/P2

Suggested title: `feat: scope Remnant threads and Hermes context`

### Problem

Threads are stored globally with `added_by` but no enforced owner or visibility.
Hermes currently routes provider tools without runtime session or agent identity,
so Remnant falls back to initialization state. Remnant accepts conversation
messages during prefetch, but current Hermes only supplies the query and session
ID. Remnant also does not implement Hermes' `queue_prefetch()` hook.

### Remnant changes

- Add owner, visibility, and optional session/profile scope to threads.
- Apply the central scope policy to list, update, stale, resolve, and search.
- Implement a bounded per-session prefetch cache keyed by normalized query,
  viewer scope, embedding model, and a memory-generation counter.
- Implement `queue_prefetch()` without allowing background work to outlive its
  session indefinitely.
- Keep a bounded recent-injection set when conversation-message dedup context is
  unavailable.

### Hermes changes

These changes belong in `NousResearch/hermes-agent` and should be proposed in a
separate PR:

- Pass runtime `session_id`, profile/agent identity, platform user, workspace,
  and agent context to `handle_tool_call()`.
- Consider extending `prefetch()` with an optional messages/context parameter
  using signature introspection for backward compatibility.
- Add a provider contract test fixture that Remnant can run against the current
  Hermes interface.

### Acceptance tests

- One agent cannot list or mutate another agent's private threads.
- Concurrent gateway sessions use their runtime identities, not provider startup
  defaults.
- A queued prefetch is reused only when query and authorization scope match.
- Cache invalidation occurs after a relevant memory write or visibility change.
- Existing providers that implement the old Hermes signatures continue to work.

## Package 9: observability, scaling gates, and quality controls

Priority: P2

Suggested title: `chore: add Remnant quality and performance gates`

### Implementation

- Expand health output with extraction queue age, dead letters, vault scan
  status, database size, active corpus size, embedding coverage by model,
  prefetch outcome rates, and p50/p95 latency.
- Make `is_available()` check local prerequisites without network calls:
  configuration validity, database parent writability, and optional dependency
  availability. Report degraded keyword-only operation separately.
- Measure exact semantic scan latency by corpus size. Introduce ANN only after
  the configured corpus or latency ceiling is exceeded.
- When ANN is added, retain exact scan as a fallback and evaluation oracle.
- Add a `gliner` optional dependency extra and prevent unexpected model download
  unless the feature is explicitly enabled.
- Add GitHub Actions for supported Python versions, pytest, Ruff, build, and an
  empty-to-current schema migration test.
- Fix all current Ruff findings and enforce the clean baseline in CI.
- Update README counts and feature claims, especially trust ranking, dedup
  reinforcement, endpoint protocol, passage retrieval, and structured claims.

### Acceptance tests and gates

- CI passes on every supported Python version.
- Ruff reports zero findings.
- The health command remains bounded on a production-sized database.
- Exact retrieval remains below the documented p95 target at the supported
  corpus ceiling.
- An ANN implementation must meet the documented recall delta and latency gain
  before it can become default.

## Suggested dependency order

```text
1 privacy policy
├── 3 vault reconciliation
├── 5 scoped retrieval
│   └── 6 quality ranking
└── 8 scoped threads and Hermes context

2 durable extraction ── 4 LLM adapter

7 transactional lifecycle depends on 1

9 observability and CI can start early, then be completed after 2-8
```

## Rollout strategy

1. Back up the SQLite database and validate integrity before every schema
   migration.
2. Land migrations disabled behind compatibility code where possible.
3. Deploy privacy and durability fixes before enabling new retrieval behavior.
4. Run the retrieval evaluator against a production-shaped redacted corpus.
5. Enable ranking and prefetch changes per profile, with rollback flags.
6. Observe at least one full extraction and vault-index cycle before removing
   compatibility paths.

## Definition of done

The plan is complete when:

- all P0 regression tests pass;
- every retrieval lane uses one authorization policy before ranking;
- extraction and vault indexing distinguish failure from empty success;
- lifecycle updates are transactional across projections;
- feedback measurably and safely affects ranking;
- Hermes passes runtime identity to memory tools;
- quality and latency changes are gated by reproducible evaluation; and
- CI enforces tests, lint, builds, and schema migrations.
