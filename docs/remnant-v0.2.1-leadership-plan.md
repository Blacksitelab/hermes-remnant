# Remnant v0.2.1 leadership plan

Status: implementation handoff

Implementation status: WP-01, WP-02, the production-ranker portion of WP-03,
the first calibrated-lane portion of WP-04, the evidence safeguard in WP-05,
and the named setup profiles in WP-09 are implemented on the current branch.
Held-out corpus expansion, tokenizer-aware allocation, large-corpus benchmark,
ANN selection, provider-neutral comparison, and optional profile/session
synthesis remain gated follow-up work.

Repository scope: `Blacksitelab/hermes-remnant` only

Baseline: Remnant `v0.2.0`, commit
`fe3b8bc00310e7fa4e84526cdbf0d086c3b7c293`, schema 14

Objective: make Remnant the most effective and efficient Hermes memory
provider by unifying every recall route, validating it end to end, and closing
the remaining correctness, calibration, scale, and setup gaps.

This plan is deliberately written so a lower-cost implementation model can
execute it without inventing architecture or acceptance criteria.

## Rules for the implementation model

1. Modify only `Blacksitelab/hermes-remnant`. Treat Hermes and every competing
   provider repository as read-only references.
2. Start every branch from current `main`; use the `codex/` prefix.
3. Implement one work package per pull request in the dependency order below.
4. Do not perform opportunistic refactors. A changed file must be named in the
   work package or justified in the PR description.
5. Preserve the Hermes `MemoryProvider` contract and keep `sync_turn()`
   non-blocking.
6. Apply authorization before retrieval, ranking, logging, metrics, or model
   calls. Inaccessible memories must never consume top-k positions.
7. Treat all recalled text as untrusted data. Never render it as instructions.
8. Preserve immutable evidence and audit history. Superseding a claim must not
   delete its source memory.
9. Every migration must be additive, restartable, and tested against a copied
   schema-14 database.
10. Do not add a mandatory network dependency for keyword recall.
11. Run the package-specific tests plus the complete validation matrix before
    requesting review.
12. Update this document's checklist in the same PR; do not mark an item done
    without linking its test or measurement.

## Optimum default profile

The following defaults are part of the v0.2.1 baseline and must remain aligned
between `RemnantConfig`, `get_config_schema()`, examples, and tests:

| Setting | Default | Reason |
|---|---:|---|
| `structured_claim_extraction_v2` | `true` | Preserve time, scope, modality, and provenance at ingest |
| `claim_reconciliation_enabled` | `true` | Prevent silent accumulation of incompatible active truth |
| `claim_aware_ranking_enabled` | `true` | Resolve versions before selection and injection |
| `resolved_context_enabled` | `true` | Label uncertainty and fence recalled data |
| `recent_turn_overlay_enabled` | `true` | Guarantee useful read-after-write recall while extraction remains async |
| `relation_evidence_enabled` | `true` | Exclude unsupported or historical graph relations |
| `ranking_profile` | `claims-v1` | Make the active algorithm observable and reproducible |
| `runtime_identity_enabled` | `false` | Requires stable gateway identity; anonymous fallback is session-scoped |

Explicit values already present in `remnant.json` remain authoritative. Never
overwrite a user's `false` during load or setup. Runtime identity may become a
future default only after Hermes guarantees a stable user/platform identity or
Remnant supplies a validated local-profile fallback that survives sessions.

## Target recall architecture

All automatic and tool-driven recall must call one application service:

```text
RecallRequest
  -> stable runtime identity
  -> authorization/profile scope
  -> lexical, semantic, and graph candidate discovery
  -> pending-turn overlay
  -> claim/evidence enrichment
  -> temporal, conditional, and conflict resolution
  -> lane-calibrated relevance and bounded evidence ranking
  -> version grouping and diversity selection
  -> token-budgeted untrusted-data rendering or structured tool result
  -> diagnostics
```

`prefetch`, `memory_search`, `memory_reflect`, evaluation, and any graph-backed
recall must not reimplement or skip these stages.

## Dependency and PR order

```text
WP-01 unified recall service
  -> WP-02 migrate every recall consumer
      -> WP-03 end-to-end evaluation
          -> WP-04 calibrated ranking
          -> WP-05 evidence-aware reconciliation
              -> WP-06 context efficiency
                  -> WP-07 scale gates and ANN decision
                      -> WP-08 product and provider integration
                          -> WP-09 release evidence
```

WP-04 and WP-05 may be developed in parallel only after WP-03 lands. Everything
else is sequential.

## WP-01: Introduce one recall service

Suggested PR: `refactor: centralize authorized claim-aware recall`

Primary files:

- add `remnant/recall.py`;
- update `remnant/search.py`, `remnant/resolve.py`, `remnant/ranking.py`;
- add `tests/test_recall_service.py`.

Implementation:

1. Add immutable `RecallRequest` and `RecallResponse` dataclasses. Request fields
   must include query, effective actor, session, strategy, result limit, profile
   scope, query time, token budget, current messages, and output mode.
2. Add `RecallService.recall(request)`. Move orchestration into it but keep the
   existing search, resolver, ranker, and compiler as focused components.
3. Return selected results plus stage diagnostics: candidate counts, filtered
   counts, timings, degraded mode, ranking profile, and empty-result reason.
4. Centralize deadline checks. Remote embedding may time out; authorized BM25
   results must remain available.
5. Resolve claims before top-k selection. A superseded result may appear only
   for explicit historical intent or inside a labelled unresolved group.
6. Deduplicate pending turns against committed evidence and current messages.
7. Do not catch broad exceptions around the entire correctness pipeline. Catch
   errors per optional stage, record degradation, and retain the last safe result.

Acceptance tests:

- an inaccessible high-scoring row never enters candidates returned by the
  service;
- a current claim beats its superseded version before top-k truncation;
- a historical query can retrieve a superseded time-valid claim;
- an unresolved conflict retains both labelled evidence versions;
- embedding timeout returns lexical results inside the configured deadline;
- the next turn is recallable from the overlay without waiting for extraction;
- diagnostics contain no raw private memory or external user identifier.

## WP-02: Migrate every recall consumer

Suggested PR: `fix: make all memory recall conflict aware`

Primary files:

- `remnant/prefetch.py`;
- `remnant/tools.py`;
- `remnant/reflect.py`;
- `remnant/__init__.py`;
- relevant tool, prefetch, and reflection tests.

Implementation:

1. Make prefetch a thin adapter from Hermes arguments to `RecallRequest`.
2. Make `memory_search` use the service and return resolution status, validity,
   scope, provenance, ranking explanation, and grouped conflicting evidence.
3. Make `memory_reflect` synthesise only over the service's resolved result.
   Its prompt must preserve uncertainty labels and cite source memory IDs.
4. Route graph search through the same authorization and evidence filtering.
5. Remove duplicate token budgeting and context formatting from prefetch.
6. Preserve tool response compatibility: existing fields remain; new fields
   are additive.

Acceptance tests:

- prefetch, search tool, and reflect receive the same ordered memory IDs for an
  equivalent request;
- none of those routes exposes superseded truth for a present-tense query;
- reflect cannot state one side of an unresolved conflict as settled fact;
- graph strategy cannot bypass profile or visibility scope;
- existing Hermes tool schemas and JSON serialization remain compatible.

## WP-03: Replace the synthetic-only evidence path

Suggested PR: `test: exercise real provider recall end to end`

Primary files:

- `remnant/evaluation/runner.py`;
- `remnant/evaluation/schema.py`;
- `evaluation/cases/` and `evaluation/baselines/`;
- `docs/evaluation.md`;
- evaluator tests.

Implementation:

1. Make the evaluator instantiate `RemnantMemoryProvider` and exercise its real
   lifecycle rather than calling search and resolution directly.
2. Ensure retrieval and context layers invoke `RecallService` and therefore the
   production ranker.
3. Add at least 120 held-out natural-language cases without unique query tokens
   copied verbatim into the answer memory.
4. Include pronouns, entity aliases, paraphrases, noisy tool output, reordered
   facts, ambiguous dates, recurring conditions, explicit corrections, weak
   inferred updates, and distractors from the same subject/predicate.
5. Add an answer layer adapter with deterministic fixtures for CI and an
   optional real-model mode for release evaluation.
6. Record extraction accuracy separately from retrieval accuracy so a seeded
   claim cannot hide extraction failure.
7. Add negative controls: stale-only, unauthorized-only, and random rankers must
   fail the appropriate gates.

Required metrics:

- extraction precision/recall for structured fields;
- recall@1/3/5, MRR, nDCG@5;
- answer correctness and confidently wrong answer rate;
- stale-claim exposure and unresolved-conflict collapse rate;
- duplicate top-k occupancy and context precision;
- injected tokens and model-call count;
- p50/p95 per stage and full prefetch latency;
- degraded-mode success rate.

Do not make a competitive claim from this corpus. Provider comparison requires
the same held-out inputs and budgets through provider-neutral adapters.

## WP-04: Calibrate retrieval and ranking

Suggested PR: `perf: calibrate claim-aware hybrid ranking`

Primary files: `remnant/search.py`, `remnant/ranking.py`, evaluator baselines,
ranking tests.

Implementation:

1. Stop normalising heterogeneous scores by one absolute maximum.
2. Preserve the native score and lane name for BM25, cosine, graph, overlay,
   and expansion candidates.
3. Normalise within each lane using a deterministic, documented method, then
   fuse lanes with reciprocal-rank fusion or weights selected on the training
   split only.
4. Apply temporal applicability and conflict lifecycle before evidence quality.
5. Bound trust/confidence/verification/corroboration so quality cannot rescue an
   irrelevant candidate.
6. Group semantic paraphrases and claim versions before diversity penalties.
7. Version all weight sets and emit the version in diagnostics/evaluation.

Gates:

- held-out recall@5 must not regress more than `0.01`;
- stale exposure and duplicate occupancy must not increase;
- no category may lose more than `0.03` MRR;
- ranking remains deterministic for identical database and request state.

## WP-05: Strengthen reconciliation with evidence hierarchy

Suggested PR: `feat: require evidence to supersede durable claims`

Primary files: `remnant/reconcile.py`, `remnant/claims.py`, `remnant/ingest.py`,
schema migration only if required, reconciliation tests.

Implementation:

1. Replace the single confidence threshold with explicit deterministic rules.
2. Treat direct user corrections as stronger than inferred updates.
3. Require corroboration, verification, trusted source authority, or explicit
   correction before replacing a durable verified claim.
4. Keep conditional alternatives active when scopes are disjoint.
5. Use event time for truth applicability and observation time for evidence
   ordering; never silently substitute one for the other.
6. Send only ambiguous cases to an optional bounded classifier. Validate its
   response and fall back to `unresolved` on error.
7. Persist decision rule, inputs, confidence, model/version, and affected claim
   IDs in the audit record.

Acceptance tests must cover false corrections, low-confidence later statements,
trusted older facts, recurring schedules, future commitments, negation, and
three-way conflicts.

## WP-06: Make context allocation token-accurate

Suggested PR: `perf: budget resolved memory context by evidence class`

Primary files: `remnant/context.py`, provider tokenizer adapter, context tests.

Implementation:

1. Accept the deployment tokenizer when Hermes exposes one; retain the current
   conservative counter as an offline fallback.
2. Allocate the budget rather than repeatedly halving remaining space:
   `60%` current applicable claims, `20%` unresolved/conditional evidence,
   `15%` supporting provenance, `5%` recent-turn overlay. Redistribute unused
   capacity deterministically.
3. Prefer complete compact claims over truncated long documents. Use passage or
   summary text before character truncation.
4. Include provenance IDs and uncertainty labels while avoiding verbose ranking
   diagnostics in the LLM context.
5. Fix and test Unicode truncation output.

Gates: never exceed the configured budget; median injected context below 1,000
tokens; p95 below 2,000; no prompt-fence escape in adversarial tests.

## WP-07: Establish scale gates before adding ANN

Suggested PR: `perf: publish retrieval scale envelope`

Primary files: benchmark utilities under `evaluation/`, maintenance health,
architecture roadmap, performance tests excluded from ordinary unit CI.

Implementation:

1. Generate reproducible 5k, 25k, 100k, and 1m-memory stores with realistic
   content lengths, identities, visibility, claim versions, and embedding gaps.
2. Measure database size, ingest throughput, exact-vector p50/p95, full prefetch
   p50/p95, peak memory, and recall@5.
3. Publish hardware, Python, SQLite, embedding dimensions, warm/cold cache state,
   and configuration with each report.
4. Add an ANN implementation only if exact-vector p95 breaches the documented
   supported target. Keep exact search as the oracle.
5. Any ANN candidate must lose no more than `0.01` recall@5 and must demonstrate
   a material latency or memory improvement.

## WP-08: Close product-level gaps

Suggested PRs should remain independent and are ordered by measured need:

1. `feat: add tiered document recall` — document abstract, overview, and passage
   levels inspired by the capability gap exposed by OpenViking/ByteRover.
2. `feat: add derived user representation` — compact, source-linked profile
   synthesis refreshed on a configurable cadence; never replace raw evidence.
3. `feat: capture bounded tool outcomes` — extract durable outcomes, not raw
   secrets or arbitrary tool payloads; maintain explicit allow/deny policy.
4. `feat: add session summaries as evidence` — source-linked summaries with
   expiry and invalidation when underlying turns change.
5. `feat: add memory feedback signals` — explicit helpful/unhelpful feedback
   influences bounded ranking quality and remains reversible.

Each feature requires an ablation showing improvement on a failing evaluation
category before implementation. Do not add it solely for provider parity.

## WP-09: Setup, migration, and release evidence

Suggested PR: `release: validate remnant leadership profile`

Implementation:

1. Add named setup presets: `claim_aware` (default), `legacy`, and
   `claim_aware_shadow`.
2. Print the resolved preset and deviations in health output.
3. Warn when runtime identity is enabled but the current gateway has no stable
   identity; explain that recall becomes session-scoped.
4. Add a dry-run config migration command that reports changed defaults without
   rewriting explicit values.
5. Run a redacted production-shaped soak for at least seven days and publish
   aggregate metrics only.
6. Run provider-neutral comparisons without committing or opening work in any
   external repository.
7. Update README, changelog, provider comparison, migration guide, and release
   notes with measured claims only.

## Complete validation matrix

Run after every work package:

```bash
python -m pytest -q
python -m ruff check remnant tests
python -m remnant.evaluate --cases evaluation/cases/leadership.jsonl --layer retrieval
python -m remnant.evaluate --cases evaluation/cases/leadership.jsonl --layer context
python -m build
git diff --check
```

Also run the package-specific failure, performance, migration, or answer layer
described above. Record commands and results in the PR body.

## Leadership release gates

Remnant may be described as the leading provider for local, auditable temporal
memory only when all gates pass:

| Dimension | Gate |
|---|---|
| Retrieval | Held-out recall@5 >= 0.85 and MRR >= 0.70 |
| Answer quality | Confidently wrong answers < 5% overall and < 10% per category |
| Temporal correctness | Stale-claim exposure < 1% |
| Conflict safety | Unresolved-conflict collapse < 1% |
| Isolation | Zero unauthorized memory exposure in the full matrix |
| Read-after-write | 100% eligible next-turn recall |
| Efficiency | Median context < 1,000 tokens; p95 <= configured 2,000-token cap |
| Latency | Local lexical p95 < 100 ms; full hybrid p95 < 500 ms at supported scale |
| Resilience | Useful lexical recall when embedding/extraction endpoints fail |
| Durability | No acknowledged-turn loss across crash/retry/shutdown tests |

An overall “best Hermes memory provider” claim additionally requires a reviewed
provider-neutral comparison. Until then, state the narrower measured claim and
publish limitations beside results.

## Definition of done

- all recall consumers use `RecallService`;
- claim-aware defaults and explicit rollback overrides are tested;
- production ranker and provider lifecycle are exercised by evaluation;
- held-out, natural-language, negative-control, isolation, and failure suites
  meet their gates;
- scale envelope and token/latency measurements are published reproducibly;
- migrations and rollback preserve evidence and user configuration;
- full tests, Ruff, evaluator, build, and `git diff --check` pass;
- documentation contains no unsupported competitive claim;
- no external repository was modified.
