# Changelog

## 0.3.2 - simpler recall, unchanged capabilities

- Route recent unprocessed turns and conversation deduplication through the
  shared recall service for both tools and automatic prefetch.
- Consolidate search-result filtering and remove unused context formatting and
  allocation implementations. Ownership and scope checks remain in place.
- Remove unused Pydantic and Watchfiles runtime dependencies.
- Reuse the query norm during exact vector scoring and compute ranking bounds
  once per score lane. Keep the same ranking formula and vector format.
- Report the context compiler's token count in prefetch diagnostics.
- Preserve committed recall if the optional pending-turn lookup fails.

All existing tools, dreams, Echo, graph retrieval, claims, imports and recovery
remain available with unchanged defaults. No schema or embedding migration.

## 0.3.1 - profile state and fleet recovery

- Scoped vault reindexing no longer forgets notes outside the selected scope.
- Schema 17 gives dream cooldowns, budgets, topic caches and night watermarks
  a profile owner, and separates thread ownership from author attribution.
  Unambiguous legacy ownership is migrated; ambiguous rows require an explicit map.
- Runtime identities import memory files from the configured filesystem profile.
- Add offline recovery from explicitly ordered, read-only store snapshots,
  retaining memory IDs and remapping colliding turn IDs and their evidence links.
  Recovery refuses conflicting owners and publishes only a verified new database.
- Correct the 0.3.0 validation scope: its 6,223-row shared-store check did not
  cover the other production databases or all services.

## 0.3.0 - profile isolation and retrieval efficiency

- Enforce profile ownership across search, context, graph, imports, threads,
  feedback, vault indexing, dreams, and model-backed claim backfill. Legacy
  visibility labels no longer grant cross-profile access.
- Migrate vault mappings to schema 16 with profile/path keys; preserve source
  rows and existing owners. Include the profile in runtime identity v2.
- Preserve high-similarity corrections and uncertain paraphrases; retain
  duplicate observation provenance and reject unsupported duplicate labels.
- Stream float32 vectors over the full eligible corpus, filter incompatible
  vectors and low-scoring results, and bound prefetch database work by time.
- Bound query caches and queued work, invalidate on committed evidence changes,
  and suppress repeated context only after delivery.
- Clear stale embeddings on content changes and retry missing derived vectors;
  periodically compact disposable caches, diagnostics, and Echo data.
- Preserve and test the existing model-backfill utility from BSL-AI.
- Isolate test diaries and correct scale benchmark dimensions and measurements.

## 0.2.2 - faster extraction

### Changed

- Bounded extraction input and model context to keep single-turn work within a
  predictable background budget.
- Disabled model reasoning by default for extraction and added native structured
  JSON output controls for Ollama and its OpenAI-compatible endpoint.
- Reduced generated claim fields and moved entity discovery to the existing
  local GLiNER/regex pipeline while retaining legacy response compatibility.
- Kept extracted-memory visibility provider-controlled and added defensive
  invalid-response retry handling.

### Verification

- 403 automated tests pass locally.
- Ruff, bytecode compilation, `git diff --check`, and package build pass.

## 0.2.1 - claim-aware recall

### Changed

- Made structured claims, reconciliation, claim-aware ranking, resolved
  context, recent-turn overlay, and evidence-backed relation traversal the
  defaults for new configurations.
- Kept runtime identity opt-in because gateways without a stable platform user
  identity intentionally fall back to session isolation, which prevents
  cross-session recall.
- Added a dependency-ordered v0.2.1 implementation handoff plan focused on one
  recall pipeline, provider-neutral evidence, calibrated ranking, and scale.
- Added a held-out adversarial release corpus with negative controls and CI
  recall/staleness gates.
- Added token-counter-aware evidence-class context allocation and a disposable
  scale-envelope benchmark for exact-vector/claim-aware recall measurements.

## 0.2.0 - release candidate

### Added

- Hermes lifecycle parity for queued prefetch, session end, pre-compression,
  built-in memory writes, delegation, session switching, and backup paths.
- Runtime agent/workspace identity scoping with non-primary write protection.
- Schema 14 claim metadata for observation/event time, validity, scope,
  modality, conflict type, resolution status, extractor version, and source
  turn provenance.
- Opt-in structured extraction, conservative claim reconciliation, claim-aware
  ranking, provenance-aware context compilation, and a bounded recent-turn
  read-after-write overlay.
- Bounded health signals for schema, claim coverage, unresolved states, and
  extraction age; CI for supported Python versions, tests, release-track Ruff,
  and package builds.

### Compatibility and rollout

- Existing databases migrate additively to schema 14; no memory rows are
  deleted or re-scoped by the migration.
- New correctness behaviors default to `false` for compatibility. Enable the
  seven release-track flags together for a canary, measure the evaluation and
  health gates, then make the rollout decision per deployment.
- Rollback is configuration-only: disable the flags and retain the evidence
  and claim metadata for later inspection.

### Verification

- 379 automated tests pass locally.
- The whole repository passes Ruff, `git diff --check`, bytecode compilation,
  and `python -m build` for the 0.2.0 wheel and sdist.
