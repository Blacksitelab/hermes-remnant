# Changelog

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
