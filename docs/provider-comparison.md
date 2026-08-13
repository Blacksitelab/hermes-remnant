# Remnant positioning among Hermes memory providers

Snapshot: 2026-08-13. This is a capability comparison, not an apples-to-apples
quality benchmark. External-provider facts come from the current
[Hermes provider guide](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory-providers/).
Remnant's numerical claims come only from its committed deterministic corpus.

| Provider | Deployment described by Hermes | Stated differentiator | Remnant assessment |
|---|---|---|---|
| Remnant 0.2 | Local SQLite; optional local HTTP models | Structured temporal claims, deterministic reconciliation, evidence graph, transactional edits, recent-turn overlay | Strongest where privacy, auditability, qualified truth, and local rollback matter; less turnkey than a managed cloud service |
| Hindsight | Cloud or local embedded PostgreSQL | Knowledge graph and reflect synthesis | Mature graph/synthesis reference point; Remnant now has graph and reflect plus stricter temporal/conflict lifecycle controls |
| Mem0 | Cloud or self-hosted | Server-side LLM extraction and OSS modes | Easier hosted path; Remnant emphasizes bounded local retrieval and observable deterministic policy |
| Honcho | Cloud | Dialectic user modelling and session context | Rich managed user modelling; Remnant avoids mandatory cloud disclosure and recurring service cost |
| OpenViking | Self-hosted | Filesystem hierarchy and tiered loading | Strong document hierarchy; Remnant is currently stronger on claim lifecycle than hierarchical corpus navigation |
| Holographic | Local | HRR algebra and trust scoring | Lightweight local alternative; Remnant has broader explicit tools, evidence provenance, and temporal claims but more moving parts |
| Supermemory | Cloud or self-hosted | Context fencing, session graph ingest, multi-container | Strong ingestion/isolation reference; Remnant now fences context and hashes runtime identity, but still needs live Hermes-scale soak evidence |

## Where Remnant is strongest

- local-first operation with keyword-only degradation when model endpoints fail;
- immutable backing evidence plus versioned temporal and conditional claims;
- unresolved conflicts are grouped and labelled instead of silently becoming truth;
- edits, forgets, embeddings, claims, relation evidence, and audits share atomic
  lifecycle transactions;
- runtime identity, session overlay, authorization, and cache keys are scoped
  before recall;
- deterministic 240-case evaluation is committed and reproducible in CI.

## Remaining gaps before claiming overall leadership

- no independent cross-provider answer-quality benchmark yet;
- exact vector scan is intentionally retained until measured p95 justifies ANN;
- installation still relies on Hermes' third-party plugin flow rather than
  appearing in the bundled provider picker;
- model-backed extraction quality depends on the operator's configured model;
- production soak data across large multi-user Hermes gateways is not yet
  published.

## Release gates for a leadership claim

Do not claim that Remnant is “best” until the same held-out corpus is run through
each provider under equivalent privacy, latency, and token budgets. At minimum,
publish recall@5, answer correctness, stale-claim exposure, duplicate occupancy,
p50/p95 latency, injected tokens, model-call count, and failure-mode results.
