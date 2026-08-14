# Memory architecture roadmap

PR #33 establishes the safety and measurement foundation: a Hermes-compatible
provider boundary, read-only retrieval, bounded exact-vector recall, explicit
ownership, maintenance health, and repeatable retrieval evaluation.

The remaining structural upgrades are intentionally gated on evaluation data.
They must improve a versioned retrieval suite and stay within the prefetch
latency budget before becoming production defaults.

## Passage-level vault retrieval

Index long vault notes as heading-aware passages with a stable parent document
identifier, passage ordinal, heading path, and source offset. Retrieval should
select passages while the injected context exposes both the passage and parent
note provenance. Keep the current whole-note record as the migration fallback
until recall@k and context precision improve on the evaluation corpus.

## Claim model

Represent extracted facts as versioned claims with subject, predicate, object,
qualifiers, validity interval, confidence, provenance, and contradiction links.
The retrieval layer should select the highest-quality currently-valid claim,
while the UI/tool surface retains the historical record and explains why a
claim won. This replaces heuristic text-level contradiction handling; it is not
safe to infer or backfill automatically without a curated evaluation set.

## Approximate nearest-neighbour retrieval

Use an ANN index only once exact-vector scans exceed the configured corpus or
latency ceiling in production metrics. Compare ANN recall against the exact
scan baseline before enabling it, and retain a bounded exact fallback for small
or rebuilding indexes.

## Required operating metrics

- retrieval recall@k, MRR, and context precision from versioned cases;
- prefetch p50/p95 latency, deadline misses, and injection rate;
- embedding coverage, extraction-queue age, DB integrity, and FTS parity;
- explicit useful/wrong feedback, correction rate, and contradiction rate.

## Outcome-aware memory utility

The next measurement-gated research track is Remnant Echo: a bounded shadow
system for learning whether recalled evidence actually helps or harms Hermes on
a class of task. Its schema, lifecycle integration, resource limits, tests,
rollout gates, and ordered pull-request work packages are specified in
[`remnant-echo-implementation-plan.md`](remnant-echo-implementation-plan.md).
