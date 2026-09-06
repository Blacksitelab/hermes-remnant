# Remnant 0.3.2 simplification validation

Baseline: `47fd3fa` (0.3.1). This release preserves all feature defaults, tool
schemas, schema 17, stored memories and embedding compatibility.

The production Python modules lose 212 net lines. Recent-turn overlay and
conversation deduplication now live in the shared recall service, search lanes
share their final filtering, and unused formatting/allocation paths are gone.
Pydantic and Watchfiles were declared dependencies with no callers; both are
removed from package requirements. Echo, dreams, graphs, reflection, claims,
threads, vault indexing, imports and recovery remain available.

## Behaviour and packaging

- 455 tests pass (453 existing plus two shared-recall regressions).
- New regressions verify identical pending-turn context through tools and
  prefetch, profile/session exclusion, conversation deduplication, and retention
  of committed recall when pending-turn lookup fails.
- The 240 leadership and 120 held-out scenarios retain identical retrieval
  metrics: recall@5 and context precision 1.0, stale exposure zero. These are
  prepared-claim regression cases, not evidence of perfect extraction or answers.
- Whole-package Ruff, compilation and wheel/sdist build pass. A clean virtual
  environment imports every installed-wheel module and passes database health
  without either removed dependency installed.
- Schema, model, configured defaults and live memory data require no migration.

## Controlled performance

The existing benchmark uses 5,000 owned memories, 768-dimensional float32
vectors and seven probes on CPython 3.12.3. Embedding inference is an instant
stub. Before and after ran sequentially on the same machine.

| Measurement | 0.3.1 | 0.3.2 |
| --- | ---: | ---: |
| Median semantic scoring | 239.543 ms | 197.855 ms |
| Median provider prefetch | 255.519 ms | 215.123 ms |
| Peak Python scoring allocations | 0.540 MiB | 0.540 MiB |
| Context delivered | 7/7 | 7/7 |

Semantic scoring is 17.4% faster and provider prefetch 15.8% faster in this
controlled run. Top-100 IDs **and exact floating-point scores** are identical.
The query norm is computed once per scan, retaining the original accumulation
order and cosine formula. Ranking also computes each lane's bounds once.

This does not establish an equivalent improvement in whole-agent latency,
model-inference cost, total fleet RAM or final answer accuracy. Allocation
tracing is separate from timing and measures Python allocations only.

Reproduce against each checkout:

```bash
PYTHONPATH=. python evaluation/benchmarks/retrieval_hardening.py
```

[Machine-readable comparison](../../evaluation/baselines/simplification-2026-09-06.json).
The [first-principles analysis](../memory-first-principles.md) describes a
separate, bounded accuracy experiment; it is not shipped behaviour.
