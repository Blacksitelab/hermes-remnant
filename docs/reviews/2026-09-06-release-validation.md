# Remnant 0.3.0 validation

Baseline: `142aff3` (0.2.2). Release: profile isolation, correction retention,
compact semantic scoring, deadline-aware prefetch, and bounded derived caches.

## Results

- 443 local tests pass; Ruff, compilation, wheel/sdist builds, and whitespace checks pass.
- The 240 leadership and 120 held-out cases retain recall@5 and context precision
  of 1.0 in both retrieval and context layers. These are prepared-claim tests,
  supplemented by actual extraction-to-provider regressions.
- A controlled corpus of 5,000 owned memories with 768-dimensional vectors has
  identical top-100 IDs and scores before and after. Seven-probe medians:

| Measurement | 0.2.2 | 0.3.0 |
| --- | ---: | ---: |
| Semantic scoring | 386.7 ms | 235.0 ms |
| Provider prefetch | 423.7 ms | 248.5 ms |
| Peak Python scoring allocations | 135.85 MiB | 0.54 MiB |

That is 39.2% faster semantic scoring and 41.4% faster provider prefetch
in this synthetic benchmark. Model inference is stubbed. Allocation tracing runs
separately from timing and measures Python allocations, not total service RAM.
The reproducible probe is `evaluation/benchmarks/retrieval_hardening.py`; run it
with `PYTHONPATH` pointing at each checkout. The general scale harness now reports
eligible profile rows separately from total database size.

## Deployment-shaped checks

A consistent copy of BSL-AI's database migrated to schema 16 while preserving the
fingerprint of all 6,223 memory IDs, contents, owners, and statuses. All existing
records belong to Claire. Claire receives context; Margaret, Sasha, Yuki, and the
default profile receive none of it, including memories sourced from the vault.
Eight missing derived embeddings repaired successfully on the copy.

Using the gateway's Python 3.11.15 runtime and real configured extraction and
embedding endpoints, two synthetic turns were extracted and the correction was
recalled. The first live check exposed a rendering bug: low-confidence conflicting
claims were hidden behind the older claim. The fix now renders both as unresolved.
High-confidence explicit corrections supersede through the existing policy.
No synthetic test turn was inserted into the production database.

## Operational boundaries

Profile ownership is enforced by Remnant's provider and tool APIs. The SQLite
file remains available to the machine administrator; this is not OS-account or
container isolation. Legacy shared/fleet labels no longer grant cross-profile
access. Vault mappings are keyed by profile and path, and runtime identity v2
includes the profile. Existing v1 runtime records require explicit owner mapping;
configured-owner deployments retain their keys.

The model can still produce uncertain or incorrect extractions. A visible
unresolved claim is preferable to silently dropping its evidence; synthetic
retrieval scores do not imply perfect model accuracy. Embedding model/prefix
changes and approximate indexes remain separate experiments requiring quality
measurement and a deliberate re-embedding migration.

Back up SQLite before updating. Rollback across schema 16 requires restoring the
matching database backup and old source. Cache cleanup retains raw turns and
memories; SQLite reuses freed pages rather than immediately shrinking the file.

Machine-readable measurements: `evaluation/baselines/retrieval-hardening-2026-09-06.json`.
