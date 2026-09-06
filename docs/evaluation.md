# Evaluation and release gates

Remnant's deterministic quality laboratory has three layers:

1. `retrieval` checks which evidence labels survive search and claim resolution;
2. `context` checks the exact untrusted-data block injected into Hermes;
3. `answer` checks an optional answer function against required and forbidden
   phrases.

The committed leadership corpus covers dynamic updates, false contradictions,
conditions, long-lived facts, historical queries, unresolved conflicts,
paraphrases, distractors, next-turn recall, vault/conversation competition,
visibility, and runtime identity isolation. Each category contains twenty
scenarios and includes explicit observation/query timestamps.

The held-out adversarial corpus at
`evaluation/cases/heldout-adversarial.jsonl` contains 120 natural-language
scenarios (ten per category). It intentionally avoids the synthetic
`signal-*` identifiers used by the leadership regression corpus and exercises
question wording, inflection, possessives, conditions, pending turns, and
profile ownership boundaries. It is a release gate, not a claim of
cross-provider superiority.

Reports include recall@1/3/5, MRR, nDCG@5, context precision, wrong-answer rate,
stale-claim exposure, duplicate occupancy, token estimates, per-stage latency,
category summaries, ranking profile, schema, embedding model, commit, and seed.
The stable report serializer removes timings and normalizes opaque database IDs.

Release procedure:

```bash
python -m remnant.evaluate --cases evaluation/cases/leadership.jsonl --layer retrieval
python -m remnant.evaluate --cases evaluation/cases/leadership.jsonl --layer context
python -m remnant.evaluate --cases evaluation/cases/heldout-adversarial.jsonl --layer retrieval
python -m remnant.evaluate --cases evaluation/cases/heldout-adversarial.jsonl --layer context
python -m remnant.evaluation.scale --sizes 5000 --probes 5 --output scale-report.json
python -m pytest -q
python -m ruff check remnant tests
python -m build
git diff --check
```

No behavior flag becomes a default solely because this synthetic suite passes.
Before deployment-wide enablement, run shadow extraction/reconciliation against
a representative private corpus and apply the thresholds in the leadership
implementation plan. Keep raw private cases and model prompts out of Git.

The scenario runner seeds prepared memories and claim projections; its scores
measure retrieval/resolution, not extraction-model quality. The regression suites
`test_retrieval_hardening.py` and `test_profile_isolation.py` additionally exercise
extraction JSON through storage and provider context, high-similarity corrections,
failed embedding repair, database contention, and profile-boundary attacks.

Scale report v2 defaults to 768 dimensions. `full_prefetch_ms` measures the actual
provider hook; `recall_service_ms` reports the separate recall service. Timings run
without tracemalloc. `peak_python_allocations_mb` comes from a separate untimed
semantic scan and excludes SQLite/native allocations. Reopening a connection is
labelled explicitly because it does not clear the operating-system page cache.
