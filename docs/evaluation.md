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

Reports include recall@1/3/5, MRR, nDCG@5, context precision, wrong-answer rate,
stale-claim exposure, duplicate occupancy, token estimates, per-stage latency,
category summaries, ranking profile, schema, embedding model, commit, and seed.
The stable report serializer removes timings and normalizes opaque database IDs.

Release procedure:

```bash
python -m remnant.evaluate --cases evaluation/cases/leadership.jsonl --layer retrieval
python -m remnant.evaluate --cases evaluation/cases/leadership.jsonl --layer context
python -m pytest -q
python -m ruff check remnant tests
python -m build
git diff --check
```

No behavior flag becomes a default solely because this synthetic suite passes.
Before deployment-wide enablement, run shadow extraction/reconciliation against
a representative private corpus and apply the thresholds in the leadership
implementation plan. Keep raw private cases and model prompts out of Git.
