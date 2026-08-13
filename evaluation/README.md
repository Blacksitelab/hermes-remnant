# Remnant leadership evaluation corpus

This directory contains synthetic, non-secret, versioned scenarios used to
gate Remnant behavior. `cases/leadership.jsonl` has twenty cases for each of
the twelve required categories. `baselines/` contains timing-free reports so
commits can be compared byte-for-byte in CI.

Run the deterministic layers with:

```bash
python -m remnant.evaluate --cases evaluation/cases/leadership.jsonl --layer retrieval
python -m remnant.evaluate --cases evaluation/cases/leadership.jsonl --layer context
```

The runner creates one temporary SQLite database per scenario and never opens
the configured production database. The optional `answer` layer is deliberately
not run in CI because a real answer model consumes tokens; callers may supply a
bounded answer adapter through the Python API.

Synthetic scenarios are regression evidence for Remnant itself, not a claim
that Remnant outperforms external providers. Any competitive claim requires a
separately reviewed, provider-neutral benchmark and deployment-shaped data.
