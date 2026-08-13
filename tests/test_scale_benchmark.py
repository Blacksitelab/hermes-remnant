from __future__ import annotations

from pathlib import Path

from remnant.context import compile_context
from remnant.evaluation.scale import benchmark_scale


def test_context_budget_prefers_complete_evidence_across_classes():
    memories = [
        {"id": "current", "content": "Current preference", "visibility": "private"},
        {
            "id": "conflict",
            "content": "Conflicting preference",
            "visibility": "private",
            "claim_status": "unresolved",
        },
        {
            "id": "document",
            "content": "Supporting document passage",
            "visibility": "shared",
            "source": "vault",
            "type": "document",
        },
        {
            "id": "recent",
            "content": "Recent unprocessed preference",
            "visibility": "private",
            "pending": True,
        },
    ]
    counter = lambda value: len(value.split())  # noqa: E731
    rendered = compile_context(memories, token_budget=100, token_counter=counter)
    assert counter(rendered) <= 100
    assert "Current preference" in rendered
    assert "Conflicting preference" in rendered
    assert "Supporting document passage" in rendered
    assert "Recent unprocessed preference" in rendered


def test_scale_benchmark_is_reproducible_and_isolated(tmp_path: Path):
    report = benchmark_scale(sizes=(20,), work_dir=tmp_path, probes=2, seed=7)
    assert report["benchmark"] == "remnant-scale-envelope-v1"
    assert report["configuration"]["seed"] == 7
    store = report["stores"][0]
    assert store["size"] == 20
    assert store["embedding_rows"] == 18
    assert store["recall_at_5"] == 1.0
    assert store["exact_vector_ms"]["p95"] >= store["exact_vector_ms"]["p50"]
    assert (tmp_path / "scale-20.db").is_file()
