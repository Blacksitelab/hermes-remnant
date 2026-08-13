from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from remnant.evaluation.metrics import ranking_metrics
from remnant.evaluation.runner import evaluate_scenarios, stable_report_json
from remnant.evaluation.schema import CATEGORIES, load_cases, validate_case


def _case(category: str = "dynamic_update") -> dict:
    return {
        "schema_version": 1,
        "case_id": f"{category}.001",
        "category": category,
        "persona": "test-user-a",
        "sessions": [
            {
                "observed_at": "2026-01-01T12:00:00Z",
                "turns": [
                    {
                        "label": "old-preference",
                        "user": "test-user-a prefers light editor theme",
                        "assistant": "Noted.",
                        "subject": "test-user-a",
                        "confidence": 0.9,
                    }
                ],
            },
            {
                "observed_at": "2026-02-01T12:00:00Z",
                "turns": [
                    {
                        "label": "new-preference",
                        "user": "test-user-a now prefers dark editor theme",
                        "assistant": "Understood.",
                        "subject": "test-user-a",
                        "confidence": 0.95,
                        "conflict_type": "update",
                        "valid_from": "2026-02-01T12:00:00Z",
                    }
                ],
            },
        ],
        "query": {
            "text": "test-user-a editor theme",
            "at": "2026-02-02T12:00:00Z",
        },
        "expected": {
            "answer_contains": ["dark"],
            "answer_must_not_contain": ["light"],
            "supporting_memory_labels": ["new-preference"],
            "stale_memory_labels": ["old-preference"],
        },
    }


def test_schema_rejects_unknown_category_and_bad_timestamp():
    unknown = _case()
    unknown["category"] = "made_up"
    with pytest.raises(ValueError, match="unknown category"):
        validate_case(unknown)
    malformed = _case()
    malformed["query"]["at"] = "next Thursday-ish"
    with pytest.raises(ValueError, match="ISO-8601"):
        validate_case(malformed)


def test_every_required_category_is_declared():
    assert len(CATEGORIES) == 12


def test_committed_leadership_corpus_has_twenty_cases_per_category():
    root = Path(__file__).resolve().parents[1]
    cases = load_cases(root / "evaluation" / "cases" / "leadership.jsonl")
    counts = Counter(case["category"] for case in cases)
    assert len(cases) == 240
    assert counts == {category: 20 for category in CATEGORIES}


def test_scenario_runner_is_stable_and_does_not_expose_stale_claim():
    first = evaluate_scenarios([_case()], layer="context")
    second = evaluate_scenarios([_case()], layer="context")
    assert stable_report_json(first) == stable_report_json(second)
    assert first["summary"]["recall_at_1"] == 1.0
    assert first["summary"]["stale_claim_exposure"] == 0
    assert "dark editor theme" in first["details"][0]["context"]
    assert "light editor theme" not in first["details"][0]["context"]


def test_deliberately_stale_retriever_fails_dynamic_case_metric():
    metrics = ranking_metrics(["old-preference"], {"new-preference"})
    assert metrics["recall_at_1"] == 0.0
    assert metrics["mrr"] == 0.0


def test_stable_report_is_valid_json():
    report = evaluate_scenarios([_case()], layer="retrieval")
    assert json.loads(stable_report_json(report))["summary"]["cases"] == 1
