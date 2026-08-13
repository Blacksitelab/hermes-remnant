"""Stable metrics for retrieval, context, and optional answer evaluation."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def ranking_metrics(returned: list[str], expected: set[str]) -> dict[str, float]:
    """Compute standard ranking metrics for one scenario."""
    ranks = [index + 1 for index, label in enumerate(returned) if label in expected]
    result: dict[str, float] = {}
    for k in (1, 3, 5):
        result[f"recall_at_{k}"] = len(expected.intersection(returned[:k])) / len(expected)
    result["mrr"] = 1.0 / min(ranks) if ranks else 0.0
    dcg = sum(
        1.0 / math.log2(index + 2)
        for index, label in enumerate(returned[:5])
        if label in expected
    )
    ideal = sum(1.0 / math.log2(index + 2) for index in range(min(5, len(expected))))
    result["ndcg_at_5"] = dcg / ideal if ideal else 0.0
    return result


def summarize(details: list[dict[str, Any]]) -> dict[str, Any]:
    """Produce byte-stable aggregate values; timings stay only in details."""
    if not details:
        return {"cases": 0, "macro_score": 0.0, "categories": {}}
    metric_names = ("recall_at_1", "recall_at_3", "recall_at_5", "mrr", "ndcg_at_5")
    overall = {
        name: round(sum(float(row[name]) for row in details) / len(details), 4)
        for name in metric_names
    }
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        by_category[str(row["category"])].append(row)
    categories = {
        category: {
            name: round(sum(float(row[name]) for row in rows) / len(rows), 4)
            for name in metric_names
        }
        for category, rows in sorted(by_category.items())
    }
    context_precision = (
        sum(float(row.get("context_precision", 0.0)) for row in details) / len(details)
    )
    stale_exposure = sum(int(row.get("stale_claim_exposure", 0)) for row in details)
    duplicate_occupancy = sum(int(row.get("duplicate_top_k_occupancy", 0)) for row in details)
    wrong = sum(1 for row in details if row.get("answer_grade") == "wrong")
    answer_values = {"correct": 1.0, "partial": 0.5, "blank": 0.0, "wrong": -1.0}
    graded = [row for row in details if row.get("answer_grade") in answer_values]
    macro = (
        sum(answer_values[str(row["answer_grade"])] for row in graded) / len(graded)
        if graded
        else overall["recall_at_5"]
    )
    return {
        "cases": len(details),
        **overall,
        "context_precision": round(context_precision, 4),
        "macro_score": round(macro, 4),
        "wrong_answer_rate": round(wrong / len(graded), 4) if graded else None,
        "stale_claim_exposure": stale_exposure,
        "duplicate_top_k_occupancy": duplicate_occupancy,
        "categories": categories,
    }


__all__ = ["ranking_metrics", "summarize"]
