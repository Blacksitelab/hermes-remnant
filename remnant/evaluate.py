"""Repeatable retrieval evaluation for Remnant.

Cases are JSON objects with ``query`` and ``expected_ids`` fields, plus
optional ``agent_id``, ``strategy``, and ``limit`` overrides.  The evaluator
measures only retrieval quality and latency; it never mutates memory state.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import RemnantConfig
from .db import RemnantDB, default_db_path, open_db
from .embed import Embedder
from .search import search


def evaluate_cases(
    db: RemnantDB,
    config: RemnantConfig,
    embedder: Embedder,
    cases: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate recall@k, MRR, and latency for explicit retrieval cases."""
    details: list[dict[str, Any]] = []
    reciprocal_ranks: list[float] = []
    recalls: list[float] = []
    latencies: list[float] = []
    for raw in cases:
        query = str(raw.get("query") or "").strip()
        expected = {str(mid) for mid in raw.get("expected_ids", []) if str(mid)}
        if not query or not expected:
            raise ValueError("every case requires non-empty query and expected_ids")
        limit = max(1, int(raw.get("limit", config.search_limit)))
        started = time.perf_counter()
        rows = search(
            db,
            config,
            query,
            agent_id=raw.get("agent_id") or config.agent_id,
            strategy=str(raw.get("strategy") or config.default_search_strategy),
            limit=limit,
            embedder=embedder,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        returned = [str(row["id"]) for row in rows]
        matched = expected.intersection(returned)
        rank = next((i + 1 for i, mid in enumerate(returned) if mid in expected), None)
        recalls.append(len(matched) / len(expected))
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        latencies.append(elapsed_ms)
        details.append(
            {
                "query": query,
                "expected_ids": sorted(expected),
                "returned_ids": returned,
                "matched_ids": sorted(matched),
                "first_relevant_rank": rank,
                "latency_ms": round(elapsed_ms, 3),
            }
        )
    count = len(details)
    sorted_latency = sorted(latencies)
    p95_index = min(count - 1, max(0, int(count * 0.95) - 1))
    return {
        "cases": count,
        "recall_at_k": round(sum(recalls) / count, 4) if count else 0.0,
        "mrr": round(sum(reciprocal_ranks) / count, 4) if count else 0.0,
        "latency_ms": {
            "mean": round(sum(latencies) / count, 3) if count else 0.0,
            "p95": round(sorted_latency[p95_index], 3) if count else 0.0,
        },
        "details": details,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Remnant from versioned case files.")
    parser.add_argument("--cases", type=Path, required=True, help="JSONL scenarios or legacy JSON.")
    parser.add_argument("--db", type=Path, default=None, help="Database path override.")
    parser.add_argument("--agent", default="default", help="Default agent identity for cases.")
    parser.add_argument(
        "--layer",
        choices=("retrieval", "context", "answer"),
        default="retrieval",
        help="Leadership evaluation layer for schema-v1 JSONL cases.",
    )
    args = parser.parse_args(argv)
    if args.cases.suffix.casefold() == ".jsonl":
        from .evaluation.runner import evaluate_scenarios, stable_report_json
        from .evaluation.schema import load_cases

        report = evaluate_scenarios(load_cases(args.cases), layer=args.layer)
        print(stable_report_json(report), end="")
        return 0
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        parser.error("--cases must contain a JSON array")
    db = open_db(args.db or default_db_path())
    config = RemnantConfig(agent_id=args.agent)
    embedder = Embedder(db, config)
    try:
        print(json.dumps(evaluate_cases(db, config, embedder, cases), indent=2, sort_keys=True))
        return 0
    finally:
        embedder.close()
        db.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
