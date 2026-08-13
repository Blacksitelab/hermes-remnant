"""Isolated deterministic runner for leadership evaluation scenarios."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from ..claims import record_claim_from_memory
from ..config import RemnantConfig
from ..context import conservative_token_count
from ..db import SCHEMA_VERSION, open_db
from ..recall import RecallRequest, RecallService
from .metrics import ranking_metrics, summarize
from .schema import validate_case


class DeterministicEmbedder:
    """Small stable embedding used only by offline fixtures."""

    _model = "evaluation-hash-v1"

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * 64
        for token in str(text or "").casefold().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:2], "big") % len(vector)] += 1.0
        norm = sum(value * value for value in vector) ** 0.5
        return [value / norm for value in vector] if norm else vector


def _seed_case(db: Any, case: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    label_to_id: dict[str, str] = {}
    id_to_label: dict[str, str] = {}
    for session_index, session in enumerate(case["sessions"]):
        for turn_index, turn in enumerate(session["turns"]):
            label = str(turn["label"])
            agent = str(turn.get("agent") or case["persona"])
            if turn.get("pending"):
                turn_id = db.insert_turn_with_extraction(
                    session_id=str(
                        turn.get("session_id")
                        or case["query"].get("session_id")
                        or case["case_id"]
                    ),
                    agent_id=agent,
                    user_text=str(turn["user"]),
                    assistant_text=str(turn.get("assistant") or ""),
                )
                pending_id = f"pending-{turn_id}"
                label_to_id[label] = pending_id
                id_to_label[pending_id] = label
                continue
            memory_id = db.insert_memory(
                content=str(turn["user"]),
                source=str(turn.get("source") or "conversation"),
                source_id=str(turn.get("source_id") or f"{session_index}:{turn_index}"),
                agent=agent,
                visibility=str(turn.get("visibility") or "private"),
                type=str(turn.get("type") or "fact"),
                confidence=float(turn.get("confidence", 0.8)),
                trust_score=float(turn.get("trust_score", 0.7)),
                metadata=dict(turn.get("metadata") or {}),
            )
            label_to_id[label] = memory_id
            id_to_label[memory_id] = label
            subject = str(turn.get("subject") or case["persona"])
            if subject and subject != "general":
                claim_data = {
                    key: turn[key]
                    for key in (
                        "confidence", "conflict_type", "event_at", "valid_from", "valid_to",
                        "scope_type", "scope_value", "conditions", "modality",
                    )
                    if key in turn
                }
                claim_data["observed_at"] = session["observed_at"]
                claim_data["extractor_version"] = "evaluation-v1"
                record_claim_from_memory(
                    db,
                    memory_id=memory_id,
                    subject=subject,
                    fact=str(turn["user"]),
                    confidence=float(turn.get("confidence", 0.8)),
                    claim_data=claim_data,
                    reconciliation_enabled=True,
                )
    return label_to_id, id_to_label


def _answer_grade(answer: str, expected: dict[str, Any]) -> str:
    normalized = answer.casefold()
    required = [value.casefold() for value in expected["answer_contains"]]
    forbidden = [value.casefold() for value in expected["answer_must_not_contain"]]
    if not normalized.strip():
        return "blank"
    if any(value in normalized for value in forbidden):
        return "wrong"
    hits = sum(value in normalized for value in required)
    if hits == len(required):
        return "correct"
    return "partial" if hits else "wrong"


def _evaluate_case(
    case: dict[str, Any],
    *,
    layer: str,
    answerer: Callable[[str, str], str] | None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="remnant-eval-") as temp_dir:
        db = open_db(Path(temp_dir) / "evaluation.db")
        try:
            label_to_id, id_to_label = _seed_case(db, case)
            query = str(case["query"]["text"])
            config = RemnantConfig(
                agent_id=str(case["query"].get("agent") or case["persona"]),
                default_search_strategy=str(case["query"].get("strategy") or "keyword"),
                prefetch_embedding_timeout_ms=0,
                search_limit=5,
            )
            query_at = datetime.fromisoformat(str(case["query"]["at"]).replace("Z", "+00:00"))
            response = RecallService(db, config).recall(
                RecallRequest(
                    query=query,
                    agent_id=config.agent_id,
                    session_id=str(case["query"].get("session_id") or case["case_id"]),
                    strategy=config.default_search_strategy,
                    limit=5,
                    now=query_at,
                    include_pending=True,
                    output_mode="context",
                ),
                embedder=DeterministicEmbedder(),
            )
            returned_labels = [id_to_label.get(str(row.get("id")), "") for row in response.results]
            returned_labels = [label for label in returned_labels if label]
            expected_labels = set(case["expected"]["supporting_memory_labels"])
            metrics = ranking_metrics(returned_labels, expected_labels)
            context = response.context
            relevant_lines = sum(
                1 for label in expected_labels if label_to_id[label][:12] in context
            )
            rendered_items = max(1, context.count("\n- ["))
            detail: dict[str, Any] = {
                "case_id": case["case_id"],
                "category": case["category"],
                **metrics,
                "expected_labels": sorted(expected_labels),
                "returned_labels": returned_labels,
                "context_precision": relevant_lines / rendered_items,
                "stale_claim_exposure": sum(
                    1
                    for label in returned_labels
                    if label in set(case["expected"].get("stale_memory_labels") or [])
                ),
                "duplicate_top_k_occupancy": max(
                    0, len(returned_labels) - len(set(returned_labels))
                ),
                "injected_tokens": conservative_token_count(context),
                "timing_ms": {
                    "recall": response.diagnostics.get("elapsed_ms", 0.0),
                },
            }
            if layer in {"context", "answer"}:
                detail["context"] = context
            if layer == "answer":
                answer = answerer(query, context) if answerer else context
                detail["answer"] = answer
                detail["answer_grade"] = _answer_grade(answer, case["expected"])
            return detail
        finally:
            db.close()


def evaluate_scenarios(
    cases: list[dict[str, Any]],
    *,
    layer: str = "retrieval",
    answerer: Callable[[str, str], str] | None = None,
) -> dict[str, Any]:
    """Run cases against isolated throwaway databases; production is untouched."""
    if layer not in {"retrieval", "context", "answer"}:
        raise ValueError("layer must be retrieval, context, or answer")
    validated = [validate_case(case) for case in cases]
    details = [_evaluate_case(case, layer=layer, answerer=answerer) for case in validated]
    return {
        "schema_version": 1,
        "layer": layer,
        "configuration": {
            "ranking_profile": "claims-v1",
            "embedding_model": DeterministicEmbedder._model,
            "random_seed": 0,
            "remnant_schema": SCHEMA_VERSION,
            "commit_sha": os.environ.get("GITHUB_SHA", "local"),
        },
        "summary": summarize(details),
        "details": details,
    }


def stable_report_json(report: dict[str, Any]) -> str:
    """Return a byte-stable report with explicitly variable timings removed."""
    stable = json.loads(json.dumps(report))
    for detail in stable.get("details", []):
        detail.pop("timing_ms", None)
        # Context carries opaque database UUID references. Preserve the
        # semantic text in live reports but normalize those run-local IDs for
        # committed reproducibility comparisons.
        if "context" in detail:
            detail["context"] = re.sub(r"m:[0-9a-f-]{1,12}", "m:<opaque>", detail["context"])
    return json.dumps(stable, indent=2, sort_keys=True) + "\n"


__all__ = ["DeterministicEmbedder", "evaluate_scenarios", "stable_report_json"]
