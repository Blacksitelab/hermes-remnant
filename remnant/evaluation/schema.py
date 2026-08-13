"""Versioned, dependency-free validation for Remnant evaluation cases."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

CATEGORIES = frozenset(
    {
        "dynamic_update",
        "static_false_contradiction",
        "conditional_fact",
        "stable_long_time",
        "historical_query",
        "unresolved_conflict",
        "duplicate_paraphrase",
        "distractor_entity",
        "immediate_next_turn",
        "vault_vs_conversation",
        "visibility_scope",
        "runtime_isolation",
    }
)


def _timestamp(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty ISO-8601 timestamp")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{path} is not a valid ISO-8601 timestamp") from exc
    return value


def validate_case(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize one schema-v1 leadership scenario."""
    if not isinstance(raw, dict):
        raise ValueError("case must be an object")
    if raw.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    case_id = str(raw.get("case_id") or "").strip()
    if not case_id:
        raise ValueError("case_id is required")
    category = str(raw.get("category") or "")
    if category not in CATEGORIES:
        raise ValueError(f"unknown category: {category}")
    persona = str(raw.get("persona") or "").strip()
    if not persona:
        raise ValueError("persona is required")
    sessions = raw.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("sessions must be a non-empty list")
    labels: set[str] = set()
    normalized_sessions: list[dict[str, Any]] = []
    for session_index, session in enumerate(sessions):
        if not isinstance(session, dict):
            raise ValueError(f"sessions[{session_index}] must be an object")
        observed_at = _timestamp(
            session.get("observed_at"), f"sessions[{session_index}].observed_at"
        )
        turns = session.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"sessions[{session_index}].turns must be non-empty")
        normalized_turns: list[dict[str, Any]] = []
        for turn_index, turn in enumerate(turns):
            if not isinstance(turn, dict) or not str(turn.get("user") or "").strip():
                raise ValueError(
                    f"sessions[{session_index}].turns[{turn_index}].user is required"
                )
            row = dict(turn)
            row["user"] = str(turn["user"]).strip()
            row["assistant"] = str(turn.get("assistant") or "").strip()
            row["label"] = str(
                turn.get("label") or f"session-{session_index}-turn-{turn_index}"
            )
            if row["label"] in labels:
                raise ValueError(f"duplicate memory label: {row['label']}")
            labels.add(row["label"])
            normalized_turns.append(row)
        normalized_sessions.append(
            {**session, "observed_at": observed_at, "turns": normalized_turns}
        )
    query = raw.get("query")
    if not isinstance(query, dict) or not str(query.get("text") or "").strip():
        raise ValueError("query.text is required")
    query_at = _timestamp(query.get("at"), "query.at")
    expected = raw.get("expected")
    if not isinstance(expected, dict):
        raise ValueError("expected must be an object")
    supporting = expected.get("supporting_memory_labels") or []
    if not isinstance(supporting, list) or not supporting:
        raise ValueError("expected.supporting_memory_labels must be non-empty")
    unknown = set(map(str, supporting)) - labels
    if unknown:
        raise ValueError(f"unknown supporting labels: {sorted(unknown)}")
    return {
        **raw,
        "case_id": case_id,
        "category": category,
        "persona": persona,
        "sessions": normalized_sessions,
        "query": {**query, "text": str(query["text"]).strip(), "at": query_at},
        "expected": {
            **expected,
            "answer_contains": list(map(str, expected.get("answer_contains") or [])),
            "answer_must_not_contain": list(
                map(str, expected.get("answer_must_not_contain") or [])
            ),
            "supporting_memory_labels": list(map(str, supporting)),
        },
    }


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL or a JSON array and validate every case."""
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.casefold() == ".jsonl":
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        values = json.loads(text)
    if not isinstance(values, list):
        raise ValueError("case file must contain JSONL objects or a JSON array")
    cases = [validate_case(value) for value in values]
    ids = [case["case_id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case_id values must be unique")
    return cases


__all__ = ["CATEGORIES", "load_cases", "validate_case"]
