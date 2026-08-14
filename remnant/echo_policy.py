"""Deterministic query archetypes and bounded Echo utility policy."""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections.abc import Iterable
from typing import Any

from .config import RemnantConfig
from .echo_types import EchoUtilityView

ARCHETYPES = (
    "preference",
    "historical",
    "project_state",
    "document_lookup",
    "troubleshooting",
    "person_or_entity",
    "procedure",
    "delegation",
    "general_recall",
    "unknown",
)
GLOBAL_ARCHETYPE = "global"
POLICY_VERSION = "echo-v1"

_HISTORY_RE = re.compile(
    r"\b(when|before|previously|used to|historical|history|at that time|last time|"
    r"in \d{4}|\d{4}-[01]\d-[0-3]\d)\b",
    re.I,
)
_DOCUMENT_RE = re.compile(r"\b(file|document|note|vault|markdown|folder|path)\b", re.I)
_TROUBLESHOOT_RE = re.compile(
    r"\b(error|failure|broken|bug|issue|traceback|exception|doesn't work|not working)\b",
    re.I,
)
_PREFERENCE_RE = re.compile(r"\b(prefer|preference|like|likes|favorite|favourite|usually)\b", re.I)
_PROJECT_RE = re.compile(r"\b(status|state|progress|milestone|todo|blocked|project)\b", re.I)
_PROCEDURE_RE = re.compile(r"\b(how do|how to|configure|setup|install|steps|procedure)\b", re.I)
_DELEGATION_RE = re.compile(r"\b(delegate|delegated|subagent|hand off|child task)\b", re.I)
_PERSON_RE = re.compile(r"\b(who is|person|user|team|colleague|friend|owner)\b", re.I)

_SIGNAL_WEIGHTS: dict[str, tuple[int, float, str]] = {
    "explicit_useful": (1, 1.0, "explicit"),
    "explicit_wrong": (-1, 1.0, "explicit"),
    "explicit_correction": (-1, 0.9, "explicit"),
    "tool_contradiction": (-1, 0.85, "inferred"),
    "tool_supported": (1, 0.6, "inferred"),
    "counterfactual_support": (1, 0.4, "inferred"),
    "counterfactual_harm": (-1, 0.5, "inferred"),
    "support_assessment": (1, 0.2, "inferred"),
}


def fingerprint(value: str) -> str:
    normalized = re.sub(r"\s+", " ", str(value or "").casefold()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def viewer_key_hash(value: str | None) -> str:
    return fingerprint(value or "anonymous")


def profile_scope_hash(scope: Iterable[str] | None) -> str:
    values = sorted(
        {
            str(item).strip().replace("\\", "/")
            for item in (scope or [])
            if str(item).strip()
        }
    )
    return fingerprint("\n".join(values))


def activation_key(
    *,
    viewer_hash: str,
    session_id: str,
    query_hash: str,
    context_hash: str,
    generation: int,
    policy_version: str,
) -> str:
    return fingerprint(
        "|".join(
            (
                viewer_hash,
                session_id,
                query_hash,
                context_hash,
                str(generation),
                policy_version,
            )
        )
    )


def classify_query(query: str) -> str:
    """Return one bounded archetype without an LLM call."""
    value = str(query or "").strip()
    if not value:
        return "unknown"
    if _HISTORY_RE.search(value):
        return "historical"
    if _DOCUMENT_RE.search(value):
        return "document_lookup"
    if _TROUBLESHOOT_RE.search(value):
        return "troubleshooting"
    if _PREFERENCE_RE.search(value):
        return "preference"
    if _PROJECT_RE.search(value):
        return "project_state"
    if _DELEGATION_RE.search(value):
        return "delegation"
    if _PROCEDURE_RE.search(value):
        return "procedure"
    if _PERSON_RE.search(value):
        return "person_or_entity"
    if re.search(r"\b(remember|recall|what did|tell me|according to)\b", value, re.I):
        return "general_recall"
    return "unknown"


def signal_spec(signal_type: str) -> tuple[int, float, str] | None:
    return _SIGNAL_WEIGHTS.get(str(signal_type or "").strip())


def canonical_pair(first: str, second: str) -> tuple[str, str] | None:
    a, b = str(first or ""), str(second or "")
    if not a or not b or a == b:
        return None
    return (a, b) if a < b else (b, a)


def _decay(value: float, age_s: float, half_life_days: float) -> float:
    if value <= 0:
        return 0.0
    return value * 0.5 ** (max(0.0, age_s) / (half_life_days * 86400.0))


def combine_utility_rows(
    rows: list[dict[str, Any]],
    *,
    memory_id: str,
    archetype: str,
    config: RemnantConfig,
    now: float | None = None,
) -> EchoUtilityView | None:
    """Combine current-archetype and 25%-weighted global aggregates."""
    if not rows:
        return None
    now = time.time() if now is None else float(now)
    explicit_positive = explicit_negative = 0
    evaluator_samples = 0
    explicit_pos_mass = explicit_neg_mass = 0.0
    inferred_pos_mass = inferred_neg_mass = 0.0
    for row in rows:
        row_archetype = str(row.get("query_archetype") or "")
        factor = (
            0.25
            if row_archetype == GLOBAL_ARCHETYPE and archetype != GLOBAL_ARCHETYPE
            else 1.0
        )
        age_s = now - float(row.get("last_signal_at") or now)
        explicit_pos_mass += factor * _decay(
            float(row.get("explicit_positive_mass") or 0.0),
            age_s,
            config.echo_explicit_feedback_half_life_days,
        )
        explicit_neg_mass += factor * _decay(
            float(row.get("explicit_negative_mass") or 0.0),
            age_s,
            config.echo_explicit_feedback_half_life_days,
        )
        inferred_pos_mass += factor * _decay(
            float(row.get("inferred_positive_mass") or 0.0),
            age_s,
            config.echo_utility_half_life_days,
        )
        inferred_neg_mass += factor * _decay(
            float(row.get("inferred_negative_mass") or 0.0),
            age_s,
            config.echo_utility_half_life_days,
        )
        explicit_positive += round(factor * int(row.get("explicit_positive") or 0))
        explicit_negative += round(factor * int(row.get("explicit_negative") or 0))
        evaluator_samples += round(factor * int(row.get("evaluator_samples") or 0))
    positive_mass = explicit_pos_mass + inferred_pos_mass
    negative_mass = explicit_neg_mass + inferred_neg_mass
    prior = 4.0
    utility_mean = (2.0 + positive_mass) / (prior + positive_mass + negative_mass)
    observations = positive_mass + negative_mass
    confidence = 1.0 - math.exp(-observations / 10.0)
    harm_risk = negative_mass / max(1e-9, observations)
    centered = 2.0 * (utility_mean - 0.5)
    adjustment = 0.0
    if observations >= config.echo_min_observations:
        adjustment = max(-config.echo_max_rank_adjustment, min(
            config.echo_max_rank_adjustment,
            config.echo_max_rank_adjustment * confidence * centered,
        ))
    return EchoUtilityView(
        memory_id=memory_id,
        query_archetype=archetype,
        utility_mean=utility_mean,
        harm_risk=harm_risk,
        confidence=confidence,
        observations=observations,
        adjustment=adjustment,
        explicit_positive=explicit_positive,
        explicit_negative=explicit_negative,
        evaluator_samples=evaluator_samples,
        policy_version=str(rows[0].get("policy_version") or config.echo_policy_version),
    )


__all__ = [
    "ARCHETYPES",
    "GLOBAL_ARCHETYPE",
    "POLICY_VERSION",
    "activation_key",
    "canonical_pair",
    "classify_query",
    "combine_utility_rows",
    "fingerprint",
    "profile_scope_hash",
    "signal_spec",
    "viewer_key_hash",
]
