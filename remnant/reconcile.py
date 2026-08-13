"""Deterministic, auditable claim reconciliation policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReconciliationDecision:
    decision: str
    confidence: float
    rule: str
    supersede: bool = False


def decide_reconciliation(
    *,
    conflict_type: str,
    confidence: float,
    active: dict[str, Any] | None,
    active_evidence: dict[str, Any] | None = None,
    candidate_evidence: dict[str, Any] | None = None,
) -> ReconciliationDecision:
    """Validate a proposed conflict label before any lifecycle transition.

    A model label alone is not enough to replace a verified durable fact. A
    direct transition/correction, corroborated observation, or trusted source
    may supersede it; otherwise the evidence remains explicitly unresolved.
    """
    if active is None:
        return ReconciliationDecision("compatible", confidence, "no-competitor")
    if conflict_type == "duplicate":
        return ReconciliationDecision("duplicate", confidence, "exact-or-model-duplicate")
    if conflict_type == "conditional":
        return ReconciliationDecision("conditional", confidence, "disjoint-or-explicit-scope")
    candidate = candidate_evidence or {}
    active_evidence = active_evidence or {}
    explicit_correction = bool(candidate.get("explicit_correction"))
    corroborated = bool(
        candidate.get("verified")
        or float(candidate.get("trust_score") or 0.0) >= 0.85
        or int(candidate.get("seen_count") or 1) >= 2
    )
    if (
        conflict_type == "update"
        and confidence >= 0.75
        and (not active_evidence.get("verified") or explicit_correction or corroborated)
    ):
        return ReconciliationDecision("update", confidence, "explicit-high-confidence-update", True)
    if conflict_type == "update" and active_evidence.get("verified"):
        return ReconciliationDecision(
            "unresolved",
            confidence,
            "verified-claim-needs-correction-or-corroboration",
        )
    if conflict_type == "contradiction":
        return ReconciliationDecision("contradiction", confidence, "explicit-conflict")
    return ReconciliationDecision("unresolved", confidence, "conservative-fallback")


__all__ = ["ReconciliationDecision", "decide_reconciliation"]
