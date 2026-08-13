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
) -> ReconciliationDecision:
    """Validate a proposed conflict label before any lifecycle transition."""
    if active is None:
        return ReconciliationDecision("compatible", confidence, "no-competitor")
    if conflict_type == "duplicate":
        return ReconciliationDecision("duplicate", confidence, "exact-or-model-duplicate")
    if conflict_type == "conditional":
        return ReconciliationDecision("conditional", confidence, "disjoint-or-explicit-scope")
    if conflict_type == "update" and confidence >= 0.75:
        return ReconciliationDecision("update", confidence, "explicit-high-confidence-update", True)
    if conflict_type == "contradiction":
        return ReconciliationDecision("contradiction", confidence, "explicit-conflict")
    return ReconciliationDecision("unresolved", confidence, "conservative-fallback")


__all__ = ["ReconciliationDecision", "decide_reconciliation"]
