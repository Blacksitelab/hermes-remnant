"""Typed data contracts for Remnant Echo."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .context import RenderedMemory


@dataclass(frozen=True)
class EchoReceiptDraft:
    """A receipt candidate that is not durable until Hermes consumes context."""

    activation_key: str
    session_id: str
    agent_id: str
    viewer_key_hash: str
    profile_scope_hash: str
    query_fingerprint: str
    query_archetype: str
    context_hash: str
    memory_generation: int
    token_count: int
    policy_version: str
    items: tuple[RenderedMemory, ...] = ()


@dataclass(frozen=True)
class EchoUtilityView:
    """Decayed utility evidence for one memory/archetype combination."""

    memory_id: str
    query_archetype: str
    utility_mean: float
    harm_risk: float
    confidence: float
    observations: float
    adjustment: float
    explicit_positive: int = 0
    explicit_negative: int = 0
    evaluator_samples: int = 0
    policy_version: str = ""


@dataclass(frozen=True)
class EchoSignalInput:
    """One bounded outcome signal before it is persisted."""

    memory_id: str
    agent_id: str
    viewer_key_hash: str
    query_archetype: str
    signal_type: str
    direction: int
    weight: float
    source: str
    receipt_id: str | None = None
    paired_memory_id: str | None = None
    evaluator_version: str | None = None


@dataclass(frozen=True)
class EchoDiagnostics:
    """Bounded diagnostics returned to recall callers."""

    archetype: str
    policy_version: str
    influence: float
    changed_count: int = 0
    utility_hits: int = 0
    budget_bypassed: bool = False
    details: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "EchoDiagnostics",
    "EchoReceiptDraft",
    "EchoSignalInput",
    "EchoUtilityView",
]
