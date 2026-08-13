"""Conservative structured-claim projections for fact memories.

Claims make common facts easier to inspect and version without pretending that
an extractor is authoritative.  Every claim points at an immutable backing
memory; uncertain sentences are stored as a ``states`` claim rather than
discarded or aggressively parsed.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from .db import RemnantDB
from .reconcile import decide_reconciliation


def _claim_parts(subject: str, fact: str) -> tuple[str, str]:
    """Extract only a small stable predicate vocabulary; otherwise retain text."""
    cleaned = fact.strip().rstrip(".?!")
    subject_re = re.escape(subject.strip())
    remainder = re.sub(rf"^{subject_re}\s+", "", cleaned, count=1, flags=re.I)
    # Temporal/transition adverbs belong to claim metadata, not the predicate
    # key; keeping them here would split "prefers" and "now prefers" into two
    # unrelated claim histories.
    remainder = re.sub(r"^(?:now|currently|still)\s+", "", remainder, flags=re.I)
    if remainder != cleaned:
        match = re.match(
            r"^(prefers?|likes?|uses?|works on|owns?|lives in|is named)\s+(.+)$",
            remainder,
            re.I,
        )
        if match:
            return re.sub(r"\s+", "_", match.group(1).lower()), match.group(2).strip()
    return "states", cleaned


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalise_value(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold().strip(" .!?"))


def classify_claim_conflict(
    *,
    fact: str,
    existing: dict[str, Any] | None,
    claim_data: dict[str, Any] | None = None,
) -> str:
    """Classify a candidate without allowing recency alone to rewrite truth.

    The extractor may provide a stronger label.  Otherwise this deliberately
    conservative local classifier only treats explicit transition language as
    an update; all other competing values remain auditable rather than being
    silently superseded.
    """
    supplied = str((claim_data or {}).get("conflict_type") or "").lower().strip()
    if supplied in {
        "update", "contradiction", "conditional", "compatible", "duplicate", "unresolved"
    }:
        return supplied
    if existing is None:
        return "compatible"
    old = _normalise_value(existing.get("object"))
    _, new_object = _claim_parts(str(existing.get("subject") or ""), fact)
    new = _normalise_value(new_object)
    if old == new:
        return "duplicate"
    if re.search(
        r"\b(switched|changed|updated|now prefer|no longer|instead of|from .+ to)\b",
        fact,
        re.I,
    ):
        return "update"
    if (claim_data or {}).get("conditions") or (claim_data or {}).get("scope_type"):
        return "conditional"
    return "unresolved"


def record_claim_from_memory(
    db: RemnantDB,
    *,
    memory_id: str,
    subject: str,
    fact: str,
    confidence: float = 0.5,
    contradicted: bool = False,
    claim_data: dict[str, Any] | None = None,
    reconciliation_enabled: bool = False,
    source_turn_id: int | None = None,
    agent_id: str | None = None,
) -> str | None:
    """Project a fact into an immutable, optionally reconciled claim.

    Legacy mode retains the historical newest-value supersession behavior for
    existing installations.  Release-track mode records conflict semantics and
    only supersedes a predecessor for an explicit high-confidence update.
    """
    subject = subject.strip()
    if not subject or subject.lower() == "general":
        return None
    predicate, object_value = _claim_parts(subject, fact)
    if data_predicate := str((claim_data or {}).get("predicate") or "").strip():
        predicate = re.sub(r"\s+", "_", data_predicate.casefold())
    if data_object := str((claim_data or {}).get("object") or "").strip():
        object_value = data_object
    if not object_value:
        return None
    data = dict(claim_data or {})
    observed_at = str(data.get("observed_at") or _now_iso())
    active = db.get_active_claim(subject, predicate, agent_id=agent_id)
    conflict_type = classify_claim_conflict(fact=fact, existing=active, claim_data=data)
    if not reconciliation_enabled:
        if (
            active
            and _normalise_value(active.get("object")) != _normalise_value(object_value)
            and not contradicted
        ):
            db.supersede_claims(subject=subject, predicate=predicate, agent_id=agent_id)
        conflict_type = "update" if active else "compatible"
    active_evidence = db.get_memory(str(active.get("memory_id"))) if active else None
    candidate_evidence = {
        "source": data.get("source"),
        "verified": bool(data.get("verified")),
        "trust_score": data.get("trust_score"),
        "seen_count": data.get("seen_count", 1),
        "explicit_correction": bool(
            re.search(
                r"\b(switched|changed|updated|now prefer|no longer|instead of|from .+ to)\b",
                fact,
                re.I,
            )
        ),
    }
    decision = decide_reconciliation(
        conflict_type=conflict_type,
        confidence=float(data.get("confidence", confidence) or confidence),
        active=active,
        active_evidence=active_evidence,
        candidate_evidence=candidate_evidence,
    )
    if reconciliation_enabled and decision.supersede:
        db.supersede_claims(subject=subject, predicate=predicate, agent_id=agent_id)
    elif reconciliation_enabled and decision.decision == "duplicate":
        return str(active.get("id")) if active else None
    if reconciliation_enabled:
        conflict_type = decision.decision

    qualifiers = data.get("qualifiers")
    if not isinstance(qualifiers, dict):
        qualifiers = {}
    conditions = data.get("conditions")
    if conditions:
        qualifiers["conditions"] = conditions
    status = "contradicted" if contradicted or conflict_type == "contradiction" else "active"
    resolution_status = "contradicted" if status == "contradicted" else conflict_type
    claim_id = db.create_claim(
        memory_id=memory_id,
        subject=subject,
        predicate=predicate,
        object=object_value,
        confidence=float(data.get("confidence", confidence) or confidence),
        qualifiers=qualifiers or None,
        status=status,
        observed_at=observed_at,
        event_at=data.get("event_at"),
        valid_from=data.get("valid_from") or (observed_at if conflict_type == "update" else None),
        valid_to=data.get("valid_to"),
        scope_type=data.get("scope_type"),
        scope_value=data.get("scope_value"),
        modality=str(data.get("modality") or "asserted"),
        conflict_type=conflict_type,
        resolution_status=resolution_status,
        extractor_version=data.get("extractor_version"),
        source_turn_id=source_turn_id,
    )
    if reconciliation_enabled:
        db.write_audit(
            actor="system",
            action="claim_reconcile",
            memory_id=memory_id,
            details={
                "claim_id": claim_id,
                "candidate_claim_id": active.get("id") if active else None,
                "decision": decision.decision,
                "rule": decision.rule,
                "rule_version": "reconcile-v1",
                "confidence": decision.confidence,
                "source_turn_id": source_turn_id,
                "superseded": decision.supersede,
            },
        )
    return claim_id


__all__ = ["record_claim_from_memory", "classify_claim_conflict"]
