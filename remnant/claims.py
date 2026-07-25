"""Conservative structured-claim projections for fact memories.

Claims make common facts easier to inspect and version without pretending that
an extractor is authoritative.  Every claim points at an immutable backing
memory; uncertain sentences are stored as a ``states`` claim rather than
discarded or aggressively parsed.
"""

from __future__ import annotations

import re

from .db import RemnantDB


def _claim_parts(subject: str, fact: str) -> tuple[str, str]:
    """Extract only a small stable predicate vocabulary; otherwise retain text."""
    cleaned = fact.strip().rstrip(".?!")
    subject_re = re.escape(subject.strip())
    remainder = re.sub(rf"^{subject_re}\s+", "", cleaned, count=1, flags=re.I)
    if remainder != cleaned:
        match = re.match(
            r"^(prefers?|likes?|uses?|works on|owns?|lives in|is named)\s+(.+)$",
            remainder,
            re.I,
        )
        if match:
            return re.sub(r"\s+", "_", match.group(1).lower()), match.group(2).strip()
    return "states", cleaned


def record_claim_from_memory(
    db: RemnantDB,
    *,
    memory_id: str,
    subject: str,
    fact: str,
    confidence: float = 0.5,
    contradicted: bool = False,
) -> str | None:
    """Project a fact memory into a claim and version competing assertions."""
    subject = subject.strip()
    if not subject or subject.lower() == "general":
        return None
    predicate, object_value = _claim_parts(subject, fact)
    if not object_value:
        return None
    active = db.get_active_claim(subject, predicate)
    if active and str(active["object"]).casefold().strip() != object_value.casefold().strip() and not contradicted:
        db.supersede_claims(subject=subject, predicate=predicate)
    return db.create_claim(
        memory_id=memory_id,
        subject=subject,
        predicate=predicate,
        object=object_value,
        confidence=confidence,
        status="contradicted" if contradicted else "active",
    )
