"""Observable claim-aware ranking applied after candidate resolution."""

from __future__ import annotations

import re
from typing import Any

from .db import RemnantDB

RANKING_PROFILE = "claims-v1"


def _quality_rows(db: RemnantDB, ids: list[str]) -> dict[str, dict[str, Any]]:
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    with db.read() as cur:
        cur.execute(
            f"SELECT id, trust_score, confidence, verified, seen_count, source "
            f"FROM memories WHERE id IN ({placeholders})",
            ids,
        )
        return {str(row["id"]): dict(row) for row in cur.fetchall()}


def rank_results(
    db: RemnantDB,
    results: list[dict[str, Any]],
    *,
    profile: str = RANKING_PROFILE,
) -> list[dict[str, Any]]:
    """Rank resolved candidates without mutating evidence or lifecycle state."""
    if not results:
        return []
    quality_rows = _quality_rows(
        db, [str(row["id"]) for row in results if row.get("id") and not row.get("pending")]
    )
    raw_max = max((abs(float(row.get("score") or 0.0)) for row in results), default=1.0)
    raw_max = raw_max or 1.0
    seen_content: set[str] = set()
    ranked: list[dict[str, Any]] = []
    for row in results:
        item = dict(row)
        relevance = max(0.01, abs(float(item.get("score") or 0.0)) / raw_max)
        claim = item.get("claim") or {}
        lifecycle = str(item.get("claim_status") or claim.get("status") or "active")
        applicability = 1.0
        reasons: list[str] = []
        if item.get("valid_at_query") is False:
            applicability *= 0.25
            reasons.append("outside-validity")
        if item.get("condition_match") is False:
            applicability *= 0.70
            reasons.append("condition-mismatch")
        if lifecycle in {"superseded", "historical"}:
            applicability *= 0.45
            reasons.append("historical")
        if lifecycle in {"contradicted", "unresolved"}:
            applicability *= 0.85
            reasons.append("uncertain")

        evidence = quality_rows.get(str(item.get("id")), {})
        trust = float(evidence.get("trust_score") or 0.5)
        confidence = float(claim.get("confidence") or evidence.get("confidence") or 0.5)
        verified = 1.0 if evidence.get("verified") else 0.0
        corroboration = min(1.0, max(0.0, float(evidence.get("seen_count") or 1) - 1) / 4)
        signal = 0.40 * trust + 0.35 * confidence + 0.15 * verified + 0.10 * corroboration
        bounded_quality = 0.80 + 0.40 * min(1.0, max(0.0, signal))

        content_key = re.sub(r"\s+", " ", str(item.get("content") or "").casefold()).strip()
        diversity = 0.65 if content_key in seen_content else 1.0
        seen_content.add(content_key)
        final = relevance * applicability * bounded_quality * diversity
        item["score"] = final
        item["ranking"] = {
            "profile": profile,
            "relevance": relevance,
            "applicability": applicability,
            "applicability_reasons": reasons or ["current-and-applicable"],
            "quality": {
                "trust": trust,
                "confidence": confidence,
                "verified": verified,
                "corroboration": corroboration,
                "bounded": bounded_quality,
            },
            "diversity": diversity,
            "final": final,
        }
        ranked.append(item)
    ranked.sort(key=lambda item: float(item["score"]), reverse=True)
    return ranked


__all__ = ["RANKING_PROFILE", "rank_results"]
