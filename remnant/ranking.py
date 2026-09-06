"""Observable claim-aware ranking applied after candidate resolution."""

from __future__ import annotations

import re
from typing import Any

from .db import RemnantDB

RANKING_PROFILE = "claims-v1"

_SOURCE_AUTHORITY = {
    "manual": 1.0,
    "conversation": 0.9,
    "import": 0.8,
    "vault": 0.85,
    "hindsight": 0.7,
    "dream": 0.55,
    "cron": 0.6,
    "sensor": 0.65,
    "email": 0.7,
}


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
    # Search lanes have different score domains (BM25, cosine, graph and RRF).
    # Normalize within each lane before combining evidence quality; one raw
    # score must never be compared directly with another lane's score.
    lane_values: dict[str, list[float]] = {}
    for row in results:
        lane = str(row.get("_score_lane") or "unknown")
        value = float(row.get("score") or 0.0)
        lane_values.setdefault(lane, []).append(value)
    lane_bounds = {lane: (min(values), max(values)) for lane, values in lane_values.items()}

    def _relevance(row: dict[str, Any]) -> tuple[float, float, str]:
        lane = str(row.get("_score_lane") or "unknown")
        native = float(row.get("score") or 0.0)
        low, high = lane_bounds[lane]
        if high <= low:
            normalized = 1.0 if native > 0 else 0.01
        else:
            normalized = (native - low) / (high - low)
            normalized = 0.05 + 0.95 * max(0.0, min(1.0, normalized))
        return normalized, native, lane

    seen_content: set[str] = set()
    ranked: list[dict[str, Any]] = []
    for row in results:
        item = dict(row)
        relevance, native_score, score_lane = _relevance(item)
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
        source_authority = _SOURCE_AUTHORITY.get(str(evidence.get("source") or ""), 0.65)
        signal = (
            0.35 * trust
            + 0.30 * confidence
            + 0.15 * verified
            + 0.10 * corroboration
            + 0.10 * source_authority
        )
        bounded_quality = 0.80 + 0.40 * min(1.0, max(0.0, signal))

        content_key = re.sub(r"\s+", " ", str(item.get("content") or "").casefold()).strip()
        diversity = 0.65 if content_key in seen_content else 1.0
        seen_content.add(content_key)
        final = relevance * applicability * bounded_quality * diversity
        item["score"] = final
        item["ranking"] = {
            "profile": profile,
            "score_lane": score_lane,
            "native_score": native_score,
            "relevance": relevance,
            "applicability": applicability,
            "applicability_reasons": reasons or ["current-and-applicable"],
            "quality": {
                "trust": trust,
                "confidence": confidence,
                "verified": verified,
                "corroboration": corroboration,
                "source_authority": source_authority,
                "bounded": bounded_quality,
            },
            "diversity": diversity,
            "final": final,
        }
        ranked.append(item)
    ranked.sort(key=lambda item: float(item["score"]), reverse=True)
    return ranked


__all__ = ["RANKING_PROFILE", "rank_results"]
