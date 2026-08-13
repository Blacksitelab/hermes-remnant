"""Claim-aware resolution between candidate retrieval and context formatting.

The search lanes remain responsible for finding candidates.  This module is a
small, deterministic policy layer that groups versions, applies validity and
condition hints, and annotates uncertainty.  It never deletes evidence and it
does not write during recall.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from .db import RemnantDB

_HISTORY_RE = re.compile(
    r"\b(when|then|before|previously|used to|historical|history|at that time)\b",
    re.I,
)
_DATE_RE = re.compile(r"\b(20\d{2}-[01]\d-[0-3]\d)\b")


def retrieval_query(query: str) -> str:
    """Remove intent-only temporal words that weaken strict lexical search."""
    value = _HISTORY_RE.sub(" ", query or "")
    value = re.sub(r"\s+", " ", value).strip(" ?.,")
    return value or str(query or "").strip()


def _parse_qualifiers(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, json.JSONDecodeError):
            return {}
    return {}


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _is_valid_at(claim: dict[str, Any], at: datetime) -> bool:
    start = _parse_time(claim.get("valid_from") or claim.get("event_at"))
    end = _parse_time(claim.get("valid_to"))
    if start and start > at:
        return False
    if end and end <= at:
        return False
    return True


def _condition_matches(claim: dict[str, Any], query: str) -> bool:
    q = query.casefold()
    scope = " ".join(
        str(v or "") for v in (claim.get("scope_type"), claim.get("scope_value"))
    ).casefold()
    qualifiers = _parse_qualifiers(claim.get("qualifiers"))
    conditions = qualifiers.get("conditions") or []
    condition_text = " ".join(str(v) for v in conditions).casefold()
    if not scope and not condition_text:
        return True
    tokens = set(re.findall(r"[a-z0-9_-]+", scope + " " + condition_text))
    query_tokens = set(re.findall(r"[a-z0-9_-]+", q))
    return bool(tokens & query_tokens)


def resolve_results(
    db: RemnantDB,
    results: list[dict[str, Any]],
    *,
    query: str,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Resolve claim-bearing results while retaining evidence metadata."""
    if not results:
        return []
    now = now or datetime.now(timezone.utc)
    date_match = _DATE_RE.search(query or "")
    if date_match:
        try:
            now = datetime.fromisoformat(date_match.group(1)).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    claims = db.get_claims_for_memories([str(row.get("id")) for row in results if row.get("id")])
    history = bool(_HISTORY_RE.search(query or ""))
    enriched: list[dict[str, Any]] = []
    for row in results:
        item = dict(row)
        claim = claims.get(str(row.get("id")))
        if claim:
            item["claim"] = claim
            item["claim_status"] = claim.get("resolution_status") or claim.get("status")
            item["conflict_type"] = claim.get("conflict_type")
            item["condition_match"] = _condition_matches(claim, query)
            item["valid_at_query"] = _is_valid_at(claim, now)
        enriched.append(item)

    # Candidate lanes may return the same evidence text through multiple rows
    # or legacy duplicate memories. Prefer a claim-bearing projection, then
    # confidence and relevance, before claim grouping consumes context slots.
    by_content: dict[str, dict[str, Any]] = {}
    for item in enriched:
        key = re.sub(r"\s+", " ", str(item.get("content") or "").casefold()).strip()
        previous = by_content.get(key)
        item_key = (
            bool(item.get("claim")),
            float((item.get("claim") or {}).get("confidence", 0.0) or 0.0),
            float(item.get("score", 0.0) or 0.0),
        )
        previous_key = (
            bool((previous or {}).get("claim")),
            float(((previous or {}).get("claim") or {}).get("confidence", 0.0) or 0.0),
            float((previous or {}).get("score", 0.0) or 0.0),
        )
        if previous is None or item_key > previous_key:
            by_content[key] = item
    enriched = list(by_content.values())

    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    unclaimed: list[dict[str, Any]] = []
    for item in enriched:
        claim = item.get("claim")
        if not claim:
            unclaimed.append(item)
            continue
        key = (
            str(claim.get("subject") or "").casefold(),
            str(claim.get("predicate") or "").casefold(),
            str(claim.get("scope_type") or "").casefold(),
            str(claim.get("scope_value") or "").casefold(),
        )
        groups.setdefault(key, []).append(item)

    selected = list(unclaimed)
    for items in groups.values():
        if history and not date_match:
            selected.extend(items)
            continue
        matching = [item for item in items if item.get("condition_match")]
        if matching:
            items = matching
        valid = [item for item in items if item.get("valid_at_query", True)]
        if valid:
            items = valid
        # A contradiction/unresolved group stays visible as one result so the
        # context compiler can state uncertainty without spending all slots on
        # paraphrases and historical versions.
        items.sort(
            key=lambda item: (
                item.get("claim_status") not in {"contradicted", "unresolved"},
                item.get("claim", {}).get("confidence", 0.5),
                item.get("score", 0.0),
                item.get("claim", {}).get("updated_at", ""),
            ),
            reverse=True,
        )
        winner = dict(items[0])
        if len(items) > 1:
            winner["claim_group"] = [
                {
                    "id": str(item.get("id")),
                    "content": item.get("content", ""),
                    "claim": item.get("claim", {}),
                }
                for item in items
            ]
            statuses = {str(item.get("claim_status") or "") for item in items}
            if "unresolved" in statuses or "contradiction" in {
                str(item.get("conflict_type") or "") for item in items
            }:
                winner["claim_status"] = "unresolved"
                winner["conflict_type"] = "unresolved"
        selected.append(winner)

    selected.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
    return selected


__all__ = ["resolve_results", "retrieval_query"]
