"""Ingest pipeline: sync_turn fast path, transient filter, dedup, storage.

`sync_turn` writes the raw turn and enqueues extraction in a single SQLite
transaction so it returns in well under 10ms. Dedup combines BM25 candidates
with cosine similarity on embeddings.
"""

from __future__ import annotations

import logging
import re

from .config import RemnantConfig
from .db import RemnantDB
from .embed import Embedder, cosine

log = logging.getLogger("remnant.ingest")

# Transient-state detector. Rejects facts containing:
#   - percentages ("32%", "percent")
#   - clock times / am / pm
#   - "currently", "now", "today", "is at", "right now"
_TRANSIENT_RE = re.compile(
    r"\b\d{1,3}\s*%\b|\bpercent\b"  # percentages
    r"|\b\d{1,2}:\d{2}\s*(?:am|pm)?\b"  # times
    r"|\b(currently|now|is at|today|tonight|this morning|right now)\b",
    re.IGNORECASE,
)


def is_transient(text: str) -> bool:
    """True if the fact looks like transient state and should be rejected."""
    return bool(_TRANSIENT_RE.search(text or ""))


def store_memory(
    db: RemnantDB,
    embedder: Embedder,
    config: RemnantConfig,
    *,
    fact: str,
    entity: str,
    session_id: str,
    agent_id: str,
    visibility: str = "private",
    source_turn_id: int | None = None,
) -> str | None:
    """Store a fact with dedup. Returns the memory id, or None if deduped away."""
    fact = fact.strip()
    if not fact or is_transient(fact):
        return None

    # Dedup: BM25 candidates, then text-normalization + cosine similarity.
    candidates = db.candidate_facts(fact, agent_id=agent_id, limit=config.dedup_candidates)
    if candidates:
        norm = _normalize(fact)
        for c in candidates:
            # Cheap text near-dup first (no embedding work needed).
            if norm == _normalize(c.get("content", "")):
                log.debug("dedup (text): %s ~= %s", fact, c["content"])
                return None
        # Then cosine on embeddings for semantic near-dup.
        new_vec = embedder.embed(fact) if embedder else []
        for c in candidates:
            existing_vec = c.get("embedding", []) or []
            if existing_vec and new_vec:
                sim = cosine(new_vec, existing_vec)
                if sim >= config.dedup_cosine_threshold:
                    log.debug("dedup (cos=%.3f): %s ~= %s", sim, fact, c["content"])
                    return None

    embedding = embedder.embed(fact) if embedder else None
    if embedding is None:
        embedding = []
    mid = db.insert_memory(
        content=fact,
        source="manual",
        agent=agent_id,
        visibility=visibility,
        source_id=str(source_turn_id) if source_turn_id is not None else None,
        type="fact",
        tags=[entity] if entity else None,
        metadata=(
            {"session_id": session_id, "entity": entity}
            if entity
            else {"session_id": session_id}
        ),
        embedding=embedding or None,
        embed_model=getattr(embedder, "_model", None) if embedder else None,
    )
    return mid


def _normalize(s: str) -> str:
    s = (s or "").lower()
    s = s.strip().strip(".,!?;:")
    return re.sub(r"\s+", " ", s).strip()


def ingest_turn(
    db: RemnantDB,
    *,
    user_text: str,
    assistant_text: str,
    session_id: str,
    agent_id: str,
) -> int:
    """Fast path: persist the raw turn AND enqueue extraction atomically.

    This is the part called from sync_turn and must be sub-10ms. It does no
    network calls and no embedding work.
    """
    turn_id = db.insert_turn(
        session_id=session_id,
        agent_id=agent_id,
        user_text=user_text,
        assistant_text=assistant_text,
    )
    db.enqueue_extraction(
        turn_id=turn_id,
        session_id=session_id,
        agent_id=agent_id,
        user_text=user_text,
        assistant_text=assistant_text,
    )
    return turn_id


__all__ = ["is_transient", "store_memory", "ingest_turn"]
