"""Ingest pipeline: sync_turn fast path, transient filter, dedup, storage,
and contradiction detection.

`sync_turn` writes the raw turn and enqueues extraction in a single SQLite
transaction so it returns in well under 10ms. Dedup combines BM25 candidates
with cosine similarity on embeddings. Contradiction detection compares a new
fact against existing memories sharing an entity using a lightweight local
heuristic (negation words, antonyms); ambiguous cases are flagged for the
LLM extraction pass to resolve rather than blocking storage.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .config import RemnantConfig
from .db import RemnantDB
from .embed import Embedder, cosine
from .entity import link_memory_entities

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

# Contradiction heuristic: negation flip words. If a new fact and an existing
# fact share an entity and differ only by negation, flag both.
_NEGATION_WORDS = {
    "not", "no", "never", "none", "n't", "cannot", "cant", "without",
    "isn't", "wasn't", "aren't", "weren't", "don't", "doesn't", "didn't",
    "won't", "wouldn't", "shouldn't", "couldn't", "isn", "wasn", "aren",
}
# Common antonym pairs (lowercased). A local-only first pass.
# Stored as tuples (not sets) so the two sides are addressable by index.
_ANTONYMS = [
    ("on", "off"),
    ("online", "offline"),
    ("up", "down"),
    ("open", "closed"),
    ("enabled", "disabled"),
    ("yes", "no"),
    ("true", "false"),
    ("active", "inactive"),
    ("private", "shared"),
    ("light", "dark"),
    ("start", "stop"),
    ("allow", "block"),
    ("accept", "reject"),
    ("pass", "fail"),
]


def is_transient(text: str) -> bool:
    """True if the fact looks like transient state and should be rejected."""
    return bool(_TRANSIENT_RE.search(text or ""))


def detect_contradiction(new_fact: str, existing: str) -> bool:
    """Lightweight local contradiction heuristic. No LLM.

    Returns True when the two facts look like they assert opposites:
      - one is the negation of the other (a negation word flips between them)
      - they contain a direct antonym pair on the same subject
    This is a conservative first pass; ambiguous cases are left for the LLM
    extraction pass to confirm (we flag metadata, we don't block storage).
    """
    new_tokens = _tokenize(new_fact)
    old_tokens = _tokenize(existing)
    new_neg = sum(1 for t in new_tokens if t in _NEGATION_WORDS)
    old_neg = sum(1 for t in old_tokens if t in _NEGATION_WORDS)
    # Tokens minus negation words; used for overlap/antonym checks.
    new_set = set(new_tokens) - _NEGATION_WORDS
    old_set = set(old_tokens) - _NEGATION_WORDS
    # Negation flip: one side adds/removes a negation word while the rest of
    # the tokens largely overlap (allowing for morphology like "likes"/"like").
    if abs(new_neg - old_neg) >= 1 and new_set and old_set:
        overlap = new_set & old_set
        smaller = min(len(new_set), len(old_set))
        if overlap and len(overlap) >= smaller - 1:
            return True
    # Antonym pair present on a shared subject.
    for pair in _ANTONYMS:
        if (pair[0] in new_tokens and pair[1] in old_tokens) or (
            pair[1] in new_tokens and pair[0] in old_tokens
        ):
            # Only flag when the rest of the tokens largely overlap.
            overlap = new_set & old_set
            if len(overlap) >= max(1, min(len(new_set), len(old_set)) - 2):
                return True
    return False


def _tokenize(text: str) -> list[str]:
    return [t.lower().strip(".,!?;:\"'()[]") for t in re.split(r"\s+", text or "") if t]


def _normalize(s: str) -> str:
    s = (s or "").lower()
    s = s.strip().strip(".,!?;:")
    return re.sub(r"\s+", " ", s).strip()


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
    entities: list[dict[str, Any]] | None = None,
) -> str | None:
    """Store a fact with dedup + contradiction flagging. Returns memory id.

    ``entities`` is an optional list of typed entity dicts
    (``{"name", "type", "aliases"}``) from the extraction LLM. When supplied,
    each is resolved, linked to the memory via ``memory_entities``, and used
    to seed ``relations`` edges between co-occurring entities. When omitted,
    ``entity`` (the legacy single-subject string) is treated as one entity of
    unknown type for backward compatibility.
    """
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
        new_vec = embedder.embed(fact) if embedder else None
        if new_vec:
            for c in candidates:
                existing_vec = c.get("embedding", []) or []
                if existing_vec:
                    sim = cosine(new_vec, existing_vec)
                    if sim >= config.dedup_cosine_threshold:
                        log.debug("dedup (cos=%.3f): %s ~= %s", sim, fact, c["content"])
                        return None

    # Contradiction detection: compare against existing active memories
    # that share an entity with this fact. Flag both via metadata, do not
    # block storage.
    contradiction_targets: list[str] = []
    if entities:
        for ent in entities:
            name = (ent.get("name") or "").strip()
            if not name:
                continue
            eid = db.find_entity_by_name(name, agent_id=agent_id)
            if not eid:
                continue
            for m in db.get_memories_for_entity(eid, agent_id=agent_id):
                if detect_contradiction(fact, m.get("content", "")):
                    contradiction_targets.append(m["id"])
                    _flag_contradiction(db, m["id"], fact)
            if contradiction_targets:
                break

    embedding = embedder.embed(fact) if embedder else None
    # embed() returns None on remote failure; pass None through so no embedding
    # row is stored (insert_memory only writes a row when embedding is truthy).
    meta: dict[str, Any] = {"session_id": session_id}
    if entity:
        meta["entity"] = entity
    if contradiction_targets:
        meta["contradicts"] = contradiction_targets
    mid = db.insert_memory(
        content=fact,
        source="manual",
        agent=agent_id,
        visibility=visibility,
        source_id=str(source_turn_id) if source_turn_id is not None else None,
        type="fact",
        tags=[entity] if entity else None,
        metadata=meta,
        embedding=embedding or None,
        embed_model=getattr(embedder, "_model", None) if embedder else None,
    )
    if mid is None:
        return None

    # Wire the entity graph: resolve + link typed entities (or the legacy
    # single `entity` subject) and seed relations between co-occurring ones.
    if entities:
        link_memory_entities(db, memory_id=mid, entities=entities, agent_id=agent_id)
    elif entity:
        link_memory_entities(
            db,
            memory_id=mid,
            entities=[{"name": entity, "type": None, "aliases": []}],
            agent_id=agent_id,
        )

    return mid


def _flag_contradiction(db: RemnantDB, memory_id: str, new_fact: str) -> None:
    """Append a `contradicts` entry to an existing memory's metadata."""
    mem = db.get_memory(memory_id)
    if mem is None:
        return
    meta = mem.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    contradicts = set(meta.get("contradicts") or [])
    contradicts.add(new_fact)
    meta["contradicts"] = sorted(contradicts)
    db.set_memory_field(
        memory_id,
        "metadata",
        meta,
        actor="system",
        action="contradiction_flag",
        details={"new_fact": new_fact},
    )


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
