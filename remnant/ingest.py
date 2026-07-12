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


def _initial_trust_score(source: str) -> float:
    """Source-based initial trust score for a new memory (issue #11)."""
    return {
        "import": 0.9,
        "manual": 0.9,
        "vault": 0.8,
        "hindsight": 0.7,
        "conversation": 0.6,
        "dream": 0.6,
    }.get(source, 0.5)


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
    source: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    source_text: str | None = None,
) -> str | None:
    """Store a fact with dedup + contradiction flagging. Returns memory id.

    ``entities`` is an optional list of typed entity dicts
    (``{"name", "type", "aliases"}``) from the extraction LLM. When supplied,
    each is resolved, linked to the memory via ``memory_entities``, and used
    to seed ``relations`` edges between co-occurring entities. When omitted,
    ``entity`` (the legacy single-subject string) is treated as one entity of
    unknown type for backward compatibility.

    ``source`` (optional) overrides the default source inference. When None
    the source defaults to ``conversation`` (when ``source_turn_id`` is set)
    or ``manual`` otherwise. When supplied, the value is used verbatim and
    must satisfy the ``memories.source`` CHECK constraint (e.g. ``dream``).

    ``tags`` (optional) overrides the default ``[entity]`` tag list. When
    None the legacy single-element tag list is used (or no tags when there
    is no entity).

    ``metadata`` (optional) is merged into the per-memory metadata dict
    alongside the internally-managed ``session_id`` / ``entity`` /
    ``contradicts`` keys; caller keys win on collision.

    ``source_text`` (issue #21) is the original text/fact used to derive
    sentence co-occurrence for relation seeding. When supplied, relations
    are only created between entities that appear together in the same
    sentence/paragraph.
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
    if metadata:
        meta.update(metadata)
    resolved_source = source if source is not None else (
        "conversation" if source_turn_id is not None else "manual"
    )
    mid = db.insert_memory(
        content=fact,
        source=resolved_source,
        agent=agent_id,
        visibility=visibility,
        source_id=str(source_turn_id) if source_turn_id is not None else None,
        type="fact",
        tags=tags if tags is not None else ([entity] if entity else None),
        metadata=meta,
        trust_score=_initial_trust_score(resolved_source),
        embedding=embedding or None,
        embed_model=getattr(embedder, "_model", None) if embedder else None,
    )
    if mid is None:
        return None

    # Wire the entity graph: resolve + link typed entities (or the legacy
    # single `entity` subject) and seed relations between co-occurring ones.
    if entities:
        link_memory_entities(
            db, memory_id=mid, entities=entities, agent_id=agent_id,
            text=source_text or fact,
        )
    elif entity:
        link_memory_entities(
            db,
            memory_id=mid,
            entities=[{"name": entity, "type": None, "aliases": []}],
            agent_id=agent_id,
            text=source_text or fact,
        )

    # Corroboration boost (issue #11): for each entity linked to this new
    # memory, find other active memories sharing the entity and bump their
    # trust_score (and the new memory's own) by +0.05 capped at 0.95.
    # Contradicted memories are excluded from the boost.
    _corroborate(db, mid, agent_id=agent_id, contradiction_targets=contradiction_targets)

    return mid


def _flag_contradiction(db: RemnantDB, memory_id: str, new_fact: str) -> None:
    """Append a `contradicts` entry to an existing memory's metadata and apply a
    trust penalty (issue #11): trust_score drops by 0.1, floored at 0.3."""
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
    current = float(mem.get("trust_score") or 0.5)
    new_score = max(current - 0.1, 0.3)
    db.set_memory_field(
        memory_id,
        "trust_score",
        new_score,
        actor="system",
        action="trust_penalty",
        details={"new_fact": new_fact, "delta": -0.1},
    )


def _corroborate(
    db: RemnantDB,
    mid: str,
    *,
    agent_id: str | None,
    contradiction_targets: list[str],
) -> None:
    """Corroboration boost (issue #11).

    For each entity linked to the new memory, find other active memories that
    share that entity. Each such memory (not in ``contradiction_targets``) gets
    +0.05 trust_score (capped at 0.95). The new memory itself is bumped once if
    at least one corroborating active memory was found.
    """
    # Collect the entity ids linked to the new memory directly from
    # memory_entities (no db helper exists for this direction).
    with db.read() as cur:
        cur.execute(
            "SELECT entity_id FROM memory_entities WHERE memory_id=?",
            (mid,),
        )
        linked_eids = [r["entity_id"] for r in cur.fetchall()]
    if not linked_eids:
        return

    contradicted = set(contradiction_targets)
    corroborated_self = False
    boosted: set[str] = set()
    for eid in linked_eids[:5]:
        others = db.get_memories_for_entity(eid, agent_id=agent_id)
        for m in others:
            other_id = m.get("id")
            if not other_id or other_id == mid:
                continue
            if other_id in contradicted or other_id in boosted:
                continue
            if m.get("status") != "active":
                continue
            boosted.add(other_id)
            entity_name = db.entity_name_for(eid)
            # get_memories_for_entity does not select trust_score; fetch the
            # current row so the boost is applied to the real value.
            current_mem = db.get_memory(other_id)
            if current_mem is None:
                continue
            current = float(current_mem.get("trust_score") or 0.5)
            new_score = min(current + 0.05, 0.95)
            db.set_memory_field(
                other_id,
                "trust_score",
                new_score,
                actor="system",
                action="trust_corroborate",
                details={"shared_entity": entity_name, "source_memory": mid},
            )
            corroborated_self = True

    if corroborated_self:
        new_mem = db.get_memory(mid)
        if new_mem is not None:
            current = float(new_mem.get("trust_score") or 0.5)
            new_score = min(current + 0.05, 0.95)
            db.set_memory_field(
                mid,
                "trust_score",
                new_score,
                actor="system",
                action="trust_corroborate",
                details={"source_memory": mid},
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
    return db.insert_turn_with_extraction(
        session_id=session_id,
        agent_id=agent_id,
        user_text=user_text,
        assistant_text=assistant_text,
    )


__all__ = ["is_transient", "store_memory", "ingest_turn", "_initial_trust_score"]
