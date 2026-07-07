"""Entity extraction, resolution, alias normalization, and memory linking.

Phase 3 wires the entity graph that Phase 1 stubbed out. This module is the
single integration point used by both the async extraction worker and the
manual `memory_store` tool path:

- `resolve_and_link`: resolve an entity name to its canonical id (creating it
  if needed), link it to a memory via `memory_entities`, and return the id +
  display name.
- `seed_relations`: create `related_to` edges between entities that co-occur
  in the same memory, with a strength derived from co-occurrence count.
- `extract_entities`: lightweight local entity extraction from text (proper
  nouns + typed keywords). No LLM; the extraction worker can still supply
  richer typed entities from its own LLM parse and pass them through.

Entity types (per spec): person, service, project, concept, place, tool.
Aliases are normalized by lowercasing + stripping punctuation; resolution
matches on name first, then any alias, scoped to the agent when one is given.

Issue #5 additions:
- ``_STOPLIST``: a module-level set of capitalized phrases (date/time terms,
  country/region names, generic tech nouns) dropped from regex extraction. It
  is configurable by passing a custom ``stoplist`` to ``extract_entities``.
- ``link_memory_entities(..., min_memories=N)``: a frequency threshold that
  defers entity creation until a name is sighted in >= N distinct memories.
  ``min_memories=1`` (default) keeps the original always-link behaviour.
- ``extract_and_link_entities``: the unified entry point. When typed entities
  from the LLM are supplied it links them directly and skips the regex pass,
  preventing double extraction noise.
"""

from __future__ import annotations

import re
from typing import Any

from .db import RemnantDB, _normalize_entity_name

# Recognized entity types (spec: person, service, project, concept, place, tool).
ENTITY_TYPES = {"person", "service", "project", "concept", "place", "tool"}

# Lightweight typed-keyword heuristics. Purely local; the extraction LLM may
# override these with explicit types in its JSON response.
_TYPE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("service", ["server", "daemon", "service", "api", "endpoint", "proxy", "agent"]),
    ("project", ["project", "repo", "repository", "pipeline", "migration", "build"]),
    ("tool", ["tool", "cli", "editor", "ide", "framework", "library", "plugin"]),
    ("place", ["lab", "datacenter", "office", "region", "rack", "site"]),
    ("person", ["sister", "brother", "colleague", "friend", "manager", "wife", "husband"]),
    ("concept", ["policy", "preference", "rule", "protocol", "standard", "plan"]),
]

# Proper nouns: capitalized words / multi-word capitalized phrases.
_PROPER_RE = re.compile(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\b")

# Common false positives to drop from proper-noun extraction.
_STOPWORDS = {
    "The", "A", "An", "Is", "Are", "Was", "Were", "Be", "This", "That",
    "These", "Those", "It", "We", "They", "He", "She", "I", "You",
    "User", "Assistant", "Remnant",
}

# Issue #5: a configurable stoplist of capitalized phrases that are NOT entities
# — they are temporal / geographic / generic-tech context rather than durable
# subjects worth tracking. Matched on the lowercased extracted phrase (after
# leading ``_STOPWORDS`` are stripped), so multi-word entries like "new zealand"
# or "hawke's bay" match the full phrase while a proper noun such as "Proxmox"
# never matches.
#
# Kept module-level (mirroring ``_STOPWORDS``) so callers and tests can extend
# it without touching the regex. The extraction LLM path is unaffected: typed
# entities from the model bypass ``extract_entities`` entirely.
_STOPLIST: set[str] = {
    # --- date / time terms -------------------------------------------------
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "today", "tonight", "tomorrow", "yesterday",
    "morning", "evening", "afternoon", "night", "midnight",
    "noon", "dawn", "dusk",
    # Year-like standalone tokens (the regex only captures capitalized forms,
    # but a few all-caps / abbreviated variants sneak through).
    "2024", "2025", "2026",
    # --- countries / regions / states / cities as location context ----------
    # Listed when they are merely *where* something happens, not a project or
    # organisation. A proper-noun lab / project named after a place still wins
    # because it is a multi-word phrase that does not match these single /
    # listed entries.
    "new zealand", "hawke's bay", "hawkes bay", "hawke bay",
    "pacific", "auckland", "wellington", "christchurch", "dunedin",
    "australia", "canada", "united states", "united kingdom", "europe",
    "asia", "africa", "americas",
    "north island", "south island",
    # --- generic tech nouns (only matched as the full standalone phrase) ---
    # "server", "api", "proxy" etc. are not specific systems/tools for this
    # lab; a proper name like "Proxmox Server" is a multi-word phrase that does
    # NOT match these single-word entries, so real systems survive.
    "server", "api", "proxy", "database", "endpoint", "client",
    "service", "daemon", "agent", "tool", "framework", "library",
    "plugin", "editor", "ide", "cli", "repository", "build", "pipeline",
    "migration", "project", "module", "package", "config", "backup",
}

# Issue #22: common English nouns/adjectives/verbs that regex extraction often
# treats as proper nouns when they appear capitalized at the start of a
# sentence. These are not durable named entities.
_COMMON_NOUNS: set[str] = {
    "people", "time", "year", "work", "life", "world", "man", "day", "thing",
    "woman", "child", "use", "way", "eye", "hand", "part", "place", "case",
    "week", "company", "system", "program", "question", "number", "group",
    "problem", "fact", "point", "right", "home", "water", "room", "area",
    "money", "story", "month", "lot", "book", "line", "kind", "head", "word",
    "house", "friend", "father", "mother", "girl", "boy", "side", "car",
    "information", "nothing", "everything", "something", "anything",
    "everyone", "someone", "anyone", "nobody", "somebody", "anybody",
    "everybody",
    "good", "bad", "new", "old", "first", "last", "long", "great", "little",
    "high", "small", "different", "large", "next", "early", "young",
    "important", "public", "same", "able", "certain", "clear", "full",
    "special", "free", "open", "short", "true", "possible", "hard",
    "strong", "whole", "easy", "real", "simple", "single", "early", "late",
    "local", "general", "main", "major", "following", "final", "initial",
    "total", "current", "modern", "available", "specific", "various",
    "personal", "private", "shared", "public", "common",
    # Issue #22 follow-up: extraction noise that leaked through in production
    # re-extraction. These are common verbs/participles/abstract nouns that
    # appear capitalized at sentence starts.
    "involving", "related", "summary", "decision", "agents", "agent",
    "after", "better", "key", "takeaways",
    "notes", "note", "overview", "context", "background", "introduction",
    "conclusion", "result", "results", "output", "action",
    "actions", "item", "items", "task", "tasks", "step", "steps",
    "phase", "update", "updates", "change", "changes", "fix", "fixes",
    "issue", "issues", "error", "errors", "warning", "warnings",
    "key takeaways", "related notes", "soul", "cv", "rtx",
}

# Issue #22: common English function words and generic response/sentence-starter
# words that the regex path sometimes extracts when they are sentence-initial
# and capitalized. These are not durable named entities.
_FUNCTION_WORDS: set[str] = {
    "and", "or", "but", "if", "then", "else", "for", "to", "of", "in",
    "on", "at", "by", "with", "from", "into", "onto", "upon", "as", "is",
    "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "have", "has", "had", "not", "no", "nor", "so", "than", "that", "this",
    "these", "those", "it", "its", "we", "us", "our", "they", "them",
    "their", "he", "him", "his", "she", "her", "you", "your", "i", "me",
    "my", "who", "whom", "whose", "what", "which", "when", "where", "why",
    "how", "all", "any", "some", "none", "both", "each", "every", "few",
    "more", "most", "other", "such", "only", "own", "same", "very", "just",
    "the", "a", "an",
    # Generic response words / capitalized sentence starters (issue #22).
    "yes", "no", "ok", "okay", "sure", "let", "well", "thanks", "please",
    "great", "good", "maybe", "will", "can", "could", "would", "should",
    "may", "might", "must", "need", "want", "like", "think", "know", "see",
    "make", "take", "come", "go", "get", "give", "look", "use", "find",
    "tell", "ask", "say", "said", "mean", "seem", "feel", "try", "keep",
    "put", "set", "run", "move", "turn", "start", "stop", "show", "help",
    "call", "called", "using", "add", "added", "done", "still", "also",
    "here", "there", "then", "thus", "however", "actually", "basically",
    "specifically", "currently", "recently", "now",
}

# Lowercased stopwords for case-insensitive filtering. Remnant is kept because
# the system name is a legitimate durable subject when it appears in text.
_STOPWORDS_LOWER: set[str] = {w.lower() for w in _STOPWORDS if w.lower() != "remnant"}


def normalize_aliases(aliases: list[str]) -> list[str]:
    """Lowercase + strip punctuation from each alias, drop empties/dups."""
    out: list[str] = []
    seen: set[str] = set()
    for a in aliases or []:
        n = _normalize_entity_name(a)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def guess_type(name: str, context: str = "") -> str | None:
    """Best-effort local type guess from name + surrounding context. No LLM."""
    text = f"{name} {context}".lower()
    for etype, keywords in _TYPE_KEYWORDS:
        if any(kw in text for kw in keywords):
            return etype
    return None


# Issue #22: sentence-aware extraction helpers.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def _split_sentences(text: str) -> list[str]:
    """Split text into rough sentences/paragraph chunks."""
    return [s.strip() for s in _SENTENCE_RE.split(text or "") if s.strip()]


def _clean_entity_match(match: str) -> str:
    """Strip leading stopwords and normalize whitespace from a regex match."""
    name = match.strip()
    parts = name.split()
    while parts and parts[0] in _STOPWORDS:
        parts = parts[1:]
    return " ".join(parts).strip()


def _is_subword(inner: str, outer: str) -> bool:
    """True when ``inner`` appears as a whole-word/phrase part of ``outer``."""
    return bool(re.search(r"\b" + re.escape(inner) + r"\b", outer, re.IGNORECASE))


def _suppress_substring_entities(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop entities that are strict substrings of a longer extracted entity.

    For example, if both ``Project Alpha`` and ``Alpha`` are extracted, keep
    the longer phrase and drop the bare ``Alpha``. Preserves the original
    candidate order after suppression.
    """
    by_length = sorted(candidates, key=lambda c: len(c["name"]), reverse=True)
    kept_keys: set[str] = set()
    kept: list[dict[str, Any]] = []
    for c in by_length:
        key = c["name"].lower()
        if any(key != k and _is_subword(key, k) for k in kept_keys):
            continue
        kept.append(c)
        kept_keys.add(key)
    # Restore original order.
    order = {id(c): i for i, c in enumerate(candidates)}
    kept.sort(key=lambda c: order[id(c)])
    return kept


def _entity_salience(info: dict[str, Any]) -> float:
    """Score an entity by frequency, sentence spread, and specificity.

    More mentions, broader sentence spread, and longer/multi-word names
    score higher. This lets us keep the top-N high-signal entities per memory.
    """
    mentions = info.get("mentions", 1)
    sentence_spread = len(info.get("sentence_ids", set()))
    word_count = len(info["name"].split())
    return mentions * (1.0 + 0.3 * sentence_spread) + 0.5 * word_count


def extract_entities(
    text: str,
    *,
    stoplist: set[str] | None = None,
    max_entities: int | None = 15,
) -> list[dict[str, Any]]:
    """Lightweight local entity extraction. No LLM, pure regex/keywords.

    Returns a list of ``{"name": str, "type": str|None, "aliases": [str]}``
    dicts. Proper nouns become entities; type is guessed from context if a
    keyword match exists, otherwise left None for the LLM extraction pass to
    fill in.

    ``stoplist`` (default: the module-level ``_STOPLIST``) drops capitalized
    phrases that are date/time terms, country/region names used as location
    context, or generic tech nouns without a proper name. Matching is on the
    lowercased phrase so multi-word entries like ``"new zealand"`` match while a
    real system such as ``"Proxmox Server"`` is preserved.

    ``max_entities`` (issue #22) caps the result to the top-N most salient
    entities. When None/0, all surviving candidates are returned (test mode).
    The default production callers pass 15.
    """
    if not text:
        return []
    deny = stoplist if stoplist is not None else _STOPLIST
    # Per-sentence extraction so we can later build sentence-co-occurrence
    # relations instead of complete graphs (issue #21).
    sentences = _split_sentences(text)
    raw: dict[str, dict[str, Any]] = {}
    for sid, sentence in enumerate(sentences):
        for match in _PROPER_RE.findall(sentence):
            name = _clean_entity_match(match)
            if not name or name in _STOPWORDS:
                continue
            key = name.lower()
            if key in raw:
                continue
            # Drop stopwords, common nouns, function words, and short noise.
            if (
                key in deny
                or key in _STOPWORDS_LOWER
                or key in _FUNCTION_WORDS
                or key in _COMMON_NOUNS
            ):
                continue
            if len(key) < 3 and not (name.isupper() or name.istitle()):
                continue
            raw[key] = {
                    "name": name,
                    "mentions": 0,
                    "sentence_ids": set(),
                    "type": guess_type(name, text),
                }
            raw[key]["mentions"] += 1
            raw[key]["sentence_ids"].add(sid)

    if not raw:
        return []

    candidates = list(raw.values())
    candidates = _suppress_substring_entities(candidates)
    candidates.sort(key=_entity_salience, reverse=True)
    if max_entities:
        candidates = candidates[:max_entities]

    return [
        {"name": c["name"], "type": c["type"], "aliases": []}
        for c in candidates
    ]


def _cooccurring_pairs_from_text(
    text: str,
    entity_ids: list[str],
    db: RemnantDB,
) -> set[tuple[str, str]]:
    """Return entity-id pairs that co-occur in the same sentence/paragraph.

    Uses the canonical display name and aliases for each entity. The pair is
    sorted so ``(a,b)`` and ``(b,a)`` collapse to a single undirected edge.
    """
    if not text or len(entity_ids) < 2:
        return set()
    entities_by_id = db.get_entities_batch(entity_ids)
    names_by_id: dict[str, set[str]] = {}
    for eid, row in entities_by_id.items():
        names: set[str] = {row["name"].lower()}
        aliases = row.get("aliases") or ""
        if aliases:
            for a in aliases.split(","):
                a = a.strip().lower()
                if a:
                    names.add(a)
        names_by_id[eid] = names

    sentences = _split_sentences(text)
    pairs: set[tuple[str, str]] = set()
    for sentence in sentences:
        lower = sentence.lower()
        present = [eid for eid in entity_ids if any(n in lower for n in names_by_id.get(eid, set()))]
        if len(present) < 2:
            continue
        for i, a in enumerate(present):
            for b in present[i + 1 :]:
                pair: tuple[str, str] = (a, b) if a < b else (b, a)
                pairs.add(pair)
    return pairs


def _rank_entity_ids_by_salience(
    entity_ids: list[str],
    db: RemnantDB,
) -> list[str]:
    """Re-order entity ids by the salience of their canonical names.

    Used as a last-resort ranking when the source text is unavailable.
    """
    rows = db.get_entities_batch(entity_ids)
    scored: list[tuple[float, str]] = []
    for eid in entity_ids:
        row = rows.get(eid)
        if not row:
            continue
        name = row.get("name") or ""
        salience = len(name.split()) * 0.5 + len(name) * 0.05
        scored.append((salience, eid))
    scored.sort(reverse=True)
    return [eid for _, eid in scored]


def extract_high_signal_entities(
    text: str,
    *,
    stoplist: set[str] | None = None,
    max_entities: int = 15,
) -> list[dict[str, Any]]:
    """Production entry point for entity extraction (issue #22).

    Returns at most ``max_entities`` (default 15) high-signal entities ranked by
    salience. This is the path used by vault indexing and import fallbacks.
    """
    return extract_entities(text, stoplist=stoplist, max_entities=max_entities)


def resolve_and_link(
    db: RemnantDB,
    *,
    memory_id: str,
    entity_name: str,
    agent_id: str | None = None,
    entity_type: str | None = None,
    aliases: list[str] | None = None,
    relation_role: str | None = None,
) -> tuple[str, str]:
    """Resolve `entity_name` to a canonical entity id, link it to `memory_id`,
    and return ``(entity_id, display_name)``.

    The display name is the original (non-normalized) name for UX; the id is
    the graph key. Aliases are normalized and merged into the entity row +
    the `entity_aliases` index.
    """
    display = (entity_name or "").strip()
    eid = db.resolve_entity(
        display, agent_id, entity_type=entity_type, aliases=aliases
    )
    if eid:
        db.link_entity(
            memory_id=memory_id,
            entity_id=eid,
            agent_id=agent_id,
            relation_role=relation_role,
        )
    return eid, display


def seed_relations(
    db: RemnantDB,
    *,
    memory_id: str,
    entity_ids: list[str],
    relation_type: str = "related_to",
    strength: float = 0.5,
    text: str | None = None,
    max_entities: int = 15,
) -> None:
    """Create undirected edges between co-occurring entities.

    Issue #21: no more complete graphs. When ``text`` is provided, only entity
    pairs that appear together in the same sentence/paragraph get an edge. When
    ``text`` is unavailable we fall back to a complete graph among the top
    ``max_entities`` entities (capped at 15 by default). Repeated co-occurrence
    strengthens the edge via the ``ON CONFLICT ... MAX`` upsert in
    ``db.add_relation``.
    """
    unique = [e for e in entity_ids if e]
    unique = list(dict.fromkeys(unique))[:max_entities]
    if len(unique) < 2:
        return

    if text:
        pairs = _cooccurring_pairs_from_text(text, unique, db)
        for a, b in pairs:
            db.add_relation(
                entity_a=a,
                entity_b=b,
                relation_type=relation_type,
                strength=strength,
                source_memory_id=memory_id,
            )
        return

    # Fallback complete graph (only when source text is unavailable).
    for i, a in enumerate(unique):
        for b in unique[i + 1 :]:
            if a == b:
                continue
            db.add_relation(
                entity_a=a,
                entity_b=b,
                relation_type=relation_type,
                strength=strength,
                source_memory_id=memory_id,
            )


def _rank_entities_by_text(
    entities: list[dict[str, Any]],
    text: str | None,
    max_entities: int,
) -> list[dict[str, Any]]:
    """Return the top ``max_entities`` entities by in-text salience.

    If ``text`` is unavailable, preserve the original order and cap.
    """
    if max_entities <= 0 or len(entities) <= max_entities:
        return list(entities)
    if not text:
        return list(entities[:max_entities])
    lower = text.lower()
    scored: list[tuple[float, dict[str, Any]]] = []
    for ent in entities:
        name = (ent.get("name") or "").strip()
        if not name:
            continue
        key = name.lower()
        mentions = lower.count(key)
        aliases = ent.get("aliases") or []
        for alias in aliases:
            mentions += lower.count(str(alias).lower())
        word_count = len(name.split())
        score = mentions * (1.0 + 0.5 * word_count)
        scored.append((score, ent))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ent for _, ent in scored[:max_entities]]

def link_memory_entities(
    db: RemnantDB,
    *,
    memory_id: str,
    entities: list[dict[str, Any]],
    agent_id: str | None = None,
    min_memories: int = 1,
    text: str | None = None,
    max_entities: int = 15,
) -> list[str]:
    """Resolve + link a list of typed entity dicts to a memory.

    Each dict: ``{"name": str, "type": str|None, "aliases": [str]}``.
    Returns the list of resolved entity ids linked to *this* memory (used for
    relation seeding).

    ``min_memories`` (issue #5) gates persistence: a newly extracted entity is
    only created/linked once it has been sighted in at least this many distinct
    memories. ``min_memories=1`` (the default) restores the original
    always-link behaviour and is used by the typed LLM path and the low-level
    API. ``min_memories>1`` defers creation via the ``entity_sightings`` table:
    the first sighting records a pending row, and the sighting that crosses the
    threshold creates the entity and links it to every sighted memory.
    Existing single-mention entities already in the DB are left in place.

    ``text`` (issue #21) provides sentence co-occurrence signal for relation
    seeding. ``max_entities`` caps the number of entities linked/related per
    memory to avoid graph bloat.
    """
    # Cap to the top-N salient entities before any persistence work.
    entities = _rank_entities_by_text(entities, text, max_entities)

    if min_memories <= 1:
        ids: list[str] = []
        for ent in entities:
            name = (ent.get("name") or "").strip()
            if not name:
                continue
            eid, _ = resolve_and_link(
                db,
                memory_id=memory_id,
                entity_name=name,
                agent_id=agent_id,
                entity_type=ent.get("type"),
                aliases=ent.get("aliases") or [],
            )
            if eid:
                ids.append(eid)
        if len(ids) >= 2:
            seed_relations(db, memory_id=memory_id, entity_ids=ids, text=text)
        return ids

    # Threshold path: defer entity creation until >= min_memories sightings.
    linked_ids: list[str] = []
    for ent in entities:
        name = (ent.get("name") or "").strip()
        if not name:
            continue
        name_key = _normalize_entity_name(name)
        if not name_key:
            continue
        eid = db.find_entity_by_name(name, agent_id=agent_id)
        existing = db.count_entity_links(eid) if eid else 0
        # Linking this memory would bring the total to ``existing + 1``.
        if existing >= min_memories - 1 or existing + 1 >= min_memories:
            if not eid:
                eid = db.resolve_entity(
                    name, agent_id, entity_type=ent.get("type"),
                    aliases=ent.get("aliases") or [],
                )
            if eid:
                db.link_entity(
                    memory_id=memory_id, entity_id=eid,
                    agent_id=agent_id, relation_role=None,
                )
                linked_ids.append(eid)
                db.clear_entity_sightings(name_key, agent_id)
            continue
        # Below threshold: record a sighting and promote only when the count
        # of distinct sighted memories reaches ``min_memories``.
        db.record_entity_sighting(name_key, agent_id, memory_id)
        sight_mids = db.entity_sighting_memory_ids(name_key, agent_id)
        if len(sight_mids) >= min_memories:
            if not eid:
                eid = db.resolve_entity(
                    name, agent_id, entity_type=ent.get("type"),
                    aliases=ent.get("aliases") or [],
                )
            if eid:
                for smid in sight_mids:
                    db.link_entity(
                        memory_id=smid, entity_id=eid,
                        agent_id=agent_id, relation_role=None,
                    )
                linked_ids.append(eid)
                db.clear_entity_sightings(name_key, agent_id)
    if len(linked_ids) >= 2:
        seed_relations(db, memory_id=memory_id, entity_ids=linked_ids, text=text)
    return linked_ids


def extract_and_link_entities(
    db: RemnantDB,
    *,
    memory_id: str,
    text: str,
    typed_entities: list[dict[str, Any]] | None = None,
    agent_id: str | None = None,
    min_memories: int = 1,
    stoplist: set[str] | None = None,
    max_entities: int = 15,
) -> list[str]:
    """Unified entity linking entry point (issue #5/#21/#22).

    When ``typed_entities`` is supplied (the LLM extraction path), link those
    directly and DO NOT additionally run the regex ``extract_entities`` pass —
    the model has already curated the entities, so re-running the regex would
    only add noise (dates, generic nouns, single-mention proper nouns).

    When ``typed_entities`` is empty/None (the regex fallback path used by
    ``import_sources`` and ``vault``), extract entities locally with
    ``extract_high_signal_entities`` (capped at ``max_entities``) and link them
    through ``link_memory_entities``.

    ``text`` is forwarded to relation seeding so only sentence-co-occurring
    entity pairs receive edges, preventing complete-graph relation explosions.
    """
    if typed_entities:
        entities = typed_entities
    else:
        entities = extract_high_signal_entities(
            text, stoplist=stoplist, max_entities=max_entities,
        )
    if not entities:
        return []
    return link_memory_entities(
        db, memory_id=memory_id, entities=entities,
        agent_id=agent_id, min_memories=min_memories,
        text=text, max_entities=max_entities,
    )


__all__ = [
    "ENTITY_TYPES",
    "_STOPWORDS",
    "_STOPLIST",
    "extract_entities",
    "extract_high_signal_entities",
    "extract_and_link_entities",
    "guess_type",
    "link_memory_entities",
    "normalize_aliases",
    "resolve_and_link",
    "seed_relations",
]
