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


def extract_entities(
    text: str,
    *,
    stoplist: set[str] | None = None,
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
    """
    if not text:
        return []
    deny = stoplist if stoplist is not None else _STOPLIST
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for match in _PROPER_RE.findall(text):
        name = match.strip()
        if not name:
            continue
        # A multi-word capitalized phrase may begin with a stopword
        # ("The Proxmox", "A Project Alpha"); strip leading stopwords so the
        # real proper noun ("Proxmox") is what we extract.
        parts = name.split()
        while parts and parts[0] in _STOPWORDS:
            parts = parts[1:]
        name = " ".join(parts).strip()
        if not name or name in _STOPWORDS:
            continue
        key = name.lower()
        if key in seen:
            continue
        # Issue #5: drop dates, generic places, and bare tech nouns.
        if key in deny:
            continue
        seen.add(key)
        out.append({
            "name": name,
            "type": guess_type(name, text),
            "aliases": [],
        })
    return out


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
) -> None:
    """Create undirected edges between every pair of co-occurring entities.

    Relations are seeded from entities that appear together in the same memory.
    Strength is a fixed baseline (0.5); repeated co-occurrence strengthens the
    edge via the `ON CONFLICT ... MAX` upsert in `db.add_relation`.
    """
    unique = [e for e in entity_ids if e]
    unique = list(dict.fromkeys(unique))  # dedupe, preserve order
    if len(unique) < 2:
        return
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


def link_memory_entities(
    db: RemnantDB,
    *,
    memory_id: str,
    entities: list[dict[str, Any]],
    agent_id: str | None = None,
    min_memories: int = 1,
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
    """
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
            seed_relations(db, memory_id=memory_id, entity_ids=ids)
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
        seed_relations(db, memory_id=memory_id, entity_ids=linked_ids)
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
) -> list[str]:
    """Unified entity linking entry point (issue #5).

    When ``typed_entities`` is supplied (the LLM extraction path), link those
    directly and DO NOT additionally run the regex ``extract_entities`` pass —
    the model has already curated the entities, so re-running the regex would
    only add noise (dates, generic nouns, single-mention proper nouns).

    When ``typed_entities`` is empty/None (the regex fallback path used by
    ``import_sources`` and ``vault``), extract entities locally with
    ``extract_entities`` and link them through ``link_memory_entities``.

    ``min_memories`` is forwarded to ``link_memory_entities`` so the regex
    fallback can be gated by the frequency threshold while the typed path
    bypasses it.
    """
    if typed_entities:
        entities = typed_entities
    else:
        entities = extract_entities(text, stoplist=stoplist)
    if not entities:
        return []
    return link_memory_entities(
        db, memory_id=memory_id, entities=entities,
        agent_id=agent_id, min_memories=min_memories,
    )


__all__ = [
    "ENTITY_TYPES",
    "_STOPWORDS",
    "_STOPLIST",
    "extract_entities",
    "extract_and_link_entities",
    "guess_type",
    "link_memory_entities",
    "normalize_aliases",
    "resolve_and_link",
    "seed_relations",
]
