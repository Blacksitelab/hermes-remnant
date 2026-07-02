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


def extract_entities(text: str) -> list[dict[str, Any]]:
    """Lightweight local entity extraction. No LLM, pure regex/keywords.

    Returns a list of ``{"name": str, "type": str|None, "aliases": [str]}``
    dicts. Proper nouns become entities; type is guessed from context if a
    keyword match exists, otherwise left None for the LLM extraction pass to
    fill in.
    """
    if not text:
        return []
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
) -> list[str]:
    """Resolve + link a list of typed entity dicts to a memory.

    Each dict: ``{"name": str, "type": str|None, "aliases": [str]}``.
    Returns the list of resolved entity ids (for relation seeding).
    """
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


__all__ = [
    "ENTITY_TYPES",
    "extract_entities",
    "guess_type",
    "link_memory_entities",
    "normalize_aliases",
    "resolve_and_link",
    "seed_relations",
]
