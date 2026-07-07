"""Re-classify existing `related_to` relations into typed relations.

Works directly on the production DB by re-reading the source memory text
for each relation, finding the sentence where the two entities co-occur,
and applying lexical pattern matching to classify the relation type.

Usage:
    python -m remnant.classify_relations --dry-run   # preview
    python -m remnant.classify_relations --yes        # apply
"""

from __future__ import annotations

import argparse
import logging
import re
import sqlite3
import sys
from collections import Counter

log = logging.getLogger("remnant.classify_relations")

# ─── relation patterns ──────────────────────────────────────────────────────
# Each pattern is (regex, relation_type). Patterns are checked in order;
# first match wins. Patterns are case-insensitive and look for the verbal
# cue between or around the two entity names.

# Directional patterns: "X <verb> Y" → relation from X to Y
# We capture which entity appears before/after the verb to determine direction.
_DIRECTIONAL_PATTERNS = [
    # uses / utilizes / employs / runs on
    (r"\b(\w[\w\s]{0,30}?)\s+(?:uses|utilizes|employs|runs on|relies on|depends on)\s+(.+?)(?:[.,;:\n]|$)", "uses"),
    (r"\b(\w[\w\s]{0,30}?)\s+(?:is built on|is powered by|is based on|is running on)\s+(.+?)(?:[.,;:\n]|$)", "uses"),

    # owns / has / contains
    (r"\b(\w[\w\s]{0,30}?)\s+(?:owns|has|contains|includes|comprises)\s+(.+?)(?:[.,;:\n]|$)", "owns"),
    (r"\b(\w[\w\s]{0,30}?)\s+(?:is part of|belongs to|is a member of|is a component of)\s+(.+?)(?:[.,;:\n]|$)", "part_of"),

    # manages / manages / controls / administers
    (r"\b(\w[\w\s]{0,30}?)\s+(?:manages|controls|administers|orchestrates|coordinates)\s+(.+?)(?:[.,;:\n]|$)", "manages"),

    # created / built / developed / wrote
    (r"\b(\w[\w\s]{0,30}?)\s+(?:created|built|developed|wrote|designed|authored|established|set up|configured)\s+(.+?)(?:[.,;:\n]|$)", "created"),

    # depends on / requires / needs
    (r"\b(\w[\w\s]{0,30}?)\s+(?:depends on|requires|needs|consumes)\s+(.+?)(?:[.,;:\n]|$)", "depends_on"),

    # interacts with / connects to / talks to / sends to
    (r"\b(\w[\w\s]{0,30}?)\s+(?:connects to|talks to|sends to|interacts with|communicates with|integrates with|syncs with|pulls from|pushes to)\s+(.+?)(?:[.,;:\n]|$)", "interacts_with"),

    # monitors / watches / tracks / observes
    (r"\b(\w[\w\s]{0,30}?)\s+(?:monitors|watches|tracks|observes|surveys|scans)\s+(.+?)(?:[.,;:\n]|$)", "monitors"),

    # references / links to / cites / mentions
    (r"\b(\w[\w\s]{0,30}?)\s+(?:references|links to|cites|mentions|points to|refers to)\s+(.+?)(?:[.,;:\n]|$)", "references"),
]

# Bidirectional patterns — relation type doesn't depend on order
_BIDIRECTIONAL_PATTERNS = [
    (r"\b(\w[\w\s]{0,30}?)\s+(?:and|or|with|alongside|beside|plus|&)\s+(.+?)(?:[.,;:\n]|$)", "co_occurs"),
]


def classify_relation_in_text(
    sentence: str,
    name_a: str,
    name_b: str,
) -> str:
    """Classify the relation between two entities based on the sentence text.

    Returns a relation type string. Falls back to "related_to" if no pattern matches.
    """
    lower = sentence.lower()
    na = name_a.lower()
    nb = name_b.lower()

    # Find positions of both entities
    pos_a = lower.find(na)
    pos_b = lower.find(nb)
    if pos_a == -1 or pos_b == -1:
        return "related_to"

    # Determine which entity comes first
    if pos_a < pos_b:
        first_name, second_name = na, nb
        first_pos, second_pos = pos_a, pos_b
        a_is_first = True
    else:
        first_name, second_name = nb, na
        first_pos, second_pos = pos_b, pos_a
        a_is_first = False

    # Extract text between the two entities
    between = lower[first_pos + len(first_name):second_pos].strip()
    # Also check text after second entity for "X is part of Y" patterns
    after = lower[second_pos + len(second_name):].strip()[:100]
    # And text before first entity
    before = lower[:first_pos].strip()[-100:]

    # Check each verb type — directional types check the "between" text
    # for the verb, and determine direction from which entity comes first

    # Type: uses (first uses second)
    if any(v in between for v in _VERBS_BY_TYPE["uses"]):
        return "uses"

    # Type: depends_on (first depends on second)
    if any(v in between for v in _VERBS_BY_TYPE["depends_on"]):
        return "depends_on"

    # Type: owns (first owns/has/contains second)
    if any(v in between for v in _VERBS_BY_TYPE["owns"]):
        return "owns"

    # Type: manages (first manages/controls second)
    if any(v in between for v in _VERBS_BY_TYPE["manages"]):
        return "manages"

    # Type: created (first created/built second)
    if any(v in between for v in _VERBS_BY_TYPE["created"]):
        return "created"

    # Type: interacts_with (first connects to second)
    if any(v in between for v in _VERBS_BY_TYPE["interacts_with"]):
        return "interacts_with"

    # Type: monitors (first monitors second)
    if any(v in between for v in _VERBS_BY_TYPE["monitors"]):
        return "monitors"

    # Type: references (first references/links to second)
    if any(v in between for v in _VERBS_BY_TYPE["references"]):
        return "references"

    # Type: part_of — "X is part of Y" → first is part_of second
    if any(v in between for v in _VERBS_BY_TYPE["part_of"]):
        return "part_of"

    # Reverse direction checks: "Y is used by X" → X uses Y
    # Check if "is used by" / "is managed by" etc. appears between entities
    reverse_verbs = {
        "is used by": "uses",
        "is owned by": "owns",
        "is managed by": "manages",
        "is controlled by": "manages",
        "is created by": "created",
        "is built by": "created",
        "is developed by": "created",
        "is monitored by": "monitors",
        "is referenced by": "references",
        "is part of": "part_of",
    }
    for verb, rel_type in reverse_verbs.items():
        if verb in between:
            # In this case, the SECOND entity is the actor
            # (e.g., "Y is used by X" means X uses Y)
            # But since we're storing undirected relations, we just need the type
            return rel_type

    # Check for simple conjunction "X and Y" → co_occurs
    if any(v in between for v in _VERBS_BY_TYPE["co_occurs"]):
        return "co_occurs"

    return "related_to"


# Verb keywords indexed by relation type for fast lookup
_VERBS_BY_TYPE = {
    "uses": ["uses", "utilizes", "employs", "runs on", "relies on", "is built on", "is powered by", "is based on", "is running on"],
    "owns": ["owns", "has", "contains", "includes", "comprises"],
    "part_of": ["is part of", "belongs to", "is a member of", "is a component of"],
    "manages": ["manages", "controls", "administers", "orchestrates", "coordinates"],
    "created": ["created", "built", "developed", "wrote", "designed", "authored", "established", "set up", "configured"],
    "depends_on": ["depends on", "requires", "needs", "consumes"],
    "interacts_with": ["connects to", "talks to", "sends to", "interacts with", "communicates with", "integrates with", "syncs with", "pulls from", "pushes to"],
    "monitors": ["monitors", "watches", "tracks", "observes", "surveys", "scans"],
    "references": ["references", "links to", "cites", "mentions", "points to", "refers to"],
    "co_occurs": ["and", "or", "with", "alongside", "beside", "plus"],
}


def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    if not text:
        return []
    # Simple sentence splitter
    parts = re.split(r"[.!?]+|\n\n+", text)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 10]


def classify_all_relations(db_path: str, dry_run: bool = True) -> dict:
    """Re-classify all related_to relations in the DB.

    For each relation, find the source memory, locate the sentence where
    the two entities co-occur, and apply lexical pattern matching.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Get all related_to relations with entity names and source memory
    c.execute("""
        SELECT r.rowid, r.entity_a, r.entity_b, r.strength,
               e1.name, e2.name, r.source_memory_id
        FROM relations r
        JOIN entities e1 ON r.entity_a = e1.id
        JOIN entities e2 ON r.entity_b = e2.id
        WHERE r.relation_type = 'related_to'
    """)
    relations = c.fetchall()
    log.info("Found %d related_to relations to classify", len(relations))

    # Cache memory texts
    memory_cache: dict[str, str] = {}
    type_counts: Counter = Counter()
    updates: list[tuple[str, str]] = []  # (rowid, new_type)

    for row in relations:
        rowid, eid_a, eid_b, strength, name_a, name_b, source_mid = row

        # Get source memory text
        if source_mid not in memory_cache:
            c.execute("SELECT content FROM memories WHERE id = ?", (source_mid,))
            mrow = c.fetchone()
            memory_cache[source_mid] = mrow[0] if mrow else ""

        text = memory_cache.get(source_mid, "")
        if not text:
            type_counts["related_to"] += 1
            continue

        # Find the sentence where both entities co-occur
        sentences = split_sentences(text)
        best_type = "related_to"
        for sentence in sentences:
            lower = sentence.lower()
            if name_a.lower() in lower and name_b.lower() in lower:
                rel_type = classify_relation_in_text(sentence, name_a, name_b)
                if rel_type != "related_to":
                    best_type = rel_type
                    break  # Found a typed relation, stop searching

        type_counts[best_type] += 1
        if best_type != "related_to":
            updates.append((best_type, rowid))

    log.info("Classification results:")
    for rt, count in type_counts.most_common():
        log.info("  %s: %d", rt, count)

    if not dry_run:
        # Apply updates
        for new_type, rowid in updates:
            c.execute("UPDATE relations SET relation_type = ? WHERE rowid = ?", (new_type, rowid))
        conn.commit()
        log.info("Applied %d typed relation updates", len(updates))

        # VACUUM
        c.execute("VACUUM")
        conn.commit()

    conn.close()
    return {"type_counts": dict(type_counts), "updates": len(updates), "dry_run": dry_run}


def main():
    parser = argparse.ArgumentParser(description="Classify related_to relations into typed relations")
    parser.add_argument("--dry-run", action="store_true", help="Preview without applying changes")
    parser.add_argument("--yes", action="store_true", help="Apply changes (default is dry-run)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    db_path = "/home/jd/.hermes/remnant/remnant.db"
    result = classify_all_relations(db_path, dry_run=not args.yes)

    print(f"\n{'DRY RUN' if result['dry_run'] else 'APPLIED'}: {result['updates']} relations would be{' ' if result['dry_run'] else ' '}typed")
    print(f"Type distribution: {result['type_counts']}")


if __name__ == "__main__":
    main()