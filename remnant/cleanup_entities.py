"""Entity graph cleanup script (issue #17).

Provides a safe, idempotent cleanup mechanism for the entity graph. Noise
entities are identified by:

- being in a stoplist (date/time terms, generic tech nouns, common English
  stopwords)
- having no letters or being very short (length <= 2)
- having fewer than ``min_memories`` active memory links

In dry-run mode the script only reports what it would delete. In live mode it
deletes the noise entities; because the schema uses ``ON DELETE CASCADE``,
associated ``entity_aliases``, ``memory_entities``, and ``relations`` rows are
removed automatically.

Usage::

    python -m remnant.cleanup_entities --dry-run
    python -m remnant.cleanup_entities --yes
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .db import RemnantDB, _normalize_entity_name, default_db_path, open_db
from .entity import _STOPLIST

log = logging.getLogger("remnant.cleanup_entities")

# Basic English stopwords that the LLM might emit as standalone "entities".
_DEFAULT_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "we", "they", "he", "she",
    "i", "you", "me", "him", "her", "us", "them", "my", "your", "his", "our",
    "their", "and", "or", "but", "if", "then", "else", "of", "to", "in",
    "for", "on", "with", "at", "by", "from", "as", "about", "into", "through",
    "during", "before", "after", "above", "below", "between", "among",
    "within", "without", "against", "under", "over", "again", "further",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "can", "will", "should", "now", "also", "one", "two", "three", "new",
    "old", "first", "last", "long", "great", "little", "good", "bad", "big",
}


def _is_noise(name: str, stoplist: set[str]) -> tuple[bool, str]:
    """Return (is_noise, reason) for an entity name."""
    n = (name or "").strip()
    if not n:
        return True, "empty"
    if len(n) <= 2:
        return True, "too_short"
    if not re.search(r"[a-zA-Z]", n):
        return True, "no_letters"
    key = _normalize_entity_name(n)
    if key in stoplist or key in _DEFAULT_STOPWORDS:
        return True, "stopword"
    return False, ""


def _build_stoplist(extra: set[str] | None = None) -> set[str]:
    stoplist = set(_STOPLIST) | _DEFAULT_STOPWORDS
    if extra:
        stoplist |= {_normalize_entity_name(x) for x in extra}
    return stoplist


def find_noise_entities(
    db: RemnantDB,
    *,
    stoplist: set[str] | None = None,
    min_memories: int = 2,
) -> list[tuple[str, str, str]]:
    """Identify noise entities.

    Returns a list of ``(entity_id, entity_name, reason)`` tuples. Reasons are:
    ``stopword``, ``too_short``, ``no_letters``, or ``too_few_links``.
    """
    stoplist = _build_stoplist(stoplist)
    noise: list[tuple[str, str, str]] = []
    with db.read() as cur:
        cur.execute("SELECT id, name FROM entities")
        rows = [dict(r) for r in cur.fetchall()]
    for row in rows:
        eid = row["id"]
        name = row["name"] or ""
        is_noise, reason = _is_noise(name, stoplist)
        if is_noise:
            noise.append((eid, name, reason))
            continue
        with db.read() as cur:
            cur.execute(
                "SELECT COUNT(DISTINCT memory_id) AS c FROM memory_entities "
                "WHERE entity_id=? AND memory_id IN (SELECT id FROM memories WHERE status='active')",
                (eid,),
            )
            count = int(cur.fetchone()["c"])
        if count < min_memories:
            noise.append((eid, name, "too_few_links"))
    return noise


def cleanup_entities(
    db: RemnantDB,
    *,
    dry_run: bool = True,
    stoplist: set[str] | None = None,
    min_memories: int = 2,
) -> dict[str, Any]:
    """Delete noise entities from the graph.

    In dry-run mode no rows are deleted; the returned stats dict reports what
    would be removed. In live mode the entities are deleted and the schema's
    ``ON DELETE CASCADE`` handles associated aliases, memory links, and
    relations.
    """
    noise = find_noise_entities(db, stoplist=stoplist, min_memories=min_memories)
    by_reason: dict[str, int] = {}
    for _, _, reason in noise:
        by_reason[reason] = by_reason.get(reason, 0) + 1

    if not dry_run and noise:
        ids = [eid for eid, _, _ in noise]
        placeholders = ",".join("?" for _ in ids)
        with db.transaction() as cur:
            cur.execute(
                f"DELETE FROM entities WHERE id IN ({placeholders})",
                ids,
            )

    return {
        "dry_run": dry_run,
        "min_memories": min_memories,
        "deleted": 0 if dry_run else len(noise),
        "would_delete": len(noise),
        "by_reason": by_reason,
    }


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clean noise entities from the Remnant entity graph."
    )
    parser.add_argument(
        "--db", type=Path, default=None,
        help="Path to the Remnant SQLite DB (default: REMNANT_DB_HOME or ~/.hermes/remnant/remnant.db)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be deleted without making changes.",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Perform the deletion. Required when not using --dry-run.",
    )
    parser.add_argument(
        "--min-memories", type=int, default=2,
        help="Minimum number of active-memory links an entity must have to survive (default: 2).",
    )
    parser.add_argument(
        "--extra-stoplist", type=str, default="",
        help="Comma-separated list of additional names/tokens to treat as noise.",
    )
    args = parser.parse_args(argv)

    _configure_logging()

    if not args.dry_run and not args.yes:
        parser.error("--yes is required for live deletion; use --dry-run to preview")

    db_path = args.db or default_db_path()
    db = open_db(db_path)
    try:
        extra = {x.strip() for x in args.extra_stoplist.split(",") if x.strip()}
        stats = cleanup_entities(
            db,
            dry_run=args.dry_run,
            stoplist=extra,
            min_memories=args.min_memories,
        )
        action = "would delete" if args.dry_run else "deleted"
        log.info(
            "entity cleanup %s: %d entities %s (by reason: %s)",
            "dry-run" if args.dry_run else "live",
            stats["would_delete"],
            action,
            stats["by_reason"],
        )
        if stats["would_delete"] and args.dry_run:
            log.info("re-run with --yes to apply")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
