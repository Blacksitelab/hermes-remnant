"""Restartable, dry-run-first claim projection backfill."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .claims import record_claim_from_memory
from .db import RemnantDB, default_db_path, open_db

EXTRACTOR_VERSION = "claims-v2-backfill"


def _subject(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (TypeError, json.JSONDecodeError):
            metadata = {}
    if isinstance(metadata, dict) and metadata.get("entity"):
        return str(metadata["entity"])
    tags = row.get("tags")
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except (TypeError, json.JSONDecodeError):
            tags = []
    return str(tags[0]) if isinstance(tags, list) and tags else "general"


def backfill_claims(
    db: RemnantDB,
    *,
    batch: int = 100,
    dry_run: bool = True,
) -> dict[str, int | bool | str]:
    """Project only memories that do not already have the current extractor version."""
    with db.read() as cur:
        cur.execute(
            "SELECT m.* FROM memories m LEFT JOIN claims c ON c.memory_id=m.id "
            "AND c.extractor_version=? WHERE m.status='active' AND m.type='fact' "
            "AND c.id IS NULL ORDER BY m.created_at, m.id LIMIT ?",
            (EXTRACTOR_VERSION, max(1, int(batch))),
        )
        rows = [dict(row) for row in cur.fetchall()]
    eligible = [row for row in rows if _subject(row).casefold() != "general"]
    written = 0
    if not dry_run:
        for row in eligible:
            claim_id = record_claim_from_memory(
                db,
                memory_id=str(row["id"]),
                subject=_subject(row),
                fact=str(row["content"]),
                confidence=float(row.get("confidence") or 0.5),
                claim_data={
                    "observed_at": row.get("created_at") or row.get("timestamp"),
                    "extractor_version": EXTRACTOR_VERSION,
                },
                reconciliation_enabled=False,
                agent_id=row.get("agent"),
            )
            written += int(claim_id is not None)
    return {
        "dry_run": dry_run,
        "extractor_version": EXTRACTOR_VERSION,
        "scanned": len(rows),
        "eligible": len(eligible),
        "written": written,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill Remnant claims safely.")
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    db = open_db(args.db or default_db_path())
    try:
        report = backfill_claims(db, batch=args.batch, dry_run=not args.yes)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    finally:
        db.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["EXTRACTOR_VERSION", "backfill_claims"]
