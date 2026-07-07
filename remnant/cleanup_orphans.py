"""CLI tool: purge orphaned forgotten memories (issue #23).

Orphans are forgotten memories with no source_id and no vault_files entry.
They were created during broken vault reindexes and are safe to delete.

Usage:
    python -m remnant.cleanup_orphans [--dry-run] [--yes]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .db import RemnantDB, default_db_path


def cleanup_orphans(db: RemnantDB, *, dry_run: bool = True) -> dict:
    """Find and delete orphaned forgotten memories.

    Returns ``{"found": int, "deleted": int, "dry_run": bool}``.
    """
    orphan_ids = db.find_orphan_forgotten_memory_ids()
    result = {"found": len(orphan_ids), "deleted": 0, "dry_run": dry_run}
    if dry_run or not orphan_ids:
        return result

    for mid in orphan_ids:
        db.hard_delete_memory(mid)
        result["deleted"] += 1
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Purge orphaned forgotten memories")
    parser.add_argument("--dry-run", action="store_true", help="Report only, don't delete")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    db = RemnantDB(default_db_path())

    result = cleanup_orphans(db, dry_run=args.dry_run)
    print(f"Found {result['found']} orphaned forgotten memories.")
    if args.dry_run:
        print("Dry run — no changes made. Run with --yes to delete.")
        return 0

    if not args.yes:
        confirm = input(f"Delete {result['found']} orphans? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return 1

    # Re-run with deletion
    result = cleanup_orphans(db, dry_run=False)
    print(f"Deleted {result['deleted']} orphaned memories.")
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())