# ruff: noqa: E501
"""Re-extract entities and rebuild relations for all active memories.

Issue #21/#22 follow-up: existing memories were extracted with the old
over-extracting code (avg 39.4 entities/memory, max 524). This script
re-runs the new capped extractor (max 15, with common-noun filtering)
and rebuilds the entity graph from scratch.

Usage:
    python -m remnant.reextract --dry-run   # preview
    python -m remnant.reextract             # execute
    python -m remnant.reextract --batch 50  # batch size for progress
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

from remnant.db import RemnantDB
from remnant.entity import extract_high_signal_entities, seed_relations


def reextract(
    db_path: str,
    *,
    dry_run: bool = False,
    batch: int = 50,
) -> dict:
    """Re-extract entities and rebuild relations for all active memories.

    Returns a summary dict with counts.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    c = conn.cursor()

    # Get all active memories with content
    c.execute("SELECT id, content, agent FROM memories WHERE status='active' AND content IS NOT NULL AND content != ''")
    memories = c.fetchall()
    total = len(memories)

    if dry_run:
        # Sample: show what extraction would produce for first 5
        print(f"Dry run: {total} active memories to re-extract")
        sample = memories[:5]
        for mid, content, agent in sample:
            entities = extract_high_signal_entities(content, max_entities=15)
        # Current state
        c.execute("SELECT COUNT(*) FROM memory_entities")
        me_before = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM relations")
        rel_before = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM entities")
        ent_before = c.fetchone()[0]
        print(f"\nCurrent: {ent_before} entities, {me_before} memory_entity links, {rel_before} relations")
        conn.close()
        return {"dry_run": True, "memories": total, "entities_before": ent_before, "links_before": me_before, "relations_before": rel_before}

    # Step 1: Clear existing memory_entities and relations
    print("Step 1: Clearing existing entity links and relations...")
    c.execute("DELETE FROM memory_entities")
    me_deleted = c.rowcount
    c.execute("DELETE FROM relations")
    rel_deleted = c.rowcount
    conn.commit()
    print(f"  Deleted {me_deleted} memory_entity links, {rel_deleted} relations")

    # Step 2: Re-extract and link entities for each memory
    print(f"Step 2: Re-extracting entities for {total} memories...")
    db = RemnantDB(db_path)
    t0 = time.perf_counter()
    total_entities_linked = 0
    total_relations_seeded = 0

    for i, (mid, content, agent) in enumerate(memories):
        entities = extract_high_signal_entities(content, max_entities=15)
        entity_ids = []
        for ent in entities:
            name = ent["name"]
            etype = ent.get("type")
            try:
                # Use agent_id=None so entities are global (avoids
                # per-agent duplicate entities for the same name).
                eid = db.resolve_entity(name, None, entity_type=etype, aliases=ent.get("aliases") or [])
                if eid:
                    db.link_entity(memory_id=mid, entity_id=eid, agent_id=None)
                    entity_ids.append(eid)
                    total_entities_linked += 1
            except Exception as e:
                print(f"  WARN: failed to link entity '{name}' for memory {mid[:8]}: {e}", file=sys.stderr)

        # Seed relations using co-occurrence from text
        if entity_ids:
            try:
                seed_relations(db, memory_id=mid, entity_ids=entity_ids, text=content, max_entities=15)
                total_relations_seeded += 1
            except Exception as e:
                print(f"  WARN: failed to seed relations for memory {mid[:8]}: {e}", file=sys.stderr)

        if (i + 1) % batch == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{total}] {rate:.1f} memories/s, {total_entities_linked} entities linked, {total_relations_seeded} relations seeded")

    elapsed = time.perf_counter() - t0
    print(f"  Done: {total} memories in {elapsed:.1f}s, {total_entities_linked} entities linked, {total_relations_seeded} relations seeded")

    # Step 3: Clean up orphaned entities (no memory links)
    print("Step 3: Cleaning up orphaned entities...")
    c.execute("""DELETE FROM entities WHERE id NOT IN (
        SELECT DISTINCT entity_id FROM memory_entities
    )""")
    orphans = c.rowcount
    conn.commit()
    print(f"  Deleted {orphans} orphaned entities")

    # Step 4: VACUUM
    print("Step 4: VACUUM...")
    conn.execute("VACUUM")
    conn.commit()

    # Final state
    c.execute("SELECT COUNT(*) FROM entities")
    ent_after = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM memory_entities")
    me_after = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM relations")
    rel_after = c.fetchone()[0]

    # Entity distribution
    c.execute("""SELECT e.name, COUNT(me.memory_id) as links
        FROM entities e JOIN memory_entities me ON e.id = me.entity_id
        GROUP BY e.id ORDER BY links DESC LIMIT 10""")
    top_entities = c.fetchall()

    import os
    db_size = os.path.getsize(db_path) / 1024 / 1024

    print("\n=== Results ===")
    print(f"Entities: {ent_after}")
    print(f"Memory-entity links: {me_after}")
    print(f"Relations: {rel_after}")
    print(f"Orphaned entities deleted: {orphans}")
    print(f"DB size: {db_size:.1f} MB")
    print("\nTop 10 entities:")
    for name, links in top_entities:
        print(f"  {name}: {links}")

    conn.close()
    return {
        "memories_processed": total,
        "entities_linked": total_entities_linked,
        "relations_seeded": total_relations_seeded,
        "orphans_deleted": orphans,
        "entities_after": ent_after,
        "links_after": me_after,
        "relations_after": rel_after,
        "db_size_mb": db_size,
    }


def main():
    parser = argparse.ArgumentParser(description="Re-extract entities and rebuild relations")
    parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    parser.add_argument("--batch", type=int, default=50, help="Progress batch size")
    parser.add_argument("--db", default=None, help="Database path (default: from config)")
    args = parser.parse_args()

    if args.db:
        db_path = args.db
    else:
        from remnant.config import load_config
        load_config("/home/jd/.hermes")
        # Try to find the DB path from config or default location
        db_path = str(Path("/home/jd/.hermes/remnant/remnant.db"))

    result = reextract(db_path, dry_run=args.dry_run, batch=args.batch)
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
