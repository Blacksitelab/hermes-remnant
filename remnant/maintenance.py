"""Safe operational maintenance for a shared Remnant database.

The commands here are deliberately explicit and dry-run first.  They are for
operators, not agent tools: agents must not be able to retag ownership or run
database maintenance through the model-facing memory surface.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Sequence
from typing import Any

from .db import SCHEMA_VERSION, RemnantDB, default_db_path, open_db
from .identity import effective_identity


def health_report(db: RemnantDB) -> dict[str, Any]:
    """Return bounded health signals needed to operate the memory service."""
    with db.read() as cur:
        cur.execute("SELECT status, COUNT(*) AS count FROM memories GROUP BY status")
        memories = {row["status"]: int(row["count"]) for row in cur.fetchall()}
        cur.execute("SELECT COUNT(*) AS count FROM embeddings")
        embeddings = int(cur.fetchone()["count"])
        cur.execute("SELECT status, COUNT(*) AS count FROM extraction_queue GROUP BY status")
        extraction = {row["status"]: int(row["count"]) for row in cur.fetchall()}
        cur.execute("SELECT outcome, COUNT(*) AS count FROM prefetch_stats GROUP BY outcome")
        prefetch = {row["outcome"]: int(row["count"]) for row in cur.fetchall()}
        cur.execute("SELECT COUNT(*) AS count FROM memories_fts")
        fts_rows = int(cur.fetchone()["count"])
        cur.execute("SELECT COUNT(*) AS count FROM memories")
        memory_rows = int(cur.fetchone()["count"])
        cur.execute("SELECT COUNT(*) AS count FROM claims")
        claim_rows = int(cur.fetchone()["count"])
        cur.execute(
            "SELECT resolution_status, COUNT(*) AS count FROM claims "
            "GROUP BY resolution_status"
        )
        claims_by_resolution = {
            str(row["resolution_status"] or "active"): int(row["count"])
            for row in cur.fetchall()
        }
        cur.execute(
            "SELECT MIN(created_at) AS oldest FROM turns "
            "WHERE extraction_status IN ('pending','running','retry_wait')"
        )
        oldest_pending = cur.fetchone()["oldest"]
        cur.execute(
            "SELECT COUNT(*) AS count FROM turns "
            "WHERE extraction_status IN ('pending','running','retry_wait')"
        )
        pending_overlay_count = int(cur.fetchone()["count"])
        pending_age_s = (
            round(max(0.0, time.time() - float(oldest_pending)), 3)
            if oldest_pending is not None
            else 0.0
        )
        cur.execute("PRAGMA integrity_check")
        integrity = str(cur.fetchone()[0])
    return {
        "schema_version": SCHEMA_VERSION,
        "integrity": integrity,
        "memories_by_status": memories,
        "memory_rows": memory_rows,
        "fts_rows": fts_rows,
        "embedding_coverage": round(embeddings / memory_rows, 4) if memory_rows else 1.0,
        "embeddings": embeddings,
        "claims": claim_rows,
        "claim_coverage": round(claim_rows / memory_rows, 4) if memory_rows else 1.0,
        "claims_by_resolution": claims_by_resolution,
        "extraction_queue": extraction,
        "pending_overlay_count": pending_overlay_count,
        "pending_extraction_age_s": pending_age_s,
        "prefetch_outcomes": prefetch,
    }


def migrate_legacy_default_agent(
    db: RemnantDB,
    *,
    target_agent: str,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Retag legacy ``agent='default'`` rows with an explicit owner.

    This migration is intentionally opt-in: it cannot infer ownership safely.
    It also leaves provenance and visibility untouched, so operators can audit
    the exact target before applying it.
    """
    target = (target_agent or "").strip()
    if not target or target == "default":
        raise ValueError("target_agent must be a non-default explicit agent id")
    with db.read() as cur:
        cur.execute("SELECT id, status, source, visibility FROM memories WHERE agent='default'")
        rows = [dict(row) for row in cur.fetchall()]
    by_status = dict(Counter(str(row["status"]) for row in rows))
    if not dry_run and rows:
        db.migrate_memory_agent("default", target)
    return {
        "dry_run": dry_run,
        "target_agent": target,
        "would_migrate": len(rows),
        "migrated": 0 if dry_run else len(rows),
        "by_status": by_status,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect and maintain Remnant safely.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("health", help="Print database health as JSON.")
    migrate = subparsers.add_parser(
        "migrate-default-agent",
        help="Retag legacy default-owned memories.",
    )
    migrate.add_argument("--agent", required=True, help="Explicit owner to assign to legacy rows.")
    migrate.add_argument(
        "--dry-run", action="store_true", help="Preview only; this is the default."
    )
    migrate.add_argument("--yes", action="store_true", help="Apply the migration.")
    identity = subparsers.add_parser(
        "identity", help="Preview a privacy-preserving runtime identity mapping."
    )
    identity.add_argument("--configured-agent", default="default")
    identity.add_argument("--session", default="diagnostic")
    identity.add_argument("--platform", default="cli")
    identity.add_argument("--agent-identity", default="")
    identity.add_argument("--workspace", default="")
    identity.add_argument("--user-id", default="")
    args = parser.parse_args(argv)
    if args.command == "identity":
        report = effective_identity(
            configured_agent=args.configured_agent,
            session_id=args.session,
            runtime_identity_enabled=True,
            platform=args.platform,
            agent_identity=args.agent_identity,
            agent_workspace=args.workspace,
            user_id=args.user_id,
        ).diagnostic()
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    db = open_db(default_db_path())
    try:
        if args.command == "health":
            print(json.dumps(health_report(db), indent=2, sort_keys=True))
            return 0
        if not args.yes:
            result = migrate_legacy_default_agent(db, target_agent=args.agent, dry_run=True)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        result = migrate_legacy_default_agent(db, target_agent=args.agent, dry_run=False)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    finally:
        db.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
