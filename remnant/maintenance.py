"""Safe operational maintenance for a shared Remnant database.

The commands here are deliberately explicit and dry-run first.  They are for
operators, not agent tools: agents must not be able to retag ownership or run
database maintenance through the model-facing memory surface.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sqlite3
import time
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import RemnantConfig
from .db import SCHEMA_VERSION, RemnantDB, default_db_path, open_db
from .identity import effective_identity
from .lifecycle import backfill_relation_evidence


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return round(ordered[index], 3)


def availability_report(
    config: RemnantConfig | None = None, *, db_path: Path | None = None
) -> dict[str, Any]:
    """Inspect local prerequisites without making any network request."""
    config = config or RemnantConfig()
    path = db_path or default_db_path()
    unavailable: list[str] = []
    degraded: list[str] = []
    parent = path.parent
    writable_parent = parent if parent.exists() else parent.parent
    if not writable_parent.exists() or not os.access(writable_parent, os.W_OK):
        unavailable.append("database parent is not writable")
    if path.exists():
        try:
            connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
            row = connection.execute(
                "SELECT value FROM schema_meta WHERE key='version'"
            ).fetchone()
            connection.close()
            version = int(row[0]) if row else 0
            if version > SCHEMA_VERSION:
                unavailable.append(f"database schema {version} is newer than {SCHEMA_VERSION}")
            elif version < SCHEMA_VERSION:
                degraded.append(f"database schema {version} requires migration")
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            unavailable.append(f"database cannot be inspected: {type(exc).__name__}")
    for label, value, enabled in (
        ("embedding", config.embed_url, True),
        ("extraction", config.extract_url, config.extract_enabled),
    ):
        parsed = urlparse(str(value or ""))
        if enabled and (parsed.scheme not in {"http", "https"} or not parsed.netloc):
            degraded.append(f"{label} endpoint is not a valid HTTP URL")
    if importlib.util.find_spec("gliner") is None:
        degraded.append("GLiNER is not installed; regex entity extraction remains available")
    status = "unavailable" if unavailable else ("degraded" if degraded else "available")
    return {
        "available": not unavailable,
        "status": status,
        "unavailable_reasons": unavailable,
        "degraded_reasons": degraded,
        "optional_dependencies": {"gliner": importlib.util.find_spec("gliner") is not None},
    }


def health_report(db: RemnantDB) -> dict[str, Any]:
    """Return bounded health signals needed to operate the memory service."""
    with db.read() as cur:
        cur.execute("SELECT status, COUNT(*) AS count FROM memories GROUP BY status")
        memories = {row["status"]: int(row["count"]) for row in cur.fetchall()}
        cur.execute(
            "SELECT COALESCE(model, 'unknown') AS model, dimensions, COUNT(*) AS count "
            "FROM embeddings GROUP BY COALESCE(model, 'unknown'), dimensions"
        )
        embeddings_by_model = [dict(row) for row in cur.fetchall()]
        embeddings = sum(int(row["count"]) for row in embeddings_by_model)
        cur.execute("SELECT status, COUNT(*) AS count FROM extraction_queue GROUP BY status")
        extraction = {row["status"]: int(row["count"]) for row in cur.fetchall()}
        cur.execute("SELECT outcome, COUNT(*) AS count FROM prefetch_stats GROUP BY outcome")
        prefetch = {row["outcome"]: int(row["count"]) for row in cur.fetchall()}
        cur.execute(
            "SELECT COALESCE(reason, 'none') AS reason, COUNT(*) AS count "
            "FROM prefetch_stats GROUP BY COALESCE(reason, 'none')"
        )
        prefetch_reasons = {str(row["reason"]): int(row["count"]) for row in cur.fetchall()}
        cur.execute(
            "SELECT elapsed_ms FROM prefetch_stats WHERE elapsed_ms IS NOT NULL "
            "ORDER BY id DESC LIMIT 1000"
        )
        prefetch_latencies = [float(row["elapsed_ms"]) for row in cur.fetchall()]
        cur.execute(
            "SELECT operation, outcome, COUNT(*) AS count, SUM(input_units) AS input_units, "
            "SUM(output_units) AS output_units, AVG(elapsed_ms) AS avg_elapsed_ms "
            "FROM operation_metrics GROUP BY operation, outcome"
        )
        operation_metrics = [dict(row) for row in cur.fetchall()]
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
            "SELECT COALESCE(extractor_version, 'legacy') AS version, COUNT(*) AS count "
            "FROM claims GROUP BY COALESCE(extractor_version, 'legacy')"
        )
        claims_by_version = {
            str(row["version"]): int(row["count"]) for row in cur.fetchall()
        }
        cur.execute(
            "SELECT COUNT(*) AS count, MIN(created_at) AS oldest FROM claims "
            "WHERE resolution_status IN ('unresolved','contradicted')"
        )
        unresolved = dict(cur.fetchone())
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
        cur.execute(
            "SELECT extraction_status, COUNT(*) AS count FROM turns "
            "GROUP BY extraction_status"
        )
        turn_extraction = {
            str(row["extraction_status"]): int(row["count"]) for row in cur.fetchall()
        }
        cur.execute(
            "SELECT COALESCE(MAX(id), 0) AS watermark FROM turns "
            "WHERE extraction_status='completed'"
        )
        extraction_watermark = int(cur.fetchone()["watermark"])
        cur.execute("SELECT COUNT(*) AS count FROM entities")
        entities = int(cur.fetchone()["count"])
        cur.execute("SELECT COUNT(*) AS count FROM relations")
        relations = int(cur.fetchone()["count"])
        cur.execute("SELECT COUNT(*) AS count FROM relation_evidence WHERE active=1")
        active_relation_evidence = int(cur.fetchone()["count"])
        cur.execute("SELECT MAX(indexed_at) AS last_scan FROM vault_files")
        last_vault_scan = cur.fetchone()["last_scan"]
        cur.execute(
            "SELECT key, value FROM dream_state WHERE key IN ('day_run_ts','night_run_ts')"
        )
        dream_runs = {str(row["key"]): json.loads(row["value"]) for row in cur.fetchall()}
        cur.execute("PRAGMA integrity_check")
        integrity = str(cur.fetchone()[0])
    unresolved_oldest = unresolved.get("oldest")
    try:
        unresolved_age_s = max(
            0.0,
            time.time()
            - datetime.fromisoformat(str(unresolved_oldest).replace("Z", "+00:00")).timestamp(),
        )
    except (TypeError, ValueError):
        unresolved_age_s = 0.0
    availability = availability_report(db_path=Path(db.path))
    total_prefetch = sum(prefetch.values())
    return {
        "availability": availability,
        "schema_version": SCHEMA_VERSION,
        "integrity": integrity,
        "memories_by_status": memories,
        "memory_rows": memory_rows,
        "fts_rows": fts_rows,
        "embedding_coverage": round(embeddings / memory_rows, 4) if memory_rows else 1.0,
        "embeddings": embeddings,
        "embeddings_by_model_dimension": embeddings_by_model,
        "claims": claim_rows,
        "claim_coverage": round(claim_rows / memory_rows, 4) if memory_rows else 1.0,
        "claims_by_resolution": claims_by_resolution,
        "claims_by_extractor_version": claims_by_version,
        "unresolved_conflicts": int(unresolved.get("count") or 0),
        "oldest_unresolved_age_s": round(unresolved_age_s, 3),
        "extraction_queue": extraction,
        "turn_extraction": turn_extraction,
        "extraction_watermark_turn_id": extraction_watermark,
        "pending_overlay_count": pending_overlay_count,
        "pending_extraction_age_s": pending_age_s,
        "prefetch_outcomes": prefetch,
        "prefetch_reasons": prefetch_reasons,
        "prefetch_injection_rate": round(prefetch.get("injected", 0) / total_prefetch, 4)
        if total_prefetch
        else 0.0,
        "prefetch_latency_ms": {
            "p50": _percentile(prefetch_latencies, 0.50),
            "p95": _percentile(prefetch_latencies, 0.95),
            "sample": len(prefetch_latencies),
        },
        "operation_metrics": operation_metrics,
        "entities": entities,
        "relations": relations,
        "active_relation_evidence": active_relation_evidence,
        "last_vault_scan": last_vault_scan,
        "last_dream_runs": dream_runs,
    }


def backup_database(db: RemnantDB, output: Path) -> dict[str, Any]:
    """Create a verified SQLite backup without overwriting an existing file."""
    target = output.resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    destination = sqlite3.connect(target)
    try:
        with db._lock:
            db._conn.backup(destination)
        integrity = str(destination.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise RuntimeError(f"backup integrity check failed: {integrity}")
    finally:
        destination.close()
    return {"backup": str(target), "schema_version": SCHEMA_VERSION, "integrity": integrity}


def restore_database(backup: Path, output: Path) -> dict[str, Any]:
    """Restore a verified backup to a new path; never overwrite live storage."""
    source_path = backup.resolve()
    target_path = output.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if target_path.exists():
        raise FileExistsError(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(f"file:{source_path.as_posix()}?mode=ro", uri=True)
    destination = sqlite3.connect(target_path)
    try:
        source_integrity = str(source.execute("PRAGMA integrity_check").fetchone()[0])
        if source_integrity != "ok":
            raise RuntimeError(f"source integrity check failed: {source_integrity}")
        source.backup(destination)
        restored_integrity = str(destination.execute("PRAGMA integrity_check").fetchone()[0])
        if restored_integrity != "ok":
            raise RuntimeError(f"restore integrity check failed: {restored_integrity}")
    finally:
        source.close()
        destination.close()
    return {"restored": str(target_path), "integrity": restored_integrity}


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
    backup = subparsers.add_parser("backup", help="Create a verified SQLite backup.")
    backup.add_argument("--output", type=Path, required=True)
    restore = subparsers.add_parser("restore", help="Restore a backup to a new path.")
    restore.add_argument("--backup", type=Path, required=True)
    restore.add_argument("--output", type=Path, required=True)
    evidence = subparsers.add_parser(
        "backfill-relation-evidence", help="Backfill relation provenance safely."
    )
    evidence.add_argument("--limit", type=int, default=1000)
    evidence.add_argument("--yes", action="store_true")
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
    if args.command == "restore":
        print(
            json.dumps(
                restore_database(args.backup, args.output), indent=2, sort_keys=True
            )
        )
        return 0
    db = open_db(default_db_path())
    try:
        if args.command == "health":
            print(json.dumps(health_report(db), indent=2, sort_keys=True))
            return 0
        if args.command == "backup":
            print(json.dumps(backup_database(db, args.output), indent=2, sort_keys=True))
            return 0
        if args.command == "backfill-relation-evidence":
            report = backfill_relation_evidence(
                db, dry_run=not args.yes, limit=args.limit
            )
            print(json.dumps(report, indent=2, sort_keys=True))
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
