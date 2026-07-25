from __future__ import annotations

import pytest

from remnant.db import default_db_path, open_db
from remnant.maintenance import health_report, migrate_legacy_default_agent


def test_health_report_is_read_only_and_reports_core_counts():
    db = open_db(default_db_path())
    try:
        mid = db.insert_memory(content="health fact", agent="default")
        report = health_report(db)
        assert report["integrity"] == "ok"
        assert report["memory_rows"] == 1
        assert report["fts_rows"] == 1
        assert report["embeddings"] == 0
        assert db.get_memory(mid)["agent"] == "default"
    finally:
        db.close()


def test_legacy_agent_migration_is_dry_run_first_and_audited():
    db = open_db(default_db_path())
    try:
        mid = db.insert_memory(content="legacy fact", agent="default")
        preview = migrate_legacy_default_agent(db, target_agent="claire", dry_run=True)
        assert preview["would_migrate"] == 1
        assert db.get_memory(mid)["agent"] == "default"

        applied = migrate_legacy_default_agent(db, target_agent="claire", dry_run=False)
        assert applied["migrated"] == 1
        assert db.get_memory(mid)["agent"] == "claire"
        assert db.list_audit(memory_id=mid, action="migrate_legacy_agent")
    finally:
        db.close()


def test_legacy_agent_migration_requires_explicit_owner():
    db = open_db(default_db_path())
    try:
        with pytest.raises(ValueError):
            migrate_legacy_default_agent(db, target_agent="default")
    finally:
        db.close()
