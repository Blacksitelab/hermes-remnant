"""Tests for orphan cleanup (issue #23)."""

from __future__ import annotations

import pytest

from remnant.cleanup_orphans import cleanup_orphans
from remnant.db import RemnantDB


@pytest.fixture
def db(tmp_path):
    return RemnantDB(tmp_path / "test.db")


def test_cleanup_orphans_dry_run_finds_orphans(db):
    # Insert a normal active memory
    mid1 = db.insert_memory(content="Active memory", agent="test")
    # Insert a forgotten orphan (no source_id, no vault_files)
    mid2 = db.insert_memory(content="Orphaned memory", agent="test")
    db.deactivate_memory(mid2)

    result = cleanup_orphans(db, dry_run=True)
    assert result["found"] >= 1
    assert result["deleted"] == 0
    assert result["dry_run"] is True


def test_cleanup_orphans_deletes(db):
    mid = db.insert_memory(content="Orphan to delete", agent="test")
    db.deactivate_memory(mid)

    result = cleanup_orphans(db, dry_run=False)
    assert result["deleted"] >= 1
    # Memory should be deactivated/gone
    mem = db.get_memory(mid)
    assert mem is None or mem.get("status") != "active"


def test_cleanup_orphans_preserves_active(db):
    mid = db.insert_memory(content="Active memory", agent="test")

    result = cleanup_orphans(db, dry_run=False)
    assert result["found"] == 0
    assert db.get_memory(mid) is not None