from __future__ import annotations

from pathlib import Path

import pytest

from remnant.config import RemnantConfig
from remnant.db import open_db
from remnant.maintenance import (
    availability_report,
    backup_database,
    health_report,
    restore_database,
)


def test_health_reports_operational_dimensions(tmp_path: Path):
    db = open_db(tmp_path / "health.db")
    try:
        memory_id = db.insert_memory(
            content="Kris uses Remnant", agent="a", embedding=[1.0, 0.0], embed_model="e1"
        )
        db.create_claim(
            memory_id=memory_id,
            subject="Kris",
            predicate="uses",
            object="Remnant",
            resolution_status="unresolved",
            extractor_version="claims-v2",
        )
        first = db.resolve_entity("Kris", "a")
        second = db.resolve_entity("Remnant", "a")
        db.add_relation(entity_a=first, entity_b=second, source_memory_id=memory_id)
        db.record_prefetch("s", "injected", elapsed_ms=12.0, result_count=1)
        db.record_operation(
            "embedding", "cache_hit", elapsed_ms=1.0, input_units=12, output_units=2
        )
        report = health_report(db)
        assert report["integrity"] == "ok"
        assert report["embeddings_by_model_dimension"][0]["model"] == "e1"
        assert report["claims_by_extractor_version"]["claims-v2"] == 1
        assert report["unresolved_conflicts"] == 1
        assert report["active_relation_evidence"] == 1
        assert report["prefetch_latency_ms"]["p95"] == 12.0
        assert report["operation_metrics"][0]["operation"] == "embedding"
    finally:
        db.close()


def test_backup_restore_round_trip_never_overwrites(tmp_path: Path):
    source = open_db(tmp_path / "source.db")
    backup = tmp_path / "backup.db"
    restored = tmp_path / "restored.db"
    try:
        memory_id = source.insert_memory(content="release evidence", agent="a")
        assert backup_database(source, backup)["integrity"] == "ok"
        with pytest.raises(FileExistsError):
            backup_database(source, backup)
    finally:
        source.close()
    assert restore_database(backup, restored)["integrity"] == "ok"
    db = open_db(restored)
    try:
        assert db.get_memory(memory_id)["content"] == "release evidence"
    finally:
        db.close()
    with pytest.raises(FileExistsError):
        restore_database(backup, restored)


def test_availability_distinguishes_degraded_and_unavailable(tmp_path: Path):
    degraded = availability_report(
        RemnantConfig(embed_url="not-a-url", extract_enabled=False),
        db_path=tmp_path / "new.db",
    )
    assert degraded["available"] is True
    assert degraded["status"] == "degraded"
    unavailable = availability_report(db_path=tmp_path / "missing" / "nested" / "db.sqlite")
    assert unavailable["available"] is False
    assert unavailable["status"] == "unavailable"
