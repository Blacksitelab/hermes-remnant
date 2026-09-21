from __future__ import annotations

from pathlib import Path

import pytest

from remnant.config import RemnantConfig
from remnant.db import open_db
from remnant.ingest import store_outcome
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
            content="Alex uses Remnant", agent="a", embedding=[1.0, 0.0], embed_model="e1"
        )
        db.create_claim(
            memory_id=memory_id,
            subject="Alex",
            predicate="uses",
            object="Remnant",
            resolution_status="unresolved",
            extractor_version="claims-v2",
        )
        first = db.resolve_entity("Alex", "a")
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
        assert report["semantic_scan"]["configured_limit"] == 0
        assert report["semantic_scan"]["ann_recommended"] is False
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


def test_store_outcome_rejects_nested_and_whitespace_secrets(tmp_path: Path):
    db = open_db(tmp_path / "outcomes.db")
    config = RemnantConfig(agent_id="owner")
    try:
        cases = [
            ("password hunter2", None),
            ("authorization Bearer abc", None),
            ("safe", {"nested": {"token": "abc"}}),
            ("safe", [{"authorization": "Bearer abc"}]),
            ("safe", {"nested": {"password": "abc"}}),
            ("safe", {"nested": [{"token": "abc"}]}),
            ("safe", {"nested": {"secret": "abc"}}),
            ("safe", {"nested": {"bearer": "abc"}}),
            ("safe", {"nested": {"authorization": "abc"}}),
        ]
        for fact, metadata in cases:
            with pytest.raises(ValueError, match="^outcome rejected$"):
                store_outcome(
                    db,
                    config,
                    operation_id="op-" + str(cases.index((fact, metadata))),
                    fact=fact,
                    metadata=metadata,
                )
        assert db.get_memory_operation(agent="owner", operation_id="op-0") is None
    finally:
        db.close()


def test_operation_insert_keeps_creation_audit_and_receipt(tmp_path: Path):
    db = open_db(tmp_path / "operation-audit.db")
    try:
        memory_id = db.insert_memory(
            content="durable outcome", agent="owner", operation_id="op-audit"
        )
        receipt = db.get_memory_operation(agent="owner", operation_id="op-audit")
        audit = db.list_audit()
        assert receipt is not None
        assert receipt["memory_id"] == memory_id
        assert receipt["audit_id"] == audit[0]["id"]
        assert audit[0]["action"] == "create"
    finally:
        db.close()


def _counts(db):
    return tuple(
        db._conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ("memories", "audit_log", "memory_operations")
    )


def _rows(db, table, where="", params=()):
    return [dict(row) for row in db._conn.execute(f"SELECT * FROM {table} {where}", params)]


def test_outcome_receipt_survives_hard_delete_and_replay_is_missing(tmp_path: Path):
    db = open_db(tmp_path / "missing.db")
    try:
        config = RemnantConfig(agent_id="owner")
        result = store_outcome(db, config, operation_id="op", fact="durable fact")
        receipt = db.get_memory_operation(agent="owner", operation_id="op")
        assert receipt is not None
        assert db.get_memory(result["memory_id"]) is not None
        assert db.hard_delete_memory(result["memory_id"])
        before = {table: _rows(db, table) for table in ("audit_log", "memory_operations")}
        counts = _counts(db)
        with pytest.raises(ValueError, match="^operation target missing$"):
            store_outcome(db, config, operation_id="op", fact="durable fact")
        assert _counts(db) == counts
        assert {table: _rows(db, table) for table in before} == before
        assert db.get_memory_operation(agent="owner", operation_id="op") == receipt
    finally:
        db.close()


def test_outcome_replay_conflict_and_malformed_embedding_are_durable(tmp_path: Path):
    db = open_db(tmp_path / "replay.db")
    config = RemnantConfig(agent_id="owner")
    try:
        first = store_outcome(db, config, operation_id="op", fact="same", embedding=[1.0])
        snapshots = {
            table: _rows(db, table) for table in ("memories", "audit_log", "memory_operations")
        }
        counts = _counts(db)
        assert (
            store_outcome(db, config, operation_id="op", fact="same", embedding=[1.0])["status"]
            == "already_stored"
        )
        assert _counts(db) == counts
        assert {table: _rows(db, table) for table in snapshots} == snapshots
        conflict_snapshots = {
            table: _rows(db, table) for table in ("memories", "audit_log", "memory_operations")
        }
        with pytest.raises(ValueError, match="^operation conflict$"):
            store_outcome(db, config, operation_id="op", fact="different")
        assert _counts(db) == counts
        assert {
            table: _rows(db, table) for table in conflict_snapshots
        } == conflict_snapshots
        malformed = store_outcome(
            db, config, operation_id="bad-embedding", fact="lexical", embedding=[float("nan")]
        )
        assert _counts(db) == (2, 2, 2)
        lexical = db.get_memory(malformed["memory_id"])
        assert lexical is not None
        assert lexical == _rows(db, "memories", "WHERE id=?", (malformed["memory_id"],))[0]
        assert lexical["content"] == "lexical"
        assert lexical["agent"] == "owner"
        assert lexical["type"] == "fact"
        assert lexical["source"] == "conversation"
        assert lexical["metadata"] is None
        malformed_receipt = db.get_memory_operation(agent="owner", operation_id="bad-embedding")
        assert malformed_receipt is not None
        assert malformed_receipt == _rows(
            db,
            "memory_operations",
            "WHERE agent=? AND operation_id=?",
            ("owner", "bad-embedding"),
        )[0]
        assert malformed_receipt["agent"] == "owner"
        assert malformed_receipt["operation_id"] == "bad-embedding"
        assert malformed_receipt["memory_id"] == malformed["memory_id"]
        audit = db.list_audit()
        creation = next(row for row in audit if row["memory_id"] == malformed["memory_id"])
        raw_creation = _rows(db, "audit_log", "WHERE id=?", (creation["id"],))[0]
        assert set(creation) == set(raw_creation)
        assert all(creation[key] == raw_creation[key] for key in set(creation) - {"details"})
        assert creation["actor"] == "owner"
        assert creation["action"] == "create"
        assert creation["memory_id"] == malformed["memory_id"]
        assert creation["details"] == {
            "source": "conversation",
            "type": "fact",
            "operation_id": "bad-embedding",
        }
        assert (
            db._conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE memory_id=?", (malformed["memory_id"],)
            ).fetchone()[0]
            == 0
        )
        assert first["memory_id"] != malformed["memory_id"]
    finally:
        db.close()


def test_outcome_failure_rolls_back_and_retry_stores_once(tmp_path: Path, monkeypatch):
    db = open_db(tmp_path / "rollback.db")
    config = RemnantConfig(agent_id="owner")
    try:
        monkeypatch.setattr(
            db, "_write_audit", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("injected"))
        )
        with pytest.raises(RuntimeError, match="injected"):
            store_outcome(db, config, operation_id="op", fact="retry me")
        assert _counts(db) == (0, 0, 0)
        monkeypatch.undo()
        assert store_outcome(db, config, operation_id="op", fact="retry me")["status"] == "stored"
        assert _counts(db) == (1, 1, 1)
    finally:
        db.close()


def test_memory_operations_migration_preserves_legacy_row(tmp_path: Path):
    path = tmp_path / "migrate.db"
    db = open_db(path)
    try:
        mid = db.insert_memory(content="legacy", agent="owner")
        db._conn.execute("DROP TABLE memory_operations")
        db._conn.execute(
            "CREATE TABLE memory_operations("
            "agent TEXT NOT NULL, operation_id TEXT NOT NULL, payload_hash TEXT NOT NULL, "
            "memory_id TEXT NOT NULL, audit_id INTEGER NOT NULL, "
            "PRIMARY KEY(agent, operation_id))"
        )
        db._conn.execute(
            "INSERT INTO memory_operations VALUES('owner','legacy','hash',?,0)", (mid,)
        )
    finally:
        db.close()
    reopened = open_db(path)
    try:
        assert (
            reopened.get_memory_operation(agent="owner", operation_id="legacy")["memory_id"] == mid
        )
        assert (
            store_outcome(
                reopened, RemnantConfig(agent_id="owner"), operation_id="new", fact="new"
            )["status"]
            == "stored"
        )
    finally:
        reopened.close()


def test_same_operation_id_isolated_by_owner_and_outcome_shape(tmp_path: Path):
    db = open_db(tmp_path / "owners.db")
    try:
        a = store_outcome(
            db, RemnantConfig(agent_id="a"), operation_id="same", fact="A", metadata={"k": "v"}
        )
        b = store_outcome(db, RemnantConfig(agent_id="b"), operation_id="same", fact="B")
        assert a["memory_id"] != b["memory_id"]
        row = db.get_memory(a["memory_id"])
        assert {row[k] for k in ("agent", "visibility", "type", "source")} == {
            "a",
            "private",
            "fact",
            "conversation",
        }
        assert row["metadata"] == {"k": "v"}
        for table in ("entities", "relations", "memory_entities", "relation_evidence", "claims"):
            assert _rows(db, table) == []
        assert _rows(db, "relations", "WHERE source_memory_id IS NOT NULL") == []
    finally:
        db.close()


_SECRET_FORMS = [
    (f"{keyword}-{separator or 'space'}", f"{keyword}{separator}{value}")
    for keyword in ("password", "token", "secret", "bearer", "authorization")
    for separator, value in ((" ", "hunter2"), (":", "abc"), ("=", "abc"))
] + [("sk-redaction", "sk-synthetic-outcome-token-1234567890"), ("url-credential", "https://u:p@example.com")]


@pytest.mark.parametrize("label,fact", _SECRET_FORMS, ids=lambda case: case[0])
def test_direct_outcome_secret_matrix_rejects_before_storage(tmp_path: Path, label: str, fact: str):
    db = open_db(tmp_path / "direct-secrets.db")
    try:
        with pytest.raises(ValueError, match="^outcome rejected$") as exc:
            store_outcome(db, RemnantConfig(agent_id="owner"), operation_id="op", fact=fact)
        assert "sentinel" not in str(exc.value).lower()
        assert _counts(db) == (0, 0, 0)
        assert db._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 0
    finally:
        db.close()


@pytest.mark.parametrize(
    "label,value",
    _SECRET_FORMS,
    ids=lambda case: case[0],
)
def test_outcome_secret_matrix_rejects_before_storage(tmp_path: Path, label: str, value: str):
    db = open_db(tmp_path / "secrets.db")
    try:
        with pytest.raises(ValueError, match="^outcome rejected$") as exc:
            store_outcome(
                db,
                RemnantConfig(agent_id="owner"),
                operation_id="op",
                fact="safe",
                metadata={"context": {"value": value}},
            )
        assert "sentinel" not in str(exc.value).lower()
        assert _counts(db) == (0, 0, 0)
        assert db._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 0
    finally:
        db.close()
