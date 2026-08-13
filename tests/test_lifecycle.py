from __future__ import annotations

from pathlib import Path

import pytest

from remnant.config import RemnantConfig
from remnant.db import open_db
from remnant.lifecycle import MemoryLifecycle, backfill_relation_evidence


class _Embedder:
    _model = "test"

    @staticmethod
    def embed(_text: str) -> list[float]:
        return [1.0, 0.0]


def test_replacement_rolls_back_every_projection_when_audit_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    db = open_db(tmp_path / "rollback.db")
    try:
        old = db.insert_memory(content="old", agent="a", embedding=[0.0, 1.0])
        entity = db.resolve_entity("Kris", "a")
        db.link_entity(memory_id=old, entity_id=entity, agent_id="a")

        def fail(*_args, **_kwargs):
            raise RuntimeError("fault after projections")

        monkeypatch.setattr(db, "_write_audit", fail)
        with pytest.raises(RuntimeError, match="fault"):
            MemoryLifecycle(db, RemnantConfig(agent_id="a"), _Embedder()).replace(
                original_ids=[old], content="new", actor="test", agent_id="a"
            )
        assert db.get_memory(old)["status"] == "active"
        with db.read() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM memories")
            assert cur.fetchone()["n"] == 1
            cur.execute("SELECT COUNT(*) AS n FROM embeddings")
            assert cur.fetchone()["n"] == 1
            cur.execute("SELECT COUNT(*) AS n FROM memory_entities")
            assert cur.fetchone()["n"] == 1
    finally:
        db.close()


def test_forget_deactivates_claim_and_relation_evidence_atomically(tmp_path: Path):
    db = open_db(tmp_path / "forget.db")
    try:
        memory_id = db.insert_memory(content="Kris uses Remnant", agent="a")
        claim_id = db.create_claim(
            memory_id=memory_id, subject="Kris", predicate="uses", object="Remnant"
        )
        first = db.resolve_entity("Kris", "a")
        second = db.resolve_entity("Remnant", "a")
        db.add_relation(entity_a=first, entity_b=second, source_memory_id=memory_id)
        MemoryLifecycle(db, RemnantConfig(agent_id="a"), None).forget(
            memory_id, actor="test", agent_id="a"
        )
        assert db.get_memory(memory_id)["status"] == "forgotten"
        assert db.get_claim_for_memory(memory_id)["resolution_status"] == "historical"
        with db.read() as cur:
            cur.execute(
                "SELECT active, claim_id FROM relation_evidence WHERE memory_id=?",
                (memory_id,),
            )
            evidence = cur.fetchone()
        assert evidence["active"] == 0
        assert evidence["claim_id"] == claim_id
    finally:
        db.close()


def test_replacement_creates_claim_and_transfers_relation_evidence(tmp_path: Path):
    db = open_db(tmp_path / "replace.db")
    try:
        old = db.insert_memory(content="Kris uses Remnant", agent="a")
        old_claim = db.create_claim(
            memory_id=old, subject="Kris", predicate="uses", object="Remnant"
        )
        first = db.resolve_entity("Kris", "a")
        second = db.resolve_entity("Remnant", "a")
        db.add_relation(entity_a=first, entity_b=second, source_memory_id=old)
        result = MemoryLifecycle(db, RemnantConfig(agent_id="a"), _Embedder()).replace(
            original_ids=[old],
            content="Kris uses Remnant daily",
            actor="test",
            agent_id="a",
        )
        replacement = db.get_claim_for_memory(result["memory_id"])
        assert replacement["id"] == result["claim_id"]
        assert replacement["status"] == "active"
        assert db.get_claim_for_memory(old)["status"] == "superseded"
        with db.read() as cur:
            cur.execute(
                "SELECT memory_id, claim_id, active FROM relation_evidence "
                "ORDER BY memory_id",
            )
            evidence = [dict(row) for row in cur.fetchall()]
        assert {row["memory_id"] for row in evidence} == {old, result["memory_id"]}
        assert next(row for row in evidence if row["memory_id"] == old)["active"] == 0
        new_evidence = next(
            row for row in evidence if row["memory_id"] == result["memory_id"]
        )
        assert new_evidence["active"] == 1
        assert new_evidence["claim_id"] != old_claim
    finally:
        db.close()


def test_relation_evidence_backfill_is_dry_run_and_idempotent(tmp_path: Path):
    db = open_db(tmp_path / "backfill.db")
    try:
        memory_id = db.insert_memory(content="Kris uses Remnant", agent="a")
        first = db.resolve_entity("Kris", "a")
        second = db.resolve_entity("Remnant", "a")
        # Simulate a legacy relation by inserting directly without evidence.
        with db.transaction() as cur:
            cur.execute(
                "INSERT INTO relations(entity_a, entity_b, relation_type, strength, "
                "source_memory_id, created_at) VALUES(?,?,?,?,?,?)",
                (min(first, second), max(first, second), "uses", 0.8, memory_id, "2026-01-01"),
            )
        assert backfill_relation_evidence(db, dry_run=True)["written"] == 0
        assert backfill_relation_evidence(db, dry_run=False)["written"] == 1
        assert backfill_relation_evidence(db, dry_run=False)["written"] == 0
    finally:
        db.close()
