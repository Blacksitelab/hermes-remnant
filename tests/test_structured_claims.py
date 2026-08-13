from __future__ import annotations

from pathlib import Path

from remnant.claims import record_claim_from_memory
from remnant.db import open_db
from remnant.extract import _parse_facts
from remnant.reextract_claims import backfill_claims


def test_structured_parser_preserves_fields_and_nulls_invalid_timestamp():
    facts = _parse_facts(
        '{"facts":[{"fact":"Kris now prefers dark mode","subject":"Kris",'
        '"predicate":"prefers","object":"dark mode","durability":"durable",'
        '"modality":"asserted","valid_from":"not-a-date","conditions":["at home"]}]}'
    )
    assert facts[0]["predicate"] == "prefers"
    assert facts[0]["object"] == "dark mode"
    assert facts[0]["valid_from"] is None
    assert facts[0]["diagnostics"] == ["invalid_valid_from"]


def test_claim_uses_structured_spo_and_reconciliation_is_audited(tmp_path: Path):
    db = open_db(tmp_path / "claims.db")
    try:
        memory_id = db.insert_memory(content="Kris uses Signal", agent="a")
        claim_id = record_claim_from_memory(
            db,
            memory_id=memory_id,
            subject="Kris",
            fact="Kris uses Signal",
            claim_data={
                "predicate": "contact method",
                "object": "Signal",
                "confidence": 0.9,
                "observed_at": "2026-01-01T00:00:00Z",
            },
            reconciliation_enabled=True,
            agent_id="a",
        )
        claim = db.get_claim_for_memory(memory_id)
        assert claim["id"] == claim_id
        assert claim["predicate"] == "contact_method"
        assert claim["object"] == "Signal"
        assert db.list_audit(memory_id=memory_id)[0]["action"] == "claim_reconcile"
    finally:
        db.close()


def test_claim_competitors_never_cross_agent_scope(tmp_path: Path):
    db = open_db(tmp_path / "scope.db")
    try:
        first = db.insert_memory(content="Kris prefers dark", agent="agent-a")
        second = db.insert_memory(content="Kris prefers light", agent="agent-b")
        record_claim_from_memory(
            db, memory_id=first, subject="Kris", fact="Kris prefers dark", agent_id="agent-a"
        )
        record_claim_from_memory(
            db, memory_id=second, subject="Kris", fact="Kris prefers light", agent_id="agent-b"
        )
        assert db.get_claim_for_memory(first)["status"] == "active"
        assert db.get_claim_for_memory(second)["status"] == "active"
    finally:
        db.close()


def test_claim_backfill_is_dry_run_restartable_and_non_mutating(tmp_path: Path):
    db = open_db(tmp_path / "backfill.db")
    try:
        memory_id = db.insert_memory(
            content="Kris owns a printer",
            agent="a",
            tags=["Kris"],
            metadata={"entity": "Kris"},
        )
        before = db.get_memory(memory_id)
        dry = backfill_claims(db, dry_run=True)
        assert dry["eligible"] == 1 and dry["written"] == 0
        assert db.get_claim_for_memory(memory_id) is None
        applied = backfill_claims(db, dry_run=False)
        repeated = backfill_claims(db, dry_run=False)
        assert applied["written"] == 1
        assert repeated["written"] == 0
        assert db.get_memory(memory_id)["content"] == before["content"]
    finally:
        db.close()
