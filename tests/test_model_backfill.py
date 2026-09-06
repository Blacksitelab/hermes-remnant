"""Tests for model-backed historical claim projection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from remnant.claims import record_claim_from_memory
from remnant.db import open_db
from remnant.llm import LLMResponseError
from remnant.model_backfill import (
    MODEL_BACKFILL_VERSION,
    _target_memories,
    apply_claim_projection,
    parse_claim_batch,
    parse_single_claim,
)


def test_parse_claim_batch_accepts_only_requested_memory_ids():
    text = json.dumps(
        {
            "claims": [
                {
                    "memory_id": "m-1",
                    "claim": {
                        "subject": "Kris",
                        "predicate": "prefers",
                        "object": "dark mode",
                        "confidence": 0.91,
                        "valid_from": None,
                        "valid_to": None,
                        "event_at": None,
                        "scope_type": None,
                        "scope_value": None,
                        "conditions": [],
                        "modality": "asserted",
                    },
                },
                {"memory_id": "m-2", "claim": None},
            ]
        }
    )

    parsed = parse_claim_batch(text, allowed_ids={"m-1", "m-2"})

    assert parsed["m-1"]["predicate"] == "prefers"
    assert parsed["m-1"]["object"] == "dark mode"
    assert "m-2" not in parsed


def test_parse_single_claim_has_no_id_assignment_surface():
    text = json.dumps(
        {
            "claim": {
                "subject": "Kris",
                "predicate": "uses",
                "object": "Gitea",
                "confidence": 0.8,
                "observed_at": None,
                "event_at": None,
                "valid_from": None,
                "valid_to": None,
                "scope_type": None,
                "scope_value": None,
                "conditions": [],
                "modality": "asserted",
            }
        }
    )

    parsed = parse_single_claim(text)

    assert parsed["subject"] == "Kris"
    assert parsed["object"] == "Gitea"


def test_parse_single_claim_recovers_id_typo_and_invalid_timestamp():
    text = json.dumps(
        {
            "claims": [
                {
                    "memory_id": "m-typo",
                    "claim": {
                        "subject": "Kris",
                        "predicate": "uses",
                        "object": "Gitea",
                        "confidence": 0.8,
                        "observed_at": "not-a-date",
                        "event_at": None,
                        "valid_from": None,
                        "valid_to": None,
                        "scope_type": None,
                        "scope_value": None,
                        "conditions": [],
                        "modality": "asserted",
                    },
                }
            ]
        }
    )

    parsed = parse_claim_batch(
        text,
        allowed_ids={"m-1"},
        recover_single_id=True,
    )

    assert parsed["m-1"]["object"] == "Gitea"
    assert parsed["m-1"]["observed_at"] is None


def test_parse_single_claim_accepts_exact_duplicate_entries():
    claim = {
        "subject": "Kris",
        "predicate": "uses",
        "object": "Gitea",
        "confidence": 0.8,
        "observed_at": None,
        "event_at": None,
        "valid_from": None,
        "valid_to": None,
        "scope_type": None,
        "scope_value": None,
        "conditions": [],
        "modality": "asserted",
    }
    text = json.dumps(
        {
            "claims": [
                {"memory_id": "m-1", "claim": claim},
                {"memory_id": "m-1", "claim": claim},
            ]
        }
    )

    parsed = parse_claim_batch(
        text,
        allowed_ids={"m-1"},
        recover_single_id=True,
    )

    assert parsed["m-1"]["object"] == "Gitea"


def test_parse_claim_batch_rejects_unknown_and_duplicate_ids():
    unknown = json.dumps(
        {"claims": [{"memory_id": "not-requested", "claim": {"subject": "Kris"}}]}
    )
    with pytest.raises(LLMResponseError, match="unknown memory_id"):
        parse_claim_batch(unknown, allowed_ids={"m-1"})

    duplicate = json.dumps(
        {
            "claims": [
                {"memory_id": "m-1", "claim": None},
                {"memory_id": "m-1", "claim": None},
            ]
        }
    )
    with pytest.raises(LLMResponseError, match="duplicate memory_id"):
        parse_claim_batch(duplicate, allowed_ids={"m-1"})


def test_apply_claim_projection_preserves_memory_and_audits_replacement(tmp_path: Path):
    db = open_db(tmp_path / "remnant.db")
    try:
        memory_id = db.insert_memory(
            content="Kris prefers dark mode",
            source="conversation",
            agent="default",
            type="fact",
        )
        record_claim_from_memory(
            db,
            memory_id=memory_id,
            subject="Kris",
            fact="Kris prefers dark mode",
            claim_data={"extractor_version": "legacy"},
            agent_id="default",
        )

        result = apply_claim_projection(
            db,
            memory_id=memory_id,
            claim={
                "subject": "Kris",
                "predicate": "prefers",
                "object": "dark mode",
                "confidence": 0.94,
                "valid_from": None,
                "valid_to": None,
                "event_at": None,
                "scope_type": None,
                "scope_value": None,
                "conditions": [],
                "modality": "asserted",
            },
            extractor_version="claims-v3-model-backfill",
            actor="test-model-backfill",
        )

        assert result["updated"] is True
        assert db.get_memory(memory_id)["content"] == "Kris prefers dark mode"
        stored = db.get_claim_for_memory(memory_id)
        assert stored["object"] == "dark mode"
        assert stored["confidence"] == pytest.approx(0.94)
        assert stored["extractor_version"] == "claims-v3-model-backfill"
        audit = db.list_audit(memory_id=memory_id, action="claim_model_backfill")
        assert len(audit) == 1
        assert audit[0]["details"]["before"]["extractor_version"] == "legacy"
        assert audit[0]["details"]["after"]["extractor_version"] == "claims-v3-model-backfill"
    finally:
        db.close()


def test_target_selection_leaves_current_structured_claims_alone(tmp_path: Path):
    db = open_db(tmp_path / "remnant.db")
    try:
        legacy = db.insert_memory(content="Kris uses Gitea", agent="default", type="fact")
        record_claim_from_memory(
            db,
            memory_id=legacy,
            subject="Kris",
            fact="Kris uses Gitea",
            claim_data={"extractor_version": "legacy"},
            agent_id="default",
        )
        current = db.insert_memory(content="Kris uses GitHub", agent="default", type="fact")
        record_claim_from_memory(
            db,
            memory_id=current,
            subject="Kris",
            fact="Kris uses GitHub",
            claim_data={"extractor_version": "claims-v2"},
            agent_id="default",
        )
        unclaimed = db.insert_memory(
            content="Kris reviews pull requests", agent="default", type="fact"
        )

        targets = _target_memories(db, extractor_version=MODEL_BACKFILL_VERSION, limit=None)
        target_ids = {row["id"] for row in targets}

        assert legacy in target_ids
        assert unclaimed in target_ids
        assert current not in target_ids
    finally:
        db.close()


def test_apply_claim_projection_creates_missing_claim_row(tmp_path: Path):
    db = open_db(tmp_path / "remnant.db")
    try:
        memory_id = db.insert_memory(
            content="Kris reviews pull requests",
            source="conversation",
            agent="default",
            type="fact",
        )
        result = apply_claim_projection(
            db,
            memory_id=memory_id,
            claim={
                "subject": "Kris",
                "predicate": "reviews",
                "object": "pull requests",
                "confidence": 0.8,
                "valid_from": None,
                "valid_to": None,
                "event_at": None,
                "scope_type": None,
                "scope_value": None,
                "conditions": [],
                "modality": "asserted",
            },
        )

        assert result["created"] is True
        stored = db.get_claim_for_memory(memory_id)
        assert stored is not None
        assert stored["subject"] == "Kris"
    finally:
        db.close()
