from __future__ import annotations

from pathlib import Path

from remnant import RemnantMemoryProvider
from remnant.claims import record_claim_from_memory
from remnant.config import RemnantConfig, apply_config_preset, load_config, save_config
from remnant.context import compile_context
from remnant.db import default_db_path, open_db
from remnant.maintenance import health_report
from remnant.resolve import resolve_results


def test_claim_aware_profile_is_the_new_configuration_default(tmp_path: Path):
    config = RemnantConfig()
    assert config.structured_claim_extraction_v2 is True
    assert config.claim_reconciliation_enabled is True
    assert config.claim_aware_ranking_enabled is True
    assert config.resolved_context_enabled is True
    assert config.recent_turn_overlay_enabled is True
    assert config.relation_evidence_enabled is True
    assert config.ranking_profile == "claims-v1"
    # Stable runtime identity is a deployment prerequisite, not a safe generic
    # default: the identity fallback deliberately isolates anonymous sessions.
    assert config.runtime_identity_enabled is False

    schema_defaults = {
        row["key"]: row.get("default")
        for row in RemnantMemoryProvider().get_config_schema()
    }
    for key in (
        "structured_claim_extraction_v2",
        "claim_reconciliation_enabled",
        "claim_aware_ranking_enabled",
        "resolved_context_enabled",
        "recent_turn_overlay_enabled",
        "relation_evidence_enabled",
    ):
        assert schema_defaults[key] is True
    assert schema_defaults["ranking_profile"] == "claims-v1"
    assert schema_defaults["runtime_identity_enabled"] is False

    # Explicit rollback values in an existing config must remain authoritative.
    home = tmp_path / "legacy-overrides"
    save_config(
        {
            "claim_aware_ranking_enabled": False,
            "resolved_context_enabled": False,
            "ranking_profile": "legacy",
        },
        home,
    )
    loaded = load_config(home)
    assert loaded.claim_aware_ranking_enabled is False
    assert loaded.resolved_context_enabled is False
    assert loaded.ranking_profile == "legacy"


def test_named_config_profiles_preserve_unrelated_settings():
    values = {"agent_id": "coder", "injection_token_budget": 1234}
    updated = apply_config_preset(values, "legacy")
    assert updated["agent_id"] == "coder"
    assert updated["injection_token_budget"] == 1234
    assert updated["claim_aware_ranking_enabled"] is False
    assert updated["ranking_profile"] == "legacy"


def test_claim_metadata_migrates_and_resolves_without_losing_history():
    db = open_db(default_db_path())
    try:
        first = db.insert_memory(content="Sven prefers light mode", agent="default")
        record_claim_from_memory(
            db,
            memory_id=first,
            subject="Sven",
            fact="Sven prefers light mode",
            claim_data={"confidence": 0.9, "extractor_version": "claims-v2"},
            reconciliation_enabled=True,
            source_turn_id=11,
        )
        second = db.insert_memory(content="Sven now prefers dark mode", agent="default")
        record_claim_from_memory(
            db,
            memory_id=second,
            subject="Sven",
            fact="Sven now prefers dark mode",
            claim_data={
                "confidence": 0.9,
                "conflict_type": "update",
                "valid_from": "2026-08-01T00:00:00Z",
                "conditions": ["at home"],
                "extractor_version": "claims-v2",
            },
            reconciliation_enabled=True,
            source_turn_id=12,
        )
        old = db.get_claim_for_memory(first)
        new = db.get_claim_for_memory(second)
        assert old and new
        assert old["status"] == "superseded"
        assert new["source_turn_id"] == 12
        assert new["resolution_status"] == "update"
        assert new["valid_from"] == "2026-08-01T00:00:00Z"
        resolved = resolve_results(
            db,
            [
                {"id": first, "content": "Sven prefers light mode", "score": 0.8},
                {"id": second, "content": "Sven now prefers dark mode", "score": 0.7},
            ],
            query="Sven preference",
        )
        assert [row["id"] for row in resolved] == [second]
        context = compile_context(resolved)
        assert "Conditional memory" in context
        assert "m:" in context
        assert "instructions" in context
    finally:
        db.close()


def test_health_report_exposes_schema_and_claim_lifecycle():
    db = open_db(default_db_path())
    try:
        report = health_report(db)
        assert report["schema_version"] == 15
        assert "claims_by_resolution" in report
        assert "pending_extraction_age_s" in report
    finally:
        db.close()


def test_runtime_identity_scopes_provider_and_non_primary_skips_writes(
    tmp_path: Path,
):
    home = tmp_path / "hermes"
    home.mkdir()
    save_config(
        {
            "extract_enabled": False,
            "runtime_identity_enabled": True,
            "agent_id": "legacy",
        },
        home,
    )
    provider = RemnantMemoryProvider()
    provider.initialize(
        session_id="s1",
        hermes_home=str(home),
        agent_identity="coder",
        agent_workspace="hermes",
        agent_context="subagent",
        platform="cli",
    )
    try:
        assert provider._config.agent_id.startswith("identity:v1:")  # type: ignore[union-attr]
        provider.sync_turn("should not persist", "", session_id="s1")
        assert provider._db.pending_extraction_count(agent_id="hermes:coder") == 0  # type: ignore[union-attr]
    finally:
        provider.shutdown()


def test_recent_turn_overlay_recalls_pending_turn_without_durable_match(
    tmp_path: Path,
):
    home = tmp_path / "hermes-overlay"
    home.mkdir()
    save_config(
        {
            "extract_enabled": False,
            "recent_turn_overlay_enabled": True,
            "resolved_context_enabled": True,
            "prefetch_embedding_timeout_ms": 0,
        },
        home,
    )
    provider = RemnantMemoryProvider()
    provider.initialize(session_id="s1", hermes_home=str(home))
    try:
        provider._worker.stop()  # type: ignore[union-attr]
        provider._db.insert_turn_with_extraction(  # type: ignore[union-attr]
            session_id="s1",
            agent_id="default",
            user_text="The printer is an Elegoo Centauri Carbon V1",
            assistant_text="Noted.",
        )
        context = provider.prefetch("what did I say in the previous turn?", session_id="s1")
        assert "Recent unprocessed turn" in context
        assert "Elegoo Centauri Carbon V1" in context
        job = provider._db.claim_next_extraction(agent_id="default")  # type: ignore[union-attr]
        provider._db.complete_extraction(job["id"], fact_count=0)  # type: ignore[union-attr]
        provider._last_injected_hash.clear()
        after = provider.prefetch("what did I say in the previous turn?", session_id="s1")
        assert "Recent unprocessed turn" not in after
    finally:
        provider.shutdown()
