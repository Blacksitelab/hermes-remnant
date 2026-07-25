"""Structured claim projection and versioning tests."""

from __future__ import annotations

from pathlib import Path

from remnant.config import RemnantConfig
from remnant.db import open_db
from remnant.ingest import store_memory


class _Embedder:
    _model = "test"

    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0] if "dark" in text.lower() else [0.0, 1.0]


def test_new_fact_creates_source_backed_claim_and_versions_prior_value(tmp_path: Path):
    db = open_db(tmp_path / "remnant.db")
    cfg = RemnantConfig()
    emb = _Embedder()
    try:
        first = store_memory(
            db, emb, cfg, fact="Sven prefers dark mode", entity="Sven",
            session_id="s", agent_id="agent",
        )
        second = store_memory(
            db, emb, cfg, fact="Sven prefers light mode", entity="Sven",
            session_id="s", agent_id="agent",
        )
        first_claim = db.get_claim_for_memory(first)
        second_claim = db.get_claim_for_memory(second)
        assert first_claim["predicate"] == "prefers"
        assert first_claim["status"] == "superseded"
        assert second_claim["object"] == "light mode"
        assert second_claim["status"] == "active"
        assert first_claim["valid_to"]
    finally:
        db.close()
