from __future__ import annotations

from remnant.config import RemnantConfig
from remnant.db import open_db
from remnant.ingest import store_memory
from remnant.recall import RecallRequest, RecallService


class _Embedder:
    _model = "test"

    @staticmethod
    def embed(text: str) -> list[float]:
        return [1.0, 0.0] if "dark" in text.casefold() else [0.0, 1.0]


def test_default_recall_groups_ambiguous_claims_before_limit(tmp_path):
    db = open_db(tmp_path / "recall.db")
    config = RemnantConfig(agent_id="agent", default_search_strategy="keyword")
    embedder = _Embedder()
    try:
        first = store_memory(
            db,
            embedder,
            config,
            fact="Sven prefers dark mode",
            entity="Sven",
            session_id="s",
            agent_id="agent",
        )
        second = store_memory(
            db,
            embedder,
            config,
            fact="Sven prefers light mode",
            entity="Sven",
            session_id="s",
            agent_id="agent",
        )
        response = RecallService(db, config).recall(
            RecallRequest(
                query="Sven preference",
                agent_id="agent",
                strategy="keyword",
                limit=1,
            ),
            embedder=embedder,
        )
        assert response.results
        result = response.results[0]
        assert result["claim_status"] == "unresolved"
        assert {row["id"] for row in result["claim_group"]} == {first, second}
        assert response.diagnostics["ranking_profile"] == "claims-v1"
    finally:
        db.close()


def test_recall_context_skips_oversized_candidate_and_keeps_later_fact(tmp_path):
    db = open_db(tmp_path / "budget.db")
    config = RemnantConfig(injection_token_budget=80)
    long_id = db.insert_memory(content="x" * 2000, agent="default")
    short_id = db.insert_memory(content="Sven likes tea", agent="default")
    try:
        response = RecallService(db, config).recall(
            RecallRequest(
                query="remember Sven preference",
                agent_id="default",
                limit=5,
                output_mode="context",
            ),
            candidates=[
                {"id": long_id, "content": "x" * 2_000, "visibility": "private"},
                {"id": short_id, "content": "Sven likes tea", "visibility": "private"},
            ],
        )
        assert "Sven likes tea" in response.context
        assert "x" * 100 not in response.context
        assert response.diagnostics["selected_count"] == 1
    finally:
        db.close()
