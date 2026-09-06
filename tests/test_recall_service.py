from __future__ import annotations

from remnant import RemnantMemoryProvider
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


def test_pending_turns_use_the_same_scoped_recall_path_in_tools_and_prefetch(tmp_path):
    db = open_db(tmp_path / "pending.db")
    config = RemnantConfig(agent_id="claire", echo_enabled=False)
    provider = RemnantMemoryProvider()
    provider._db, provider._config = db, config
    provider._embedder = _Embedder()
    try:
        for owner, session, text in (
            ("claire", "s", "My project uses amber deployment gates."),
            ("sasha", "s", "My project uses blue deployment gates."),
            ("claire", "other", "My project uses green deployment gates."),
        ):
            db.insert_turn_with_extraction(
                agent_id=owner, session_id=session, user_text=text, assistant_text="OK",
            )
        query = "What did I say about my project?"
        response = RecallService(db, config).recall(RecallRequest(
            query=query, agent_id="claire", session_id="s", include_pending=True,
            output_mode="context",
        ))
        context = provider.prefetch(query, session_id="s")
        assert context == response.context
        assert "amber" in context and "unprocessed" in context.lower()
        assert "blue" not in context and "green" not in context
        assert len(response.results) == 1
        assert response.results[0]["agent_id"] == "claire"
        assert provider.prefetch(query, session_id="s", messages=[{
            "role": "user", "content": "My project uses amber deployment gates.",
        }]) == ""
    finally:
        db.close()


def test_pending_queue_failure_retains_committed_recall(tmp_path, monkeypatch):
    db = open_db(tmp_path / "queue-failure.db")
    try:
        mid = db.insert_memory(content="The project uses amber deployment gates.", agent="claire")

        def unavailable(**kwargs):
            raise RuntimeError("queue unavailable")

        monkeypatch.setattr(db, "get_pending_turns", unavailable)
        response = RecallService(db, RemnantConfig(agent_id="claire")).recall(RecallRequest(
            query="project deployment", agent_id="claire", include_pending=True,
            strategy="keyword", output_mode="context",
        ))
        assert response.rendered_ids == (mid,)
        assert "amber" in response.context
        assert response.diagnostics["degraded"]
        assert response.diagnostics["reason"] == "pending_overlay_failed"
    finally:
        db.close()
