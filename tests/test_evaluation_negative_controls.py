from __future__ import annotations

from remnant.config import RemnantConfig
from remnant.db import open_db
from remnant.evaluation.metrics import ranking_metrics
from remnant.recall import RecallRequest, RecallService


def test_unauthorized_only_control_returns_no_candidate(tmp_path):
    db = open_db(tmp_path / "negative.db")
    try:
        db.insert_memory(
            content="Morgan's private recovery code is cedar",
            agent="other-user",
            visibility="private",
        )
        response = RecallService(db, RemnantConfig()).recall(
            RecallRequest(
                query="Morgan recovery code",
                agent_id="current-user",
                strategy="keyword",
            )
        )
        assert response.results == []
    finally:
        db.close()


def test_random_or_stale_only_control_fails_relevance_metrics():
    assert ranking_metrics(["wrong-memory"], {"target-memory"})["recall_at_5"] == 0.0
    assert ranking_metrics(["old-version"], {"new-version"})["mrr"] == 0.0
