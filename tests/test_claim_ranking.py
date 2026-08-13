from __future__ import annotations

from pathlib import Path

from remnant.context import compile_context, conservative_token_count
from remnant.db import open_db
from remnant.ranking import rank_results


def test_verified_quality_breaks_equal_relevance_tie(tmp_path: Path):
    db = open_db(tmp_path / "rank.db")
    try:
        weak = db.insert_memory(content="Kris prefers dark", confidence=0.2, trust_score=0.2)
        strong = db.insert_memory(content="Kris prefers dark mode", confidence=0.9, trust_score=0.9)
        db.set_memory_field(strong, "verified", 1, actor="test", action="verify")
        ranked = rank_results(
            db,
            [
                {"id": weak, "content": "Kris prefers dark", "score": 1.0},
                {"id": strong, "content": "Kris prefers dark mode", "score": 1.0},
            ],
        )
        assert ranked[0]["id"] == strong
        assert 0.8 <= ranked[0]["ranking"]["quality"]["bounded"] <= 1.2
    finally:
        db.close()


def test_relevance_is_demoted_not_erased_by_low_quality(tmp_path: Path):
    db = open_db(tmp_path / "relevance.db")
    try:
        relevant = db.insert_memory(content="exact", confidence=0.1, trust_score=0.1)
        weak_match = db.insert_memory(content="other", confidence=1.0, trust_score=1.0)
        ranked = rank_results(
            db,
            [
                {"id": relevant, "content": "exact", "score": 1.0},
                {"id": weak_match, "content": "other", "score": 0.2},
            ],
        )
        assert ranked[0]["id"] == relevant
        assert ranked[0]["score"] > 0
    finally:
        db.close()


def test_ranker_normalizes_scores_within_search_lane(tmp_path: Path):
    db = open_db(tmp_path / "lanes.db")
    try:
        first = db.insert_memory(content="first", source="manual")
        second = db.insert_memory(content="second", source="conversation")
        ranked = rank_results(
            db,
            [
                {"id": first, "content": "first", "score": 100.0, "_score_lane": "keyword"},
                {"id": second, "content": "second", "score": 0.9, "_score_lane": "semantic"},
            ],
        )
        assert {row["ranking"]["score_lane"] for row in ranked} == {"keyword", "semantic"}
        assert all("source_authority" in row["ranking"]["quality"] for row in ranked)
    finally:
        db.close()


def test_context_truncates_items_and_stays_inside_budget():
    memories = [
        {"id": "one", "content": "x" * 900, "visibility": "private"},
        {"id": "two", "content": "compact fact", "visibility": "private"},
    ]
    rendered = compile_context(memories, token_budget=100)
    assert conservative_token_count(rendered) <= 100
    assert "…" in rendered
    assert "compact fact" in rendered


def test_historical_keyword_search_can_load_superseded_evidence(tmp_path: Path):
    from remnant.config import RemnantConfig
    from remnant.search import search

    db = open_db(tmp_path / "history.db")
    try:
        old = db.insert_memory(content="Kris preferred light mode", agent="a")
        new = db.insert_memory(content="Kris prefers dark mode", agent="a")
        db.supersede(old, new)
        config = RemnantConfig(agent_id="a", claim_aware_ranking_enabled=True)
        current = search(db, config, "Kris light mode", agent_id="a", strategy="keyword")
        history = search(
            db,
            config,
            "Previously Kris light mode",
            agent_id="a",
            strategy="keyword",
        )
        assert all(row["id"] != old for row in current)
        assert any(row["id"] == old for row in history)
    finally:
        db.close()


def test_context_honours_exact_custom_token_counter():
    counter = lambda value: len(value.split())  # noqa: E731
    rendered = compile_context(
        [{"id": "one", "content": "word " * 100, "visibility": "private"}],
        token_budget=30,
        token_counter=counter,
    )
    assert counter(rendered) <= 30
