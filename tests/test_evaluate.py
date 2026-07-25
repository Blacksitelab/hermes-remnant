from __future__ import annotations

from remnant.config import RemnantConfig
from remnant.db import default_db_path, open_db
from remnant.evaluate import evaluate_cases
from remnant.ingest import store_memory


class _Embedder:
    _model = "test"

    @staticmethod
    def embed(_text: str) -> list[float]:
        return [1.0]


def test_evaluate_cases_reports_retrieval_metrics():
    db = open_db(default_db_path())
    cfg = RemnantConfig(default_search_strategy="keyword")
    emb = _Embedder()
    try:
        mid = store_memory(
            db, emb, cfg, fact="Sven prefers dark mode", entity="Sven",
            session_id="seed", agent_id="default", source="manual",
        )
        result = evaluate_cases(
            db, cfg, emb,
            [{"query": "Sven dark mode", "expected_ids": [mid], "strategy": "keyword"}],
        )
        assert result["cases"] == 1
        assert result["recall_at_k"] == 1.0
        assert result["mrr"] == 1.0
        assert result["details"][0]["first_relevant_rank"] == 1
    finally:
        db.close()
