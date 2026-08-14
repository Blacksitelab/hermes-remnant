from __future__ import annotations

import time

from remnant.config import RemnantConfig
from remnant.context import compile_context_details
from remnant.db import open_db
from remnant.echo import EchoService
from remnant.echo_worker import EchoWorker
from remnant.ingest import ingest_turn, store_memory


class _Embedder:
    _model = "test"

    @staticmethod
    def embed(text: str) -> list[float]:
        return [1.0, 0.0]


def _context(memory_id: str = "m1"):
    return compile_context_details(
        [
            {
                "id": memory_id,
                "content": "Sven prefers dark mode",
                "score": 0.9,
                "evidence_class": "current_claim",
                "claim_status": "active",
            }
        ],
        token_budget=120,
    )


def _draft(service: EchoService, memory_id: str = "m1"):
    return service.build_receipt_draft(
        query="What is Sven's preference?",
        session_id="s1",
        agent_id="agent",
        viewer_key="viewer",
        profile_scope=[],
        memory_generation=1,
        context=_context(memory_id),
    )


def test_echo_receipt_is_exact_and_closes_to_persisted_turn(tmp_path):
    db = open_db(tmp_path / "echo.db")
    config = RemnantConfig(agent_id="agent", echo_initial_sample_rate=0.0)
    service = EchoService(db, config)
    try:
        draft = _draft(service)
        assert draft is not None
        receipt_id = service.activate_receipt(draft)
        assert receipt_id
        receipt = service.store.get_receipt(receipt_id)
        assert receipt is not None
        assert receipt["status"] == "open"
        assert [item["memory_id"] for item in receipt["items"]] == ["m1"]
        turn_id = ingest_turn(
            db,
            user_text="What is Sven's preference?",
            assistant_text="Sven prefers dark mode.",
            session_id="s1",
            agent_id="agent",
        )
        assert service.close_receipt(
            session_id="s1",
            viewer_key="viewer",
            query="What is Sven's preference?",
            turn_id=turn_id,
        ) == receipt_id
        assert service.store.get_receipt(receipt_id)["status"] == "closed"
    finally:
        db.close()


def test_echo_receipt_is_created_only_when_context_is_consumed(tmp_path):
    db = open_db(tmp_path / "echo-consume.db")
    config = RemnantConfig(agent_id="agent", echo_initial_sample_rate=0.0)
    service = EchoService(db, config)
    try:
        from remnant import RemnantMemoryProvider

        provider = RemnantMemoryProvider()
        provider._echo = service
        draft = _draft(service)
        assert provider._consume_prefetch_result({"context": "", "_echo_draft": draft}) == ""
        assert service.receipt_id_for_activation(draft.activation_key) is None
        assert provider._consume_prefetch_result(
            {"context": draft.context_hash, "_echo_draft": draft}
        )
        assert service.receipt_id_for_activation(draft.activation_key)
    finally:
        db.close()


def test_echo_feedback_aggregates_and_influences_only_after_threshold(tmp_path):
    db = open_db(tmp_path / "echo-utility.db")
    config = RemnantConfig(
        agent_id="agent",
        echo_shadow_mode=False,
        echo_rank_influence=1.0,
        echo_min_observations=1,
        echo_max_rank_adjustment=0.1,
    )
    service = EchoService(db, config)
    try:
        for _ in range(3):
            service.record_feedback(
                memory_id="m1",
                feedback="useful",
                agent_id="agent",
                viewer_key="viewer",
                query="What is Sven's preference?",
            )
        assert service.aggregate(limit=20) == 6
        results, diagnostics = service.adjust_results(
            [{"id": "m1", "score": 0.5}],
            query="What is Sven's preference?",
            agent_id="agent",
            viewer_key="viewer",
        )
        assert results[0]["score"] > 0.5
        assert diagnostics.utility_hits == 2
        assert results[0]["ranking"]["echo"]["policy"] == "echo-v1"
    finally:
        db.close()


def test_echo_worker_persists_inferred_signal_with_retry_safe_job(tmp_path):
    db = open_db(tmp_path / "echo-worker.db")
    config = RemnantConfig(agent_id="agent", echo_max_jobs_per_day=10)
    service = EchoService(db, config)
    embedder = _Embedder()
    try:
        memory_id = store_memory(
            db,
            embedder,
            config,
            fact="Sven prefers dark mode",
            entity="Sven",
            session_id="s1",
            agent_id="agent",
        )
        assert memory_id
        draft = _draft(service, memory_id)
        receipt_id = service.activate_receipt(draft)
        job_id = service.store.enqueue_job(
            receipt_id=receipt_id,
            job_type="single",
            target_ids=[memory_id],
            priority=1.0,
            evaluator_version="echo-v1",
        )
        worker = EchoWorker(
            service,
            config,
            evaluator=lambda job, payload: [
                {"memory_id": memory_id, "signal_type": "counterfactual_support", "confidence": 1}
            ],
        )
        # A sampled job must wait until the injected receipt is matched to a
        # persisted turn; evaluating with an empty answer would be misleading.
        assert worker.run_once() is False
        turn_id = ingest_turn(
            db,
            user_text="What is Sven's preference?",
            assistant_text="Sven prefers dark mode.",
            session_id="s1",
            agent_id="agent",
        )
        service.close_receipt(
            session_id="s1",
            viewer_key="viewer",
            query="What is Sven's preference?",
            turn_id=turn_id,
        )
        assert worker.run_once() is True
        with db.read() as cur:
            cur.execute("SELECT status FROM echo_jobs WHERE id=?", (job_id,))
            assert cur.fetchone()["status"] == "done"
            cur.execute("SELECT COUNT(*) AS n FROM echo_signals WHERE memory_id=?", (memory_id,))
            assert cur.fetchone()["n"] == 1
        assert service.health()["echo_utility"] == 1
    finally:
        db.close()


def test_expired_receipt_cancels_pending_evaluator_job(tmp_path):
    db = open_db(tmp_path / "echo-expiry.db")
    config = RemnantConfig(agent_id="agent", echo_initial_sample_rate=0.0)
    service = EchoService(db, config)
    try:
        draft = _draft(service)
        receipt_id = service.activate_receipt(draft)
        job_id = service.store.enqueue_job(
            receipt_id=receipt_id,
            job_type="single",
            target_ids=["m1"],
            priority=1.0,
            evaluator_version="echo-v1",
        )
        service.store.compact(now=time.time() + 301)
        with db.read() as cur:
            cur.execute("SELECT status FROM echo_jobs WHERE id=?", (job_id,))
            assert cur.fetchone()["status"] == "skipped"
    finally:
        db.close()


def test_echo_config_rejects_unbounded_rank_influence():
    try:
        RemnantConfig.from_dict({"echo_max_rank_adjustment": 0.5})
    except ValueError as exc:
        assert "echo_max_rank_adjustment" in str(exc)
    else:
        raise AssertionError("invalid Echo ranking bound was accepted")
