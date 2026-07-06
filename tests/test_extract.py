"""Issue #13 tests: extraction worker startup sweep, LIFO queue ordering, and
timing logs. Runs without a live Ollama by stubbing the LLM call.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from remnant.config import RemnantConfig
from remnant.db import default_db_path, open_db
from remnant.embed import Embedder
from remnant.extract import ExtractionWorker


def _hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fake_embed(db, config, dim=8):
    emb = Embedder.__new__(Embedder)
    emb._db = db
    emb._model = config.embed_model
    emb._url = config.embed_url
    emb._timeout = config.embed_timeout
    emb._client = None

    def embed(text: str) -> list[float]:
        import hashlib

        cached = db.get_cached_embedding(emb._model, _hash(text))
        if cached is not None:
            return cached
        words = [w.lower() for w in text.strip().split()]
        vec = [0.0] * dim
        for w in words:
            h = int.from_bytes(hashlib.sha256(w.encode("utf-8")).digest()[:4], "big") % dim
            vec[h] += 1.0
        n = sum(v * v for v in vec) ** 0.5
        if n:
            vec = [v / n for v in vec]
        db.put_cached_embedding(emb._model, _hash(text), vec)
        return vec

    emb.embed = embed
    emb.close = lambda: None
    return emb


@pytest.fixture()
def hermes_home(tmp_path: Path) -> Path:
    home = tmp_path / "hermes"
    (home / "remnant").mkdir(parents=True, exist_ok=True)
    return home


def _open_db(hermes_home: Path):
    return open_db(default_db_path())


def _insert_turn(db, *, agent_id="default", session_id="s", user="u", assistant="a"):
    return db.insert_turn(
        session_id=session_id,
        agent_id=agent_id,
        user_text=user,
        assistant_text=assistant,
    )


# ===========================================================================
# get_unextracted_turns
# ===========================================================================


def test_get_unextracted_turns_returns_unqueued_unextracted(hermes_home: Path):
    db = _open_db(hermes_home)
    try:
        tid = _insert_turn(db, user="Sven prefers dark mode", assistant="ok")
        rows = db.get_unextracted_turns()
        assert len(rows) == 1
        assert rows[0]["id"] == tid
        assert rows[0]["user_text"] == "Sven prefers dark mode"
    finally:
        db.close()


def test_get_unextracted_turns_excludes_queued(hermes_home: Path):
    db = _open_db(hermes_home)
    try:
        tid = _insert_turn(db)
        db.enqueue_extraction(
            turn_id=tid, session_id="s", agent_id="default",
            user_text="u", assistant_text="a",
        )
        assert db.get_unextracted_turns() == []
    finally:
        db.close()


def test_get_unextracted_turns_excludes_already_extracted(hermes_home: Path):
    db = _open_db(hermes_home)
    try:
        tid = _insert_turn(db)
        # Insert a conversation memory sourced from this turn; it counts as
        # already extracted even though there is no extraction_queue row.
        db.insert_memory(
            content="Sven prefers dark mode",
            source="conversation",
            source_id=str(tid),
            agent="default",
            type="fact",
        )
        assert db.get_unextracted_turns() == []
    finally:
        db.close()


def test_get_unextracted_turns_filters_by_agent(hermes_home: Path):
    db = _open_db(hermes_home)
    try:
        _insert_turn(db, agent_id="alice")
        _insert_turn(db, agent_id="bob")
        rows = db.get_unextracted_turns(agent_id="alice")
        assert len(rows) == 1
        assert rows[0]["agent_id"] == "alice"
    finally:
        db.close()


def test_get_unextracted_turns_orders_by_id_desc(hermes_home: Path):
    db = _open_db(hermes_home)
    try:
        t1 = _insert_turn(db, user="first")
        t2 = _insert_turn(db, user="second")
        t3 = _insert_turn(db, user="third")
        rows = db.get_unextracted_turns()
        ids = [r["id"] for r in rows]
        assert ids == sorted([t1, t2, t3], reverse=True)
    finally:
        db.close()


# ===========================================================================
# claim_next_extraction: LIFO ordering
# ===========================================================================


def test_claim_next_extraction_is_lifo(hermes_home: Path):
    db = _open_db(hermes_home)
    try:
        # Enqueue three jobs in order; LIFO means the last one is claimed first.
        db.enqueue_extraction(
            turn_id=_insert_turn(db, user="first"), session_id="s", agent_id="default",
            user_text="first", assistant_text="a",
        )
        db.enqueue_extraction(
            turn_id=_insert_turn(db, user="second"), session_id="s", agent_id="default",
            user_text="second", assistant_text="a",
        )
        db.enqueue_extraction(
            turn_id=_insert_turn(db, user="third"), session_id="s", agent_id="default",
            user_text="third", assistant_text="a",
        )
        job = db.claim_next_extraction(agent_id="default")
        assert job is not None
        assert job["user_text"] == "third"
        db.complete_extraction(int(job["id"]))
        job2 = db.claim_next_extraction(agent_id="default")
        assert job2 is not None
        assert job2["user_text"] == "second"
    finally:
        db.close()


def test_claim_next_extraction_lifo_with_agent_filter(hermes_home: Path):
    db = _open_db(hermes_home)
    try:
        db.enqueue_extraction(
            turn_id=_insert_turn(db, agent_id="alice", user="alice1"),
            session_id="s", agent_id="alice", user_text="alice1", assistant_text="a",
        )
        db.enqueue_extraction(
            turn_id=_insert_turn(db, agent_id="bob", user="bob1"),
            session_id="s", agent_id="bob", user_text="bob1", assistant_text="a",
        )
        db.enqueue_extraction(
            turn_id=_insert_turn(db, agent_id="alice", user="alice2"),
            session_id="s", agent_id="alice", user_text="alice2", assistant_text="a",
        )
        job = db.claim_next_extraction(agent_id="alice")
        assert job is not None
        assert job["agent_id"] == "alice"
        assert job["user_text"] == "alice2"
    finally:
        db.close()


# ===========================================================================
# Startup sweep
# ===========================================================================


def test_queue_startup_sweep_enqueues_unextracted_turns(hermes_home: Path, monkeypatch):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        _insert_turn(db, user="first unextracted turn")
        _insert_turn(db, user="second unextracted turn")
        # A third turn that is already queued must NOT be re-enqueued.
        queued_tid = _insert_turn(db, user="already queued")
        db.enqueue_extraction(
            turn_id=queued_tid, session_id="s", agent_id="default",
            user_text="already queued", assistant_text="a",
        )

        worker = ExtractionWorker(db, emb, cfg)
        calls: list[int] = []
        original = db.enqueue_extraction

        def spy_enqueue(*, turn_id, session_id, agent_id, user_text, assistant_text):
            calls.append(turn_id)
            return original(
                turn_id=turn_id, session_id=session_id, agent_id=agent_id,
                user_text=user_text, assistant_text=assistant_text,
            )

        monkeypatch.setattr(db, "enqueue_extraction", spy_enqueue)

        worker.queue_startup_sweep()
        worker._enqueue_startup()

        # The two unextracted turns are enqueued; the already-queued one is not.
        assert len(calls) == 2
        assert queued_tid not in calls
        # _pending_startup is cleared after _enqueue_startup runs.
        assert worker._pending_startup == []
    finally:
        db.close()


def test_queue_startup_sweep_logs_count(hermes_home: Path, caplog):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        _insert_turn(db, user="turn one")
        _insert_turn(db, user="turn two")
        _insert_turn(db, user="turn three")
        worker = ExtractionWorker(db, emb, cfg)
        with caplog.at_level(logging.INFO, logger="remnant.extract"):
            worker.queue_startup_sweep()
        assert any(
            "startup sweep" in rec.message and "3" in rec.message
            for rec in caplog.records
        )
    finally:
        db.close()


# ===========================================================================
# Timing logs
# ===========================================================================


def test_process_emits_timing_log(hermes_home: Path, caplog, monkeypatch):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        worker = ExtractionWorker(db, emb, cfg)
        # Stub _extract to return one fact and avoid any network call.
        monkeypatch.setattr(
            worker, "_extract", lambda u, a: [{"fact": "Sven prefers dark mode", "entity": "Sven"}]
        )
        # Stub store_memory so no DB writes happen beyond what we assert.
        from remnant import ingest as ingest_mod

        monkeypatch.setattr(
            ingest_mod, "store_memory", lambda *a, **k: "mid"
        )
        job = {
            "id": 1, "turn_id": 42, "session_id": "s", "agent_id": "default",
            "user_text": "u", "assistant_text": "a",
        }
        with caplog.at_level(logging.INFO, logger="remnant.extract"):
            worker._process(job)
        assert any(
            "extraction complete" in rec.message and "turn_id=42" in rec.message
            and "duration_ms=" in rec.message
            for rec in caplog.records
        )
    finally:
        db.close()