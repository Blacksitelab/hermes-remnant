"""Regressions through extraction, storage, retrieval, and context delivery."""

import json
import sqlite3
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from remnant import RemnantMemoryProvider
from remnant.config import RemnantConfig
from remnant.db import open_db
from remnant.embed import Embedder
from remnant.extract import ExtractionWorker
from remnant.ingest import store_memory
from remnant.maintenance import repair_embeddings
from remnant.search import search
from remnant.vault import index_file


class Vectors:
    _model = "test"
    _dim = 2

    def __init__(self):
        self.calls = []
        self.vector = [1.0, 0.0]

    def embed(self, text, **kwargs):
        self.calls.append(text)
        return self.vector


@pytest.fixture
def db(tmp_path):
    database = open_db(tmp_path / "regression.db")
    yield database
    database.close()


def provider(db, **options):
    instance = RemnantMemoryProvider()
    instance._db = db
    instance._config = RemnantConfig(agent_id="a", embed_model="test", embed_dim=2,
                                     echo_enabled=False, **options)
    instance._embedder = Vectors()
    return instance


def test_extraction_keeps_high_similarity_correction_and_recall_uses_it(db, monkeypatch):
    p = provider(db)
    worker = ExtractionWorker(db, p._embedder, p._config)
    facts = [
        {"fact": "Morgan prefers dark mode and avoids light mode.",
         "object": "dark mode and avoids light mode", "conflict_type": "compatible"},
        {"fact": "Morgan now prefers light mode and avoids dark mode.",
         "object": "light mode and avoids dark mode", "conflict_type": "update"},
    ]
    monkeypatch.setattr("remnant.extract.extract_high_signal_entities", lambda *a, **k: [])
    try:
        for fact in facts:
            payload = {"subject": "Morgan", "predicate": "prefers", "confidence": .95, **fact}
            monkeypatch.setattr(
                "remnant.extract.chat", lambda **k: json.dumps({"facts": [payload]}),
            )
            db.insert_turn_with_extraction(session_id="s", agent_id="a",
                                           user_text=fact["fact"], assistant_text="OK")
            job = db.claim_next_extraction(agent_id="a")
            assert worker._process(job) == 1
            db.complete_extraction(job["id"], fact_count=1)
        with db.read() as cur:
            assert cur.execute("SELECT count(*) FROM memories").fetchone()[0] == 2
        active = db.get_active_claim("Morgan", "prefers", agent_id="a")
        assert active["object"] == facts[1]["object"]
        assert active["resolution_status"] == "update"
        context = p.prefetch("What mode does Morgan prefer?", session_id="fresh")
        assert "light mode and avoids dark mode" in context
        assert "prefers dark mode and avoids light mode" not in context
    finally:
        worker._client.close()


def test_equivalent_observation_keeps_provenance_and_retry_is_idempotent(db):
    p = provider(db)
    args = dict(fact="Morgan uses Linux.", entity="Morgan", session_id="s", agent_id="a")
    first = store_memory(db, p._embedder, p._config, source_turn_id=1, **args)
    for _ in range(2):
        assert store_memory(db, p._embedder, p._config, source_turn_id=2, **args) is None
    assert db.get_memory(first)["seen_count"] == 2
    with db.read() as cur:
        records = cur.execute(
            "SELECT details FROM audit_log WHERE action='memory_duplicate'",
        ).fetchall()
    assert [json.loads(row[0])["source_turn_id"] for row in records] == [2]


@pytest.mark.parametrize("confidence", [.5, .95])
def test_model_omitting_transition_cannot_hide_the_correction(db, monkeypatch, confidence):
    p = provider(db)
    worker = ExtractionWorker(db, p._embedder, p._config)
    monkeypatch.setattr("remnant.extract.extract_high_signal_entities", lambda *a, **k: [])
    try:
        for mode in ("dark", "light"):
            # The model drops 'Correction'/'now' from its fact and conflict label.
            fact = {"fact": f"Morgan prefers {mode} mode", "subject": "Morgan",
                    "predicate": "prefers", "object": f"{mode} mode", "confidence": confidence}
            monkeypatch.setattr("remnant.extract.chat", lambda **k: json.dumps({"facts": [fact]}))
            text = fact["fact"] if mode == "dark" else "Correction: Morgan now prefers light mode"
            db.insert_turn_with_extraction(session_id="s", agent_id="a",
                                           user_text=text, assistant_text="OK")
            job = db.claim_next_extraction("a")
            worker._process(job)
            db.complete_extraction(job["id"], fact_count=1)
        ctx = p.prefetch("What mode does Morgan prefer?", session_id="new")
        assert "light mode" in ctx
        if confidence < .75:
            assert "dark mode" in ctx
            assert "Unresolved" in ctx or "unresolved" in ctx
        else:
            assert "dark mode" not in ctx
    finally:
        worker._client.close()


def test_vault_failed_update_clears_vector_and_unchanged_file_retries(db, tmp_path):
    cfg = RemnantConfig(vault_path=str(tmp_path), embed_model="test", embed_dim=2)
    emb = Vectors()
    note = tmp_path / "project.md"
    note.write_text("# Project\nOriginal backup policy.")
    mid = index_file(db, cfg, emb, note)
    note.write_text("# Project\nChanged backup policy.")
    emb.vector = None
    assert index_file(db, cfg, emb, note) == mid
    assert db.get_memory_embedding(mid) == []
    emb.vector = [0.0, 1.0]
    assert index_file(db, cfg, emb, note) == mid
    assert db.get_memory_embedding(mid) == [0.0, 1.0]
    assert len(db.list_memories(limit=10)) == 1


def test_repair_does_not_attach_vector_to_concurrently_changed_text(db):
    mid = db.insert_memory(content="Original policy", agent="a")
    def embed(text, **kwargs):
        db.update_memory_content(mid, content="Changed policy")
        return [1.0, 0.0]
    emb = SimpleNamespace(_model="test", _dim=2, embed=embed)
    result = repair_embeddings(db, RemnantConfig(agent_id="a"), emb)
    assert result == {"candidates": 1, "repaired": 0}
    assert db.get_memory_embedding(mid) == []


@pytest.mark.parametrize("vector", [[1.0], [float("nan"), 1.0], [0.0, 0.0],
                                    [float("inf"), 1.0]])
def test_invalid_remote_vectors_are_not_cached(db, vector):
    emb = Embedder(db, RemnantConfig(embed_dim=2))
    emb._client.close()
    emb._client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"embedding": vector})
        if all(abs(x) != float("inf") and x == x for x in vector)
        else httpx.Response(200, content=json.dumps({"embedding": vector}))))
    try:
        assert emb.embed("A fact") is None
        assert emb.embed_query("A query") is None
        with db.read() as cur:
            assert cur.execute("SELECT count(*) FROM embedding_cache").fetchone()[0] == 0
    finally:
        emb.close()


def test_query_cache_is_bounded_and_never_persistent(db, monkeypatch):
    emb = Embedder(db, RemnantConfig(embed_dim=2))
    calls = []
    monkeypatch.setattr(emb, "_embed_remote", lambda text, **k: calls.append(text) or [1., 0.])
    try:
        for i in range(140):
            emb.embed_query(f"query {i}")
        assert len(emb._query_cache) == 128
        emb.embed_query("query 139")
        assert len(calls) == 140
        emb._model = "changed"
        emb.embed_query("query 139")
        assert len(calls) == 141
        with db.read() as cur:
            assert cur.execute("SELECT count(*) FROM embedding_cache").fetchone()[0] == 0
    finally:
        emb.close()


def test_complete_scan_filters_each_score_model_dimension_and_scope(db):
    emb = Vectors()
    target = db.insert_memory(content="Old important fact", agent="a", embedding=[1., 0.],
                              embed_model="test:latest")
    # Identical recent content and vectors avoid conflating relevance with age.
    for i in range(5001):
        db.insert_memory(content=f"Distractor {i}", agent="a", embedding=[0., 1.],
                         embed_model="test")
    for kwargs in ({"agent": "b"}, {"embed_model": "other"}, {"embedding": [1., 0., 0.]}):
        args = dict(content="Out of scope", agent="a", embedding=[1., 0.], embed_model="test")
        db.insert_memory(**(args | kwargs))
    cfg = RemnantConfig(embed_model="test", min_semantic_score=.3)
    rows = search(db, cfg, "No lexical match", agent_id="a", strategy="semantic", embedder=emb)
    assert [row["id"] for row in rows] == [target]


def test_queued_queries_use_fresh_vectors_and_only_delivery_suppresses(db):
    p = provider(db, prefetch_cache_ttl_s=0)
    db.insert_memory(content="Morgan uses Linux", agent="a", embedding=[1., 0.], embed_model="test")
    p.queue_prefetch("Morgan software", session_id="s")
    p.queue_prefetch("Morgan operating system", session_id="s")
    assert p._embedder.calls == ["Morgan software", "Morgan operating system"]
    assert p._last_injected_hash == {}
    assert p.prefetch("Morgan operating system", session_id="s")
    assert p.prefetch("Morgan operating system", session_id="s") == ""


def test_pending_prefetch_work_is_coalesced_and_bounded(db):
    p = provider(db, prefetch_cache_max_entries=2)
    submitted = []
    p._prefetch_executor = SimpleNamespace(submit=submitted.append)
    for i in range(10):
        p.queue_prefetch(f"Morgan query {i}", session_id="s")
    p.queue_prefetch("Morgan q", session_id="other")
    p.queue_prefetch("Morgan q", session_id="third")
    assert len(submitted) == 2


def test_external_committed_evidence_invalidates_queued_context(db):
    p = provider(db)
    mid = db.insert_memory(content="Morgan uses Linux", agent="a", embedding=[1., 0.],
                           embed_model="test")
    p.queue_prefetch("Morgan software", session_id="s")
    other = open_db(db.path)
    try:
        other.update_memory_content(mid, content="Morgan uses FreeBSD")
    finally:
        other.close()
    context = p.prefetch("Morgan software", session_id="s")
    assert "Morgan uses Linux" not in context


def test_foreground_deadline_bounds_shared_lock_and_measures_wait(db):
    p = provider(db, injection_prefetch_deadline_ms=60)
    locked, release = threading.Event(), threading.Event()
    def hold():
        with db.read():
            locked.set()
            release.wait(2)
    thread = threading.Thread(target=hold)
    thread.start()
    assert locked.wait(1)
    try:
        start = time.monotonic()
        assert p.prefetch("Morgan software") == ""
        elapsed = (time.monotonic() - start) * 1000
        assert 40 < elapsed < 200
    finally:
        release.set()
        thread.join()
    db.flush_diagnostics()
    with db.read() as cur:
        row = cur.execute("SELECT elapsed_ms,reason FROM prefetch_stats").fetchone()
    assert abs(row["elapsed_ms"] - elapsed) < 20
    assert row["reason"] == "deadline"


def test_telemetry_does_not_wait_for_competing_writer(db):
    p = provider(db)
    writer = sqlite3.connect(db.path)
    writer.execute("BEGIN IMMEDIATE")
    try:
        started = time.monotonic()
        assert p.prefetch("hi") == ""
        db.flush_diagnostics()
        assert time.monotonic() - started < .2
        assert len(db._diagnostics) == 1
    finally:
        writer.rollback()
        writer.close()
    db.flush_diagnostics()
    assert not db._diagnostics


def test_retention_removes_only_reproducible_cache_data(db):
    mid = db.insert_memory(content="Keep the source evidence", agent="a")
    db.insert_turn(session_id="s", agent_id="a", user_text="Keep the raw turn", assistant_text="")
    for i in range(5):
        db.put_cached_embedding("test", str(i), [1., 0.])
    db.compact_caches(max_entries=2)
    with db.read() as cur:
        assert cur.execute("SELECT count(*) FROM embedding_cache").fetchone()[0] == 2
        assert cur.execute("SELECT count(*) FROM turns").fetchone()[0] == 1
    assert db.get_memory(mid)["content"] == "Keep the source evidence"
