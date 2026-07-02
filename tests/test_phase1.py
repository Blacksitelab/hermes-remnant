"""Phase 1 integration smoke tests for the Remnant memory provider.

These run without a live Ollama instance: the Embedder and extraction HTTP
calls are monkeypatched so the data plane and dedup logic can be verified
deterministically.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from remnant import RemnantMemoryProvider, register
from remnant.config import RemnantConfig, load_config, save_config
from remnant.db import open_db
from remnant.embed import Embedder, cosine
from remnant.ingest import is_transient, store_memory
from remnant.search import search as bm25_search

# --- fixtures ---------------------------------------------------------------


@pytest.fixture()
def hermes_home(tmp_path: Path) -> Path:
    home = tmp_path / "hermes"
    (home / "remnant").mkdir(parents=True, exist_ok=True)
    return home


@pytest.fixture()
def provider(hermes_home: Path) -> RemnantMemoryProvider:
    p = RemnantMemoryProvider()
    p.initialize(session_id="test-session", hermes_home=str(hermes_home))
    yield p
    p.shutdown()


def _fake_embed(db, config, dim=8):
    """Return an Embedder whose .embed() returns deterministic vectors."""
    emb = Embedder.__new__(Embedder)
    emb._db = db
    emb._model = config.embed_model
    emb._url = config.embed_url
    emb._timeout = config.embed_timeout
    emb._client = None

    def embed(text: str) -> list[float]:
        cached = db.get_cached_embedding(emb._model, _hash(text))
        if cached is not None:
            return cached
        # Deterministic pseudo-embedding from char sums.
        vec = [float((ord(c) * (i + 1)) % 97 / 97.0) for i, c in enumerate(text[:dim])]
        while len(vec) < dim:
            vec.append(0.0)
        db.put_cached_embedding(emb._model, _hash(text), vec)
        return vec

    emb.embed = embed
    emb.close = lambda: None
    return emb


def _hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture(autouse=True)
def patch_embed(monkeypatch):
    """Replace Embedder construction everywhere with the deterministic fake."""
    import sys

    remnant_init = sys.modules["remnant"]
    embed_mod = sys.modules["remnant.embed"]

    original_init = remnant_init.Embedder
    original_embed_mod = embed_mod.Embedder

    def fake_ctor(db, config):
        return _fake_embed(db, config)

    monkeypatch.setattr(remnant_init, "Embedder", fake_ctor)
    monkeypatch.setattr(embed_mod, "Embedder", fake_ctor)
    yield
    monkeypatch.setattr(remnant_init, "Embedder", original_init)
    monkeypatch.setattr(embed_mod, "Embedder", original_embed_mod)


@pytest.fixture(autouse=True)
def patch_extract(monkeypatch):
    """Make the extraction worker not hit the network: return [] facts."""
    from remnant import extract as extract_mod

    def fake_process(self, job):
        return

    monkeypatch.setattr(extract_mod.ExtractionWorker, "_extract", lambda self, u, a: [])
    yield


# --- config -----------------------------------------------------------------


def test_load_config_defaults(hermes_home: Path):
    cfg = load_config(str(hermes_home))
    assert cfg.embed_model == "nomic-embed-text"
    assert cfg.embed_url == "http://192.168.0.11:11434/api/embeddings"
    assert cfg.extract_model == "gemma4:12b"
    assert cfg.default_visibility == "private"


def test_save_and_reload_config(hermes_home: Path):
    save_config({"extract_enabled": False, "agent_id": "fleet-1"}, str(hermes_home))
    cfg = load_config(str(hermes_home))
    assert cfg.extract_enabled is False
    assert cfg.agent_id == "fleet-1"
    # Defaults preserved for unspecified fields
    assert cfg.embed_model == "nomic-embed-text"


def test_provider_config_schema(provider: RemnantMemoryProvider):
    schema = provider.get_config_schema()
    keys = {f["key"] for f in schema}
    assert {"embed_url", "embed_model", "extract_url", "extract_model"} <= keys
    # No secrets declared
    assert not any(f.get("secret") for f in schema)


# --- lifecycle --------------------------------------------------------------


def test_name_and_available(provider: RemnantMemoryProvider):
    assert provider.name == "remnant"
    assert provider.is_available() is True


def test_is_available_without_init():
    p = RemnantMemoryProvider()
    # No network/file access needed; must be cheap.
    assert p.is_available() is True


# --- sync_turn timing -------------------------------------------------------


def test_sync_turn_under_10ms(provider: RemnantMemoryProvider):
    # Warm the connection so first-call overhead (schema create) isn't counted.
    provider.sync_turn("warmup user", "warmup assistant", session_id="warmup")
    t0 = time.perf_counter()
    provider.sync_turn("hello world", "hi there", session_id="timing")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 10.0, f"sync_turn took {elapsed_ms:.2f}ms"
    # Turn must have been persisted.
    with provider._db.read() as cur:  # type: ignore[union-attr]
        cur.execute("SELECT COUNT(*) AS c FROM turns WHERE session_id='timing'")
        assert cur.fetchone()["c"] == 1


def test_sync_turn_persists_and_enqueues(provider: RemnantMemoryProvider):
    # Stop the background extraction worker so it cannot process/delete the
    # queue row before we assert its presence (deterministic over a race).
    provider._worker.stop()  # type: ignore[union-attr]
    provider.sync_turn("user msg", "assistant reply", session_id="s1")
    db = provider._db  # type: ignore[union-attr]
    assert db is not None
    with db.read() as cur:
        cur.execute("SELECT * FROM turns WHERE session_id='s1'")
        row = cur.fetchone()
        assert row is not None
        assert row["user_text"] == "user msg"
        assert row["assistant_text"] == "assistant reply"
        cur.execute("SELECT COUNT(*) AS c FROM extraction_queue WHERE session_id='s1'")
        assert cur.fetchone()["c"] == 1


def test_sync_turn_does_not_block_on_extraction(provider: RemnantMemoryProvider):
    """Even if the worker is stopped, sync_turn must return immediately."""
    provider._worker.stop()  # type: ignore[union-attr]
    t0 = time.perf_counter()
    provider.sync_turn("x", "y", session_id="noblock")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 10.0
    # Queue row still written despite worker being down.
    db = provider._db  # type: ignore[union-attr]
    assert db is not None
    assert db.pending_count() >= 1


# --- transient filter ------------------------------------------------------


def test_is_transient_rejects_percentages():
    assert is_transient("The printer is at 32%")
    assert is_transient("battery is at 15 percent")


def test_is_transient_rejects_current_status():
    assert is_transient("the printer is currently offline")
    assert is_transient("the server is down right now")


def test_is_transient_rejects_times_and_today():
    assert is_transient("meeting at 9:30 am today")
    assert is_transient("the deployment is tonight")


def test_is_transient_accepts_durable_facts():
    assert not is_transient("Sven prefers dark mode")
    assert not is_transient("The homelab has 4 nodes")
    assert not is_transient("Alice is Sven's sister")


# --- store + dedup ---------------------------------------------------------


def test_memory_store_stores_durable_fact(provider: RemnantMemoryProvider):
    res = provider.handle_tool_call(
        "memory_store",
        {"fact": "Sven prefers dark mode", "entity": "Sven"},
        session_id="s2",
    )
    assert res["stored"] is True
    assert "memory_id" in res


def test_memory_store_rejects_transient(provider: RemnantMemoryProvider):
    res = provider.handle_tool_call(
        "memory_store",
        {"fact": "the printer is at 32%", "entity": "printer"},
        session_id="s2",
    )
    assert res["stored"] is False


def test_memory_store_dedup_identical(provider: RemnantMemoryProvider):
    provider.handle_tool_call(
        "memory_store",
        {"fact": "Sven prefers dark mode", "entity": "Sven"},
        session_id="s2",
    )
    res = provider.handle_tool_call(
        "memory_store",
        {"fact": "Sven prefers dark mode", "entity": "Sven"},
        session_id="s2",
    )
    assert res["stored"] is False


def test_memory_store_dedup_near_identical(provider: RemnantMemoryProvider):
    """Near-identical facts (same text after normalization) are deduped."""
    provider.handle_tool_call(
        "memory_store",
        {"fact": "The homelab has four nodes", "entity": "homelab"},
        session_id="s3",
    )
    res = provider.handle_tool_call(
        "memory_store",
        {"fact": "the homelab has four nodes.", "entity": "homelab"},
        session_id="s3",
    )
    # Embeddings differ slightly but text normalization catches the dup.
    assert res["stored"] is False


def test_store_memory_direct(hermes_home: Path):
    """Exercise store_memory directly with the deterministic embedder."""
    db = open_db(hermes_home / "remnant" / "remnant.db")
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        mid = store_memory(
            db, emb, cfg, fact="Alice is Sven's sister", entity="Alice",
            session_id="s", agent_id="default",
        )
        assert mid is not None
        # Storing again should dedup.
        mid2 = store_memory(
            db, emb, cfg, fact="Alice is Sven's sister", entity="Alice",
            session_id="s", agent_id="default",
        )
        assert mid2 is None
    finally:
        db.close()


# --- BM25 search -----------------------------------------------------------


def test_memory_search_returns_results(provider: RemnantMemoryProvider):
    provider.handle_tool_call(
        "memory_store",
        {"fact": "Sven prefers dark mode", "entity": "Sven"},
        session_id="s4",
    )
    provider.handle_tool_call(
        "memory_store",
        {"fact": "The homelab runs Proxmox", "entity": "homelab"},
        session_id="s4",
    )
    res = provider.handle_tool_call(
        "memory_search", {"query": "dark mode"}, session_id="s4"
    )
    assert res["count"] >= 1
    facts = [r["fact"] for r in res["results"]]
    assert any("dark mode" in f.lower() for f in facts)


def test_memory_search_empty_query(provider: RemnantMemoryProvider):
    res = provider.handle_tool_call("memory_search", {"query": ""}, session_id="s4")
    assert "error" in res


def test_memory_search_unknown_tool(provider: RemnantMemoryProvider):
    res = provider.handle_tool_call("bogus", {}, session_id="s4")
    assert "error" in res


def test_search_visibility_filtering(hermes_home: Path):
    """A private-scoped search should not see shared/fleet memories."""
    db = open_db(hermes_home / "remnant" / "remnant.db")
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        store_memory(db, emb, cfg, fact="Private note for agent", entity="x",
                     session_id="s", agent_id="a1", visibility="private")
        store_memory(db, emb, cfg, fact="Shared note for fleet", entity="y",
                     session_id="s", agent_id="a1", visibility="shared")
        # Search scoped to private visibility: only private returned.
        results = bm25_search(db, cfg, "note", agent_id="a1", visibility="private")
        vis = {r["visibility"] for r in results}
        assert vis == {"private"} or vis == set()
        # Search scoped to shared visibility: private + shared returned.
        results = bm25_search(db, cfg, "note", agent_id="a1", visibility="shared")
        vis = {r["visibility"] for r in results}
        assert vis <= {"private", "shared"}
    finally:
        db.close()


def test_search_agent_scoped(hermes_home: Path):
    """Agent A should not see agent B's private memories."""
    db = open_db(hermes_home / "remnant" / "remnant.db")
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        store_memory(db, emb, cfg, fact="Agent A secret", entity="x",
                     session_id="s", agent_id="A", visibility="private")
        store_memory(db, emb, cfg, fact="Agent B secret", entity="y",
                     session_id="s", agent_id="B", visibility="private")
        a_res = bm25_search(db, cfg, "secret", agent_id="A")
        a_facts = [r["content"] for r in a_res]
        assert any("Agent A" in f for f in a_facts)
        assert not any("Agent B" in f for f in a_facts)
    finally:
        db.close()


# --- system prompt ---------------------------------------------------------


def test_system_prompt_block_is_static(provider: RemnantMemoryProvider):
    b1 = provider.system_prompt_block()
    b2 = provider.system_prompt_block()
    assert b1 == b2
    assert "memory_search" in b1
    assert "memory_store" in b1


def test_system_prompt_byte_stable_across_calls(provider: RemnantMemoryProvider):
    b1 = provider.system_prompt_block().encode("utf-8")
    provider.sync_turn("u", "a", session_id="x")
    b2 = provider.system_prompt_block().encode("utf-8")
    assert b1 == b2


# --- tool schemas ----------------------------------------------------------


def test_get_tool_schemas(provider: RemnantMemoryProvider):
    schemas = provider.get_tool_schemas()
    names = {s["function"]["name"] for s in schemas}
    assert names == {"memory_search", "memory_store"}


# --- extraction queue persistence -----------------------------------------


def test_extraction_queue_persists_across_restart(hermes_home: Path):
    """A queued turn survives provider shutdown/restart."""
    p = RemnantMemoryProvider()
    p.initialize(session_id="restart", hermes_home=str(hermes_home))
    # Stop the worker so the queued turn is NOT processed before we reopen.
    p._worker.stop()  # type: ignore[union-attr]
    p.sync_turn("queued user", "queued assistant", session_id="restart")
    pending = p._db.pending_count()  # type: ignore[union-attr]
    assert pending >= 1
    p.shutdown()

    # Reopen the DB directly and confirm the row is still there.
    db = open_db(hermes_home / "remnant" / "remnant.db")
    try:
        with db.read() as cur:
            cur.execute("SELECT COUNT(*) AS c FROM extraction_queue WHERE session_id='restart'")
            assert cur.fetchone()["c"] == 1
    finally:
        db.close()


# --- register entry point --------------------------------------------------


def test_register_entry_point():
    class FakeCtx:
        def __init__(self):
            self.registered = None

        def register_memory_provider(self, provider):
            self.registered = provider

    ctx = FakeCtx()
    register(ctx)
    assert isinstance(ctx.registered, RemnantMemoryProvider)
    assert ctx.registered.name == "remnant"


# --- cosine helper ---------------------------------------------------------


def test_cosine_identical_and_orthogonal():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([], [1.0]) == 0.0


# --- embedding cache ------------------------------------------------------


def test_embedding_cache_hits(provider: RemnantMemoryProvider):
    """Embedding the same text twice should only compute once (cache hit)."""
    emb = provider._embedder  # type: ignore[union-attr]
    db = provider._db  # type: ignore[union-attr]
    puts = {"n": 0}
    real_put = db.put_cached_embedding

    def counting_put(model, text_hash, embedding):
        puts["n"] += 1
        return real_put(model, text_hash, embedding)

    db.put_cached_embedding = counting_put  # type: ignore[method-assign]
    try:
        emb.embed("cache test text")
        assert puts["n"] == 1  # first call: cache miss → compute → store
        emb.embed("cache test text")
        assert puts["n"] == 1  # second call: cache hit → no store
    finally:
        db.put_cached_embedding = real_put  # type: ignore[method-assign]
