"""Phase 2 tests: semantic search, RRF fusion, proactive prefetch, reflection.

Run without a live Ollama: the Embedder is monkeypatched with deterministic
vectors and the reflect LLM call is stubbed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from remnant import RemnantMemoryProvider
from remnant.config import RemnantConfig
from remnant.db import default_db_path, open_db
from remnant.embed import Embedder, cosine
from remnant.ingest import store_memory
from remnant.prefetch import _expand_queries, _needs_memory
from remnant.search import _rrf_fuse
from remnant.search import search as hybrid_search

# --- shared fakes (mirror test_phase1) -------------------------------------


def _hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fake_embed(db, config, dim=8):
    """Deterministic embedder whose vectors encode semantic similarity.

    Facts containing the same key noun produce overlapping vectors so cosine
    ranking is meaningful and stable.
    """
    emb = Embedder.__new__(Embedder)
    emb._db = db
    emb._model = config.embed_model
    emb._url = config.embed_url
    emb._timeout = config.embed_timeout
    emb._client = None

    def _seed(text: str) -> list[float]:
        words = [w.lower() for w in text.strip().split()]
        vec = [0.0] * dim
        for w in words:
            h = abs(hash(w)) % dim
            vec[h] += 1.0
        # Normalize so cosine is well-defined.
        n = sum(v * v for v in vec) ** 0.5
        if n:
            vec = [v / n for v in vec]
        return vec

    def embed(text: str) -> list[float]:
        cached = db.get_cached_embedding(emb._model, _hash(text))
        if cached is not None:
            return cached
        vec = _seed(text)
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


@pytest.fixture()
def provider(hermes_home: Path) -> RemnantMemoryProvider:
    p = RemnantMemoryProvider()
    p.initialize(session_id="p2-session", hermes_home=str(hermes_home))
    yield p
    p.shutdown()


@pytest.fixture(autouse=True)
def patch_embed(monkeypatch):
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
    from remnant import extract as extract_mod

    monkeypatch.setattr(extract_mod.ExtractionWorker, "_extract", lambda self, u, a: [])
    yield


def _seed_memories(db, emb, cfg, items, agent_id="default"):
    for fact, entity, vis in items:
        store_memory(
            db, emb, cfg, fact=fact, entity=entity,
            session_id="seed", agent_id=agent_id, visibility=vis,
        )


# --- intent classifier ------------------------------------------------------


@pytest.mark.parametrize(
    "query,expected",
    [
        ("hey", False),
        ("how are you", False),
        ("thanks!", False),
        ("ok", False),
        ("what did we decide about the homelab", True),
        ("remember Sven's preference", True),
        ("status of the proxmox node", True),
        ("who is Alice", True),
        ("Sven", True),  # short proper-noun lookup
        ("", False),
    ],
)
def test_needs_memory_classifier(query, expected):
    assert _needs_memory(query) is expected


def test_expand_queries_proper_nouns():
    terms = _expand_queries("What did we decide about Proxmox node alpha")
    assert any("Proxmox" in t for t in terms)
    assert len(terms) <= 3
    assert len(terms) >= 1


def test_expand_queries_empty():
    assert _expand_queries("") == []


# --- semantic search --------------------------------------------------------


def test_semantic_search_ranks_relevant(hermes_home: Path):
    db = open_db(default_db_path())
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        _seed_memories(db, emb, cfg, [
            ("Sven prefers dark mode for all editors", "Sven", "private"),
            ("The homelab runs Proxmox on four nodes", "homelab", "private"),
            ("Alice is Sven's sister", "Alice", "private"),
        ])
        results = hybrid_search(
            db, cfg, "editor dark mode preference",
            agent_id="default", limit=10, strategy="semantic", embedder=emb,
        )
        assert results, "semantic search should return results"
        top = results[0]
        assert "dark mode" in top["content"].lower()
        # Score is cosine similarity in [-1, 1].
        assert top["score"] >= 0.0
    finally:
        db.close()


def test_semantic_search_uses_bm25_prefilter(hermes_home: Path):
    """Semantic search must not load embeddings for the whole DB.

    With many memories, only the BM25 candidate set (<= SEMANTIC_CANDIDATE_LIMIT)
    should have embeddings loaded. We verify by checking results are bounded
    and that a memory with no FTS token match but a relevant embedding is still
    reachable via the recency fallback (or BM25). Here we just assert no crash
    and a bounded result count.
    """
    db = open_db(default_db_path())
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        _seed_memories(db, emb, cfg, [
            (f"fact number {i} about topic {i % 5}", f"e{i}", "private")
            for i in range(60)
        ])
        results = hybrid_search(
            db, cfg, "topic 0", agent_id="default",
            limit=10, strategy="semantic", embedder=emb,
        )
        assert len(results) <= 10
    finally:
        db.close()


# --- RRF fusion -------------------------------------------------------------


def test_rrf_fusion_merges_ranks():
    kw = [
        {"id": "a", "content": "alpha", "visibility": "private", "agent_id": "x"},
        {"id": "b", "content": "beta", "visibility": "private", "agent_id": "x"},
        {"id": "c", "content": "gamma", "visibility": "private", "agent_id": "x"},
    ]
    sem = [
        {"id": "b", "content": "beta", "visibility": "private", "agent_id": "x"},
        {"id": "a", "content": "alpha", "visibility": "private", "agent_id": "x"},
        {"id": "d", "content": "delta", "visibility": "private", "agent_id": "x"},
    ]
    fused = _rrf_fuse(kw, sem)
    ids = [r["id"] for r in fused]
    # 'a' and 'b' appear in both lists => higher fused score than c/d.
    assert ids[0] in {"a", "b"}
    assert ids[1] in {"a", "b"}
    # RRF score uses k=60: score = 1/(60+rank_kw) + 1/(60+rank_sem).
    a_score = next(r["score"] for r in fused if r["id"] == "a")
    c_score = next(r["score"] for r in fused if r["id"] == "c")
    assert a_score > c_score


def test_auto_strategy_fuses(hermes_home: Path):
    db = open_db(default_db_path())
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        _seed_memories(db, emb, cfg, [
            ("Sven prefers dark mode", "Sven", "private"),
            ("The homelab runs Proxmox", "homelab", "private"),
        ])
        results = hybrid_search(
            db, cfg, "dark mode", agent_id="default",
            limit=10, strategy="auto", embedder=emb,
        )
        assert results
        assert any("dark mode" in r["content"].lower() for r in results)
    finally:
        db.close()


# --- prefetch ---------------------------------------------------------------


def test_prefetch_skips_greetings(provider: RemnantMemoryProvider):
    assert provider.prefetch("hey", session_id="s") == {}
    assert provider.prefetch("how are you", session_id="s") == {}


def test_prefetch_injects_relevant_facts(provider: RemnantMemoryProvider):
    # Seed a fact via the tool path.
    provider.handle_tool_call(
        "memory_store",
        {"fact": "Sven prefers dark mode for editors", "entity": "Sven"},
        session_id="seed",
    )
    provider.handle_tool_call(
        "memory_store",
        {"fact": "The homelab runs Proxmox on four nodes", "entity": "homelab"},
        session_id="seed",
    )
    res = provider.prefetch("what did Sven decide about dark mode", session_id="ask")
    assert res, "prefetch should inject relevant context"
    assert "context" in res
    assert "dark mode" in res["context"].lower()
    assert res["token_estimate"] > 0
    assert "memories" in res and res["memories"]


def test_prefetch_within_deadline(provider: RemnantMemoryProvider):
    provider.handle_tool_call(
        "memory_store",
        {"fact": "Sven prefers dark mode for editors", "entity": "Sven"},
        session_id="seed",
    )
    t0 = time.perf_counter()
    res = provider.prefetch("remember Sven dark mode preference", session_id="t")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < provider._config.injection_prefetch_deadline_ms
    assert res  # and it injected something


def test_prefetch_token_budget_enforced(provider: RemnantMemoryProvider):
    # Lower the budget drastically so only ~1 short line fits.
    provider._config.injection_token_budget = 30
    for i in range(20):
        provider.handle_tool_call(
            "memory_store",
            {"fact": f"Sven preference number {i} is about topic {i}", "entity": "Sven"},
            session_id="seed",
        )
    res = provider.prefetch("what are Sven preferences", session_id="bud")
    if res:
        assert res["token_estimate"] <= 30


def test_prefetch_diff_based_dedup(provider: RemnantMemoryProvider):
    provider.handle_tool_call(
        "memory_store",
        {"fact": "Sven prefers dark mode for editors", "entity": "Sven"},
        session_id="seed",
    )
    q = "remember Sven dark mode preference"
    r1 = provider.prefetch(q, session_id="dup")
    assert r1, "first call should inject"
    h1 = r1["hash"]
    # Same query + same memories => same context hash => suppressed.
    r2 = provider.prefetch(q, session_id="dup")
    assert r2 == {}, "unchanged context should be suppressed (diff-based dedup)"
    assert provider._last_injected_hash.get("dup") == h1


def test_prefetch_dedup_against_messages(provider: RemnantMemoryProvider):
    provider.handle_tool_call(
        "memory_store",
        {"fact": "Sven prefers dark mode for editors", "entity": "Sven"},
        session_id="seed",
    )
    # The fact text is already in the conversation => should be deduped away.
    messages = [
        {"role": "user", "content": "Sven prefers dark mode for editors"},
        {"role": "assistant", "content": "got it"},
    ]
    res = provider.prefetch(
        "remember Sven dark mode preference",
        session_id="msgdedup",
        messages=messages,  # type: ignore[arg-type]
    )
    if res:
        for m in res["memories"]:
            assert "dark mode" not in m["content"].lower() or m["content"] not in [
                msg["content"] for msg in messages
            ]


def test_prefetch_disabled(provider: RemnantMemoryProvider):
    provider._config.prefetch_enabled = False
    assert provider.prefetch("what did we decide", session_id="off") == {}


def test_queue_prefetch_does_not_crash(provider: RemnantMemoryProvider):
    provider.queue_prefetch("next turn query")
    assert provider._prefetch_queue  # queued something


# --- memory_reflect ---------------------------------------------------------


def test_memory_reflect_tool_mock(provider: RemnantMemoryProvider, monkeypatch):
    provider.handle_tool_call(
        "memory_store",
        {"fact": "Sven prefers dark mode for editors", "entity": "Sven"},
        session_id="seed",
    )
    provider.handle_tool_call(
        "memory_store",
        {"fact": "The homelab runs Proxmox on four nodes", "entity": "homelab"},
        session_id="seed",
    )
    # Stub the reflect LLM call so no network is used.
    from remnant import reflect as reflect_mod

    def fake_call_llm(config, user_content):
        return "Sven prefers dark mode; the homelab runs Proxmox on four nodes."

    monkeypatch.setattr(reflect_mod, "_call_llm", fake_call_llm)
    res = provider.handle_tool_call(
        "memory_reflect",
        {"question": "What do we know about Sven and the homelab?"},
        session_id="reflect",
    )
    parsed = json.loads(res)
    assert "synthesis" in parsed
    assert parsed["synthesis"]
    assert parsed["count"] >= 1
    assert isinstance(parsed["source_ids"], list)
    assert parsed["source_ids"]


def test_memory_reflect_empty_question(provider: RemnantMemoryProvider):
    res = provider.handle_tool_call(
        "memory_reflect", {"question": ""}, session_id="reflect"
    )
    assert "error" in json.loads(res)


def test_memory_reflect_no_memories(provider: RemnantMemoryProvider, monkeypatch):
    from remnant import reflect as reflect_mod

    monkeypatch.setattr(reflect_mod, "_call_llm", lambda c, u: "nothing")
    res = provider.handle_tool_call(
        "memory_reflect",
        {"question": "anything about unrelated topic zzz"},
        session_id="reflect",
    )
    # With no matching memories, hybrid_search returns [] and reflect short-circuits.
    parsed = json.loads(res)
    assert parsed["count"] == 0
    assert parsed["synthesis"] == ""


# --- cosine sanity (re-confirm for semantic) --------------------------------


def test_cosine_semantic_helper():
    assert cosine([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


# --- config defaults --------------------------------------------------------


def test_phase2_config_defaults():
    cfg = RemnantConfig()
    assert cfg.injection_token_budget == 2000
    assert cfg.injection_prefetch_deadline_ms == 500
    assert cfg.prefetch_enabled is True
    assert cfg.reflect_model == "gemma4:12b"
    assert cfg.reflect_url == cfg.extract_url
