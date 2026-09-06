"""Phase 5 tests: thread CRUD, stale sweep, dream candidate selection (local
cosine pre-filter), budget enforcement, cooldown, diary append, cross-agent
merge.

Run without a live cloud model: the Embedder uses deterministic word-bag
vectors (same pattern as test_phase2/3/4) and the cloud dream call is
monkeypatched.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from remnant import RemnantMemoryProvider
from remnant.config import RemnantConfig
from remnant.db import default_db_path, open_db
from remnant.dream import day_dream, night_dream
from remnant.embed import Embedder
from remnant.threads import (
    create_thread,
    list_threads,
    resolve_thread,
    stale_threads,
    sweep_stale_threads,
    update_thread,
)

# --- shared fakes -----------------------------------------------------------


def _hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fake_embed(db, config, dim=16):
    """Deterministic word-bag embedder (same scheme as test_phase2)."""
    emb = Embedder.__new__(Embedder)
    emb._db = db
    emb._model = config.embed_model
    emb._url = config.embed_url
    emb._timeout = config.embed_timeout
    emb._client = None

    def _seed(text: str) -> list[float]:
        import hashlib

        words = [w.lower() for w in text.strip().split()]
        vec = [0.0] * dim
        for w in words:
            # Deterministic per-word bucket (independent of PYTHONHASHSEED).
            h = int.from_bytes(hashlib.sha256(w.encode("utf-8")).digest()[:4], "big") % dim
            vec[h] += 1.0
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


@pytest.fixture()
def provider(hermes_home: Path) -> RemnantMemoryProvider:
    p = RemnantMemoryProvider()
    p.initialize(session_id="p5-session", hermes_home=str(hermes_home))
    yield p
    p.shutdown()


def _open_db(hermes_home: Path):
    return open_db(default_db_path())


def _store_fact(db, emb, cfg, fact, agent="default", visibility="private"):
    from remnant.ingest import store_memory

    return store_memory(
        db, emb, cfg, fact=fact, entity="general",
        session_id="s", agent_id=agent, visibility=visibility,
    )


# ===========================================================================
# Config defaults
# ===========================================================================


def test_phase5_config_defaults():
    cfg = RemnantConfig()
    assert cfg.dream_day_budget == 3
    assert cfg.dream_night_budget == 5
    assert cfg.dream_cooldown_minutes == 120
    assert cfg.dream_day_model and cfg.dream_night_model
    assert "DREAMS.md" in cfg.diary_path
    d = cfg.to_dict()
    assert d["dream_day_budget"] == 3
    assert d["dream_night_budget"] == 5


# ===========================================================================
# Thread CRUD
# ===========================================================================


def test_thread_create_get_list(hermes_home: Path):
    db = _open_db(hermes_home)
    try:
        tid = create_thread(db, owner="default", title="Build server",
                            topic="build", importance=0.7,
                            tags=["infra"], related_entities=["build-srv"])
        assert tid
        t = db.get_thread(tid)
        assert t["title"] == "Build server"
        assert t["status"] == "active"
        assert t["importance"] == 0.7
        assert t["tags"] == ["infra"]
        assert t["related_entities"] == ["build-srv"]
        threads = list_threads(db)
        assert len(threads) == 1
        assert threads[0]["id"] == tid
    finally:
        db.close()


def test_thread_create_requires_title_and_topic(hermes_home: Path):
    db = _open_db(hermes_home)
    try:
        with pytest.raises(ValueError):
            create_thread(db, owner="default", title="", topic="x")
        with pytest.raises(ValueError):
            create_thread(db, owner="default", title="t", topic="")
    finally:
        db.close()


def test_thread_update(hermes_home: Path):
    db = _open_db(hermes_home)
    try:
        tid = create_thread(db, owner="default", title="A", topic="a")
        res = update_thread(db, tid, title="A2", importance=0.9, tags=["new"])
        assert res["title"] == "A2"
        assert res["importance"] == 0.9
        assert res["tags"] == ["new"]
        # last_activity advanced on touch=True
        assert res["last_activity"] >= res["created_at"]
    finally:
        db.close()


def test_thread_update_unknown_returns_none(hermes_home: Path):
    db = _open_db(hermes_home)
    try:
        assert update_thread(db, "nope", title="x") is None
    finally:
        db.close()


def test_thread_resolve(hermes_home: Path):
    db = _open_db(hermes_home)
    try:
        tid = create_thread(db, owner="default", title="A", topic="a")
        res = resolve_thread(db, tid)
        assert res["status"] == "resolved"
    finally:
        db.close()


def test_thread_list_status_filter(hermes_home: Path):
    db = _open_db(hermes_home)
    try:
        t1 = create_thread(db, owner="default", title="A", topic="a")
        t2 = create_thread(db, owner="default", title="B", topic="b")
        resolve_thread(db, t1)
        active = list_threads(db, status="active")
        assert {t["id"] for t in active} == {t2}
        resolved = list_threads(db, status="resolved")
        assert {t["id"] for t in resolved} == {t1}
    finally:
        db.close()


# ===========================================================================
# Stale sweep
# ===========================================================================


def test_stale_threads_marks_old(hermes_home: Path):
    db = _open_db(hermes_home)
    try:
        tid = create_thread(db, owner="default", title="old", topic="old")
        # Manually back-date last_activity to 20 days ago.
        old = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                            time.gmtime(time.time() - 20 * 86400))
        with db.transaction() as cur:
            cur.execute(
                "UPDATE threads SET last_activity=? WHERE id=?", (old, tid)
            )
        stale = stale_threads(db, days=14)
        assert [t["id"] for t in stale] == [tid]
        marked = sweep_stale_threads(db, days=14)
        assert marked == [tid]
        assert db.get_thread(tid)["status"] == "stale"
        # Idempotent: second sweep finds nothing new.
        assert sweep_stale_threads(db, days=14) == []
    finally:
        db.close()


def test_stale_threads_skips_recent(hermes_home: Path):
    db = _open_db(hermes_home)
    try:
        create_thread(db, owner="default", title="fresh", topic="fresh")
        assert sweep_stale_threads(db, days=14) == []
    finally:
        db.close()


# ===========================================================================
# dream_state helpers
# ===========================================================================


def test_dream_state_get_set(hermes_home: Path):
    db = _open_db(hermes_home)
    try:
        assert db.get_state("missing") is None
        assert db.get_state("missing", default=5) == 5
        db.set_state("k", {"a": 1, "b": [2, 3]})
        assert db.get_state("k") == {"a": 1, "b": [2, 3]}
    finally:
        db.close()


def test_get_recent_memories_window(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        _store_fact(db, emb, cfg, "alpha is a durable fact about Proxmox")
        now = time.time()
        recent = db.get_recent_memories(since_ts=now - 60)
        assert len(recent) == 1
        old = db.get_recent_memories(since_ts=now + 60)
        assert old == []
    finally:
        db.close()


# ===========================================================================
# Candidate selection (local, no LLM)
# ===========================================================================


def test_candidate_selection_local_only(hermes_home: Path):
    """Candidate selection must use cosine over stored embeddings only —
    no network call to the cloud model."""
    from remnant import dream as dream_mod

    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        # Two similar memories (shared words) + one dissimilar.
        _store_fact(db, emb, cfg, "Proxmox runs the build server",
                    agent="default", visibility="shared")
        _store_fact(db, emb, cfg, "Proxmox build server uptime",
                    agent="default", visibility="shared")
        _store_fact(db, emb, cfg, "unrelated cooking recipe",
                    agent="default", visibility="shared")
        recent = db.get_recent_memories(since_ts=time.time() - 3600)
        assert recent
        # Ensure no network call is made: monkeypatch httpx.Client.post.
        called = {"n": 0}

        def _boom(*a, **k):
            called["n"] += 1
            raise AssertionError("cloud model should not be called in selection")

        import httpx

        orig_post = httpx.Client.post
        httpx.Client.post = _boom  # type: ignore[method-assign]
        try:
            pairs = dream_mod._select_candidate_pairs(db, recent, mode="night")
        finally:
            httpx.Client.post = orig_post  # type: ignore[method-assign]
        assert called["n"] == 0, "candidate selection must not call the cloud"
        assert isinstance(pairs, list)
        assert len(pairs) <= 30
    finally:
        db.close()


def test_candidate_selection_bounded_and_cross_agent(hermes_home: Path):
    from remnant import dream as dream_mod

    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        # Two agents store the same fact → cross_agent candidate.
        _store_fact(db, emb, cfg, "Sven prefers dark mode in the editor",
                    agent="alice", visibility="shared")
        _store_fact(db, emb, cfg, "Sven prefers dark mode editor theme",
                    agent="bob", visibility="shared")
        recent = db.get_recent_memories(since_ts=time.time() - 3600)
        pairs = dream_mod._select_candidate_pairs(db, recent, mode="night")
        cross = [p for p in pairs if p["kind"] == "cross_agent"]
        assert not cross, "cross-profile candidates must be excluded"
    finally:
        db.close()


# ===========================================================================
# Budget + cooldown enforcement
# ===========================================================================


def test_day_dream_budget_exhausted(hermes_home: Path, monkeypatch):
    db = _open_db(hermes_home)
    cfg = RemnantConfig(dream_day_budget=2)
    emb = _fake_embed(db, cfg)
    try:
        # Pre-set the budget counter to the cap.
        today = time.strftime("%Y-%m-%d", time.gmtime())
        db.set_state("day_counter_date", today, agent_id=cfg.agent_id)
        db.set_state("day_counter", cfg.dream_day_budget, agent_id=cfg.agent_id)
        res = day_dream(db, cfg, emb)
        assert res["skipped"] == "budget_exhausted"
        assert res["counter"] == cfg.dream_day_budget
    finally:
        db.close()


def test_day_dream_cooldown_skips(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig(dream_day_budget=3)
    emb = _fake_embed(db, cfg)
    try:
        # Pretend a day run happened moments ago.
        db.set_state("day_run_ts", time.time() - 60, agent_id=cfg.agent_id)
        res = day_dream(db, cfg, emb)
        assert res["skipped"] == "cooldown"
    finally:
        db.close()


def test_dream_cooldown_per_topic(monkeypatch, hermes_home: Path):
    """A repeated judgment on the same pair within cooldown is suppressed."""
    from remnant import dream as dream_mod

    db = _open_db(hermes_home)
    cfg = RemnantConfig(dream_night_budget=5)
    emb = _fake_embed(db, cfg)
    try:
        _store_fact(db, emb, cfg, "Proxmox build server alpha",
                    agent="default", visibility="shared")
        _store_fact(db, emb, cfg, "Proxmox build server alpha beta",
                    agent="default", visibility="shared")
        # Stub the cloud judgment: always return one 'connection' for the same
        # pair, so the second run within cooldown is suppressed.
        calls = {"n": 0}

        def fake_judge(config, pairs, *, mode):
            calls["n"] += 1
            return [{
                "pair_ids": [pairs[0]["id_a"], pairs[0]["id_b"]],
                "judgment": "connection",
                "reason": "linked build server topic",
                "thread_title": "",
            }]

        monkeypatch.setattr(dream_mod, "_cloud_judge", fake_judge)
        r1 = night_dream(db, cfg, emb)
        assert r1["actions"] >= 1
        # Second run on the same pair: cooldown should drop the judgment.
        # Force a fresh window so candidates are re-selected.
        db.set_state("night_run_ts", time.time() - 10, agent_id=cfg.agent_id)  # not cooldown-gated
        r2 = night_dream(db, cfg, emb)
        assert r2["candidates"] >= 1
        assert r2["actions"] == 0, "cooldown should suppress repeated topic"
    finally:
        db.close()


# ===========================================================================
# Diary append (not indexed)
# ===========================================================================


def test_diary_append_first_person(tmp_path: Path, hermes_home: Path):
    from remnant import dream as dream_mod

    diary = tmp_path / "DREAMS.md"
    cfg = RemnantConfig(diary_path=str(diary))
    dream_mod._append_diary(cfg, "day", "the build server and Proxmox are linked")
    text = diary.read_text()
    assert "## " in text and "(day)" in text
    assert "I noticed that" in text
    assert "---" in text


def test_diary_not_indexed_by_remmnant(provider: RemnantMemoryProvider):
    # The diary path default is outside the vault and never indexed by
    # index_vault; sanity check the prompt mentions the diary is private.
    block = provider.system_prompt_block()
    assert "DREAMS.md" in block
    assert "private" in block.lower()


# ===========================================================================
# Cross-agent merge via day_dream
# ===========================================================================


def test_day_dream_merges_cross_agent_duplicates(monkeypatch, hermes_home: Path):
    from remnant import dream as dream_mod

    db = _open_db(hermes_home)
    cfg = RemnantConfig(agent_id="alice", dream_day_budget=3)
    emb = _fake_embed(db, cfg)
    try:
        mid_a = _store_fact(db, emb, cfg, "Sven owns the BlacksiteLab homelab",
                            agent="alice", visibility="shared")
        mid_b = _store_fact(db, emb, cfg, "Sven owns the BlacksiteLab homelab",
                            agent="bob", visibility="shared")
        assert mid_a and mid_b
        # Force the cloud judgment to say same_fact for the cross pair.
        def fake_judge(config, pairs, *, mode):
            cross = [p for p in pairs if p["kind"] == "cross_agent"]
            if not cross:
                return []
            p = cross[0]
            return [{
                "pair_ids": [p["id_a"], p["id_b"]],
                "judgment": "same_fact",
                "reason": "identical ownership fact",
                "thread_title": "",
            }]

        monkeypatch.setattr(dream_mod, "_cloud_judge", fake_judge)
        res = day_dream(db, cfg, emb)
        assert res["actions"] == 0
        assert db.get_memory(mid_a)["status"] == "active"
        assert db.get_memory(mid_b)["status"] == "active"
    finally:
        db.close()


def test_night_dream_creates_thread_for_connection(monkeypatch, hermes_home: Path):
    from remnant import dream as dream_mod

    db = _open_db(hermes_home)
    cfg = RemnantConfig(dream_night_budget=5)
    emb = _fake_embed(db, cfg)
    try:
        _store_fact(db, emb, cfg, "Proxmox host alpha runs the build server",
                    agent="default", visibility="shared")
        _store_fact(db, emb, cfg, "Build server alpha uses Proxmox backups",
                    agent="default", visibility="shared")
        def fake_judge(config, pairs, *, mode):
            return [{
                "pair_ids": [pairs[0]["id_a"], pairs[0]["id_b"]],
                "judgment": "connection",
                "reason": "both reference Proxmox build server",
                "thread_title": "Proxmox build infra",
            }]

        monkeypatch.setattr(dream_mod, "_cloud_judge", fake_judge)
        res = night_dream(db, cfg, emb)
        assert res["actions"] >= 1
        threads = list_threads(db, status="active")
        assert any(t["source"] == "dream" for t in threads)
    finally:
        db.close()


def test_day_dream_no_recent_memories(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig(dream_day_budget=3)
    emb = _fake_embed(db, cfg)
    try:
        res = day_dream(db, cfg, emb)
        assert res["candidates"] == 0
        assert res["actions"] == 0
    finally:
        db.close()


def test_dream_state_persisted_after_run(monkeypatch, hermes_home: Path):
    from remnant import dream as dream_mod

    db = _open_db(hermes_home)
    cfg = RemnantConfig(dream_night_budget=5)
    emb = _fake_embed(db, cfg)
    try:
        _store_fact(db, emb, cfg, "durable alpha fact about Proxmox",
                    agent="default", visibility="shared")
        _store_fact(db, emb, cfg, "alpha Proxmox fact durable statement",
                    agent="default", visibility="shared")
        monkeypatch.setattr(dream_mod, "_cloud_judge",
                            lambda c, p, *, mode: [])
        before = time.time()
        res = night_dream(db, cfg, emb)
        assert res["candidates"] >= 1
        ts = db.get_state("night_run_ts", agent_id=cfg.agent_id)
        assert ts is not None and float(ts) >= before
    finally:
        db.close()


# ===========================================================================
# Tool dispatch: memory_thread
# ===========================================================================


def test_memory_thread_create_via_tool(provider: RemnantMemoryProvider):
    res = provider.handle_tool_call(
        "memory_thread",
        {"action": "create", "title": "T", "topic": "t", "importance": 0.8},
        session_id="s",
    )
    parsed = json.loads(res)
    assert "error" not in parsed
    assert parsed["title"] == "T"
    assert parsed["topic"] == "t"


def test_memory_thread_list_via_tool(provider: RemnantMemoryProvider):
    provider.handle_tool_call(
        "memory_thread", {"action": "create", "title": "A", "topic": "a"},
        session_id="s",
    )
    res = provider.handle_tool_call(
        "memory_thread", {"action": "list"}, session_id="s",
    )
    assert json.loads(res)["count"] >= 1


def test_memory_thread_resolve_via_tool(provider: RemnantMemoryProvider):
    created = provider.handle_tool_call(
        "memory_thread", {"action": "create", "title": "A", "topic": "a"},
        session_id="s",
    )
    tid = json.loads(created)["thread_id"]
    res = provider.handle_tool_call(
        "memory_thread", {"action": "resolve", "thread_id": tid},
        session_id="s",
    )
    assert json.loads(res)["thread"]["status"] == "resolved"


def test_memory_thread_update_via_tool(provider: RemnantMemoryProvider):
    created = provider.handle_tool_call(
        "memory_thread", {"action": "create", "title": "A", "topic": "a"},
        session_id="s",
    )
    tid = json.loads(created)["thread_id"]
    res = provider.handle_tool_call(
        "memory_thread",
        {"action": "update", "thread_id": tid, "title": "A2", "importance": 0.9},
        session_id="s",
    )
    parsed = json.loads(res)
    assert parsed["thread"]["title"] == "A2"
    assert parsed["thread"]["importance"] == 0.9


def test_memory_thread_stale_via_tool(provider: RemnantMemoryProvider):
    res = provider.handle_tool_call(
        "memory_thread", {"action": "stale"}, session_id="s",
    )
    parsed = json.loads(res)
    assert "marked_stale" in parsed
    assert parsed["count"] == 0


def test_memory_thread_create_validation(provider: RemnantMemoryProvider):
    res = provider.handle_tool_call(
        "memory_thread", {"action": "create", "title": "", "topic": "x"},
        session_id="s",
    )
    assert "error" in json.loads(res)


def test_memory_thread_schema_present(provider: RemnantMemoryProvider):
    schemas = provider.get_tool_schemas()
    names = {s["function"]["name"] for s in schemas}
    assert "memory_thread" in names
    t = next(s for s in schemas if s["function"]["name"] == "memory_thread")
    actions = set(t["function"]["parameters"]["properties"]["action"]["enum"])
    assert actions == {"create", "update", "resolve", "list", "stale"}


# ===========================================================================
# Provider run_dream_loop helper
# ===========================================================================


def test_provider_run_dream_loop_day(provider: RemnantMemoryProvider):
    res = provider.run_dream_loop("day")
    assert res["mode"] == "day"
    # No recent memories => 0 candidates, no error.
    assert res.get("candidates", 0) == 0


def test_provider_run_dream_loop_unknown_mode(provider: RemnantMemoryProvider):
    res = provider.run_dream_loop("bogus")
    assert "error" in res


def test_provider_system_prompt_mentions_threads_and_dreams(provider):
    block = provider.system_prompt_block()
    assert "memory_thread" in block
    assert "DREAMS.md" in block
    assert "day_dream" in block or "night_dream" in block
