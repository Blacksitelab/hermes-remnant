"""Tests for the dream pipeline's ``source='dream'`` memories (issue #14) and
the typed-entity noise filter (issue #10).

These run without a live cloud model: the Embedder uses the same deterministic
word-bag vectors as test_phase2/3/5 and ``_cloud_judge`` is monkeypatched.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from remnant.config import RemnantConfig
from remnant.db import default_db_path, open_db
from remnant.dream import day_dream, night_dream
from remnant.embed import Embedder
from remnant.ingest import store_memory

# --- shared fakes (mirror test_phase5) -------------------------------------


def _hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fake_embed(db, config, dim=16):
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


def _open_db(hermes_home: Path):
    return open_db(default_db_path())


def _store_fact(db, emb, cfg, fact, agent="default", visibility="private"):
    return store_memory(
        db, emb, cfg, fact=fact, entity="general",
        session_id="s", agent_id=agent, visibility=visibility,
    )


# ===========================================================================
# Issue #14: store_memory source parameter
# ===========================================================================


def test_store_memory_explicit_source_dream(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        mid = store_memory(
            db, emb, cfg,
            fact="Dream reflection: Proxmox and the build server are linked",
            entity="dream",
            session_id="dream",
            agent_id="default",
            visibility="private",
            source="dream",
        )
        assert mid is not None
        m = db.get_memory(mid)
        assert m["source"] == "dream"
    finally:
        db.close()


def test_store_memory_default_source_conversation_with_turn(hermes_home: Path):
    """When source is omitted and source_turn_id is set, source is
    ``conversation`` (preserved behaviour)."""
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        turn_id = db.insert_turn(
            session_id="s", agent_id="default", user_text="u", assistant_text="a",
        )
        mid = store_memory(
            db, emb, cfg,
            fact="Sven prefers dark mode for the editor",
            entity="Sven",
            session_id="s",
            agent_id="default",
            source_turn_id=turn_id,
        )
        assert mid is not None
        assert db.get_memory(mid)["source"] == "conversation"
    finally:
        db.close()


def test_store_memory_default_source_manual_without_turn(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        mid = store_memory(
            db, emb, cfg,
            fact="Sven owns the BlacksiteLab homelab",
            entity="Sven",
            session_id="s",
            agent_id="default",
        )
        assert mid is not None
        assert db.get_memory(mid)["source"] == "manual"
    finally:
        db.close()


# ===========================================================================
# Issue #14: day_dream / night_dream store source='dream' memories
# ===========================================================================


def _seed_pair(db, emb, cfg, fact_a, fact_b):
    _store_fact(db, emb, cfg, fact_a, agent=cfg.agent_id, visibility="shared")
    _store_fact(db, emb, cfg, fact_b, agent=cfg.agent_id, visibility="shared")
    return db.get_recent_memories(since_ts=time.time() - 3600)


def test_day_dream_stores_dream_memory_for_connection(monkeypatch, hermes_home: Path):
    from remnant import dream as dream_mod

    db = _open_db(hermes_home)
    cfg = RemnantConfig(dream_day_budget=3)
    emb = _fake_embed(db, cfg)
    try:
        _seed_pair(
            db, emb, cfg,
            "Proxmox host alpha runs the build server",
            "Build server alpha uses Proxmox backups",
        )

        def fake_judge(config, pairs, *, mode):
            return [{
                "pair_ids": [pairs[0]["id_a"], pairs[0]["id_b"]],
                "judgment": "connection",
                "reason": "both reference the Proxmox build server",
                "thread_title": "Proxmox build infra",
            }]

        monkeypatch.setattr(dream_mod, "_cloud_judge", fake_judge)
        res = day_dream(db, cfg, emb)
        assert res["actions"] >= 1
        # At least one source='dream' memory now exists.
        with db.read() as cur:
            cur.execute(
                "SELECT COUNT(*) AS c FROM memories WHERE source='dream'"
            )
            assert cur.fetchone()["c"] >= 1
        # And the reflection text is present.
        with db.read() as cur:
            cur.execute(
                "SELECT content FROM memories WHERE source='dream'"
            )
            rows = cur.fetchall()
            assert any("Dream reflection:" in r["content"] for r in rows)
    finally:
        db.close()


def test_night_dream_stores_dream_memory_for_connection(monkeypatch, hermes_home: Path):
    from remnant import dream as dream_mod

    db = _open_db(hermes_home)
    cfg = RemnantConfig(dream_night_budget=5)
    emb = _fake_embed(db, cfg)
    try:
        _seed_pair(
            db, emb, cfg,
            "Proxmox host alpha runs the build server",
            "Build server alpha uses Proxmox backups",
        )

        def fake_judge(config, pairs, *, mode):
            return [{
                "pair_ids": [pairs[0]["id_a"], pairs[0]["id_b"]],
                "judgment": "connection",
                "reason": "shared Proxmox build infra topic",
                "thread_title": "",
            }]

        monkeypatch.setattr(dream_mod, "_cloud_judge", fake_judge)
        res = night_dream(db, cfg, emb)
        assert res["actions"] >= 1
        with db.read() as cur:
            cur.execute(
                "SELECT COUNT(*) AS c FROM memories WHERE source='dream'"
            )
            assert cur.fetchone()["c"] >= 1
    finally:
        db.close()


def test_dream_merge_uses_source_dream(monkeypatch, hermes_home: Path):
    """Cross-profile facts never enter dream consolidation."""
    from remnant import dream as dream_mod

    db = _open_db(hermes_home)
    cfg = RemnantConfig(agent_id="alice", dream_day_budget=3)
    emb = _fake_embed(db, cfg)
    try:
        mid_a = _store_fact(
            db, emb, cfg, "Sven owns the BlacksiteLab homelab",
            agent="alice", visibility="shared",
        )
        mid_b = _store_fact(
            db, emb, cfg, "Sven owns the BlacksiteLab homelab",
            agent="bob", visibility="shared",
        )
        assert mid_a and mid_b

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


# ===========================================================================
# Issue #10: filter_typed_entities
# ===========================================================================


def test_filter_typed_entities_drops_function_words():
    from remnant.extract import filter_typed_entities

    ents = [{"name": n, "type": None, "aliases": []} for n in
            ["no", "if", "the", "and", "when", "not"]]
    assert filter_typed_entities(ents) == []


def test_filter_typed_entities_drops_short_lowercase_noise():
    from remnant.extract import filter_typed_entities

    ents = [{"name": n, "type": None, "aliases": []} for n in ["a", "an", "x", "ok"]]
    # "ok" is len 2 lowercase, not title/upper -> dropped.
    assert filter_typed_entities(ents) == []


def test_filter_typed_entities_keeps_proper_nouns_and_acronyms():
    from remnant.extract import filter_typed_entities

    ents = [{"name": n, "type": None, "aliases": []} for n in
            ["Proxmox", "Remnant", "AI"]]
    kept = {e["name"] for e in filter_typed_entities(ents)}
    assert kept == {"Proxmox", "Remnant", "AI"}


def test_filter_typed_entities_drops_empties_and_duplicates():
    from remnant.extract import filter_typed_entities

    ents = [
        {"name": "Proxmox", "type": "service", "aliases": []},
        {"name": "  ", "type": None, "aliases": []},
        {"name": "", "type": None, "aliases": []},
        {"name": "proxmox", "type": None, "aliases": []},  # dup of Proxmox
        {"name": "Sven", "type": "person", "aliases": ["svenny"]},
    ]
    out = filter_typed_entities(ents)
    names = [e["name"] for e in out]
    assert names == ["Proxmox", "Sven"]
    assert out[1]["aliases"] == ["svenny"]
