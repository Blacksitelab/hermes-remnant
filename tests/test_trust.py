"""Tests for source-based initial trust, contradiction penalty,
corroboration boost, and retrieval reinforcement (issue #11).

These run without a live Ollama: the Embedder uses the same deterministic
word-bag vectors as ``test_dream.py``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from remnant.config import RemnantConfig
from remnant.db import default_db_path, open_db
from remnant.embed import Embedder
from remnant.ingest import store_memory
from remnant.search import search


# --- shared fakes (mirror test_dream.py) -----------------------------------


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fake_embed(db, config, dim=16):
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
def patch_extract(monkeypatch):
    from remnant import extract as extract_mod

    monkeypatch.setattr(extract_mod.ExtractionWorker, "_extract", lambda self, u, a: [])
    yield


def _open_db(hermes_home: Path):
    return open_db(default_db_path())


def _store(db, emb, cfg, fact, *, agent="default", source=None, visibility="private"):
    return store_memory(
        db, emb, cfg,
        fact=fact,
        entity="general",
        session_id="s",
        agent_id=agent,
        visibility=visibility,
        source=source,
    )


# ===========================================================================
# 1. Source-based initial trust
# ===========================================================================


def test_source_conversation_trust_06(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        mid = _store(db, emb, cfg, "Proxmox host alpha runs build box seven",
                     agent="default", source="conversation")
        assert mid is not None
        # conversation seeds at 0.6; corroboration may bump it +0.05 only if
        # another memory shares the 'general' entity. Here there is none, so
        # the trust_score stays at the seed value.
        assert db.get_memory(mid)["trust_score"] == pytest.approx(0.6)
    finally:
        db.close()


def test_source_vault_trust_08(hermes_home: Path):
    """A vault document inserted via vault.index_file seeds trust_score 0.8.

    We exercise the db.insert_memory call directly with source='vault' to
    mirror what vault.index_file does for a brand-new note.
    """
    db = _open_db(hermes_home)
    try:
        mid = db.insert_memory(
            content="Vault note body",
            source="vault",
            agent="default",
            type="document",
            trust_score=0.8,
        )
        assert db.get_memory(mid)["trust_score"] == pytest.approx(0.8)
    finally:
        db.close()


def test_source_manual_trust_09(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        # No source_turn_id and no source -> manual, seed 0.9. Use a unique
        # entity so no corroborating memory shares it (keeps the boost off).
        mid = store_memory(
            db, emb, cfg,
            fact="Sven owns the BlacksiteLab homelab",
            entity="BlacksiteLab",
            session_id="s",
            agent_id="default",
        )
        assert mid is not None
        assert db.get_memory(mid)["trust_score"] == pytest.approx(0.9)
    finally:
        db.close()


# ===========================================================================
# 2. Contradiction penalty
# ===========================================================================


def test_contradiction_penalty_lowers_trust(hermes_home: Path):
    """A contradiction flag reduces the existing memory's trust_score by 0.1
    with a floor of 0.3."""
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        # Seed a memory with the negated form as the truth. Use the typed
        # `entities` list so the contradiction-detection path runs on the next
        # insert (the legacy `entity` string skips contradiction detection).
        mid1 = store_memory(
            db, emb, cfg,
            fact="The build server is online",
            entity="build server",
            entities=[{"name": "build server", "type": None, "aliases": []}],
            session_id="s",
            agent_id="default",
            source="manual",
        )
        assert mid1 is not None
        before = db.get_memory(mid1)["trust_score"]
        assert before == pytest.approx(0.9)

        # Now store the negation; it shares the entity and flips a negation,
        # so detect_contradiction fires and _flag_contradiction runs.
        mid2 = store_memory(
            db, emb, cfg,
            fact="The build server is offline",
            entity="build server",
            entities=[{"name": "build server", "type": None, "aliases": []}],
            session_id="s",
            agent_id="default",
            source="manual",
        )
        assert mid2 is not None
        after = db.get_memory(mid1)["trust_score"]
        # mid1 is in contradiction_targets, so the corroboration boost skips it.
        # Net: 0.9 - 0.1 = 0.8.
        assert after == pytest.approx(0.8)
    finally:
        db.close()


# ===========================================================================
# 3. Corroboration boost
# ===========================================================================


def test_corroboration_boost_raises_shared_entity_trust(hermes_home: Path):
    """A new memory sharing an entity with an existing active memory bumps the
    existing memory's trust_score by +0.05 (capped at 0.95)."""
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        # Seed an existing memory with a known entity.
        mid1 = store_memory(
            db, emb, cfg,
            fact="Proxmox host alpha runs the build server",
            entity="Proxmox",
            session_id="s",
            agent_id="default",
            source="manual",
        )
        assert mid1 is not None
        assert db.get_memory(mid1)["trust_score"] == pytest.approx(0.9)

        # Store a second memory that shares the 'Proxmox' entity. The two
        # facts are not negations of each other, so no contradiction; the
        # corroboration boost should raise mid1's trust_score by +0.05 (capped
        # 0.95). With a seed of 0.9, mid1 goes to 0.95.
        mid2 = store_memory(
            db, emb, cfg,
            fact="Proxmox backups run nightly",
            entity="Proxmox",
            session_id="s",
            agent_id="default",
            source="manual",
        )
        assert mid2 is not None
        after = db.get_memory(mid1)["trust_score"]
        assert after == pytest.approx(0.95)
    finally:
        db.close()


# ===========================================================================
# 4. Retrieval reinforcement
# ===========================================================================


def test_search_reinforces_returned_memories(hermes_home: Path):
    """A search increments seen_count and bumps trust_score by +0.02 (cap
    0.95) for each returned memory."""
    db = _open_db(hermes_home)
    cfg = RemnantConfig(default_search_strategy="keyword")
    emb = _fake_embed(db, cfg)
    try:
        mid = store_memory(
            db, emb, cfg,
            fact="Proxmox host alpha runs the build server",
            entity="Proxmox",
            session_id="s",
            agent_id="default",
            source="manual",
        )
        assert mid is not None
        before = db.get_memory(mid)
        before_trust = before["trust_score"]
        before_seen = before["seen_count"]

        results = search(db, cfg, "Proxmox build server", agent_id="default",
                        strategy="keyword")
        assert any(r["id"] == mid for r in results)

        after = db.get_memory(mid)
        assert after["seen_count"] == before_seen + 1
        assert after["trust_score"] == pytest.approx(min(before_trust + 0.02, 0.95))
    finally:
        db.close()