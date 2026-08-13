"""Tests for entity graph cleanup (issue #17).

These run without a live Ollama.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from remnant.cleanup_entities import cleanup_entities, find_noise_entities
from remnant.config import RemnantConfig
from remnant.db import default_db_path, open_db
from remnant.embed import Embedder
from remnant.ingest import store_memory


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
            h = int.from_bytes(
                hashlib.sha256(w.encode("utf-8")).digest()[:4], "big"
            ) % dim
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


@pytest.fixture(autouse=True)
def patch_extract(monkeypatch):
    from remnant import extract as extract_mod
    monkeypatch.setattr(extract_mod.ExtractionWorker, "_extract", lambda self, u, a: [])
    yield


def _open_db(hermes_home: Path):
    return open_db(default_db_path())


@pytest.fixture()
def db_with_entities(tmp_path: Path):
    db = _open_db(tmp_path)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)

    # Two memories sharing a real entity.
    store_memory(db, emb, cfg, fact="Proxmox alpha runs the build server",
                 entity="Proxmox", session_id="s", agent_id="a")
    store_memory(db, emb, cfg, fact="Proxmox backups run nightly",
                 entity="Proxmox", session_id="s", agent_id="a")

    # One memory linked to a noise stopword entity (manually create entity and link).
    mid = store_memory(db, emb, cfg, fact="The server is online",
                       entity="server", session_id="s", agent_id="a")
    assert mid is not None
    eid_noise = db.resolve_entity("server", "a", entity_type="service", aliases=[])
    assert eid_noise is not None
    db.link_entity(memory_id=mid, entity_id=eid_noise, agent_id="a")

    # One very short entity.
    eid_short = db.resolve_entity("AB", "a", entity_type=None, aliases=[])
    assert eid_short is not None
    db.link_entity(memory_id=mid, entity_id=eid_short, agent_id="a")

    # One single-linked entity (should be removed with min_memories=2).
    eid_lonely = db.resolve_entity("LonelySystem", "a", entity_type="tool", aliases=[])
    assert eid_lonely is not None
    db.link_entity(memory_id=mid, entity_id=eid_lonely, agent_id="a")

    yield db, eid_noise, eid_short, eid_lonely
    db.close()


def test_find_noise_entities(db_with_entities):
    db, eid_noise, eid_short, eid_lonely = db_with_entities
    noise = find_noise_entities(db, min_memories=2)
    ids = {eid for eid, _, _ in noise}
    assert eid_noise in ids
    assert eid_short in ids
    assert eid_lonely in ids


def test_dry_run_reports_but_does_not_delete(db_with_entities):
    db, eid_noise, eid_short, eid_lonely = db_with_entities
    stats = cleanup_entities(db, dry_run=True, min_memories=2)
    assert stats["dry_run"] is True
    assert stats["would_delete"] == 3
    assert stats["deleted"] == 0
    # Entities still exist.
    for eid in (eid_noise, eid_short, eid_lonely):
        assert db.get_entity(eid) is not None


def test_live_cleanup_deletes_noise(db_with_entities):
    db, eid_noise, eid_short, eid_lonely = db_with_entities
    stats = cleanup_entities(db, dry_run=False, min_memories=2)
    assert stats["dry_run"] is False
    assert stats["deleted"] == 3
    for eid in (eid_noise, eid_short, eid_lonely):
        assert db.get_entity(eid) is None


def test_live_cleanup_preserves_real_shared_entity(db_with_entities):
    db, *_ = db_with_entities
    real = db.find_entity_by_name("Proxmox", agent_id="a")
    assert real is not None
    cleanup_entities(db, dry_run=False, min_memories=2)
    assert db.get_entity(real) is not None


def test_min_memories_one_keeps_lonely(db_with_entities):
    db, _, _, eid_lonely = db_with_entities
    stats = cleanup_entities(db, dry_run=False, min_memories=1)
    assert stats["deleted"] == 2  # noise + short only
    assert db.get_entity(eid_lonely) is not None


def test_extra_stoplist(db_with_entities):
    db, *_ = db_with_entities
    real = db.find_entity_by_name("Proxmox", agent_id="a")
    stats = cleanup_entities(db, dry_run=False, min_memories=2, stoplist={"proxmox"})
    assert real is not None
    assert db.get_entity(real) is None
    assert stats["deleted"] == 4
