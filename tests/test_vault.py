"""Issue #12 tests: vault reindex updates memories in place instead of
forget+insert, so the memory_id, entity links, trust_score, and retrieval
history are preserved across reindex cycles. Only a deleted file forgets.

These tests are isolated from test_phase4.py and use the same fake-embedder
pattern so they run without a live Ollama.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from remnant.config import RemnantConfig
from remnant.db import default_db_path, open_db
from remnant.embed import Embedder
from remnant.vault import index_file, index_vault


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fake_embed(db, config, dim=8):
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


@pytest.fixture()
def vault(tmp_path: Path) -> Path:
    v = tmp_path / "vault"
    v.mkdir()
    return v


def _open_db(hermes_home: Path):
    return open_db(default_db_path())


def _write_note(path: Path, body: str, frontmatter: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frontmatter:
        import yaml

        fm = yaml.safe_dump(frontmatter, sort_keys=False).strip()
        content = f"---\n{fm}\n---\n{body}"
    else:
        content = body
    path.write_text(content, encoding="utf-8")


# ===========================================================================
# Issue #12: in-place update preserves memory_id across reindex
# ===========================================================================


def test_reindex_keeps_memory_id_and_updates_content_and_hash(
    hermes_home: Path, vault: Path
):
    db = _open_db(hermes_home)
    cfg = RemnantConfig(vault_path=str(vault))
    emb = _fake_embed(db, cfg)
    note = vault / "Inbox" / "idea.md"
    _write_note(note, "# Idea\noriginal content about Proxmox")
    try:
        first_mid = index_file(db, cfg, emb, note)
        assert first_mid
        before = db.get_memory(first_mid)
        assert before["status"] == "active"
        assert "original content" in before["content"]
        old_hash = before["content_hash"]

        # Mutate and re-index. memory_id must stay the same; content + hash
        # must change; status must still be active (no forgotten row created).
        _write_note(note, "# Idea\nrevised content about Proxmox backups")
        second_mid = index_file(db, cfg, emb, note)
        assert second_mid == first_mid

        after = db.get_memory(second_mid)
        assert after["status"] == "active"
        assert "revised content" in after["content"]
        assert after["content_hash"] != old_hash
        assert after["content_hash"] == hashlib.sha256(
            after["content"].encode()
        ).hexdigest()

        # No forgotten memory was created.
        with db.read() as cur:
            cur.execute(
                "SELECT COUNT(*) AS c FROM memories "
                "WHERE source='vault' AND source_id='Inbox/idea.md' "
                "AND status='forgotten'"
            )
            assert cur.fetchone()["c"] == 0

        # vault_files still maps the path to the same single memory.
        assert db.get_vault_memory("Inbox/idea.md") == first_mid
    finally:
        db.close()


def test_reindex_unchanged_file_keeps_memory_id_and_no_forgotten(
    hermes_home: Path, vault: Path
):
    db = _open_db(hermes_home)
    cfg = RemnantConfig(vault_path=str(vault))
    emb = _fake_embed(db, cfg)
    note = vault / "Inbox" / "stable.md"
    _write_note(note, "# Stable\nunchanging content about homelab")
    try:
        first_mid = index_vault(db, cfg, emb)
        assert first_mid["indexed"] == 1
        mid = db.get_vault_memory("Inbox/stable.md")
        assert mid

        # Re-index without changing the file. memory_id must stay; no new
        # forgotten rows; indexed count is 0 (skipped).
        second = index_vault(db, cfg, emb)
        assert second["indexed"] == 0
        assert second["skipped"] == 1
        assert second["forgotten"] == 0
        assert db.get_vault_memory("Inbox/stable.md") == mid

        with db.read() as cur:
            cur.execute(
                "SELECT COUNT(*) AS c FROM memories "
                "WHERE source='vault' AND status='forgotten'"
            )
            assert cur.fetchone()["c"] == 0
    finally:
        db.close()


def test_deleting_file_marks_memory_forgotten(hermes_home: Path, vault: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig(vault_path=str(vault))
    emb = _fake_embed(db, cfg)
    note = vault / "Inbox" / "doomed.md"
    _write_note(note, "# Doomed\nthis note will be deleted")
    try:
        index_vault(db, cfg, emb)
        mid = db.get_vault_memory("Inbox/doomed.md")
        assert mid
        assert db.get_memory(mid)["status"] == "active"

        import os

        os.remove(note)
        stats = index_vault(db, cfg, emb)
        assert stats["forgotten"] == 1
        assert db.get_memory(mid)["status"] == "forgotten"
        assert db.get_vault_hash("Inbox/doomed.md") is None
        assert db.get_vault_memory("Inbox/doomed.md") is None
    finally:
        db.close()


def test_reindex_preserves_trust_score_and_entity_links(
    hermes_home: Path, vault: Path
):
    """In-place update must not clobber trust_score, seen_count, or entity
    links — the whole point of issue #12 over the old forget+insert path."""
    db = _open_db(hermes_home)
    cfg = RemnantConfig(vault_path=str(vault))
    emb = _fake_embed(db, cfg)
    note = vault / "Projects" / "alpha.md"
    _write_note(note, "# Project Alpha\nSven runs Project Alpha on Proxmox.")
    try:
        mid = index_file(db, cfg, emb, note)
        assert mid
        # Bump trust_score and seen_count to non-default values; these must
        # survive a re-index of the same path with changed content.
        db.set_memory_field(mid, "trust_score", 0.95, actor="test")
        db.increment_seen_count(mid)
        db.increment_seen_count(mid)
        before = db.get_memory(mid)
        assert before["trust_score"] == 0.95
        # seen_count starts at 1 on insert; two increments => 3.
        assert before["seen_count"] == 3
        # Confirm an entity link exists.
        with db.read() as cur:
            cur.execute(
                "SELECT COUNT(*) AS c FROM memory_entities WHERE memory_id=?", (mid,)
            )
            n_before = cur.fetchone()["c"]
        assert n_before >= 1

        _write_note(note, "# Project Alpha\nSven expanded Project Alpha on Proxmox.")
        new_mid = index_file(db, cfg, emb, note)
        assert new_mid == mid

        after = db.get_memory(mid)
        assert after["trust_score"] == 0.95
        assert after["seen_count"] == 3
        # Entity link row preserved (same memory_id => same PK).
        with db.read() as cur:
            cur.execute(
                "SELECT COUNT(*) AS c FROM memory_entities WHERE memory_id=?", (mid,)
            )
            n_after = cur.fetchone()["c"]
        assert n_after >= 1
    finally:
        db.close()


def test_update_memory_content_writes_vault_update_audit(
    hermes_home: Path, vault: Path
):
    db = _open_db(hermes_home)
    cfg = RemnantConfig(vault_path=str(vault))
    emb = _fake_embed(db, cfg)
    note = vault / "Inbox" / "audited.md"
    _write_note(note, "# Audited\nfirst body")
    try:
        mid = index_file(db, cfg, emb, note)
        _write_note(note, "# Audited\nsecond body")
        index_file(db, cfg, emb, note)
        rows = db.list_audit(memory_id=mid, action="vault_update")
        assert rows, "a vault_update audit row should be written on reindex"
        details = rows[0]["details"]
        if isinstance(details, str):
            details = json.loads(details)
        assert "content_hash" in details
    finally:
        db.close()