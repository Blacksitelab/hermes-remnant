from __future__ import annotations

from pathlib import Path

import pytest

from remnant.config import RemnantConfig
from remnant.db import open_db
from remnant.import_sources import import_hindsight, import_memory_store
from remnant.secrets import SecretLikeContentError
from remnant.vault import index_file


class Embedder:
    _model = "g47-test"

    def __init__(self):
        self.calls = 0

    def embed(self, _text):
        self.calls += 1
        return [1.0]


def _profile(home: Path, text: str) -> None:
    path = home / "profiles" / "alpha" / "MEMORY.md"
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")


@pytest.mark.parametrize("mode", ({}, {"dry_run": True}, {"shadow": True}))
def test_memory_store_rejects_persisted_literal_before_output(tmp_path: Path, mode):
    _profile(tmp_path, "- safe fact\n")
    import remnant.import_sources as sources
    original = sources.discover_memory_store_entries
    sources.discover_memory_store_entries = lambda _home: iter([
        ("alpha", "sk_live_derived.md", "- safe fact\n")
    ])
    db = open_db(tmp_path / "db.sqlite")
    emb = Embedder()
    try:
        with pytest.raises(SecretLikeContentError):
            import_memory_store(db, RemnantConfig(agent_id="alpha"), emb, tmp_path, **mode)
        assert db.list_memories(agent_id="alpha") == []
        assert db.list_audit(action="import") == []
        log = tmp_path / "remnant" / "shadow.log"
        assert not log.exists()
        assert emb.calls == 0
    finally:
        sources.discover_memory_store_entries = original
        db.close()


@pytest.mark.parametrize("mode", ({}, {"dry_run": True}, {"shadow": True}))
def test_hindsight_rejects_credential_like_query_before_side_effects(
    tmp_path: Path, mode, monkeypatch,
):
    import remnant.import_sources as sources

    monkeypatch.setattr(
        sources, "_hindsight_recall",
        lambda query, *, limit, bank_id: [{"content": "safe hindsight fact"}],
    )
    db = open_db(tmp_path / "db.sqlite")
    emb = Embedder()
    try:
        with pytest.raises(SecretLikeContentError):
            import_hindsight(db, RemnantConfig(agent_id="alpha"), emb,
                             queries=["safe query sk_live_12345678901234567890"], **mode)
        assert db.list_memories(agent_id="alpha") == []
        assert db.list_audit(action="import") == []
        assert not (tmp_path / "remnant" / "shadow.log").exists()
        assert emb.calls == 0
    finally:
        db.close()


def test_exempt_memory_path_still_rejects_explicit_credential(tmp_path: Path):
    _profile(tmp_path, "- /home/alice/project/MEMORY.md\n- password: 'secret-value'\n")
    db = open_db(tmp_path / "db.sqlite")
    try:
        with pytest.raises(SecretLikeContentError):
            import_memory_store(db, RemnantConfig(agent_id="alpha"), Embedder(), tmp_path)
        assert db.list_memories(agent_id="alpha") == []
    finally:
        db.close()


def test_vault_rejection_leaves_index_unchanged_and_skips_embedding(tmp_path: Path):
    vault = tmp_path / "vault"
    note = vault / "Notes" / "bad.md"
    note.parent.mkdir(parents=True)
    note.write_text("---\ntags: ['token: abcdefghijklmnop']\n---\nnormal body", encoding="utf-8")
    db = open_db(tmp_path / "db.sqlite")
    emb = Embedder()
    try:
        with pytest.raises(SecretLikeContentError):
            index_file(db, RemnantConfig(vault_path=str(vault)), emb, note)
        assert db.get_vault_hash("Notes/bad.md") is None
        assert db.get_vault_memory("Notes/bad.md") is None
        assert emb.calls == 0
    finally:
        db.close()


def test_memory_store_rejects_late_batch_without_side_effects(tmp_path: Path, monkeypatch):
    import remnant.import_sources as sources

    monkeypatch.setattr(sources, "discover_memory_store_entries", lambda _home: iter([
        ("alpha", "MEMORY.md", "- safe fact\n"),
        ("alpha", "MEMORY.md", "- late password: 'secret-value'\n"),
    ]))
    db = open_db(tmp_path / "db.sqlite")
    emb = Embedder()
    try:
        with pytest.raises(SecretLikeContentError):
            import_memory_store(db, RemnantConfig(agent_id="alpha"), emb, tmp_path)
        assert db.list_memories(agent_id="alpha") == []
        assert db.list_audit(action="import") == []
        assert not (tmp_path / "remnant" / "shadow.log").exists()
        assert emb.calls == 0
    finally:
        db.close()
