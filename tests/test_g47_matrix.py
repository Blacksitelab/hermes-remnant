from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from remnant.config import RemnantConfig
from remnant.db import open_db
from remnant.db import RemnantDB
from remnant.ingest import store_memory
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


@pytest.fixture(params=(
    pytest.param({"dry_run": False, "shadow": False}, id="normal"),
    pytest.param({"dry_run": True, "shadow": False}, id="dry-run"),
    pytest.param({"dry_run": False, "shadow": True}, id="shadow"),
))
def import_mode(request):
    return request.param


@pytest.fixture
def safe_literal():
    value = "".join(("sk", "-", "p1fixture", "x" * 24))
    assert value.startswith("sk-")
    return value


def _rejecting_db() -> Mock:
    db = Mock(spec_set=RemnantDB)
    for name in dir(RemnantDB):
        if not name.startswith("_") and callable(getattr(RemnantDB, name, None)):
            setattr(db, name, Mock(side_effect=AssertionError("unexpected DB call")))
    return db


def _assert_rejection(exc, field):
    assert exc.type is SecretLikeContentError
    assert str(exc.value) == "secret_like_content"
    assert exc.value.reason_code == "secret_like_content"
    assert exc.value.field == field


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
    monkeypatch.setattr(sources, "discover_memory_store_entries", lambda _home: iter([("alpha", "MEMORY.md", "- safe fact\n"), ("alpha", "MEMORY.md", "- late password: 'secret-value'\n")]))
    db = open_db(tmp_path / "db.sqlite"); emb = Embedder()
    try:
        with pytest.raises(SecretLikeContentError): import_memory_store(db, RemnantConfig(agent_id="alpha"), emb, tmp_path)
        assert db.list_memories(agent_id="alpha") == [] and db.list_audit(action="import") == []
        assert not (tmp_path / "remnant" / "shadow.log").exists() and emb.calls == 0
    finally: db.close()


def test_hindsight_rejects_late_content_before_side_effects(tmp_path: Path, monkeypatch, import_mode, safe_literal):
    import remnant.import_sources as sources
    monkeypatch.setattr(sources, "_hindsight_recall", lambda query, *, limit, bank_id: [{"content": "safe"}] if query == "safe" else [{"content": "another safe"}, {"content": safe_literal}])
    db, emb, home = _rejecting_db(), Embedder(), tmp_path / "hermes"
    with pytest.raises(SecretLikeContentError) as exc: import_hindsight(db, RemnantConfig(agent_id="alpha"), emb, queries=["safe", "late"], hermes_home=home, **import_mode)
    _assert_rejection(exc, "import.content")
    assert db.mock_calls == [] and emb.calls == 0 and not (home / "remnant" / "shadow.log").exists()


def test_hindsight_rejects_nested_recalled_metadata_before_side_effects(tmp_path: Path, monkeypatch, import_mode, safe_literal):
    import remnant.import_sources as sources
    monkeypatch.setattr(sources, "_hindsight_recall", lambda query, *, limit, bank_id: [{"content": "safe", "metadata": {"outer": [{"value": safe_literal}]}}])
    db, emb = _rejecting_db(), Embedder()
    with pytest.raises(SecretLikeContentError) as exc: import_hindsight(db, RemnantConfig(agent_id="alpha"), emb, queries=["safe"], hermes_home=tmp_path / "hermes", **import_mode)
    _assert_rejection(exc, "import.recalled_metadata.outer[0].value")
    assert db.mock_calls == [] and emb.calls == 0


def test_store_memory_rejects_nested_caller_metadata_before_db_or_embedding(safe_literal):
    db, emb = _rejecting_db(), Embedder()
    with pytest.raises(SecretLikeContentError) as exc: store_memory(db, emb, RemnantConfig(agent_id="alpha"), fact="safe fact", entity="safe subject", session_id="safe-session", agent_id="alpha", metadata={"outer": [{"value": safe_literal}]})
    _assert_rejection(exc, "metadata.outer[0].value")
    assert db.mock_calls == [] and emb.calls == 0
