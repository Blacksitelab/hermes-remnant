from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from unittest.mock import Mock

import pytest

from remnant.config import RemnantConfig
from remnant.db import RemnantDB, open_db
from remnant.import_sources import import_hindsight, import_memory_store
from remnant.ingest import store_memory
from remnant.secrets import SecretLikeContentError, classify_text
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
    assert any(item.kind == "literal" for item in classify_text(value))
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


def _assert_quiet(caplog, capsys):
    caplog.set_level(logging.DEBUG)
    captured = capsys.readouterr()
    assert not caplog.records and not captured.out and not captured.err


def _db_digest(db) -> str:
    conn = db._conn
    objects = conn.execute(
        "SELECT type, name, sql FROM sqlite_master "
        "WHERE type IN ('table', 'index', 'trigger', 'view') ORDER BY type, name"
    ).fetchall()
    snapshot = [tuple(row) for row in objects]
    for (name,) in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ):
        try:
            rows = conn.execute(f'SELECT rowid, * FROM "{name}"').fetchall()
        except sqlite3.OperationalError:
            rows = conn.execute(f'SELECT * FROM "{name}"').fetchall()
        snapshot.append((name, [tuple(row) for row in rows]))
    return hashlib.sha256(json.dumps(snapshot, default=str).encode()).hexdigest()


def test_memory_store_rejects_persisted_metadata_literal_before_output(
    tmp_path: Path, import_mode, safe_literal, caplog, capsys, monkeypatch
):
    import remnant.import_sources as sources
    extract = Mock(side_effect=AssertionError("entity extraction reached"))
    shadow = Mock(side_effect=AssertionError("shadow writer reached"))
    monkeypatch.setattr(sources, "extract_and_link_entities", extract)
    monkeypatch.setattr(sources, "write_shadow_log", shadow)
    safe_path = tmp_path / "profiles" / "alpha" / "MEMORY.md"
    synthetic_path = tmp_path / "profiles" / "alpha" / f"MEMORY-{safe_literal}.md"
    _profile(tmp_path, "- safe fact\n")
    calls = 0

    def discover(home):
        nonlocal calls
        calls += 1
        path = safe_path if calls == 1 else synthetic_path
        yield "alpha", str(path), "- safe fact"

    monkeypatch.setattr(sources, "discover_memory_store_entries", discover)
    db, emb = _rejecting_db(), Embedder()
    with pytest.raises(SecretLikeContentError) as exc:
        import_memory_store(
            db, RemnantConfig(agent_id="alpha"), emb, tmp_path, **import_mode
        )
    _assert_rejection(exc, "import.metadata.imported_from")
    assert calls == 2
    assert db.mock_calls == [] and emb.calls == 0
    assert extract.call_count == shadow.call_count == 0
    assert not (tmp_path / "remnant" / "shadow.log").exists()
    _assert_quiet(caplog, capsys)


def test_hindsight_rejects_credential_like_query_before_side_effects(
    tmp_path: Path, import_mode, safe_literal, monkeypatch, caplog, capsys,
):
    import remnant.import_sources as sources

    calls = []
    extract = Mock(side_effect=AssertionError("entity extraction reached"))
    shadow = Mock(side_effect=AssertionError("shadow writer reached"))
    monkeypatch.setattr(sources, "extract_and_link_entities", extract)
    monkeypatch.setattr(sources, "write_shadow_log", shadow)

    def recall(query, *, limit, bank_id):
        calls.append((query, bank_id, limit))
        return [{"content": "safe hindsight fact"}]

    monkeypatch.setattr(
        sources, "_hindsight_recall", recall,
    )
    db, emb = _rejecting_db(), Embedder()
    with pytest.raises(SecretLikeContentError) as exc:
        import_hindsight(
            db, RemnantConfig(agent_id="alpha"), emb,
            queries=["safe query", f"late {safe_literal}"],
            hermes_home=tmp_path / "hermes", **import_mode,
        )
    _assert_rejection(exc, "import.query")
    assert calls == [("safe query", "hermes-alpha", sources.HINDSIGHT_QUERY_LIMIT)]
    assert extract.call_count == shadow.call_count == 0
    assert db.mock_calls == [] and emb.calls == 0
    assert not (tmp_path / "hermes" / "remnant" / "shadow.log").exists()
    _assert_quiet(caplog, capsys)


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


def test_memory_store_rejects_late_batch_without_side_effects(
    tmp_path: Path, import_mode, safe_literal, caplog, capsys, monkeypatch
):
    import remnant.import_sources as sources
    extract = Mock(side_effect=AssertionError("entity extraction reached"))
    shadow = Mock(side_effect=AssertionError("shadow writer reached"))
    monkeypatch.setattr(sources, "extract_and_link_entities", extract)
    monkeypatch.setattr(sources, "write_shadow_log", shadow)
    _profile(tmp_path, f"- safe fact\n- late {safe_literal}\n")
    db, emb = _rejecting_db(), Embedder()
    with pytest.raises(SecretLikeContentError) as exc:
        import_memory_store(
            db, RemnantConfig(agent_id="alpha"), emb, tmp_path, **import_mode
        )
    _assert_rejection(exc, "import.content")
    assert db.mock_calls == [] and emb.calls == 0
    assert extract.call_count == shadow.call_count == 0
    assert not (tmp_path / "remnant" / "shadow.log").exists()
    _assert_quiet(caplog, capsys)


@pytest.mark.parametrize(
    ("bad_part", "relative_path", "seed_index"),
    (("passage", "Notes/late.md", 0), ("tags", "Tags/custom.md", 1),
     ("metadata", "Nested/frontmatter.md", 2), ("title", "Titles/later.md", 3)),
)
def test_vault_rejects_late_content_without_mutating_seeded_state(
    tmp_path: Path, bad_part: str, relative_path: str, seed_index: int,
    safe_literal: str, caplog, capsys, monkeypatch
):
    import remnant.vault as vault_module
    from remnant.entity import resolve_and_link, seed_relations

    def deterministic_entities(db, *, memory_id, text, typed_entities, agent_id, min_memories):
        del typed_entities, min_memories
        ids = [resolve_and_link(db, memory_id=memory_id, entity_name=name,
                                entity_type="concept", agent_id=agent_id)[0]
               for name in ("Alice", "Bob")]
        seed_relations(db, memory_id=memory_id, entity_ids=ids, text=text)

    monkeypatch.setattr(vault_module, "extract_and_link_entities", deterministic_entities)
    vault = tmp_path / "vault"
    note = vault / relative_path
    note.parent.mkdir(parents=True)
    safe_body = ("Alice and Bob discussed a harmless project note. "
                 "This deliberately contains enough ordinary text to create "
                 "more than one passage in the configured vault index. "
                 "The first passage is safe and remains independent from the "
                 "later validation boundary, with enough words to exceed the "
                 "configured splitter limit without relying on incidental wrapping.\n\n"
                 "## Second\nCarol recorded another harmless fact for the seed.\n")
    note.write_text("---\ntags: [safe, seeded]\ncustom:\n  nested: [one, two]\n---\n"
                    + safe_body, encoding="utf-8")
    db = open_db(tmp_path / "db.sqlite")
    cfg = RemnantConfig(vault_path=str(vault), vault_passage_chars=128)
    seed_emb = Embedder()
    try:
        assert index_file(db, cfg, seed_emb, note)
        passages = db.get_vault_passages(relative_path, agent_id=cfg.agent_id)
        assert len(passages) > 1
        seed_memory = db.get_memory(passages[0]["memory_id"])
        assert seed_memory and seed_memory["content"]
        assert seed_memory["metadata"]
        db.set_memory_field(passages[0]["memory_id"], "trust_score", 0.91, actor="test")
        for _ in range(6):
            db.increment_seen_count(passages[0]["memory_id"])
        db.put_cached_embedding("seed-model", "seed-hash", [0.25])
        db.write_audit(
            actor="test", action="seed", memory_id=passages[0]["memory_id"],
            details={"seed": True},
        )
        db.insert_turn(
            session_id="seed-session", agent_id="alpha", user_text="u", assistant_text="a"
        )
        with db.read() as cur:
            assert cur.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] > 0
            assert cur.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0] > 0
            assert cur.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] > 0
            assert cur.execute("SELECT COUNT(*) FROM memory_entities").fetchone()[0] > 0
            assert cur.execute("SELECT COUNT(*) FROM entities").fetchone()[0] > 0
            assert cur.execute("SELECT COUNT(*) FROM relations").fetchone()[0] > 0
            assert cur.execute("SELECT COUNT(*) FROM relation_evidence").fetchone()[0] > 0
        caplog.clear()
        before = _db_digest(db)
        before_changes = db._conn.total_changes
        with db.read() as cur:
            before_embeddings = cur.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        if bad_part == "passage":
            body = (
                "Alice and Bob discussed a harmless project note. "
                "This safe first passage is intentionally longer than 128 characters so the "
                "runtime literal cannot share its passage. More harmless words keep it stable.\n\n"
                f"## Second\nlater value {safe_literal}"
            )
            frontmatter = "tags: [safe, seeded]\ncustom:\n  nested: [one, two]"
        elif bad_part == "tags":
            body = safe_body
            frontmatter = f"tags: [safe, {safe_literal}]\ncustom:\n  nested: [one, two]"
        elif bad_part == "metadata":
            body = safe_body
            frontmatter = f"tags: [safe, seeded]\ncustom:\n  nested: {safe_literal}"
        else:
            body = ("Alice and Bob discussed a harmless project note. "
                    "This prefix is intentionally long enough to be the safe first passage "
                    "before the later title heading and its validation boundary. "
                    "More harmless words keep the first passage independent.\n\n"
                    f"# {safe_literal}\n")
            frontmatter = "tags: [safe, seeded]\ncustom:\n  nested: [one, two]"
        note.write_text(f"---\n{frontmatter}\n---\n{body}", encoding="utf-8")
        assert len(body.split("\n\n", 1)[0]) > cfg.vault_passage_chars
        assert safe_literal not in body.split("\n\n", 1)[0]
        input_digest = hashlib.sha256(note.read_bytes()).hexdigest()
        emb = Embedder()
        with pytest.raises(SecretLikeContentError) as exc:
            index_file(db, cfg, emb, note)
        expected_field = {"passage": "vault.content", "tags": "vault.tags[1]",
                          "metadata": "vault.metadata.fm_custom.nested",
                          "title": "vault.metadata.title"}[bad_part]
        _assert_rejection(exc, expected_field)
        assert _db_digest(db) == before
        assert db._conn.total_changes == before_changes
        assert emb.calls == 0
        with db.read() as cur:
            assert cur.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == before_embeddings
        assert hashlib.sha256(note.read_bytes()).hexdigest() == input_digest
        assert not caplog.records and not capsys.readouterr().out and not capsys.readouterr().err
    finally:
        db.close()


def test_hindsight_rejects_late_content_before_side_effects(
    tmp_path: Path, monkeypatch, import_mode, safe_literal, caplog, capsys
):
    import remnant.import_sources as sources
    extract = Mock(side_effect=AssertionError("entity extraction reached"))
    shadow = Mock(side_effect=AssertionError("shadow writer reached"))
    monkeypatch.setattr(sources, "extract_and_link_entities", extract)
    monkeypatch.setattr(sources, "write_shadow_log", shadow)

    def recall(query, *, limit, bank_id):
        return [{"content": "safe"}] if query == "safe" else [
            {"content": "another safe"}, {"content": safe_literal}
        ]

    monkeypatch.setattr(sources, "_hindsight_recall", recall)
    db, emb, home = _rejecting_db(), Embedder(), tmp_path / "hermes"
    with pytest.raises(SecretLikeContentError) as exc:
        import_hindsight(
            db, RemnantConfig(agent_id="alpha"), emb,
            queries=["safe", "late"], hermes_home=home, **import_mode
        )
    _assert_rejection(exc, "import.content")
    assert extract.call_count == shadow.call_count == 0
    assert db.mock_calls == [] and emb.calls == 0 and not (home / "remnant" / "shadow.log").exists()
    _assert_quiet(caplog, capsys)


def test_hindsight_rejects_nested_recalled_metadata_before_side_effects(
    tmp_path: Path, monkeypatch, import_mode, safe_literal, caplog, capsys
):
    import remnant.import_sources as sources

    extract = Mock(side_effect=AssertionError("entity extraction reached"))
    shadow = Mock(side_effect=AssertionError("shadow writer reached"))
    monkeypatch.setattr(sources, "extract_and_link_entities", extract)
    monkeypatch.setattr(sources, "write_shadow_log", shadow)

    monkeypatch.setattr(
        sources,
        "_hindsight_recall",
        lambda query, *, limit, bank_id: [
            {"content": "safe earlier row"},
            {"content": "safe", "metadata": {"outer": [{"value": safe_literal}]}}
        ],
    )
    db, emb = _rejecting_db(), Embedder()
    with pytest.raises(SecretLikeContentError) as exc:
        import_hindsight(
            db, RemnantConfig(agent_id="alpha"), emb, queries=["safe"],
            hermes_home=tmp_path / "hermes", **import_mode
        )
    _assert_rejection(exc, "import.recalled_metadata.outer[0].value")
    assert extract.call_count == shadow.call_count == 0
    assert db.mock_calls == [] and emb.calls == 0
    assert not (tmp_path / "hermes" / "remnant" / "shadow.log").exists()
    _assert_quiet(caplog, capsys)


def test_store_memory_rejects_nested_caller_metadata_before_db_or_embedding(
    safe_literal, caplog, capsys
):
    db, emb = _rejecting_db(), Embedder()
    with pytest.raises(SecretLikeContentError) as exc:
        store_memory(
            db, emb, RemnantConfig(agent_id="alpha"), fact="safe fact",
            entity="safe subject", session_id="safe-session", agent_id="alpha",
            metadata={"outer": [{"value": safe_literal}]},
        )
    _assert_rejection(exc, "metadata.outer[0].value")
    assert db.mock_calls == [] and emb.calls == 0
    _assert_quiet(caplog, capsys)
