"""Phase 4 tests: vault indexing, exclusions, frontmatter parsing, hash-based
re-index skip, deleted-file forget, profile_scope filtering, and locked-note
content hiding.

Run without a live Ollama: the Embedder is monkeypatched with deterministic
word-bag vectors (same pattern as test_phase2/test_phase3) and the extraction
LLM call is stubbed to return no facts.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from remnant import RemnantMemoryProvider
from remnant.config import (
    DEFAULT_VAULT_EXCLUDE,
    DEFAULT_VAULT_PATH,
    DEFAULT_VAULT_REINDEX_INTERVAL_S,
    RemnantConfig,
)
from remnant.db import default_db_path, open_db
from remnant.embed import Embedder
from remnant.search import search as hybrid_search
from remnant.vault import (
    _file_hash,
    _parse_frontmatter,
    _relative_path,
    _should_index,
    index_file,
    index_vault,
)

# --- shared fakes (mirror test_phase2/test_phase3) -------------------------


def _hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fake_embed(db, config, dim=8):
    """Deterministic word-bag embedder (same as test_phase2)."""
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
    """A fresh fake vault directory with a couple of notes."""
    v = tmp_path / "vault"
    v.mkdir()
    return v


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
def provider(hermes_home: Path, vault: Path) -> RemnantMemoryProvider:
    """A provider whose vault_path points at the per-test fake vault."""
    cfg_path = hermes_home / "remnant.json"
    import json

    cfg_path.write_text(json.dumps({"vault_path": str(vault)}))
    p = RemnantMemoryProvider()
    p.initialize(session_id="p4-session", hermes_home=str(hermes_home))
    yield p
    p.shutdown()


def _open_db(hermes_home: Path):
    return open_db(default_db_path())


def _write_note(path: Path, body: str, frontmatter: dict | None = None) -> None:
    """Write a markdown note with optional YAML frontmatter."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if frontmatter:
        import yaml

        fm = yaml.safe_dump(frontmatter, sort_keys=False).strip()
        content = f"---\n{fm}\n---\n{body}"
    else:
        content = body
    path.write_text(content, encoding="utf-8")


# ===========================================================================
# Config defaults
# ===========================================================================


def test_phase4_config_defaults():
    cfg = RemnantConfig()
    assert cfg.vault_path == DEFAULT_VAULT_PATH
    assert cfg.vault_exclude == DEFAULT_VAULT_EXCLUDE
    assert "90_" in cfg.vault_exclude and "99_ARCHIVE" in cfg.vault_exclude
    assert cfg.profile_scope == []
    assert cfg.vault_reindex_interval_s == DEFAULT_VAULT_REINDEX_INTERVAL_S
    # A fresh config still works with from_dict/to_dict round-trip.
    d = cfg.to_dict()
    assert d["vault_path"] == DEFAULT_VAULT_PATH
    assert d["vault_exclude"] == list(DEFAULT_VAULT_EXCLUDE)


# ===========================================================================
# _should_index + _relative_path + _file_hash
# ===========================================================================


def test_should_index_excludes_90_to_95_and_archive():
    excludes = DEFAULT_VAULT_EXCLUDE
    assert _should_index("Inbox/note.md", excludes)
    assert _should_index("Projects/Alpha/spec.md", excludes)
    # Excluded top-level folders.
    assert not _should_index("90_Scratch/junk.md", excludes)
    assert not _should_index("91_Notes/x.md", excludes)
    assert not _should_index("92_Daily/log.md", excludes)
    assert not _should_index("93_Templates/t.md", excludes)
    assert not _should_index("94_Logs/l.md", excludes)
    assert not _should_index("95_Attachments/a.md", excludes)
    assert not _should_index("99_ARCHIVE/old.md", excludes)
    # Trailing slash on the pattern should be tolerated.
    assert not _should_index("99_ARCHIVE/sub/deep.md", excludes)
    # An empty path is never indexed.
    assert not _should_index("", excludes)


def test_should_index_no_excludes_means_everything_allowed():
    assert _should_index("anything/here.md", [])
    assert _should_index("99_ARCHIVE/old.md", [])


def test_relative_path_posix_and_strips_root(vault: Path):
    sub = vault / "Projects" / "Alpha"
    sub.mkdir(parents=True)
    f = sub / "spec.md"
    f.write_text("x")
    rel = _relative_path(f, vault)
    assert rel == "Projects/Alpha/spec.md"
    # Already-relative input via absolute path on a different root still yields
    # the suffix when the root prefix matches.
    assert _relative_path(vault / "a.md", vault) == "a.md"


def test_file_hash_is_sha256_hex(tmp_path: Path):
    import hashlib

    f = tmp_path / "x.md"
    f.write_text("hello\n")
    h = _file_hash(f)
    assert isinstance(h, str) and len(h) == 64
    assert h == hashlib.sha256(b"hello\n").hexdigest()


# ===========================================================================
# Frontmatter parsing
# ===========================================================================


def test_parse_frontmatter_basic():
    text = "---\ntype: note\ntags: [a, b]\nstatus: active\n---\nBody text here."
    fm, body = _parse_frontmatter(text)
    assert fm["type"] == "note"
    assert fm["tags"] == ["a", "b"]
    assert fm["status"] == "active"
    assert "Body text here." in body


def test_parse_frontmatter_tags_as_csv_string():
    text = "---\ntags: one, two, three\n---\nbody"
    fm, body = _parse_frontmatter(text)
    assert fm["tags"] == ["one", "two", "three"]
    assert body.startswith("body")


def test_parse_frontmatter_locked_true():
    text = "---\nlocked: true\n---\nsecret body"
    fm, body = _parse_frontmatter(text)
    assert fm["locked"] is True
    assert "secret body" in body


def test_parse_frontmatter_no_fence_returns_whole_text():
    text = "Just a body, no frontmatter."
    fm, body = _parse_frontmatter(text)
    assert fm == {}
    assert body == text


def test_parse_frontmatter_unterminated_fence_returns_whole_text():
    text = "---\ntype: note\nbut no closing fence\nbody"
    fm, body = _parse_frontmatter(text)
    # No valid closing fence => treat the whole thing as body.
    assert fm == {}


def test_parse_frontmatter_carries_spec_keys():
    text = (
        "---\n"
        "type: note\n"
        "tags: [x]\n"
        "status: active\n"
        "created: 2024-01-01\n"
        "updated: 2024-06-01\n"
        "author: Sven\n"
        "locked: true\n"
        "custom: extra\n"
        "---\n"
        "body"
    )
    fm, _ = _parse_frontmatter(text)
    for k in ("type", "tags", "status", "created", "updated", "author", "locked"):
        assert k in fm
    assert fm["custom"] == "extra"


# ===========================================================================
# index_file
# ===========================================================================


def test_index_file_creates_document_memory(hermes_home: Path, vault: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig(vault_path=str(vault))
    emb = _fake_embed(db, cfg)
    note = vault / "Inbox" / "idea.md"
    _write_note(note, "# Idea\nA durable idea about Proxmox backups.")
    try:
        mid = index_file(db, cfg, emb, note)
        assert mid
        mem = db.get_memory(mid)
        assert mem["type"] == "document"
        assert mem["source"] == "vault"
        assert mem["source_id"] == "Inbox/idea.md"
        assert "Proxmox" in mem["content"]
        # vault_files row written with the hash + memory_id.
        assert db.get_vault_hash("Inbox/idea.md") == _file_hash(note)
        assert db.get_vault_memory("Inbox/idea.md") == mid
    finally:
        db.close()


def test_index_file_stores_frontmatter_metadata(hermes_home: Path, vault: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig(vault_path=str(vault))
    emb = _fake_embed(db, cfg)
    note = vault / "Notes" / "spec.md"
    _write_note(
        note,
        "# Spec\nThe spec for the build.",
        frontmatter={
            "type": "spec", "tags": ["alpha"], "status": "active",
            "created": "2024-01-01", "updated": "2024-06-01",
            "author": "Sven", "locked": True,
        },
    )
    try:
        mid = index_file(db, cfg, emb, note)
        mem = db.get_memory(mid)
        meta = mem["metadata"]
        assert meta["vault_path"] == "Notes/spec.md"
        assert meta["type"] == "spec"
        assert meta["tags"] == ["alpha"]
        assert meta["status"] == "active"
        assert meta["author"] == "Sven"
        assert meta["locked"] is True
        # tags column also populated from frontmatter.
        assert mem["tags"] == ["alpha"]
    finally:
        db.close()


def test_index_file_skips_excluded_paths(hermes_home: Path, vault: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig(vault_path=str(vault))
    emb = _fake_embed(db, cfg)
    note = vault / "90_Scratch" / "junk.md"
    _write_note(note, "scratch junk")
    try:
        assert index_file(db, cfg, emb, note) is None
        assert db.get_vault_hash("90_Scratch/junk.md") is None
    finally:
        db.close()


def test_index_file_skips_non_markdown(hermes_home: Path, vault: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig(vault_path=str(vault))
    emb = _fake_embed(db, cfg)
    note = vault / "image.png"
    note.write_bytes(b"\x89PNG\r\n\x1a\n")
    try:
        assert index_file(db, cfg, emb, note) is None
    finally:
        db.close()


def test_index_file_has_embedding_and_entity_links(hermes_home: Path, vault: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig(vault_path=str(vault))
    emb = _fake_embed(db, cfg)
    note = vault / "Projects" / "alpha.md"
    _write_note(note, "# Project Alpha\nSven runs Project Alpha on Proxmox.")
    try:
        mid = index_file(db, cfg, emb, note)
        assert mid
        vec = db.get_memory_embedding(mid)
        assert vec, "document memory should have an embedding"
        # Entity link exists (Proxmox is a proper noun extracted by the regex).
        with db.read() as cur:
            cur.execute(
                "SELECT COUNT(*) AS c FROM memory_entities WHERE memory_id=?", (mid,)
            )
            n = cur.fetchone()["c"]
        assert n >= 1
    finally:
        db.close()


# ===========================================================================
# index_vault: hash skip, re-index, deleted forget
# ===========================================================================


def test_index_vault_indexes_new_files(hermes_home: Path, vault: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig(vault_path=str(vault))
    emb = _fake_embed(db, cfg)
    _write_note(vault / "Inbox" / "a.md", "# A\nNote A about Proxmox.")
    _write_note(vault / "Inbox" / "b.md", "# B\nNote B about homelab.")
    try:
        stats = index_vault(db, cfg, emb)
        assert stats["indexed"] == 2
        assert stats["skipped"] == 0
        assert stats["forgotten"] == 0
        # Two document memories in the DB.
        with db.read() as cur:
            cur.execute(
                "SELECT COUNT(*) AS c FROM memories WHERE source='vault' AND status='active'"
            )
            assert cur.fetchone()["c"] == 2
    finally:
        db.close()


def test_index_vault_skips_unchanged_files_on_second_run(
    hermes_home: Path, vault: Path
):
    db = _open_db(hermes_home)
    cfg = RemnantConfig(vault_path=str(vault))
    emb = _fake_embed(db, cfg)
    note = vault / "Inbox" / "a.md"
    _write_note(note, "# A\nNote A about Proxmox.")
    try:
        first = index_vault(db, cfg, emb)
        assert first["indexed"] == 1
        second = index_vault(db, cfg, emb)
        assert second["indexed"] == 0
        assert second["skipped"] == 1
        assert second["forgotten"] == 0
    finally:
        db.close()


def test_index_vault_reindexes_changed_file(hermes_home: Path, vault: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig(vault_path=str(vault))
    emb = _fake_embed(db, cfg)
    note = vault / "Inbox" / "a.md"
    _write_note(note, "# A\noriginal content")
    try:
        index_vault(db, cfg, emb)
        old_mid = db.get_vault_memory("Inbox/a.md")
        assert old_mid
        # Mutate the file and re-index.
        _write_note(note, "# A\nchanged content about Proxmox")
        stats = index_vault(db, cfg, emb)
        assert stats["indexed"] == 1
        new_mid = db.get_vault_memory("Inbox/a.md")
        assert new_mid
        # Old memory was forgotten (status preserved, not deleted).
        assert db.get_memory(old_mid)["status"] == "forgotten"
        assert db.get_memory(new_mid)["status"] == "active"
        assert "changed content" in db.get_memory(new_mid)["content"]
        # vault_files now points at the new memory only.
        assert db.get_vault_memory("Inbox/a.md") == new_mid
    finally:
        db.close()


def test_index_vault_forgets_deleted_file(hermes_home: Path, vault: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig(vault_path=str(vault))
    emb = _fake_embed(db, cfg)
    note = vault / "Inbox" / "doomed.md"
    _write_note(note, "# Doomed\nthis note will be deleted")
    try:
        index_vault(db, cfg, emb)
        mid = db.get_vault_memory("Inbox/doomed.md")
        assert mid
        # Delete the file and re-index.
        os.remove(note)
        stats = index_vault(db, cfg, emb)
        assert stats["forgotten"] == 1
        # Memory row preserved but forgotten.
        assert db.get_memory(mid)["status"] == "forgotten"
        # vault_files row removed.
        assert db.get_vault_hash("Inbox/doomed.md") is None
        assert db.get_vault_memory("Inbox/doomed.md") is None
    finally:
        db.close()


def test_index_vault_excludes_directly_at_walk(hermes_home: Path, vault: Path):
    """Excluded top-level folders are pruned at the directory level, so their
    files are never even hashed."""
    db = _open_db(hermes_home)
    cfg = RemnantConfig(vault_path=str(vault))
    emb = _fake_embed(db, cfg)
    _write_note(vault / "Inbox" / "keep.md", "# Keep\nkeep me")
    _write_note(vault / "90_Scratch" / "drop.md", "# Drop\nexclude me")
    _write_note(vault / "99_ARCHIVE" / "sub" / "old.md", "# Old\narchived")
    try:
        stats = index_vault(db, cfg, emb)
        assert stats["indexed"] == 1
        assert db.get_vault_hash("Inbox/keep.md") is not None
        assert db.get_vault_hash("90_Scratch/drop.md") is None
        assert db.get_vault_hash("99_ARCHIVE/sub/old.md") is None
    finally:
        db.close()


def test_index_vault_force_reindexes_everything(hermes_home: Path, vault: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig(vault_path=str(vault))
    emb = _fake_embed(db, cfg)
    note = vault / "Inbox" / "a.md"
    _write_note(note, "# A\ncontent")
    try:
        index_vault(db, cfg, emb)
        # force=True re-indexes even though the hash is unchanged.
        stats = index_vault(db, cfg, emb, force=True)
        assert stats["indexed"] == 1
        assert stats["skipped"] == 0
    finally:
        db.close()


def test_index_vault_missing_vault_path_is_safe(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig(vault_path=str(hermes_home / "no_such_vault"))
    emb = _fake_embed(db, cfg)
    try:
        stats = index_vault(db, cfg, emb)
        assert stats["indexed"] == 0
        assert stats["forgotten"] == 0
    finally:
        db.close()


def test_index_vault_no_duplicate_documents(hermes_home: Path, vault: Path):
    """A single path maps to a single active memory across re-indexes."""
    db = _open_db(hermes_home)
    cfg = RemnantConfig(vault_path=str(vault))
    emb = _fake_embed(db, cfg)
    note = vault / "Inbox" / "a.md"
    _write_note(note, "# A\ncontent")
    try:
        for _ in range(3):
            index_vault(db, cfg, emb)
        with db.read() as cur:
            cur.execute(
                "SELECT COUNT(*) AS c FROM memories "
                "WHERE source='vault' AND source_id='Inbox/a.md' AND status='active'"
            )
            assert cur.fetchone()["c"] == 1
    finally:
        db.close()


# ===========================================================================
# profile_scope filtering in search
# ===========================================================================


def _seed_vault_for_scope(db, cfg, emb, vault: Path):
    # Each note mentions the same keyword "alpha" so a single BM25 query
    # matches all of them; profile_scope then decides which are visible.
    _write_note(vault / "Projects" / "alpha.md", "# Project Alpha\nAlpha build details.")
    _write_note(vault / "Personal" / "diary.md", "# Diary\nDiary mentions alpha.")
    _write_note(vault / "Inbox" / "quick.md", "# Quick\nQuick alpha note.")
    index_vault(db, cfg, emb)


def test_profile_scope_filters_vault_documents(hermes_home: Path, vault: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig(vault_path=str(vault))
    emb = _fake_embed(db, cfg)
    _seed_vault_for_scope(db, cfg, emb, vault)
    try:
        # No scope: all three documents match the shared keyword "alpha".
        all_res = hybrid_search(db, cfg, "alpha", agent_id="default")
        vault_res = [r for r in all_res if r.get("source") == "vault"]
        assert {r["source_id"] for r in vault_res} == {
            "Projects/alpha.md", "Personal/diary.md", "Inbox/quick.md"
        }
        # Scoped to Projects/ only: only the alpha note remains.
        scoped = hybrid_search(
            db, cfg, "alpha", agent_id="default",
            profile_scope=["Projects/"],
        )
        scoped_ids = {r["source_id"] for r in scoped if r.get("source") == "vault"}
        assert scoped_ids == {"Projects/alpha.md"}
    finally:
        db.close()


def test_profile_scope_empty_means_no_filtering(hermes_home: Path, vault: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig(vault_path=str(vault))
    emb = _fake_embed(db, cfg)
    _seed_vault_for_scope(db, cfg, emb, vault)
    try:
        # Empty list => no additional filtering (same as None).
        res = hybrid_search(
            db, cfg, "alpha", agent_id="default", profile_scope=[],
        )
        assert any(r["source_id"] == "Personal/diary.md" for r in res)
    finally:
        db.close()


def test_profile_scope_does_not_affect_non_vault_memories(
    hermes_home: Path, vault: Path
):
    db = _open_db(hermes_home)
    cfg = RemnantConfig(vault_path=str(vault))
    emb = _fake_embed(db, cfg)
    _seed_vault_for_scope(db, cfg, emb, vault)
    # Also store a regular fact that mentions "alpha" so it competes in BM25.
    from remnant.ingest import store_memory

    store_memory(
        db, emb, cfg, fact="Sven prefers dark mode with alpha colors", entity="Sven",
        session_id="s", agent_id="default",
    )
    try:
        res = hybrid_search(
            db, cfg, "alpha", agent_id="default", profile_scope=["Projects/"],
        )
        # The fact survives even though it doesn't live under Projects/.
        assert any("dark mode" in r["content"] for r in res)
        # Vault docs not under Projects/ are filtered out.
        assert all(
            r.get("source_id") in (None, "Projects/alpha.md")
            for r in res
        )
    finally:
        db.close()


def test_profile_scope_from_config_used_by_default(hermes_home: Path, vault: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig(vault_path=str(vault), profile_scope=["Projects/"])
    emb = _fake_embed(db, cfg)
    _seed_vault_for_scope(db, cfg, emb, vault)
    try:
        # No explicit profile_scope arg => falls back to config.profile_scope.
        res = hybrid_search(db, cfg, "alpha", agent_id="default")
        vault_ids = {r["source_id"] for r in res if r.get("source") == "vault"}
        assert vault_ids == {"Projects/alpha.md"}
    finally:
        db.close()


# ===========================================================================
# Locked notes: content hidden from other agents
# ===========================================================================


def test_locked_note_content_hidden_from_other_agent(
    hermes_home: Path, vault: Path
):
    db = _open_db(hermes_home)
    # The vault memories are owned by cfg.agent_id = "owner".
    cfg = RemnantConfig(vault_path=str(vault), agent_id="owner")
    emb = _fake_embed(db, cfg)
    secret = vault / "Personal" / "secret.md"
    _write_note(
        secret,
        "# Secret\nhunter2 is the passphrase for BlacksiteLab.",
        frontmatter={"locked": True},
    )
    public = vault / "Projects" / "public.md"
    _write_note(public, "# Public\nhunter2 appears in the public build log too.")
    index_vault(db, cfg, emb)
    try:
        # Owner agent: sees the secret content (BM25 matches "hunter2").
        owner_res = hybrid_search(db, cfg, "hunter2", agent_id="owner")
        owner_contents = " ".join(r["content"] for r in owner_res)
        assert "hunter2" in owner_contents
        assert any("passphrase" in r["content"] for r in owner_res)

        # Another agent querying the same DB: content masked, only metadata shown.
        viewer_cfg = RemnantConfig(vault_path=str(vault), agent_id="intruder")
        intruder_res = hybrid_search(
            db, viewer_cfg, "hunter2", agent_id=None,
        )
        locked_rows = [r for r in intruder_res if r.get("locked")]
        assert locked_rows, "locked note should still be returned (masked)"
        for r in locked_rows:
            assert "passphrase" not in r["content"]
            assert r["content"] == "[locked note: content hidden]"
            # Metadata (path etc.) is still visible.
            assert r.get("source_id") == "Personal/secret.md"
            assert r.get("source") == "vault"
        # The public (unlocked) note is still returned with its real content.
        public_rows = [
            r for r in intruder_res if r.get("source_id") == "Projects/public.md"
        ]
        assert public_rows
        assert "hunter2" in public_rows[0]["content"]
    finally:
        db.close()


def test_locked_note_visible_to_owner_unmasked(hermes_home: Path, vault: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig(vault_path=str(vault), agent_id="owner")
    emb = _fake_embed(db, cfg)
    secret = vault / "Personal" / "secret.md"
    _write_note(
        secret,
        "# Secret\nSensitive hunter2 details here.",
        frontmatter={"locked": True},
    )
    index_vault(db, cfg, emb)
    try:
        res = hybrid_search(db, cfg, "hunter2", agent_id="owner")
        # Owner sees real content, no locked flag.
        rows = [r for r in res if r.get("source_id") == "Personal/secret.md"]
        assert rows
        assert rows[0]["content"] != "[locked note: content hidden]"
        assert "hunter2" in rows[0]["content"]
        assert not rows[0].get("locked")
    finally:
        db.close()


def test_unlocked_note_content_shown_to_everyone(hermes_home: Path, vault: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig(vault_path=str(vault), agent_id="owner")
    emb = _fake_embed(db, cfg)
    note = vault / "Projects" / "public.md"
    _write_note(note, "# Public\nhunter2 build is green.")
    index_vault(db, cfg, emb)
    try:
        viewer_cfg = RemnantConfig(vault_path=str(vault), agent_id="intruder")
        res = hybrid_search(db, viewer_cfg, "hunter2", agent_id=None)
        rows = [r for r in res if r.get("source_id") == "Projects/public.md"]
        assert rows
        assert "hunter2" in rows[0]["content"]
        assert not rows[0].get("locked")
    finally:
        db.close()


# ===========================================================================
# Tool dispatch: memory_import + memory_search profile_scope
# ===========================================================================


def test_memory_import_tool_runs_vault(provider: RemnantMemoryProvider, vault: Path):
    _write_note(vault / "Inbox" / "a.md", "# A\nA note about Proxmox.")
    res = provider.handle_tool_call(
        "memory_import", {"source": "vault"}, session_id="imp",
    )
    assert "error" not in res
    assert res["source"] == "vault"
    assert res["indexed"] >= 1


def test_memory_import_hindsight_now_implemented(provider: RemnantMemoryProvider):
    # Phase 6: hindsight is now a real import source (monkeypatched in the
    # migration suite to avoid real recall calls); the dispatch returns stats
    # rather than the Phase-4 "not implemented" error.
    res = provider.handle_tool_call(
        "memory_import", {"source": "hindsight", "dry_run": True}, session_id="imp",
    )
    assert "error" not in res
    assert res["source"] == "hindsight"
    assert "stats" in res


def test_memory_import_rejects_unknown_source(provider: RemnantMemoryProvider):
    res = provider.handle_tool_call(
        "memory_import", {"source": "bogus"}, session_id="imp",
    )
    assert "error" in res


def test_memory_import_schema_has_source_enum(provider: RemnantMemoryProvider):
    schemas = provider.get_tool_schemas()
    imp = next(s for s in schemas if s["function"]["name"] == "memory_import")
    assert set(imp["function"]["parameters"]["properties"]["source"]["enum"]) == {
        "vault", "hindsight", "memory_store",
    }


def test_memory_search_schema_has_profile_scope(provider: RemnantMemoryProvider):
    schemas = provider.get_tool_schemas()
    s = next(s for s in schemas if s["function"]["name"] == "memory_search")
    props = s["function"]["parameters"]["properties"]
    assert "profile_scope" in props
    assert props["profile_scope"]["type"] == "array"


def test_memory_search_tool_profile_scope_arg(
    provider: RemnantMemoryProvider, vault: Path
):
    # Both notes mention "alpha" so a single-token query matches both.
    _write_note(vault / "Projects" / "alpha.md", "# Alpha\nAlpha project details.")
    _write_note(vault / "Personal" / "diary.md", "# Diary\nDiary mentions alpha.")
    provider.handle_tool_call("memory_import", {"source": "vault"}, session_id="imp")
    # Without scope: both documents are reachable.
    res = provider.handle_tool_call(
        "memory_search", {"query": "alpha"}, session_id="s",
    )
    ids = {r["source_id"] for r in res["results"] if r.get("source_id")}
    assert "Projects/alpha.md" in ids and "Personal/diary.md" in ids
    # With scope: only Projects/.
    res_scoped = provider.handle_tool_call(
        "memory_search",
        {"query": "alpha", "profile_scope": ["Projects/"]},
        session_id="s",
    )
    scoped_ids = {r["source_id"] for r in res_scoped["results"] if r.get("source_id")}
    assert scoped_ids == {"Projects/alpha.md"}


def test_memory_search_tool_locked_masking_for_other_agent(
    provider: RemnantMemoryProvider, vault: Path
):
    # provider's agent_id is "default" (the owner).
    _write_note(
        vault / "Personal" / "secret.md",
        "# Secret\nhunter2 is the passphrase.",
        frontmatter={"locked": True},
    )
    provider.handle_tool_call("memory_import", {"source": "vault"}, session_id="imp")
    # Owner sees content (BM25 matches the single token "hunter2").
    owner = provider.handle_tool_call(
        "memory_search", {"query": "hunter2"}, session_id="s",
    )
    assert any("hunter2" in r["fact"] for r in owner["results"])

    # A different agent uses a provider configured with a different agent_id so
    # the owner check fails. We build a second provider pointing at the same
    # home so it reads the same DB.
    import json

    home = Path(provider._hermes_home)  # type: ignore[attr-defined]
    cfg_path = home / "remnant.json"
    cfg_path.write_text(json.dumps({
        "vault_path": str(vault), "agent_id": "intruder",
    }))
    intruder = RemnantMemoryProvider()
    intruder.initialize(session_id="intr", hermes_home=str(home))
    try:
        res = intruder.handle_tool_call(
            "memory_search", {"query": "hunter2"}, session_id="intr",
        )
        locked = [r for r in res["results"] if r.get("locked")]
        assert locked
        for r in locked:
            assert "hunter2" not in r["fact"]
            assert r["fact"] == "[locked note: content hidden]"
    finally:
        intruder.shutdown()


# ===========================================================================
# reindex_vault() helper on the provider
# ===========================================================================


def test_provider_reindex_vault_returns_stats(provider: RemnantMemoryProvider, vault: Path):
    _write_note(vault / "Inbox" / "a.md", "# A\nA note.")
    stats = provider.reindex_vault()
    assert stats["indexed"] >= 1
    # Second run is a no-op (hash match).
    stats2 = provider.reindex_vault()
    assert stats2["indexed"] == 0
    assert stats2["skipped"] >= 1


def test_provider_reindex_vault_handles_deleted(provider: RemnantMemoryProvider, vault: Path):
    note = vault / "Inbox" / "a.md"
    _write_note(note, "# A\nA note.")
    provider.reindex_vault()
    os.remove(note)
    stats = provider.reindex_vault()
    assert stats["forgotten"] == 1


def test_provider_system_prompt_mentions_vault_and_import(provider: RemnantMemoryProvider):
    block = provider.system_prompt_block()
    assert "memory_import" in block
    assert "vault" in block.lower()
    assert "profile_scope" in block


# ===========================================================================
# DB helpers for vault_files
# ===========================================================================


def test_get_set_vault_hash_roundtrip(hermes_home: Path):
    db = _open_db(hermes_home)
    try:
        assert db.get_vault_hash("a/b.md") is None
        # memory_id may be None for a tracked-but-not-yet-linked file.
        db.set_vault_hash("a/b.md", "deadbeef", memory_id=None)
        assert db.get_vault_hash("a/b.md") == "deadbeef"
        assert db.get_vault_memory("a/b.md") is None
        # Linking a real memory updates the row.
        mid = db.insert_memory(
            content="doc", source="vault", source_id="a/b.md",
            agent="default", type="document",
        )
        db.set_vault_hash("a/b.md", "cafef00d", memory_id=mid)
        assert db.get_vault_hash("a/b.md") == "cafef00d"
        assert db.get_vault_memory("a/b.md") == mid
    finally:
        db.close()


def test_mark_vault_forgotten_for_missing(hermes_home: Path):
    db = _open_db(hermes_home)
    try:
        mid_keep = db.insert_memory(
            content="keep doc", source="vault", source_id="keep.md",
            agent="default", type="document",
        )
        mid_gone = db.insert_memory(
            content="gone doc", source="vault", source_id="gone.md",
            agent="default", type="document",
        )
        db.set_vault_hash("keep.md", "h1", memory_id=mid_keep)
        db.set_vault_hash("gone.md", "h2", memory_id=mid_gone)
        # Present on disk: keep.md only; gone.md is missing.
        forgotten = db.mark_vault_forgotten_for_missing({"keep.md"})
        assert forgotten == [mid_gone]
        assert db.get_memory(mid_gone)["status"] == "forgotten"
        assert db.get_memory(mid_keep)["status"] == "active"
        assert db.get_vault_hash("gone.md") is None
        assert db.get_vault_hash("keep.md") == "h1"
    finally:
        db.close()


def test_mark_vault_forgotten_single(hermes_home: Path):
    db = _open_db(hermes_home)
    try:
        mid = db.insert_memory(content="doc", source="vault", source_id="x.md",
                               agent="default", type="document")
        db.set_vault_hash("x.md", "h", memory_id=mid)
        gone = db.mark_vault_forgotten("x.md")
        assert gone == mid
        assert db.get_memory(mid)["status"] == "forgotten"
        assert db.get_vault_hash("x.md") is None
    finally:
        db.close()


# --- timing helper sanity ---------------------------------------------------


def test_last_reindex_ts_none_then_set(hermes_home: Path, vault: Path):
    from remnant.vault import last_reindex_ts

    db = _open_db(hermes_home)
    cfg = RemnantConfig(vault_path=str(vault))
    emb = _fake_embed(db, cfg)
    try:
        assert last_reindex_ts(db) is None
        _write_note(vault / "Inbox" / "a.md", "# A\nnote")
        index_vault(db, cfg, emb)
        ts = last_reindex_ts(db)
        assert ts is not None
        # Roughly now (within a generous window).
        assert abs(ts - time.time()) < 10_000
    finally:
        db.close()
