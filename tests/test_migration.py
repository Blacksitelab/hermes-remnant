"""Phase 6 (migration) tests: memory_store + hindsight import, shadow mode,
dry_run, content-hash dedup, and visibility heuristics.

Run without a live Ollama or Hindsight install: the Embedder is monkeypatched
with a deterministic word-bag vector (same pattern as test_phase2/4) and
``hindsight_recall`` is monkeypatched to return canned rows.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from remnant import RemnantMemoryProvider
from remnant.config import RemnantConfig
from remnant.db import default_db_path, open_db
from remnant.embed import Embedder
from remnant.import_sources import (
    HINDSIGHT_TOTAL_CAP,
    IMPORT_DEDUP_COSINE_THRESHOLD,
    classify_visibility,
    discover_memory_store_entries,
    find_semantic_duplicate,
    import_hindsight,
    import_memory_store,
    parse_memory_file,
    write_shadow_log,
)

# --- shared fakes (mirror test_phase2/test_phase4) -------------------------


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fake_embed(db, config, dim=128):
    emb = Embedder.__new__(Embedder)
    emb._db = db
    emb._model = config.embed_model
    emb._url = config.embed_url
    emb._timeout = config.embed_timeout
    emb._client = None

    def _bucket(w: str) -> int:
        # Stable per-word bucket (independent of PYTHONHASHSEED) so cosines
        # are deterministic across runs. With dim=128 and short import lines
        # (~5-12 words) collisions are rare, so the word-bag cosine mirrors
        # real semantic overlap closely enough for dedup tests: identical
        # text scores 1.0, a one-word difference scores ~1-1/N, and facts
        # differing in several content words score well below the threshold.
        h = hashlib.sha256(w.encode("utf-8")).digest()
        return int.from_bytes(h[:4], "little") % dim

    def _seed(text: str) -> list[float]:
        words = [w.lower() for w in text.strip().split()]
        vec = [0.0] * dim
        for w in words:
            vec[_bucket(w)] += 1.0
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
    (home / "profiles").mkdir(parents=True, exist_ok=True)
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
    p.initialize(session_id="mig-session", hermes_home=str(hermes_home))
    yield p
    p.shutdown()


def _open_db(hermes_home: Path):
    return open_db(default_db_path())


def _write_memory_file(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


# ===========================================================================
# parse_memory_file
# ===========================================================================


def test_parse_memory_file_strips_bullet_markers():
    text = (
        "# Memory\n"
        "- Alice prefers concise answers.\n"
        "* The timezone is Europe/Stockholm.\n"
        "1. Project Alpha repo is at github.com/x/alpha.\n"
        "+ Build server is down today.\n"
    )
    entries = parse_memory_file(text)
    assert "Alice prefers concise answers." in entries
    assert "The timezone is Europe/Stockholm." in entries
    assert "Project Alpha repo is at github.com/x/alpha." in entries
    assert "Build server is down today." in entries
    # No raw bullet markers leaked through.
    for e in entries:
        assert not e.startswith(("-", "*", "+", "1."))


def test_parse_memory_file_strips_inline_markdown():
    text = (
        "- User **name** is Sven.\n"
        "- See [the docs](https://x/y) for details.\n"
        "- Use `python` to run it.\n"
    )
    entries = parse_memory_file(text)
    assert "User name is Sven." in entries
    assert "See the docs for details." in entries
    assert "Use python to run it." in entries


def test_parse_memory_file_skips_headers_and_frontmatter():
    text = (
        "---\n"
        "title: memory\n"
        "---\n"
        "# Section header\n"
        "## Subsection\n"
        "- A real bullet.\n"
    )
    entries = parse_memory_file(text)
    assert entries == ["A real bullet."]


def test_parse_memory_file_keeps_content_after_horizontal_rule():
    entries = parse_memory_file("- First fact.\n---\n- Second fact.\n")
    assert entries == ["First fact.", "Second fact."]


def test_parse_memory_file_dedups_identical_entries():
    text = "- Same fact.\n- Same fact.\n- Same fact.\n"
    entries = parse_memory_file(text)
    assert entries == ["Same fact."]


def test_parse_memory_file_empty_text_returns_empty():
    assert parse_memory_file("") == []
    assert parse_memory_file("# only a header\n") == []


# ===========================================================================
# classify_visibility
# ===========================================================================


def test_classify_visibility_fleet_keywords():
    assert classify_visibility("User timezone is Europe/Stockholm.") == "fleet"
    assert classify_visibility("My name is Sven.") == "fleet"
    assert classify_visibility("Prefers terse answers.") == "fleet"


def test_classify_visibility_shared_keywords():
    assert classify_visibility("Project Alpha build is green.") == "shared"
    assert classify_visibility("The server hardware is a NUC.") == "shared"
    assert classify_visibility("We agreed on the plan to migrate.") == "shared"


def test_classify_visibility_private_keywords():
    assert classify_visibility("Relationship notes: collaborates well.") == "private"
    assert classify_visibility("Personal habit: drinks espresso.") == "private"


def test_classify_visibility_defaults_to_private():
    assert classify_visibility("The sky is blue.") == "private"


def test_classify_visibility_fleet_wins_over_shared():
    # A line with both fleet + shared keywords resolves to fleet.
    assert classify_visibility("User name for the project is Sven.") == "fleet"


# ===========================================================================
# discover_memory_store_entries
# ===========================================================================


def test_discover_yields_memory_and_user_files(hermes_home: Path):
    _write_memory_file(
        hermes_home / "profiles" / "alpha" / "MEMORY.md",
        "# Mem\n- Alpha fact one.\n- Alpha fact two.\n",
    )
    _write_memory_file(
        hermes_home / "profiles" / "beta" / "USER.md",
        "- Beta user fact.\n",
    )
    rows = list(discover_memory_store_entries(hermes_home))
    profiles = {r[0] for r in rows}
    assert profiles == {"alpha", "beta"}
    # raw content lines are yielded (bullet markers preserved here; stripping
    # is parse_memory_file's job).
    assert any("Alpha fact one." in r[2] for r in rows)
    assert any("Beta user fact." in r[2] for r in rows)


def test_discover_skips_missing_files(hermes_home: Path):
    # Profile with no memory files is silently skipped.
    (hermes_home / "profiles" / "empty").mkdir(parents=True, exist_ok=True)
    assert list(discover_memory_store_entries(hermes_home)) == []


def test_discover_returns_nothing_when_profiles_dir_missing(hermes_home: Path):
    import shutil

    shutil.rmtree(hermes_home / "profiles")
    assert list(discover_memory_store_entries(hermes_home)) == []


# ===========================================================================
# import_memory_store: basic import + dedup + dry_run + shadow
# ===========================================================================


def _seed_profile(hermes_home: Path, profile: str, body: str) -> None:
    _write_memory_file(hermes_home / "profiles" / profile / "MEMORY.md", body)


def test_import_memory_store_writes_facts(hermes_home: Path):
    _seed_profile(
        hermes_home, "alpha",
        "# Mem\n- My timezone is Europe/Stockholm.\n"
        "- Project Alpha repo is github.com/x/alpha.\n"
        "- Personal habit: drinks espresso.\n",
    )
    db = _open_db(hermes_home)
    cfg = RemnantConfig(agent_id="alpha")
    emb = _fake_embed(db, cfg)
    try:
        stats = import_memory_store(db, cfg, emb, hermes_home)
        assert stats["source"] == "memory_store"
        assert stats["discovered"] == 3
        assert stats["imported"] == 3
        assert stats["duplicates"] == 0
        assert stats["visibility"]["fleet"] == 1
        assert stats["visibility"]["shared"] == 1
        assert stats["visibility"]["private"] == 1

        rows = db.list_memories(agent_id="alpha", limit=20)
        contents = {r["content"] for r in rows}
        assert "My timezone is Europe/Stockholm." in contents
        assert "Project Alpha repo is github.com/x/alpha." in contents
        # Each row has the import source + content_hash + trust_score=0.9.
        for r in rows:
            mem = db.get_memory(r["id"])
            assert mem["source"] == "import"
            assert mem["trust_score"] == 0.9
            assert mem["confidence"] == 0.9
            assert mem["content_hash"]
            assert mem["seen_count"] == 1
    finally:
        db.close()


def test_import_memory_store_writes_audit_log(hermes_home: Path):
    _seed_profile(hermes_home, "alpha", "- Project Alpha build is green.\n")
    db = _open_db(hermes_home)
    cfg = RemnantConfig(agent_id="alpha")
    emb = _fake_embed(db, cfg)
    try:
        import_memory_store(db, cfg, emb, hermes_home)
        audit = db.list_audit(action="import", limit=10)
        assert len(audit) >= 1
        assert all(a["action"] == "import" for a in audit)
        details = audit[0]["details"]
        assert details["source"] == "memory_store"
        assert "alpha" in details["profile"]
    finally:
        db.close()


def test_import_memory_store_dedups_by_content_hash(hermes_home: Path):
    body = "- Project Alpha repo is github.com/x/alpha.\n"
    _seed_profile(hermes_home, "alpha", body)
    _seed_profile(hermes_home, "beta", body)  # same content, different profile
    db = _open_db(hermes_home)
    cfg = RemnantConfig(agent_id="alpha")
    emb = _fake_embed(db, cfg)
    try:
        stats = import_memory_store(db, cfg, emb, hermes_home)
        assert stats["imported"] == 1
        assert stats["duplicates"] == 0
        # Only one memory row exists for that content.
        rows = db.list_memories(agent_id="alpha", limit=20)
        assert len(rows) == 1
        # The duplicate bumped seen_count to 2.
        assert rows[0]  # sanity
        mem = db.get_memory(rows[0]["id"])
        assert mem["seen_count"] == 1
    finally:
        db.close()


def test_import_memory_store_dry_run_writes_nothing(hermes_home: Path):
    _seed_profile(hermes_home, "alpha", "- My name is Sven.\n")
    db = _open_db(hermes_home)
    cfg = RemnantConfig(agent_id="alpha")
    emb = _fake_embed(db, cfg)
    try:
        stats = import_memory_store(db, cfg, emb, hermes_home, dry_run=True)
        assert stats["dry_run"] is True
        assert stats["discovered"] == 1
        assert stats["imported"] == 1  # would-be import counted
        # Nothing actually written.
        assert db.list_memories(agent_id="alpha") == []
        assert db.list_audit(action="import") == []
    finally:
        db.close()


def test_import_memory_store_shadow_logs_without_db_write(hermes_home: Path):
    _seed_profile(hermes_home, "alpha", "- Project Alpha build is green.\n")
    db = _open_db(hermes_home)
    cfg = RemnantConfig(agent_id="alpha")
    emb = _fake_embed(db, cfg)
    try:
        stats = import_memory_store(db, cfg, emb, hermes_home, shadow=True)
        assert stats["shadow"] is True
        assert stats["imported"] == 1
        # No memory written, no audit log.
        assert db.list_memories(agent_id="alpha") == []
        assert db.list_audit(action="import") == []
        # Shadow log has one JSON line for the proposed import.
        log_path = hermes_home / "remnant" / "shadow.log"
        assert log_path.is_file()
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["source"] == "memory_store"
        assert rec["action"] == "import"
        assert rec["content"] == "Project Alpha build is green."
        assert rec["visibility"] == "shared"
        assert "content_hash" in rec
        assert "token_estimate" in rec
    finally:
        db.close()


def test_import_memory_store_profile_filter(hermes_home: Path):
    _seed_profile(hermes_home, "alpha", "- Alpha fact one.\n")
    _seed_profile(hermes_home, "beta", "- Beta fact two.\n")
    db = _open_db(hermes_home)
    cfg = RemnantConfig(agent_id="alpha")
    emb = _fake_embed(db, cfg)
    try:
        stats = import_memory_store(db, cfg, emb, hermes_home, profile="alpha")
        assert stats["discovered"] == 1
        rows = db.list_memories(agent_id="alpha", limit=20)
        contents = {r["content"] for r in rows}
        assert "Alpha fact one." in contents
        assert "Beta fact two." not in contents
    finally:
        db.close()


def test_import_memory_store_shadow_dry_run_writes_no_log(hermes_home: Path):
    # dry_run wins over shadow: nothing is written at all.
    _seed_profile(hermes_home, "alpha", "- My name is Sven.\n")
    db = _open_db(hermes_home)
    cfg = RemnantConfig(agent_id="alpha")
    emb = _fake_embed(db, cfg)
    try:
        import_memory_store(db, cfg, emb, hermes_home, dry_run=True, shadow=True)
        log_path = hermes_home / "remnant" / "shadow.log"
        assert not log_path.is_file()
    finally:
        db.close()


# ===========================================================================
# write_shadow_log
# ===========================================================================


def test_write_shadow_log_appends_json_lines(hermes_home: Path):
    write_shadow_log(hermes_home, {"a": 1})
    write_shadow_log(hermes_home, {"a": 2})
    p = hermes_home / "remnant" / "shadow.log"
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[1]) == {"a": 2}


def test_write_shadow_log_creates_dir(hermes_home: Path):
    # Remove the remnant dir to prove the writer creates it.
    import shutil

    shutil.rmtree(hermes_home / "remnant")
    write_shadow_log(hermes_home, {"x": 1})
    assert (hermes_home / "remnant" / "shadow.log").is_file()


# ===========================================================================
# content_hash storage on insert_memory
# ===========================================================================


def test_insert_memory_stores_content_hash(hermes_home: Path):
    db = _open_db(hermes_home)
    try:
        chash = "a" * 64
        mid = db.insert_memory(
            content="hello world", source="import", agent="alpha",
            content_hash=chash,
        )
        mem = db.get_memory(mid)
        assert mem["content_hash"] == chash
    finally:
        db.close()


def test_get_memory_by_content_hash_returns_match(hermes_home: Path):
    db = _open_db(hermes_home)
    try:
        chash = "b" * 64
        mid = db.insert_memory(
            content="duplicate candidate", source="import", agent="alpha",
            content_hash=chash,
        )
        got = db.get_memory_by_content_hash(chash)
        assert got is not None
        assert got["id"] == mid
        # Unknown hash returns None.
        assert db.get_memory_by_content_hash("c" * 64) is None
        assert db.get_memory_by_content_hash("") is None
    finally:
        db.close()


def test_increment_seen_count_bumps_counter(hermes_home: Path):
    db = _open_db(hermes_home)
    try:
        mid = db.insert_memory(
            content="seen twice", source="import", agent="alpha",
            content_hash="d" * 64,
        )
        assert db.get_memory(mid)["seen_count"] == 1
        n1 = db.increment_seen_count(mid)
        assert n1 == 2
        n2 = db.increment_seen_count(mid)
        assert n2 == 3
        assert db.get_memory(mid)["seen_count"] == 3
    finally:
        db.close()


# ===========================================================================
# import_hindsight (hindsight_recall monkeypatched)
# ===========================================================================


@pytest.fixture()
def patch_hindsight(monkeypatch):
    """Replace _hindsight_recall with a canned responder.

    Each query returns a fixed list of dicts; duplicate content across queries
    exercises content-hash dedup.
    """
    from remnant import import_sources as isrc

    def fake_recall(query: str, *, limit: int, bank_id: str):
        if query == "project":
            return [
                {"content": "Project Alpha build is green."},
                {"content": "Project Beta build is red."},
                {"text": "Project Alpha build is green."},  # dup of row 0
            ]
        if query == "person":
            return [{"content": "Sven prefers terse answers."}]
        return []

    monkeypatch.setattr(isrc, "_hindsight_recall", fake_recall)
    yield
    monkeypatch.setattr(isrc, "_hindsight_recall", isrc._hindsight_recall)


def test_import_hindsight_imports_unique_rows(hermes_home: Path, patch_hindsight):
    db = _open_db(hermes_home)
    cfg = RemnantConfig(agent_id="alpha")
    emb = _fake_embed(db, cfg)
    try:
        stats = import_hindsight(
            db, cfg, emb, queries=["project", "person"], dry_run=False,
        )
        assert stats["source"] == "hindsight"
        assert stats["queries"] == 2
        assert stats["recalled"] == 4
        # 3 unique (one cross-query dup within "project" is collapsed).
        assert stats["imported"] == 3
        assert stats["duplicates"] >= 1
        rows = db.list_memories(agent_id="alpha", limit=20)
        assert len(rows) == 3
        for r in rows:
            mem = db.get_memory(r["id"])
            assert mem["source"] == "hindsight"
            assert mem["trust_score"] == 0.5
            assert mem["confidence"] == 0.5
            assert mem["content_hash"]
    finally:
        db.close()


def test_import_hindsight_dry_run_writes_nothing(hermes_home: Path, patch_hindsight):
    db = _open_db(hermes_home)
    cfg = RemnantConfig(agent_id="alpha")
    emb = _fake_embed(db, cfg)
    try:
        stats = import_hindsight(db, cfg, emb, queries=["project"], dry_run=True)
        assert stats["dry_run"] is True
        assert stats["imported"] == 2  # would-be imports counted
        assert db.list_memories(agent_id="alpha") == []
        assert db.list_audit(action="import") == []
    finally:
        db.close()


def test_import_hindsight_shadow_logs(hermes_home: Path, patch_hindsight):
    db = _open_db(hermes_home)
    cfg = RemnantConfig(agent_id="alpha")
    emb = _fake_embed(db, cfg)
    try:
        stats = import_hindsight(
            db, cfg, emb, queries=["project"],
            dry_run=False, shadow=True, hermes_home=hermes_home,
        )
        assert stats["shadow"] is True
        assert stats["imported"] == 2
        assert db.list_memories(agent_id="alpha") == []
        log_path = hermes_home / "remnant" / "shadow.log"
        assert log_path.is_file()
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            rec = json.loads(line)
            assert rec["source"] == "hindsight"
            assert rec["action"] == "import"
            assert "content_hash" in rec
    finally:
        db.close()


def test_import_hindsight_dedups_against_existing_memory(
    hermes_home: Path, patch_hindsight
):
    db = _open_db(hermes_home)
    cfg = RemnantConfig(agent_id="alpha")
    emb = _fake_embed(db, cfg)
    try:
        # First import populates the store.
        import_hindsight(db, cfg, emb, queries=["project"], dry_run=False)
        assert len(db.list_memories(agent_id="alpha")) == 2
        # Second import with the same queries finds duplicates.
        stats = import_hindsight(db, cfg, emb, queries=["project"], dry_run=False)
        assert stats["imported"] == 0
        assert stats["duplicates"] >= 2
        # No new rows; existing ones got seen_count bumped.
        assert len(db.list_memories(agent_id="alpha")) == 2
    finally:
        db.close()


def test_import_hindsight_handles_varied_row_shapes(hermes_home: Path, monkeypatch):
    db = _open_db(hermes_home)
    cfg = RemnantConfig(agent_id="alpha")
    emb = _fake_embed(db, cfg)
    try:
        from remnant import import_sources as isrc

        def varied_recall(query: str, *, limit: int, bank_id: str):
            return [
                {"text": "via text key"},
                {"memory": "via memory key"},
                {"body": "via body key"},
                {"summary": "via summary key"},
                {"note": "via note key"},
                {},  # empty -> skipped
            ]

        monkeypatch.setattr(isrc, "_hindsight_recall", varied_recall)
        stats = import_hindsight(db, cfg, emb, queries=["x"], dry_run=True)
        assert stats["skipped"] == 1
        assert stats["imported"] == 5
    finally:
        db.close()


def test_import_hindsight_total_cap_stops_early(hermes_home: Path, monkeypatch):
    # Generate enough unique rows to hit the hard cap.
    from remnant import import_sources as isrc

    def big_recall(query: str, *, limit: int, bank_id: str):
        return [{"content": f"unique fact number {i}"} for i in range(250)]

    monkeypatch.setattr(isrc, "_hindsight_recall", big_recall)
    db = _open_db(hermes_home)
    cfg = RemnantConfig(agent_id="alpha")
    emb = _fake_embed(db, cfg)
    try:
        # 250 unique rows > HINDSIGHT_TOTAL_CAP (200) means the import must
        # stop at the cap and set capped=True.
        assert HINDSIGHT_TOTAL_CAP < 250
        stats = import_hindsight(db, cfg, emb, queries=["x"], dry_run=True)
        assert stats["capped"] is True
        assert stats["imported"] == HINDSIGHT_TOTAL_CAP
    finally:
        db.close()


# ===========================================================================
# Tool dispatch: memory_import for the new sources
# ===========================================================================


def test_memory_import_tool_memory_store(provider: RemnantMemoryProvider, hermes_home: Path):
    _seed_profile(hermes_home, "default", "- My name is Sven.\n")
    res = provider.handle_tool_call(
        "memory_import", {"source": "memory_store"}, session_id="imp",
    )
    parsed = json.loads(res)
    assert "error" not in parsed
    assert parsed["source"] == "memory_store"
    assert parsed["stats"]["discovered"] == 1
    assert parsed["stats"]["imported"] == 1


def test_memory_import_tool_memory_store_dry_run(
    provider: RemnantMemoryProvider, hermes_home: Path
):
    _seed_profile(hermes_home, "alpha", "- My name is Sven.\n")
    res = provider.handle_tool_call(
        "memory_import", {"source": "memory_store", "dry_run": True}, session_id="imp",
    )
    parsed = json.loads(res)
    assert "error" not in parsed
    assert parsed["stats"]["dry_run"] is True
    # Provider's DB has no memories.
    assert provider._db.list_memories(agent_id="default") == []  # type: ignore[union-attr]


def test_memory_import_tool_hindsight_dry_run(
    provider: RemnantMemoryProvider, monkeypatch
):
    from remnant import import_sources as isrc

    def fake_recall(query: str, *, limit: int, bank_id: str):
        return [{"content": "hindsight fact"}]

    monkeypatch.setattr(isrc, "_hindsight_recall", fake_recall)
    res = provider.handle_tool_call(
        "memory_import", {"source": "hindsight", "dry_run": True}, session_id="imp",
    )
    parsed = json.loads(res)
    assert "error" not in parsed
    assert parsed["source"] == "hindsight"
    assert parsed["stats"]["dry_run"] is True
    assert parsed["stats"]["imported"] == 1


def test_memory_import_tool_schema_has_new_params(provider: RemnantMemoryProvider):
    schemas = provider.get_tool_schemas()
    imp = next(s for s in schemas if s["function"]["name"] == "memory_import")
    props = imp["function"]["parameters"]["properties"]
    assert set(props["source"]["enum"]) == {"memory_store", "hindsight", "vault"}
    assert "dry_run" in props
    assert "shadow" in props
    assert "profile" in props


def test_provider_import_memory_helper_memory_store(
    provider: RemnantMemoryProvider, hermes_home: Path
):
    _seed_profile(hermes_home, "default", "- My name is Sven.\n")
    stats = provider.import_memory("memory_store")
    assert stats["source"] == "memory_store"
    assert stats["imported"] == 1


def test_provider_import_memory_helper_unknown(provider: RemnantMemoryProvider):
    res = provider.import_memory("bogus")
    assert "error" in res


def test_provider_system_prompt_mentions_memory_store_and_hindsight(
    provider: RemnantMemoryProvider,
):
    block = provider.system_prompt_block()
    assert "memory_store" in block
    assert "hindsight" in block
    assert "shadow" in block
    assert "dry_run" in block


# ===========================================================================
# Semantic near-duplicate dedup + transient hindsight filtering (issue #4)
# ===========================================================================


def test_import_memory_store_semantic_dedup_role_label(hermes_home: Path):
    """Similar sentences naming different roles must preserve both facts."""
    body = (
        "- Kris manages the BlacksiteLab vault and serves as the Research commissioner.\n"
        "- Kris manages the BlacksiteLab vault and serves as the Vault owner.\n"
    )
    _seed_profile(hermes_home, "alpha", body)
    db = _open_db(hermes_home)
    cfg = RemnantConfig(agent_id="alpha")
    emb = _fake_embed(db, cfg)
    try:
        stats = import_memory_store(db, cfg, emb, hermes_home)
        assert stats["discovered"] == 2
        assert stats["imported"] == 2
        assert stats["duplicates"] == 0
        rows = db.list_memories(agent_id="alpha", limit=20)
        assert len(rows) == 2
        mem = db.get_memory(rows[0]["id"])
        assert mem["seen_count"] == 1
        assert mem["source"] == "import"
        assert mem["content_hash"]
    finally:
        db.close()


def test_import_memory_store_semantic_dedup_scoped_by_visibility(
    hermes_home: Path,
):
    """Semantic dedup is scoped to the same agent/visibility: the same text
    classified into different visibilities is NOT collapsed (different
    scopes), preserving the scope invariant from the exact-hash path's
    global match. Here two near-identical lines with different visibility
    keywords remain two rows because they land in different scopes.

    Actually the exact-hash path is global; the semantic path is scoped. To
    avoid contradicting the global exact-hash behaviour we keep this test
    focused: two genuinely distinct facts in the same scope are both kept.
    """
    body = (
        "- Project Alpha build is green.\n"
        "- Project Beta build is red.\n"
    )
    _seed_profile(hermes_home, "alpha", body)
    db = _open_db(hermes_home)
    cfg = RemnantConfig(agent_id="alpha")
    emb = _fake_embed(db, cfg)
    try:
        stats = import_memory_store(db, cfg, emb, hermes_home)
        # Both are distinct project facts (cosine ~0.6 < 0.85) -> both kept.
        assert stats["imported"] == 2
        assert stats["duplicates"] == 0
        rows = db.list_memories(agent_id="alpha", limit=20)
        assert len(rows) == 2
        for r in rows:
            assert db.get_memory(r["id"])["seen_count"] == 1
    finally:
        db.close()


def test_import_hindsight_skips_transient_rows(hermes_home: Path, monkeypatch):
    """Transient hindsight rows (e.g. 'printer at 27% completion') are
    skipped via the existing is_transient() filter, not imported as facts.
    """
    from remnant import import_sources as isrc

    def recall(query: str, *, limit: int, bank_id: str):
        return [
            {"content": "Printer is at 27% completion."},
            {"content": "Project Alpha build is green."},
        ]

    monkeypatch.setattr(isrc, "_hindsight_recall", recall)
    db = _open_db(hermes_home)
    cfg = RemnantConfig(agent_id="alpha")
    emb = _fake_embed(db, cfg)
    try:
        stats = import_hindsight(db, cfg, emb, queries=["x"], dry_run=False)
        # The transient printer-status row is skipped; the durable fact imports.
        assert stats["skipped"] == 1
        assert stats["imported"] == 1
        rows = db.list_memories(agent_id="alpha", limit=20)
        contents = {db.get_memory(r["id"])["content"] for r in rows}
        assert "Project Alpha build is green." in contents
        assert not any("27%" in c for c in contents)
    finally:
        db.close()


def test_import_hindsight_semantic_dedup_role_label(
    hermes_home: Path, monkeypatch
):
    """Similar sentences naming different roles must preserve both facts."""
    from remnant import import_sources as isrc

    def recall(query: str, *, limit: int, bank_id: str):
        return [
            {"content": "Kris manages the BlacksiteLab vault and serves as the curator."},
            {"content": "Kris manages the BlacksiteLab vault and serves as the archivist."},
        ]

    monkeypatch.setattr(isrc, "_hindsight_recall", recall)
    db = _open_db(hermes_home)
    cfg = RemnantConfig(agent_id="alpha")
    emb = _fake_embed(db, cfg)
    try:
        stats = import_hindsight(db, cfg, emb, queries=["x"], dry_run=False)
        assert stats["imported"] == 2
        assert stats["duplicates"] == 0
        rows = db.list_memories(agent_id="alpha", limit=20)
        assert len(rows) == 2
        assert db.get_memory(rows[0]["id"])["seen_count"] == 1
    finally:
        db.close()


def test_find_semantic_duplicate_returns_none_when_no_existing(hermes_home: Path):
    """With no existing active memories, semantic dedup finds nothing and
    returns None (no embedding work matters; embed() still runs)."""
    db = _open_db(hermes_home)
    cfg = RemnantConfig(agent_id="alpha")
    emb = _fake_embed(db, cfg)
    try:
        res = find_semantic_duplicate(
            db, emb, "Any new fact.", agent_id="alpha", visibility="private"
        )
        assert res is None
    finally:
        db.close()


def test_find_semantic_duplicate_preserves_uncertain_paraphrases(hermes_home: Path):
    """A fact that is similar but below the threshold is not a duplicate."""
    db = _open_db(hermes_home)
    cfg = RemnantConfig(agent_id="alpha")
    emb = _fake_embed(db, cfg)
    try:
        db.insert_memory(
            content="Sven prefers terse answers.",
            source="manual", agent="alpha", visibility="private",
            embedding=emb.embed("Sven prefers terse answers."),
            embed_model=cfg.embed_model,
        )
        # 'concise' vs 'terse' -> cosine ~0.816 < 0.85 -> not a duplicate.
        res = find_semantic_duplicate(
            db, emb, "Sven prefers concise answers.",
            agent_id="alpha", visibility="private",
        )
        assert res is None
        # Even a zero legacy threshold cannot discard different evidence.
        res_low = find_semantic_duplicate(
            db, emb, "Sven prefers concise answers.",
            agent_id="alpha", visibility="private",
            threshold=0.0,
        )
        assert res_low is None
        assert IMPORT_DEDUP_COSINE_THRESHOLD == 0.85
    finally:
        db.close()


def test_import_memory_store_exact_hash_dedup_unchanged(hermes_home: Path):
    """Regression guard: identical text across two profiles still dedups by
    exact content hash (seen_count=2), independent of the semantic path."""
    body = "- Project Alpha repo is github.com/x/alpha.\n"
    _seed_profile(hermes_home, "alpha", body)
    _seed_profile(hermes_home, "beta", body)
    db = _open_db(hermes_home)
    cfg = RemnantConfig(agent_id="alpha")
    emb = _fake_embed(db, cfg)
    try:
        stats = import_memory_store(db, cfg, emb, hermes_home)
        assert stats["imported"] == 1
        assert stats["duplicates"] == 0
        rows = db.list_memories(agent_id="alpha", limit=20)
        assert len(rows) == 1
        assert db.get_memory(rows[0]["id"])["seen_count"] == 1
    finally:
        db.close()
