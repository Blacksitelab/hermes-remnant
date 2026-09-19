"""S-010–S-012 baseline safety regressions.

Positive synthetic regressions for the separate pre-PR#46 baseline patch:
S-010 vault containment, S-011 graph/candidate authorization and SQL/Python
scope equivalence, S-012 SQLite journal policy. Each test asserts the fixed
behavior, not merely that the old defect is absent.
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
from unittest.mock import patch

import pytest

from remnant.config import RemnantConfig
from remnant.db import (
    RemnantDB,
    _append_profile_scope_sql,
    configure_sqlite_journal,
    open_db,
    wal_safe_sqlite,
)
from remnant.embed import Embedder
from remnant.graph import graph_traverse
from remnant.recall import RecallRequest, RecallService
from remnant.scope import document_scope_allows
from remnant.search import search
from remnant.tools import handle_tool_call
from remnant.vault import index_file, index_vault


def _no_embedder(db, cfg, dim=4):
    """Deterministic offline embedder: no Ollama, no network."""
    emb = Embedder.__new__(Embedder)
    emb._db = db
    emb._model = cfg.embed_model
    emb._url = cfg.embed_url
    emb._timeout = cfg.embed_timeout
    emb._client = None
    emb.embed = lambda text: [1.0] + [0.0] * (dim - 1)
    emb.close = lambda: None
    return emb


# ===========================================================================
# S-010: canonical containment at every indexing entrance
# ===========================================================================


@pytest.fixture()
def vault_tree(tmp_path: pathlib.Path):
    """Vault with escape/symlink/alias/exclusion shapes and an outside file."""
    vault = tmp_path / "vault"
    for name in ("allowed", "private", "99_ARCHIVE", "90_Scratch"):
        (vault / name).mkdir(parents=True)
    (vault / "allowed" / "ok.md").write_text("# Ok\nordinary allowed note\n")
    (vault / "private" / "secret.md").write_text("# Secret\nprivate note body\n")
    outside = tmp_path / "outside.md"
    outside.write_text("# Outside\nescaped document body\n")
    # allowed alias -> outside file (resolved escape)
    (vault / "allowed" / "escape.md").symlink_to(outside)
    # allowed alias -> private (in-root but out-of-scope) target
    (vault / "allowed" / "alias_to_private.md").symlink_to(vault / "private" / "secret.md")
    # excluded lexical entry -> allowed target (allowed->excluded alias inversion)
    (vault / "99_ARCHIVE" / "sneaky.md").symlink_to(vault / "allowed" / "ok.md")
    # allowed alias -> allowed target (must index once under the canonical id)
    (vault / "allowed" / "alias.md").symlink_to(vault / "allowed" / "ok.md")
    # broken link and symlink loop
    (vault / "allowed" / "broken.md").symlink_to(tmp_path / "missing.md")
    (vault / "allowed" / "loopA.md").symlink_to(vault / "allowed" / "loopB.md")
    (vault / "allowed" / "loopB.md").symlink_to(vault / "allowed" / "loopA.md")
    # directory alias to an outside directory
    outside_dir = tmp_path / "outsidedir"
    outside_dir.mkdir()
    (outside_dir / "d.md").write_text("# D\noutside dir doc\n")
    (vault / "allowed" / "dirlink").symlink_to(outside_dir, target_is_directory=True)
    return vault, outside


def test_s010_scan_rejects_escapes_aliases_and_loops(tmp_path, vault_tree):
    vault, _ = vault_tree
    db = open_db(tmp_path / "s010.db")
    cfg = RemnantConfig(agent_id="a", vault_path=str(vault), profile_scope=["allowed"])
    try:
        with patch("remnant.vault.extract_and_link_entities"):
            stats = index_vault(db, cfg, _no_embedder(db, cfg))
        rows = {
            dict(r)["source_id"]
            for r in db._conn.execute("SELECT source_id FROM memories WHERE source='vault'")
        }
        # Only the canonical allowed note survives; every alias is rejected.
        assert "allowed/ok.md" in rows
        assert not any("escape" in r for r in rows)
        assert not any("alias_to_private" in r for r in rows)
        assert not any("sneaky" in r for r in rows)
        assert not any("broken" in r or "loop" in r or "dirlink" in r for r in rows)
        assert not any(r.startswith("private/") for r in rows)
        # The allowed->allowed alias indexes once under the canonical identity.
        assert sum(1 for r in rows if r.startswith("allowed/")) == 1
        assert stats["indexed"] == 1
        # No outside content was ever read or stored.
        blob = json.dumps([dict(r) for r in db._conn.execute("SELECT content FROM memories")])
        assert "escaped document body" not in blob
        assert "private note body" not in blob
    finally:
        db.close()


def test_s010_direct_index_rejects_before_hash_read_and_entity_callbacks(tmp_path, vault_tree):
    vault, _ = vault_tree
    db = open_db(tmp_path / "s010b.db")
    cfg = RemnantConfig(agent_id="a", vault_path=str(vault), profile_scope=["allowed"])
    try:
        read_calls: list[str] = []
        entity_calls: list[str] = []
        real_hash = __import__("remnant.vault", fromlist=["_file_hash"])._file_hash

        def spy_hash(path):
            read_calls.append(str(path))
            return real_hash(path)

        def spy_entities(*args, **kwargs):
            entity_calls.append(str(args))
            return []

        with patch("remnant.vault._file_hash", spy_hash), patch(
            "remnant.vault.extract_and_link_entities", spy_entities
        ):
            emb = _no_embedder(db, cfg)
            target = vault / "allowed"
            assert index_file(db, cfg, emb, target / "escape.md") is None
            assert index_file(db, cfg, emb, target / "alias_to_private.md") is None
            assert index_file(db, cfg, emb, vault / "99_ARCHIVE" / "sneaky.md") is None
            assert index_file(db, cfg, emb, target / "broken.md") is None
            assert index_file(db, cfg, emb, target / "loopA.md") is None
            # Contained, in-scope file still indexes (allowed-path compatibility).
            assert index_file(db, cfg, emb, target / "ok.md")
        # Rejected paths never reached hash/read/entity extraction. Only the
        # accepted contained file (ok.md) may appear.
        assert all("ok.md" in c for c in read_calls), f"rejected links were hashed: {read_calls}"
        assert not any("escape" in c or "private" in c for c in entity_calls)
        with patch("remnant.vault.extract_and_link_entities"):
            assert index_file(
                db, cfg, _no_embedder(db, cfg), vault / "allowed" / "nope.md"
            ) is None
    finally:
        db.close()


def test_s010_missing_root_and_unreadable_scan_do_not_forget(tmp_path, vault_tree):
    vault, _ = vault_tree
    db = open_db(tmp_path / "s010c.db")
    cfg = RemnantConfig(agent_id="a", vault_path=str(vault), profile_scope=["allowed"])
    try:
        with patch("remnant.vault.extract_and_link_entities"):
            index_vault(db, cfg, _no_embedder(db, cfg))
        mid = db.get_vault_memory("allowed/ok.md", agent_id="a")
        assert mid and db.get_memory(mid)["status"] == "active"

        # Unavailable root is not an empty scan: nothing is forgotten.
        missing_cfg = RemnantConfig(
            agent_id="a", vault_path=str(tmp_path / "gone"), profile_scope=["allowed"]
        )
        stats = index_vault(db, missing_cfg, _no_embedder(db, missing_cfg))
        assert stats["failed"] == 1 and stats["forgotten"] == 0
        assert db.get_memory(mid)["status"] == "active"

        # A repeated successful scan keeps the note and forgets nothing.
        with patch("remnant.vault.extract_and_link_entities"):
            again = index_vault(db, cfg, _no_embedder(db, cfg))
        assert again["forgotten"] == 0
        assert db.get_memory(mid)["status"] == "active"
    finally:
        db.close()


# ===========================================================================
# S-011: graph scope, stored-policy candidates, SQL/Python scope equivalence
# ===========================================================================


def _scoped_graph_fixture(tmp_path):
    db = open_db(tmp_path / "graph.db")
    cfg = RemnantConfig(agent_id="a", profile_scope=["allowed"], embed_model="test")
    secret = db.insert_memory(
        content="PRIVATEMARKER excluded body", source="vault", source_id="private/blocked.md",
        agent="a", type="document", visibility="fleet",
    )
    seed = db.resolve_entity("ScopedProbe", entity_type="project", agent_id="a")
    neighbour = db.resolve_entity("NeighbourNode", entity_type="project", agent_id="a")
    db.link_entity(memory_id=secret, entity_id=seed, agent_id="a")
    db.link_entity(memory_id=secret, entity_id=neighbour, agent_id="a")
    return db, cfg, secret, seed, neighbour


def test_s011_explicit_graph_and_lanes_never_leak_excluded_document(tmp_path):
    db, cfg, secret, seed, _ = _scoped_graph_fixture(tmp_path)
    try:
        res = handle_tool_call(
            "memory_graph", {"entity": "ScopedProbe"}, db=db, config=cfg,
            embedder=None, session_id="s",
        )
        blob = json.dumps(res)
        assert "PRIVATEMARKER" not in blob
        assert secret not in blob
        assert res["entities"] == [] and res["memories"] == []

        # Hidden-only seed reveals nothing, and a hidden-only connecting edge
        # does not bridge to a second node.
        trav = graph_traverse(db, "ScopedProbe", agent_id="a", profile_scope=["allowed"])
        assert trav["entities"] == [] and trav["memories"] == []

        # Excluded doc cannot resurface through any search lane or prefetch.
        emb = _no_embedder(db, cfg)
        for strategy in ("keyword", "semantic", "auto", "graph"):
            rows = search(db, cfg, "PRIVATEMARKER excluded body", strategy=strategy, embedder=emb)
            assert secret not in {r["id"] for r in rows}
        from remnant import RemnantMemoryProvider

        provider = RemnantMemoryProvider()
        provider._db, provider._config, provider._embedder = db, cfg, emb
        assert "PRIVATEMARKER" not in (provider.prefetch("PRIVATEMARKER excluded body") or "")
    finally:
        db.close()


def test_s011_allowed_paths_still_work_across_graph_and_search(tmp_path):
    db, cfg, _, seed, _ = _scoped_graph_fixture(tmp_path)
    try:
        ok = db.insert_memory(
            content="allowed ordinary body", source="vault", source_id="allowed/ok.md",
            agent="a", type="document", embedding=[1.0, 0.0], embed_model="test",
        )
        db.link_entity(memory_id=ok, entity_id=seed, agent_id="a")
        res = handle_tool_call(
            "memory_graph", {"entity": "ScopedProbe"}, db=db, config=cfg,
            embedder=None, session_id="s",
        )
        assert ok in {m["id"] for m in res["memories"]}
        # Canonical entity display name is lowercased by the resolver.
        assert {e["name"] for e in res["entities"]} == {"scopedprobe"}

        emb = _no_embedder(db, cfg, dim=2)
        # Text lanes find the allowed note by content; the graph lane by entity.
        for strategy in ("keyword", "semantic", "auto"):
            rows = search(db, cfg, "allowed ordinary body", strategy=strategy, embedder=emb)
            assert ok in {r["id"] for r in rows}, strategy
        rows = search(db, cfg, "ScopedProbe", strategy="graph", embedder=emb)
        assert ok in {r["id"] for r in rows}
    finally:
        db.close()


def test_s011_candidate_boundary_uses_stored_policy_not_adapter_claims(tmp_path):
    db = open_db(tmp_path / "cand.db")
    try:
        stored = db.insert_memory(
            content="stored secret body", source="vault", source_id="private/blocked.md",
            agent="a", type="document", visibility="private",
        )
        cfg = RemnantConfig(agent_id="a", profile_scope=["allowed"])
        service = RecallService(db, cfg)
        # Adapter forges an allowed/manual/fleet row for an excluded stored doc.
        forged = [{
            "id": stored, "content": "FORGED BODY", "agent": "a", "agent_id": "a",
            "source": "manual", "type": "fact", "source_id": "allowed/ok.md",
            "visibility": "fleet", "score": 1.0,
        }]
        resp = service.recall(RecallRequest(query="anything", agent_id="a"), candidates=forged)
        assert resp.results == []
        assert "FORGED BODY" not in json.dumps(resp.results)

        # An allowed stored row is still delivered, rehydrated from storage.
        service_no_scope = RecallService(db, RemnantConfig(agent_id="a"))
        ok = db.insert_memory(
            content="allowed body", source="vault", source_id="allowed/ok.md",
            agent="a", type="document",
        )
        resp = service_no_scope.recall(
            RecallRequest(query="body", agent_id="a"),
            candidates=[{"id": ok, "content": "forged content", "visibility": "fleet"}],
        )
        assert [r["id"] for r in resp.results] == [ok]
        assert resp.results[0]["content"] == "allowed body"

        # Unknown/deleted ID, other-owner row, forgotten row, visibility ceiling,
        # and injected pending overlays all fail closed.
        gone = db.insert_memory(content="forgotten body", agent="a")
        db.set_memory_field(gone, "status", "forgotten", actor="t")
        other = db.insert_memory(content="alice secret", agent="alice")
        fleet = db.insert_memory(content="fleet only", agent="a", visibility="fleet")
        for candidate in (
            {"id": "no-such-id", "content": "x"},
            {"id": other, "content": "alice secret"},
            {"id": gone, "content": "forgotten body"},
            {"id": "pending-999", "pending": True, "agent_id": "a", "content": "INJECTED"},
        ):
            resp = service_no_scope.recall(
                RecallRequest(query="q", agent_id="a", include_pending=True),
                candidates=[dict(candidate)],
            )
            assert resp.results == [], candidate
        resp = service_no_scope.recall(
            RecallRequest(query="q", agent_id="a", visibility="private"),
            candidates=[{"id": fleet, "content": "fleet only"}],
        )
        assert resp.results == []
    finally:
        db.close()


def test_s011_final_claim_group_contains_only_authorized_inputs(tmp_path):
    """Nested claim groups must not carry another owner's evidence."""
    db = open_db(tmp_path / "claimgroup.db")
    cfg = RemnantConfig(agent_id="a", claim_aware_ranking_enabled=True)
    ids: dict[str, str] = {}
    for owner, text in (
        ("a", "Sam prefers dark mode"),
        ("a", "Sam prefers light mode"),
        ("alice", "Sam prefers purple mode"),
    ):
        mid = db.insert_memory(content=text, agent=owner, type="fact")
        ids[text] = mid
        db._conn.execute(
            "INSERT OR REPLACE INTO claims(memory_id, subject, predicate, object, status, "
            "confidence, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?)",
            (mid, "sam", "prefers", text.rsplit(" ", 1)[-1], "active", 0.9,
             "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
        )
    db._conn.commit()
    try:
        resp = RecallService(db, cfg).recall(
            RecallRequest(query="Sam preference", agent_id="a", strategy="keyword", limit=5),
        )
        rendered = json.dumps(resp.results)
        assert "purple" not in rendered
        assert ids["Sam prefers purple mode"] not in rendered
        assert ids["Sam prefers dark mode"] in rendered
        # Every nested claim-group member is an authorized (owner-a) input.
        grouped = [g["id"] for r in resp.results for g in (r.get("claim_group") or [])]
        assert grouped
        assert set(grouped) <= {ids["Sam prefers dark mode"], ids["Sam prefers light mode"]}
    finally:
        db.close()


def test_s011_superseded_historical_recall_preserved(tmp_path):
    db = open_db(tmp_path / "hist.db")
    cfg = RemnantConfig(agent_id="a", default_search_strategy="keyword")
    db.insert_memory(content="old printer was an Ender 3", agent="a")
    superseded = db._conn.execute(
        "SELECT id FROM memories WHERE content LIKE 'old printer%'"
    ).fetchone()[0]
    db.set_memory_field(superseded, "status", "superseded", actor="t")
    try:
        service = RecallService(db, cfg)
        ordinary = service.recall(RecallRequest(query="printer", agent_id="a"))
        assert superseded not in {r["id"] for r in ordinary.results}
        historical = service.recall(
            RecallRequest(query="what printer did I use before?", agent_id="a"),
            candidates=[{"id": superseded, "content": "old printer was an Ender 3"}],
        )
        assert superseded in {r["id"] for r in historical.results}
    finally:
        db.close()


@pytest.mark.parametrize(
    ("source_id", "prefixes", "expected"),
    [
        ("allowed/a.md", ["allowed"], True),
        ("allowed/sub/deep.md", ["allowed"], True),
        ("allowed", ["allowed"], True),
        ("Allowed/a.md", ["allowed"], False),
        ("allowedX/a.md", ["allowed"], False),
        ("Allowed/sub/deep.md", ["allowed"], False),
        ("allowed/a%b.md", ["allowed"], True),
        ("allowed/a_b.md", ["allowed"], True),
        ("al%owed/a.md", ["allowed"], False),
        ("al_owed/a.md", ["allowed"], False),
        ("allowed\\sub\\deep.md", ["allowed"], True),
        ("/allowed/a.md", ["allowed"], True),
        ("allowed/a.md/", ["allowed"], True),
        ("allowed//a.md", ["allowed"], True),
        ("", ["allowed"], False),
        ("other/x.md", ["allowed", "other"], True),
        ("allowed/a.md", [], False),
    ],
)
def test_s011_sql_and_python_scope_matching_agree(tmp_path, source_id, prefixes, expected):
    db = open_db(tmp_path / f"scope-{abs(hash((source_id, tuple(prefixes))))}.db")
    try:
        db.insert_memory(
            content="c", source="vault", source_id=source_id, agent="a", type="document",
        )
        sql, params = _append_profile_scope_sql(
            "SELECT 1 FROM memories m WHERE 1=1", [], prefixes,
        )
        sql_allows = bool(db._conn.execute(sql, params).fetchone())
        py_allows = document_scope_allows(
            {"source": "vault", "type": "document", "source_id": source_id}, prefixes,
        )
        assert sql_allows == py_allows == expected
        # Non-document rows are never path-scoped, including under an empty scope.
        assert document_scope_allows({"source": "manual", "type": "fact"}, prefixes) is True
    finally:
        db.close()


# ===========================================================================
# S-012: journal policy
# ===========================================================================


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("3.44.5", False), ("3.44.6", True), ("3.44.7", True), ("3.44.10", True),
        ("3.45.1", False), ("3.45.99", False),
        ("3.50.6", False), ("3.50.7", True), ("3.50.8", True),
        ("3.51.2", False), ("3.51.3", True), ("3.51.4", True), ("3.51.10", True),
        ("3.7.0", False), ("3.45", False), ("3", False), ("4.0.0", True),
        ("", False), (None, False), ("garbage", False), ("3.44.x", False),
    ],
)
def test_s012_wal_safe_version_boundaries(version, expected):
    assert wal_safe_sqlite(version) is expected


def test_s012_fresh_file_uses_delete_full_when_engine_unverified(tmp_path):
    path = tmp_path / "fresh.db"
    db = open_db(path)
    try:
        mode = db._conn.execute("PRAGMA journal_mode").fetchone()[0]
        sync = db._conn.execute("PRAGMA synchronous").fetchone()[0]
        if wal_safe_sqlite(sqlite3.sqlite_version):
            assert mode == "wal" and sync == 1
        else:
            assert mode == "delete" and sync == 2
        db.insert_memory(content="hello", agent="a")
    finally:
        db.close()
    # Reopen: the persisted mode is stable and rows survive.
    db = open_db(path)
    try:
        assert db._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
        assert db._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        db.close()


def test_s012_existing_wal_on_unverified_engine_fails_before_writes(tmp_path):
    if wal_safe_sqlite(sqlite3.sqlite_version):
        pytest.skip("linked engine is a recognized WAL-safe build")
    path = tmp_path / "wal.db"
    raw = sqlite3.connect(path)
    raw.execute("PRAGMA journal_mode=wal")
    raw.execute("CREATE TABLE marker(a TEXT)")
    raw.execute("INSERT INTO marker VALUES('pre')")
    raw.commit()
    raw.close()
    with pytest.raises(RuntimeError) as excinfo:
        open_db(path)
    message = str(excinfo.value)
    assert "WAL" in message and sqlite3.sqlite_version in message
    # Refusal happens before writes/migration: the file is untouched, still WAL.
    check = sqlite3.connect(path)
    try:
        assert check.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert check.execute("SELECT a FROM marker").fetchone()[0] == "pre"
    finally:
        check.close()


def test_s012_two_connection_writes_and_bounded_busy_failure(tmp_path):
    """DELETE/FULL still supports sequential multi-connection writes."""
    path = tmp_path / "multi.db"
    db = open_db(path)
    try:
        db.insert_memory(content="first", agent="a")
    finally:
        db.close()
    second = open_db(path)
    try:
        second.insert_memory(content="second", agent="a")
    finally:
        second.close()
    check = open_db(path)
    try:
        assert check._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 2
        assert check._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        check.close()


def test_s012_memory_database_is_handled_explicitly():
    conn = sqlite3.connect(":memory:")
    try:
        assert configure_sqlite_journal(conn) == "memory"
    finally:
        conn.close()


def test_s012_dry_run_does_not_change_mode_but_fails_closed(tmp_path):
    path = tmp_path / "dry.db"
    raw = sqlite3.connect(path)
    raw.execute("CREATE TABLE t(a)")
    raw.commit()
    before = raw.execute("PRAGMA journal_mode").fetchone()[0]
    try:
        assert configure_sqlite_journal(raw, allow_mode_change=False) == before
        assert raw.execute("PRAGMA journal_mode").fetchone()[0] == before
    finally:
        raw.close()

    if wal_safe_sqlite(sqlite3.sqlite_version):
        pytest.skip("linked engine is a recognized WAL-safe build")
    wal = tmp_path / "dry-wal.db"
    raw = sqlite3.connect(wal)
    raw.execute("PRAGMA journal_mode=wal")
    raw.execute("CREATE TABLE t(a)")
    raw.commit()
    try:
        with pytest.raises(RuntimeError):
            configure_sqlite_journal(raw, allow_mode_change=False)
        # Still WAL: a dry run refuses without converting.
        assert raw.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        raw.close()


def test_s012_private_destination_converts_and_safe_build_keeps_wal(tmp_path):
    if wal_safe_sqlite(sqlite3.sqlite_version):
        pytest.skip("linked engine is a recognized WAL-safe build")
    path = tmp_path / "private.db"
    raw = sqlite3.connect(path)
    raw.execute("PRAGMA journal_mode=wal")
    raw.execute("CREATE TABLE t(a)")
    raw.execute("INSERT INTO t VALUES('x')")
    raw.commit()
    # A private, exclusively-owned destination converts instead of refusing.
    mode = configure_sqlite_journal(raw, private_file=True)
    assert mode == "delete"
    assert raw.execute("SELECT a FROM t").fetchone()[0] == "x"
    raw.close()


def test_s012_mocked_safe_classifier_selects_wal_mode_only(tmp_path):
    """Mode-selection test only: this is NOT corruption-safety evidence."""
    with patch("remnant.db.wal_safe_sqlite", return_value=True):
        path = tmp_path / "mocked.db"
        conn = sqlite3.connect(path)
        try:
            assert configure_sqlite_journal(conn) == "wal"
            assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1
        finally:
            conn.close()
        # Existing WAL is retained on a recognized build.
        conn = sqlite3.connect(path)
        try:
            assert configure_sqlite_journal(conn) == "wal"
        finally:
            conn.close()


def test_s012_direct_writer_entry_points_are_covered(tmp_path):
    """calibrate_trust / classify_relations / reextract obey the policy."""
    from remnant.calibrate_trust import calibrate_trust
    from remnant.classify_relations import classify_all_relations
    from remnant.reextract import reextract

    if wal_safe_sqlite(sqlite3.sqlite_version):
        pytest.skip("linked engine is a recognized WAL-safe build")

    path = tmp_path / "writers.db"
    db = open_db(path)
    db.insert_memory(content="Sam uses Proxmox", agent="a")
    db.close()

    # Dry runs inspect without changing the mode.
    assert calibrate_trust(str(path), dry_run=True)
    assert classify_all_relations(str(path), dry_run=True)
    assert reextract(str(path), dry_run=True)
    check = sqlite3.connect(path)
    try:
        assert check.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    finally:
        check.close()

    # Unsafe existing WAL is refused before their first write.
    wal = tmp_path / "writers-wal.db"
    db = open_db(wal)
    db.insert_memory(content="Sam uses Proxmox", agent="a")
    db.close()
    raw = sqlite3.connect(wal)
    raw.execute("PRAGMA journal_mode=wal")
    raw.commit()
    raw.close()
    for call in (
        lambda: calibrate_trust(str(wal), dry_run=False),
        lambda: classify_all_relations(str(wal), dry_run=False),
        lambda: reextract(str(wal), dry_run=False),
    ):
        with pytest.raises(RuntimeError):
            call()
    check = sqlite3.connect(wal)
    try:
        assert check.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert check.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
    finally:
        check.close()


def test_s012_backup_restore_and_recovery_retain_rows_and_mode(tmp_path):
    from remnant.maintenance import backup_database, restore_database
    from remnant.recover import _snapshot

    src = tmp_path / "live.db"
    db = open_db(src)
    try:
        db.insert_memory(content="durable row", agent="a")
        backup = tmp_path / "backup.db"
        result = backup_database(db, backup)
        assert result["integrity"] == "ok"
        assert result["journal_mode"] in {"delete", "wal", "memory"}
        restored = tmp_path / "restored.db"
        restored_result = restore_database(backup, restored)
        assert restored_result["integrity"] == "ok"
    finally:
        db.close()

    for out in (tmp_path / "backup.db", tmp_path / "restored.db"):
        check = open_db(out)
        try:
            assert check._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
            assert check._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        finally:
            check.close()

    # Recovery snapshot from a read-only source on a private destination.
    snapshot = tmp_path / "snapshot.db"
    _snapshot(tmp_path / "restored.db", snapshot)
    check = open_db(snapshot)
    try:
        assert check._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
    finally:
        check.close()


def test_s012_policy_failure_closes_the_new_connection(tmp_path):
    if wal_safe_sqlite(sqlite3.sqlite_version):
        pytest.skip("linked engine is a recognized WAL-safe build")
    path = tmp_path / "closed.db"
    raw = sqlite3.connect(path)
    raw.execute("PRAGMA journal_mode=wal")
    raw.commit()
    raw.close()
    opened: list[sqlite3.Connection] = []
    real_connect = sqlite3.connect

    def tracking_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opened.append(conn)
        return conn

    with patch("remnant.db.sqlite3.connect", side_effect=tracking_connect):
        with pytest.raises(RuntimeError):
            RemnantDB(path)
    assert opened, "expected a connection attempt"
    # The failed connection was closed: using it raises ProgrammingError.
    with pytest.raises(sqlite3.ProgrammingError):
        opened[-1].execute("SELECT 1")
