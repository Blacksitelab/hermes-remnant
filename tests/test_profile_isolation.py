"""No visibility label, caller override, or background path grants another profile access."""

import json

import pytest

from remnant import RemnantMemoryProvider
from remnant.config import RemnantConfig
from remnant.db import open_db
from remnant.dream import day_dream
from remnant.graph import graph_traverse
from remnant.import_sources import import_hindsight, import_memory_store
from remnant.recall import RecallRequest, RecallService
from remnant.search import search
from remnant.vault import index_file, index_vault


class Vectors:
    _model = "test"
    _dim = 2

    def embed(self, text, **kwargs):
        return [1.0, 0.0]


@pytest.fixture
def db(tmp_path):
    database = open_db(tmp_path / "profiles.db")
    yield database
    database.close()


def make_provider(db, agent):
    p = RemnantMemoryProvider()
    p._db = db
    p._config = RemnantConfig(agent_id=agent, embed_model="test", embed_dim=2,
                              echo_enabled=False)
    p._embedder = Vectors()
    return p


@pytest.mark.parametrize("visibility", ["private", "shared", "fleet"])
@pytest.mark.parametrize("source", ["conversation", "vault"])
def test_all_retrieval_lanes_exclude_other_profiles(db, visibility, source):
    cfg = RemnantConfig(agent_id="bob", embed_model="test")
    alice = db.insert_memory(content="Alice secret project", source=source, agent="alice",
                             visibility=visibility, embedding=[1., 0.], embed_model="test")
    bob = db.insert_memory(content="Bob project", agent="bob", embedding=[1., 0.],
                           embed_model="test")
    for strategy in ("keyword", "semantic", "auto", "graph"):
        rows = search(db, cfg, "project", strategy=strategy, embedder=Vectors())
        assert alice not in {row["id"] for row in rows}
        if strategy != "graph":
            assert bob in {row["id"] for row in rows}
    service = RecallService(db, cfg)
    result = service.recall(RecallRequest(query="project", agent_id="bob"), candidates=[
        {"id": alice, "content": "Alice secret project", "agent": "bob", "score": 1.},
    ])
    assert result.results == []
    assert "Alice secret" not in make_provider(db, "bob").prefetch("project")


def test_caller_cannot_override_profile_or_edit_guessed_memory_id(db):
    alice = db.insert_memory(content="Alice secret project", agent="alice", visibility="fleet")
    p = make_provider(db, "bob")
    result = json.loads(p.handle_tool_call("memory_search", {"query": "secret"}, agent_id="alice"))
    assert result["results"] == []
    for action in ("update", "forget", "feedback", "share"):
        result = json.loads(p.handle_tool_call("memory_edit", {
            "action": action, "memory_id": alice, "content": "Changed", "feedback": "useful",
        }, agent_id="alice"))
        assert result.get("error")
        assert "Alice secret project" not in json.dumps(result)
    assert db.get_memory(alice)["status"] == "active"


def test_graph_does_not_expose_other_profiles_entity_names(db):
    root = db.resolve_entity("Common Project")  # Legacy global entity.
    secret = db.resolve_entity("Alice Secret Client", agent_id="alice")
    alice = db.insert_memory(content="Alice secret relationship", agent="alice", visibility="fleet")
    bob = db.insert_memory(content="Bob common project", agent="bob")
    db.link_entity(memory_id=alice, entity_id=root, agent_id="alice")
    db.link_entity(memory_id=bob, entity_id=root, agent_id="bob")
    db.link_entity(memory_id=alice, entity_id=secret, agent_id="alice")
    db.add_relation(entity_a=root, entity_b=secret, source_memory_id=alice)
    for evidence_only in (False, True):
        result = graph_traverse(db, "Common Project", agent_id="bob", evidence_only=evidence_only)
        assert "Alice Secret" not in json.dumps(result)
        assert alice not in {row["id"] for row in result["memories"]}


def test_vault_same_path_updates_and_deletions_are_profile_owned(db, tmp_path):
    note = tmp_path / "policy.md"
    note.write_text("# Policy\nAlice original policy")
    alice_cfg = RemnantConfig(agent_id="alice", vault_path=str(tmp_path), embed_dim=2)
    bob_cfg = RemnantConfig(agent_id="bob", vault_path=str(tmp_path), embed_dim=2)
    alice = index_file(db, alice_cfg, Vectors(), note)
    note.write_text("# Policy\nBob replacement policy")
    bob = index_file(db, bob_cfg, Vectors(), note)
    assert alice != bob
    assert "Alice original" in db.get_memory(alice)["content"]
    assert db.get_vault_memory("policy.md", agent_id="alice") == alice
    assert db.get_vault_memory("policy.md", agent_id="bob") == bob
    note.unlink()
    index_vault(db, bob_cfg, Vectors())
    assert db.get_memory(bob)["status"] == "forgotten"
    assert db.get_memory(alice)["status"] == "active"


def test_imports_use_only_current_profile_files_and_hindsight_bank(db, tmp_path, monkeypatch):
    for name in ("alice", "bob"):
        folder = tmp_path / "profiles" / name
        folder.mkdir(parents=True)
        (folder / "MEMORY.md").write_text(f"- {name} secret preference")
    cfg = RemnantConfig(agent_id="bob")
    stats = import_memory_store(db, cfg, Vectors(), tmp_path)
    assert stats["imported"] == 1
    assert "alice" not in json.dumps(db.list_memories(agent_id="bob"))
    banks = []
    monkeypatch.setattr("remnant.import_sources._hindsight_recall",
                        lambda query, **kwargs: banks.append(kwargs["bank_id"]) or [])
    import_hindsight(db, cfg, Vectors(), queries=["preference"])
    assert banks == ["hermes-bob"]
    p = make_provider(db, "bob")
    denied = json.loads(p.handle_tool_call("memory_import", {
        "source": "memory_store", "profile": "alice",
    }))
    assert denied.get("error")


def test_threads_and_dreams_cannot_cross_profiles(db, monkeypatch):
    tid = db.insert_thread(title="Alice secret", topic="Private plans", added_by="alice")
    p = make_provider(db, "bob")
    assert json.loads(p.handle_tool_call("memory_thread", {"action": "list"}))["threads"] == []
    assert json.loads(p.handle_tool_call("memory_thread", {
        "action": "resolve", "thread_id": tid,
    })).get("error")
    for agent in ("alice", "bob"):
        db.insert_memory(content=f"{agent} secret topic", agent=agent, visibility="fleet",
                         embedding=[1., 0.], embed_model="test")
    calls = []
    monkeypatch.setattr("remnant.dream._cloud_judge", lambda *a, **k: calls.append(a) or [])
    assert day_dream(db, p._config, p._embedder)["actions"] == 0
    assert calls == []


def test_v15_vault_migration_preserves_ownership_and_evidence(tmp_path):
    path = tmp_path / "legacy.db"
    old = open_db(path)
    mid = old.insert_memory(content="Legacy Alice note", agent="alice", source="vault")
    with old.read() as cur:
        cur.executescript("""
            DROP TABLE vault_files;
            DROP TABLE vault_passages;
            CREATE TABLE vault_files(path TEXT PRIMARY KEY,hash TEXT,
                                     memory_id TEXT,indexed_at TEXT);
            CREATE TABLE vault_passages(path TEXT,ordinal INTEGER,memory_id TEXT,heading_path TEXT,
                                       start_offset INTEGER,end_offset INTEGER,
                                       PRIMARY KEY(path,ordinal));
            UPDATE schema_meta SET value='15' WHERE key='version';
        """)
        cur.execute("INSERT INTO vault_files VALUES('note.md','hash',?,'2026-09-06')", (mid,))
        cur.execute("INSERT INTO vault_passages VALUES('note.md',0,?,'',0,10)", (mid,))
    old.close()
    new = open_db(path)
    try:
        assert new.get_vault_memory("note.md", agent_id="alice") == mid
        assert new.get_vault_memory("note.md", agent_id="bob") is None
        assert new.get_memory(mid)["content"] == "Legacy Alice note"
        with new.read() as cur:
            assert cur.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        new.close()


def test_named_profiles_and_runtime_identity_cannot_alias_each_other(tmp_path):
    from remnant.config import load_config
    from remnant.identity import effective_identity

    keys = []
    for profile in ("alice", "bob"):
        home = tmp_path / "profiles" / profile
        home.mkdir(parents=True)
        (home / "remnant.json").write_text('{"agent_id":"copied-config"}')
        cfg = load_config(home)
        assert cfg.agent_id == profile
        assert cfg.diary_path == str(home / "remnant" / "DREAMS.md")
        keys.append(effective_identity(
            configured_agent=cfg.agent_id, runtime_identity_enabled=True,
            session_id="same-session", agent_identity="same-runtime", platform="cli",
        ).storage_key)
    assert keys[0] != keys[1]


def test_model_backfill_sends_only_selected_profiles_evidence(db):
    from remnant.model_backfill import run_model_backfill

    db.insert_memory(content="Alice private project", agent="alice")
    bob = db.insert_memory(content="Bob private project", agent="bob")
    prompts = []
    def model(prompt, allowed):
        prompts.append(prompt)
        assert allowed == {bob}
        return '{"claims":[]}'
    result = run_model_backfill(db, RemnantConfig(agent_id="bob"), model_call=model)
    assert result["targeted"] == 1
    assert prompts and "Alice" not in prompts[0]


def test_scoped_vault_reindex_preserves_out_of_scope_mappings(db, tmp_path):
    for folder in ('inside', 'outside'):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / 'note.md').write_text(f'# {folder}\nOriginal note')
    cfg = RemnantConfig(agent_id='alice', vault_path=str(tmp_path), embed_dim=2)
    index_vault(db, cfg, Vectors())
    outside = db.get_vault_memory('outside/note.md', agent_id='alice')
    cfg.profile_scope = ['inside']
    (tmp_path / 'inside/note.md').unlink()
    result = index_vault(db, cfg, Vectors())
    assert result['forgotten'] == 1
    assert db.get_vault_memory('outside/note.md', agent_id='alice') == outside
    assert db.get_memory(outside)['status'] == 'active'
    cfg.profile_scope = []
    index_vault(db, cfg, Vectors())
    assert db.get_vault_memory('outside/note.md', agent_id='alice') == outside


def test_dream_state_is_owned_including_night_window(db, monkeypatch):
    import time

    from remnant import dream

    now = time.time()
    today = time.strftime('%Y-%m-%d', time.gmtime(now))
    for key, value in (('day_run_ts', now), ('night_run_ts', now),
                       ('day_counter_date', today), ('day_counter', 999),
                       ('recent_topics', {'secret': now})):
        db.set_state(key, value, agent_id='alice')
    seen = []
    monkeypatch.setattr(db, 'get_recent_memories',
                        lambda **kw: seen.append(kw) or [])
    cfg = RemnantConfig(agent_id='bob', trust_decay_enabled=False)
    assert 'skipped' not in day_dream(db, cfg, Vectors())
    dream.night_dream(db, cfg, Vectors())
    assert seen[-1]['agent_id'] == 'bob'
    assert seen[-1]['since_ts'] < now - 86000
    assert db.get_state('day_counter', agent_id='bob') == 0
    assert db.get_state('recent_topics', agent_id='bob') is None
    assert db.get_state('day_counter', agent_id='alice') == 999
    assert dream._window_start(db, 'night', now, agent_id='alice') == now


def test_runtime_identity_import_uses_filesystem_profile(db, tmp_path):
    from remnant.identity import effective_identity

    home = tmp_path / 'profiles' / 'bob'
    home.mkdir(parents=True)
    (home / 'MEMORY.md').write_text('- Bob prefers a midnight editor theme.')
    identity = effective_identity(configured_agent='bob', session_id='test',
                                  runtime_identity_enabled=True)
    p = make_provider(db, identity.storage_key)
    p._effective_identity = identity
    p._hermes_home = str(home)
    result = json.loads(p.handle_tool_call('memory_import', {'source': 'memory_store'}))
    assert result['stats']['imported'] == 1
    assert db.list_memories(agent_id=identity.storage_key)
    assert db.list_memories(agent_id='bob') == []
    assert p.import_memory('memory_store', profile='bob', dry_run=True)['discovered'] == 1
    assert p.import_memory('memory_store', profile='alice').get('error')
    assert json.loads(p.handle_tool_call('memory_import', {
        'source': 'memory_store', 'profile': 'alice',
    })).get('error')


@pytest.mark.parametrize('mixed', [False, True])
def test_legacy_system_thread_and_state_migration_requires_known_owner(tmp_path, mixed):
    path = tmp_path / 'legacy-state.db'
    old = open_db(path)
    old.insert_memory(content='Alice evidence', agent='alice')
    if mixed:
        old.insert_memory(content='Bob evidence', agent='bob')
    tid = old.insert_thread(title='Existing dream', topic='Legacy topic',
                            added_by='system', owner='alice')
    old._conn.executescript('''
        ALTER TABLE threads DROP COLUMN owner;
        DROP TABLE dream_state;
        CREATE TABLE dream_state(key TEXT PRIMARY KEY, value TEXT NOT NULL,
                                 updated_at TEXT NOT NULL);
        INSERT INTO dream_state VALUES('night_run_ts','123','2026-09-06');
        UPDATE schema_meta SET value='16' WHERE key='version';
    ''')
    old.close()
    new = open_db(path)
    try:
        thread = new.get_thread(tid)
        assert thread['added_by'] == 'system'  # Authorship is retained.
        assert thread['owner'] == (None if mixed else 'alice')
        assert bool(new.list_threads(agent_id='alice')) is (not mixed)
        assert new.list_threads(agent_id='bob') == []
        assert new.get_state('night_run_ts', agent_id='alice') == (None if mixed else 123)
        if mixed:
            assert new.get_state('night_run_ts') == 123  # Preserved for explicit mapping.
        with pytest.raises(ValueError, match='owner'):
            new.insert_thread(title='Unowned', topic='Unowned')
    finally:
        new.close()
