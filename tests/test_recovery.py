"""Recovery keeps source evidence and remaps overlapping numeric turn IDs."""
import json
import sqlite3
from pathlib import Path

import pytest

from remnant.db import open_db
from remnant.recover import build_recovery


def seed(path, owner):
    db = open_db(path)
    turn = db.insert_turn_with_extraction(session_id='same', agent_id=owner,
                                         user_text=f'{owner} evidence', assistant_text='ack')
    memory = db.insert_memory(content=f'{owner} secret', agent=owner,
                              source='conversation', source_id=str(turn))
    db.create_claim(memory_id=memory, subject=owner, predicate='prefers', object='tea',
                    confidence=.9, source_turn_id=turn)
    db.write_audit(actor=owner, action='memory_duplicate', memory_id=memory,
                   details={'source_turn_id': turn})
    db.insert_thread(title=f'{owner} topic', topic='private', owner=owner)
    db.close()
    return memory


def test_recovery_merges_fleet_and_remaps_turns_without_changing_inputs(tmp_path):
    sources = []
    ids = []
    for owner in ('alice', 'bob'):
        path = tmp_path / f'{owner}.db'
        ids.append(seed(path, owner))
        sources.append({'path': str(path), 'owners': {owner: owner}})
    # The same source corpus, previously copied to a shared database.
    clone = tmp_path / 'shared.db'
    with sqlite3.connect(sources[0]['path']) as src, sqlite3.connect(clone) as dst:
        src.backup(dst)
    sources.append({'path': str(clone), 'owners': {'alice': 'alice'}})
    before = [Path(s['path']).read_bytes() for s in sources]
    output = tmp_path / 'result.db'
    report = build_recovery({'sources': sources}, output)
    assert report['memories_by_owner'] == {'alice': 1, 'bob': 1}
    assert report['threads_by_owner'] == {'alice': 1, 'bob': 1}
    assert report['turns'] == 2
    db = open_db(output)
    try:
        assert {m['id'] for m in db.list_memories()} == set(ids)
        rows = list(db._conn.execute('''
            SELECT m.agent,t.agent_id,c.source_turn_id,m.source_id FROM memories m
            JOIN claims c ON c.memory_id=m.id JOIN turns t ON t.id=c.source_turn_id
        '''))
        assert all(r[0] == r[1] and str(r[2]) == r[3] for r in rows)
        assert db._conn.execute('SELECT COUNT(*) FROM extraction_queue').fetchone()[0] == 2
        for row in db._conn.execute(
            "SELECT details,m.agent,t.agent_id FROM audit_log a "
            "JOIN memories m ON m.id=a.memory_id "
            "JOIN turns t ON t.id=json_extract(a.details,'$.source_turn_id') "
            "WHERE action='memory_duplicate'"
        ):
            assert json.loads(row[0])['source_turn_id'] and row[1] == row[2]
        assert db._conn.execute('PRAGMA foreign_key_check').fetchone() is None
    finally:
        db.close()
    assert [Path(s['path']).read_bytes() for s in sources] == before
    with pytest.raises(FileExistsError):
        build_recovery({'sources': sources}, output)


def test_recovery_rejects_unknown_owners_and_conflicting_memory(tmp_path):
    a = tmp_path / 'a.db'
    mid = seed(a, 'alice')
    output = tmp_path / 'result.db'
    with pytest.raises(ValueError, match='unmapped owner'):
        build_recovery({'sources': [{'path': str(a), 'owners': {'bob': 'bob'}}]}, output)
    assert not output.exists()
    b = tmp_path / 'b.db'
    with sqlite3.connect(a) as src, sqlite3.connect(b) as dst:
        src.backup(dst)
    db = open_db(b)
    db.set_memory_field(mid, 'content', 'Changed evidence', actor='alice')
    db.close()
    with pytest.raises(ValueError, match='conflicting memory'):
        build_recovery({'sources': [{'path': str(p), 'owners': {'alice': 'alice'}}
                                    for p in (a, b)]}, output)
    assert not output.exists()


def test_recovery_requires_explicit_mapping_for_ambiguous_legacy_state(tmp_path):
    path = tmp_path / 'mixed.db'
    seed(path, 'alice')
    db = open_db(path)
    db.insert_memory(content='Bob evidence', agent='bob')
    tid = db.insert_thread(title='Legacy dream', topic='Private topic', owner='alice')
    db._conn.execute('UPDATE threads SET owner=NULL WHERE id=?', (tid,))
    db.set_state('night_run_ts', 123)
    db.close()
    spec = {'path': str(path), 'owners': {'alice': 'alice', 'bob': 'bob'}}
    output = tmp_path / 'result.db'
    with pytest.raises(ValueError, match='unmapped owner'):
        build_recovery({'sources': [spec]}, output)
    assert not output.exists()
    spec.update(thread_owners={tid: 'alice'}, dream_owner='alice')
    build_recovery({'sources': [spec]}, output)
    db = open_db(output)
    try:
        assert db.get_thread(tid)['owner'] == 'alice'
        assert db.get_state('night_run_ts', agent_id='alice') == 123
        assert db.get_state('night_run_ts', agent_id='bob') is None
    finally:
        db.close()


def test_recovery_refuses_combining_profile_owners(tmp_path):
    with pytest.raises(ValueError, match='cannot share'):
        build_recovery({'sources': [{
            'path': str(tmp_path / 'unused.db'),
            'owners': {'alice': 'alice', 'bob': 'alice'},
        }]}, tmp_path / 'result.db')


def test_recovery_infers_nullable_legacy_link_owners_from_backing_memory(tmp_path):
    source = tmp_path / 'legacy-links.db'
    memory = seed(source, 'alice')
    db = open_db(source)
    entity = db.resolve_entity('Legacy shared node')
    db.link_entity(memory_id=memory, entity_id=entity, agent_id='alice')
    db._conn.execute('UPDATE memory_entities SET agent=NULL')
    db._conn.execute("INSERT INTO entity_sightings VALUES('legacy',NULL,?,'2026-09-06')",
                     (memory,))
    db.close()
    output = tmp_path / 'recovered.db'
    build_recovery({'sources': [{'path': str(source), 'owners': {'alice': 'alice'}}]}, output)
    db = open_db(output)
    try:
        for table in ('memory_entities', 'entity_sightings'):
            assert db._conn.execute(f'SELECT agent FROM {table}').fetchone()[0] == 'alice'
        assert db.get_entity(entity)['agent'] is None
    finally:
        db.close()
