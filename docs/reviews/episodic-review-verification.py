"""Isolated acceptance/security checks for the independent PR #46 review."""
from __future__ import annotations

import hashlib
import json
import os
import runpy
import sqlite3
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from remnant import RemnantMemoryProvider, history as h
from remnant.config import RemnantConfig
from remnant.context import conservative_token_count
from remnant.db import open_db
from remnant.evaluation.runner import evaluate_scenarios
from remnant.evaluation.schema import load_cases
from remnant.maintenance import backup_database, health_report, restore_database
from remnant.tools import handle_tool_call
from remnant.vault import index_vault

PROBES = runpy.run_path(str(Path(__file__).with_name('episodic-review-probes.py')))
fixture, BASE = PROBES['fixture'], PROBES['BASE']
RESULTS = {}


def run(root):
    for corpus, expected in [('leadership', 240), ('heldout-adversarial', 120)]:
        result = evaluate_scenarios(load_cases(f'evaluation/cases/{corpus}.jsonl'))['summary']
        assert result['cases'] == expected and result['recall_at_5'] >= .85
        assert result['stale_claim_exposure'] == 0
        if corpus == 'heldout-adversarial':
            assert result['duplicate_top_k_occupancy'] == 0
        RESULTS[corpus] = result

    db = open_db(root / 'fresh.db')
    health = health_report(db)
    assert health['integrity'] == 'ok' and health['schema_version'] == 18
    backup = backup_database(db, root / 'backup.db')
    restored = restore_database(root / 'backup.db', root / 'restored.db')
    RESULTS['fresh_backup_restore'] = {'health': health['integrity'], 'version': health['schema_version'],
        'backup': backup['integrity'], 'restore': restored['integrity']}
    db.close()

    service = fixture(root, 'security', [(1, 's', 'user', 'visible quasar', BASE),
        (2, 's', 'assistant', 'compacted quasar', BASE+1),
        (3, 's', 'user', 'rewound quasar', BASE+2), (4, 's', 'user', 'hidden quasar', BASE+3),
        (5, 's', 'tool', 'tool quasar', BASE+4), (6, 's', 'system', 'system quasar', BASE+5)])
    conn = sqlite3.connect(service.archive.path)
    conn.execute('UPDATE messages SET active=0,compacted=1 WHERE id=2')
    conn.execute('UPDATE messages SET active=0 WHERE id=3')
    conn.execute("UPDATE messages SET display_kind='hidden' WHERE id=4")
    conn.commit()
    conn.close()
    before = hashlib.sha256(service.archive.path.read_bytes()).hexdigest()
    with patch.object(h, 'chat', side_effect=AssertionError('unexpected model')):
        result = service.recall({'session_id': 's', 'synthesize': False,
            'agent_id': 'foreign', 'profile': 'foreign', 'path': '/not/an/archive'})
        ids = [e['source']['message_id'] for e in result['evidence']]
        assert ids == [1, 2]
        assert service.recall({'session_id': "s' OR 1=1 --", 'synthesize': False})['status'] == 'no_evidence'
        assert not service.recall({'session_id': 's', 'around_message_id': 3, 'synthesize': False})['evidence']
    after = hashlib.sha256(service.archive.path.read_bytes()).hexdigest()
    assert before == after
    link_home = root / 'symlink'
    link_home.mkdir()
    (link_home / 'state.db').symlink_to(service.archive.path)
    assert not h.HistoryArchive(link_home/'state.db', profile_home=link_home).available()
    RESULTS['visibility_sql_path_archive_immutability'] = 'passed: compacted allowed; rewind/hidden/tool/system/SQL injection/override/symlink rejected; archive SHA unchanged'

    # Foreign citation, malformed output, and model outage cannot add unchecked prose.
    for output in ['not json', json.dumps({'statements': [{'text':'foreign', 'sources':[{'session_id':'foreign','message_id':1}]}]})]:
        with patch.object(h, 'chat', return_value=output) as chat:
            result = service.recall({'session_id': 's'})
            assert chat.call_count == 1 and result['status'] == 'partial'
            assert all(s['uncertainty'] == 'source-excerpt' for s in result['statements'])
    with patch.object(h, 'chat', side_effect=TimeoutError) as chat:
        result = service.recall({'session_id': 's'})
        assert chat.call_count == 1 and result['evidence']
    RESULTS['model_failure_fallback'] = 'passed malformed/foreign citations/timeout; one attempted call each'

    # Full pipeline serialization, not just its compact internal estimate.
    maximum = (0, 0, 0)
    for size in range(1, 150, 4):
        conn = sqlite3.connect(service.archive.path)
        conn.execute("DELETE FROM messages")
        conn.executemany('INSERT INTO messages(id,session_id,role,content,timestamp) VALUES(?,?,?,?,?)',
            [(i, 's', 'user', 'a'*size, BASE+i) for i in range(1, 61)])
        conn.commit()
        conn.close()
        result = service.recall({'session_id': 's', 'synthesize': False})
        compact = conservative_token_count(h._json(result))
        wire = conservative_token_count(json.dumps(result, ensure_ascii=False, default=str))
        if wire > maximum[0]:
            maximum = (wire, compact, size)
    RESULTS['wire_output_max'] = {'wire_tokens': maximum[0], 'compact_tokens': maximum[1], 'message_chars': maximum[2]}
    assert maximum[0] > 4000

    # Exercise ordinary provider entry points with a trap history object.
    class Trap:
        def __getattr__(self, name):
            raise AssertionError('ordinary path touched history: '+name)
    class Embed:
        _model = 'synthetic'
        def embed(self, text, **kw):
            return None
    provider = RemnantMemoryProvider()
    provider._db, provider._config, provider._embedder = service.db, service.config, Embed()
    provider._history = Trap()
    provider.sync_turn('Discuss 2024-06-14 project history', 'fine')
    provider.prefetch('Discuss 2024-06-14 project history')
    provider.queue_prefetch('Discuss 2024-06-14 project history')
    RESULTS['ordinary_path_history_trap'] = 'passed sync_turn/prefetch/queue_prefetch with date prose'
    service.db.close()

    # A real concurrent Remnant lock exceeds the supposed archive-request deadline.
    service = fixture(root, 'lock', [(1, 's', 'user', 'owned', BASE)],
        config=RemnantConfig(agent_id='a', runtime_identity_enabled=True, history_summary_enabled=False))
    service.db.insert_turn(session_id='s', agent_id='a', user_text='owned', assistant_text='')
    locked = threading.Event()
    def hold():
        with service.db.read():
            locked.set()
            time.sleep(3.4)
    thread = threading.Thread(target=hold)
    thread.start()
    locked.wait()
    start = time.perf_counter()
    result = service.recall({'session_id': 's', 'synthesize': False})
    elapsed = time.perf_counter()-start
    thread.join()
    RESULTS['request_lock_deadline'] = {'elapsed_s': elapsed, 'status': result['status']}
    assert elapsed > 3.0
    service.db.close()

    # Independent DB connections reserve daily calls transactionally.
    path = root/'concurrent.db'
    db = open_db(path)
    for i in range(64):
        db.enqueue_history_summary(archive_key='a', agent_id='owner', session_id=str(i))
    second = open_db(path)
    def claim(handle):
        return [handle.claim_history_summary(agent_id='owner') for _ in range(25)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = [r for batch in pool.map(claim, [db, second]) for r in batch if r]
    assert len(rows) == 20 and len({r['id'] for r in rows}) == 20
    RESULTS['concurrent_daily_budget'] = {'claimed': len(rows), 'unique': len({r['id'] for r in rows})}
    second.close()
    db.close()

    # Pre-existing full-codebase findings: synthetic vault escape and graph scope bypass.
    vault = root/'vault'
    vault.mkdir()
    (vault/'allowed').mkdir()
    outside = root/'outside.md'
    outside.write_text('Synthetic external document.\n')
    (vault/'allowed'/'linked.md').symlink_to(outside)
    db = open_db(root/'vault.db')
    config = RemnantConfig(agent_id='audit', vault_path=str(vault), profile_scope=['allowed'])
    with patch('remnant.vault.extract_and_link_entities'):
        stats = index_vault(db, config, None)
    rows = db._conn.execute('SELECT source_id,content FROM memories').fetchall()
    RESULTS['vault_symlink_escape'] = {'indexed': stats['indexed'], 'rows': [dict(r) for r in rows]}
    assert stats['indexed'] == 1
    mid = db.insert_memory(content='Synthetic excluded document.', source='vault', source_id='private/blocked.md',
                           agent='audit', type='document')
    eid = db.resolve_entity('ScopeProbe', entity_type='project', agent_id='audit')
    db.link_entity(memory_id=mid, entity_id=eid, agent_id='audit')
    result = handle_tool_call('memory_graph', {'entity':'ScopeProbe'}, db=db, config=config, embedder=None, session_id='s')
    RESULTS['graph_scope_bypass'] = {'returned_excluded_memory': any(r['id'] == mid for r in result['memories'])}
    assert RESULTS['graph_scope_bypass']['returned_excluded_memory']
    db.close()

    # Installed Hermes integration on a fresh disposable source only.
    hermes_home = root/'installed-home'
    hermes_home.mkdir()
    os.environ['HERMES_HOME'] = str(hermes_home)
    os.environ['REMNANT_DB_HOME'] = str(root/'unused')
    sys.path.insert(0, '/home/jd/.hermes/hermes-agent')
    from hermes_state import SessionDB
    archive_path = hermes_home/'state.db'
    archive = SessionDB(db_path=archive_path)
    archive.create_session('fresh', 'cli', profile_name='default')
    mid = archive.append_message('fresh', 'user', 'installed Hermes quasar', timestamp=BASE)
    archive.close()
    db = open_db(root/'installed-remnant.db')
    service = h.HistoryService(db, RemnantConfig(history_summary_enabled=False), profile_home=hermes_home)
    result = service.recall({'start':'2024-06-14', 'synthesize':False})
    assert any(e['source']['message_id'] == mid for e in result['evidence'])
    RESULTS['installed_hermes'] = {'status': result['status'], 'message_id': mid}
    db.close()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='remnant-sentinel-checks-') as temp:
        run(Path(temp))
    print(json.dumps(RESULTS, indent=2))
