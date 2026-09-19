"""Independent synthetic probes. Run: PYTHONPATH=. .venv/bin/python docs/reviews/episodic-review-probes.py

No live archive/config or external model calls. Assertions pin observed candidate defects,
not desired behavior. This is review evidence, not an implementation test suite.
"""
from __future__ import annotations

import gc
import json
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from remnant import history as h
from remnant.config import RemnantConfig
from remnant.context import conservative_token_count
from remnant.db import open_db


SCHEMA = """
CREATE TABLE sessions(id TEXT PRIMARY KEY, source TEXT, user_id TEXT, started_at REAL,
 ended_at REAL, title TEXT, profile_name TEXT, rewind_count INTEGER DEFAULT 0,
 hidden INTEGER DEFAULT 0);
CREATE TABLE messages(id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT,
 timestamp REAL, active INTEGER DEFAULT 1, compacted INTEGER DEFAULT 0, display_kind TEXT,
 _compressed_summary INTEGER DEFAULT 0);
CREATE INDEX idx_message_session_time ON messages(session_id,timestamp);
CREATE VIRTUAL TABLE messages_fts USING fts5(content,content='messages',content_rowid='id');
"""
BASE = datetime(2024, 6, 14, tzinfo=timezone.utc).timestamp()
RESULTS = {}


def fixture(root, name, messages, *, config=None):
    home = root / name
    home.mkdir()
    conn = sqlite3.connect(home / 'state.db')
    conn.executescript(SCHEMA)
    for sid in dict.fromkeys(row[1] for row in messages):
        conn.execute("INSERT INTO sessions(id,source,started_at) VALUES(?,'interactive',?)", (sid, BASE))
    conn.executemany('INSERT INTO messages(id,session_id,role,content,timestamp) VALUES(?,?,?,?,?)', messages)
    conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
    conn.commit()
    conn.close()
    db = open_db(home / 'remnant.db')
    service = h.HistoryService(db, config or RemnantConfig(history_summary_enabled=False), profile_home=home)
    return service


def drain(service, args):
    seen, pages, counts, tokens = [], [], [], []
    for _ in range(100):
        result = service.recall({**args, 'synthesize': False})
        seen.extend(x['source']['message_id'] for x in result['evidence'])
        pages.append(result)
        counts.append(len(result['evidence']))
        tokens.append(conservative_token_count(h._json(result)))
        if not result['has_more']:
            break
        args = {**args, 'cursor': result['next_cursor']}
    return seen, pages, counts, tokens


def run(root):
    service = fixture(root, 'large', [(i, 's', 'user', 'x' * 1900, BASE+i) for i in range(1, 25)])
    seen, pages, counts, tokens = drain(service, {'session_id': 's'})
    RESULTS['large_excerpts'] = {'expected_messages': 24, 'returned': len(seen), 'page_counts': counts,
                                'has_more': pages[-1]['has_more'], 'coverage': pages[-1]['coverage']}
    assert not seen and not pages[-1]['has_more']
    service.db.close()

    service = fixture(root, 'cursor', [(i, 's', 'user', 'short text', BASE+i) for i in range(1, 50)])
    service._now = datetime(2024, 6, 15, tzinfo=timezone.utc)
    first = service.recall({'relative': 'yesterday', 'synthesize': False})
    service._now = datetime(2024, 6, 16, tzinfo=timezone.utc)
    second = service.recall({'relative': 'yesterday', 'synthesize': False, 'cursor': first['next_cursor']})
    RESULTS['relative_cursor'] = {'first': first['resolved_range'], 'second': second['resolved_range'],
                                  'second_status': second['status'], 'second_messages': len(second['evidence'])}
    assert first['next_cursor'] and first['resolved_range']['start_utc'] != second['resolved_range']['start_utc']
    for args in ({'relative': '999999999999999999999d'}, {'start': '9999-12-31'}):
        try:
            service.recall({**args, 'synthesize': False})
        except Exception as exc:
            RESULTS.setdefault('invalid_bounds', []).append({'input': args, 'exception': type(exc).__name__})
    service.db.close()

    service = fixture(root, 'tailmatch', [(1, 's', 'user', 'x' * 2200 + ' quasar', BASE)])
    result = service.recall({'query': 'quasar', 'synthesize': False})
    RESULTS['match_after_projection'] = {'status': result['status'], 'evidence': len(result['evidence']),
                                         'has_more': result['has_more'], 'coverage': result['coverage']}
    assert result['status'] == 'no_evidence' and not result['evidence']
    service.db.close()

    service = fixture(root, 'runtime', [(1, 's', 'user', 'project alpha', BASE)],
                      config=RemnantConfig(runtime_identity_enabled=True, history_summary_enabled=False, agent_id='alice'))
    service.db.insert_turn(session_id='s', agent_id='alice', user_text='owned', assistant_text='')
    row = service.archive.get_messages('s')[0]
    kw = dict(agent_id='alice', remnant_db=service.db, runtime_identity_enabled=True, trusted_session_ids=set())
    RESULTS['runtime_reference'] = {'session_valid': service.archive.validate_session('s', **kw) is not None,
                                     'reference_valid': service.archive.validate_reference(h._reference_from_message(row), **kw) is not None}
    assert RESULTS['runtime_reference'] == {'session_valid': True, 'reference_valid': False}
    service.db.close()

    service = fixture(root, 'summary', [(i, 's', 'user', f'message {i}', BASE+i) for i in range(1, 61)],
                      config=RemnantConfig(history_summary_enabled=True))
    row = service.archive.get_messages('s')[29]
    ref = h._reference_from_message(row)
    captured = []
    def fake_chat(**kw):
        captured.append(kw)
        return json.dumps({'topics': ['alpha'], 'statements': [{'topic': 'alpha', 'text': 'Discussed alpha',
            'kind': 'discussion', 'sources': [ref]}]})
    service.enqueue_session('s')
    with patch.object(h, 'chat', fake_chat):
        completed = service.process_one_summary()
    cached = service.db.get_history_summary(archive_key=service.archive.archive_key, agent_id='default', session_id='s')
    RESULTS['unsupplied_summary_reference'] = {'completed': completed, 'reference_in_prompt': h._json(ref) in captured[0]['user'],
        'cache_status': cached['status'], 'coverage': json.loads(cached['coverage_json'])}
    assert completed and h._json(ref) not in captured[0]['user'] and cached['status'] == 'ready'
    service.db.close()

    service = fixture(root, 'edited', [(1, 's', 'user', 'We discussed options.', BASE),
                                       (2, 's', 'assistant', 'Adopt option blue.', BASE+1)])
    refs = [h._reference_from_message(row) for row in service.archive.get_messages('s')]
    service.db.enqueue_history_summary(archive_key=service.archive.archive_key, agent_id='default', session_id='s')
    claim = service.db.claim_history_summary(agent_id='default')
    service.db.complete_history_summary(claim['id'], source_version=service.archive.source_version('s'),
        summary={'topics': ['option'], 'statements': [{'text': 'Adopt option blue.', 'sources': refs}]}, coverage={})
    conn = sqlite3.connect(service.archive.path)
    conn.execute("UPDATE messages SET content='Actually adopt option red.' WHERE id=2")
    conn.commit()
    conn.close()
    result = service.recall({'session_id': 's', 'synthesize': False})
    RESULTS['partly_invalid_summary'] = {'summaries_used': result['coverage']['summaries_used'],
        'evidence': [(e['excerpt'], [r['message_id'] for r in e.get('sources', [e['source']])]) for e in result['evidence']]}
    assert any(e['summary'] and 'blue' in e['excerpt'] for e in result['evidence'])
    service.db.close()

    # Cache-only matches never supplement nonempty raw matches.
    service = fixture(root, 'cacheunion', [(1, 'raw', 'user', 'quasar raw discussion', BASE),
                                          (2, 'cached', 'user', 'a star discussion', BASE+1)])
    row = service.archive.get_messages('cached')[0]
    service.db.enqueue_history_summary(archive_key=service.archive.archive_key, agent_id='default', session_id='cached')
    claim = service.db.claim_history_summary(agent_id='default')
    service.db.complete_history_summary(claim['id'], source_version=service.archive.source_version('cached'),
        summary={'topics': ['quasar'], 'statements': [{'text': 'quasar discussion', 'sources': [h._reference_from_message(row)]}]}, coverage={})
    seen, pages, counts, tokens = drain(service, {'query': 'quasar'})
    RESULTS['cache_raw_union'] = {'ids': seen, 'page_counts': counts, 'has_more': pages[-1]['has_more']}
    assert seen == [1]
    service.db.close()

    # The worker claims by owner only, not archive key.
    a = fixture(root, 'archive_a', [(1, 's', 'user', 'Common question', BASE),
        (2, 's', 'assistant', 'A private detail', BASE+1)], config=RemnantConfig(history_summary_enabled=True))
    b = fixture(root, 'archive_b', [(1, 's', 'user', 'Common question', BASE),
        (2, 's', 'assistant', 'B private detail', BASE+1)], config=RemnantConfig(history_summary_enabled=True))
    b.db.close()
    b.db = a.db
    a.enqueue_session('s')
    captured = []
    def other_chat(**kw):
        captured.append(kw)
        return json.dumps({'topics': ['b'], 'statements': [{'text': 'B private detail',
            'sources': [{'session_id':'s','message_id':1}, {'session_id':'s','message_id':2}]}]})
    with patch.object(h, 'chat', other_chat):
        completed = b.process_one_summary()
    row = a.db.get_history_summary(archive_key=a.archive.archive_key, agent_id='default', session_id='s')
    recalled = a.recall({'session_id': 's', 'synthesize': False})
    leaked = any('B private detail' in e['excerpt'] for e in recalled['evidence'])
    RESULTS['cross_archive_claim'] = {'processed_by_b': completed, 'a_cache_status': row['status'],
        'b_content_in_a_cache': 'B private detail' in row['summary_json'], 'b_content_returned_to_a': leaked}
    assert completed and leaked
    a.db.close()

    # First initialization before state.db creation stays unavailable.
    home = root / 'late'
    home.mkdir()
    db = open_db(home / 'remnant.db')
    service = h.HistoryService(db, RemnantConfig(history_summary_enabled=False), profile_home=home)
    conn = sqlite3.connect(home / 'state.db')
    conn.executescript(SCHEMA)
    conn.close()
    RESULTS['late_archive'] = service.recall({'query': 'quasar', 'synthesize': False})['status']
    assert RESULTS['late_archive'] == 'unavailable'
    db.close()

    # A session title is not SQL-projected or bounded. Use synthetic data only.
    service = fixture(root, 'output', [(i, 's', 'user', 'short message', BASE+i) for i in range(1, 70)])
    seen, pages, counts, tokens = drain(service, {'session_id': 's'})
    RESULTS['serialized_budget'] = {'max_tokens': max(tokens), 'reported_tokens': [p['token_estimate'] for p in pages], 'actual_tokens': tokens}
    service.db.close()

    # Large raw LIKE scan, even when FTS exists.
    service = fixture(root, 'scale', [(i, f's{i//100}', 'user', 'ordinary text '*80 + (' quasar' if i%1000 == 0 else ''), BASE+i) for i in range(1, 30001)])
    times = []
    for _ in range(5):
        started = time.perf_counter()
        result = service.recall({'query': 'no_such_token', 'synthesize': False})
        times.append((time.perf_counter()-started)*1000)
    RESULTS['large_archive_no_hit'] = {'messages': 30000, 'message_chars': 1120, 'runs_ms': times, 'status': result['status'], 'coverage': result['coverage']}
    gc.collect()
    before = len(list(Path('/proc/self/fd').iterdir()))
    gc.disable()
    for _ in range(50):
        service.archive.get_session('s1')
    after = len(list(Path('/proc/self/fd').iterdir()))
    gc.enable()
    gc.collect()
    RESULTS['connection_lifecycle'] = {'before_fds': before, 'after_50_reads_before_gc': after, 'after_gc': len(list(Path('/proc/self/fd').iterdir()))}
    assert after >= before+50
    service.db.close()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='remnant-sentinel-') as directory:
        run(Path(directory))
    print(json.dumps(RESULTS, indent=2))
