"""Bounded independent correction checks; synthetic data/stub models only.
Run from repo root: PYTHONPATH=. .venv/bin/python docs/reviews/episodic-correction-probes.py
Preserves prior defect reproductions; reports desired-behavior PASS/FAIL without stopping.
"""
from __future__ import annotations

import gc
import json
import runpy
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from remnant import history as h
from remnant.config import RemnantConfig
from remnant.context import conservative_token_count
from remnant.db import open_db

old = runpy.run_path('docs/reviews/episodic-review-probes.py')
fixture, BASE, SCHEMA = old['fixture'], old['BASE'], old['SCHEMA']
RESULTS = {}


def record(name, passed, **data):
    RESULTS[name] = {'result': 'PASS' if passed else 'FAIL', **data}


def drain(service, args, cap=100):
    pages, seen = [], []
    for _ in range(cap):
        result = service.recall({**args, 'synthesize': False})
        pages.append(result)
        seen.extend(e['source']['message_id'] for e in result['evidence'])
        if not result['has_more']:
            break
        if not result['next_cursor']:
            break
        args = {**args, 'cursor': result['next_cursor']}
    return seen, pages


def cache(service, sid, *, text='quasar discussion', refs=None):
    refs = refs or [h._reference_from_message(r) for r in service.archive.get_messages(sid)]
    service.db.enqueue_history_summary(archive_key=service.archive.archive_key,
                                      agent_id=service.agent_id, session_id=sid)
    claim = service.db.claim_history_summary(agent_id=service.agent_id,
                                            archive_key=service.archive.archive_key,
                                            daily_limit=1000)  # Seed historical cache; no model calls.
    assert claim and claim['session_id'] == sid
    assert service.db.complete_history_summary(
        claim['id'], source_version=service.archive.source_version(sid),
        summary={'topics': ['quasar'], 'statements': [{'text': text, 'sources': refs}]},
        coverage={}, claim_token=claim['claim_token'], archive_key=service.archive.archive_key)


def run(root):
    # S-001: same owner, colliding IDs, shared Remnant DB, different trusted archives.
    config = RemnantConfig(history_summary_enabled=True)
    a = fixture(root, 'a', [(1, 's', 'user', 'Common question', BASE),
                           (2, 's', 'assistant', 'A private detail', BASE+1)], config=config)
    b = fixture(root, 'b', [(1, 's', 'user', 'Common question', BASE),
                           (2, 's', 'assistant', 'B private detail', BASE+1)], config=config)
    b.db.close()
    b.db = a.db
    a.enqueue_session('s')
    with patch.object(h, 'chat', side_effect=AssertionError('foreign model call')) as model:
        processed = b.process_one_summary()
    row = a.db.get_history_summary(archive_key=a.archive.archive_key,
                                   agent_id='default', session_id='s')
    output = a.recall({'session_id': 's', 'synthesize': False})
    record('S-001-existing-archives', not processed and model.call_count == 0
           and row['status'] == 'pending' and 'B private detail' not in json.dumps(output),
           processed=processed, model_calls=model.call_count, foreign_job_status=row['status'])
    missing = h.HistoryService(a.db, config, profile_home=root/'missing-worker')
    missing.process_one_summary()
    row = a.db.get_history_summary(archive_key=a.archive.archive_key,
                                   agent_id='default', session_id='s')
    record('S-001-missing-worker-partition', row['status'] == 'pending',
           foreign_job_status=row['status'], foreign_job_attempts=row['attempts'])
    a.db.close()

    # S-002: use provider's exact JSON serialization, drain every continuation.
    for label, text, sessions, per in [('ascii', 'x'*1900, 1, 24),
                                     ('cjk', '界'*1900, 1, 24),
                                     ('multi', '界'*1900, 21, 9),
                                     ('ids-36', 'x'*1900, 20, 9),
                                     ('ids-60', 'x'*1900, 20, 9),
                                     ('long-ids', 'x'*1900, 20, 9)]:
        width = {'long-ids': 110, 'ids-36': 36, 'ids-60': 60}.get(label, 0)
        rows = [(j*per+i+1, ('s'+str(j)).ljust(width, 'z'),
                 'user', text, BASE+i) for j in range(sessions) for i in range(per)]
        service = fixture(root, label, rows)
        args = {'session_id': 's0'} if sessions == 1 else {'start': '2024-06-14'}
        seen, pages = drain(service, args)
        wire = [conservative_token_count(json.dumps(p, ensure_ascii=False, default=str)) for p in pages]
        expected = sorted(r[0] for r in rows)
        first_state = h._decode_cursor(pages[0]['next_cursor'],
            h._fingerprint(service._validate_request(h.HistoryRequest(
                start=args.get('start'), session_id=args.get('session_id'), synthesize=False))[0],
                           service.agent_id)) if pages[0]['next_cursor'] else {}
        record('S-002-'+label, sorted(seen) == expected and max(wire) <= 4000
               and not pages[-1]['has_more'], expected=len(expected), returned=len(seen),
               unique=len(set(seen)), missing=sorted(set(expected)-set(seen))[:25],
               pages=len(pages), max_wire_tokens=max(wire), last_status=pages[-1]['status'],
               last_has_more=pages[-1]['has_more'],
               last_next_cursor=bool(pages[-1]['next_cursor']),
               first_ids=[e['source']['message_id'] for e in pages[0]['evidence']],
               first_cursor_offset_total=sum(first_state.get('offsets', {}).values()),
               counts_accurate=all(p['coverage']['messages_returned'] ==
                                   sum(not e['summary'] for e in p['evidence']) for p in pages))
        service.db.close()

    # S-003: unseen-only reference is now rejected; mixed valid/invalid must also fail closed.
    for mixed in (False, True):
        service = fixture(root, f'summary-{mixed}',
                          [(i, 's', 'user', f'message {i}', BASE+i) for i in range(1, 61)], config=config)
        rows = service.archive.get_messages('s')
        unseen = h._reference_from_message(rows[29])
        refs = ([h._reference_from_message(rows[0])] if mixed else []) + [unseen]
        service.enqueue_session('s')
        with patch.object(h, 'chat', return_value=json.dumps({'topics': ['quasar'], 'statements':
                          [{'text': 'Claim depends on unseen message 30', 'sources': refs}]})) as model:
            processed = service.process_one_summary()
        stored = service.db.get_history_summary(archive_key=service.archive.archive_key,
                                                agent_id='default', session_id='s')
        record('S-003-mixed-summary' if mixed else 'S-003-unseen-only',
               not processed and stored['status'] != 'ready', processed=processed,
               status=stored['status'], unseen_supplied=h._json(unseen) in model.call_args.kwargs['user'],
               stored_summary=json.loads(stored['summary_json'] or '{}'))
        service.db.close()

    service = fixture(root, 'synthesis', [(1, 's', 'user', 'Generic discussion', BASE)])
    ref = h._reference_from_message(service.archive.get_messages('s')[0])
    with patch.object(h, 'chat', return_value=json.dumps({'statements': [
        {'text': 'Unsupported joint claim', 'sources': [ref, {'session_id': 'foreign', 'message_id': 999}]}]})):
        output = service.recall({'session_id': 's'})
    record('S-003-mixed-synthesis', not any(s['text'] == 'Unsupported joint claim' for s in output['statements']),
           statements=output['statements'])
    service.db.close()

    for mutation in ('edit', 'range', 'hidden', 'rewind', 'delete'):
        service = fixture(root, 'mutate-'+mutation, [(1, 's', 'user', 'Generic discussion', BASE),
                          (2, 's', 'assistant', 'Adopt option blue', BASE+86400)])
        cache(service, 's', text='Adopt option blue')
        sql = {'edit': "UPDATE messages SET content='Actually adopt option red' WHERE id=2",
               'hidden': "UPDATE messages SET display_kind='hidden' WHERE id=2",
               'rewind': 'UPDATE messages SET active=0,compacted=0 WHERE id=2',
               'delete': 'DELETE FROM messages WHERE id=2'}.get(mutation)
        if sql:
            with sqlite3.connect(service.archive.path) as conn:
                conn.execute(sql)
        args = {'session_id': 's', 'synthesize': False}
        if mutation == 'range':
            args['start'] = '2024-06-14'
        output = service.recall(args)
        record('S-003-invalidated-'+mutation, output['coverage']['summaries_used'] == 0
               and not any(e['summary'] for e in output['evidence']),
               summaries_used=output['coverage']['summaries_used'], raw_ids=[e['source']['message_id'] for e in output['evidence']])
        service.db.close()

    service = fixture(root, 'tail-sample', [(i, 's', 'user', f'message {i}', BASE+i) for i in range(1, 201)])
    sample = service.archive.get_message_sample('s')
    prompt, supplied, truncated = h._summary_input(sample)
    record('S-003-true-tail-sample', supplied[0]['id'] == 1 and supplied[-1]['id'] == 200,
           supplied_ids=[r['id'] for r in supplied])
    service.db.close()

    # S-004 and S-005: identity branches and frozen calendar/rolling bounds.
    service = fixture(root, 'identity', [(1, 's', 'user', 'project alpha', BASE)],
                      config=RemnantConfig(agent_id='alice', runtime_identity_enabled=True, history_summary_enabled=False))
    service.db.insert_turn(session_id='s', agent_id='alice', user_text='owned', assistant_text='')
    ref = h._reference_from_message(service.archive.get_messages('s')[0])
    results = {}
    for owner, trusted in [('alice', set()), ('bob', set()), ('unknown', set()), ('current', {'s'})]:
        results[owner] = service.archive.validate_reference(ref, agent_id=owner, remnant_db=service.db,
                            runtime_identity_enabled=True, trusted_session_ids=trusted) is not None
    record('S-004-runtime', results == {'alice': True, 'bob': False, 'unknown': False, 'current': True}, **results)
    service.db.close()
    for relative in ('yesterday', '2d'):
        service = fixture(root, 'clock-'+relative, [(i, 's', 'user', 'short text', BASE+i) for i in range(1, 86)])
        service._now = datetime(2024, 6, 15, tzinfo=timezone.utc)
        first = service.recall({'relative': relative, 'synthesize': False})
        service._now = datetime(2024, 6, 16, tzinfo=timezone.utc)
        seen, rest = drain(service, {'relative': relative, 'cursor': first['next_cursor']})
        all_ids = [e['source']['message_id'] for e in first['evidence']] + seen
        record('S-005-'+relative, sorted(all_ids) == list(range(1,86)) and
               all(p['resolved_range'] == first['resolved_range'] for p in rest),
               returned=len(all_ids), all_ranges_frozen=all(p['resolved_range'] == first['resolved_range'] for p in rest))
        service.db.close()

    # S-006: returned evidence must contain the actual late match, not just its ID.
    service = fixture(root, 'tailmatch', [(1, 's', 'user', 'x'*2200+' quasar', BASE)])
    output = service.recall({'query': 'quasar', 'synthesize': False})
    record('S-006-match-local-evidence', any('quasar' in e['excerpt'] for e in output['evidence']),
           returned_ids=[e['source']['message_id'] for e in output['evidence']],
           excerpt_contains_query=any('quasar' in e['excerpt'] for e in output['evidence']))
    service.db.close()
    for count in (1,25):
        service = fixture(root, f'union-{count}', [(1, 'raw', 'user', 'quasar raw discussion', BASE)] +
                          [(i+2, f'cached{i}', 'user', 'star discussion', BASE+i+1) for i in range(count)])
        for i in range(count):
            cache(service, f'cached{i}')
        seen, pages = drain(service, {'query': 'quasar'})
        record(f'S-006-union-{count}', sorted(seen) == list(range(1,count+2)),
               expected=count+1, returned=seen, pages=len(pages), last_has_more=pages[-1]['has_more'],
               page_ids=[[e['source']['message_id'] for e in p['evidence']] for p in pages])
        service.db.close()

    # S-007: descriptors, real lock timeout, and contention between discovery/reauthorization.
    service = fixture(root, 'deadline', [(1, 's', 'user', 'owned', BASE)],
                      config=RemnantConfig(agent_id='alice', runtime_identity_enabled=True, history_summary_enabled=False))
    service.db.insert_turn(session_id='s', agent_id='alice', user_text='owned', assistant_text='')
    gc.collect()
    gc.disable()
    try:
        before = len(list(Path('/proc/self/fd').iterdir()))
        for _ in range(50):
            service.archive.get_session('s')
        after = len(list(Path('/proc/self/fd').iterdir()))
    finally:
        gc.enable()
        gc.collect()
    record('S-007-close', after <= before+1, before=before, after=after)
    def lock_thread(duration):
        ready = threading.Event()
        def hold():
            with service.db.read():
                ready.set()
                time.sleep(duration)
        thread = threading.Thread(target=hold)
        thread.start()
        assert ready.wait(1)
        return thread
    thread = lock_thread(3.3)
    start = time.monotonic()
    output = service.recall({'session_id': 's', 'synthesize': False})
    elapsed = time.monotonic()-start
    thread.join()
    record('S-007-initial-lock', elapsed < 3.2 and output['status'] == 'partial',
           elapsed_s=elapsed, status=output['status'], complete=output['coverage']['discovery_complete'])
    original = service.archive.validate_session
    threads = []
    def lock_after_discovery(*args, **kwargs):
        result = original(*args, **kwargs)
        if not threads:
            threads.append(lock_thread(3.3))
        return result
    exception = None
    with patch.object(service.archive, 'validate_session', lock_after_discovery):
        try:
            output = service.recall({'session_id': 's', 'synthesize': False})
        except Exception as exc:
            exception = type(exc).__name__
    for thread in threads:
        thread.join()
    record('S-007-reauthorization-lock', exception is None and output['status'] == 'partial',
           escaped_exception=exception, deadline_s=h.HISTORY_QUERY_DEADLINE_S)
    service.db.close()

    # S-008: original extremes plus numeric conversion in a structurally valid cursor.
    service = fixture(root, 'bounds', [(1, 's', 'user', 'text', BASE)])
    for name, args in [('relative', {'relative': '999999999999999999999d'}),
                       ('date', {'start': '9999-12-31'}),
                       ('anchor', {'session_id': 's', 'around_message_id': 10**100})]:
        output = service.recall(args)
        record('S-008-'+name, output['status'] == 'invalid_request', status=output['status'])
    request, resolved = service._validate_request(h.HistoryRequest(start='2024-06-14', synthesize=False))
    token = h._encode_cursor({'v': 1, 'fingerprint': h._fingerprint(request, service.agent_id),
                             'resolved_range': resolved.as_dict(), 'mode': 'date', 'after': [10**400, 's']})
    exception = None
    try:
        output = service.recall({'start': '2024-06-14', 'cursor': token, 'synthesize': False})
    except Exception as exc:
        exception = type(exc).__name__
    record('S-008-cursor-number', exception is None and output['status'] == 'invalid_request',
           escaped_exception=exception, cursor_bytes=len(token))
    service.db.close()

    # S-009: recover once present, then reject replacement with an escaping link.
    home = root/'late'
    home.mkdir()
    db = open_db(home/'remnant.db')
    service = h.HistoryService(db, RemnantConfig(history_summary_enabled=False), profile_home=home)
    before = service.recall({'query': 'quasar', 'synthesize': False})['status']
    with sqlite3.connect(home/'state.db') as conn:
        conn.executescript(SCHEMA)
    after = service.recall({'query': 'quasar', 'synthesize': False})['status']
    (home/'state.db').rename(root/'state.db')
    (home/'state.db').symlink_to(root/'state.db')
    escaped = service.recall({'query': 'quasar', 'synthesize': False})['status']
    record('S-009-recovery-containment', before == escaped == 'unavailable' and after != 'unavailable',
           missing=before, created=after, escaped_symlink=escaped)
    db.close()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='remnant-correction-review-') as temp:
        run(Path(temp))
    print(json.dumps(RESULTS, indent=2))
