"""Disposable build/import/migration and temporal checks; paths supplied explicitly."""
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import remnant
from remnant.db import SCHEMA_VERSION, open_db

mode = sys.argv[1]
if mode == 'import':
    assert remnant.__version__ == version('hermes-remnant')
    print(json.dumps({'import_path': remnant.__file__, 'version': remnant.__version__,
                      'python': sys.version, 'sqlite': sqlite3.sqlite_version}))
elif mode == 'migration-seed':
    db = open_db(Path(sys.argv[2]))
    assert SCHEMA_VERSION == 17
    db.insert_turn(session_id='migration', agent_id='owner', user_text='synthetic user', assistant_text='synthetic assistant')
    db.insert_memory(content='synthetic durable fact', source='manual', agent='owner')
    rows = {table: [dict(r) for r in db._conn.execute('SELECT * FROM '+table)] for table in ['turns','memories','claims']}
    Path(sys.argv[3]).write_text(json.dumps(rows, sort_keys=True))
    db.close()
    print('seeded genuine schema 17 with durable evidence')
elif mode == 'migration-check':
    db = open_db(Path(sys.argv[2]))
    assert SCHEMA_VERSION == 18
    rows = {table: [dict(r) for r in db._conn.execute('SELECT * FROM '+table)] for table in ['turns','memories','claims']}
    assert rows == json.loads(Path(sys.argv[3]).read_text())
    assert db._conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    db.close()
    db = open_db(Path(sys.argv[2]))
    assert db._conn.execute('SELECT COUNT(*) FROM history_summaries').fetchone()[0] == 0
    db.close()
    print('schema 17 -> 18 -> reopen: durable rows identical, cache empty, integrity ok')
elif mode == 'dates':
    from remnant.history import resolve_range
    checks = []
    for day, zone, hours in [('2024-02-29','UTC',24), ('2024-09-29','Pacific/Auckland',23),
                             ('2024-04-07','Pacific/Auckland',25), ('2024-03-10','America/New_York',23),
                             ('2024-11-03','America/New_York',25)]:
        r = resolve_range(start=day, timezone_name=zone)
        actual = (r.end_utc-r.start_utc).total_seconds()/3600
        assert actual == hours
        checks.append({'day':day, 'zone':zone, 'hours':actual})
    for relative, start, end in [('last_week', '2024-06-03T00:00:00Z', '2024-06-10T00:00:00Z'),
                                  ('yesterday', '2024-06-13T00:00:00Z','2024-06-14T00:00:00Z')]:
        r = resolve_range(relative=relative, reference_now=datetime(2024,6,14,12,tzinfo=timezone.utc))
        assert r.as_dict()['start_utc'] == start and r.as_dict()['end_utc'] == end
    print(json.dumps({'calendar_boundaries': checks, 'fixed_clock_relative': 'passed'}))
elif mode == 'source-scan':
    # Inventory all tracked source, flag executable/deserialization primitives and secret
    # indicators by location only. Never scan untracked reports, config, or environment.
    import ast
    import re
    import subprocess
    files = subprocess.check_output(['git','ls-files','remnant','tests'], text=True).splitlines()
    calls, indicators, totals = [], [], {'modules':0, 'lines':0}
    for filename in files:
        if not filename.endswith('.py'):
            continue
        text = Path(filename).read_text()
        tree = ast.parse(text)
        totals['modules'] += 1
        totals['lines'] += len(text.splitlines())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if name in {'eval','exec','pickle.loads','pickle.load','os.system','subprocess.run','yaml.load'}:
                    calls.append({'file':filename, 'line':node.lineno, 'call':name})
        for number, line in enumerate(text.splitlines(), 1):
            if re.search(r'BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY|\b(?:sk-proj-|ghp_)[A-Za-z0-9]{20,}', line):
                indicators.append({'file':filename,'line':number})
    print(json.dumps({'inventory': totals, 'risky_primitives': calls, 'secret_indicators': indicators}, indent=2))
elif mode == 'performance':
    import statistics
    baseline = json.loads(Path(sys.argv[2]).read_text())
    candidate = json.loads(Path(sys.argv[3]).read_text())
    probes = json.loads(Path(sys.argv[4]).read_text())
    assert baseline['top100'] == candidate['top100']
    assert baseline['scores'] == candidate['scores']
    keys = ['rows','dimensions','semantic_ms','provider_prefetch_ms','provider_delivered','probes','python_scoring_peak_mib']
    print(json.dumps({'baseline': {k: baseline[k] for k in keys},
        'candidate': {k: candidate[k] for k in keys}, 'top100_and_scores_identical': True,
        'prefetch_delta_percent': (candidate['provider_prefetch_ms']/baseline['provider_prefetch_ms']-1)*100,
        'history_30000_no_hit_median_ms': statistics.median(probes['large_archive_no_hit']['runs_ms'])}, indent=2))
else:
    raise ValueError(mode)
