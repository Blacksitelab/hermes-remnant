"""Offline, additive fleet recovery from explicitly ordered SQLite snapshots.

Sources are read only. The first source wins overlapping evidence; conflicting
memory content, status or ownership is rejected. Numeric turn IDs are remapped.
The result is published as a new file only after integrity and boundary checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from typing import Any

from .db import open_db

# Rebuild disposable caches/telemetry instead of merging stale ranking state.
TABLES = (
    'turns', 'memories', 'entities', 'entity_aliases', 'claims', 'memory_entities',
    'relations', 'relation_evidence', 'embeddings', 'threads', 'vault_files',
    'vault_passages', 'dream_state', 'entity_sightings', 'audit_log',
    'echo_receipts', 'echo_receipt_items', 'echo_signals', 'extraction_queue', 'echo_jobs',
)
DISPOSABLE = (
    'embedding_cache', 'prefetch_stats', 'operation_metrics', 'echo_utility',
    'echo_pair_utility', 'echo_daily_metrics',
)
OWNER_COLUMNS = {
    'turns': 'agent_id', 'memories': 'agent', 'entities': 'agent',
    'entity_aliases': 'agent', 'memory_entities': 'agent', 'entity_sightings': 'agent',
    'threads': 'owner', 'vault_files': 'agent', 'vault_passages': 'agent',
    'dream_state': 'owner', 'echo_receipts': 'agent_id', 'echo_signals': 'agent_id',
    'extraction_queue': 'agent_id',
}
INTEGER_IDS = {'turns', 'audit_log', 'echo_signals', 'extraction_queue', 'echo_jobs'}
TURN_COLUMNS = {
    'claims': 'source_turn_id', 'echo_receipts': 'turn_id',
    'echo_receipt_items': 'source_turn_id', 'extraction_queue': 'turn_id',
}


def _snapshot(source: Path, destination: Path) -> None:
    with closing(sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True)) as src:
        with closing(sqlite3.connect(destination)) as dst:
            src.backup(dst)
    db = open_db(destination)
    try:
        if db._conn.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError(f'database integrity failed: {source}')
        if db._conn.execute('PRAGMA foreign_key_check').fetchone():
            raise ValueError(f'foreign key check failed: {source}')
    finally:
        db.close()


def _signature(table: str, row: dict[str, Any]) -> bytes:
    if table == 'turns':
        keys = ('agent_id', 'session_id', 'user_text', 'assistant_text', 'created_at')
    elif table == 'extraction_queue':
        keys = ('turn_id',)
    elif table == 'echo_jobs':
        keys = ('receipt_id', 'job_type', 'target_ids')
    else:
        keys = tuple(row)  # Keep distinct events with distinct original IDs.
    return hashlib.sha256(json.dumps([row[k] for k in keys],
                                     ensure_ascii=False).encode()).digest()


def _remap_turn_json(text: str | None, turns: dict[int, int]) -> str | None:
    if not text:
        return text
    def remap(value: Any) -> Any:
        if isinstance(value, list):
            return [remap(item) for item in value]
        if isinstance(value, dict):
            return {
                key: (str(turns[int(item)]) if isinstance(item, str) else turns[int(item)])
                if key in ('turn_id', 'source_turn_id') and item is not None
                else remap(item)
                for key, item in value.items()
            }
        return value
    original = json.loads(text)
    updated = remap(original)
    return json.dumps(updated, ensure_ascii=False) if original != updated else text


def build_recovery(manifest: dict[str, Any], output: str | Path) -> dict[str, Any]:
    """Build a new database; never update an input or replace an existing output."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    specs = manifest['sources']
    if not specs or len({str(Path(s['path']).resolve()) for s in specs}) != len(specs):
        raise ValueError('sources must be nonempty and distinct')
    owner_sources: dict[str, str] = {}
    for spec in specs:
        for source_owner, target_owner in spec.get('owners', {}).items():
            if owner_sources.setdefault(target_owner, source_owner) != source_owner:
                raise ValueError('distinct profile owners cannot share a destination')
        if not spec.get('owners') or any(not v or v == 'system' for v in spec['owners'].values()):
            raise ValueError('each source needs an explicit nonempty owner mapping')
    report: dict[str, Any] = {'sources': [], 'disposable_tables_rebuilt': list(DISPOSABLE)}
    with tempfile.TemporaryDirectory(prefix='remnant-recovery-', dir=output.parent) as temp:
        draft = Path(temp) / 'result.db'
        db = open_db(draft)
        conn = db._conn
        conn.executescript('''
            CREATE TABLE recovery_turn_map(source TEXT, old_id INTEGER, new_id INTEGER,
                                           PRIMARY KEY(source,old_id));
            CREATE TABLE recovery_claim_map(source TEXT, old_id TEXT, new_id TEXT,
                                            PRIMARY KEY(source,old_id));
            CREATE TEMP TABLE seen_integer_rows(table_name TEXT, signature BLOB, new_id INTEGER,
                                                PRIMARY KEY(table_name,signature));
        ''')
        try:
            for index, spec in enumerate(specs):
                source_path = Path(spec['path']).resolve()
                snapshot = Path(temp) / f'source-{index}.db'
                _snapshot(source_path, snapshot)
                source = sqlite3.connect(snapshot)
                source.row_factory = sqlite3.Row
                owner_map = spec['owners']
                turn_map: dict[int, int] = {}
                claim_map: dict[str, str] = {}
                counts: dict[str, Any] = {}
                try:
                    with db.transaction():
                        for table in TABLES:
                            counts[table] = {'source': 0, 'inserted': 0, 'overlap': 0}
                            columns = list(conn.execute(f'PRAGMA table_info({table})'))
                            pk = [r['name'] for r in sorted(columns, key=lambda r: r['pk'])
                                  if r['pk']]
                            for raw in source.execute(f'SELECT * FROM {table} ORDER BY rowid'):
                                row = dict(raw)
                                counts[table]['source'] += 1
                                owner_col = OWNER_COLUMNS.get(table)
                                if owner_col:
                                    owner = row[owner_col]
                                    if table == 'threads' and not owner:
                                        owner = spec.get('thread_owners', {}).get(row['id'])
                                    if table == 'dream_state' and not owner:
                                        owner = spec.get('dream_owner')
                                    if owner in owner_map:
                                        row[owner_col] = owner_map[owner]
                                    elif table in ('entities', 'entity_aliases') and not owner:
                                        pass  # Legacy graph nodes require owned backing evidence.
                                    else:
                                        raise ValueError(
                                            f'unmapped owner in {source_path}: {table}')
                                if table == 'turns':
                                    original_turn = row['id']
                                turn_col = TURN_COLUMNS.get(table)
                                if turn_col and row[turn_col] is not None:
                                    row[turn_col] = turn_map[int(row[turn_col])]
                                if (table == 'memories'
                                    and row['source'] == 'conversation'):
                                    if row['source_id'] is not None:
                                        row['source_id'] = str(turn_map[int(row['source_id'])])
                                json_col = {'memories': 'metadata', 'audit_log': 'details',
                                            'claims': 'qualifiers'}.get(table)
                                if json_col:
                                    row[json_col] = _remap_turn_json(row[json_col], turn_map)
                                if table == 'relation_evidence' and row['claim_id']:
                                    row['claim_id'] = claim_map[row['claim_id']]
                                if table == 'echo_receipt_items' and row['item_kind'] == 'pending':
                                    row['memory_id'] = f"pending:{row['source_turn_id']}"
                                if table in INTEGER_IDS:
                                    signature = _signature(table, row)
                                    existing = conn.execute(
                                        'SELECT new_id FROM seen_integer_rows '
                                        'WHERE table_name=? AND signature=?', (table, signature),
                                    ).fetchone()
                                    original_id = row.pop('id')
                                else:
                                    existing = conn.execute(
                                        f"SELECT * FROM {table} WHERE "
                                        + ' AND '.join(f'{key} IS ?' for key in pk),
                                        [row[key] for key in pk],
                                    ).fetchone()
                                    if table == 'claims' and existing is None:
                                        existing = conn.execute(
                                            'SELECT * FROM claims WHERE memory_id=?',
                                            (row['memory_id'],),
                                        ).fetchone()
                                if existing is not None:
                                    if table == 'memories':
                                        if any(row[k] != existing[k] for k in
                                               ('agent', 'content', 'status', 'superseded_by',
                                                'source', 'source_id')):
                                            raise ValueError(
                                                'conflicting memory content, status or ownership')
                                        if row['updated_at'] > existing['updated_at']:
                                            raise ValueError(
                                                'newer overlapping memory: reorder source priority')
                                    elif owner_col and table not in INTEGER_IDS:
                                        if row[owner_col] != existing[owner_col]:
                                            raise ValueError(f'conflicting ownership in {table}')
                                    counts[table]['overlap'] += 1
                                    new_id = existing['new_id'] if table in INTEGER_IDS else None
                                else:
                                    names = list(row)
                                    cursor = conn.execute(
                                        f"INSERT INTO {table} ({','.join(names)}) VALUES "
                                        f"({','.join('?' for _ in names)})", list(row.values()),
                                    )
                                    counts[table]['inserted'] += 1
                                    new_id = cursor.lastrowid
                                    if table in INTEGER_IDS:
                                        conn.execute('INSERT INTO seen_integer_rows VALUES(?,?,?)',
                                                     (table, signature, new_id))
                                if table == 'turns':
                                    turn_map[original_turn] = int(new_id)
                                    conn.execute('INSERT INTO recovery_turn_map VALUES(?,?,?)',
                                                 (str(source_path), original_id, new_id))
                                if table == 'claims':
                                    claim_map[raw['id']] = existing['id'] if existing else row['id']
                                    conn.execute('INSERT INTO recovery_claim_map VALUES(?,?,?)',
                                                 (str(source_path), raw['id'],
                                                  claim_map[raw['id']]))
                    report['sources'].append({'path': str(source_path), 'tables': counts})
                finally:
                    source.close()
            # All relational checks run before the new database is made available.
            if conn.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('recovered database failed integrity check')
            if conn.execute('PRAGMA foreign_key_check').fetchone():
                raise ValueError('recovered database has broken references')
            bad_turns = conn.execute('''
                SELECT COUNT(*) FROM memories m
                LEFT JOIN turns t ON m.source_id=CAST(t.id AS TEXT)
                WHERE m.source='conversation' AND m.source_id IS NOT NULL
                  AND (t.id IS NULL OR m.agent<>t.agent_id)
            ''').fetchone()[0]
            bad_claims = conn.execute('''
                SELECT COUNT(*) FROM claims c LEFT JOIN turns t ON c.source_turn_id=t.id
                JOIN memories m ON m.id=c.memory_id WHERE c.source_turn_id IS NOT NULL
                  AND (t.id IS NULL OR m.agent<>t.agent_id)
            ''').fetchone()[0]
            if bad_turns or bad_claims:
                raise ValueError('recovered evidence crosses a profile boundary')
            report['memories_by_owner'] = dict(conn.execute(
                'SELECT agent,COUNT(*) FROM memories GROUP BY agent'))
            report['threads_by_owner'] = dict(conn.execute(
                'SELECT owner,COUNT(*) FROM threads GROUP BY owner'))
            report['integrity'] = 'ok'
            report['turns'] = conn.execute('SELECT COUNT(*) FROM turns').fetchone()[0]
            conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        finally:
            db.close()
        os.chmod(draft, 0o600)
        os.link(draft, output)  # Fails rather than overwriting a concurrently created output.
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = build_recovery(json.loads(args.manifest.read_text()), args.output)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'sources'}))


if __name__ == '__main__':
    main()
