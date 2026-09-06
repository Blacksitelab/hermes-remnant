"""SQLite storage for Remnant: schema, migrations, CRUD.

- WAL mode for concurrent reader + single writer.
- FTS5 on memory text for BM25 keyword search.
- float32 blob embeddings stored in a separate `embeddings` table.
- Entity graph tables (`entities`, `memory_entities`, `relations`).
- `extraction_queue` table persists pending turns across restarts.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import struct
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from .scope import VISIBILITY_ORDER, normalize_profile_scope, path_in_profile_scope

SCHEMA_VERSION = 17

# Shared DB home: a single SQLite database used across all Hermes profiles /
# agents. Provider APIs enforce ownership; only operator APIs allow unscoped
# reads. REMNANT_DB_HOME overrides the file location (also used by tests).
DEFAULT_DB_HOME = Path("~/.hermes/remnant").expanduser()
DB_FILENAME = "remnant.db"


def default_db_path() -> Path:
    """Return the shared Remnant DB path.

    Honors the ``REMNANT_DB_HOME`` env var (primarily for tests); defaults to
    ``~/.hermes/remnant/remnant.db``. Config remains profile-scoped under
    ``hermes_home``; only the DB is shared.
    """
    home = Path(os.environ.get("REMNANT_DB_HOME", str(DEFAULT_DB_HOME)))
    return home / DB_FILENAME

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    user_text TEXT NOT NULL,
    assistant_text TEXT NOT NULL,
    created_at REAL NOT NULL,
    extraction_status TEXT NOT NULL DEFAULT 'pending',
    extraction_attempts INTEGER NOT NULL DEFAULT 0,
    extraction_fact_count INTEGER,
    extraction_completed_at REAL,
    extraction_error TEXT
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL CHECK(type IN ('fact','observation','conversation','document','thread')),
    content TEXT NOT NULL,
    source TEXT NOT NULL CHECK(source IN (
        'conversation','vault','email','cron','sensor',
        'manual','import','hindsight','dream'
    )),
    source_id TEXT,
    agent TEXT,
    visibility TEXT DEFAULT 'private',
    timestamp TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    trust_score REAL DEFAULT 0.5,
    verified INTEGER DEFAULT 0,
    superseded_by TEXT,
    status TEXT DEFAULT 'active',
    tags TEXT,
    metadata TEXT,
    content_hash TEXT,
    seen_count INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mem_agent ON memories(agent);
CREATE INDEX IF NOT EXISTS idx_mem_visibility ON memories(visibility);
CREATE INDEX IF NOT EXISTS idx_mem_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_mem_source ON memories(source);
CREATE INDEX IF NOT EXISTS idx_mem_content_hash ON memories(content_hash);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, tags,
    content='memories', content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT,
    type TEXT,
    aliases TEXT,
    agent TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_agent ON entities(agent);

CREATE TABLE IF NOT EXISTS entity_aliases (
    entity_id TEXT NOT NULL,
    alias TEXT NOT NULL,
    agent TEXT,
    PRIMARY KEY(entity_id, alias, agent),
    FOREIGN KEY(entity_id) REFERENCES entities(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_aliases_alias ON entity_aliases(alias);

CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    relation_role TEXT,
    agent TEXT,
    PRIMARY KEY(memory_id, entity_id),
    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_me_entity ON memory_entities(entity_id);
CREATE INDEX IF NOT EXISTS idx_me_agent ON memory_entities(agent);
CREATE INDEX IF NOT EXISTS idx_me_memory ON memory_entities(memory_id);

CREATE TABLE IF NOT EXISTS relations (
    entity_a TEXT NOT NULL,
    entity_b TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    strength REAL,
    source_memory_id TEXT,
    created_at TEXT,
    PRIMARY KEY(entity_a, entity_b, relation_type),
    FOREIGN KEY(entity_a) REFERENCES entities(id) ON DELETE CASCADE,
    FOREIGN KEY(entity_b) REFERENCES entities(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_relations_a ON relations(entity_a);
CREATE INDEX IF NOT EXISTS idx_relations_b ON relations(entity_b);

CREATE TABLE IF NOT EXISTS relation_evidence (
    entity_a TEXT NOT NULL,
    entity_b TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    claim_id TEXT,
    strength REAL NOT NULL DEFAULT 0.5,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(entity_a, entity_b, relation_type, memory_id),
    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_relation_evidence_memory
    ON relation_evidence(memory_id, active);
CREATE INDEX IF NOT EXISTS idx_relation_evidence_relation
    ON relation_evidence(entity_a, entity_b, relation_type, active);

CREATE TABLE IF NOT EXISTS embeddings (
    memory_id TEXT PRIMARY KEY,
    model TEXT,
    embedding BLOB,
    dimensions INTEGER,
    created_at TEXT,
    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS extraction_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id INTEGER NOT NULL,
    session_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    user_text TEXT NOT NULL,
    assistant_text TEXT NOT NULL,
    enqueued_at REAL NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    next_attempt_at REAL NOT NULL DEFAULT 0,
    started_at REAL,
    last_error TEXT,
    FOREIGN KEY(turn_id) REFERENCES turns(id)
);
CREATE INDEX IF NOT EXISTS idx_queue_status ON extraction_queue(status);

CREATE TABLE IF NOT EXISTS embedding_cache (
    model TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY(model, text_hash)
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    memory_id TEXT,
    details TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_memory ON audit_log(memory_id);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action);

CREATE TABLE IF NOT EXISTS vault_files (
    agent TEXT NOT NULL DEFAULT '',
    path TEXT NOT NULL,
    hash TEXT NOT NULL,
    memory_id TEXT,
    indexed_at TEXT NOT NULL,
    PRIMARY KEY(agent, path)
);
CREATE INDEX IF NOT EXISTS idx_vault_hash ON vault_files(hash);
CREATE INDEX IF NOT EXISTS idx_vault_memory ON vault_files(memory_id);

CREATE TABLE IF NOT EXISTS vault_passages (
    agent TEXT NOT NULL DEFAULT '',
    path TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    memory_id TEXT NOT NULL,
    heading_path TEXT,
    start_offset INTEGER,
    end_offset INTEGER,
    PRIMARY KEY(agent, path, ordinal),
    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_vault_passages_memory ON vault_passages(memory_id);

-- Structured claim projections retain a versioned, queryable view of facts.
-- The backing memory is always the source of truth and is never overwritten.
CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL UNIQUE,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    qualifiers TEXT,
    confidence REAL DEFAULT 0.5,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK(status IN ('active', 'superseded', 'contradicted')),
    valid_from TEXT,
    valid_to TEXT,
    observed_at TEXT,
    event_at TEXT,
    scope_type TEXT,
    scope_value TEXT,
    modality TEXT DEFAULT 'asserted',
    conflict_type TEXT,
    resolution_status TEXT DEFAULT 'active',
    extractor_version TEXT,
    source_turn_id INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_claims_subject_predicate
    ON claims(subject, predicate, status, updated_at);

-- Phase 5: topic threads + dream-loop machine state.
CREATE TABLE IF NOT EXISTS threads (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    topic TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK(status IN ('active','stale','resolved')),
    importance REAL DEFAULT 0.5,
    tags TEXT,
    related_entities TEXT,
    source TEXT,
    added_by TEXT,
    owner TEXT,
    created_at TEXT NOT NULL,
    last_activity TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_threads_status ON threads(status);
CREATE INDEX IF NOT EXISTS idx_threads_topic ON threads(topic);
CREATE INDEX IF NOT EXISTS idx_threads_last_activity ON threads(last_activity);

CREATE TABLE IF NOT EXISTS dream_state (
    owner TEXT NOT NULL DEFAULT '',
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(owner, key)
);

-- Issue #5: pending entity sightings. When the regex extraction path defers
-- entity creation until a name has been sighted in >= ``entity_min_memories``
-- distinct memories, each sighting is recorded here keyed by the normalized
-- name + agent. Once the threshold is reached the entity is created, linked
-- to every sighted memory, and the sighting rows are cleared. Entities that
-- pre-date this mechanism (single-mention, already linked) are left alone.
CREATE TABLE IF NOT EXISTS entity_sightings (
    name_key TEXT NOT NULL,
    agent TEXT,
    memory_id TEXT NOT NULL,
    seen_at TEXT NOT NULL,
    PRIMARY KEY(name_key, agent, memory_id)
);
CREATE INDEX IF NOT EXISTS idx_sightings_name ON entity_sightings(name_key, agent);

-- Prefetch stats: track every prefetch() call and its outcome so we can
-- measure the empty-return rate, latency, and which rejection path fires.
CREATE TABLE IF NOT EXISTS prefetch_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    outcome TEXT NOT NULL,          -- 'injected' or 'empty'
    reason TEXT,                    -- why empty (deadline/no_results/greeting/etc.)
    elapsed_ms REAL,                 -- wall-clock time spent in prefetch()
    result_count INTEGER DEFAULT 0,  -- number of memories injected (0 if empty)
    token_estimate INTEGER DEFAULT 0,
    query TEXT,                     -- the user query (truncated for storage)
    agent_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_prefetch_outcome ON prefetch_stats(outcome);
CREATE INDEX IF NOT EXISTS idx_prefetch_created ON prefetch_stats(created_at);

-- Echo: consumed-context receipts and bounded outcome-aware utility.
CREATE TABLE IF NOT EXISTS echo_receipts (
    id TEXT PRIMARY KEY,
    activation_key TEXT NOT NULL UNIQUE,
    session_id TEXT NOT NULL,
    turn_id INTEGER,
    agent_id TEXT NOT NULL,
    viewer_key_hash TEXT NOT NULL,
    profile_scope_hash TEXT NOT NULL,
    query_fingerprint TEXT NOT NULL,
    query_archetype TEXT NOT NULL,
    context_hash TEXT NOT NULL,
    memory_generation INTEGER NOT NULL,
    rendered_count INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    policy_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('open','closed','expired')),
    outcome TEXT,
    created_at REAL NOT NULL,
    closed_at REAL,
    FOREIGN KEY(turn_id) REFERENCES turns(id)
);
CREATE INDEX IF NOT EXISTS idx_echo_receipt_match
    ON echo_receipts(session_id, query_fingerprint, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_echo_receipt_created ON echo_receipts(created_at);
CREATE INDEX IF NOT EXISTS idx_echo_receipt_turn ON echo_receipts(turn_id);

CREATE TABLE IF NOT EXISTS echo_receipt_items (
    receipt_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    item_kind TEXT NOT NULL CHECK(item_kind IN ('memory','pending')),
    source_turn_id INTEGER,
    evidence_class TEXT NOT NULL,
    score_lane TEXT,
    base_score REAL NOT NULL,
    base_rank INTEGER NOT NULL,
    rendered_tokens INTEGER NOT NULL,
    rendered_hash TEXT NOT NULL,
    claim_status TEXT,
    PRIMARY KEY(receipt_id, memory_id),
    FOREIGN KEY(receipt_id) REFERENCES echo_receipts(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_echo_item_memory
    ON echo_receipt_items(memory_id, receipt_id);

CREATE TABLE IF NOT EXISTS echo_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    receipt_id TEXT,
    memory_id TEXT NOT NULL,
    paired_memory_id TEXT,
    agent_id TEXT NOT NULL,
    viewer_key_hash TEXT NOT NULL,
    query_archetype TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    direction INTEGER NOT NULL CHECK(direction IN (-1, 1)),
    weight REAL NOT NULL CHECK(weight > 0 AND weight <= 1),
    source TEXT NOT NULL,
    evaluator_version TEXT,
    created_at REAL NOT NULL,
    aggregated_at REAL,
    FOREIGN KEY(receipt_id) REFERENCES echo_receipts(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_echo_signal_pending
    ON echo_signals(aggregated_at, created_at);
CREATE INDEX IF NOT EXISTS idx_echo_signal_memory
    ON echo_signals(memory_id, query_archetype, created_at);

CREATE TABLE IF NOT EXISTS echo_utility (
    agent_id TEXT NOT NULL,
    viewer_key_hash TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    query_archetype TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    explicit_positive_mass REAL NOT NULL DEFAULT 0,
    explicit_negative_mass REAL NOT NULL DEFAULT 0,
    inferred_positive_mass REAL NOT NULL DEFAULT 0,
    inferred_negative_mass REAL NOT NULL DEFAULT 0,
    explicit_positive INTEGER NOT NULL DEFAULT 0,
    explicit_negative INTEGER NOT NULL DEFAULT 0,
    evaluator_samples INTEGER NOT NULL DEFAULT 0,
    effective_observations REAL NOT NULL DEFAULT 0,
    utility_mean REAL NOT NULL DEFAULT 0.5,
    harm_risk REAL NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0,
    last_signal_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY(agent_id, viewer_key_hash, memory_id, query_archetype, policy_version)
);
CREATE INDEX IF NOT EXISTS idx_echo_utility_lookup
    ON echo_utility(agent_id, viewer_key_hash, query_archetype, memory_id);
CREATE INDEX IF NOT EXISTS idx_echo_utility_updated ON echo_utility(updated_at);

CREATE TABLE IF NOT EXISTS echo_pair_utility (
    agent_id TEXT NOT NULL,
    viewer_key_hash TEXT NOT NULL,
    first_memory_id TEXT NOT NULL,
    second_memory_id TEXT NOT NULL,
    query_archetype TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    positive_mass REAL NOT NULL DEFAULT 0,
    negative_mass REAL NOT NULL DEFAULT 0,
    sample_count INTEGER NOT NULL DEFAULT 0,
    synergy_score REAL NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0,
    last_signal_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY(agent_id, viewer_key_hash, first_memory_id, second_memory_id,
                query_archetype, policy_version),
    CHECK(first_memory_id < second_memory_id)
);
CREATE INDEX IF NOT EXISTS idx_echo_pair_first
    ON echo_pair_utility(agent_id, viewer_key_hash, first_memory_id, query_archetype);
CREATE INDEX IF NOT EXISTS idx_echo_pair_second
    ON echo_pair_utility(agent_id, viewer_key_hash, second_memory_id, query_archetype);

CREATE TABLE IF NOT EXISTS echo_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    receipt_id TEXT NOT NULL,
    job_type TEXT NOT NULL CHECK(job_type IN ('single','pair')),
    target_ids TEXT NOT NULL,
    priority REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK(status IN ('pending','running','done','failed','skipped')),
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at REAL NOT NULL DEFAULT 0,
    started_at REAL,
    last_error TEXT,
    evaluator_version TEXT NOT NULL,
    created_at REAL NOT NULL,
    completed_at REAL,
    FOREIGN KEY(receipt_id) REFERENCES echo_receipts(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_echo_job_ready
    ON echo_jobs(status, next_attempt_at, priority DESC, id);

CREATE TABLE IF NOT EXISTS echo_daily_metrics (
    day TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    metric TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    total REAL NOT NULL DEFAULT 0,
    maximum REAL NOT NULL DEFAULT 0,
    PRIMARY KEY(day, agent_id, metric)
);

CREATE TABLE IF NOT EXISTS operation_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation TEXT NOT NULL,
    outcome TEXT NOT NULL,
    elapsed_ms REAL NOT NULL DEFAULT 0,
    input_units INTEGER NOT NULL DEFAULT 0,
    output_units INTEGER NOT NULL DEFAULT 0,
    agent_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_operation_metrics_kind
    ON operation_metrics(operation, outcome, created_at);
"""

# FTS5 triggers keep the index in sync with the base table.
# memories.id is a TEXT PK; FTS5 uses an integer rowid. We map via the
# hidden `rowid` of the memories table and carry content + tags columns.
_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, tags)
    VALUES (new.rowid, new.content, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
    VALUES ('delete', old.rowid, old.content, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
    VALUES ('delete', old.rowid, old.content, old.tags);
    INSERT INTO memories_fts(rowid, content, tags)
    VALUES (new.rowid, new.content, new.tags);
END;
"""


def _pack_embedding(vec: list[float]) -> bytes:
    if not vec or not all(math.isfinite(value) for value in vec):
        raise ValueError("embedding must contain finite values")
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack_embedding(blob: bytes | None) -> list[float]:
    if not blob:
        return []
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _uuid() -> str:
    return str(uuid.uuid4())


class RemnantDB:
    """Thread-safe handle to the Remnant SQLite database."""

    def __init__(self, db_path: str | Path):
        self.path = str(db_path)
        # sqlite connections are not safe for overlapping cursor activity even
        # with check_same_thread=False. WAL helps *separate connections*;
        # this lock protects the provider's single shared connection.
        self._lock = threading.RLock()
        self._deadline: ContextVar[float | None] = ContextVar("remnant_deadline", default=None)
        self._diagnostics: deque[tuple[str, tuple[Any, ...]]] = deque(maxlen=2048)
        self._diagnostics_lock = threading.Lock()
        self._conn = self._open()
        self._migrate()

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,  # autocommit; explicit transactions used where needed
            check_same_thread=False,
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.row_factory = sqlite3.Row
        return conn

    def _migrate(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.executescript(_SCHEMA)
            cur.executescript(_FTS_TRIGGERS)
            self._apply_migrations(cur)
            self._migrate_profile_state(cur)
            vault_columns = {row["name"] for row in cur.execute("PRAGMA table_info(vault_files)")}
            if "agent" not in vault_columns:
                # Preserve every legacy mapping under its backing memory's owner.
                cur.executescript("""
                    BEGIN IMMEDIATE;
                    CREATE TABLE vault_files_v16 (
                        agent TEXT NOT NULL DEFAULT '', path TEXT NOT NULL, hash TEXT NOT NULL,
                        memory_id TEXT, indexed_at TEXT NOT NULL, PRIMARY KEY(agent,path));
                    INSERT INTO vault_files_v16
                        SELECT COALESCE(m.agent,''),v.path,v.hash,v.memory_id,v.indexed_at
                        FROM vault_files v LEFT JOIN memories m ON m.id=v.memory_id;
                    CREATE TABLE vault_passages_v16 (
                        agent TEXT NOT NULL DEFAULT '', path TEXT NOT NULL,
                        ordinal INTEGER NOT NULL,
                        memory_id TEXT NOT NULL, heading_path TEXT, start_offset INTEGER,
                        end_offset INTEGER, PRIMARY KEY(agent,path,ordinal),
                        FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE);
                    INSERT INTO vault_passages_v16
                        SELECT COALESCE(m.agent,''),v.path,v.ordinal,v.memory_id,
                               v.heading_path,v.start_offset,v.end_offset
                        FROM vault_passages v LEFT JOIN memories m ON m.id=v.memory_id;
                    DROP TABLE vault_files;
                    DROP TABLE vault_passages;
                    ALTER TABLE vault_files_v16 RENAME TO vault_files;
                    ALTER TABLE vault_passages_v16 RENAME TO vault_passages;
                    CREATE INDEX idx_vault_hash ON vault_files(hash);
                    CREATE INDEX idx_vault_memory ON vault_files(memory_id);
                    CREATE INDEX idx_vault_passages_memory ON vault_passages(memory_id);
                    COMMIT;
                """)
            cur.execute(
                "INSERT OR IGNORE INTO schema_meta(key,value) VALUES('memory_generation','0')"
            )
            # Cache invalidation follows committed evidence changes across processes.
            for table in ("memories", "claims", "relation_evidence", "embeddings"):
                for event in ("INSERT", "UPDATE", "DELETE"):
                    cur.execute(
                        f"CREATE TRIGGER IF NOT EXISTS generation_{table}_{event} "
                        f"AFTER {event} ON {table} BEGIN UPDATE schema_meta "
                        "SET value=CAST(value AS INTEGER)+1 WHERE key='memory_generation'; END"
                    )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_embedding_cache_created "
                "ON embedding_cache(created_at)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_embeddings_model_dimensions "
                "ON embeddings(model, dimensions)"
            )
            cur.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES(?,?)",
                ("version", str(SCHEMA_VERSION)),
            )

    def _migrate_profile_state(self, cur: sqlite3.Cursor) -> None:
        """Keep authorship separate from ownership; never guess in a mixed store."""
        thread_cols = {row["name"] for row in cur.execute("PRAGMA table_info(threads)")}
        state_cols = {row["name"] for row in cur.execute("PRAGMA table_info(dream_state)")}
        if "owner" in thread_cols and "owner" in state_cols:
            return
        owners = [row[0] for row in cur.execute(
            "SELECT agent FROM memories WHERE agent IS NOT NULL AND agent NOT IN ('','system') "
            "UNION SELECT agent_id FROM turns WHERE agent_id NOT IN ('','system')"
        )]
        sole_owner = owners[0] if len(owners) == 1 else None
        cur.execute("BEGIN IMMEDIATE")
        try:
            if "owner" not in thread_cols:
                cur.execute("ALTER TABLE threads ADD COLUMN owner TEXT")
                cur.execute(
                    "UPDATE threads SET owner=CASE WHEN added_by IS NOT NULL "
                    "AND added_by NOT IN ('','system') THEN added_by ELSE ? END",
                    (sole_owner,),
                )
            if "owner" not in state_cols:
                cur.execute("ALTER TABLE dream_state RENAME TO dream_state_legacy")
                cur.execute(
                    "CREATE TABLE dream_state(owner TEXT NOT NULL DEFAULT '', key TEXT NOT NULL, "
                    "value TEXT NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY(owner,key))"
                )
                cur.execute(
                    "INSERT INTO dream_state SELECT ?,key,value,updated_at FROM dream_state_legacy",
                    (sole_owner or "",),
                )
                cur.execute("DROP TABLE dream_state_legacy")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_threads_owner ON threads(owner)")
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

    def _apply_migrations(self, cur: sqlite3.Cursor) -> None:
        """Idempotent column additions for tables created in earlier phases.

        CREATE TABLE IF NOT EXISTS never alters existing tables, so we add
        new columns explicitly and ignore the OperationalError if they exist.
        """
        cur.execute("PRAGMA table_info(memory_entities)")
        cols = {row["name"] for row in cur.fetchall()}
        if "agent" not in cols:
            try:
                cur.execute("ALTER TABLE memory_entities ADD COLUMN agent TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_me_agent ON memory_entities(agent)"
                )
            except sqlite3.OperationalError:
                pass
            try:
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_me_memory ON memory_entities(memory_id)"
                )
            except sqlite3.OperationalError:
                pass
        cur.execute("PRAGMA table_info(entities)")
        cols = {row["name"] for row in cur.fetchall()}
        if "agent" not in cols:
            try:
                cur.execute("ALTER TABLE entities ADD COLUMN agent TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_entities_agent ON entities(agent)"
                )
            except sqlite3.OperationalError:
                pass
        # entity_aliases.agent scopes aliases per agent (agent-scoped
        # resolution). Added for Phase 3; CREATE TABLE IF NOT EXISTS won't
        # alter an existing table, so backfill the column for old DBs.
        cur.execute("PRAGMA table_info(entity_aliases)")
        cols = {row["name"] for row in cur.fetchall()}
        if "agent" not in cols:
            try:
                cur.execute("ALTER TABLE entity_aliases ADD COLUMN agent TEXT")
            except sqlite3.OperationalError:
                pass

        # Phase 11: durable extraction lifecycle. A queue row alone cannot
        # distinguish a successful zero-fact extraction from a failed request,
        # because successful jobs are deleted after processing.
        cur.execute("PRAGMA table_info(turns)")
        turn_cols = {row["name"] for row in cur.fetchall()}
        turn_additions = {
            "extraction_status": "TEXT NOT NULL DEFAULT 'pending'",
            "extraction_attempts": "INTEGER NOT NULL DEFAULT 0",
            "extraction_fact_count": "INTEGER",
            "extraction_completed_at": "REAL",
            "extraction_error": "TEXT",
        }
        for name, declaration in turn_additions.items():
            if name not in turn_cols:
                try:
                    cur.execute(f"ALTER TABLE turns ADD COLUMN {name} {declaration}")
                except sqlite3.OperationalError:
                    pass

        cur.execute("PRAGMA table_info(extraction_queue)")
        queue_cols = {row["name"] for row in cur.fetchall()}
        queue_additions = {
            "next_attempt_at": "REAL NOT NULL DEFAULT 0",
            "started_at": "REAL",
            "last_error": "TEXT",
        }
        for name, declaration in queue_additions.items():
            if name not in queue_cols:
                try:
                    cur.execute(f"ALTER TABLE extraction_queue ADD COLUMN {name} {declaration}")
                except sqlite3.OperationalError:
                    pass
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_queue_ready "
            "ON extraction_queue(status, next_attempt_at)"
        )
        # Phase 12: additive claim metadata.  Existing claim status values are
        # intentionally retained for backwards-compatible CHECK constraints;
        # richer lifecycle state lives in resolution_status/conflict_type.
        cur.execute("PRAGMA table_info(claims)")
        claim_cols = {row["name"] for row in cur.fetchall()}
        claim_additions = {
            "observed_at": "TEXT",
            "event_at": "TEXT",
            "scope_type": "TEXT",
            "scope_value": "TEXT",
            "modality": "TEXT DEFAULT 'asserted'",
            "conflict_type": "TEXT",
            "resolution_status": "TEXT DEFAULT 'active'",
            "extractor_version": "TEXT",
            "source_turn_id": "INTEGER",
        }
        for name, declaration in claim_additions.items():
            if name not in claim_cols:
                try:
                    cur.execute(f"ALTER TABLE claims ADD COLUMN {name} {declaration}")
                except sqlite3.OperationalError:
                    pass
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_claims_resolution "
            "ON claims(subject, predicate, resolution_status, valid_from, valid_to)"
        )
        # Phase 6 (migration): content_hash + seen_count on memories, and
        # widen the source CHECK to allow 'import' and 'hindsight'. CREATE
        # TABLE IF NOT EXISTS never alters an existing table, so backfill the
        # columns for old DBs. The CHECK constraint cannot be altered in place;
        # for pre-existing DBs we rely on the fact that sqlite only enforces
        # CHECK on INSERT/UPDATE of the column, and old rows already pass. A
        # fresh DB picks up the new CHECK from _SCHEMA above.
        cur.execute("PRAGMA table_info(memories)")
        cols = {row["name"] for row in cur.fetchall()}
        if "content_hash" not in cols:
            try:
                cur.execute("ALTER TABLE memories ADD COLUMN content_hash TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_mem_content_hash "
                    "ON memories(content_hash)"
                )
            except sqlite3.OperationalError:
                pass
        if "seen_count" not in cols:
            try:
                cur.execute(
                    "ALTER TABLE memories ADD COLUMN seen_count INTEGER DEFAULT 1"
                )
            except sqlite3.OperationalError:
                pass

        # Phase 7 (migration): widen the memories.source CHECK constraint to
        # allow 'dream'. SQLite does not support ALTER TABLE on CHECK, so we
        # rebuild the table when the current schema lacks 'dream'.
        cur.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='memories'"
        )
        create_sql = cur.fetchone()[0]
        if "'dream'" not in create_sql:
            self._conn.execute("PRAGMA foreign_keys=OFF")
            try:
                cur.execute("BEGIN IMMEDIATE")
                try:
                    self._rebuild_memories_table(cur)
                    cur.execute("COMMIT")
                except Exception:
                    cur.execute("ROLLBACK")
                    raise
            finally:
                self._conn.execute("PRAGMA foreign_keys=ON")
            # Recreate FTS5 triggers against the rebuilt table and rebuild the
            # FTS index because rowids may have shifted during the table swap.
            cur.executescript(_FTS_TRIGGERS)
            try:
                cur.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            except sqlite3.OperationalError:
                pass

    def _rebuild_memories_table(self, cur: sqlite3.Cursor) -> None:
        """Recreate the memories table with the widened source CHECK."""
        cur.execute(
            """
            CREATE TABLE memories_new (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL CHECK(
                    type IN ('fact','observation','conversation','document','thread')
                ),
                content TEXT NOT NULL,
                source TEXT NOT NULL CHECK(source IN (
                    'conversation','vault','email','cron','sensor',
                    'manual','import','hindsight','dream'
                )),
                source_id TEXT,
                agent TEXT,
                visibility TEXT DEFAULT 'private',
                timestamp TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                trust_score REAL DEFAULT 0.5,
                verified INTEGER DEFAULT 0,
                superseded_by TEXT,
                status TEXT DEFAULT 'active',
                tags TEXT,
                metadata TEXT,
                content_hash TEXT,
                seen_count INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cur.execute("INSERT INTO memories_new SELECT * FROM memories")
        cur.execute("DROP TABLE memories")
        cur.execute("ALTER TABLE memories_new RENAME TO memories")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_agent ON memories(agent)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_visibility ON memories(visibility)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_status ON memories(status)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_source ON memories(source)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_content_hash ON memories(content_hash)")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        """Context manager yielding a cursor inside BEGIN/COMMIT (write txn)."""
        with self.read() as cur:
            cur.execute("BEGIN IMMEDIATE;")
            try:
                yield cur
                cur.execute("COMMIT;")
            except Exception:
                self._conn.set_progress_handler(None, 0)
                if self._conn.in_transaction:
                    cur.execute("ROLLBACK;")
                raise

    @contextmanager
    def read(self) -> Iterator[sqlite3.Cursor]:
        """Read-only cursor serialized on this connection.

        WAL allows concurrent readers from different connections, but this
        provider intentionally holds one connection shared by extraction,
        prefetch, and tool calls. Serialize cursor use to avoid interleaved
        statements and transaction state corruption.
        """
        deadline = self._deadline.get()
        remaining = max(0.0, deadline - time.monotonic()) if deadline is not None else None
        acquired = (self._lock.acquire() if remaining is None
                    else self._lock.acquire(timeout=remaining))
        if not acquired:
            raise TimeoutError("memory database deadline exceeded")
        try:
            if deadline is not None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("memory database deadline exceeded")
                self._conn.execute(
                    f"PRAGMA busy_timeout={max(0, int((deadline - time.monotonic()) * 1000))}"
                )
                self._conn.set_progress_handler(lambda: time.monotonic() >= deadline, 1000)
            cur = self._conn.cursor()
            try:
                yield cur
            finally:
                cur.close()
        finally:
            if deadline is not None:
                self._conn.set_progress_handler(None, 0)
                self._conn.execute("PRAGMA busy_timeout=5000")
            self._lock.release()

    @contextmanager
    def deadline(self, until: float) -> Iterator[None]:
        previous = self._deadline.get()
        token = self._deadline.set(min(until, previous) if previous is not None else until)
        try:
            yield
        finally:
            self._deadline.reset(token)

    @property
    def memory_generation(self) -> int:
        with self.read() as cur:
            return int(cur.execute(
                "SELECT value FROM schema_meta WHERE key='memory_generation'"
            ).fetchone()[0])

    def iter_embeddings(
        self, *, agent_id: str | None, profile_scope: list[str] | None,
        model: str | None, dimensions: int, visibility: str | None = None,
        limit: int = 0,
    ) -> Iterator[dict[str, Any]]:
        """Stream committed vectors on an independent read-only WAL connection.

        Scoring never holds the shared writer lock, and no recency truncation
        applies by default. A caller's deadline interrupts SQLite and scoring.
        """
        sql = (
            "SELECT m.id, m.content, m.visibility, m.agent AS agent_id, "
            "m.timestamp AS created_at, m.updated_at, e.embedding "
            "FROM memories m JOIN embeddings e ON e.memory_id=m.id "
            "WHERE m.status='active' AND e.dimensions=?"
        )
        params: list[Any] = [dimensions]
        if model is not None:
            base = model.removesuffix(":latest")
            alias = base + ":latest" if ":" not in base.rsplit("/", 1)[-1] else base
            sql += " AND e.model IN (?,?)"
            params.extend((base, alias))
        if agent_id is not None:
            sql += " AND m.agent=?"
            params.append(agent_id)
        if visibility in VISIBILITY_ORDER:
            allowed = [key for key, value in VISIBILITY_ORDER.items()
                       if value <= VISIBILITY_ORDER[visibility]]
            sql += " AND m.visibility IN (" + ",".join("?" for _ in allowed) + ")"
            params.extend(allowed)
        sql, params = _append_profile_scope_sql(sql, params, profile_scope)
        if limit > 0:
            sql += " LIMIT ?"
            params.append(limit)
        deadline = self._deadline.get()
        connection = sqlite3.connect(Path(self.path).resolve().as_uri() + "?mode=ro", uri=True,
                                     timeout=0.0)
        connection.row_factory = sqlite3.Row
        if deadline is not None:
            connection.set_progress_handler(lambda: time.monotonic() >= deadline, 1000)
        try:
            cursor = connection.execute(sql, params)
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("semantic scan deadline exceeded")
                rows = cursor.fetchmany(64)
                if not rows:
                    break
                for row in rows:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError("semantic scan deadline exceeded")
                    yield dict(row)
        finally:
            connection.close()

    # -- prefetch stats --------------------------------------------------------

    def record_prefetch(
        self,
        session_id: str,
        outcome: str,
        reason: str | None = None,
        elapsed_ms: float = 0.0,
        result_count: int = 0,
        token_estimate: int = 0,
        query: str = "",
        agent_id: str | None = None,
    ) -> None:
        """Record a prefetch() call outcome for diagnostics.

        Called on every prefetch return path (injected or empty) so we can
        measure the empty-return rate, latency distribution, and which
        rejection path fires most often.
        """
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat()
        # Truncate query for storage safety.
        q = (query or "")[:500]
        with self._diagnostics_lock:
            self._diagnostics.append((
                "prefetch", (session_id, outcome, reason, elapsed_ms, result_count,
                             token_estimate, q, agent_id, ts),
            ))

    def record_operation(
        self, operation: str, outcome: str, *, elapsed_ms: float = 0.0,
        input_units: int = 0, output_units: int = 0, agent_id: str | None = None,
    ) -> None:
        """Buffer bounded telemetry; foreground callers never acquire a DB write lock."""
        with self._diagnostics_lock:
            self._diagnostics.append((
                "operation", (str(operation)[:64], str(outcome)[:32],
                              max(0.0, float(elapsed_ms)), max(0, int(input_units)),
                              max(0, int(output_units)), agent_id, _now_iso()),
            ))

    def flush_diagnostics(self) -> None:
        """Best-effort background batch. Retain bounded records if a writer is busy."""
        if not self._lock.acquire(blocking=False):
            return
        batch: list[tuple[str, tuple[Any, ...]]] = []
        try:
            if self._conn.in_transaction:
                return
            with self._diagnostics_lock:
                for _ in range(min(256, len(self._diagnostics))):
                    batch.append(self._diagnostics.popleft())
            if not batch:
                return
            self._conn.execute("PRAGMA busy_timeout=0")
            self._conn.execute("BEGIN IMMEDIATE")
            self._conn.executemany(
                "INSERT INTO prefetch_stats(session_id,outcome,reason,elapsed_ms,result_count,"
                "token_estimate,query,agent_id,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                [row for kind, row in batch if kind == "prefetch"],
            )
            self._conn.executemany(
                "INSERT INTO operation_metrics(operation,outcome,elapsed_ms,input_units,"
                "output_units,agent_id,created_at) VALUES(?,?,?,?,?,?,?)",
                [row for kind, row in batch if kind == "operation"],
            )
            self._conn.commit()
        except sqlite3.Error:
            if self._conn.in_transaction:
                self._conn.rollback()
            with self._diagnostics_lock:
                self._diagnostics.extendleft(reversed(batch))
        finally:
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._lock.release()

    def compact_caches(self, *, max_entries: int = 10000, max_age_days: int = 30) -> None:
        """Expire reproducible caches and telemetry, never memories or raw evidence."""
        self.flush_diagnostics()
        with self.transaction() as cur:
            cur.execute("DELETE FROM embedding_cache WHERE created_at < ?",
                        (time.time() - max(0, max_age_days) * 86400,))
            cur.execute(
                "DELETE FROM embedding_cache WHERE rowid IN (SELECT rowid FROM embedding_cache "
                "ORDER BY created_at DESC LIMIT -1 OFFSET ?)", (max(0, max_entries),),
            )
            for table in ("operation_metrics", "prefetch_stats"):
                cur.execute(f"DELETE FROM {table} WHERE id <= "
                            f"COALESCE((SELECT MAX(id)-10000 FROM {table}),0)")

    # -- turns -----------------------------------------------------------------

    def insert_turn(
        self,
        *,
        session_id: str,
        agent_id: str,
        user_text: str,
        assistant_text: str,
    ) -> int:
        now = time.time()
        with self.transaction() as cur:
            cur.execute(
                "INSERT INTO turns(session_id, agent_id, user_text, assistant_text, created_at) "
                "VALUES(?,?,?,?,?)",
                (session_id, agent_id, user_text, assistant_text, now),
            )
            return int(cur.lastrowid)

    def insert_turn_with_extraction(
        self,
        *,
        session_id: str,
        agent_id: str,
        user_text: str,
        assistant_text: str,
    ) -> int:
        """Persist a turn and its extraction job in one transaction.

        A restart must never leave a durable turn without the queue row that
        causes it to be processed.  Keeping both writes together also avoids
        requiring recovery scans during normal operation.
        """
        now = time.time()
        with self.transaction() as cur:
            cur.execute(
                "INSERT INTO turns(session_id, agent_id, user_text, assistant_text, created_at) "
                "VALUES(?,?,?,?,?)",
                (session_id, agent_id, user_text, assistant_text, now),
            )
            turn_id = int(cur.lastrowid)
            cur.execute(
                "INSERT INTO extraction_queue(turn_id, session_id, agent_id, user_text, "
                "assistant_text, enqueued_at, attempts, status) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (turn_id, session_id, agent_id, user_text, assistant_text, now, 0, "pending"),
            )
            return turn_id

    # -- extraction queue ------------------------------------------------------

    def enqueue_extraction(
        self,
        *,
        turn_id: int,
        session_id: str,
        agent_id: str,
        user_text: str,
        assistant_text: str,
    ) -> int:
        now = time.time()
        with self.transaction() as cur:
            cur.execute(
                "INSERT INTO extraction_queue(turn_id, session_id, agent_id, user_text, "
                "assistant_text, enqueued_at, attempts, status) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (turn_id, session_id, agent_id, user_text, assistant_text, now, 0, "pending"),
            )
            return int(cur.lastrowid)

    def claim_next_extraction(self, agent_id: str | None = None) -> dict[str, Any] | None:
        """Atomically claim the next pending extraction job.

        LIFO ordering (``ORDER BY id DESC``) so the most recently enqueued turn
        is processed first — issue #13.
        """
        with self.transaction() as cur:
            if agent_id is None:
                cur.execute(
                    "SELECT * FROM extraction_queue WHERE status='pending' "
                    "AND next_attempt_at <= ? ORDER BY id DESC LIMIT 1",
                    (time.time(),),
                )
            else:
                cur.execute(
                    "SELECT * FROM extraction_queue WHERE status='pending' AND agent_id=? "
                    "AND next_attempt_at <= ? ORDER BY id DESC LIMIT 1",
                    (agent_id, time.time()),
                )
            row = cur.fetchone()
            if row is None:
                return None
            qid = int(row["id"])
            cur.execute(
                "UPDATE extraction_queue SET attempts=attempts+1, status='running', "
                "started_at=? WHERE id=?",
                (time.time(), qid),
            )
            cur.execute(
                "UPDATE turns SET extraction_status='running', "
                "extraction_attempts=extraction_attempts+1, extraction_error=NULL "
                "WHERE id=?",
                (int(row["turn_id"]),),
            )
            return dict(row)

    def complete_extraction(self, queue_id: int, *, fact_count: int = 0) -> None:
        with self.transaction() as cur:
            cur.execute("SELECT turn_id FROM extraction_queue WHERE id=?", (queue_id,))
            row = cur.fetchone()
            if row is None:
                return
            cur.execute(
                "UPDATE turns SET extraction_status='completed', extraction_fact_count=?, "
                "extraction_completed_at=?, extraction_error=NULL WHERE id=?",
                (max(0, int(fact_count)), time.time(), int(row["turn_id"])),
            )
            cur.execute("DELETE FROM extraction_queue WHERE id=?", (queue_id,))

    def fail_extraction(self, queue_id: int, *, error: str = "") -> None:
        """Retry a failed job with backoff, then retain a dead-letter marker."""
        with self.transaction() as cur:
            cur.execute("SELECT turn_id, attempts FROM extraction_queue WHERE id=?", (queue_id,))
            row = cur.fetchone()
            if row is None:
                return
            attempts = int(row["attempts"] or 0)
            bounded_error = str(error or "extraction failed")[:500]
            if attempts < 3:
                delay = min(300.0, 2.0 ** max(0, attempts - 1))
                cur.execute(
                    "UPDATE extraction_queue SET status='pending', next_attempt_at=?, "
                    "started_at=NULL, last_error=? WHERE id=?",
                    (time.time() + delay, bounded_error, queue_id),
                )
                cur.execute(
                    "UPDATE turns SET extraction_status='retry_wait', "
                    "extraction_error=? WHERE id=?",
                    (bounded_error, int(row["turn_id"])),
                )
                return
            cur.execute(
                "UPDATE turns SET extraction_status='dead_letter', extraction_error=? WHERE id=?",
                (bounded_error, int(row["turn_id"])),
            )
            cur.execute("DELETE FROM extraction_queue WHERE id=? AND attempts >= 3", (queue_id,))

    def pending_count(self) -> int:
        with self.read() as cur:
            cur.execute("SELECT COUNT(*) AS c FROM extraction_queue WHERE status='pending'")
            return int(cur.fetchone()["c"])

    def recover_stale_extractions(self, *, max_age_s: float = 900.0) -> int:
        """Return abandoned running jobs to the retryable queue."""
        cutoff = time.time() - max(1.0, float(max_age_s))
        with self.transaction() as cur:
            cur.execute(
                "SELECT id, turn_id FROM extraction_queue "
                "WHERE status='running' AND COALESCE(started_at, enqueued_at) < ?",
                (cutoff,),
            )
            rows = cur.fetchall()
            for row in rows:
                cur.execute(
                    "UPDATE extraction_queue SET status='pending', next_attempt_at=0, "
                    "started_at=NULL, last_error=? WHERE id=?",
                    ("recovered stale running job", int(row["id"])),
                )
                cur.execute(
                    "UPDATE turns SET extraction_status='retry_wait', "
                    "extraction_error=? WHERE id=?",
                    ("recovered stale running job", int(row["turn_id"])),
                )
            return len(rows)

    def get_unextracted_turns(
        self, agent_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Return turns that have neither an extraction_queue row nor a
        ``source='conversation'`` memory with ``source_id = str(turn_id)``.

        Used by the extraction worker startup sweep so turns that were stored
        but never extracted (e.g. crash between ``insert_turn`` and
        ``enqueue_extraction``) are recovered on restart.
        """
        sql = (
            "SELECT t.id, t.session_id, t.agent_id, t.user_text, t.assistant_text, "
            "t.created_at "
            "FROM turns t "
            "LEFT JOIN extraction_queue q ON q.turn_id = t.id "
            "LEFT JOIN memories m ON m.source_id = CAST(t.id AS TEXT) "
            "AND m.source = 'conversation' "
            "WHERE q.id IS NULL AND m.id IS NULL "
            "AND COALESCE(t.extraction_status, 'pending') IN ('pending', 'retry_wait')"
        )
        params: list[Any] = []
        if agent_id is not None:
            sql += " AND t.agent_id=?"
            params.append(agent_id)
        sql += " ORDER BY t.id DESC LIMIT ?"
        params.append(limit)
        with self.read() as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    # -- memories --------------------------------------------------------------

    def insert_memory(
        self,
        *,
        content: str,
        source: str = "manual",
        agent: str | None = None,
        visibility: str = "private",
        source_id: str | None = None,
        type: str = "fact",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        confidence: float = 0.5,
        trust_score: float = 0.5,
        content_hash: str | None = None,
        embedding: list[float] | None = None,
        embed_model: str | None = None,
    ) -> str:
        now = _now_iso()
        mid = _uuid()
        tags_json = json.dumps(tags) if tags else None
        meta_json = json.dumps(metadata, default=str) if metadata else None
        with self.transaction() as cur:
            cur.execute(
                "INSERT INTO memories(id, type, content, source, source_id, agent, "
                "visibility, timestamp, confidence, trust_score, verified, superseded_by, "
                "status, tags, metadata, content_hash, seen_count, "
                "created_at, updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,0,NULL,?,?,?,?,1,?,?)",
                (
                    mid, type, content, source, source_id, agent, visibility,
                    now, confidence, trust_score, "active", tags_json, meta_json,
                    content_hash, now, now,
                ),
            )
            if embedding:
                blob = _pack_embedding(embedding)
                cur.execute(
                    "INSERT OR REPLACE INTO embeddings(memory_id, model, embedding, "
                    "dimensions, created_at) VALUES(?,?,?,?,?)",
                    (mid, embed_model, blob, len(embedding), now),
                )
            return mid

    def replace_memories_atomic(
        self,
        *,
        original_ids: list[str],
        content: str,
        source: str,
        agent: str | None,
        visibility: str,
        type: str,
        tags: list[str] | None,
        metadata: dict[str, Any] | None,
        confidence: float,
        trust_score: float,
        embedding: list[float] | None,
        embed_model: str | None,
        claim_projection: dict[str, Any] | None,
        actor: str,
        action: str,
    ) -> dict[str, Any]:
        """Create a replacement and supersede originals in one transaction."""
        if not original_ids:
            raise ValueError("at least one original memory is required")
        now = _now_iso()
        new_id = _uuid()
        tags_json = json.dumps(tags) if tags else None
        metadata_json = json.dumps(metadata, default=str) if metadata else None
        with self.transaction() as cur:
            placeholders = ",".join("?" for _ in original_ids)
            cur.execute(
                f"SELECT id, content, agent, visibility, status, trust_score, tags "
                f"FROM memories WHERE id IN ({placeholders})",
                original_ids,
            )
            originals = [dict(row) for row in cur.fetchall()]
            if {row["id"] for row in originals} != set(original_ids):
                raise KeyError("one or more original memories do not exist")
            if any(row["status"] != "active" for row in originals):
                raise ValueError("all original memories must be active")
            cur.execute(
                "INSERT INTO memories(id, type, content, source, source_id, agent, "
                "visibility, timestamp, confidence, trust_score, verified, superseded_by, "
                "status, tags, metadata, content_hash, seen_count, created_at, updated_at) "
                "VALUES(?,?,?,?,NULL,?,?,?,?,?,0,NULL,'active',?,?,NULL,1,?,?)",
                (
                    new_id, type, content, source, agent, visibility, now, confidence,
                    trust_score, tags_json, metadata_json, now, now,
                ),
            )
            if embedding:
                cur.execute(
                    "INSERT INTO embeddings(memory_id, model, embedding, dimensions, created_at) "
                    "VALUES(?,?,?,?,?)",
                    (new_id, embed_model, _pack_embedding(embedding), len(embedding), now),
                )
            new_claim_id: str | None = None
            if claim_projection:
                new_claim_id = _uuid()
                cur.execute(
                    "INSERT INTO claims(id, memory_id, subject, predicate, object, qualifiers, "
                    "confidence, status, valid_from, valid_to, observed_at, event_at, "
                    "scope_type, scope_value, modality, conflict_type, resolution_status, "
                    "extractor_version, source_turn_id, created_at, updated_at) "
                    "VALUES(?,?,?,?,?,?,?,'active',?,NULL,?,?,?,?,?,?,?,'active',?,?,?)",
                    (
                        new_claim_id,
                        new_id,
                        claim_projection["subject"],
                        claim_projection["predicate"],
                        claim_projection["object"],
                        json.dumps(claim_projection.get("qualifiers"), default=str)
                        if claim_projection.get("qualifiers")
                        else None,
                        float(claim_projection.get("confidence") or confidence),
                        claim_projection.get("valid_from") or now,
                        claim_projection.get("observed_at") or now,
                        claim_projection.get("event_at"),
                        claim_projection.get("scope_type"),
                        claim_projection.get("scope_value"),
                        claim_projection.get("modality") or "asserted",
                        "update",
                        claim_projection.get("extractor_version") or "lifecycle-v1",
                        claim_projection.get("source_turn_id"),
                        now,
                        now,
                    ),
                )
            for original_id in original_ids:
                cur.execute(
                    "INSERT OR IGNORE INTO memory_entities("
                    "memory_id, entity_id, relation_role, agent) "
                    "SELECT ?, entity_id, relation_role, agent FROM memory_entities "
                    "WHERE memory_id=?",
                    (new_id, original_id),
                )
                cur.execute(
                    "INSERT OR IGNORE INTO relation_evidence(entity_a, entity_b, "
                    "relation_type, memory_id, claim_id, strength, active, created_at, "
                    "updated_at) SELECT entity_a, entity_b, relation_type, ?, ?, strength, "
                    "1, ?, ? FROM relation_evidence WHERE memory_id=? AND active=1",
                    (new_id, new_claim_id, now, now, original_id),
                )
                cur.execute(
                    "UPDATE memories SET status='superseded', superseded_by=?, updated_at=? "
                    "WHERE id=? AND status='active'",
                    (new_id, now, original_id),
                )
                cur.execute(
                    "UPDATE claims SET status='superseded', resolution_status='superseded', "
                    "valid_to=COALESCE(valid_to, ?), updated_at=? WHERE memory_id=? "
                    "AND status='active'",
                    (now, now, original_id),
                )
                cur.execute(
                    "UPDATE relation_evidence SET active=0, updated_at=? WHERE memory_id=?",
                    (now, original_id),
                )
                self._write_audit(
                    cur,
                    actor,
                    "supersede",
                    original_id,
                    {"superseded_by": new_id},
                )
            snapshots = [
                {
                    "id": row["id"],
                    "content": row["content"],
                    "visibility": row["visibility"],
                    "status": row["status"],
                    "trust_score": row["trust_score"],
                    "tags": row["tags"],
                }
                for row in originals
            ]
            audit_id = self._write_audit(
                cur,
                actor,
                action,
                new_id,
                {
                    "original_ids": original_ids,
                    "replacement_id": new_id,
                    "before_id": original_ids[0] if len(original_ids) == 1 else None,
                    "before": snapshots[0] if len(snapshots) == 1 else snapshots,
                    "after_id": new_id,
                    "merged_from": original_ids if len(original_ids) > 1 else None,
                },
            )
        return {"memory_id": new_id, "claim_id": new_claim_id, "audit_id": audit_id}

    def transition_memory_atomic(
        self,
        memory_id: str,
        *,
        status: str | None = None,
        visibility: str | None = None,
        actor: str,
        action: str,
    ) -> int:
        """Atomically transition memory, claims, relation evidence, and audit."""
        if status is None and visibility is None:
            raise ValueError("status or visibility is required")
        now = _now_iso()
        with self.transaction() as cur:
            cur.execute("SELECT * FROM memories WHERE id=?", (memory_id,))
            before = cur.fetchone()
            if before is None:
                raise KeyError(memory_id)
            if status is not None:
                cur.execute(
                    "UPDATE memories SET status=?, updated_at=? WHERE id=?",
                    (status, now, memory_id),
                )
                if status != "active":
                    cur.execute(
                        "UPDATE claims SET resolution_status='historical', "
                        "valid_to=COALESCE(valid_to, ?), updated_at=? WHERE memory_id=?",
                        (now, now, memory_id),
                    )
                    cur.execute(
                        "UPDATE relation_evidence SET active=0, updated_at=? WHERE memory_id=?",
                        (now, memory_id),
                    )
            if visibility is not None:
                cur.execute(
                    "UPDATE memories SET visibility=?, updated_at=? WHERE id=?",
                    (visibility, now, memory_id),
                )
            before_detail: Any = (
                before["visibility"]
                if visibility is not None and status is None
                else {
                    "id": memory_id,
                    "content": before["content"],
                    "visibility": before["visibility"],
                    "status": before["status"],
                    "trust_score": before["trust_score"],
                }
            )
            return self._write_audit(
                cur,
                actor,
                action,
                memory_id,
                {
                    "before": before_detail,
                    "before_status": before["status"],
                    "after_status": status or before["status"],
                    "before_visibility": before["visibility"],
                    "after_visibility": visibility or before["visibility"],
                    "after": visibility or before["visibility"],
                },
            )

    def deactivate_memory(self, memory_id: str) -> None:
        with self.transaction() as cur:
            cur.execute(
                "UPDATE memories SET status='inactive', updated_at=? WHERE id=?",
                (_now_iso(), memory_id),
            )

    def find_active_memory_by_content(
        self, content: str, *, agent_id: str
    ) -> dict[str, Any] | None:
        """Find one exact active memory for a mirrored built-in write."""
        with self.read() as cur:
            cur.execute(
                "SELECT * FROM memories WHERE content=? AND agent=? AND status='active' "
                "ORDER BY updated_at DESC LIMIT 1",
                (content, agent_id),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    # -- structured claims ----------------------------------------------------

    def create_claim(
        self,
        *,
        memory_id: str,
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 0.5,
        qualifiers: dict[str, Any] | None = None,
        status: str = "active",
        observed_at: str | None = None,
        event_at: str | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
        scope_type: str | None = None,
        scope_value: str | None = None,
        modality: str = "asserted",
        conflict_type: str | None = None,
        resolution_status: str | None = None,
        extractor_version: str | None = None,
        source_turn_id: int | None = None,
    ) -> str:
        """Create a versioned claim projection backed by ``memory_id``."""
        now = _now_iso()
        claim_id = _uuid()
        with self.transaction() as cur:
            cur.execute(
                "INSERT INTO claims(id, memory_id, subject, predicate, object, qualifiers, "
                "confidence, status, valid_from, valid_to, observed_at, event_at, "
                "scope_type, scope_value, modality, conflict_type, resolution_status, "
                "extractor_version, source_turn_id, created_at, updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    claim_id, memory_id, subject, predicate, object,
                    json.dumps(qualifiers, default=str) if qualifiers else None,
                    confidence, status, valid_from, valid_to, observed_at or now,
                    event_at, scope_type, scope_value, modality, conflict_type,
                    resolution_status or status, extractor_version, source_turn_id,
                    now, now,
                ),
            )
        return claim_id

    def replace_claim_projection(
        self,
        *,
        memory_id: str,
        subject: str,
        predicate: str,
        object: str,
        confidence: float,
        qualifiers: dict[str, Any] | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
        observed_at: str | None = None,
        event_at: str | None = None,
        scope_type: str | None = None,
        scope_value: str | None = None,
        modality: str = "asserted",
        extractor_version: str,
        source_turn_id: int | None = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        """Replace one memory's claim projection without changing its memory.

        ``claims.memory_id`` is unique, so a model-backed migration must update
        the existing projection in place.  The before/after rows are retained
        in the audit log; the immutable backing memory remains untouched.
        """
        now = _now_iso()
        with self.transaction() as cur:
            cur.execute("SELECT * FROM claims WHERE memory_id=?", (memory_id,))
            before_row = cur.fetchone()
            before = dict(before_row) if before_row else None
            if before is None:
                claim_id = _uuid()
                cur.execute(
                    "INSERT INTO claims(id, memory_id, subject, predicate, object, qualifiers, "
                    "confidence, status, valid_from, valid_to, observed_at, event_at, "
                    "scope_type, scope_value, modality, conflict_type, resolution_status, "
                    "extractor_version, source_turn_id, created_at, updated_at) "
                    "VALUES(?,?,?,?,?,?,?,'active',?,?,?,?,?,?,?,NULL,'active',?,?,?,?)",
                    (
                        claim_id,
                        memory_id,
                        subject,
                        predicate,
                        object,
                        json.dumps(qualifiers, default=str) if qualifiers else None,
                        float(confidence),
                        valid_from,
                        valid_to,
                        observed_at or now,
                        event_at,
                        scope_type,
                        scope_value,
                        modality,
                        extractor_version,
                        source_turn_id,
                        now,
                        now,
                    ),
                )
                operation = "created"
            else:
                claim_id = str(before["id"])
                cur.execute(
                    "UPDATE claims SET subject=?, predicate=?, object=?, qualifiers=?, "
                    "confidence=?, valid_from=?, valid_to=?, observed_at=?, event_at=?, "
                    "scope_type=?, scope_value=?, modality=?, extractor_version=?, "
                    "source_turn_id=COALESCE(?, source_turn_id), updated_at=? "
                    "WHERE memory_id=?",
                    (
                        subject,
                        predicate,
                        object,
                        json.dumps(qualifiers, default=str) if qualifiers else None,
                        float(confidence),
                        valid_from if valid_from is not None else before.get("valid_from"),
                        valid_to if valid_to is not None else before.get("valid_to"),
                        observed_at if observed_at is not None else before.get("observed_at"),
                        event_at if event_at is not None else before.get("event_at"),
                        scope_type if scope_type is not None else before.get("scope_type"),
                        scope_value if scope_value is not None else before.get("scope_value"),
                        modality or before.get("modality") or "asserted",
                        extractor_version,
                        source_turn_id,
                        now,
                        memory_id,
                    ),
                )
                operation = "updated"
            cur.execute("SELECT * FROM claims WHERE memory_id=?", (memory_id,))
            after = dict(cur.fetchone())
            audit_id = self._write_audit(
                cur,
                actor,
                "claim_model_backfill",
                memory_id,
                {
                    "operation": operation,
                    "claim_id": claim_id,
                    "before": before,
                    "after": after,
                },
            )
        return {
            "claim_id": claim_id,
            "updated": operation == "updated",
            "created": operation == "created",
            "audit_id": audit_id,
        }

    def supersede_claims(
        self,
        *,
        subject: str,
        predicate: str,
        except_memory_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[str]:
        """Mark active conflicting versions as superseded and retain their history."""
        with self.transaction() as cur:
            sql = (
                "SELECT c.id FROM claims c JOIN memories m ON m.id=c.memory_id "
                "WHERE c.subject=? COLLATE NOCASE "
                "AND c.predicate=? COLLATE NOCASE AND c.status='active'"
            )
            params: list[Any] = [subject, predicate]
            if except_memory_id:
                sql += " AND c.memory_id<>?"
                params.append(except_memory_id)
            if agent_id is not None:
                sql += " AND m.agent=?"
                params.append(agent_id)
            cur.execute(sql, params)
            ids = [str(row["id"]) for row in cur.fetchall()]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                cur.execute(
                    f"UPDATE claims SET status='superseded', valid_to=?, updated_at=? "
                    f"WHERE id IN ({placeholders})",
                    [_now_iso(), _now_iso(), *ids],
                )
            return ids

    def get_claim_for_memory(self, memory_id: str) -> dict[str, Any] | None:
        with self.read() as cur:
            cur.execute("SELECT * FROM claims WHERE memory_id=?", (memory_id,))
            row = cur.fetchone()
        if row is None:
            return None
        claim = dict(row)
        if claim.get("qualifiers"):
            try:
                claim["qualifiers"] = json.loads(claim["qualifiers"])
            except (json.JSONDecodeError, TypeError):
                pass
        return claim

    def get_active_claim(
        self, subject: str, predicate: str, *, agent_id: str | None = None
    ) -> dict[str, Any] | None:
        """Return the newest active version of a subject/predicate claim."""
        with self.read() as cur:
            sql = (
                "SELECT c.* FROM claims c JOIN memories m ON m.id=c.memory_id "
                "WHERE c.subject=? COLLATE NOCASE AND c.predicate=? COLLATE NOCASE "
                "AND c.status='active'"
            )
            params: list[Any] = [subject, predicate]
            if agent_id is not None:
                sql += " AND m.agent=?"
                params.append(agent_id)
            sql += " ORDER BY c.updated_at DESC LIMIT 1"
            cur.execute(sql, params)
            row = cur.fetchone()
        return dict(row) if row else None

    def get_claims_for_memories(self, memory_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Return claim projections keyed by backing memory id in one query."""
        if not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        with self.read() as cur:
            cur.execute(
                f"SELECT * FROM claims WHERE memory_id IN ({placeholders})",
                memory_ids,
            )
            return {str(row["memory_id"]): dict(row) for row in cur.fetchall()}

    def hard_delete_memory(self, memory_id: str) -> bool:
        """Permanently delete a memory and its cascading rows.

        Unlike ``deactivate_memory``, this performs a real DELETE. Intended for
        orphan cleanup (issue #23) where the memory is a duplicate/garbage row
        with no links to vault files or other memories.
        """
        with self.transaction() as cur:
            cur.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            # The memories_ad trigger maintains the external-content FTS row.
            # Capture this rowcount before issuing any other statement.
            return cur.rowcount > 0

    def search_bm25(
        self,
        query: str,
        *,
        agent_id: str | None = None,
        visibility: str | None = None,
        profile_scope: list[str] | None = None,
        include_historical: bool = False,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """BM25 keyword search over active memories, optionally filtered.

        A scoped read includes only that owner, regardless of source or visibility.
        """
        fts_query = _to_fts_query(query)
        if not fts_query:
            return []
        sql = (
            "SELECT m.id, m.content, m.visibility, m.agent AS agent_id, "
            "m.timestamp AS created_at, m.updated_at, "
            "bm25(memories_fts) AS score "
            "FROM memories_fts JOIN memories m ON m.rowid = memories_fts.rowid "
            "WHERE memories_fts MATCH ? AND "
            + ("m.status IN ('active','superseded')" if include_historical else "m.status='active'")
        )
        params: list[Any] = [fts_query]
        if agent_id is not None:
            sql += " AND m.agent=?"
            params.append(agent_id)
        if visibility is not None:
            sql += " AND m.visibility=?"
            params.append(visibility)
        sql, params = _append_profile_scope_sql(sql, params, profile_scope)
        sql += " ORDER BY score ASC LIMIT ?"
        params.append(limit)
        with self.read() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["score"] = -float(d["score"])
            out.append(d)
        return out

    def candidate_facts(
        self, query: str, *, agent_id: str, limit: int = 8
    ) -> list[dict[str, Any]]:
        """Fetch fact text for conservative equivalence checks."""
        fts_query = _to_fts_query(query)
        if not fts_query:
            return []
        sql = (
            "SELECT m.id, m.content, m.visibility "
            "FROM memories_fts JOIN memories m ON m.rowid = memories_fts.rowid "
            "WHERE memories_fts MATCH ? AND m.status='active' AND m.agent=? AND m.type='fact'"
        )
        params: list[Any] = [fts_query, agent_id]
        sql += " ORDER BY bm25(memories_fts) ASC LIMIT ?"
        params.append(limit)
        with self.read() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [dict(row) for row in rows]

    def get_memory_embedding(self, memory_id: str) -> list[float]:
        with self.read() as cur:
            cur.execute("SELECT embedding FROM embeddings WHERE memory_id=?", (memory_id,))
            row = cur.fetchone()
        return _unpack_embedding(row["embedding"]) if row and row["embedding"] else []

    def repair_embedding_candidates(
        self, *, agent_id: str, model: str, dimensions: int,
        profile_scope: list[str] | None = None, memory_ids: list[str] | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        base = model.removesuffix(":latest")
        alias = base + ":latest" if ":" not in base.rsplit("/", 1)[-1] else base
        sql = (
            "SELECT m.id,m.content FROM memories m LEFT JOIN embeddings e ON e.memory_id=m.id "
            "WHERE m.status='active' AND m.agent=? "
            "AND (e.memory_id IS NULL OR e.model IS NULL OR e.model NOT IN (?,?) "
            "OR e.dimensions!=? OR length(e.embedding)!=?)"
        )
        params: list[Any] = [agent_id, base, alias, dimensions, dimensions * 4]
        sql, params = _append_profile_scope_sql(sql, params, profile_scope)
        if memory_ids is not None:
            if not memory_ids:
                return []
            sql += " AND m.id IN (" + ",".join("?" for _ in memory_ids) + ")"
            params.extend(memory_ids)
        sql += " LIMIT ?"
        params.append(max(1, limit))
        with self.read() as cur:
            return [dict(row) for row in cur.execute(sql, params).fetchall()]

    def put_memory_embedding(
        self, memory_id: str, *, expected_content: str, embedding: list[float], model: str,
    ) -> bool:
        """Do not attach a repaired vector if its source changed during the network call."""
        blob = _pack_embedding(embedding)
        with self.transaction() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO embeddings"
                "(memory_id,model,embedding,dimensions,created_at) "
                "SELECT id,?,?,?,? FROM memories WHERE id=? AND content=? AND status='active'",
                (model, blob, len(embedding), _now_iso(), memory_id, expected_content),
            )
            return cur.rowcount == 1

    def record_duplicate(
        self, memory_id: str, *, source_turn_id: int | None, session_id: str, agent_id: str,
    ) -> None:
        """Retain corroboration provenance without counting extraction retries twice."""
        with self.transaction() as cur:
            if source_turn_id is not None:
                if cur.execute(
                    "SELECT 1 FROM memories WHERE id=? AND source_id=? "
                    "UNION ALL SELECT 1 FROM audit_log WHERE memory_id=? "
                    "AND action='memory_duplicate' AND json_extract(details,'$.source_turn_id')=?",
                    (memory_id, str(source_turn_id), memory_id, source_turn_id),
                ).fetchone():
                    return
            cur.execute("UPDATE memories SET seen_count=seen_count+1 WHERE id=?", (memory_id,))
            self._write_audit(cur, agent_id, "memory_duplicate", memory_id,
                              {"source_turn_id": source_turn_id, "session_id": session_id})

    def search_by_embedding(
        self,
        memory_ids: list[str],
        *,
        agent_id: str | None = None,
        visibility: str | None = None,
        profile_scope: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return active memories in `memory_ids` with their embeddings attached.

        Used by semantic search to load embeddings only for the BM25-pre-filtered
        candidate set, never the whole table. `memory_ids` should already be
        bounded by the caller (e.g. SEMANTIC_CANDIDATE_LIMIT).

        A scoped read includes only that owner, including vault documents.
        """
        if not memory_ids:
            return []
        placeholders = ",".join("?" for _ in memory_ids)
        sql = (
            "SELECT m.id, m.content, m.visibility, m.agent AS agent_id, "
            "m.timestamp AS created_at, m.updated_at, e.embedding "
            "FROM memories m LEFT JOIN embeddings e ON e.memory_id = m.id "
            f"WHERE m.id IN ({placeholders}) AND m.status='active'"
        )
        params: list[Any] = list(memory_ids)
        if agent_id is not None:
            sql += " AND m.agent=?"
            params.append(agent_id)
        if visibility is not None:
            sql += " AND m.visibility=?"
            params.append(visibility)
        sql, params = _append_profile_scope_sql(sql, params, profile_scope)
        with self.read() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["embedding"] = _unpack_embedding(d["embedding"]) if d["embedding"] else []
            out.append(d)
        return out

    def search_all_embeddings(
        self,
        *,
        agent_id: str | None = None,
        visibility: str | None = None,
        profile_scope: list[str] | None = None,
        limit: int = 5_000,
    ) -> list[dict[str, Any]]:
        """Return a bounded, authorization-scoped embedding corpus.

        Semantic retrieval must not be gated by lexical recall: a memory with
        no shared words can still be the best answer. This exact scan is the
        correct small-corpus implementation and has an explicit ceiling for a
        future ANN replacement.
        """
        sql = (
            "SELECT m.id, m.content, m.visibility, m.agent AS agent_id, "
            "m.timestamp AS created_at, m.updated_at, e.embedding "
            "FROM memories m JOIN embeddings e ON e.memory_id = m.id "
            "WHERE m.status='active'"
        )
        params: list[Any] = []
        if agent_id is not None:
            sql += " AND m.agent=?"
            params.append(agent_id)
        if visibility is not None:
            sql += " AND m.visibility=?"
            params.append(visibility)
        sql, params = _append_profile_scope_sql(sql, params, profile_scope)
        sql += " ORDER BY m.updated_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        with self.read() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["embedding"] = _unpack_embedding(item["embedding"])
            out.append(item)
        return out

    def search_all_active(
        self,
        *,
        agent_id: str | None = None,
        visibility: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return active memory ids (optionally filtered) for candidate loading.

        Ordered by recency. Embeddings are NOT loaded here; callers pass the
        returned ids to `search_by_embedding` to fetch vectors only for the
        candidate set. This keeps the pre-filter cheap and avoids scanning the
        embeddings table blindly.

        A scoped read includes only that owner, including vault documents.
        """
        sql = (
            "SELECT m.id, m.content, m.visibility, m.agent AS agent_id, "
            "m.timestamp AS created_at "
            "FROM memories m WHERE m.status='active'"
        )
        params: list[Any] = []
        if agent_id is not None:
            sql += " AND m.agent=?"
            params.append(agent_id)
        if visibility is not None:
            sql += " AND m.visibility=?"
            params.append(visibility)
        sql += " ORDER BY m.created_at DESC LIMIT ?"
        params.append(limit)
        with self.read() as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def list_memories(
        self,
        *,
        agent_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        sql = (
            "SELECT id, content, visibility, agent AS agent_id, status AS active, "
            "timestamp AS created_at, updated_at "
            "FROM memories WHERE status='active'"
        )
        params: list[Any] = []
        if agent_id is not None:
            sql += " AND agent=?"
            params.append(agent_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self.read() as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    # -- entities --------------------------------------------------------------

    def resolve_entity(
        self,
        name: str,
        agent_id: str | None = None,
        *,
        entity_type: str | None = None,
        aliases: list[str] | None = None,
    ) -> str:
        """Return canonical entity id for `name` within this agent's scope.

        Resolution is fuzzy on name + aliases (case-insensitive, punctuation
        stripped). The first existing match wins; otherwise a new entity row
        is created. `agent_id` scopes the canonical form per agent so two
        agents can have distinct entities that happen to share a name. When
        `agent_id` is None the entity is global (legacy Phase 1 path).

        Returns the entity *id* (UUID). Use ``find_entity_by_name`` for a
        read-only lookup, and ``entity_name_for`` to map an id back to a name.
        """
        key = _normalize_entity_name(name)
        if not key:
            return ""
        normalized_aliases = [_normalize_entity_name(a) for a in (aliases or [])]
        normalized_aliases = [a for a in normalized_aliases if a]
        with self.transaction() as cur:
            # Look for an existing entity matching name or any alias, scoped to
            # the agent when one is given. Entities created without an agent
            # (Phase 1) remain global and are matched by name only.
            match_sql = (
                "SELECT id FROM entities WHERE LOWER(name) = ? "
                "AND agent IS ? "
                "ORDER BY agent IS NULL LIMIT 1"
            )
            cur.execute(match_sql, (key, agent_id))
            row = cur.fetchone()
            if row is not None:
                eid = row["id"]
                # Merge any newly supplied aliases / type into the row.
                self._merge_entity_meta(cur, eid, entity_type, normalized_aliases, agent_id)
                return eid
            # Try alias match across the agent scope. We JOIN instead of using
            # an EXISTS subquery so the alias row's `agent` column stays in
            # scope for the ORDER BY (a correlated subquery alias is not visible
            # in the outer ORDER BY, which previously raised OperationalError).
            if normalized_aliases:
                for alias in normalized_aliases:
                    cur.execute(
                        "SELECT e.id FROM entities e JOIN entity_aliases ea "
                        "ON ea.entity_id = e.id "
                        "WHERE ea.alias = ? AND ea.agent IS ? "
                        "ORDER BY ea.agent IS NULL LIMIT 1",
                        (alias, agent_id),
                    )
                    row = cur.fetchone()
                    if row is not None:
                        eid = row["id"]
                        self._merge_entity_meta(cur, eid, entity_type, normalized_aliases, agent_id)
                        return eid
            # No match: create a new entity.
            eid = _uuid()
            cur.execute(
                "INSERT OR IGNORE INTO entities(id, name, type, aliases, agent, created_at) "
                "VALUES(?,?,?,?,?,?)",
                (eid, key, entity_type, json.dumps(normalized_aliases), agent_id, _now_iso()),
            )
            for alias in normalized_aliases:
                cur.execute(
                    "INSERT OR IGNORE INTO entity_aliases(entity_id, alias, agent) VALUES(?,?,?)",
                    (eid, alias, agent_id),
                )
            return eid

    def _merge_entity_meta(
        self,
        cur: sqlite3.Cursor,
        entity_id: str,
        entity_type: str | None,
        aliases: list[str],
        agent_id: str | None = None,
    ) -> None:
        """Update type/aliases on an existing entity without clobbering data."""
        if entity_type:
            cur.execute(
                "UPDATE entities SET type=COALESCE(type, ?) WHERE id=? AND type IS NULL",
                (entity_type, entity_id),
            )
        if aliases:
            for alias in aliases:
                cur.execute(
                    "INSERT OR IGNORE INTO entity_aliases(entity_id, alias, agent) "
                    "VALUES(?,?,?)",
                    (entity_id, alias, agent_id),
                )
            cur.execute("SELECT aliases FROM entities WHERE id=?", (entity_id,))
            row = cur.fetchone()
            existing = set()
            if row and row["aliases"]:
                try:
                    existing = {a.lower() for a in json.loads(row["aliases"])}
                except (json.JSONDecodeError, TypeError):
                    existing = set()
            merged = [a for a in aliases if a.lower() not in existing]
            if merged:
                cur.execute("SELECT aliases FROM entities WHERE id=?", (entity_id,))
                row = cur.fetchone()
                cur_aliases: list[str] = []
                if row and row["aliases"]:
                    try:
                        cur_aliases = list(json.loads(row["aliases"]))
                    except (json.JSONDecodeError, TypeError):
                        cur_aliases = []
                new_aliases = list(dict.fromkeys(cur_aliases + merged))
                cur.execute(
                    "UPDATE entities SET aliases=? WHERE id=?",
                    (json.dumps(new_aliases), entity_id),
                )

    def entity_name_for(self, entity_id: str) -> str:
        """Map an entity id back to its canonical name. '' if unknown."""
        with self.read() as cur:
            cur.execute("SELECT name FROM entities WHERE id=?", (entity_id,))
            row = cur.fetchone()
        return row["name"] if row and row["name"] else ""

    def link_entity(
        self,
        *,
        memory_id: str,
        entity_id: str,
        agent_id: str | None = None,
        relation_role: str | None = None,
    ) -> None:
        """Associate a memory with an entity (idempotent on (memory_id, entity_id))."""
        with self.transaction() as cur:
            cur.execute(
                "INSERT OR IGNORE INTO memory_entities(memory_id, entity_id, relation_role, agent) "
                "VALUES(?,?,?,?)",
                (memory_id, entity_id, relation_role, agent_id),
            )

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        with self.read() as cur:
            cur.execute("SELECT * FROM entities WHERE id=?", (entity_id,))
            row = cur.fetchone()
        return dict(row) if row else None

    def get_entities_batch(
        self, entity_ids: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Fetch multiple entity rows by id. Returns {id: row_dict}."""
        if not entity_ids:
            return {}
        placeholders = ",".join("?" for _ in entity_ids)
        with self.read() as cur:
            cur.execute(
                f"SELECT id, name, type, aliases, agent FROM entities "
                f"WHERE id IN ({placeholders})",
                entity_ids,
            )
            return {r["id"]: dict(r) for r in cur.fetchall()}

    def find_entity_by_name(self, name: str, agent_id: str | None = None) -> str | None:
        """Return entity id matching `name` (or any alias) without creating one."""
        key = _normalize_entity_name(name)
        if not key:
            return None
        with self.read() as cur:
            cur.execute(
                "SELECT id FROM entities WHERE LOWER(name)=? "
                "AND (agent IS ? OR (agent IS NULL AND EXISTS(SELECT 1 FROM memory_entities me "
                "JOIN memories m ON m.id=me.memory_id WHERE me.entity_id=entities.id "
                "AND m.agent=?))) ORDER BY agent IS NULL LIMIT 1",
                (key, agent_id, agent_id),
            )
            row = cur.fetchone()
            if row is not None:
                return row["id"]
            cur.execute(
                "SELECT e.id FROM entities e JOIN entity_aliases ea "
                "ON ea.entity_id = e.id "
                "WHERE ea.alias=? AND ea.agent IS ? "
                "ORDER BY ea.agent IS NULL LIMIT 1",
                (key, agent_id),
            )
            row = cur.fetchone()
            return row["id"] if row is not None else None

    def count_entity_links(self, entity_id: str) -> int:
        """Number of memories linked to ``entity_id`` (any status)."""
        if not entity_id:
            return 0
        with self.read() as cur:
            cur.execute(
                "SELECT COUNT(*) AS c FROM memory_entities WHERE entity_id=?",
                (entity_id,),
            )
            return int(cur.fetchone()["c"])

    def record_entity_sighting(
        self, name_key: str, agent_id: str | None, memory_id: str
    ) -> None:
        """Record a deferred entity sighting (idempotent)."""
        if not name_key or not memory_id:
            return
        with self.transaction() as cur:
            cur.execute(
                "INSERT OR IGNORE INTO entity_sightings(name_key, agent, memory_id, seen_at) "
                "VALUES(?,?,?,?)",
                (name_key, agent_id, memory_id, _now_iso()),
            )

    def entity_sighting_count(self, name_key: str, agent_id: str | None) -> int:
        """Number of distinct memories that have sighted ``name_key``."""
        if not name_key:
            return 0
        with self.read() as cur:
            cur.execute(
                "SELECT COUNT(*) AS c FROM entity_sightings WHERE name_key=? AND "
                "(agent IS NULL OR agent=?)",
                (name_key, agent_id),
            )
            return int(cur.fetchone()["c"])

    def entity_sighting_memory_ids(
        self, name_key: str, agent_id: str | None
    ) -> list[str]:
        """Memory ids of every sighting of ``name_key``."""
        if not name_key:
            return []
        with self.read() as cur:
            cur.execute(
                "SELECT memory_id FROM entity_sightings WHERE name_key=? AND "
                "(agent IS NULL OR agent=?)",
                (name_key, agent_id),
            )
            return [r["memory_id"] for r in cur.fetchall()]

    def clear_entity_sightings(self, name_key: str, agent_id: str | None) -> None:
        """Drop sighting rows for ``name_key`` (after promotion)."""
        if not name_key:
            return
        with self.transaction() as cur:
            cur.execute(
                "DELETE FROM entity_sightings WHERE name_key=? AND "
                "(agent IS NULL OR agent=?)",
                (name_key, agent_id),
            )

    # -- relations -------------------------------------------------------------

    def add_relation(
        self,
        *,
        entity_a: str,
        entity_b: str,
        relation_type: str = "related_to",
        strength: float = 0.5,
        source_memory_id: str | None = None,
    ) -> None:
        """Insert or strengthen a relation between two entities (undirected)."""
        a, b = sorted((entity_a, entity_b))
        with self.transaction() as cur:
            cur.execute(
                "INSERT INTO relations(entity_a, entity_b, relation_type, strength, "
                "source_memory_id, created_at) "
                "VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(entity_a, entity_b, relation_type) DO UPDATE SET "
                "strength=MAX(excluded.strength, relations.strength), "
                "created_at=excluded.created_at",
                (a, b, relation_type, strength, source_memory_id, _now_iso()),
            )
            if source_memory_id:
                now = _now_iso()
                cur.execute(
                    "SELECT id FROM claims WHERE memory_id=?",
                    (source_memory_id,),
                )
                claim = cur.fetchone()
                cur.execute(
                    "INSERT INTO relation_evidence(entity_a, entity_b, relation_type, "
                    "memory_id, claim_id, strength, active, created_at, updated_at) "
                    "VALUES(?,?,?,?,?,?,1,?,?) ON CONFLICT(entity_a, entity_b, "
                    "relation_type, memory_id) DO UPDATE SET strength=excluded.strength, "
                    "active=1, updated_at=excluded.updated_at",
                    (
                        a, b, relation_type, source_memory_id,
                        claim["id"] if claim else None, strength, now, now,
                    ),
                )

    def get_relations(self, entity_id: str) -> list[dict[str, Any]]:
        with self.read() as cur:
            cur.execute(
                "SELECT entity_a, entity_b, relation_type, strength, source_memory_id, "
                "created_at FROM relations WHERE entity_a=? OR entity_b=?",
                (entity_id, entity_id),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_memories_for_entity(
        self, entity_id: str, *, agent_id: str | None = None
    ) -> list[dict[str, Any]]:
        sql = (
            "SELECT m.id, m.content, m.visibility, m.agent AS agent_id, "
            "m.timestamp AS created_at, m.updated_at, m.status, m.tags, m.metadata "
            "FROM memory_entities me JOIN memories m ON m.id = me.memory_id "
            "WHERE me.entity_id=? AND m.status='active'"
        )
        params: list[Any] = [entity_id]
        if agent_id is not None:
            sql += " AND m.agent=?"
            params.append(agent_id)
        with self.read() as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    # -- graph traversal -------------------------------------------------------

    def traverse_graph(
        self,
        entity_id: str,
        *,
        depth: int = 2,
        agent_id: str | None = None,
        profile_scope: list[str] | None = None,
        evidence_only: bool = False,
    ) -> dict[str, Any]:
        """BFS over `relations` up to `depth` hops. Pure SQLite, no LLM.

        Returns ``{"entities": [...], "memories": [...]}`` where entities are
        dicts ``{id, name, type, depth}`` (the seed at depth 0) and memories
        are deduped active memories linked to any visited entity.
        """
        if agent_id is not None and not self.get_memories_for_entity(entity_id, agent_id=agent_id):
            return {"entities": [], "memories": []}
        visited: dict[str, int] = {entity_id: 0}
        order: list[str] = [entity_id]
        frontier: list[str] = [entity_id]
        for hop in range(1, depth + 1):
            if not frontier:
                break
            placeholders = ",".join("?" for _ in frontier)
            table = "relation_evidence" if evidence_only else "relations"
            memory_column = "memory_id" if evidence_only else "source_memory_id"
            relation_source = (
                f"(SELECT r.entity_a,r.entity_b FROM {table} r "
                f"JOIN memories m ON m.id=r.{memory_column} "
                "WHERE m.status='active' AND (? IS NULL OR m.agent=?)"
                + (" AND r.active=1" if evidence_only else "") + ")"
            )
            sql = (
                f"SELECT entity_a AS other FROM {relation_source} "
                f"WHERE entity_b IN ({placeholders}) UNION SELECT entity_b AS other "
                f"FROM {relation_source} WHERE entity_a IN ({placeholders})"
            )
            params = [agent_id, agent_id, *frontier, agent_id, agent_id, *frontier]
            with self.read() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
            next_frontier: list[str] = []
            for r in rows:
                other = r["other"]
                if other not in visited:
                    visited[other] = hop
                    order.append(other)
                    next_frontier.append(other)
            frontier = next_frontier

        # Load entity metadata for visited entities.
        entities_out: list[dict[str, Any]] = []
        if order:
            placeholders = ",".join("?" for _ in order)
            with self.read() as cur:
                cur.execute(
                    f"SELECT id, name, type, aliases FROM entities WHERE id IN ({placeholders})",
                    order,
                )
                rows = {r["id"]: dict(r) for r in cur.fetchall()}
            for eid in order:
                meta = rows.get(eid, {"id": eid, "name": None, "type": None, "aliases": None})
                meta = dict(meta)
                meta["depth"] = visited[eid]
                entities_out.append(meta)

        # Load active memories linked to any visited entity.
        memories_out: list[dict[str, Any]] = []
        if order:
            placeholders = ",".join("?" for _ in order)
            sql = (
                "SELECT DISTINCT m.id, m.content, m.visibility, m.agent AS agent_id, "
                "m.source, m.source_id, m.timestamp AS created_at, m.updated_at "
                f"FROM memory_entities me JOIN memories m ON m.id = me.memory_id "
                f"WHERE me.entity_id IN ({placeholders}) AND m.status='active'"
            )
            params = list(order)
            if agent_id is not None:
                sql += " AND m.agent=?"
                params.append(agent_id)
            sql, params = _append_profile_scope_sql(sql, params, profile_scope)
            with self.read() as cur:
                cur.execute(sql, params)
                memories_out = [dict(r) for r in cur.fetchall()]

        return {"entities": entities_out, "memories": memories_out}

    def search_graph(
        self,
        query_entities: list[str],
        *,
        agent_id: str | None = None,
        depth: int = 2,
        limit: int = 20,
        profile_scope: list[str] | None = None,
        evidence_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Resolve entity names from the query, traverse the graph, and return
        linked active memories. Pure SQLite (no embedding / LLM work).
        """
        seed_ids: list[str] = []
        for name in query_entities:
            eid = self.find_entity_by_name(name, agent_id=agent_id)
            if eid:
                seed_ids.append(eid)
        if not seed_ids:
            return []
        seen: dict[str, dict[str, Any]] = {}
        for eid in seed_ids:
            res = self.traverse_graph(
                eid,
                depth=depth,
                agent_id=agent_id,
                profile_scope=profile_scope,
                evidence_only=evidence_only,
            )
            for m in res["memories"]:
                mid = m["id"]
                if mid not in seen:
                    seen[mid] = m
        ranked = list(seen.values())
        ranked.sort(key=lambda r: r.get("updated_at", ""), reverse=True)
        return ranked[:limit]

    # -- edit helpers ----------------------------------------------------------

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        with self.read() as cur:
            cur.execute("SELECT * FROM memories WHERE id=?", (memory_id,))
            row = cur.fetchone()
        if row is None:
            return None
        d = dict(row)
        if d.get("metadata"):
            try:
                d["metadata"] = json.loads(d["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass
        if d.get("tags"):
            try:
                d["tags"] = json.loads(d["tags"])
            except (json.JSONDecodeError, TypeError):
                pass
        return d

    def update_memory_content(
        self,
        memory_id: str,
        *,
        content: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        embed_model: str | None = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        """In-place content update for a memory, preserving identity columns
        (memory_id, entity links, trust_score, retrieval history, etc.).

        Updates ``content``, ``tags``, ``metadata``, ``content_hash`` (sha256 of
        the new content) and ``updated_at``. Optionally replaces the embedding
        (INSERT OR REPLACE into ``embeddings``). Writes a ``vault_update`` audit
        row and rebuilds the FTS5 row so search reflects the new content.

        Returns ``{"memory": after, "audit_id": audit_id, "before": before}``
        like ``set_memory_field``. Raises ``KeyError`` if the memory is absent.
        """
        import hashlib

        before = self.get_memory(memory_id)
        if before is None:
            raise KeyError(memory_id)
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        tags_json = json.dumps(tags) if tags is not None else None
        meta_json = json.dumps(metadata, default=str) if metadata is not None else None
        now = _now_iso()
        with self.transaction() as cur:
            cur.execute(
                "UPDATE memories SET content=?, tags=?, metadata=?, content_hash=?, "
                "updated_at=? WHERE id=?",
                (content, tags_json, meta_json, content_hash, now, memory_id),
            )
            if embedding:
                blob = _pack_embedding(embedding)
                cur.execute(
                    "INSERT OR REPLACE INTO embeddings(memory_id, model, embedding, "
                    "dimensions, created_at) VALUES(?,?,?,?,?)",
                    (memory_id, embed_model, blob, len(embedding), now),
                )
            elif content != before["content"]:
                cur.execute("DELETE FROM embeddings WHERE memory_id=?", (memory_id,))
            # Rebuild the FTS5 row. The triggers fire on UPDATE already, but we
            # also do an explicit delete+insert so the index is consistent even
            # when the trigger path is bypassed by external-content quirks.
            cur.execute(
                "DELETE FROM memories_fts WHERE rowid="
                "(SELECT rowid FROM memories WHERE id=?)",
                (memory_id,),
            )
            cur.execute(
                "INSERT INTO memories_fts(rowid, content, tags) "
                "SELECT rowid, content, tags FROM memories WHERE id=?",
                (memory_id,),
            )
            audit_id = self._write_audit(
                cur, actor, "vault_update", memory_id,
                {"content_hash": content_hash},
            )
        after = self.get_memory(memory_id)
        return {"memory": after, "audit_id": audit_id, "before": before}

    def set_memory_field(
        self,
        memory_id: str,
        field: str,
        value: Any,
        *,
        actor: str = "system",
        action: str = "update",
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Atomically update one column on a memory and write an audit row.

        `field` must be one of the allowed mutable columns (whitelisted).
        Returns the updated memory row.
        """
        allowed = {
            "content", "visibility", "trust_score", "confidence",
            "verified", "status", "superseded_by", "tags", "metadata",
        }
        if field not in allowed:
            raise ValueError(f"field not mutable: {field}")
        before = self.get_memory(memory_id)
        if before is None:
            raise KeyError(memory_id)
        col_value = value
        if field in ("tags", "metadata") and isinstance(value, (dict, list)):
            col_value = json.dumps(value, default=str)
        with self.transaction() as cur:
            cur.execute(
                f"UPDATE memories SET {field}=?, updated_at=? WHERE id=?",
                (col_value, _now_iso(), memory_id),
            )
            audit_id = self._write_audit(cur, actor, action, memory_id, details or {})
        after = self.get_memory(memory_id)
        return {"memory": after, "audit_id": audit_id, "before": before}

    def supersede(self, old_id: str, new_id: str | None, *, actor: str = "system") -> int:
        """Mark `old_id` as superseded by `new_id` (or None). Returns audit id."""
        with self.transaction() as cur:
            cur.execute(
                "UPDATE memories SET status='superseded', superseded_by=?, updated_at=? WHERE id=?",
                (new_id, _now_iso(), old_id),
            )
            return self._write_audit(
                cur,
                actor,
                "supersede",
                old_id,
                {"superseded_by": new_id},
            )

    def _write_audit(
        self,
        cur: sqlite3.Cursor,
        actor: str,
        action: str,
        memory_id: str | None,
        details: dict[str, Any],
    ) -> int:
        cur.execute(
            "INSERT INTO audit_log(actor, action, memory_id, details, created_at) "
            "VALUES(?,?,?,?,?)",
            (actor, action, memory_id, json.dumps(details, default=str), _now_iso()),
        )
        return int(cur.lastrowid)

    def write_audit(
        self,
        *,
        actor: str,
        action: str,
        memory_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> int:
        """Public audit log writer (runs in its own transaction)."""
        with self.transaction() as cur:
            return self._write_audit(cur, actor, action, memory_id, details or {})

    def migrate_memory_agent(self, old_agent: str, new_agent: str) -> list[str]:
        """Retag a legacy owner atomically and retain one audit event per row."""
        with self.transaction() as cur:
            cur.execute("SELECT id FROM memories WHERE agent=?", (old_agent,))
            ids = [str(row["id"]) for row in cur.fetchall()]
            if not ids:
                return []
            cur.execute(
                "UPDATE memories SET agent=?, updated_at=? WHERE agent=?",
                (new_agent, _now_iso(), old_agent),
            )
            for memory_id in ids:
                self._write_audit(
                    cur,
                    "system",
                    "migrate_legacy_agent",
                    memory_id,
                    {"from_agent": old_agent, "to_agent": new_agent},
                )
            return ids

    def list_audit(
        self,
        *,
        memory_id: str | None = None,
        action: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        sql = "SELECT id, actor, action, memory_id, details, created_at FROM audit_log"
        where: list[str] = []
        params: list[Any] = []
        if memory_id is not None:
            where.append("memory_id=?")
            params.append(memory_id)
        if action is not None:
            where.append("action=?")
            params.append(action)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self.read() as cur:
            cur.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            if r.get("details"):
                try:
                    r["details"] = json.loads(r["details"])
                except (json.JSONDecodeError, TypeError):
                    pass
        return rows

    # -- embedding cache ------------------------------------------------------

    def get_cached_embedding(self, model: str, text_hash: str) -> list[float] | None:
        with self.read() as cur:
            cur.execute(
                "SELECT embedding FROM embedding_cache WHERE model=? AND text_hash=?",
                (model, text_hash),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return _unpack_embedding(row["embedding"])

    def put_cached_embedding(self, model: str, text_hash: str, embedding: list[float]) -> None:
        blob = _pack_embedding(embedding)
        with self.transaction() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO embedding_cache(model, text_hash, embedding, created_at) "
                "VALUES(?,?,?,?)",
                (model, text_hash, blob, time.time()),
            )

    # -- vault files (Phase 4) -------------------------------------------------

    def get_vault_hash(self, path: str, *, agent_id: str | None = None) -> str | None:
        with self.read() as cur:
            row = cur.execute(
                "SELECT hash FROM vault_files WHERE path=? AND (? IS NULL OR agent=?)",
                (path, agent_id, agent_id),
            ).fetchone()
        return row["hash"] if row else None

    def get_vault_memory(self, path: str, *, agent_id: str | None = None) -> str | None:
        with self.read() as cur:
            row = cur.execute(
                "SELECT memory_id FROM vault_files WHERE path=? AND (? IS NULL OR agent=?)",
                (path, agent_id, agent_id),
            ).fetchone()
        return row["memory_id"] if row else None

    def set_vault_hash(
        self, path: str, hash_hex: str, memory_id: str | None = None,
        *, agent_id: str | None = None,
    ) -> None:
        owner = (agent_id if agent_id is not None
                 else (self.get_memory(memory_id) or {}).get("agent", ""))
        with self.transaction() as cur:
            if memory_id:
                cur.execute(
                    "DELETE FROM vault_files WHERE path=? AND agent='' AND memory_id IS NULL",
                    (path,),
                )
            cur.execute(
                "INSERT OR REPLACE INTO vault_files(agent,path,hash,memory_id,indexed_at) "
                "VALUES(?,?,?,?,?)", (owner, path, hash_hex, memory_id, _now_iso()),
            )

    def get_vault_passages(
        self, path: str, *, agent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        with self.read() as cur:
            return [dict(row) for row in cur.execute(
                "SELECT * FROM vault_passages WHERE path=? AND (? IS NULL OR agent=?) "
                "ORDER BY ordinal", (path, agent_id, agent_id),
            ).fetchall()]

    def set_vault_passage(
        self, path: str, ordinal: int, memory_id: str, heading_path: str,
        start_offset: int, end_offset: int, *, agent_id: str | None = None,
    ) -> None:
        owner = (agent_id if agent_id is not None
                 else (self.get_memory(memory_id) or {}).get("agent", ""))
        with self.transaction() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO vault_passages "
                "(agent,path,ordinal,memory_id,heading_path,start_offset,end_offset) "
                "VALUES(?,?,?,?,?,?,?)",
                (owner, path, ordinal, memory_id, heading_path, start_offset, end_offset),
            )

    def forget_vault_passages_after(
        self, path: str, ordinal: int, *, agent_id: str | None = None,
    ) -> list[str]:
        with self.transaction() as cur:
            ids = [str(row[0]) for row in cur.execute(
                "SELECT memory_id FROM vault_passages WHERE path=? AND ordinal>? "
                "AND (? IS NULL OR agent=?)", (path, ordinal, agent_id, agent_id),
            ).fetchall()]
            for mid in ids:
                cur.execute("UPDATE memories SET status='forgotten',updated_at=? WHERE id=?",
                            (_now_iso(), mid))
            cur.execute("DELETE FROM vault_passages WHERE path=? AND ordinal>? "
                        "AND (? IS NULL OR agent=?)", (path, ordinal, agent_id, agent_id))
            return ids

    def get_all_vault_files(self, *, agent_id: str | None = None) -> list[dict[str, Any]]:
        with self.read() as cur:
            return [dict(row) for row in cur.execute(
                "SELECT * FROM vault_files WHERE (? IS NULL OR agent=?)", (agent_id, agent_id),
            ).fetchall()]

    def find_orphan_forgotten_memory_ids(self) -> list[str]:
        """Return IDs of forgotten memories with no source_id and no vault file.

        These "orphan" memories were created during broken vault reindexes: the
        vault_files row was dropped/never created, the memory was marked
        ``status='forgotten'``, and it carries no ``source_id``. They are safe to
        delete because nothing in the vault links back to them.
        """
        sql = (
            "SELECT m.id FROM memories m "
            "LEFT JOIN vault_files v ON v.memory_id = m.id "
            "WHERE m.status IN ('forgotten', 'inactive') "
            "AND m.source_id IS NULL "
            "AND v.memory_id IS NULL "
            "ORDER BY m.created_at"
        )
        with self.read() as cur:
            cur.execute(sql)
            return [r["id"] for r in cur.fetchall()]

    def mark_vault_forgotten(self, path: str, *, agent_id: str | None = None) -> str | None:
        """Forget only this owner's mappings, retaining the original evidence rows."""
        with self.transaction() as cur:
            rows = cur.execute(
                "SELECT agent,memory_id FROM vault_files WHERE path=? AND (? IS NULL OR agent=?)",
                (path, agent_id, agent_id),
            ).fetchall()
            for row in rows:
                owner = row["agent"]
                ids = [str(p[0]) for p in cur.execute(
                    "SELECT memory_id FROM vault_passages WHERE path=? AND agent=?",
                    (path, owner),
                ).fetchall()]
                if row["memory_id"]:
                    ids.append(row["memory_id"])
                for mid in set(ids):
                    cur.execute("UPDATE memories SET status='forgotten',updated_at=? WHERE id=?",
                                (_now_iso(), mid))
                cur.execute("DELETE FROM vault_passages WHERE path=? AND agent=?", (path, owner))
                cur.execute("DELETE FROM vault_files WHERE path=? AND agent=?", (path, owner))
            return rows[0]["memory_id"] if rows else None

    def mark_vault_forgotten_for_missing(
        self, present_paths: set[str], *, agent_id: str | None = None,
        profile_scope: list[str] | None = None,
    ) -> list[str]:
        forgotten = []
        for row in self.get_all_vault_files(agent_id=agent_id):
            if (row["path"] not in present_paths
                and path_in_profile_scope(row["path"], profile_scope)):
                ids = [p["memory_id"] for p in self.get_vault_passages(
                    row["path"], agent_id=row["agent"],
                )]
                mid = self.mark_vault_forgotten(row["path"], agent_id=row["agent"])
                forgotten.extend(dict.fromkeys(([mid] if mid else []) + ids))
        return forgotten

    def get_memory_by_source_id(
        self, source: str, source_id: str
    ) -> dict[str, Any] | None:
        """Return the most recent active memory matching (source, source_id)."""
        with self.read() as cur:
            cur.execute(
                "SELECT * FROM memories WHERE source=? AND source_id=? "
                "ORDER BY updated_at DESC LIMIT 1",
                (source, source_id),
            )
            row = cur.fetchone()
        if row is None:
            return None
        d = dict(row)
        if d.get("metadata"):
            try:
                d["metadata"] = json.loads(d["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass
        if d.get("tags"):
            try:
                d["tags"] = json.loads(d["tags"])
            except (json.JSONDecodeError, TypeError):
                pass
        return d

    def get_memory_by_content_hash(
        self, content_hash: str, *, agent_id: str | None = None
    ) -> dict[str, Any] | None:
        """Return the most recent active memory with a matching content_hash.

        Used by the migration import path to dedup incoming facts across
        memory_store and hindsight sources. Returns None when the hash is
        absent or the column is NULL.
        """
        if not content_hash:
            return None
        with self.read() as cur:
            cur.execute(
                "SELECT * FROM memories WHERE content_hash=? AND status='active' "
                "AND (? IS NULL OR agent=?) ORDER BY updated_at DESC LIMIT 1",
                (content_hash, agent_id, agent_id),
            )
            row = cur.fetchone()
        if row is None:
            return None
        d = dict(row)
        if d.get("metadata"):
            try:
                d["metadata"] = json.loads(d["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass
        if d.get("tags"):
            try:
                d["tags"] = json.loads(d["tags"])
            except (json.JSONDecodeError, TypeError):
                pass
        return d

    def list_active_memories_for_decay(
        self, *, batch_size: int | None = None, agent_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Return active memory ids with trust_score and updated_at.

        Used by the batch trust-decay job (issue #24). ``batch_size`` can
        bound the result for incremental passes; None returns the full set.
        """
        sql = (
            "SELECT id, trust_score, updated_at FROM memories "
            "WHERE status='active'"
        )
        params: list[Any] = []
        if agent_id is not None:
            sql += " AND agent=?"
            params.append(agent_id)
        if batch_size:
            sql += " LIMIT ?"
            params.append(batch_size)
        with self.read() as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def increment_seen_count(self, memory_id: str) -> int:
        """Bump the seen_count on a memory (duplicate re-observation).

        Returns the new seen_count value. Used by the import path to record
        that an incoming fact matched an existing memory without creating a
        new row.
        """
        with self.transaction() as cur:
            cur.execute(
                "UPDATE memories SET seen_count=COALESCE(seen_count,0)+1, "
                "updated_at=? WHERE id=?",
                (_now_iso(), memory_id),
            )
            cur.execute("SELECT seen_count FROM memories WHERE id=?", (memory_id,))
            row = cur.fetchone()
        return int(row["seen_count"]) if row and row["seen_count"] is not None else 0

    # -- threads (Phase 5) -----------------------------------------------------

    def insert_thread(
        self,
        *,
        title: str,
        topic: str,
        importance: float = 0.5,
        tags: list[str] | None = None,
        related_entities: list[str] | None = None,
        source: str = "manual",
        added_by: str = "system",
        owner: str | None = None,
    ) -> str:
        """Create a thread. Returns its id."""
        if not title.strip() or not topic.strip():
            raise ValueError("title and topic are required")
        owner = owner or added_by
        if not owner or owner == "system":
            raise ValueError("an explicit thread owner is required")
        tid = _uuid()
        now = _now_iso()
        with self.transaction() as cur:
            cur.execute(
                "INSERT INTO threads(id, title, topic, status, importance, tags, "
                "related_entities, source, added_by, created_at, last_activity, "
                "updated_at, owner) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    tid, title, topic, "active", importance,
                    json.dumps(tags) if tags else None,
                    json.dumps(related_entities) if related_entities else None,
                    source, added_by, now, now, now, owner,
                ),
            )
            return tid

    def update_thread(
        self,
        thread_id: str,
        *,
        title: str | None = None,
        status: str | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
        related_entities: list[str] | None = None,
        touch: bool = True,
    ) -> dict[str, Any] | None:
        """Update mutable fields on a thread. Returns the updated row or None."""
        before = self.get_thread(thread_id)
        if before is None:
            return None
        now = _now_iso()
        sets: list[str] = ["updated_at=?"]
        params: list[Any] = [now]
        if title is not None:
            sets.append("title=?")
            params.append(title)
        if status is not None:
            if status not in ("active", "stale", "resolved"):
                raise ValueError(f"invalid status: {status}")
            sets.append("status=?")
            params.append(status)
        if importance is not None:
            sets.append("importance=?")
            params.append(float(importance))
        if tags is not None:
            sets.append("tags=?")
            params.append(json.dumps(tags) if tags else None)
        if related_entities is not None:
            sets.append("related_entities=?")
            params.append(json.dumps(related_entities) if related_entities else None)
        if touch:
            sets.append("last_activity=?")
            params.append(now)
        params.append(thread_id)
        with self.transaction() as cur:
            cur.execute(
                f"UPDATE threads SET {','.join(sets)} WHERE id=?", params
            )
        return self.get_thread(thread_id)

    def resolve_thread(self, thread_id: str) -> dict[str, Any] | None:
        """Mark a thread resolved (preserved, never deleted)."""
        return self.update_thread(thread_id, status="resolved", touch=True)

    def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        with self.read() as cur:
            cur.execute("SELECT * FROM threads WHERE id=?", (thread_id,))
            row = cur.fetchone()
        if row is None:
            return None
        return _decode_thread(dict(row))

    def list_threads(
        self, *, status: str | None = None, limit: int = 50, agent_id: str | None = None
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM threads WHERE 1=1"
        params: list[Any] = []
        if status is not None:
            sql += " AND status=?"
            params.append(status)
        if agent_id is not None:
            sql += " AND owner=?"
            params.append(agent_id)
        sql += " ORDER BY last_activity DESC LIMIT ?"
        params.append(limit)
        with self.read() as cur:
            cur.execute(sql, params)
            return [_decode_thread(dict(r)) for r in cur.fetchall()]

    def stale_threads(self, *, days: int = 14, agent_id: str | None = None) -> list[dict[str, Any]]:
        """Return active threads whose last_activity is older than `days`."""
        cutoff = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - days * 86400)
        )
        with self.read() as cur:
            cur.execute(
                "SELECT * FROM threads WHERE status='active' AND last_activity < ? "
                "AND (? IS NULL OR owner=?) ORDER BY last_activity ASC",
                (cutoff, agent_id, agent_id),
            )
            return [_decode_thread(dict(r)) for r in cur.fetchall()]

    def sweep_stale_threads(self, *, days: int = 14, agent_id: str | None = None) -> list[str]:
        """Mark inactive active threads as stale. Returns the marked ids."""
        stale = self.stale_threads(days=days, agent_id=agent_id)
        if not stale:
            return []
        marked: list[str] = []
        with self.transaction() as cur:
            now = _now_iso()
            for t in stale:
                cur.execute(
                    "UPDATE threads SET status='stale', updated_at=? WHERE id=? "
                    "AND status='active'",
                    (now, t["id"]),
                )
                marked.append(t["id"])
        return marked

    # -- dream_state (Phase 5) -------------------------------------------------

    def get_state(self, key: str, default: Any = None, *, agent_id: str = "") -> Any:
        """Return a JSON-decoded value from dream_state, or `default`."""
        with self.read() as cur:
            cur.execute("SELECT value FROM dream_state WHERE owner=? AND key=?", (agent_id, key))
            row = cur.fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            return default

    def set_state(self, key: str, value: Any, *, agent_id: str = "") -> None:
        """Persist a JSON-serializable value under `key`."""
        with self.transaction() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO dream_state(owner, key, value, updated_at) "
                "VALUES(?,?,?,?)",
                (agent_id, key, json.dumps(value, default=str), _now_iso()),
            )

    def get_recent_memories(
        self, *, since_ts: float, agent_id: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        """Return active memories created since `since_ts` (epoch seconds).

        `since_ts` is compared against the memories.created_at ISO timestamp.
        """
        since_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(since_ts))
        sql = (
            "SELECT id, content, agent, visibility, source, type, timestamp, "
            "created_at, updated_at FROM memories WHERE status='active' "
            "AND created_at >= ?"
        )
        params: list[Any] = [since_iso]
        if agent_id is not None:
            sql += " AND agent=?"
            params.append(agent_id)
        sql += " ORDER BY created_at ASC LIMIT ?"
        params.append(limit)
        with self.read() as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def get_pending_turns(
        self,
        *,
        agent_id: str | None = None,
        session_id: str | None = None,
        max_age_s: float = 900.0,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Return recent turns whose extraction has not reached a terminal state.

        This is the read-after-write overlay source.  It deliberately returns
        only raw turns and never treats them as durable claims; callers must
        label them as unprocessed and apply the normal authorization scope.
        """
        cutoff = time.time() - max(0.0, float(max_age_s))
        sql = (
            "SELECT id, session_id, agent_id, user_text, assistant_text, "
            "created_at, extraction_status FROM turns "
            "WHERE created_at >= ? AND extraction_status IN "
            "('pending','running','retry_wait')"
        )
        params: list[Any] = [cutoff]
        if agent_id is not None:
            sql += " AND agent_id=?"
            params.append(agent_id)
        if session_id is not None:
            sql += " AND session_id=?"
            params.append(session_id)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        with self.read() as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def pending_extraction_count(
        self, *, agent_id: str | None = None, session_id: str | None = None
    ) -> int:
        """Return the number of non-terminal extraction turns."""
        sql = (
            "SELECT COUNT(*) AS n FROM turns "
            "WHERE extraction_status IN ('pending','running','retry_wait')"
        )
        params: list[Any] = []
        if agent_id is not None:
            sql += " AND agent_id=?"
            params.append(agent_id)
        if session_id is not None:
            sql += " AND session_id=?"
            params.append(session_id)
        with self.read() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        return int(row["n"] if row else 0)

    def get_memories_for_agent_scope(
        self, *, agent_id: str | None = None, visibility: str | None = "shared",
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Return active memories visible across the agent scope.

        Unscoped reads are for operator maintenance only. A supplied owner
        restricts every source and visibility label to that owner.
        """
        sql = (
            "SELECT id, content, agent, visibility, source, type, created_at "
            "FROM memories WHERE status='active'"
        )
        params: list[Any] = []
        if agent_id is not None:
            sql += " AND agent=?"
            params.append(agent_id)
        elif visibility is not None:
            sql += " AND visibility=?"
            params.append(visibility)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self.read() as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    # -- lifecycle -------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            if getattr(self, "_closed", False):
                return
            self.flush_diagnostics()
            self._conn.close()
            self._closed = True


def _to_fts_query(query: str) -> str:
    """Turn a free-text query into an FTS5 MATCH expression.

    Quotes individual tokens so punctuation/special chars don't break MATCH,
    and collapses whitespace. Empty input yields "" which callers treat as
    "no query".
    """
    if query is None:
        return ""
    tokens = [t for t in query.strip().split() if t]
    if not tokens:
        return ""
    return " ".join(f'"{t.replace(chr(34), "")}"' for t in tokens)


def _append_profile_scope_sql(
    sql: str,
    params: list[Any],
    profile_scope: list[str] | None,
    *,
    alias: str = "m",
) -> tuple[str, list[Any]]:
    """Apply vault profile scope before ranking and LIMIT.

    Non-vault memories are not path-scoped. Both current vault rows and legacy
    document rows are path-scoped, so an imported document cannot bypass the
    policy by carrying a different source label. An empty effective scope
    therefore excludes vault/document rows while retaining ordinary facts.
    """
    if profile_scope is None:
        return sql, params
    prefixes = normalize_profile_scope(profile_scope)
    if not prefixes:
        return (
            sql
            + f" AND COALESCE({alias}.source, '') <> 'vault'"
            + f" AND COALESCE({alias}.type, '') <> 'document'",
            params,
        )
    clauses: list[str] = []
    for prefix in prefixes:
        clauses.append(f"{alias}.source_id=? OR {alias}.source_id LIKE ?")
        params.extend([prefix, prefix + "/%"])
    non_document = (
        f"COALESCE({alias}.source, '') <> 'vault'"
        f" AND COALESCE({alias}.type, '') <> 'document'"
    )
    return sql + f" AND ({non_document} OR ({' OR '.join(clauses)}))", params


def _normalize_entity_name(name: str) -> str:
    """Lowercase, strip surrounding punctuation/whitespace for entity matching.

    Periods are preserved (internal initials/abbreviations like "Sven E." are
    semantically meaningful); other surrounding punctuation is stripped.
    """
    if not name:
        return ""
    s = name.strip().lower()
    # strip surrounding punctuation EXCEPT periods (initials/abbreviations).
    s = s.strip(",!?;:\"'()[]{}<>/\\|`~@#$%^&*-=+")
    import re as _re

    s = _re.sub(r"\s+", " ", s).strip()
    return s


def _decode_thread(row: dict[str, Any]) -> dict[str, Any]:
    """Decode JSON columns on a thread row."""
    for k in ("tags", "related_entities"):
        v = row.get(k)
        if v:
            try:
                row[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                pass
    return row


def open_db(db_path: str | Path) -> RemnantDB:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return RemnantDB(db_path)


__all__ = [
    "RemnantDB",
    "open_db",
    "default_db_path",
    "DEFAULT_DB_HOME",
    "_pack_embedding",
    "_unpack_embedding",
    "_normalize_entity_name",
]
