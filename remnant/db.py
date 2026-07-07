"""SQLite storage for Remnant: schema, migrations, CRUD.

- WAL mode for concurrent reader + single writer.
- FTS5 on memory text for BM25 keyword search.
- float32 blob embeddings stored in a separate `embeddings` table.
- Entity graph tables (`entities`, `memory_entities`, `relations`).
- `extraction_queue` table persists pending turns across restarts.
"""

from __future__ import annotations

import json
import os
import sqlite3
import struct
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 7

# Shared DB home: a single SQLite database used across all Hermes profiles /
# agents so cross-agent features (shared vault search, dream-loop dedup,
# graph traversal) work without merging per-profile DBs. The location can be
# overridden via the REMNANT_DB_HOME env var (used by tests for isolation).
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
    created_at REAL NOT NULL
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
    path TEXT PRIMARY KEY,
    hash TEXT NOT NULL,
    memory_id TEXT,
    indexed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vault_hash ON vault_files(hash);
CREATE INDEX IF NOT EXISTS idx_vault_memory ON vault_files(memory_id);

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
    created_at TEXT NOT NULL,
    last_activity TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_threads_status ON threads(status);
CREATE INDEX IF NOT EXISTS idx_threads_topic ON threads(topic);
CREATE INDEX IF NOT EXISTS idx_threads_last_activity ON threads(last_activity);

CREATE TABLE IF NOT EXISTS dream_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
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
        self._lock = threading.Lock()
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
            cur.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES(?,?)",
                ("version", str(SCHEMA_VERSION)),
            )

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
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN IMMEDIATE;")
            try:
                yield cur
                cur.execute("COMMIT;")
            except Exception:
                cur.execute("ROLLBACK;")
                raise

    @contextmanager
    def read(self) -> Iterator[sqlite3.Cursor]:
        """Read-only cursor (no lock; WAL allows concurrent readers)."""
        cur = self._conn.cursor()
        try:
            yield cur
        finally:
            cur.close()

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
                    "ORDER BY id DESC LIMIT 1"
                )
            else:
                cur.execute(
                    "SELECT * FROM extraction_queue WHERE status='pending' AND agent_id=? "
                    "ORDER BY id DESC LIMIT 1",
                    (agent_id,),
                )
            row = cur.fetchone()
            if row is None:
                return None
            qid = int(row["id"])
            cur.execute(
                "UPDATE extraction_queue SET attempts=attempts+1, status='running' WHERE id=?",
                (qid,),
            )
            return dict(row)

    def complete_extraction(self, queue_id: int) -> None:
        with self.transaction() as cur:
            cur.execute("DELETE FROM extraction_queue WHERE id=?", (queue_id,))

    def fail_extraction(self, queue_id: int) -> None:
        """Return the job to pending so the worker retries on next tick."""
        with self.transaction() as cur:
            cur.execute(
                "UPDATE extraction_queue SET status='pending' WHERE id=? AND attempts < 3",
                (queue_id,),
            )
            # After 3 attempts give up and drop the row; turn is still persisted.
            cur.execute("DELETE FROM extraction_queue WHERE id=? AND attempts >= 3", (queue_id,))

    def pending_count(self) -> int:
        with self.read() as cur:
            cur.execute("SELECT COUNT(*) AS c FROM extraction_queue WHERE status='pending'")
            return int(cur.fetchone()["c"])

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
            "WHERE q.id IS NULL AND m.id IS NULL"
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

    def deactivate_memory(self, memory_id: str) -> None:
        with self.transaction() as cur:
            cur.execute(
                "UPDATE memories SET status='inactive', updated_at=? WHERE id=?",
                (_now_iso(), memory_id),
            )

    def hard_delete_memory(self, memory_id: str) -> bool:
        """Permanently delete a memory and its cascading rows.

        Unlike ``deactivate_memory``, this performs a real DELETE. Intended for
        orphan cleanup (issue #23) where the memory is a duplicate/garbage row
        with no links to vault files or other memories.
        """
        with self.transaction() as cur:
            cur.execute(
                "DELETE FROM memories WHERE id=?", (memory_id,)
            )
            cur.execute(
                "DELETE FROM memories_fts WHERE rowid IN "
                "(SELECT id FROM memories WHERE id=?)", (memory_id,)
            )
            return cur.fetchone() is not None if cur.rowcount else cur.rowcount > 0

    def search_bm25(
        self,
        query: str,
        *,
        agent_id: str | None = None,
        visibility: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """BM25 keyword search over active memories, optionally filtered.

        Vault-source documents are a shared corpus: when ``agent_id`` is set,
        vault documents authored by *other* agents are still visible (subject
        to locked-note masking downstream in ``search``), so a user/agent can
        search the shared vault. Agent-scoped facts remain private to their
        owner.
        """
        fts_query = _to_fts_query(query)
        if not fts_query:
            return []
        sql = (
            "SELECT m.id, m.content, m.visibility, m.agent AS agent_id, "
            "m.timestamp AS created_at, m.updated_at, "
            "bm25(memories_fts) AS score "
            "FROM memories_fts JOIN memories m ON m.rowid = memories_fts.rowid "
            "WHERE memories_fts MATCH ? AND m.status='active'"
        )
        params: list[Any] = [fts_query]
        if agent_id is not None:
            sql += " AND (m.agent=? OR m.source='vault')"
            params.append(agent_id)
        if visibility is not None:
            sql += " AND m.visibility=?"
            params.append(visibility)
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
        """Fetch candidate memories (with embeddings) for cosine dedup."""
        fts_query = _to_fts_query(query)
        if not fts_query:
            return []
        sql = (
            "SELECT m.id, m.content, m.visibility, e.embedding "
            "FROM memories_fts JOIN memories m ON m.rowid = memories_fts.rowid "
            "LEFT JOIN embeddings e ON e.memory_id = m.id "
            "WHERE memories_fts MATCH ? AND m.status='active' AND m.agent=?"
        )
        params: list[Any] = [fts_query, agent_id]
        sql += " ORDER BY bm25(memories_fts) ASC LIMIT ?"
        params.append(limit)
        with self.read() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["embedding"] = _unpack_embedding(d["embedding"]) if d["embedding"] else []
            out.append(d)
        return out

    def get_memory_embedding(self, memory_id: str) -> list[float]:
        with self.read() as cur:
            cur.execute("SELECT embedding FROM embeddings WHERE memory_id=?", (memory_id,))
            row = cur.fetchone()
        return _unpack_embedding(row["embedding"]) if row and row["embedding"] else []

    def search_by_embedding(
        self,
        memory_ids: list[str],
        *,
        agent_id: str | None = None,
        visibility: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return active memories in `memory_ids` with their embeddings attached.

        Used by semantic search to load embeddings only for the BM25-pre-filtered
        candidate set, never the whole table. `memory_ids` should already be
        bounded by the caller (e.g. SEMANTIC_CANDIDATE_LIMIT).

        Vault-source documents are a shared corpus (see ``search_bm25``): when
        ``agent_id`` is set, vault documents authored by other agents remain
        visible so the shared vault can be semantically searched.
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
            sql += " AND (m.agent=? OR m.source='vault')"
            params.append(agent_id)
        if visibility is not None:
            sql += " AND m.visibility=?"
            params.append(visibility)
        with self.read() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["embedding"] = _unpack_embedding(d["embedding"]) if d["embedding"] else []
            out.append(d)
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

        Vault-source documents are a shared corpus (see ``search_bm25``): when
        ``agent_id`` is set, vault documents authored by other agents remain
        visible.
        """
        sql = (
            "SELECT m.id, m.content, m.visibility, m.agent AS agent_id, "
            "m.timestamp AS created_at "
            "FROM memories m WHERE m.status='active'"
        )
        params: list[Any] = []
        if agent_id is not None:
            sql += " AND (m.agent=? OR m.source='vault')"
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
                "AND (agent IS NULL OR agent = ?) "
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
                        "WHERE ea.alias = ? AND (ea.agent IS NULL OR ea.agent = ?) "
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
                "AND (agent IS NULL OR agent=?) ORDER BY agent IS NULL LIMIT 1",
                (key, agent_id),
            )
            row = cur.fetchone()
            if row is not None:
                return row["id"]
            cur.execute(
                "SELECT e.id FROM entities e JOIN entity_aliases ea "
                "ON ea.entity_id = e.id "
                "WHERE ea.alias=? AND (ea.agent IS NULL OR ea.agent=?) "
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
    ) -> dict[str, Any]:
        """BFS over `relations` up to `depth` hops. Pure SQLite, no LLM.

        Returns ``{"entities": [...], "memories": [...]}`` where entities are
        dicts ``{id, name, type, depth}`` (the seed at depth 0) and memories
        are deduped active memories linked to any visited entity.
        """
        visited: dict[str, int] = {entity_id: 0}
        order: list[str] = [entity_id]
        frontier: list[str] = [entity_id]
        for hop in range(1, depth + 1):
            if not frontier:
                break
            placeholders = ",".join("?" for _ in frontier)
            sql = (
                f"SELECT entity_a AS other FROM relations WHERE entity_b IN ({placeholders}) "
                f"UNION SELECT entity_b AS other FROM relations WHERE entity_a IN ({placeholders})"
            )
            params = list(frontier) + list(frontier)
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
                "m.timestamp AS created_at, m.updated_at "
                f"FROM memory_entities me JOIN memories m ON m.id = me.memory_id "
                f"WHERE me.entity_id IN ({placeholders}) AND m.status='active'"
            )
            params = list(order)
            if agent_id is not None:
                sql += " AND m.agent=?"
                params.append(agent_id)
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
            res = self.traverse_graph(eid, depth=depth, agent_id=agent_id)
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

    def get_vault_hash(self, path: str) -> str | None:
        """Return the stored content hash for a vault path, or None if absent."""
        with self.read() as cur:
            cur.execute("SELECT hash FROM vault_files WHERE path=?", (path,))
            row = cur.fetchone()
        return row["hash"] if row else None

    def get_vault_memory(self, path: str) -> str | None:
        """Return the memory_id linked to a vault path, or None."""
        with self.read() as cur:
            cur.execute("SELECT memory_id FROM vault_files WHERE path=?", (path,))
            row = cur.fetchone()
        return row["memory_id"] if row and row["memory_id"] else None

    def set_vault_hash(
        self, path: str, hash_hex: str, memory_id: str | None = None
    ) -> None:
        """Insert or update the vault_files row for `path` (idempotent)."""
        with self.transaction() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO vault_files(path, hash, memory_id, indexed_at) "
                "VALUES(?,?,?,?)",
                (path, hash_hex, memory_id, _now_iso()),
            )

    def get_all_vault_files(self) -> list[dict[str, Any]]:
        """Return all known vault_files rows: {path, hash, memory_id, indexed_at}."""
        with self.read() as cur:
            cur.execute(
                "SELECT path, hash, memory_id, indexed_at FROM vault_files"
            )
            return [dict(r) for r in cur.fetchall()]

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

    def mark_vault_forgotten(self, path: str) -> str | None:
        """Forget the memory linked to a vault path and remove the row.

        Returns the memory_id that was forgotten (or None if no row existed),
        so callers can audit/react. The memory row is preserved (status set to
        'forgotten', never deleted) per Remnant's "nothing is ever deleted" rule.
        """
        with self.transaction() as cur:
            cur.execute("SELECT memory_id FROM vault_files WHERE path=?", (path,))
            row = cur.fetchone()
            mid = row["memory_id"] if row else None
            if mid:
                cur.execute(
                    "UPDATE memories SET status='forgotten', updated_at=? WHERE id=?",
                    (_now_iso(), mid),
                )
            cur.execute("DELETE FROM vault_files WHERE path=?", (path,))
        return mid

    def mark_vault_forgotten_for_missing(
        self, present_paths: set[str]
    ) -> list[str]:
        """Forget memories for every vault_files row whose path is not in
        `present_paths`. Returns the list of forgotten memory ids (may be
        empty). Used by the re-index pass to handle deleted files.
        """
        forgotten: list[str] = []
        with self.transaction() as cur:
            cur.execute("SELECT path, memory_id FROM vault_files")
            rows = cur.fetchall()
            for r in rows:
                if r["path"] in present_paths:
                    continue
                mid = r["memory_id"]
                if mid:
                    cur.execute(
                        "UPDATE memories SET status='forgotten', updated_at=? WHERE id=?",
                        (_now_iso(), mid),
                    )
                    forgotten.append(mid)
                cur.execute("DELETE FROM vault_files WHERE path=?", (r["path"],))
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

    def get_memory_by_content_hash(self, content_hash: str) -> dict[str, Any] | None:
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
                "ORDER BY updated_at DESC LIMIT 1",
                (content_hash,),
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
        self, *, batch_size: int | None = None
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
    ) -> str:
        """Create a thread. Returns its id."""
        if not title.strip() or not topic.strip():
            raise ValueError("title and topic are required")
        tid = _uuid()
        now = _now_iso()
        with self.transaction() as cur:
            cur.execute(
                "INSERT INTO threads(id, title, topic, status, importance, tags, "
                "related_entities, source, added_by, created_at, last_activity, "
                "updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    tid, title, topic, "active", importance,
                    json.dumps(tags) if tags else None,
                    json.dumps(related_entities) if related_entities else None,
                    source, added_by, now, now, now,
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
        self, *, status: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM threads"
        params: list[Any] = []
        if status is not None:
            sql += " WHERE status=?"
            params.append(status)
        sql += " ORDER BY last_activity DESC LIMIT ?"
        params.append(limit)
        with self.read() as cur:
            cur.execute(sql, params)
            return [_decode_thread(dict(r)) for r in cur.fetchall()]

    def stale_threads(self, *, days: int = 14) -> list[dict[str, Any]]:
        """Return active threads whose last_activity is older than `days`."""
        cutoff = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - days * 86400)
        )
        with self.read() as cur:
            cur.execute(
                "SELECT * FROM threads WHERE status='active' AND last_activity < ? "
                "ORDER BY last_activity ASC",
                (cutoff,),
            )
            return [_decode_thread(dict(r)) for r in cur.fetchall()]

    def sweep_stale_threads(self, *, days: int = 14) -> list[str]:
        """Mark inactive active threads as stale. Returns the marked ids."""
        stale = self.stale_threads(days=days)
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

    def get_state(self, key: str, default: Any = None) -> Any:
        """Return a JSON-decoded value from dream_state, or `default`."""
        with self.read() as cur:
            cur.execute("SELECT value FROM dream_state WHERE key=?", (key,))
            row = cur.fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            return default

    def set_state(self, key: str, value: Any) -> None:
        """Persist a JSON-serializable value under `key`."""
        with self.transaction() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO dream_state(key, value, updated_at) "
                "VALUES(?,?,?)",
                (key, json.dumps(value, default=str), _now_iso()),
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

    def get_memories_for_agent_scope(
        self, *, agent_id: str | None = None, visibility: str | None = "shared",
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Return active memories visible across the agent scope.

        For cross-agent duplicate detection: when `agent_id` is None all active
        memories are considered; otherwise agent-scoped memories of `agent_id`
        plus all shared/fleet memories from other agents.
        """
        sql = (
            "SELECT id, content, agent, visibility, source, type, created_at "
            "FROM memories WHERE status='active'"
        )
        params: list[Any] = []
        if agent_id is not None:
            sql += " AND (agent=? OR visibility IN ('shared','fleet'))"
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
            self._conn.close()


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
