"""SQLite storage for Remnant: schema, migrations, CRUD.

- WAL mode for concurrent reader + single writer.
- FTS5 on memory text for BM25 keyword search.
- float32 blob embeddings stored in a separate `embeddings` table.
- Entity graph tables (`entities`, `memory_entities`, `relations`).
- `extraction_queue` table persists pending turns across restarts.
"""

from __future__ import annotations

import json
import sqlite3
import struct
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2

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
    source TEXT NOT NULL CHECK(source IN ('conversation','vault','email','cron','sensor','manual')),
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
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mem_agent ON memories(agent);
CREATE INDEX IF NOT EXISTS idx_mem_visibility ON memories(visibility);
CREATE INDEX IF NOT EXISTS idx_mem_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_mem_source ON memories(source);

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
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);

CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    relation_role TEXT,
    PRIMARY KEY(memory_id, entity_id),
    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_me_entity ON memory_entities(entity_id);

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
        conn.row_factory = sqlite3.Row
        return conn

    def _migrate(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.executescript(_SCHEMA)
            cur.executescript(_FTS_TRIGGERS)
            cur.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES(?,?)",
                ("version", str(SCHEMA_VERSION)),
            )

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
        """Atomically claim the next pending extraction job."""
        with self.transaction() as cur:
            if agent_id is None:
                cur.execute(
                    "SELECT * FROM extraction_queue WHERE status='pending' "
                    "ORDER BY id LIMIT 1"
                )
            else:
                cur.execute(
                    "SELECT * FROM extraction_queue WHERE status='pending' AND agent_id=? "
                    "ORDER BY id LIMIT 1",
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
        embedding: list[float] | None = None,
        embed_model: str | None = None,
    ) -> str:
        now = _now_iso()
        mid = _uuid()
        tags_json = json.dumps(tags) if tags else None
        meta_json = json.dumps(metadata) if metadata else None
        with self.transaction() as cur:
            cur.execute(
                "INSERT INTO memories(id, type, content, source, source_id, agent, "
                "visibility, timestamp, confidence, trust_score, verified, superseded_by, "
                "status, tags, metadata, created_at, updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,0,NULL,?,?,?,?,?)",
                (
                    mid, type, content, source, source_id, agent, visibility,
                    now, confidence, 0.5, "active", tags_json, meta_json, now, now,
                ),
            )
            if embedding is not None:
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

    def search_bm25(
        self,
        query: str,
        *,
        agent_id: str | None = None,
        visibility: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """BM25 keyword search over active memories, optionally filtered."""
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
            sql += " AND m.agent=?"
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
        candidate set. This keeps the pre-filter cheap and avoids scanning
        the embeddings table blindly.
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

    def resolve_entity(self, name: str, agent_id: str) -> str:
        """Return canonical form for `name` within this agent's scope.

        For Phase 1 we keep the previous behavior: return a stripped canonical
        name. An entity row is created (or reused) so the entity graph is
        populated for later phases. The agent scope is recorded via the
        `metadata` field on the memory, not as a separate column, per the
        Phase 1 schema.
        """
        key = name.strip()
        if not key:
            return key
        with self.transaction() as cur:
            cur.execute("SELECT id, name FROM entities WHERE name=?", (key,))
            row = cur.fetchone()
            if row is not None:
                return row["name"]
            eid = _uuid()
            cur.execute(
                "INSERT OR IGNORE INTO entities(id, name, type, aliases, created_at) "
                "VALUES(?,?,?,?,?)",
                (eid, key, None, json.dumps([]), _now_iso()),
            )
            return key

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


def open_db(db_path: str | Path) -> RemnantDB:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return RemnantDB(db_path)


__all__ = [
    "RemnantDB",
    "open_db",
    "_pack_embedding",
    "_unpack_embedding",
]
