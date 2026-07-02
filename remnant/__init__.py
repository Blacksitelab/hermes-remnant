"""Remnant memory provider for Hermes Agent.

Implements the Hermes ``MemoryProvider`` ABC. All storage is profile-scoped
under ``hermes_home`` (passed via ``initialize()``). ``sync_turn`` is
non-blocking: it persists the raw turn and enqueues extraction in a single
SQLite transaction, then wakes the background worker.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_VAULT_EXCLUDE,
    DEFAULT_VAULT_PATH,
    DEFAULT_VAULT_REINDEX_INTERVAL_S,
    RemnantConfig,
    load_config,
    save_config,
)
from .db import RemnantDB, open_db
from .dream import day_dream, night_dream
from .embed import Embedder
from .extract import ExtractionWorker
from .ingest import ingest_turn
from .prefetch import prefetch as _run_prefetch
from .tools import TOOL_SCHEMAS, handle_tool_call
from .vault import index_vault as _index_vault

log = logging.getLogger("remnant")


class _SessionEmbedder:
    """Per-session query-embedding cache wrapper.

    Wraps the real Embedder so the (single) Ollama query embedding for a given
    session is computed at most once and reused across all expanded query terms
    in a prefetch call. Falls back to the underlying embedder transparently.
    """

    def __init__(self, embedder: Embedder, query: str, qvec: list[float] | None = None):
        self._embedder = embedder
        self._query = query
        self._qvec: list[float] | None = qvec

    def embed(self, text: str) -> list[float]:
        # Reuse the cached query vector when the text matches the session query,
        # otherwise delegate to the real embedder (which has its own SQLite cache).
        if text == self._query and self._qvec is not None:
            return self._qvec
        return self._embedder.embed(text)

# --- Hermes ABC -----------------------------------------------------------
# The real ABC lives in ``agent.memory_provider`` inside a running Hermes.
# We import it defensively so the plugin still loads (and tests can run)
# without the Hermes package installed.
try:
    from agent.memory_provider import MemoryProvider  # type: ignore[import]
except Exception:  # pragma: no cover - fallback for standalone/test envs

    class MemoryProvider:  # type: ignore[no-redef]
        """Minimal ABC stub used when running outside Hermes."""

        def name(self) -> str:  # pragma: no cover
            raise NotImplementedError

        def is_available(self) -> bool:  # pragma: no cover
            raise NotImplementedError

        def initialize(self, session_id: str, **kwargs: Any) -> None:  # pragma: no cover
            raise NotImplementedError

        def get_tool_schemas(self) -> list[dict[str, Any]]:  # pragma: no cover
            raise NotImplementedError

        def handle_tool_call(  # pragma: no cover
            self, tool_name: str, args: dict[str, Any], **kwargs: Any
        ) -> Any:
            raise NotImplementedError

        def get_config_schema(self) -> list[dict[str, Any]]:  # pragma: no cover
            raise NotImplementedError

        def save_config(self, values: dict[str, Any], hermes_home: str) -> None:  # pragma: no cover
            raise NotImplementedError

        def system_prompt_block(self) -> str:  # pragma: no cover
            return ""

        def sync_turn(  # pragma: no cover
            self,
            user_content: str,
            assistant_content: str,
            *,
            session_id: str = "",
            messages: list[dict[str, Any]] | None = None,
        ) -> None:
            return None

        def shutdown(self) -> None:  # pragma: no cover
            return None


# Static, byte-stable block describing the provider to the agent. It never
# includes live data, so it is safe to return the same constant every call
# for the lifetime of a conversation.
_SYSTEM_PROMPT_BLOCK = (
    "## Remnant Memory Provider\n"
    "You have durable long-term memory via the Remnant provider.\n"
    "Use the `memory_search` tool to recall facts (keyword, semantic, auto hybrid, "
    "or graph entity-traversal strategies). Pass `profile_scope` to restrict "
    "vault documents to a set of allowed path prefixes.\n"
    "Use the `memory_store` tool to save a durable fact explicitly.\n"
    "Use the `memory_reflect` tool to synthesize an answer across stored memories.\n"
    "Use the `memory_graph` tool to explore entities and their connected memories.\n"
    "Use the `memory_edit` tool to update, merge, forget, score, or share memories. "
    "Nothing is ever deleted: forgotten memories stay in the DB but are hidden from "
    "search; updates supersede the old version while preserving it.\n"
    "Use the `memory_import` tool with `source='vault'` to re-index the Obsidian "
    "vault: new and changed notes become document memories, deleted notes are "
    "forgotten. Excluded vault folders (90_*-95_*, 99_ARCHIVE) are skipped. Locked "
    "notes are indexed but their content is hidden from other agents in search.\n"
    "Transient state (percentages, current status, timestamps) is rejected.\n"
    "Memories are scoped by agent and visibility (private/shared/fleet).\n"
    "Use the `memory_thread` tool to manage topic threads: create, update, "
    "resolve, list, or sweep stale (threads inactive 14 days are marked stale). "
    "Threads capture ongoing conversations and dream-loop suggestions; they are "
    "never deleted.\n"
    "A bounded dream loop (day_dream / night_dream, invokable from a cron timer) "
    "finds non-obvious connections across memories and writes reflections to a "
    "private DREAMS.md diary; cross-agent duplicates are merged into shared "
    "memory. The loop pre-filters candidates locally and only ever sends a small "
    "bounded list to the cloud model.\n"
)

# Config schema exposed to `hermes memory setup`. Kept minimal: only fields a
# user must configure. Endpoints/models default to the BSL1 Ollama setup.
_CONFIG_SCHEMA: list[dict[str, Any]] = [
    {
        "key": "embed_url",
        "description": "Ollama embeddings endpoint",
        "default": "http://192.168.0.11:11434/api/embeddings",
        "required": False,
    },
    {
        "key": "embed_model",
        "description": "Embedding model name",
        "default": "nomic-embed-text",
        "required": False,
    },
    {
        "key": "extract_url",
        "description": "Extraction LLM OpenAI-compatible endpoint",
        "default": "http://192.168.0.11:11434/v1/chat/completions",
        "required": False,
    },
    {
        "key": "extract_model",
        "description": "Extraction LLM model name",
        "default": "gemma4:12b",
        "required": False,
    },
    {
        "key": "extract_enabled",
        "description": "Enable async LLM extraction of facts",
        "default": True,
        "required": False,
    },
    {
        "key": "default_visibility",
        "description": "Default visibility for auto-extracted memories",
        "default": "private",
        "choices": ["private", "shared", "fleet"],
        "required": False,
    },
    {
        "key": "agent_id",
        "description": "Agent identifier scoping memories",
        "default": "default",
        "required": False,
    },
    {
        "key": "vault_path",
        "description": "Path to the Obsidian vault to index as document memories",
        "default": DEFAULT_VAULT_PATH,
        "required": False,
    },
    {
        "key": "vault_exclude",
        "description": (
            "Top-level vault folder name prefixes to exclude from indexing "
            "(e.g. 90_-95_*, 99_ARCHIVE)."
        ),
        "default": DEFAULT_VAULT_EXCLUDE,
        "required": False,
    },
    {
        "key": "profile_scope",
        "description": (
            "List of allowed vault path prefixes for this agent's document "
            "search. Empty means no additional filtering."
        ),
        "default": [],
        "required": False,
    },
    {
        "key": "vault_reindex_interval_s",
        "description": "Minimum seconds between automatic vault re-index passes",
        "default": DEFAULT_VAULT_REINDEX_INTERVAL_S,
        "required": False,
    },
]


class RemnantMemoryProvider(MemoryProvider):
    """Hermes memory provider backed by SQLite + Ollama extraction."""

    def __init__(self) -> None:
        self._config: RemnantConfig | None = None
        self._db: RemnantDB | None = None
        self._embedder: Embedder | None = None
        self._worker: ExtractionWorker | None = None
        self._session_id: str = ""
        self._hermes_home: str = ""
        self._started: bool = False
        # Phase 2: per-session injection + query-embedding caches.
        self._last_injected_hash: dict[str, str] = {}
        self._session_query_vec: dict[str, list[float]] = {}
        self._session_query: dict[str, str] = {}
        self._prefetch_queue: list[tuple[str, str]] = []

    # -- lifecycle ------------------------------------------------------------

    @property
    def name(self) -> str:
        return "remnant"

    def is_available(self) -> bool:
        """No network calls. We are available as long as we can write files."""
        return True

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        hermes_home = kwargs.get("hermes_home")
        if not hermes_home:
            raise ValueError("initialize() requires hermes_home in kwargs")
        self._hermes_home = str(hermes_home)
        self._session_id = session_id or "default"
        self._config = load_config(self._hermes_home)
        db_path = Path(self._hermes_home) / "remnant" / "remnant.db"
        self._db = open_db(db_path)
        self._embedder = Embedder(self._db, self._config)
        self._worker = ExtractionWorker(self._db, self._embedder, self._config)
        self._worker.start()
        self._started = True
        log.info("remnant initialized (home=%s, session=%s)", self._hermes_home, self._session_id)

    def shutdown(self) -> None:
        try:
            if self._worker is not None:
                self._worker.stop()
        finally:
            # Phase 4: no background vault watcher process is started here.
            # Re-index is driven by an external cron/timer calling
            # `reindex_vault()`; nothing to stop on shutdown.
            if self._embedder is not None:
                self._embedder.close()
            if self._db is not None:
                self._db.close()
            self._started = False

    # -- config ---------------------------------------------------------------

    def get_config_schema(self) -> list[dict[str, Any]]:
        return list(_CONFIG_SCHEMA)

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        save_config(values, hermes_home)

    # -- tools ----------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return list(TOOL_SCHEMAS)

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> Any:
        if self._db is None or self._config is None or self._embedder is None:
            return {"error": "provider not initialized"}
        session_id = kwargs.get("session_id", self._session_id)
        agent_id = kwargs.get("agent_id", self._config.agent_id)
        return handle_tool_call(
            tool_name,
            args,
            db=self._db,
            config=self._config,
            embedder=self._embedder,
            session_id=session_id,
            agent_id=agent_id,
        )

    # -- prompts --------------------------------------------------------------

    def system_prompt_block(self) -> str:
        return _SYSTEM_PROMPT_BLOCK

    # -- turns ----------------------------------------------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """Persist the turn and enqueue extraction without blocking."""
        if self._db is None or self._config is None:
            return
        sid = session_id or self._session_id or "default"
        try:
            ingest_turn(
                self._db,
                user_text=user_content or "",
                assistant_text=assistant_content or "",
                session_id=sid,
                agent_id=self._config.agent_id,
            )
        except Exception as e:
            log.warning("sync_turn failed: %s", e)
            return
        # Wake the background worker so it picks up the new job promptly.
        if self._worker is not None:
            self._worker.wake()

    # -- prefetch (Phase 2) ---------------------------------------------------

    def _session_embedder(self, session_id: str, query: str) -> _SessionEmbedder | None:
        """Return an embedder wrapper that caches the query vector per session.

        Computes the single Ollama query embedding once (the only network call
        in prefetch) and reuses it for every expanded query term in this
        session. Returns None if the provider isn't initialized.
        """
        if self._embedder is None or self._config is None:
            return None
        cached = self._session_query_vec.get(session_id)
        if cached is None:
            try:
                cached = self._embedder.embed(query)
            except Exception:
                cached = []
            if cached:
                self._session_query_vec[session_id] = cached
        return _SessionEmbedder(self._embedder, query, qvec=cached)

    def prefetch(
        self, query: str, *, session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Proactive memory injection before an LLM call.

        Returns a compact context block to prepend, or ``{}`` when memory isn't
        needed, the deadline is exceeded, or the context is unchanged since the
        last call in this session (diff-based suppression).
        """
        if self._db is None or self._config is None or self._embedder is None:
            return {}
        sid = session_id or self._session_id or "default"
        # Reset the per-session query-vector cache when the query changes so a
        # new query re-embeds; an identical query reuses the cached vector.
        if self._session_query.get(sid) != query:
            self._session_query.pop(sid, None)
            self._session_query[sid] = query
        return _run_prefetch(self, query, sid, messages=messages)

    def queue_prefetch(self, query: str) -> None:
        """Optionally pre-warm the next turn's prefetch. Non-blocking best-effort."""
        if self._db is None or self._config is None:
            return
        sid = self._session_id or "default"
        self._prefetch_queue.append((sid, query))
        # Keep the queue bounded.
        if len(self._prefetch_queue) > 32:
            self._prefetch_queue = self._prefetch_queue[-32:]

    # -- vault re-index (Phase 4) --------------------------------------------

    def reindex_vault(self, *, force: bool = False) -> dict[str, int]:
        """Re-walk the vault and index/forget documents. Returns stats.

        Safe to call from an external cron/timer. Uses the provider's
        configured ``vault_path`` / ``vault_exclude``. See
        ``remnant.vault.index_vault`` for the underlying implementation.
        """
        if self._db is None or self._config is None or self._embedder is None:
            return {"indexed": 0, "skipped": 0, "forgotten": 0}
        return _index_vault(
            self._db, self._config, self._embedder, force=force
        )

    # -- dream loop (Phase 5) -------------------------------------------------

    def run_dream_loop(self, mode: str = "day") -> dict[str, Any]:
        """Run a single dream-loop pass (callable from a cron/systemd timer).

        ``mode`` is ``"day"`` or ``"night"``. Returns the loop summary dict.
        Never starts a daemon: each call is one bounded pass.
        """
        if self._db is None or self._config is None or self._embedder is None:
            return {"mode": mode, "skipped": "not_initialized"}
        if mode == "day":
            return day_dream(self._db, self._config, self._embedder)
        if mode == "night":
            return night_dream(self._db, self._config, self._embedder)
        return {"mode": mode, "error": f"unknown mode: {mode}"}


def register(ctx: Any) -> None:
    """Plugin entry point discovered by Hermes."""
    ctx.register_memory_provider(RemnantMemoryProvider())


__all__ = ["RemnantMemoryProvider", "register"]
