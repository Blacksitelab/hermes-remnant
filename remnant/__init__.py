"""Remnant memory provider for Hermes Agent.

Implements the Hermes ``MemoryProvider`` ABC. Configuration stays profile-scoped
under ``hermes_home`` (loaded from ``hermes_home/remnant.json``), but the SQLite
database is **shared** across all profiles/agents at
``~/.hermes/remnant/remnant.db`` (overridable via ``REMNANT_DB_HOME``) so that
storage can be backed up centrally while provider access remains profile-owned.
``sync_turn`` is non-blocking:
it persists the raw turn and enqueues extraction in a single SQLite
transaction, then wakes the background worker.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .config import (
    DEFAULT_EXTRACT_MAX_FACTS,
    DEFAULT_EXTRACT_MAX_INPUT_TOKENS,
    DEFAULT_EXTRACT_MAX_OUTPUT_TOKENS,
    DEFAULT_EXTRACT_NUM_CTX,
    DEFAULT_EXTRACT_STRUCTURED_OUTPUT,
    DEFAULT_EXTRACT_THINK,
    DEFAULT_VAULT_EXCLUDE,
    DEFAULT_VAULT_PATH,
    DEFAULT_VAULT_REINDEX_INTERVAL_S,
    RemnantConfig,
    load_config,
    save_config,
)
from .db import RemnantDB, default_db_path, open_db
from .dream import day_dream, night_dream
from .echo import EchoService
from .echo_evaluate import EchoEvaluator
from .echo_worker import EchoWorker
from .embed import Embedder
from .extract import ExtractionWorker
from .identity import EffectiveIdentity, effective_identity
from .import_sources import import_hindsight, import_memory_store, source_profile_name
from .ingest import ingest_turn, store_memory
from .prefetch import prefetch as _run_prefetch
from .tools import TOOL_SCHEMAS, handle_tool_call
from .vault import index_vault as _index_vault

log = logging.getLogger("remnant")
__version__ = "0.3.1"


class _SessionEmbedder:
    """Per-session query-embedding cache wrapper.

    Wraps the real Embedder so the (single) Ollama query embedding for a given
    session is computed at most once and reused across all expanded query terms
    in a prefetch call. Falls back to the underlying embedder transparently.
    """

    def __init__(
        self,
        embedder: Embedder,
        query: str,
        qvec: list[float] | None = None,
        *,
        query_attempted: bool = False,
    ):
        self._embedder = embedder
        self._model = getattr(embedder, "_model", None)
        self._query = query
        self._qvec: list[float] | None = qvec
        self._query_attempted = query_attempted

    def embed(self, text: str) -> list[float] | None:
        # Reuse the cached query vector when the text matches the session query,
        # otherwise delegate to the real embedder (which has its own SQLite cache).
        # A None qvec means the query embedding failed upstream; we propagate
        # None so semantic search skips cosine comparison rather than zero-scoring.
        if text == self._query:
            # ``None`` is a meaningful cached failure.  Retrying here would
            # issue another blocking network request after prefetch has already
            # spent its embedding budget.
            if self._query_attempted:
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
    "Memory access is restricted to this profile; shared/fleet labels do not grant "
    "cross-profile access.\n"
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
    "Use `memory_import` with `source='memory_store'` to import MEMORY.md / "
    "USER.md bullets for the current Hermes profile as facts "
    "(confidence=0.9, trust_score=0.9) with fleet/shared/private visibility "
    "heuristics. Use `source='hindsight'` to issue a bounded set of broad recall "
    "queries to the Hindsight store and import unique results (trust_score=0.5). "
    "Both dedup by content hash. Use `dry_run=true` to preview counts without "
    "writing; `shadow=true` logs proposed actions to "
    "~/.hermes/remnant/shadow.log instead of importing.\n"
    "Transient state (percentages, current status, timestamps) is rejected.\n"
    "Memories are scoped by agent and visibility (private/shared/fleet).\n"
    "Use the `memory_thread` tool to manage topic threads: create, update, "
    "resolve, list, or sweep stale (threads inactive 14 days are marked stale). "
    "Threads capture ongoing conversations and dream-loop suggestions; they are "
    "never deleted.\n"
    "A bounded dream loop (day_dream / night_dream, invokable from a cron timer) "
    "finds non-obvious connections across memories and writes reflections to a "
    "private DREAMS.md diary; same-profile duplicates may be consolidated into "
    "memory. The loop pre-filters candidates locally and only ever sends a small "
    "bounded list to the cloud model.\n"
)

# Config schema exposed to `hermes memory setup`. Kept minimal: only fields a
# user must configure. Endpoints/models default to the BSL1 Ollama setup.
_CONFIG_SCHEMA: list[dict[str, Any]] = [
    {
        "key": "embed_url",
        "description": "Ollama embeddings endpoint",
        "default": "http://your-ollama-host.local:11434/api/embeddings",
        "required": False,
    },
    {
        "key": "embed_model",
        "description": "Embedding model name",
        "default": "nomic-embed-text",
        "required": False,
    },
    {
        "key": "embed_keep_alive",
        "description": "Ollama embedding model residency (finite duration recommended)",
        "default": "10m",
        "required": False,
    },
    {
        "key": "extract_url",
        "description": "Extraction LLM chat endpoint (Ollama /api/chat or OpenAI-compatible /v1)",
        "default": "http://your-ollama-host.local:11434/api/chat",
        "required": False,
    },
    {
        "key": "extract_model",
        "description": "Extraction LLM model name",
        "default": "gemma4:12b",
        "required": False,
    },
    {
        "key": "extract_keep_alive",
        "description": "Ollama extraction model residency (finite duration recommended)",
        "default": "2m",
        "required": False,
    },
    {
        "key": "extract_enabled",
        "description": "Enable async LLM extraction of facts",
        "default": True,
        "required": False,
    },
    {
        "key": "llm_protocol",
        "description": "Chat protocol; auto infers from the configured endpoint path",
        "default": "auto",
        "choices": ["auto", "ollama_native", "openai_compatible"],
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
    {
        "key": "prefetch_embedding_timeout_ms",
        "description": "Maximum milliseconds reserved for a prefetch query embedding",
        "default": 250,
        "required": False,
    },
    {
        "key": "runtime_identity_enabled",
        "description": (
            "Scope memories by stable Hermes runtime identity/workspace; enable only "
            "when the gateway supplies a stable platform user identity"
        ),
        "default": False,
        "type": "boolean",
        "required": False,
    },
    {
        "key": "structured_claim_extraction_v2",
        "description": "Preserve temporal and conditional claim metadata during extraction",
        "default": True,
        "type": "boolean",
        "required": False,
    },
    {
        "key": "claim_reconciliation_enabled",
        "description": "Classify duplicates, updates, conditions, and unresolved conflicts",
        "default": True,
        "type": "boolean",
        "required": False,
    },
    {
        "key": "claim_aware_ranking_enabled",
        "description": "Resolve claim evidence before selecting injected memories",
        "default": True,
        "type": "boolean",
        "required": False,
    },
    {
        "key": "ranking_profile",
        "description": "Versioned ranking profile used for claim-aware recall",
        "default": "claims-v1",
        "required": False,
    },
    {
        "key": "resolved_context_enabled",
        "description": "Render provenance-aware, prompt-injection-resistant context",
        "default": True,
        "type": "boolean",
        "required": False,
    },
    {
        "key": "recent_turn_overlay_enabled",
        "description": "Temporarily expose recent unprocessed turns for read-after-write recall",
        "default": True,
        "type": "boolean",
        "required": False,
    },
    {
        "key": "relation_evidence_enabled",
        "description": "Traverse only relations backed by active memory evidence",
        "default": True,
        "type": "boolean",
        "required": False,
    },
    {
        "key": "extract_num_ctx",
        "description": "Maximum context tokens reserved for one extraction request",
        "default": DEFAULT_EXTRACT_NUM_CTX,
        "required": False,
    },
    {
        "key": "extract_max_input_tokens",
        "description": "Maximum estimated input tokens sent to extraction",
        "default": DEFAULT_EXTRACT_MAX_INPUT_TOKENS,
        "required": False,
    },
    {
        "key": "extract_max_output_tokens",
        "description": "Maximum tokens generated by extraction",
        "default": DEFAULT_EXTRACT_MAX_OUTPUT_TOKENS,
        "required": False,
    },
    {
        "key": "extract_max_facts",
        "description": "Maximum durable facts accepted from one turn",
        "default": DEFAULT_EXTRACT_MAX_FACTS,
        "required": False,
    },
    {
        "key": "extract_think",
        "description": "Allow model reasoning during extraction",
        "default": DEFAULT_EXTRACT_THINK,
        "type": "boolean",
        "required": False,
    },
    {
        "key": "extract_structured_output",
        "description": "Enforce the extraction JSON schema at the model endpoint",
        "default": DEFAULT_EXTRACT_STRUCTURED_OUTPUT,
        "type": "boolean",
        "required": False,
    },
    {
        "key": "echo_enabled",
        "description": "Enable local outcome-aware Echo receipts and utility learning",
        "default": True,
        "type": "boolean",
        "required": False,
    },
    {
        "key": "echo_shadow_mode",
        "description": "Collect Echo evidence without changing visible ranking",
        "default": True,
        "type": "boolean",
        "required": False,
    },
    {
        "key": "echo_rank_influence",
        "description": "Capped Echo ranking influence; use 0 for shadow-only behavior",
        "default": 0.0,
        "required": False,
    },
    {
        "key": "echo_initial_sample_rate",
        "description": "Background counterfactual sample rate for new utility records",
        "default": 0.05,
        "required": False,
    },
    {
        "key": "echo_mature_sample_rate",
        "description": "Background counterfactual sample rate for mature records",
        "default": 0.005,
        "required": False,
    },
    {
        "key": "echo_max_evaluator_seconds_per_day",
        "description": "Daily local evaluator time budget for Echo",
        "default": 300,
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
        self._echo: EchoService | None = None
        self._echo_worker: EchoWorker | None = None
        self._session_id: str = ""
        self._hermes_home: str = ""
        self._started: bool = False
        # Phase 2: per-session injection + query-embedding caches.
        self._last_injected_hash: dict[str, str] = {}
        self._session_query_vec: dict[str, list[float]] = {}
        self._session_query: dict[str, tuple[str | None, str]] = {}
        self._prefetch_pending: set[tuple] = set()
        self._queued_prefetch: OrderedDict[
            tuple[str, str, str, str, str, int], tuple[float, dict[str, Any]]
        ] = OrderedDict()
        self._prefetch_executor: ThreadPoolExecutor | None = None
        self._prefetch_lock = threading.Lock()
        self._runtime_identity: dict[str, str] = {}
        self._effective_identity: EffectiveIdentity | None = None
        self._agent_context: str = "primary"
        self._memory_generation: int = 0
        # Optional Hermes-provided tokenizer callable. Standalone deployments
        # leave this unset and context compilation uses its conservative
        # deterministic fallback.
        self._token_counter: Any = None

    # -- lifecycle ------------------------------------------------------------

    @property
    def name(self) -> str:
        return "remnant"

    def is_available(self) -> bool:
        """Check local prerequisites without contacting Ollama or Hermes."""
        try:
            from .maintenance import availability_report

            return bool(availability_report(db_path=default_db_path())["available"])
        except (OSError, TypeError, ValueError):
            return False

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        hermes_home = kwargs.get("hermes_home")
        if not hermes_home:
            raise ValueError("initialize() requires hermes_home in kwargs")
        self._hermes_home = str(hermes_home)
        self._session_id = session_id or "default"
        self._config = load_config(self._hermes_home)
        supplied_counter = kwargs.get("token_counter")
        self._token_counter = supplied_counter if callable(supplied_counter) else None
        self._agent_context = str(kwargs.get("agent_context") or "primary")
        self._runtime_identity = {
            key: str(kwargs.get(key) or "").strip()
            for key in (
                "agent_identity", "agent_workspace", "user_id", "user_id_alt",
                "platform", "parent_session_id",
            )
            if kwargs.get(key)
        }
        self._effective_identity = effective_identity(
            configured_agent=self._config.agent_id,
            session_id=self._session_id,
            runtime_identity_enabled=self._config.runtime_identity_enabled,
            aliases=self._config.runtime_user_aliases,
            **kwargs,
        )
        self._config.agent_id = self._effective_identity.storage_key
        db_path = default_db_path()
        self._db = open_db(db_path)
        self._embedder = Embedder(self._db, self._config)
        self._echo = EchoService(self._db, self._config)
        self._echo.viewer_key = self._effective_identity.viewer_key
        if self._config.echo_enabled:
            self._echo_worker = EchoWorker(
                self._echo,
                self._config,
                evaluator=EchoEvaluator(self._config),
                model_busy=lambda: bool(self._worker and self._worker.active_jobs),
            )
            self._echo_worker.start()
        self._worker = ExtractionWorker(self._db, self._embedder, self._config)
        self._worker.start()
        self._prefetch_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="remnant-prefetch"
        )
        # Issue #13: sweep for turns that were stored but never extracted
        # (e.g. crash between insert_turn and enqueue_extraction) before the
        # worker starts draining the regular queue, then wake it to process.
        self._worker.queue_startup_sweep()
        self._worker.wake()
        self._started = True
        log.info("remnant %s initialized (profile=%s, home=%s, session=%s)",
                 __version__, self._config.agent_id, self._hermes_home, self._session_id)

    def shutdown(self) -> None:
        try:
            if self._echo_worker is not None:
                self._echo_worker.stop()
            if self._worker is not None:
                self._worker.stop()
            if self._echo is not None:
                self._echo.compact()
        finally:
            # Phase 4: no background vault watcher process is started here.
            # Re-index is driven by an external cron/timer calling
            # `reindex_vault()`; nothing to stop on shutdown.
            if self._prefetch_executor is not None:
                self._prefetch_executor.shutdown(wait=True, cancel_futures=True)
                self._prefetch_executor = None
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

    def identity_diagnostic(self) -> dict[str, str | bool]:
        """Return the effective mapping without exposing raw platform user IDs."""
        if self._effective_identity is None:
            return {"initialized": False}
        return {"initialized": True, **self._effective_identity.diagnostic()}

    # -- tools ----------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return list(TOOL_SCHEMAS)

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        if self._db is None or self._config is None or self._embedder is None:
            return json.dumps({"error": "provider not initialized"})
        session_id = kwargs.get("session_id", self._session_id)
        agent_id = self._config.agent_id
        result = handle_tool_call(
            tool_name,
            args,
            db=self._db,
            config=self._config,
            embedder=self._embedder,
            session_id=session_id,
            agent_id=agent_id,
            hermes_home=self._hermes_home,
            configured_profile=(self._effective_identity.configured_agent
                                if self._effective_identity else None),
            echo=self._echo,
        )
        # Hermes puts tool results directly into message content; the Ollama
        # cloud proxy rejects dict content ("invalid message content type:
        # map[string]interface{}"). Serialise to a JSON string so it matches
        # the wire format every other provider (Hindsight, etc.) uses.
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)

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
        if self._agent_context not in {"primary", ""}:
            log.debug("skipping memory write from agent context=%s", self._agent_context)
            return
        sid = session_id or self._session_id or "default"
        try:
            turn_id = ingest_turn(
                self._db,
                user_text=user_content or "",
                assistant_text=assistant_content or "",
                session_id=sid,
                agent_id=self._config.agent_id,
            )
        except Exception as e:
            log.warning("sync_turn failed: %s", e)
            return
        if self._echo is not None:
            self._echo.close_receipt(
                session_id=sid,
                viewer_key=(
                    self._effective_identity.viewer_key
                    if self._effective_identity is not None
                    else self._config.agent_id
                ),
                query=user_content or "",
                turn_id=turn_id,
            )
        self._invalidate_prefetch()
        # Wake the background worker so it picks up the new job promptly.
        if self._worker is not None:
            self._worker.wake()

    def _prefetch_key(
        self, query: str, session_id: str
    ) -> tuple[str, str, str, str, str, int]:
        cfg = self._config
        identity = (
            self._effective_identity.viewer_key
            if self._effective_identity is not None
            else (cfg.agent_id if cfg is not None else "default")
        )
        normalized = re.sub(r"\s+", " ", str(query or "").casefold()).strip()
        query_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        profile_json = json.dumps(
            sorted(cfg.profile_scope) if cfg is not None else [], separators=(",", ":")
        )
        profile_hash = hashlib.sha256(profile_json.encode("utf-8")).hexdigest()
        return (
            identity,
            session_id or "default",
            query_hash,
            profile_hash,
            cfg.embed_model if cfg is not None else "",
            self._memory_generation + (self._db.memory_generation if self._db else 0),
        )

    def _invalidate_prefetch(self) -> None:
        with self._prefetch_lock:
            self._memory_generation += 1
            self._queued_prefetch.clear()

    def _take_queued_prefetch(
        self, key: tuple[str, str, str, str, str, int]
    ) -> dict[str, Any] | None:
        now = time.monotonic()
        ttl = max(0.0, float(getattr(self._config, "prefetch_cache_ttl_s", 60)))
        with self._prefetch_lock:
            if ttl <= 0:
                self._queued_prefetch.clear()
                return None
            expired = [
                cache_key
                for cache_key, (created, _) in self._queued_prefetch.items()
                if now - created > ttl
            ]
            for cache_key in expired:
                self._queued_prefetch.pop(cache_key, None)
            entry = self._queued_prefetch.pop(key, None)
        return entry[1] if entry is not None else None

    def _store_queued_prefetch(
        self,
        key: tuple[str, str, str, str, str, int],
        result: dict[str, Any],
    ) -> None:
        maximum = max(1, int(getattr(self._config, "prefetch_cache_max_entries", 32)))
        with self._prefetch_lock:
            self._queued_prefetch[key] = (time.monotonic(), result)
            self._queued_prefetch.move_to_end(key)
            while len(self._queued_prefetch) > maximum:
                self._queued_prefetch.popitem(last=False)

    # -- prefetch (Phase 2) ---------------------------------------------------

    def _session_embedder(
        self, session_id: str, query: str, *, timeout_s: float | None = None,
    ) -> _SessionEmbedder | None:
        if self._embedder is None or self._config is None:
            return None
        key = (getattr(self._embedder, "_model", None), query)
        with self._prefetch_lock:
            if self._session_query.get(session_id) != key:
                self._session_query_vec.pop(session_id, None)
                self._session_query[session_id] = key
            cached = self._session_query_vec.get(session_id)
        if cached is None:
            try:
                embed = getattr(self._embedder, "embed_query", self._embedder.embed)
                try:
                    cached = embed(query, timeout=timeout_s)
                except TypeError:
                    cached = embed(query)
            except Exception:
                cached = None
            with self._prefetch_lock:
                # A concurrent foreground/background query cannot overwrite its successor.
                if cached and self._session_query.get(session_id) == key:
                    self._session_query_vec[session_id] = cached
                while len(self._session_query) > max(1, self._config.prefetch_cache_max_entries):
                    oldest = next(iter(self._session_query))
                    self._session_query.pop(oldest, None)
                    self._session_query_vec.pop(oldest, None)
        return _SessionEmbedder(self._embedder, query, qvec=cached, query_attempted=True)

    def prefetch(
        self, query: str, *, session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
    ) -> str:
        """Return safe context within the foreground database/retrieval budget."""
        if self._db is None or self._config is None or self._embedder is None:
            return ""
        started = time.monotonic()
        sid = session_id or self._session_id or "default"
        diagnostics: dict[str, Any] = {}
        result: dict[str, Any] = {}
        context = ""
        try:
            until = started + self._config.injection_prefetch_deadline_ms / 1000.0
            with self._db.deadline(until):
                key = self._prefetch_key(query, sid)
                queued = self._take_queued_prefetch(key)
                if queued and messages:
                    from .recall import _dedup_against_messages

                    items = queued.get("memories", [])
                    if len(_dedup_against_messages(items, messages)) != len(items):
                        queued = None
                if queued is not None:
                    result = queued
                    diagnostics.update(result.get("_diagnostics", {}))
                else:
                    result = _run_prefetch(self, query, sid, messages=messages,
                                           diagnostics=diagnostics)
                if key == self._prefetch_key(query, sid):
                    context = self._consume_prefetch_result(result)
                else:
                    diagnostics["reason"] = "evidence_changed"
        except (TimeoutError, sqlite3.Error):
            diagnostics["reason"] = "deadline"
        self._db.record_prefetch(
            sid, "injected" if context else "empty",
            reason=(diagnostics.get("reason") if context
                    else diagnostics.get("reason") or "suppressed"),
            elapsed_ms=(time.monotonic() - started) * 1000.0,
            result_count=len(result.get("memories", [])) if context else 0,
            token_estimate=int(result.get("token_estimate", 0)) if context else 0,
            query=query, agent_id=self._config.agent_id,
        )
        return context

    def _consume_prefetch_result(self, result: dict[str, Any]) -> str:
        """Suppress repeats and activate Echo only at actual context delivery."""
        context = str(result.get("context") or "")
        if not context:
            return ""
        sid = str(result.get("session_id") or self._session_id or "default")
        digest = hashlib.sha256(context.encode("utf-8")).hexdigest()
        with self._prefetch_lock:
            if self._last_injected_hash.get(sid) == digest:
                return ""
            self._last_injected_hash[sid] = digest
            while len(self._last_injected_hash) > 32:
                self._last_injected_hash.pop(next(iter(self._last_injected_hash)))
        draft = result.get("_echo_draft")
        if draft is not None and self._echo is not None:
            try:
                self._echo.activate_receipt(draft)
            except (TimeoutError, sqlite3.Error):
                # Optional attribution cannot discard already-compiled evidence.
                pass
        return context

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Coalesce bounded background work; never mark context delivered here."""
        if self._db is None or self._config is None or self._embedder is None:
            return
        sid = session_id or self._session_id or "default"
        try:
            with self._db.deadline(time.monotonic() + 0.01):
                key = self._prefetch_key(query, sid)
        except (TimeoutError, sqlite3.Error):
            return
        with self._prefetch_lock:
            if (key in self._queued_prefetch
                or any(pending[1] == sid for pending in self._prefetch_pending)
                or len(self._prefetch_pending) >= max(1, self._config.prefetch_cache_max_entries)):
                return
            self._prefetch_pending.add(key)

        def run() -> None:
            diagnostics: dict[str, Any] = {}
            try:
                until = time.monotonic() + self._config.injection_prefetch_deadline_ms / 1000.0
                with self._db.deadline(until):
                    result = _run_prefetch(self, query, sid, diagnostics=diagnostics)
                    if self._prefetch_key(query, sid) == key:
                        result["_diagnostics"] = diagnostics
                        self._store_queued_prefetch(key, result)
            except Exception:
                log.debug("queued prefetch skipped", exc_info=True)
            finally:
                with self._prefetch_lock:
                    self._prefetch_pending.discard(key)

        if self._prefetch_executor is not None:
            try:
                self._prefetch_executor.submit(run)
            except RuntimeError:
                with self._prefetch_lock:
                    self._prefetch_pending.discard(key)
        else:
            run()

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mirror built-in durable memory writes into Remnant."""
        if (
            action not in {"add", "replace", "remove"}
            or self._db is None
            or self._config is None
            or self._embedder is None
            or self._agent_context not in {"primary", ""}
        ):
            return
        meta = dict(metadata or {})
        if action in {"replace", "remove"}:
            old_content = str(meta.get("old_content") or content or "").strip()
            old = self._db.find_active_memory_by_content(
                old_content, agent_id=self._config.agent_id
            )
            if old is not None:
                self._db.deactivate_memory(str(old["id"]))
                self._invalidate_prefetch()
            if action == "remove":
                return
        if not content:
            return
        try:
            store_memory(
                self._db,
                self._embedder,
                self._config,
                fact=str(content).strip(),
                entity=str(meta.get("entity") or ("user" if target == "user" else "general")),
                session_id=str(meta.get("session_id") or self._session_id or "default"),
                agent_id=self._config.agent_id,
                visibility=str(meta.get("visibility") or self._config.default_visibility),
                source="manual",
                metadata={"write_origin": "builtin_memory", **meta},
            )
        except Exception:
            log.warning("failed to mirror built-in memory write", exc_info=True)
        else:
            self._invalidate_prefetch()
        if self._worker is not None:
            self._worker.wake()

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        """Offer a bounded recall block to Hermes' compressor."""
        if not messages:
            return ""
        query = ""
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                query = str(message.get("content") or "").strip()
                if query:
                    break
        if not query:
            return ""
        try:
            return self.prefetch(query, session_id=self._session_id)
        except Exception:
            log.debug("pre-compress recall failed", exc_info=True)
            return ""

    def on_delegation(
        self,
        task: str,
        result: str,
        *,
        child_session_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Persist the parent-side outcome of delegated work."""
        if self._agent_context not in {"primary", ""}:
            return
        self.sync_turn(
            f"Delegated task (child session {child_session_id or 'unknown'}): {task}",
            f"Delegated result: {result}",
            session_id=self._session_id,
        )

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        """Give already-queued turn extraction a short bounded flush window."""
        if self._worker is not None:
            self._worker.wait_until_idle(timeout_s=1.5, session_id=self._session_id)

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **_: Any,
    ) -> None:
        """Discard session-local recall state when Hermes rotates a session."""
        old_session_id = self._session_id
        self._session_id = new_session_id or "default"
        affected = {old_session_id, self._session_id, parent_session_id}
        if reset or rewound:
            affected.add(new_session_id)
        for sid in affected:
            self._last_injected_hash.pop(sid, None)
            self._session_query_vec.pop(sid, None)
            self._session_query.pop(sid, None)
        with self._prefetch_lock:
            self._queued_prefetch = OrderedDict(
                (key, value) for key, value in self._queued_prefetch.items()
                if key[1] not in affected
            )

    def backup_paths(self) -> list[str]:
        """Expose the shared database to Hermes' external-backup mechanism."""
        return [str(default_db_path())]

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

    # -- migration import (Phase 6) ------------------------------------------

    def import_memory(
        self,
        source: str,
        *,
        profile: str | None = None,
        dry_run: bool = False,
        shadow: bool = False,
    ) -> dict[str, Any]:
        """Import memories from an existing store.

        ``source`` is ``"memory_store"`` (the current profile's MEMORY.md / USER.md),
        ``"hindsight"`` (bounded broad-query recall), or ``"vault"``
        (delegates to ``reindex_vault``). Returns a stats dict. Safe to call
        from an external cron/timer.
        """
        if self._db is None or self._config is None or self._embedder is None:
            return {"error": "provider not initialized"}
        source_profile = source_profile_name(
            self._hermes_home,
            self._effective_identity.configured_agent
            if self._effective_identity else self._config.agent_id,
        )
        if profile is not None and profile != source_profile:
            return {"error": "imports are restricted to the current profile"}
        profile = source_profile
        if source == "vault":
            return self.reindex_vault()
        if source not in ("memory_store", "hindsight"):
            return {"error": f"unknown import source: {source}"}
        if source == "memory_store":
            return import_memory_store(
                self._db, self._config, self._embedder, self._hermes_home,
                dry_run=dry_run, shadow=shadow, profile=profile,
            )
        return import_hindsight(
            self._db, self._config, self._embedder,
            dry_run=dry_run, shadow=shadow, hermes_home=self._hermes_home,
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
