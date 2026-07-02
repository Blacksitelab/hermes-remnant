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

from .config import RemnantConfig, load_config, save_config
from .db import RemnantDB, open_db
from .embed import Embedder
from .extract import ExtractionWorker
from .ingest import ingest_turn
from .tools import TOOL_SCHEMAS, handle_tool_call

log = logging.getLogger("remnant")

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
    "Use the `memory_search` tool to recall facts (BM25 keyword search).\n"
    "Use the `memory_store` tool to save a durable fact explicitly.\n"
    "Transient state (percentages, current status, timestamps) is rejected.\n"
    "Memories are scoped by agent and visibility (private/shared/fleet).\n"
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


def register(ctx: Any) -> None:
    """Plugin entry point discovered by Hermes."""
    ctx.register_memory_provider(RemnantMemoryProvider())


__all__ = ["RemnantMemoryProvider", "register"]
