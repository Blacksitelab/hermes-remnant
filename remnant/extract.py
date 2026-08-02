"""Async extraction worker.

- Runs in a `ThreadPoolExecutor` (single worker by default) so `sync_turn`
  stays non-blocking.
- Pulls jobs from the persisted `extraction_queue` table; restarts don't lose
  turns.
- Calls gemma4:12b on the BSL1 OpenAI-compatible endpoint and parses facts +
  entities from the JSON response.
- Runs each extracted fact through the transient-state filter and dedup before
  storing.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any

import httpx

from .config import RemnantConfig
from .db import RemnantDB
from .embed import Embedder
from .entity import _COMMON_NOUNS, _FUNCTION_WORDS, _STOPLIST, _STOPWORDS
from .ingest import is_transient, store_memory
from .llm import LLMResponseError, chat

log = logging.getLogger("remnant.extract")

# Lowercased view of ``entity._STOPWORDS`` so the typed-entity filter can match
# case-insensitively (the source set stores title-cased forms like ``"The"``).
# Per the issue #10 spec the typed filter reuses both ``_STOPWORDS`` and
# ``_STOPLIST``; the lowercased view also covers title-cased entries.
# ``"Remnant"`` (the system's own name) is dropped from the regex stopword set
# here: the regex path stops it to avoid over-extracting it from prose, but
# when the LLM *explicitly* names ``Remnant`` as a typed entity it is a real
# durable subject and should survive (issue #10 test contract).
_STOPWORDS_LOWER: set[str] = {
    w.lower() for w in _STOPWORDS if w.lower() != "remnant"
}

# Common English function words and generic response words the LLM sometimes
# emits as "entities". Imported from ``entity`` (issue #22) so the typed-entity
# filter and the regex fallback share the same deny-list.

_EXTRACT_PROMPT = (
    "You are a memory extraction engine. Read the conversation turn and "
    "extract durable facts worth remembering long-term.\n"
    "\n"
    "Rules:\n"
    "- Only extract stable, long-lived facts (preferences, identity, projects, "
    "relationships, owned things, recurring context).\n"
    "- DO NOT extract transient state (current percentages, right-now status, "
    '"is at", "currently", "today", timestamps).\n'
    "- One fact per line, as a complete declarative sentence.\n"
    "- Also list the entities each fact is about, with a type drawn from: "
    "person, service, project, concept, place, tool.\n"
    "- Include aliases (alternate names / spellings) for each entity.\n"
    "\n"
    "Return STRICT JSON only, no prose:\n"
    '{"facts": [{"entity": "<primary entity name>", "fact": "<one-sentence fact>", '
    '"visibility": "private", "entities": [{"name": "<name>", "type": "<type>", '
    '"aliases": ["<alias>", ...]}]}]}\n'
    "\n"
    "Visibility must be one of: private, shared, fleet. Default to private.\n"
    "\n"
    "Entity names must be proper nouns, specific system names, or named "
    "concepts — NOT common English words. Do NOT extract function words "
    "(articles, prepositions, conjunctions, pronouns, generic verbs), "
    "stopwords, or short noise tokens. If you are unsure whether a word is a "
    "real entity, omit it."
)

_STOP_RE = re.compile(
    r"\b(currently|now|is at|today|tonight|this morning|right now)\b",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(r"\b\d{1,3}\s*%\b|\bpercent\b", re.IGNORECASE)
_TIME_RE = re.compile(
    r"\b\d{1,2}:\d{2}\b|\b(am|pm)\b", re.IGNORECASE
)


class ExtractionWorker:
    """Background worker that drains the extraction_queue."""

    def __init__(
        self,
        db: RemnantDB,
        embedder: Embedder,
        config: RemnantConfig,
    ):
        self._db = db
        self._embedder = embedder
        self._config = config
        self._client = httpx.Client(timeout=config.extract_timeout)
        self._stop = threading.Event()
        self._executor: Any = None
        self._future: Any = None
        self._wake = threading.Event()
        # Issue #13: startup sweep re-enqueues turns that were stored but never
        # extracted (e.g. crash between insert_turn and enqueue_extraction).
        self._pending_startup: list[dict[str, Any]] = []
        self._startup_done: threading.Event = threading.Event()

    def queue_startup_sweep(self) -> None:
        """Populate the pending-startup list from unextracted turns.

        Safe to call once at provider initialize(); the worker loop drains it
        on its first iteration before handling the regular queue.
        """
        try:
            recovered = self._db.recover_stale_extractions()
            if recovered:
                log.info("recovered %d stale extraction jobs", recovered)
            self._pending_startup = self._db.get_unextracted_turns(
                self._config.agent_id, limit=200
            )
            log.info(
                "extraction startup sweep found %d unextracted turns",
                len(self._pending_startup),
            )
        except Exception as e:
            log.warning("extraction startup sweep failed: %s", e)
            self._pending_startup = []

    def _enqueue_startup(self) -> None:
        """Enqueue every pending startup turn via the persisted queue."""
        if not self._pending_startup:
            return
        for turn in self._pending_startup:
            try:
                self._db.enqueue_extraction(
                    turn_id=int(turn["id"]),
                    session_id=turn["session_id"],
                    agent_id=turn["agent_id"],
                    user_text=turn["user_text"],
                    assistant_text=turn["assistant_text"],
                )
            except Exception as e:
                log.warning(
                    "startup sweep enqueue failed turn_id=%s: %s",
                    turn.get("id"), e,
                )
        self._pending_startup = []

    def start(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="remnant-extract"
        )
        self._future = self._executor.submit(self._loop)

    def wake(self) -> None:
        """Wake the worker loop to check for new jobs."""
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.clear()
            try:
                if not self._startup_done.is_set():
                    self._enqueue_startup()
                    self._startup_done.set()
                self._drain()
            except Exception as e:
                log.warning("extraction loop error: %s", e)
            # Wait for wake or poll interval
            self._wake.wait(timeout=2.0)

    def _drain(self) -> None:
        while not self._stop.is_set():
            job = self._db.claim_next_extraction(self._config.agent_id)
            if job is None:
                return
            try:
                fact_count = self._process(job) or 0
                self._db.complete_extraction(int(job["id"]), fact_count=int(fact_count))
            except Exception as e:
                log.warning("extraction failed for turn %s: %s", job["turn_id"], e)
                self._db.fail_extraction(int(job["id"]), error=str(e))

    def _process(self, job: dict[str, Any]) -> int:
        t0 = time.perf_counter()
        facts = self._extract(job["user_text"], job["assistant_text"])
        for f in facts:
            fact_text = f.get("fact", "").strip()
            entity = f.get("entity", "").strip() or "general"
            visibility = f.get("visibility", self._config.default_visibility)
            if not fact_text:
                continue
            if is_transient(fact_text):
                log.debug("rejected transient fact: %s", fact_text)
                continue
            # Typed entities from the LLM parse (optional). When present these
            # drive entity-graph linking + relation seeding. We keep the
            # legacy single `entity` subject as the tag/metadata subject.
            typed_entities = f.get("entities") or []
            if not typed_entities and entity and entity != "general":
                typed_entities = [{"name": entity, "type": None, "aliases": []}]
            typed_entities = filter_typed_entities(typed_entities)
            store_memory(
                self._db,
                self._embedder,
                self._config,
                fact=fact_text,
                entity=entity,
                session_id=job["session_id"],
                agent_id=job["agent_id"],
                visibility=visibility,
                source_turn_id=int(job["turn_id"]),
                entities=typed_entities,
                source_text=fact_text,
            )
        duration_ms = (time.perf_counter() - t0) * 1000.0
        log.info(
            "extraction complete turn_id=%s facts=%s duration_ms=%.1f",
            job["turn_id"], len(facts), duration_ms,
        )
        return len(facts)

    def _extract(self, user_text: str, assistant_text: str) -> list[dict[str, Any]]:
        if not self._config.extract_enabled:
            return []
        content = f"USER: {user_text}\nASSISTANT: {assistant_text}"
        t0 = time.perf_counter()
        try:
            text = chat(
                url=self._config.extract_url,
                model=self._config.extract_model,
                system=_EXTRACT_PROMPT,
                user=content,
                timeout=self._config.extract_timeout,
                protocol=getattr(self._config, "llm_protocol", None),
                temperature=0.1,
                max_tokens=4096,
                client=self._client,
            )
            facts = _parse_facts(text)
            duration_ms = (time.perf_counter() - t0) * 1000.0
            log.debug(
                "extraction LLM call succeeded duration_ms=%.1f facts=%s",
                duration_ms, len(facts),
            )
            return facts
        except (httpx.HTTPError, LLMResponseError, ValueError, json.JSONDecodeError) as e:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            log.warning(
                "extraction failed duration_ms=%.1f error=%s",
                duration_ms, e,
            )
            raise

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
        self._client.close()


def _parse_facts(text: str) -> list[dict[str, Any]]:
    """Parse the LLM response. Tolerates trailing prose / fenced blocks."""
    # Try direct JSON first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "facts" in obj:
            return obj["facts"]
    except json.JSONDecodeError:
        pass
    # Fall back to the first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict) and "facts" in obj:
                return obj["facts"]
        except json.JSONDecodeError:
            pass
    return []


def filter_typed_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter typed entities returned by the extraction LLM (issue #10/#22).

    The model can return function words, stopwords, and short noise tokens as
    entities. This helper mirrors the regex path's stopword filter so only
    durable proper nouns / named concepts reach the entity graph. When more than
    ``_MAX_TYPED_ENTITIES`` (15) survive, the list is capped to the top-N by a
    simple salience heuristic (name length + alias count) to avoid graph bloat.

    Rules (lowercased ``name`` unless noted):
      - drop empties (``name.strip() == ""``);
      - drop when present in ``entity._STOPWORDS`` or ``entity._STOPLIST``;
      - drop when ``len(name) < 3`` unless the original token is title-cased
        (so short acronyms / proper-noun initials such as ``"AI"`` survive
        while ``"no"`` does not).
    Order and alias lists are preserved. Duplicates (case-insensitive) are
    removed after the first occurrence.
    """
    if not entities:
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ent in entities:
        raw = (ent.get("name") or "").strip()
        if not raw or len(raw.strip()) == 0:
            continue
        key = raw.lower()
        if (
            key in _STOPWORDS_LOWER
            or key in _STOPLIST
            or key in _FUNCTION_WORDS
            or key in _COMMON_NOUNS
        ):
            continue
        # Drop short lowercase noise; keep all-caps acronyms (e.g. ``AI``)
        # and title-cased proper-noun initials (e.g. ``Ai``).
        if len(key) < 3 and not (raw.isupper() or raw.istitle()):
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(ent)

    # Issue #22: cap to the top 15 typed entities by simple salience.
    if len(out) > _MAX_TYPED_ENTITIES:
        scored: list[tuple[float, dict[str, Any]]] = []
        for ent in out:
            name = (ent.get("name") or "").strip()
            aliases = ent.get("aliases") or []
            score = len(name.split()) + 0.5 * len(aliases) + len(name) * 0.05
            scored.append((score, ent))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = [ent for _, ent in scored[:_MAX_TYPED_ENTITIES]]
    return out


_MAX_TYPED_ENTITIES = 15


__all__ = ["ExtractionWorker", "filter_typed_entities"]
