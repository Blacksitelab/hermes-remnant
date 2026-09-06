"""Thread management (Phase 5).

Topic threads capture ongoing conversations and dream-loop suggestions. They
are stored in the `threads` table (see ``remnant.db``). Threads are never
deleted: inactive ones are swept to ``status='stale'`` and resolved ones to
``status='resolved'``.

The stale sweep is idempotent and safe to call from a cron/timer.
"""

from __future__ import annotations

import logging
from typing import Any

from .db import RemnantDB

log = logging.getLogger("remnant.threads")

# Threads inactive for this many days are considered stale.
STALE_DAYS = 14


def create_thread(
    db: RemnantDB,
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
    """Create a thread and return its id."""
    return db.insert_thread(
        title=title,
        topic=topic,
        importance=importance,
        tags=tags,
        related_entities=related_entities,
        source=source,
        added_by=added_by,
        owner=owner,
    )


def update_thread(
    db: RemnantDB,
    thread_id: str,
    *,
    title: str | None = None,
    status: str | None = None,
    importance: float | None = None,
    tags: list[str] | None = None,
    related_entities: list[str] | None = None,
    touch: bool = True,
) -> dict[str, Any] | None:
    return db.update_thread(
        thread_id,
        title=title,
        status=status,
        importance=importance,
        tags=tags,
        related_entities=related_entities,
        touch=touch,
    )


def resolve_thread(db: RemnantDB, thread_id: str) -> dict[str, Any] | None:
    return db.resolve_thread(thread_id)


def list_threads(
    db: RemnantDB, *, status: str | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    return db.list_threads(status=status, limit=limit)


def stale_threads(db: RemnantDB, *, days: int = STALE_DAYS) -> list[dict[str, Any]]:
    return db.stale_threads(days=days)


def sweep_stale_threads(db: RemnantDB, *, days: int = STALE_DAYS) -> list[str]:
    """Mark threads inactive for `days` as stale. Returns the marked ids.

    Idempotent: a thread already stale/resolved is left alone.
    """
    marked = db.sweep_stale_threads(days=days)
    if marked:
        log.info("swept %d stale threads (>%d days inactive)", len(marked), days)
    return marked


__all__ = [
    "STALE_DAYS",
    "create_thread",
    "update_thread",
    "resolve_thread",
    "list_threads",
    "stale_threads",
    "sweep_stale_threads",
]
