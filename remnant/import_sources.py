"""Migration phase: import existing durable facts into Remnant.

Two sources are supported:

- ``memory_store``: MEMORY.md / USER.md files across all Hermes profiles under
  ``~/.hermes/profiles/``. Each bullet/numbered line becomes a fact with
  ``source='import'``, ``confidence=0.9``, ``trust_score=0.9``. Visibility is
  assigned by a keyword heuristic (fleet / shared / private).
- ``hindsight``: a bounded set of broad ``hindsight_recall`` queries pulls
  stored memories, which become facts with ``source='hindsight'``,
  ``trust_score=0.5`` and default ``private`` visibility.

Both paths dedup by content hash (sha256 of the normalized text): a duplicate
bumps ``seen_count`` on the existing memory instead of inserting a new row.

Shadow mode (``shadow=True``) writes a JSON-line record per proposed action to
``<hermes_home>/remnant/shadow.log`` instead of touching the DB — for human
comparison against Hindsight's actual injection.

``dry_run=True`` performs parsing, extraction, and dedup simulation but writes
nothing (no memories, no audit_log, no shadow log).

Token efficiency: memory_store entries are short bullets and are stored
one-per-row; hindsight recall is bounded to a small fixed set of broad queries
and a hard cap on total imported rows.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .config import RemnantConfig
from .db import RemnantDB
from .embed import Embedder
from .entity import extract_entities, link_memory_entities

log = logging.getLogger("remnant.import_sources")

# -- visibility heuristic ----------------------------------------------------

# Fleet: user-profile facts (name, timezone, preferences) — every agent sees.
_FLEET_KEYWORDS = (
    "timezone", "name", "email", "prefers", "hates", "wants",
    "language", "region", "location",
)
# Shared: cross-agent project/hardware facts — fleet-visible project context.
_SHARED_KEYWORDS = (
    "project", "repo", "hardware", "server", "network", "decision",
    "agreed", "plan", "build",
)
# Private: agent-specific relationship/working-style facts — local only.
_PRIVATE_KEYWORDS = (
    "relationship", "style", "notes", "manner", "habit", "personal",
)

# Bounded hindsight recall. A small fixed set of broad queries keeps token and
# network cost predictable; we dedup by content hash and stop after the cap.
_DEFAULT_HINDSIGHT_QUERIES = (
    "project", "preference", "decision", "person", "tool",
)
HINDSIGHT_QUERY_LIMIT = 25  # per-query result cap
HINDSIGHT_TOTAL_CAP = 200   # hard cap on total imported rows


def _content_hash(text: str) -> str:
    """sha256 hex of the normalized text. Stable across whitespace variants."""
    s = re.sub(r"\s+", " ", (text or "").strip().lower())
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# -- memory_store discovery + parsing --------------------------------------

_MEMORY_FILES = ("MEMORY.md", "USER.md")


def discover_memory_store_entries(
    hermes_home: str | Path,
) -> Iterator[tuple[str, str, str]]:
    """Yield ``(profile, file_path, content_line)`` for MEMORY.md / USER.md in
    every profile under ``<hermes_home>/profiles/``.

    ``content_line`` is the raw markdown line (bullet/numbered item or plain
    text line) stripped of surrounding whitespace. Headers, fenced code, and
    blank lines are skipped. Profiles without these files are skipped silently.
    """
    profiles_root = Path(hermes_home) / "profiles"
    if not profiles_root.is_dir():
        return
    for profile_dir in sorted(profiles_root.iterdir(), key=lambda p: p.name):
        if not profile_dir.is_dir():
            continue
        profile = profile_dir.name
        for fname in _MEMORY_FILES:
            fpath = profile_dir / fname
            if not fpath.is_file():
                continue
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                log.warning("import: cannot read %s: %s", fpath, e)
                continue
            for line in _iter_entry_lines(text):
                if line.strip():
                    yield profile, str(fpath), line.strip()


def _iter_entry_lines(text: str) -> Iterator[str]:
    """Yield raw content lines (potential entries) from a markdown file.

    Skips YAML frontmatter fences, ATX headers (``#``), horizontal rules, and
    blank lines. Bullet and numbered markers are preserved here; stripping
    happens in ``parse_memory_file``.
    """
    in_frontmatter = False
    fenced = False
    for raw in text.splitlines():
        s = raw.rstrip()
        stripped = s.strip()
        if not stripped:
            continue
        # YAML frontmatter: a leading ``---`` opens, the next ``---``/``...``
        # closes. Lines inside are skipped.
        if stripped == "---" or stripped == "...":
            if not in_frontmatter and not fenced and s == stripped:
                # opening fence only counts at the very top of the file; we
                # approximate by toggling on the first occurrence.
                in_frontmatter = not in_frontmatter
            else:
                in_frontmatter = False
            continue
        if in_frontmatter:
            continue
        # Skip fenced code blocks.
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fenced = not fenced
            continue
        if fenced:
            continue
        # Skip ATX headers and horizontal rules.
        if stripped.startswith("#"):
            continue
        if stripped in ("---", "***", "___") or re.match(r"^-{3,}$", stripped):
            continue
        yield s


def parse_memory_file(text: str) -> list[str]:
    """Parse a MEMORY.md / USER.md body into a list of cleaned entry strings.

    Recognizes markdown bullets (``-``, ``*``, ``+``) and numbered items
    (``1.``, ``2.`` …). Continuation lines are not merged; each bullet/numbered
    line is one entry. Inline markdown emphasis (``*``, ``_``, ``**``) is
    stripped. Whitespace is collapsed. Empty results are dropped.
    """
    entries: list[str] = []
    seen: set[str] = set()
    bullet_re = re.compile(r"^\s*([-*+]|[0-9]+\.)\s+(.*)$")
    for raw in _iter_entry_lines(text):
        m = bullet_re.match(raw)
        body = m.group(2) if m else raw
        cleaned = _clean_inline_md(body)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        entries.append(cleaned)
    return entries


def _clean_inline_md(text: str) -> str:
    """Strip inline markdown emphasis/links, keep the visible text."""
    # Markdown links: ``[label](url)`` -> ``label``.
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text or "")
    # Inline code: ```x``` -> ``x``.
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Bold/italic emphasis: strip ``*`` / ``_`` wrappers.
    text = re.sub(r"\*{1,3}([^*\s][^*]*?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_\s][^_]*?)_{1,3}", r"\1", text)
    return text.strip()


def classify_visibility(text: str, agent: str | None = None) -> str:
    """Heuristic visibility assignment: ``fleet`` | ``shared`` | ``private``.

    Fleet keywords win first (user-profile facts every agent should see), then
    shared (cross-agent project/hardware context), then private (agent-specific
    relationship/working-style). Defaults to ``private``.
    """
    t = (text or "").lower()
    if any(kw in t for kw in _FLEET_KEYWORDS):
        return "fleet"
    if any(kw in t for kw in _SHARED_KEYWORDS):
        return "shared"
    if any(kw in t for kw in _PRIVATE_KEYWORDS):
        return "private"
    return "private"


# -- shadow log -------------------------------------------------------------

def _shadow_log_path(hermes_home: str | Path) -> Path:
    return Path(hermes_home) / "remnant" / "shadow.log"


def write_shadow_log(hermes_home: str | Path, record: dict[str, Any]) -> Path:
    """Append a JSON-line record to ``<hermes_home>/remnant/shadow.log``.

    Creates the parent directory if needed. Returns the log path.
    """
    p = _shadow_log_path(hermes_home)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str, sort_keys=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return p


# -- import: memory_store ---------------------------------------------------

def _token_estimate(text: str) -> int:
    """Rough token estimate (~4 chars/token). Used only for the shadow log."""
    return max(1, len(text or "") // 4)


def import_memory_store(
    db: RemnantDB,
    config: RemnantConfig,
    embedder: Embedder,
    hermes_home: str | Path,
    *,
    dry_run: bool = False,
    shadow: bool = False,
    profile: str | None = None,
) -> dict[str, Any]:
    """Import MEMORY.md / USER.md entries across all (or one) Hermes profiles.

    Returns a stats dict::

        {
          "source": "memory_store",
          "discovered": int,    # parsed entries considered
          "imported": int,      # new memories written
          "duplicates": int,   # matched an existing content_hash
          "skipped": int,       # transient/empty rejected
          "visibility": {"fleet": n, "shared": n, "private": n},
          "dry_run": bool,
          "shadow": bool,
        }

    ``dry_run`` performs parsing + dedup simulation but writes nothing.
    ``shadow`` writes a JSON-line per proposed action to the shadow log instead
    of touching the DB (overrides dry_run=False semantics for writes; dry_run
    still wins when both are set).
    """
    stats = {
        "source": "memory_store",
        "discovered": 0,
        "imported": 0,
        "duplicates": 0,
        "skipped": 0,
        "visibility": {"fleet": 0, "shared": 0, "private": 0},
        "dry_run": bool(dry_run),
        "shadow": bool(shadow),
    }
    actor = config.agent_id

    for prof, fpath, raw_line in discover_memory_store_entries(hermes_home):
        if profile is not None and prof != profile:
            continue
        # Re-parse the single line through parse_memory_file so bullet markers
        # and inline markdown are stripped consistently with a full file.
        entries = parse_memory_file(raw_line)
        if not entries:
            stats["skipped"] += 1
            continue
        for entry in entries:
            if not entry.strip():
                stats["skipped"] += 1
                continue
            stats["discovered"] += 1
            vis = classify_visibility(entry, agent=prof)
            stats["visibility"][vis] += 1
            chash = _content_hash(entry)
            existing = db.get_memory_by_content_hash(chash)
            duplicate = existing is not None
            if duplicate:
                stats["duplicates"] += 1
            else:
                stats["imported"] += 1

            if dry_run:
                continue

            if shadow:
                write_shadow_log(hermes_home, {
                    "ts": _now_iso(),
                    "source": "memory_store",
                    "profile": prof,
                    "file": fpath,
                    "action": "duplicate" if duplicate else "import",
                    "content": entry,
                    "content_hash": chash,
                    "visibility": vis,
                    "duplicate_of": existing["id"] if existing else None,
                    "token_estimate": _token_estimate(entry),
                })
                continue

            if duplicate:
                db.increment_seen_count(existing["id"])
                continue

            embedding = embedder.embed(entry) if embedder else None
            embed_model = getattr(embedder, "_model", None) if embedder else None
            meta: dict[str, Any] = {
                "imported_from": fpath,
                "profile": prof,
                "content_hash": chash,
            }
            mid = db.insert_memory(
                content=entry,
                source="import",
                source_id=fpath,
                agent=actor,
                visibility=vis,
                type="fact",
                confidence=0.9,
                trust_score=0.9,
                content_hash=chash,
                metadata=meta,
                embedding=embedding or None,
                embed_model=embed_model,
            )
            if mid:
                ents = extract_entities(entry)
                if ents:
                    link_memory_entities(db, memory_id=mid, entities=ents, agent_id=actor)
                db.write_audit(
                    actor=actor,
                    action="import",
                    memory_id=mid,
                    details={"source": "memory_store", "profile": prof, "file": fpath},
                )
    return stats


# -- import: hindsight ------------------------------------------------------

_HINDSIGHT_BASE_URL = os.environ.get("HINDSIGHT_BASE_URL", "http://127.0.0.1:9514")
_HINDSIGHT_BANK_ID = os.environ.get("HINDSIGHT_BANK_ID", "hermes-claire")


def _hindsight_recall(query: str, *, limit: int) -> list[dict[str, Any]]:
    """Call the Hindsight recall API via the Python client.

    Uses ``HindsightClient`` (the installed ``hindsight`` package's client class)
    to query the local Hindsight server. Returns a list of
    ``{"content": str, "type": str, "id": str}`` dicts.

    The server URL and bank ID are configurable via ``HINDSIGHT_BASE_URL`` and
    ``HINDSIGHT_BANK_ID`` env vars (defaults: ``http://127.0.0.1:9514`` and
    ``hermes-claire``).
    """
    try:
        from hindsight import HindsightClient  # type: ignore[import]
    except Exception as e:  # pragma: no cover - tests monkeypatch this fn
        log.warning("import: HindsightClient unavailable: %s", e)
        return []
    try:
        client = HindsightClient(base_url=_HINDSIGHT_BASE_URL)
        resp = client.recall(
            bank_id=_HINDSIGHT_BANK_ID,
            query=query,
            max_tokens=4096,
        )
        results = []
        for r in (resp.results or []):
            results.append({
                "id": r.id,
                "type": r.type,
                "content": r.text,
            })
        client.close()
        return results
    except Exception as e:
        log.warning("import: hindsight recall failed for query %r: %s", query, e)
        return []


def import_hindsight(
    db: RemnantDB,
    config: RemnantConfig,
    embedder: Embedder,
    *,
    queries: list[str] | None = None,
    dry_run: bool = False,
    shadow: bool = False,
    hermes_home: str | Path | None = None,
) -> dict[str, Any]:
    """Import memories from the Hindsight store via a bounded set of broad
    queries.

    Each unique content becomes a fact with ``source='hindsight'``,
    ``trust_score=0.5``, default ``private`` visibility. Dedup is by content
    hash (sha256 of normalized text). Stops after ``HINDSIGHT_TOTAL_CAP`` new
    rows in a single call.

    Returns a stats dict::

        {
          "source": "hindsight",
          "queries": int,          # queries actually issued
          "recalled": int,          # raw rows returned by hindsight_recall
          "discovered": int,        # unique after content-hash dedup
          "imported": int,          # new memories written
          "duplicates": int,       # matched existing content_hash
          "skipped": int,           # empty/unparseable
          "capped": bool,           # hit the total-cap
          "dry_run": bool,
          "shadow": bool,
        }
    """
    qs = list(queries) if queries else list(_DEFAULT_HINDSIGHT_QUERIES)
    stats = {
        "source": "hindsight",
        "queries": 0,
        "recalled": 0,
        "discovered": 0,
        "imported": 0,
        "duplicates": 0,
        "skipped": 0,
        "capped": False,
        "dry_run": bool(dry_run),
        "shadow": bool(shadow),
    }
    actor = config.agent_id
    seen_hashes: set[str] = set()
    imported = 0

    for q in qs:
        stats["queries"] += 1
        rows = _hindsight_recall(q, limit=HINDSIGHT_QUERY_LIMIT)
        for row in rows:
            stats["recalled"] += 1
            content = _extract_content(row)
            if not content:
                stats["skipped"] += 1
                continue
            chash = _content_hash(content)
            if chash in seen_hashes:
                stats["duplicates"] += 1
                continue
            seen_hashes.add(chash)
            stats["discovered"] += 1

            existing = db.get_memory_by_content_hash(chash)
            duplicate = existing is not None
            if duplicate:
                stats["duplicates"] += 1
            else:
                stats["imported"] += 1
                imported += 1

            if dry_run:
                if imported >= HINDSIGHT_TOTAL_CAP:
                    stats["capped"] = True
                    break
                continue

            if shadow and hermes_home is not None:
                write_shadow_log(hermes_home, {
                    "ts": _now_iso(),
                    "source": "hindsight",
                    "query": q,
                    "action": "duplicate" if duplicate else "import",
                    "content": content,
                    "content_hash": chash,
                    "duplicate_of": existing["id"] if existing else None,
                    "token_estimate": _token_estimate(content),
                })
            elif duplicate:
                db.increment_seen_count(existing["id"])
            else:
                embedding = embedder.embed(content) if embedder else None
                embed_model = getattr(embedder, "_model", None) if embedder else None
                meta: dict[str, Any] = {
                    "hindsight_query": q,
                    "content_hash": chash,
                }
                mid = db.insert_memory(
                    content=content,
                    source="hindsight",
                    source_id=q,
                    agent=actor,
                    visibility="private",
                    type="fact",
                    confidence=0.5,
                    trust_score=0.5,
                    content_hash=chash,
                    metadata=meta,
                    embedding=embedding or None,
                    embed_model=embed_model,
                )
                if mid:
                    ents = extract_entities(content)
                    if ents:
                        link_memory_entities(
                            db, memory_id=mid, entities=ents, agent_id=actor
                        )
                    db.write_audit(
                        actor=actor,
                        action="import",
                        memory_id=mid,
                        details={"source": "hindsight", "query": q},
                    )

            if imported >= HINDSIGHT_TOTAL_CAP:
                stats["capped"] = True
                break
        if stats["capped"]:
            break
    return stats


def _extract_content(row: dict[str, Any]) -> str:
    """Pull a usable content string from a Hindsight recall row.

    Hindsight rows vary in shape; accept the common keys in priority order and
    fall back to a JSON dump. Returns '' when nothing usable is found (e.g. an
    empty dict), so the caller can skip it.
    """
    if not isinstance(row, dict):
        return ""
    for key in ("content", "text", "memory", "body", "summary", "note"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # An empty dict has nothing importable; skip it rather than emitting "{}".
    if not row:
        return ""
    # Last resort: stringify the row so it is still importable + dedupable.
    try:
        s = json.dumps(row, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return ""
    return s


def _now_iso() -> str:
    import time as _time

    return _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())


__all__ = [
    "discover_memory_store_entries",
    "parse_memory_file",
    "classify_visibility",
    "import_memory_store",
    "import_hindsight",
    "write_shadow_log",
    "HINDSIGHT_QUERY_LIMIT",
    "HINDSIGHT_TOTAL_CAP",
]
