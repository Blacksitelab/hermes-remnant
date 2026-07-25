"""Vault indexing (Phase 4).

Walks the Obsidian vault (markdown notes), parses YAML frontmatter, hashes
file contents with sha256, and indexes each note as a ``type='document'``
memory with ``source='vault'`` and ``source_id=<relative path>``. Hash-based
change detection means only changed/new files are re-indexed; deleted files
are marked ``forgotten`` (row preserved — nothing is ever deleted).

Exclusion patterns match the first path component (top-level vault folder), so
the agent scratch trees ``90_*``–``95_*`` and ``99_ARCHIVE/`` are never walked.

Frontmatter fields stored in memory metadata: ``type``, ``tags``, ``status``,
``created``, ``updated``, ``author``, ``locked``. Locked notes
(``locked: true``) are indexed normally so the owner agent can search them, but
their ``content`` is masked in search results returned to other agents — only
title/path + metadata are visible (see ``remnant.search``).
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Any

from .config import RemnantConfig
from .db import RemnantDB
from .embed import Embedder
from .entity import extract_and_link_entities, extract_entities, link_memory_entities

log = logging.getLogger("remnant.vault")

# Frontmatter keys carried over into memory metadata. Anything else in the
# frontmatter is still preserved under metadata too, but these are the ones
# the spec calls out as first-class.
_FRONTMATTER_KEYS = ("type", "tags", "status", "created", "updated", "author", "locked")

# Only markdown notes are indexed. Non-markdown attachments live in the vault
# but are out of scope for text indexing.
_MARKDOWN_SUFFIXES = {".md", ".markdown"}
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _relative_path(absolute_path: str | Path, vault_root: str | Path) -> str:
    """Return `absolute_path` relative to `vault_root`, using '/' separators.

    Always returns a posix-style relative string regardless of platform so
    source_ids are stable across OSes.
    """
    ap = Path(absolute_path)
    root = Path(vault_root)
    try:
        rel = ap.resolve().relative_to(root.resolve())
    except ValueError:
        # If resolution fails (e.g. non-existent path during tests), fall back
        # to a string-prefix strip without resolving symlinks.
        rel = Path(ap)
        root_abs = str(root)
        if not root_abs.endswith("/"):
            root_abs += "/"
        s = str(ap)
        if s.startswith(root_abs):
            rel = Path(s[len(root_abs):])
    return "/".join(rel.parts)


def _should_index(relative_path: str, exclude_patterns: list[str]) -> bool:
    """True if `relative_path` is allowed under the exclusion rules.

    Exclusion patterns match the *first* path component (the top-level vault
    folder). A pattern like ``90_`` excludes any folder whose name starts with
    ``90_``; ``99_ARCHIVE`` excludes that exact folder. Matching is on the
    leading path segment only, so nested folders with the same name deeper in
    the tree are still indexed.
    """
    if not relative_path:
        return False
    parts = relative_path.split("/")
    top = parts[0]
    for pat in exclude_patterns or []:
        if not pat:
            continue
        if pat.endswith("/"):
            pat = pat[:-1]
        if top == pat or top.startswith(pat):
            return False
    return True


def _file_hash(path: str | Path) -> str:
    """sha256 hex digest of the file's bytes. Streaming; memory-cheap."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a markdown note into (frontmatter_dict, body_text).

    Frontmatter is delimited by a leading ``---`` fence (YAML). When absent or
    unparseable, returns ``({}, text)``. The body is the text after the closing
    fence, with a single trailing newline preserved at most.
    """
    if not text:
        return {}, ""
    # Only a fence at the very start counts.
    if not text.startswith("---"):
        return {}, text
    # Find the closing fence on its own line.
    rest = text[3:]
    # Skip the newline immediately after the opening ``---``.
    if rest.startswith("\r\n"):
        rest = rest[2:]
    elif rest.startswith("\n"):
        rest = rest[1:]
    # Search for a line that is exactly ``---`` (optionally ``...``).
    close_idx = -1
    close_len = 0
    for marker in ("\n---\n", "\n---\r\n", "\n...\n", "\n...\r\n"):
        i = rest.find(marker)
        if i != -1 and (close_idx == -1 or i < close_idx):
            close_idx = i
            close_len = len(marker)
    # Handle a closing fence at EOF (no trailing newline).
    if close_idx == -1:
        for marker in ("\n---", "\n..."):
            if rest.endswith(marker):
                close_idx = len(rest) - len(marker)
                close_len = len(marker)
                break
    if close_idx == -1:
        # No closing fence — treat the whole thing as body.
        return {}, text
    yaml_block = rest[:close_idx]
    body = rest[close_idx + close_len:]
    try:
        import yaml

        data = yaml.safe_load(yaml_block)
    except (yaml.YAMLError, ImportError):  # type: ignore[misc]
        data = None
    if not isinstance(data, dict):
        return {}, text
    # Normalize tags: YAML may render them as a list or a comma string.
    if "tags" in data and isinstance(data["tags"], str):
        data["tags"] = [t.strip() for t in data["tags"].split(",") if t.strip()]
    elif "tags" in data and isinstance(data["tags"], list):
        data["tags"] = [str(t).strip() for t in data["tags"] if str(t).strip()]
    return data, body.rstrip() + ("\n" if body.rstrip() else "")


def _title_from(rel: str, body: str) -> str:
    """Best-effort note title: first H1 line, else the filename stem."""
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("# "):
            return s[2:].strip()
    return Path(rel).stem


def _build_metadata(frontmatter: dict[str, Any], rel: str) -> dict[str, Any]:
    """Carry the spec's first-class keys into metadata, plus the vault path."""
    meta: dict[str, Any] = {"vault_path": rel}
    for k in _FRONTMATTER_KEYS:
        if k in frontmatter:
            meta[k] = frontmatter[k]
    # Preserve any other frontmatter keys verbatim.
    for k, v in frontmatter.items():
        if k not in _FRONTMATTER_KEYS:
            meta[f"fm_{k}"] = v
    return meta


def _tags_for(frontmatter: dict[str, Any]) -> list[str] | None:
    tags = frontmatter.get("tags")
    if isinstance(tags, list) and tags:
        return [str(t) for t in tags]
    return None


def _split_passages(body: str, max_chars: int, overlap: int) -> list[dict[str, Any]]:
    """Split a note into heading-aware, overlapping retrieval passages."""
    text = (body or "").strip()
    if not text or max_chars <= 0 or len(text) <= max_chars:
        return [{"content": text, "heading_path": "", "start": 0, "end": len(text)}]
    headings: list[str] = []
    sections: list[tuple[str, str, int]] = []
    section_start = 0
    current_heading = ""
    offset = 0
    for line in text.splitlines(keepends=True):
        match = _HEADING_RE.match(line.strip())
        if match:
            if offset > section_start:
                sections.append((text[section_start:offset], current_heading, section_start))
            level = len(match.group(1))
            headings = headings[: level - 1]
            headings.append(match.group(2).strip())
            current_heading = " > ".join(headings)
            section_start = offset
        offset += len(line)
    if section_start < len(text):
        sections.append((text[section_start:], current_heading, section_start))

    out: list[dict[str, Any]] = []
    step_overlap = max(0, min(overlap, max_chars // 2))
    for section, heading_path, base in sections:
        start = 0
        while start < len(section):
            end = min(len(section), start + max_chars)
            # Prefer a paragraph or line boundary within the last quarter.
            if end < len(section):
                boundary = max(section.rfind("\n\n", start + max_chars * 3 // 4, end),
                               section.rfind("\n", start + max_chars * 3 // 4, end))
                if boundary > start:
                    end = boundary
            content = section[start:end].strip()
            if content:
                out.append({
                    "content": content,
                    "heading_path": heading_path,
                    "start": base + start,
                    "end": base + end,
                })
            if end >= len(section):
                break
            start = max(end - step_overlap, start + 1)
    return out or [{"content": text, "heading_path": "", "start": 0, "end": len(text)}]


def index_file(
    db: RemnantDB,
    config: RemnantConfig,
    embedder: Embedder,
    path: str | Path,
) -> str | None:
    """Index a single vault file. Returns the memory_id (new or existing).

    - Skips excluded paths and non-markdown files.
    - Skips unchanged files (hash matches the stored value).
    - On change: updates the existing memory in place; only forgets when the
      file is deleted. Embeds the body, links extracted entities, and updates
      the vault_files row.
    - Locked notes are indexed normally; the lock flag is in metadata so
      search can mask content for other agents.
    """
    vault_root = Path(config.vault_path)
    abs_path = Path(path)
    rel = _relative_path(abs_path, vault_root)
    if not _should_index(rel, config.vault_exclude):
        return None
    if abs_path.suffix.lower() not in _MARKDOWN_SUFFIXES:
        return None
    if not abs_path.is_file():
        return None

    try:
        hash_hex = _file_hash(abs_path)
    except OSError as e:
        log.warning("vault: cannot hash %s: %s", rel, e)
        return None

    existing_hash = db.get_vault_hash(rel)
    existing_mid = db.get_vault_memory(rel)
    if existing_hash == hash_hex and existing_mid:
        # Unchanged file — keep the existing memory, nothing to do.
        return existing_mid

    try:
        raw = abs_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        log.warning("vault: cannot read %s: %s", rel, e)
        return None

    frontmatter, body = _parse_frontmatter(raw)
    if not body.strip():
        # Empty notes still get a row so deletion-tracking works, but we don't
        # emit a memory with empty content.
        body = _title_from(rel, raw)

    title = _title_from(rel, body)
    passages = _split_passages(
        body.strip() or title,
        config.vault_passage_chars,
        config.vault_passage_overlap,
    )
    metadata = _build_metadata(frontmatter, rel)
    tags = _tags_for(frontmatter)
    locked = bool(frontmatter.get("locked"))
    if locked:
        metadata["locked"] = True

    embed_model = getattr(embedder, "_model", None) if embedder else None
    existing_passages = {p["ordinal"]: p["memory_id"] for p in db.get_vault_passages(rel)}
    # Migrate a legacy one-memory note in place: it becomes the first passage.
    if not existing_passages and existing_mid:
        existing_passages[0] = existing_mid

    memory_ids: list[str] = []
    for ordinal, passage in enumerate(passages):
        content = passage["content"]
        passage_metadata = {
            **metadata,
            "title": title,
            "parent_vault_path": rel,
            "passage_ordinal": ordinal,
            "heading_path": passage["heading_path"],
            "start_offset": passage["start"],
            "end_offset": passage["end"],
        }
        embedding = embedder.embed(content) if embedder else None
        existing_passage_id = existing_passages.get(ordinal)
        if existing_passage_id:
            db.update_memory_content(
                existing_passage_id,
                content=content,
                tags=tags,
                metadata=passage_metadata,
                embedding=embedding or None,
                embed_model=embed_model,
            )
            mid = existing_passage_id
        else:
            mid = db.insert_memory(
                content=content,
                source="vault",
                source_id=rel if ordinal == 0 else f"{rel}#p{ordinal}",
                agent=config.agent_id,
                visibility=config.default_visibility,
                type="document",
                tags=tags,
                metadata=passage_metadata,
                confidence=1.0,
                trust_score=0.8,
                embedding=embedding or None,
                embed_model=embed_model,
            )
        db.set_vault_passage(
            rel, ordinal, mid, passage["heading_path"], passage["start"], passage["end"]
        )
        extract_and_link_entities(
            db, memory_id=mid, text=content,
            typed_entities=None, agent_id=config.agent_id,
            min_memories=1,
        )
        memory_ids.append(mid)

    db.forget_vault_passages_after(rel, len(passages) - 1)
    db.set_vault_hash(rel, hash_hex, memory_id=memory_ids[0])
    return memory_ids[0]


def index_vault(
    db: RemnantDB,
    config: RemnantConfig,
    embedder: Embedder,
    *,
    force: bool = False,
) -> dict[str, int]:
    """Walk the vault, indexing changed/new files and forgetting deleted ones.

    Returns ``{"indexed": n, "skipped": n, "forgotten": n}``:
      - ``indexed``: files that were (re)indexed (new or hash-changed).
      - ``skipped``: files unchanged since last index (hash match).
      - ``forgotten``: files present in the DB but no longer on disk.

    If ``force`` is True, every non-excluded markdown file is re-indexed
    regardless of hash.
    """
    vault_root = Path(config.vault_path)
    indexed = 0
    skipped = 0
    seen_paths: set[str] = set()

    if not vault_root.is_dir():
        log.warning("vault: path does not exist or is not a directory: %s", vault_root)
        forgotten = db.mark_vault_forgotten_for_missing(seen_paths)
        return {"indexed": 0, "skipped": 0, "forgotten": len(forgotten)}

    for abs_path in _walk_markdown(vault_root, config.vault_exclude):
        rel = _relative_path(abs_path, vault_root)
        seen_paths.add(rel)
        if not force:
            try:
                hash_hex = _file_hash(abs_path)
            except OSError:
                continue
            existing = db.get_vault_hash(rel)
            if existing == hash_hex and db.get_vault_memory(rel):
                skipped += 1
                continue
        mid = index_file(db, config, embedder, abs_path)
        if mid:
            indexed += 1

    forgotten = db.mark_vault_forgotten_for_missing(seen_paths)
    return {"indexed": indexed, "skipped": skipped, "forgotten": len(forgotten)}


def _walk_markdown(
    vault_root: Path, exclude_patterns: list[str]
) -> list[Path]:
    """Yield markdown files under `vault_root` that pass the exclusion filter.

    Sorted for stable, deterministic indexing order. Uses a manual walk so the
    excluded top-level folders are pruned at the directory level (never even
    descended into), keeping token/speed costs minimal.
    """
    out: list[Path] = []
    # Sort top-level entries so exclusions are cheap and order is stable.
    try:
        top_entries = sorted(vault_root.iterdir(), key=lambda p: p.name)
    except OSError:
        return out
    for entry in top_entries:
        rel = _relative_path(entry, vault_root)
        if not _should_index(rel, exclude_patterns):
            continue
        if entry.is_dir():
            for p in sorted(entry.rglob("*")):
                if p.is_file() and p.suffix.lower() in _MARKDOWN_SUFFIXES:
                    prel = _relative_path(p, vault_root)
                    if _should_index(prel, exclude_patterns):
                        out.append(p)
        elif entry.is_file() and entry.suffix.lower() in _MARKDOWN_SUFFIXES:
            out.append(entry)
    return out


def last_reindex_ts(db: RemnantDB) -> float | None:
    """Return the most recent indexed_at timestamp across vault_files, or None.

    Stored as ISO-8601 strings; we parse to a UTC epoch seconds float so the
    reindex scheduler can compare against ``vault_reindex_interval_s``.
    """
    with db.read() as cur:
        cur.execute("SELECT MAX(indexed_at) AS m FROM vault_files")
        row = cur.fetchone()
    if not row or not row["m"]:
        return None
    import calendar

    try:
        return calendar.timegm(time.strptime(row["m"], "%Y-%m-%dT%H:%M:%SZ"))
    except (ValueError, TypeError):
        return None


__all__ = [
    "index_vault",
    "index_file",
    "last_reindex_ts",
]
