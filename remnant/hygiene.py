"""Reviewable hygiene report and approval workflow for Remnant.

``report`` uses a caller-supplied SQLite file through a read-only, explicit
transaction.  ``apply`` accepts only a hashed, owner-bound manifest and routes
mutations through ``MemoryLifecycle`` so status, claim, relation and audit
projections retain their normal semantics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

from .config import RemnantConfig
from .db import SCHEMA_VERSION, MemoryConflictError, RemnantDB
from .edit import memory_edit
from .secrets import (
    SecretFinding,
    classify_text,
    is_locked_memory,
    memory_fingerprint,
    redact_text,
    redact_value,
)

REPORT_COLUMNS = (
    "class",
    "subclass",
    "memory_id",
    "agent",
    "source",
    "created_at",
    "claim",
    "excerpt",
    "fingerprint",
    "decision",
)
_MANIFEST_VERSION = 1
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_NOOP_DECISIONS = {"", "blank", "keep", "defer", "skip"}


class HygieneError(RuntimeError):
    """A safe, user-facing hygiene workflow error."""


class HygieneConflictError(HygieneError):
    """An approval cannot be applied to the current row state."""


class HygieneValidationError(HygieneError):
    """A report or approval manifest failed structural validation."""


def _safe_identity(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HygieneValidationError(f"{label} is required")
    if any(item.kind == "literal" for item in classify_text(value, location=label)):
        raise HygieneValidationError(f"{label} is unsafe")
    return value


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_json(value: Any, default: Any) -> Any:
    if not isinstance(value, str) or not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _store_binding(db_path: str | Path) -> str:
    path = Path(db_path).expanduser().resolve()
    try:
        stat = path.stat()
    except OSError as exc:
        raise HygieneValidationError("database is unavailable") from exc
    return _canonical_hash({"path": str(path), "device": stat.st_dev, "inode": stat.st_ino})


def store_binding(db_path: str | Path) -> str:
    """Return the stable path/device/inode binding used in manifests."""
    return _store_binding(db_path)


def _snapshot_layout(path: Path) -> tuple[tuple[int, int, int, int], int, tuple[str, ...]]:
    try:
        stat = path.stat()
        with path.open("rb") as handle:
            header = handle.read(100)
    except OSError as exc:
        raise HygieneValidationError("database snapshot could not be inspected") from exc
    if len(header) < 100 or header[:16] != b"SQLite format 3\x00":
        raise HygieneValidationError("database is not a SQLite file")
    journal_version = header[18]
    if journal_version not in {1, 2}:
        raise HygieneValidationError("database journal layout is unsupported")
    signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    sidecars = tuple(
        str(candidate)
        for candidate in (
            path.with_name(path.name + "-wal"),
            path.with_name(path.name + "-shm"),
            path.with_name(path.name + "-journal"),
        )
        if candidate.exists()
    )
    return signature, journal_version, sidecars


class _ReadonlySnapshotConnection(sqlite3.Connection):
    """Connection that cleans a private WAL capture after close."""

    _snapshot_cleanup: tempfile.TemporaryDirectory[str] | None = None
    _snapshot_source: Path | None = None
    _snapshot_source_layout: tuple[tuple[int, int, int, int], int, tuple[str, ...]] | None = None
    _snapshot_source_signature: tuple[tuple[Any, ...], ...] | None = None

    def close(self) -> None:
        cleanup = getattr(self, "_snapshot_cleanup", None)
        try:
            super().close()
        finally:
            if cleanup is not None:
                cleanup.cleanup()
                self._snapshot_cleanup = None


def _assert_snapshot_layout(
    path: Path,
    expected: tuple[tuple[int, int, int, int], int, tuple[str, ...]] | None = None,
    *,
    allow_sidecars: bool = False,
) -> None:
    current = _snapshot_layout(path)
    if expected is not None and current[:2] != expected[:2]:
        raise HygieneValidationError("database snapshot changed during read")
    if expected is not None and current[2] != expected[2]:
        raise HygieneValidationError("database journal sidecars changed during read")
    if current[2] and not allow_sidecars:
        raise HygieneValidationError("database snapshot has active journal sidecars")


def _wal_source_signature(path: Path, sidecars: tuple[str, ...]) -> tuple[tuple[Any, ...], ...]:
    files = (str(path), *sidecars)
    signature: list[tuple[Any, ...]] = []
    for name in files:
        candidate = Path(name)
        stat = candidate.stat()
        signature.append(
            (
                name,
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                _file_sha256(candidate),
            )
        )
    return tuple(signature)


def _copy_wal_snapshot(
    source: Path,
    layout: tuple[tuple[int, int, int, int], int, tuple[str, ...]],
) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
    expected_sidecars = {
        str(source.with_name(source.name + "-wal")),
        str(source.with_name(source.name + "-shm")),
    }
    if set(layout[2]) != expected_sidecars:
        raise HygieneValidationError("database WAL snapshot layout is unsupported")
    # ponytail: three capture attempts, then fail closed; an unbounded copy can
    # never produce a trustworthy snapshot from a continuously changing WAL.
    for _attempt in range(3):
        cleanup: tempfile.TemporaryDirectory[str] | None = None
        try:
            before = _wal_source_signature(source, layout[2])
            cleanup = tempfile.TemporaryDirectory(prefix="hygiene-snapshot-")
            target = Path(cleanup.name) / source.name
            for name in (str(source), *layout[2]):
                shutil.copyfile(name, Path(cleanup.name) / Path(name).name)
            after = _wal_source_signature(source, layout[2])
        except (OSError, FileNotFoundError):
            if cleanup is not None:
                cleanup.cleanup()
            continue
        if before == after:
            return target, cleanup
        if cleanup is not None:
            cleanup.cleanup()
    raise HygieneValidationError("database WAL snapshot is changing")


def _assert_connection_source(connection: sqlite3.Connection) -> None:
    source = getattr(connection, "_snapshot_source", None)
    layout = getattr(connection, "_snapshot_source_layout", None)
    expected_signature = getattr(connection, "_snapshot_source_signature", None)
    if source is not None and layout is not None:
        _assert_snapshot_layout(source, layout, allow_sidecars=bool(layout[2]))
        if (
            expected_signature is not None
            and _wal_source_signature(source, layout[2]) != expected_signature
        ):
            raise HygieneValidationError("database snapshot changed during read")


def _open_readonly(db_path: str | Path) -> sqlite3.Connection:
    source_path = Path(db_path).expanduser().resolve()
    if not source_path.is_file():
        raise HygieneValidationError("database file does not exist")
    source_layout = _snapshot_layout(source_path)
    cleanup: tempfile.TemporaryDirectory[str] | None = None
    path = source_path
    layout = source_layout
    if layout[2]:
        if layout[1] != 2:
            raise HygieneValidationError("database snapshot has active journal sidecars")
        path, cleanup = _copy_wal_snapshot(source_path, source_layout)
        layout = _snapshot_layout(path)
    # A WAL reader may create -wal/-shm even with mode=ro/query_only.  An
    # immutable connection is safe only for a closed WAL snapshot with no
    # sidecars; a live WAL is copied to a private, disposable snapshot above.
    query = "mode=ro&immutable=1" if layout[1] == 2 and not layout[2] else "mode=ro"
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            path.as_uri() + "?" + query,
            uri=True,
            isolation_level=None,
            timeout=0.0,
            factory=_ReadonlySnapshotConnection,
        )
        if cleanup is not None:
            connection._snapshot_cleanup = cleanup
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA foreign_keys=ON")
        mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).casefold()
        if layout[1] == 1 and mode != "delete":
            raise HygieneValidationError("database journal layout changed during open")
        connection.execute("BEGIN")
        _assert_snapshot_layout(path, layout, allow_sidecars=bool(layout[2]))
        connection._snapshot_source = source_path
        connection._snapshot_source_layout = source_layout
        connection._snapshot_source_signature = _wal_source_signature(
            source_path, source_layout[2]
        )
        return connection
    except HygieneValidationError:
        if connection is not None:
            connection.close()
        raise
    except (OSError, sqlite3.Error) as exc:
        if connection is not None:
            connection.close()
        elif cleanup is not None:
            cleanup.cleanup()
        raise HygieneValidationError("database could not be opened read-only") from exc


def _validate_schema(connection: sqlite3.Connection) -> tuple[str, int]:
    try:
        version_row = connection.execute(
            "SELECT value FROM schema_meta WHERE key='version'"
        ).fetchone()
        version = int(version_row["value"]) if version_row else -1
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    except (sqlite3.Error, TypeError, ValueError) as exc:
        raise HygieneValidationError("database schema is unsupported") from exc
    if version != SCHEMA_VERSION or not {"memories", "claims", "audit_log"}.issubset(tables):
        raise HygieneValidationError("database schema is unsupported")
    try:
        schema_rows = connection.execute(
            "SELECT type,name,COALESCE(sql,'') AS sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
        ).fetchall()
    except sqlite3.Error as exc:
        raise HygieneValidationError("database schema could not be inspected") from exc
    schema_identity = _canonical_hash([dict(row) for row in schema_rows])
    return schema_identity, version


def _snapshot_info(connection: sqlite3.Connection, db_path: str | Path) -> dict[str, Any]:
    schema_identity, version = _validate_schema(connection)
    try:
        generation_row = connection.execute(
            "SELECT value FROM schema_meta WHERE key='memory_generation'"
        ).fetchone()
        generation = int(generation_row["value"]) if generation_row else 0
    except (sqlite3.Error, TypeError, ValueError) as exc:
        raise HygieneValidationError("database generation is unavailable") from exc
    binding = _store_binding(db_path)
    return {
        "schema_version": version,
        "schema_identity": schema_identity,
        "store_binding": binding,
        "snapshot_generation": generation,
    }


def _walk_text_fields(value: Any, field: str) -> list[tuple[str, str, list[SecretFinding]]]:
    if isinstance(value, str):
        findings = classify_text(value, location=field)
        return [(field, value, findings)] if findings else []
    if isinstance(value, dict):
        found: list[tuple[str, str, list[SecretFinding]]] = []
        for key, item in value.items():
            key_text = str(key)
            key_field = f"{field}.<key>"
            key_findings = classify_text(key_text, location=key_field)
            if key_findings:
                found.append((key_field, key_text, key_findings))
            found.extend(_walk_text_fields(item, f"{field}.{key_text}"))
        return found
    if isinstance(value, (list, tuple)):
        found = []
        for index, item in enumerate(value):
            found.extend(_walk_text_fields(item, f"{field}[{index}]"))
        return found
    return []


def _safe_value(value: Any, field: str) -> str:
    value = redact_value(value, field=field)
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return redact_text(str(value), field=field) or ""


def _claim_display(claim: dict[str, Any] | None) -> str:
    if not claim:
        return ""
    fields = (
        ("subject", claim.get("subject")),
        ("predicate", claim.get("predicate")),
        ("object", claim.get("object")),
        ("qualifiers", _parse_json(claim.get("qualifiers"), claim.get("qualifiers"))),
    )
    return "; ".join(
        f"{name}={_safe_value(value, f'claim.{name}')}" for name, value in fields
        if value not in (None, "", {})
    )


def _findings_for_memory(
    memory: dict[str, Any], claims: list[dict[str, Any]], fingerprint: str
) -> list[dict[str, Any]]:
    metadata = _parse_json(memory.get("metadata"), memory.get("metadata"))
    tags = _parse_json(memory.get("tags"), memory.get("tags"))
    fields: list[tuple[str, Any]] = [
        ("content", memory.get("content")),
        ("type", memory.get("type")),
        ("source", memory.get("source")),
        ("source_id", memory.get("source_id")),
        ("agent", memory.get("agent")),
        ("visibility", memory.get("visibility")),
        ("timestamp", memory.get("timestamp")),
        ("tags", tags),
        ("metadata", metadata),
        ("created_at", memory.get("created_at")),
        ("updated_at", memory.get("updated_at")),
    ]
    findings: list[dict[str, Any]] = []
    for field, value in fields:
        for location, original, matches in _walk_text_fields(value, field):
            for match in matches:
                findings.append(
                    {
                        "class": "secret-like" if match.kind == "literal" else "review-pointer",
                        "subclass": match.subclass,
                        "memory_id": str(memory["id"]),
                        "agent": redact_text(str(memory.get("agent") or ""), field="agent") or "",
                        "source": redact_text(
                            str(memory.get("source") or ""), field="source"
                        )
                        or "",
                        "created_at": redact_text(
                            str(memory.get("created_at") or ""), field="created_at"
                        )
                        or "",
                        "claim": "",
                        "excerpt": _safe_value(original, location)[:1000],
                        "fingerprint": fingerprint,
                        "decision": "",
                    }
                )
    for claim in claims:
        claim_fields = (
            ("subject", claim.get("subject")),
            ("predicate", claim.get("predicate")),
            ("object", claim.get("object")),
            ("qualifiers", _parse_json(claim.get("qualifiers"), claim.get("qualifiers"))),
        )
        claim_text = _claim_display(claim)
        for field, value in claim_fields:
            for location, original, matches in _walk_text_fields(value, f"claim.{field}"):
                for match in matches:
                    findings.append(
                        {
                            "class": "secret-like" if match.kind == "literal" else "review-pointer",
                            "subclass": match.subclass,
                            "memory_id": str(memory["id"]),
                            "agent": redact_text(
                                str(memory.get("agent") or ""), field="agent"
                            )
                            or "",
                            "source": redact_text(
                                str(memory.get("source") or ""), field="source"
                            )
                            or "",
                            "created_at": redact_text(
                                str(memory.get("created_at") or ""), field="created_at"
                            )
                            or "",
                            "claim": claim_text,
                            "excerpt": _safe_value(original, location)[:1000],
                            "fingerprint": fingerprint,
                            "decision": "",
                        }
                    )
        status = str(claim.get("status") or "").casefold()
        resolution = str(claim.get("resolution_status") or "").casefold()
        if status in {"contradicted", "unresolved"} or resolution in {
            "contradicted",
            "unresolved",
        } or claim.get("conflict_type"):
            findings.append(
                {
                    "class": "claim-review",
                    "subclass": resolution or status or "conflict",
                    "memory_id": str(memory["id"]),
                    "agent": redact_text(str(memory.get("agent") or ""), field="agent") or "",
                    "source": redact_text(str(memory.get("source") or ""), field="source") or "",
                    "created_at": redact_text(
                        str(memory.get("created_at") or ""), field="created_at"
                    )
                    or "",
                    "claim": claim_text,
                    "excerpt": _safe_value(
                        claim.get("conflict_type") or resolution or status, "claim.status"
                    ),
                    "fingerprint": fingerprint,
                    "decision": "",
                }
            )
    return findings


def _safe_report_row(row: dict[str, Any]) -> dict[str, Any]:
    # Do not redact memory_id/fingerprint: they are non-secret binding values
    # required to construct and validate an approval manifest.
    safe = dict(row)
    safe["memory_id"] = _safe_identity(str(safe.get("memory_id") or ""), "memory id")
    safe["fingerprint"] = _safe_identity(
        str(safe.get("fingerprint") or ""), "memory fingerprint"
    )
    for field in (
        "class", "subclass", "agent", "source", "created_at", "claim", "excerpt", "decision",
    ):
        safe[field] = redact_text(str(safe.get(field) or ""), field=field) or ""
    return safe


def _build_report(
    connection: sqlite3.Connection, db_path: str | Path, agent_id: str
) -> dict[str, Any]:
    agent_id = _safe_identity(agent_id, "agent")
    info = _snapshot_info(connection, db_path)
    try:
        memories = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM memories WHERE agent=? AND status='active' "
                "AND (superseded_by IS NULL OR superseded_by='') ORDER BY id",
                (agent_id,),
            ).fetchall()
        ]
    except sqlite3.Error as exc:
        raise HygieneValidationError("memory snapshot could not be read") from exc
    claims_by_memory: dict[str, list[dict[str, Any]]] = {str(row["id"]): [] for row in memories}
    if claims_by_memory:
        placeholders = ",".join("?" for _ in claims_by_memory)
        try:
            claim_rows = connection.execute(
                f"SELECT * FROM claims WHERE memory_id IN ({placeholders}) ORDER BY memory_id,id",
                list(claims_by_memory),
            ).fetchall()
        except sqlite3.Error as exc:
            raise HygieneValidationError("claim snapshot could not be read") from exc
        for row in claim_rows:
            claims_by_memory.setdefault(str(row["memory_id"]), []).append(dict(row))
    rows: list[dict[str, Any]] = []
    locked_excluded = 0
    literal_count = 0
    pointer_count = 0
    claim_count = 0
    flagged_memory_ids: set[str] = set()
    for memory in memories:
        if is_locked_memory(memory):
            locked_excluded += 1
            continue
        fingerprint = memory_fingerprint(memory)
        memory_findings = _findings_for_memory(
            memory, claims_by_memory.get(str(memory["id"]), []), fingerprint
        )
        if memory_findings:
            flagged_memory_ids.add(str(memory["id"]))
        for row in memory_findings:
            if row["class"] == "secret-like":
                literal_count += 1
            elif row["class"] == "review-pointer":
                pointer_count += 1
            elif row["class"] == "claim-review":
                claim_count += 1
            rows.append(_safe_report_row(row))
    rows.sort(key=lambda row: (row["memory_id"], row["class"], row["subclass"], row["excerpt"]))
    report = {
        "kind": "remnant-hygiene-report",
        "version": _MANIFEST_VERSION,
        **info,
        "agent": agent_id,
        "counts": {
            "owner_memories": len(memories),
            "eligible_memories": len(memories) - locked_excluded,
            "reportable_memories": len(memories) - locked_excluded,
            "flagged_memories": len(flagged_memory_ids),
            "finding_rows": len(rows),
            "row_count": len(rows),
            "literal_findings": literal_count,
            "pointer_findings": pointer_count,
            "claim_findings": claim_count,
            "locked_excluded": locked_excluded,
        },
        "rows": rows,
    }
    report["report_hash"] = _report_hash(report)
    return report


def report_snapshot(db_path: str | Path, agent_id: str) -> dict[str, Any]:
    """Create a redacted report without opening a writable RemnantDB."""
    agent_id = _safe_identity(str(agent_id or "").strip(), "agent")
    connection = _open_readonly(db_path)
    try:
        report = _build_report(connection, db_path, agent_id)
        _assert_connection_source(connection)
        return report
    finally:
        try:
            connection.rollback()
        finally:
            connection.close()


def generate_report(db_path: str | Path, agent_id: str) -> dict[str, Any]:
    return report_snapshot(db_path, agent_id)


def _report_hash(report: dict[str, Any]) -> str:
    body = {key: value for key, value in report.items() if key != "report_hash"}
    return _canonical_hash(body)


def _validated_report(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or report.get("report_hash") != _report_hash(report):
        raise HygieneValidationError("report hash is invalid")
    _safe_identity(str(report.get("agent") or ""), "report agent")
    safe = redact_value(report, field="report")
    if not isinstance(safe, dict) or safe.get("report_hash") != _report_hash(safe):
        raise HygieneValidationError("report contains unsafe fields")
    if not isinstance(safe.get("rows"), list):
        raise HygieneValidationError("report rows are invalid")
    for row in safe["rows"]:
        if not isinstance(row, dict):
            raise HygieneValidationError("report row is invalid")
        _safe_identity(str(row.get("memory_id") or ""), "memory id")
        _safe_identity(str(row.get("fingerprint") or ""), "memory fingerprint")
    return safe


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise HygieneValidationError("report file could not be read") from exc
    return digest.hexdigest()


def _secure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError as exc:
        raise HygieneValidationError("report directory permissions could not be secured") from exc


def _secure_write(path: Path, payload: str) -> None:
    _secure_directory(path.parent)
    fd, temp_name = tempfile.mkstemp(prefix=".hygiene-", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        os.chmod(path, 0o600)
    except OSError as exc:
        try:
            os.close(fd)
        except OSError:
            pass
        raise HygieneValidationError("report file could not be written securely") from exc
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def _csv_cell(value: Any, field: str) -> str:
    text = redact_text(str(value if value is not None else ""), field=field) or ""
    # Prefix potentially executable spreadsheet cells.  csv.writer then quotes
    # the complete value where needed; the leading apostrophe is intentional.
    if text.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + text
    return text


def _csv_payload(rows: list[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=REPORT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_cell(row.get(field), field) for field in REPORT_COLUMNS})
    return output.getvalue()


def write_report(
    report: dict[str, Any], output: str | Path, *, manifest_path: str | Path | None = None,
    format: str | None = None,
) -> dict[str, str]:
    """Write a CSV/JSON report and a safe companion manifest."""
    report = _validated_report(report)
    output_path = Path(output).expanduser().resolve()
    chosen_format = (
        format or ("json" if output_path.suffix.casefold() == ".json" else "csv")
    ).casefold()
    if chosen_format not in {"csv", "json"}:
        raise HygieneValidationError("report format must be csv or json")
    rows = [dict(row) for row in report.get("rows", [])]
    if chosen_format == "json":
        payload = json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        payload += "\n"
    else:
        payload = _csv_payload(rows)
    _safe_identity(str(output_path), "report path")
    companion_path = Path(manifest_path).expanduser() if manifest_path else Path(
        str(output_path) + ".manifest.json"
    )
    companion_path = companion_path.resolve()
    _safe_identity(str(companion_path), "report manifest path")
    _secure_write(output_path, payload)
    companion = {
        "kind": "remnant-hygiene-report-manifest",
        "version": _MANIFEST_VERSION,
        "report_hash": report.get("report_hash"),
        "report_path": str(output_path),
        "report_content_hash": _file_sha256(output_path),
        "report_format": chosen_format,
        "schema_version": report.get("schema_version"),
        "schema_identity": report.get("schema_identity"),
        "store_binding": report.get("store_binding"),
        "snapshot_generation": report.get("snapshot_generation"),
        "agent": report.get("agent"),
        "counts": report.get("counts", {}),
        "row_fingerprints": {
            str(row.get("memory_id")): row.get("fingerprint")
            for row in rows
            if row.get("memory_id") and row.get("fingerprint")
        },
    }
    _secure_write(companion_path, json.dumps(companion, indent=2, sort_keys=True) + "\n")
    return {"report": str(output_path), "manifest": str(companion_path)}


def create_approval_manifest(
    report: dict[str, Any], *, approver: str, reason: str, operations: list[dict[str, Any]],
    report_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the explicit authority document consumed by ``apply``."""
    report = _validated_report(report)
    if report_manifest_path is None:
        raise HygieneValidationError("approval must bind to a persisted report")
    companion_path = _resolve_report_manifest_path(report_manifest_path)
    companion = _load_json_file(companion_path, "report manifest could not be read")
    report_path = companion.get("report_path")
    report_content_hash = companion.get("report_content_hash")
    report_format = companion.get("report_format")
    if not isinstance(report_path, str) or not report_path.strip():
        raise HygieneValidationError("report manifest is missing the report path")
    if not isinstance(report_content_hash, str) or not _HEX64_RE.fullmatch(report_content_hash):
        raise HygieneValidationError("report manifest content binding is invalid")
    if report_format not in {"csv", "json"}:
        raise HygieneValidationError("report manifest format is invalid")
    _safe_identity(str(report_path), "report path")
    manifest = {
        "kind": "remnant-hygiene-approval",
        "version": _MANIFEST_VERSION,
        "report_hash": report.get("report_hash"),
        "report_manifest_path": str(companion_path.resolve()),
        "report_path": str(Path(report_path).expanduser().resolve()),
        "report_content_hash": report_content_hash,
        "report_format": report_format,
        "schema_version": report.get("schema_version"),
        "schema_identity": report.get("schema_identity"),
        "store_binding": report.get("store_binding"),
        "agent": _safe_identity(str(report.get("agent") or ""), "report agent"),
        "approver": _safe_identity(approver, "approver"),
        "reason": _safe_identity(reason, "approval reason"),
        "operations": operations,
    }
    normalized = _validate_manifest(manifest)
    manifest = {**manifest, "operations": normalized}
    _verify_report_manifest(manifest, normalized)
    return manifest


def _load_json_file(path: str | Path, error: str) -> dict[str, Any]:
    try:
        with Path(path).expanduser().open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HygieneValidationError(error) from exc
    if not isinstance(value, dict):
        raise HygieneValidationError(error)
    return value


def _resolve_report_manifest_path(source: str | Path) -> Path:
    path = Path(source).expanduser()
    if path.is_file():
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, ValueError, UnicodeDecodeError):
            value = None
        if isinstance(value, dict) and value.get("kind") == "remnant-hygiene-report-manifest":
            return path.resolve()
    if path.suffix.casefold() != ".json" or not path.name.endswith(".manifest.json"):
        path = Path(str(path) + ".manifest.json")
    return path.resolve()


def _report_rows(rows: Any) -> dict[str, str]:
    if not isinstance(rows, list):
        raise HygieneValidationError("report rows are missing")
    fingerprints: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise HygieneValidationError("report row is invalid")
        if any(
            isinstance(value, str)
            and redact_text(value, field=f"report.{field}") != value
            for field, value in row.items()
        ):
            raise HygieneValidationError("report contains unsafe fields")
        memory_id = _safe_identity(str(row.get("memory_id") or ""), "memory id")
        fingerprint = _safe_identity(
            str(row.get("fingerprint") or ""), "memory fingerprint"
        )
        if not _HEX64_RE.fullmatch(fingerprint):
            raise HygieneValidationError("report row fingerprint is invalid")
        prior = fingerprints.get(memory_id)
        if prior is not None and prior != fingerprint:
            raise HygieneValidationError("report contains conflicting row fingerprints")
        fingerprints[memory_id] = fingerprint
    return fingerprints


def _read_report_evidence(
    report_path: str | Path, report_format: str, expected_content_hash: str
) -> tuple[dict[str, Any] | None, dict[str, str]]:
    path = Path(report_path).expanduser().resolve()
    if _file_sha256(path) != expected_content_hash:
        raise HygieneValidationError("report content does not match approval")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise HygieneValidationError("report could not be read") from exc
    if report_format == "json":
        try:
            report = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise HygieneValidationError("report JSON is invalid") from exc
        if not isinstance(report, dict):
            raise HygieneValidationError("report JSON is invalid")
        safe_report = _validated_report(report)
        return safe_report, _report_rows(safe_report.get("rows"))
    if report_format != "csv":
        raise HygieneValidationError("report format is invalid")
    try:
        text = raw.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if tuple(reader.fieldnames or ()) != REPORT_COLUMNS:
            raise HygieneValidationError("report CSV columns are invalid")
        rows = [dict(row) for row in reader]
    except (UnicodeDecodeError, csv.Error) as exc:
        raise HygieneValidationError("report CSV is invalid") from exc
    return None, _report_rows(rows)


def _operation_id(operation: dict[str, Any]) -> str:
    candidate = operation.get("operation_id")
    if candidate is not None:
        candidate = str(candidate)
        if not _SAFE_ID_RE.fullmatch(candidate):
            raise HygieneValidationError("operation id is invalid")
        return _safe_identity(candidate, "operation id")
    return "op-" + _canonical_hash(operation)[:32]


def _validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if redact_value(manifest, field="approval") != manifest:
        raise HygieneValidationError("approval contains unsafe fields")
    required = (
        "report_hash", "schema_version", "schema_identity", "store_binding",
        "report_manifest_path", "report_path", "report_content_hash",
        "report_format",
        "agent", "approver", "reason",
    )
    if manifest.get("kind") != "remnant-hygiene-approval":
        raise HygieneValidationError("approval manifest kind is invalid")
    if manifest.get("version") != _MANIFEST_VERSION or any(
        not manifest.get(key) for key in required
    ):
        raise HygieneValidationError("approval manifest is incomplete")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise HygieneValidationError("approval schema version is invalid")
    if not all(
        _HEX64_RE.fullmatch(str(manifest.get(key))) for key in
        ("report_hash", "schema_identity", "store_binding", "report_content_hash")
    ):
        raise HygieneValidationError("approval manifest hash binding is invalid")
    if manifest.get("report_hash") == "0" * 64:
        raise HygieneValidationError("approval report hash is invalid")
    _safe_identity(manifest.get("agent"), "agent")
    approver = _safe_identity(manifest.get("approver"), "approver")
    reason = manifest.get("reason")
    _safe_identity(str(manifest.get("report_manifest_path")), "report manifest path")
    _safe_identity(str(manifest.get("report_path")), "report path")
    if (
        approver.casefold() in {"model", "assistant", "llm"}
        or not isinstance(reason, str)
        or not reason.strip()
    ):
        raise HygieneValidationError("explicit human approval is required")
    _safe_identity(reason, "approval reason")
    if not isinstance(manifest.get("operations"), list):
        raise HygieneValidationError("approval operations are invalid")
    allowed_keys = {
        "operation_id", "action", "decision", "memory_id", "memory_ids", "replacement_ids",
        "expected_fingerprint", "expected_fingerprints", "content", "replacement",
        "replacement_text", "reason",
    }
    normalized: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for raw in manifest["operations"]:
        if not isinstance(raw, dict) or set(raw) - allowed_keys:
            raise HygieneValidationError("approval operation is invalid")
        if "action" not in raw and "decision" not in raw:
            raise HygieneValidationError("approval action is missing")
        action_value = raw.get("action")
        decision_value = raw.get("decision")
        if (
            action_value not in (None, "")
            and decision_value not in (None, "")
            and str(action_value).strip().casefold() != str(decision_value).strip().casefold()
        ):
            raise HygieneValidationError("approval action and decision conflict")
        action = str(raw.get("action") or raw.get("decision") or "").strip().casefold()
        if action in _NOOP_DECISIONS:
            selected = raw.get("memory_ids") or raw.get("replacement_ids") or raw.get("memory_id")
            memory_ids = (
                [str(item) for item in selected]
                if isinstance(selected, list)
                else [str(selected or "")]
            )
        else:
            if action not in {"forget", "update", "merge"}:
                raise HygieneValidationError("approval action is invalid")
            selected = raw.get("memory_ids") or raw.get("replacement_ids")
            if action == "merge":
                if selected is None and isinstance(raw.get("memory_id"), list):
                    selected = raw["memory_id"]
                if not isinstance(selected, list):
                    selected = [raw.get("memory_id")]
                memory_ids = [str(item) for item in selected]
                if len(memory_ids) < 2:
                    raise HygieneValidationError("merge requires multiple memory ids")
            else:
                if selected is None:
                    memory_ids = [str(raw.get("memory_id") or "")]
                elif isinstance(selected, list):
                    memory_ids = [str(item) for item in selected]
                else:
                    raise HygieneValidationError("approval memory id is invalid")
        if (
            any(not item or len(item) > 256 for item in memory_ids)
            or len(set(memory_ids)) != len(memory_ids)
        ):
            raise HygieneValidationError("approval memory id is invalid")
        for memory_id in memory_ids:
            _safe_identity(memory_id, "approval memory id")
        if action in {"forget", "update"} and len(memory_ids) != 1:
            raise HygieneValidationError("approval action requires one memory id")
        if action in _NOOP_DECISIONS and len(memory_ids) != 1:
            raise HygieneValidationError("approval decision requires one memory id")
        if used_ids.intersection(memory_ids):
            raise HygieneValidationError("approval operations overlap")
        used_ids.update(memory_ids)
        expected_raw = raw.get("expected_fingerprints")
        if expected_raw is None and raw.get("expected_fingerprint") is not None:
            expected_raw = {memory_ids[0]: raw.get("expected_fingerprint")}
        if expected_raw is None:
            expected_raw = {}
        if not isinstance(expected_raw, dict) or set(map(str, expected_raw)) != set(memory_ids):
            raise HygieneValidationError("approval row fingerprints are incomplete")
        expected = {str(key): str(value) for key, value in expected_raw.items()}
        for key in expected:
            _safe_identity(key, "approval memory id")
        if not all(_HEX64_RE.fullmatch(value) for value in expected.values()):
            raise HygieneValidationError("approval row fingerprint is invalid")
        content = raw.get("content", raw.get("replacement", raw.get("replacement_text")))
        if action in {"update", "merge"}:
            if (
                not isinstance(content, str)
                or not content.strip()
                or "\x00" in content
                or len(content) > 1_000_000
            ):
                raise HygieneValidationError("replacement text is invalid")
            if any(item.kind == "literal" for item in classify_text(content)):
                raise HygieneValidationError("replacement text is secret-like")
        elif content is not None:
            raise HygieneValidationError("replacement text is not valid for this action")
        normalized.append(
            {
                "operation_id": _operation_id(raw),
                "action": action,
                "memory_ids": memory_ids,
                "expected_fingerprints": expected,
                **({"content": content} if content is not None else {}),
            }
        )
    if len({item["operation_id"] for item in normalized}) != len(normalized):
        raise HygieneValidationError("approval operation ids conflict")
    return normalized


def _verify_report_manifest(
    manifest: dict[str, Any], operations: list[dict[str, Any]]
) -> None:
    source = manifest.get("report_manifest_path")
    if not source:
        raise HygieneValidationError("approval report binding is missing")
    path = _resolve_report_manifest_path(source)
    companion = _load_json_file(path, "report manifest could not be read")
    if redact_value(companion, field="report_manifest") != companion:
        raise HygieneValidationError("report manifest contains unsafe fields")
    if (
        companion.get("kind") != "remnant-hygiene-report-manifest"
        or companion.get("version") != _MANIFEST_VERSION
    ):
        raise HygieneValidationError("report manifest kind is invalid")
    for key in (
        "report_hash", "schema_version", "schema_identity", "store_binding", "agent",
        "report_path", "report_content_hash", "report_format",
    ):
        if companion.get(key) != manifest.get(key):
            raise HygieneValidationError("approval does not match the report manifest")
    report_format = companion.get("report_format")
    report_path = manifest.get("report_path")
    content_hash = manifest.get("report_content_hash")
    if not isinstance(report_format, str) or not isinstance(report_path, str):
        raise HygieneValidationError("report evidence is incomplete")
    report, fingerprints = _read_report_evidence(report_path, report_format, str(content_hash))
    if report is not None:
        for key in ("report_hash", "schema_version", "schema_identity", "store_binding", "agent"):
            if report.get(key) != manifest.get(key):
                raise HygieneValidationError("approval does not match the report")
    companion_fingerprints = companion.get("row_fingerprints")
    if companion_fingerprints != fingerprints:
        raise HygieneValidationError("report manifest fingerprints are stale")
    for operation in operations:
        for memory_id, expected in operation["expected_fingerprints"].items():
            if fingerprints.get(memory_id) != expected:
                raise HygieneValidationError("approval row is absent from the report")


def _readback(
    db: RemnantDB,
    operation: dict[str, Any],
    audit_id: int,
    replacement_id: str | None,
) -> dict[str, Any]:
    ids = operation["memory_ids"]
    memories = {memory_id: db.get_memory(memory_id) for memory_id in ids}
    if operation["action"] == "forget":
        memory = memories[ids[0]]
        if not isinstance(memory, dict) or memory.get("status") != "forgotten":
            raise HygieneConflictError("forget read-back did not match")
    else:
        if not replacement_id:
            raise HygieneConflictError("replacement read-back is incomplete")
        replacement = db.get_memory(replacement_id)
        if replacement is None or replacement.get("status") != "active":
            raise HygieneConflictError("replacement read-back did not match")
        for memory_id in ids:
            memory = memories[memory_id]
            if (
                not isinstance(memory, dict)
                or memory.get("status") != "superseded"
                or memory.get("superseded_by") != replacement_id
            ):
                raise HygieneConflictError("replacement read-back did not match")
    return {
        "audit_ids": [int(audit_id)],
        "replacement_id": replacement_id,
        "readback": {
            memory_id: {
                "status": (memories[memory_id] or {}).get("status"),
                "fingerprint": memory_fingerprint(memories[memory_id])
                if isinstance(memories[memory_id], dict)
                else None,
            }
            for memory_id in ids
        },
    }


def _find_operation_audit(
    db: RemnantDB, operation: dict[str, Any]
) -> tuple[int, str | None] | None:
    with db.read() as cur:
        rows = cur.execute(
            "SELECT id, action, memory_id, details FROM audit_log "
            "WHERE action IN ('forget','update','merge') "
            "ORDER BY id DESC"
        ).fetchall()
    for row in rows:
        details = _parse_json(row["details"], {})
        if (
            not isinstance(details, dict)
            or details.get("operation_id") != operation["operation_id"]
        ):
            continue
        replacement_id = details.get("replacement_id") or details.get("after_id")
        if row["action"] != operation["action"]:
            continue
        if operation["action"] == "forget":
            if str(row["memory_id"]) != operation["memory_ids"][0]:
                continue
        else:
            if [str(item) for item in details.get("original_ids", [])] != operation["memory_ids"]:
                continue
            replacement = db.get_memory(str(replacement_id)) if replacement_id else None
            if (
                not isinstance(replacement, dict)
                or replacement.get("content") != operation.get("content")
            ):
                continue
        return int(row["id"]), str(replacement_id) if replacement_id else None
    return None


def _check_readonly_operation(
    connection: sqlite3.Connection, operation: dict[str, Any], agent_id: str
) -> str | None:
    for memory_id in operation["memory_ids"]:
        row = connection.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        if row is None:
            return "memory is missing"
        current = dict(row)
        if current.get("agent") != agent_id:
            return "memory is owned by another agent"
        if is_locked_memory(current):
            return "locked memory cannot be modified"
        if current.get("status") != "active" or current.get("superseded_by"):
            return "memory is no longer active"
        expected = operation["expected_fingerprints"].get(memory_id)
        if expected != memory_fingerprint(current):
            return "memory changed since report"
    return None


def _receipt_path(receipt_dir: Path, operation_id: str) -> Path:
    return receipt_dir / f"{operation_id}.json"


def _write_receipt(receipt_dir: Path, receipt: dict[str, Any]) -> None:
    _secure_write(
        _receipt_path(receipt_dir, str(receipt["operation_id"])),
        json.dumps(redact_value(receipt, field="receipt"), indent=2, sort_keys=True) + "\n",
    )


def _operation_counts(
    receipts: list[dict[str, Any]], total: int, *, dry_run: bool = False
) -> dict[str, int]:
    applied = sum(item.get("status") in {"applied", "reconciled"} for item in receipts)
    conflicted = sum(item.get("status") == "conflict" for item in receipts)
    no_op = sum(item.get("status") == "no-op" for item in receipts)
    if dry_run:
        unperformed = sum(item.get("status") == "would_apply" for item in receipts)
    else:
        unperformed = max(0, total - applied - conflicted - no_op)
    return {
        "applied": int(applied),
        "conflicted": int(conflicted),
        "no_op": int(no_op),
        "unperformed": int(unperformed),
    }


def apply_hygiene(
    db_path: str | Path,
    manifest: dict[str, Any] | str | Path,
    *,
    agent_id: str | None = None,
    dry_run: bool = False,
    receipt_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and apply an approval, or perform a completely read-only dry run."""
    if isinstance(manifest, (str, Path)):
        manifest = _load_json_file(manifest, "approval manifest could not be read")
    if not isinstance(manifest, dict):
        raise HygieneValidationError("approval manifest is invalid")
    operations = _validate_manifest(manifest)
    _verify_report_manifest(manifest, operations)
    owner = _safe_identity(str(manifest["agent"]).strip(), "agent")
    if not agent_id or str(agent_id).strip() != owner:
        raise HygieneValidationError("explicit agent does not match approval")
    if _store_binding(db_path) != manifest["store_binding"]:
        raise HygieneConflictError("approval is bound to another database")
    connection = _open_readonly(db_path)
    try:
        schema_identity, version = _validate_schema(connection)
        if schema_identity != manifest["schema_identity"] or version != manifest["schema_version"]:
            raise HygieneConflictError("approval schema binding is stale")
        if dry_run:
            receipts: list[dict[str, Any]] = []
            conflict = False
            for operation in operations:
                if operation["action"] in _NOOP_DECISIONS:
                    receipts.append({"operation_id": operation["operation_id"], "status": "no-op"})
                    continue
                error = _check_readonly_operation(connection, operation, owner)
                receipt = {
                    "operation_id": operation["operation_id"],
                    "action": operation["action"],
                    "memory_ids": operation["memory_ids"],
                    "status": "would_apply" if error is None else "conflict",
                }
                if error:
                    receipt["reason"] = error
                    conflict = True
                receipts.append(receipt)
            _assert_connection_source(connection)
            counts = _operation_counts(receipts, len(operations), dry_run=True)
            return {
                "kind": "remnant-hygiene-apply",
                "status": "conflict" if conflict else "dry-run",
                "agent": owner,
                "receipts": receipts,
                **counts,
            }
    finally:
        try:
            connection.rollback()
        finally:
            connection.close()
    db: RemnantDB | None = None
    receipt_root = Path(receipt_dir).expanduser() if receipt_dir else None
    if receipt_root:
        _secure_directory(receipt_root)
    receipts: list[dict[str, Any]] = []
    conflict = False
    try:
        db = RemnantDB(Path(db_path).expanduser())
        config = RemnantConfig(agent_id=owner)
        for operation in operations:
            if conflict:
                continue
            if operation["action"] in _NOOP_DECISIONS:
                receipt = {"operation_id": operation["operation_id"], "status": "no-op"}
                receipts.append(receipt)
                continue
            prior = _find_operation_audit(db, operation)
            if prior is not None:
                audit_id, replacement_id = prior
                receipt_data = _readback(db, operation, audit_id, replacement_id)
                receipt = {
                    "operation_id": operation["operation_id"],
                    "action": operation["action"],
                    "memory_ids": operation["memory_ids"],
                    "status": "reconciled",
                    **receipt_data,
                }
                receipts.append(receipt)
                if receipt_root:
                    _write_receipt(receipt_root, receipt)
                continue
            try:
                if operation["action"] == "forget":
                    result = memory_edit(
                        db,
                        config,
                        None,
                        action="forget",
                        actor=str(manifest["approver"]),
                        memory_id=operation["memory_ids"][0],
                        agent_id=owner,
                        expected_fingerprint=operation["expected_fingerprints"][operation["memory_ids"][0]],
                        operation_id=operation["operation_id"],
                    )
                    if result.get("error"):
                        raise PermissionError("approved operation was rejected")
                    audit_id = int(result["audit_id"])
                    replacement_id = None
                else:
                    result = memory_edit(
                        db,
                        config,
                        None,
                        action=operation["action"],
                        actor=str(manifest["approver"]),
                        memory_id=operation["memory_ids"][0],
                        memory_ids=operation["memory_ids"],
                        content=operation["content"],
                        agent_id=owner,
                        expected_fingerprints=operation["expected_fingerprints"],
                        operation_id=operation["operation_id"],
                    )
                    if result.get("error"):
                        raise PermissionError("approved operation was rejected")
                    audit_id = int(result["audit_id"])
                    replacement_id = str(result["memory_id"])
                receipt_data = _readback(db, operation, audit_id, replacement_id)
            except (MemoryConflictError, PermissionError, KeyError, ValueError) as exc:
                conflict = True
                receipt = {
                    "operation_id": operation["operation_id"],
                    "action": operation["action"],
                    "memory_ids": operation["memory_ids"],
                    "status": "conflict",
                    "reason": str(exc),
                }
                receipts.append(receipt)
                continue
            receipt = {
                "operation_id": operation["operation_id"],
                "action": operation["action"],
                "memory_ids": operation["memory_ids"],
                "status": "applied",
                **receipt_data,
            }
            receipts.append(receipt)
            if receipt_root:
                _write_receipt(receipt_root, receipt)
    finally:
        if db is not None:
            db.close()
    counts = _operation_counts(receipts, len(operations))
    return {
        "kind": "remnant-hygiene-apply",
        "status": "partial" if conflict else "applied",
        "agent": owner,
        "receipts": receipts,
        **counts,
    }


# Names used by operators and small integration harnesses.
apply_manifest = apply_hygiene
write_hygiene_report = write_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m remnant.hygiene")
    subparsers = parser.add_subparsers(dest="command", required=True)
    report = subparsers.add_parser("report")
    report.add_argument("--db", required=True)
    report.add_argument("--agent", required=True)
    report.add_argument("--output", required=True)
    report.add_argument("--manifest")
    report.add_argument("--format", choices=("csv", "json"))
    apply = subparsers.add_parser("apply")
    apply.add_argument("--db", required=True)
    apply.add_argument("--agent", required=True)
    apply.add_argument("--manifest", required=True)
    mode = apply.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    apply.add_argument("--receipt-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "report":
            report = report_snapshot(args.db, args.agent)
            paths = write_report(
                report, args.output, manifest_path=args.manifest, format=args.format
            )
            print(json.dumps({"report": paths["report"], "manifest": paths["manifest"]}))
            return 0
        result = apply_hygiene(
            args.db,
            args.manifest,
            agent_id=args.agent,
            dry_run=args.dry_run,
            receipt_dir=args.receipt_dir
            or str(Path(args.manifest).expanduser().parent / ".hygiene-receipts"),
        )
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] in {"dry-run", "applied"} else 2
    except HygieneError:
        print("hygiene: operation rejected", file=sys.stderr)
        return 2
    except (OSError, sqlite3.Error, RuntimeError):
        print("hygiene: operation failed", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI smoke test
    raise SystemExit(main())


__all__ = [
    "REPORT_COLUMNS",
    "HygieneError",
    "HygieneConflictError",
    "HygieneValidationError",
    "store_binding",
    "report_snapshot",
    "generate_report",
    "write_report",
    "create_approval_manifest",
    "apply_hygiene",
    "apply_manifest",
    "write_hygiene_report",
    "main",
]
