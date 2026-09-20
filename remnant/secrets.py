"""Small, deterministic secret-like classifier and redactor.

This module deliberately has no database or command-line imports.  It only
recognises finite, high-confidence credential forms; ordinary pointer prose is
reported separately and is never treated as a literal credential.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SecretFinding:
    """A location-only finding.  The matched value is never retained."""

    kind: str
    subclass: str
    start: int
    end: int
    location: str = ""


# Ordered from the most structurally specific forms to the generic assignment
# form.  The replacement uses only the named subclass, never the match.
_LITERAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private_key",
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "jwt",
        re.compile(
            r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b",
        ),
    ),
    (
        "bearer_token",
        re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}", re.IGNORECASE),
    ),
    (
        "github_token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{16,}|github_pat_[A-Za-z0-9_]{16,})\b"),
    ),
    (
        "gitlab_token",
        re.compile(r"\bglpat-[A-Za-z0-9_-]{16,}\b"),
    ),
    (
        "slack_token",
        re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{16,}\b"),
    ),
    (
        "aws_access_key",
        re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    ),
    (
        "provider_token",
        re.compile(
            r"\b(?:sk|pk|rk|tok|cred|secret|token|apikey|api_key|access_token)"
            r"[-_][A-Za-z0-9_-]{12,}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "credential_assignment",
        re.compile(
            r"\b(?P<key>api[_ -]?key|access[_ -]?(?:key|token)|client[_ -]?secret|"
            r"secret|token|password|passwd|passphrase|private[_ -]?key)\b"
            r"\s*(?:=|:)\s*(?:[\"'](?P<quoted>[^\"']{6,})[\"']|"
            r"(?P<bare>[A-Za-z0-9][^\s,;\]}]{5,}))",
            re.IGNORECASE,
        ),
    ),
    (
        "database_credential",
        re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s:@/]+:[^\s@/]{6,}@[^\s]+", re.IGNORECASE),
    ),
    (
        "high_entropy_token",
        re.compile(
            r"(?=[A-Za-z0-9_+/=-]{24,}\b)(?=[^\s]*[A-Z])(?=[^\s]*[a-z])"
            r"(?=[^\s]*\d)[A-Za-z0-9_+/=-]{24,}\b"
        ),
    ),
)

_POINTER_RE = re.compile(
    r"\b(?:api[ _-]?key|access[ _-]?token|password|passwd|passphrase|credential|"
    r"private[ _-]?key|secret|token)\b",
    re.IGNORECASE,
)


def _normalise_subclass(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_") or "credential"


def _literal_findings(text: str) -> list[SecretFinding]:
    findings: list[SecretFinding] = []
    for subclass, pattern in _LITERAL_PATTERNS:
        for match in pattern.finditer(text):
            actual_subclass = subclass
            if subclass == "credential_assignment":
                actual_subclass = _normalise_subclass(match.group("key"))
            findings.append(
                SecretFinding("literal", actual_subclass, match.start(), match.end())
            )
    findings.sort(key=lambda item: (item.start, -(item.end - item.start), item.subclass))
    # Keep the widest match when patterns overlap.  That prevents a generic
    # token pattern from leaving a suffix of a private key or assignment.
    selected: list[SecretFinding] = []
    for finding in findings:
        if not selected or finding.start > selected[-1].end:
            selected.append(finding)
            continue
        prior = selected[-1]
        if finding.end > prior.end:
            selected[-1] = SecretFinding(
                "literal", prior.subclass, prior.start, finding.end
            )
    return sorted(selected, key=lambda item: item.start)


def classify_text(text: str | None, *, location: str = "") -> list[SecretFinding]:
    """Return literal and pointer findings without retaining matched values."""
    if not isinstance(text, str) or not text:
        return []
    literal = _literal_findings(text)
    findings = [
        SecretFinding(item.kind, item.subclass, item.start, item.end, location)
        for item in literal
    ]
    literal_spans = [(item.start, item.end) for item in literal]
    seen_pointer: set[tuple[int, int]] = set()
    for match in _POINTER_RE.finditer(text):
        span = (match.start(), match.end())
        if span in seen_pointer or any(
            match.start() >= start and match.end() <= end for start, end in literal_spans
        ):
            continue
        seen_pointer.add(span)
        findings.append(
            SecretFinding("pointer", "credential_pointer", match.start(), match.end(), location)
        )
    findings.sort(key=lambda item: (item.start, item.kind != "literal", item.subclass))
    return findings


def _safe_location(field: str) -> str:
    """Keep field hints useful without copying a literal into the hint."""
    path = str(field or "value")
    findings = _literal_findings(path)
    if not findings:
        return path.replace("\r", "\\r").replace("\n", "\\n")
    parts: list[str] = []
    cursor = 0
    for finding in findings:
        if finding.start < cursor:
            continue
        parts.append(path[cursor:finding.start])
        parts.append("[REDACTED]")
        cursor = finding.end
    parts.append(path[cursor:])
    return "".join(parts).replace("\r", "\\r").replace("\n", "\\n")


def _marker(finding: SecretFinding, field: str) -> str:
    path = _safe_location(field)
    hint = finding.subclass or "credential"
    return f"[SECRET-LIKE] field={path} location={hint}"


def redact_text(text: str | None, *, field: str = "value") -> str | None:
    """Replace literal matches while preserving pointer prose."""
    if not isinstance(text, str) or not text:
        return text
    findings = [item for item in classify_text(text, location=field) if item.kind == "literal"]
    if not findings:
        return text
    parts: list[str] = []
    cursor = 0
    for finding in findings:
        if finding.start < cursor:
            continue
        parts.append(text[cursor:finding.start])
        parts.append(_marker(finding, field))
        cursor = finding.end
    parts.append(text[cursor:])
    return "".join(parts)


def _redacted_key(key: Any) -> tuple[Any, bool]:
    if not isinstance(key, str):
        return key, False
    if not any(item.kind == "literal" for item in classify_text(key)):
        return key, False
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"[SECRET-LIKE-KEY:{digest}]", True


def redact_value(value: Any, *, field: str = "value") -> Any:
    """Recursively redact reportable textual values in JSON-like data."""
    if isinstance(value, str):
        return redact_text(value, field=field)
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for key, item in value.items():
            safe_key, key_was_redacted = _redacted_key(key)
            key_component = "<secret-key>" if key_was_redacted else str(key)
            out[safe_key] = redact_value(item, field=f"{field}.{key_component}")
        return out
    if isinstance(value, list):
        return [redact_value(item, field=f"{field}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, tuple):
        return tuple(
            redact_value(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        )
    return value


def memory_fingerprint(row: dict[str, Any]) -> str:
    """Hash the complete mutable memory row without exposing its contents."""
    fields = (
        "id", "type", "content", "source", "source_id", "agent", "visibility",
        "timestamp", "confidence", "trust_score", "verified", "superseded_by",
        "status", "tags", "metadata", "content_hash", "seen_count", "created_at",
        "updated_at",
    )
    values: dict[str, Any] = {}
    for field in fields:
        value = row.get(field)
        if field in {"tags", "metadata"} and isinstance(value, str):
            try:
                value = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                pass
        values[field] = value
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def is_locked_memory(row: dict[str, Any]) -> bool:
    """Recognise the existing vault frontmatter lock representation."""
    metadata = row.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (TypeError, json.JSONDecodeError):
            if re.search(r"[\"']?locked[\"']?\s*:\s*(?:true|1)", metadata, re.IGNORECASE):
                return True
            metadata = None
    if isinstance(metadata, dict) and (
        metadata.get("locked") is True or str(metadata.get("locked", "")).casefold() == "true"
    ):
        return True
    tags = row.get("tags")
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except (TypeError, json.JSONDecodeError):
            tags = [tags]
    if isinstance(tags, (list, tuple)) and any(str(tag).casefold() == "locked" for tag in tags):
        return True
    return str(row.get("visibility") or "").casefold() == "locked"


# Short aliases make the helper convenient for callers while keeping the
# descriptive names used by the hygiene workflow as the primary API.
classify = classify_text
redact = redact_text
redact_record = redact_value


__all__ = [
    "SecretFinding",
    "classify_text",
    "classify",
    "redact_text",
    "redact",
    "redact_value",
    "redact_record",
    "memory_fingerprint",
    "is_locked_memory",
]
