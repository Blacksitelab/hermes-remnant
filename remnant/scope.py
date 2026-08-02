"""Shared visibility and profile-scope policy.

This module contains the policy decisions that must stay identical across
keyword, semantic, graph, dream, and tool retrieval.  It intentionally has no
database dependency so it can also be used to validate model-produced actions
before they reach SQLite.
"""

from __future__ import annotations

from collections.abc import Iterable

VISIBILITY_ORDER = {"private": 0, "shared": 1, "fleet": 2}
SHAREABLE_VISIBILITIES = frozenset({"shared", "fleet"})
_DENY_SCOPE = "__remnant_scope_denied__"


def normalize_profile_scope(values: Iterable[str] | None) -> list[str]:
    """Normalize vault path prefixes and remove duplicates."""
    out: list[str] = []
    for value in values or ():
        prefix = str(value or "").strip().replace("\\", "/").strip("/")
        if prefix and prefix not in out:
            out.append(prefix)
    return out


def path_in_profile_scope(path: str | None, prefixes: Iterable[str] | None) -> bool:
    """Return whether a vault-relative path is inside one allowed prefix."""
    normalized_path = str(path or "").replace("\\", "/").strip("/")
    normalized_prefixes = normalize_profile_scope(prefixes)
    if not normalized_prefixes:
        return True
    return any(
        normalized_path == prefix or normalized_path.startswith(prefix + "/")
        for prefix in normalized_prefixes
    )


def _prefix_contains(parent: str, child: str) -> bool:
    return child == parent or child.startswith(parent + "/")


def effective_profile_scope(
    configured: Iterable[str] | None,
    requested: Iterable[str] | None,
) -> list[str]:
    """Return a requested scope capped by the configured scope.

    ``None`` means no request was supplied.  An explicit empty request is not
    allowed to disable a configured scope.  An incompatible request produces a
    sentinel prefix that matches no real vault path, while leaving non-vault
    memories unaffected (profile scope only governs vault documents).
    """
    configured_prefixes = normalize_profile_scope(configured)
    requested_prefixes = normalize_profile_scope(requested)
    if not configured_prefixes:
        return requested_prefixes
    if not requested_prefixes:
        return configured_prefixes

    effective: list[str] = []
    for requested_prefix in requested_prefixes:
        for configured_prefix in configured_prefixes:
            if _prefix_contains(configured_prefix, requested_prefix):
                effective.append(requested_prefix)
            elif _prefix_contains(requested_prefix, configured_prefix):
                effective.append(configured_prefix)
    return list(dict.fromkeys(effective)) or [_DENY_SCOPE]


def is_shareable_visibility(visibility: str | None) -> bool:
    """Return whether content may participate in cross-agent/cloud dreams."""
    return str(visibility or "private").strip().lower() in SHAREABLE_VISIBILITIES


def visibility_allows(memory_visibility: str | None, requested: str | None) -> bool:
    """Apply the existing visibility ceiling semantics consistently."""
    if not requested:
        return True
    cap = VISIBILITY_ORDER.get(str(requested).strip().lower())
    if cap is None:
        return True
    return VISIBILITY_ORDER.get(str(memory_visibility or "private").lower(), 0) <= cap


__all__ = [
    "SHAREABLE_VISIBILITIES",
    "VISIBILITY_ORDER",
    "effective_profile_scope",
    "is_shareable_visibility",
    "normalize_profile_scope",
    "path_in_profile_scope",
    "visibility_allows",
]
