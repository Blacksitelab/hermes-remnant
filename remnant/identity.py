"""Stable, privacy-preserving runtime identity for shared Remnant databases."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


def _digest(value: str, length: int = 20) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


@dataclass(frozen=True)
class EffectiveIdentity:
    configured_agent: str
    agent_identity: str
    workspace: str
    platform: str
    platform_scope: str
    user_key: str
    agent_context: str
    session_id: str
    parent_session_id: str
    legacy: bool = False

    @property
    def storage_key(self) -> str:
        if self.legacy:
            return self.configured_agent
        components = {
            "profile": self.configured_agent,
            "agent": self.agent_identity or self.configured_agent,
            "platform": self.platform_scope,
            "user": self.user_key,
            "workspace": self.workspace,
        }
        encoded = json.dumps(components, sort_keys=True, separators=(",", ":"))
        return f"identity:v2:{_digest(encoded)}"

    @property
    def viewer_key(self) -> str:
        return self.storage_key

    def diagnostic(self) -> dict[str, str | bool]:
        """Return non-secret identity components suitable for local health output."""
        return {
            "storage_key": self.storage_key,
            "configured_agent": self.configured_agent,
            "agent_identity": self.agent_identity,
            "workspace": self.workspace,
            "platform": self.platform,
            "user_key": self.user_key,
            "agent_context": self.agent_context,
            "legacy": self.legacy,
        }


def effective_identity(
    *,
    configured_agent: str,
    session_id: str,
    runtime_identity_enabled: bool,
    aliases: dict[str, str] | None = None,
    **runtime: Any,
) -> EffectiveIdentity:
    """Construct an identity without persisting or logging raw external user IDs."""
    agent_identity = str(runtime.get("agent_identity") or configured_agent).strip()
    workspace = str(runtime.get("agent_workspace") or "").strip()
    platform = str(runtime.get("platform") or "unknown").strip().casefold()
    context = str(runtime.get("agent_context") or "primary").strip().casefold()
    parent = str(runtime.get("parent_session_id") or "").strip()
    raw_user = str(runtime.get("user_id_alt") or runtime.get("user_id") or "").strip()
    alias_map = aliases or {}
    alias = alias_map.get(f"{platform}:{raw_user}") or alias_map.get(raw_user)
    if alias:
        user_key = f"alias:{_digest(str(alias))}"
    elif raw_user:
        user_key = f"external:{_digest(f'{platform}:{raw_user}')}"
    elif platform in {"cli", "cron", "flush"}:
        user_key = f"local:{_digest(f'{workspace}:{agent_identity}')}"
    else:
        # A gateway with no stable user identity is isolated to this session;
        # missing identity can reduce recall but must never broaden access.
        user_key = f"anonymous-session:{_digest(session_id or 'default')}"
    return EffectiveIdentity(
        configured_agent=configured_agent or "default",
        agent_identity=agent_identity,
        workspace=workspace,
        platform=platform,
        platform_scope="aliased" if alias else platform,
        user_key=user_key,
        agent_context=context,
        session_id=session_id or "default",
        parent_session_id=parent,
        legacy=not runtime_identity_enabled,
    )


__all__ = ["EffectiveIdentity", "effective_identity"]
