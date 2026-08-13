from __future__ import annotations

import time
from pathlib import Path

from remnant import RemnantMemoryProvider
from remnant.config import save_config
from remnant.identity import effective_identity


class _Embedder:
    _model = "test"

    @staticmethod
    def embed(_text: str) -> list[float]:
        return [1.0]

    @staticmethod
    def close() -> None:
        return None


def test_gateway_users_have_distinct_non_secret_storage_keys():
    first = effective_identity(
        configured_agent="default",
        session_id="s1",
        runtime_identity_enabled=True,
        agent_identity="claire",
        agent_workspace="hermes",
        platform="telegram",
        user_id="raw-user-1",
    )
    second = effective_identity(
        configured_agent="default",
        session_id="s2",
        runtime_identity_enabled=True,
        agent_identity="claire",
        agent_workspace="hermes",
        platform="telegram",
        user_id="raw-user-2",
    )
    assert first.storage_key != second.storage_key
    assert "raw-user" not in str(first.diagnostic())


def test_missing_gateway_identity_is_session_isolated():
    first = effective_identity(
        configured_agent="default",
        session_id="s1",
        runtime_identity_enabled=True,
        platform="telegram",
    )
    second = effective_identity(
        configured_agent="default",
        session_id="s2",
        runtime_identity_enabled=True,
        platform="telegram",
    )
    assert first.storage_key != second.storage_key


def test_explicit_alias_merges_platform_identifiers():
    aliases = {"telegram:111": "kris", "discord:222": "kris"}
    telegram = effective_identity(
        configured_agent="default",
        session_id="s1",
        runtime_identity_enabled=True,
        platform="telegram",
        user_id="111",
        aliases=aliases,
    )
    discord = effective_identity(
        configured_agent="default",
        session_id="s2",
        runtime_identity_enabled=True,
        platform="discord",
        user_id="222",
        aliases=aliases,
    )
    assert telegram.user_key == discord.user_key
    assert telegram.storage_key == discord.storage_key


def test_legacy_identity_preserves_configured_agent():
    identity = effective_identity(
        configured_agent="legacy-owner",
        session_id="s1",
        runtime_identity_enabled=False,
        platform="telegram",
        user_id="111",
    )
    assert identity.storage_key == "legacy-owner"


def test_prefetch_cache_key_is_scoped_bounded_ttl_and_write_invalidated(
    tmp_path: Path,
):
    home = tmp_path / "hermes"
    home.mkdir()
    save_config(
        {
            "extract_enabled": False,
            "runtime_identity_enabled": True,
            "prefetch_cache_max_entries": 2,
            "prefetch_cache_ttl_s": 1,
        },
        home,
    )
    provider = RemnantMemoryProvider()
    provider.initialize(
        "s1",
        hermes_home=str(home),
        agent_identity="claire",
        agent_workspace="hermes",
        platform="telegram",
        user_id="111",
    )
    try:
        key = provider._prefetch_key("same query", "s1")
        provider._store_queued_prefetch(key, {"context": "cached"})
        assert provider._take_queued_prefetch(key) == {"context": "cached"}
        provider._store_queued_prefetch(key, {"context": "stale"})
        provider.sync_turn("a durable user statement", "noted", session_id="s1")
        assert provider._take_queued_prefetch(key) is None
        assert provider._prefetch_key("same query", "s1") != key
        keys = [provider._prefetch_key(f"query {index}", "s1") for index in range(3)]
        for index, cache_key in enumerate(keys):
            provider._store_queued_prefetch(cache_key, {"context": str(index)})
        assert len(provider._queued_prefetch) == 2
        provider._config.prefetch_cache_ttl_s = 0  # type: ignore[union-attr]
        time.sleep(0.001)
        assert provider._take_queued_prefetch(keys[-1]) is None
    finally:
        provider.shutdown()


def test_builtin_add_replace_remove_mirrors_once(tmp_path: Path):
    home = tmp_path / "mirror-home"
    home.mkdir()
    save_config({"extract_enabled": False}, home)
    provider = RemnantMemoryProvider()
    provider.initialize("s1", hermes_home=str(home))
    try:
        provider._embedder.close()  # type: ignore[union-attr]
        provider._embedder = _Embedder()  # type: ignore[assignment]
        provider.on_memory_write("add", "user", "User prefers tea", {"session_id": "s1"})
        provider.on_memory_write(
            "replace",
            "user",
            "User prefers coffee",
            {"session_id": "s1", "old_content": "User prefers tea"},
        )
        provider.on_memory_write("remove", "user", "User prefers coffee", {"session_id": "s1"})
        with provider._db.read() as cur:  # type: ignore[union-attr]
            cur.execute("SELECT content, status FROM memories ORDER BY created_at")
            rows = [(row["content"], row["status"]) for row in cur.fetchall()]
        assert rows == [
            ("User prefers tea", "inactive"),
            ("User prefers coffee", "inactive"),
        ]
    finally:
        provider.shutdown()


def test_optional_hooks_are_safe_before_initialization():
    provider = RemnantMemoryProvider()
    assert provider.identity_diagnostic() == {"initialized": False}
    assert provider.on_pre_compress([]) == ""
    provider.queue_prefetch("query")
    provider.on_memory_write("add", "memory", "fact")
    provider.on_delegation("task", "result")
    provider.on_session_end([])
