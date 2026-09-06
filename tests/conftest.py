"""Shared pytest fixtures for the Remnant test suite.

Each test gets a unique shared-DB home via the ``REMNANT_DB_HOME`` env var so
that the provider (which now opens the shared ``~/.hermes/remnant/remnant.db``
location via ``default_db_path()``) and direct ``open_db(default_db_path())``
calls hit an isolated temp DB rather than the real user database.

Config remains profile-scoped under the per-test ``hermes_home`` fixture defined
in each test module; only the DB location is shared/redirected here.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _remnant_db_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the shared Remnant DB to a per-test temp directory.

    ``default_db_path()`` reads ``REMNANT_DB_HOME`` at call time, so setting it
    here (before ``provider.initialize()`` and any direct ``open_db`` call)
    keeps every test fully isolated. ``monkeypatch`` restores the prior value on
    teardown.
    """
    db_home = tmp_path / "remnant_db_home"
    db_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("REMNANT_DB_HOME", str(db_home))
    from remnant import dream

    original = dream._expand_diary_path
    monkeypatch.setattr(
        dream, "_expand_diary_path",
        lambda path: str(tmp_path / "DREAMS.md")
        if str(path) == "~/.hermes/remnant/DREAMS.md" else original(path),
    )
