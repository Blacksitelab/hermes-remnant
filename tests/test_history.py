from __future__ import annotations

import gc
import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from remnant.config import RemnantConfig
from remnant.context import conservative_token_count
from remnant.db import HISTORY_SUMMARY_DAILY_CALL_CAP, open_db
from remnant.history import HistoryArchive, HistoryService, _reference_from_message, resolve_range

UTC = timezone.utc


def epoch(value: str) -> float:
    return datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp()


BASE = epoch("2024-06-14T00:00:00")

def make_archive(home: Path) -> Path:
    home.mkdir(exist_ok=True)
    path = home / "state.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE sessions(
            id TEXT PRIMARY KEY, source TEXT NOT NULL, user_id TEXT,
            started_at REAL NOT NULL, ended_at REAL, title TEXT,
            profile_name TEXT, rewind_count INTEGER NOT NULL DEFAULT 0,
            hidden INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE messages(
            id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL,
            content TEXT, timestamp REAL NOT NULL, active INTEGER NOT NULL DEFAULT 1,
            compacted INTEGER NOT NULL DEFAULT 0, display_kind TEXT,
            _compressed_summary INTEGER NOT NULL DEFAULT 0
        );
        CREATE VIRTUAL TABLE messages_fts USING fts5(
            content, content='messages', content_rowid='id'
        );
        """
    )
    sessions = [
        ("before", "interactive", epoch("2024-06-13T23:00:00"), None, None, None, 0, 0),
        ("day", "interactive", epoch("2024-06-14T12:00:00"), None, None, None, 0, 0),
        ("cron", "cron", epoch("2024-06-14T14:00:00"), None, None, None, 0, 0),
        ("hidden", "interactive", epoch("2024-06-14T10:00:00"), None, None, None, 0, 1),
        ("scaffold", "subagent", epoch("2024-06-14T10:00:00"), None, None, None, 0, 0),
    ]
    conn.executemany(
        "INSERT INTO sessions(id,source,started_at,ended_at,title,profile_name,"
        "rewind_count,hidden) "
        "VALUES(?,?,?,?,?,?,?,?)",
        sessions,
    )
    messages = [
        (1, "before", "user", "no date words here", epoch("2024-06-14T00:00:00"), 1, 0, None, 0),
        (
            2,
            "before",
            "assistant",
            "exclusive endpoint",
            epoch("2024-06-14T01:00:00"),
            1,
            0,
            None,
            0,
        ),
        (3, "day", "user", "project alpha decision", epoch("2024-06-14T12:00:00"), 1, 0, None, 0),
        (4, "day", "assistant", "tool scaffolding", epoch("2024-06-14T12:01:00"), 1, 0, None, 0),
        (
            5,
            "day",
            "tool",
            "project alpha hidden tool",
            epoch("2024-06-14T12:02:00"),
            1,
            0,
            None,
            0,
        ),
        (6, "day", "user", "rewound project", epoch("2024-06-14T12:03:00"), 0, 0, None, 0),
        (7, "day", "assistant", "compressed context", epoch("2024-06-14T12:04:00"), 1, 0, None, 1),
        (8, "cron", "user", "automated alpha report", epoch("2024-06-14T14:00:00"), 1, 0, None, 0),
        (9, "hidden", "user", "secret alpha", epoch("2024-06-14T10:00:00"), 1, 0, None, 0),
        (10, "scaffold", "user", "subagent alpha", epoch("2024-06-14T10:00:00"), 1, 0, None, 0),
    ]
    conn.executemany("INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?)", messages)
    conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def archive_home(tmp_path: Path) -> Path:
    return tmp_path / "hermes-profile"


@pytest.fixture
def archive(archive_home: Path) -> HistoryArchive:
    path = make_archive(archive_home)
    return HistoryArchive(path, profile_home=archive_home)


def service_for(archive: HistoryArchive, tmp_path: Path, **kwargs: object) -> HistoryService:
    db = open_db(tmp_path / "remnant.db")
    config = RemnantConfig(history_summary_enabled=kwargs.pop("history_summary_enabled", False))
    return HistoryService(
        db,
        config,
        archive=archive,
        now=datetime(2024, 6, 15, tzinfo=UTC),
    )


def test_date_recall_uses_message_time_and_visibility(
    archive: HistoryArchive, tmp_path: Path
) -> None:
    service = service_for(archive, tmp_path)
    try:
        result = service.recall({"start": "2024-06-14", "synthesize": False})
        session_ids = [row["id"] for row in result["sessions"]]
        message_ids = [row["source"]["message_id"] for row in result["evidence"]]
        assert result["status"] == "ok"
        assert session_ids == ["before", "day", "cron"]
        assert message_ids == [1, 3, 8, 2, 4]
        assert 5 not in message_ids and 6 not in message_ids and 7 not in message_ids
        assert result["sessions"][0]["link"] == "@session:default/before"
        assert result["coverage"]["summaries_missing_or_stale"] == 3

        bounded = service.recall(
            {
                "start": "2024-06-14T00:00:00+00:00",
                "end": "2024-06-14T01:00:00+00:00",
                "synthesize": False,
            }
        )
        assert [item["source"]["message_id"] for item in bounded["evidence"]] == [1]
    finally:
        service.db.close()


def test_topic_and_anchor_are_source_linked(archive: HistoryArchive, tmp_path: Path) -> None:
    service = service_for(archive, tmp_path)
    try:
        topic = service.recall({"query": "project alpha", "synthesize": False})
        assert [row["id"] for row in topic["sessions"]] == ["day", "cron"]
        assert all(
            item["source"]["link"] == "@session:default/day"
            for item in topic["evidence"]
            if item["source"]["session_id"] == "day"
        )
        around = service.recall({"session_id": "day", "around_message_id": 3, "synthesize": False})
        assert [item["source"]["message_id"] for item in around["evidence"]] == [3, 4]
        assert all(item["source"]["role"] in {"user", "assistant"} for item in around["evidence"])
    finally:
        service.db.close()


def test_date_pages_have_truthful_keyset_cursor(archive_home: Path, tmp_path: Path) -> None:
    path = make_archive(archive_home)
    conn = sqlite3.connect(path)
    base = epoch("2024-06-14T02:00:00")
    for index in range(25):
        sid = f"page-{index:02d}"
        conn.execute(
            "INSERT INTO sessions(id,source,started_at,profile_name) VALUES(?,?,?,?)",
            (sid, "interactive", base + index, None),
        )
        conn.execute(
            "INSERT INTO messages(id,session_id,role,content,timestamp) VALUES(?,?,?,?,?)",
            (100 + index, sid, "user", f"page {index}", base + index),
        )
    conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
    conn.commit()
    conn.close()
    archive = HistoryArchive(path, profile_home=archive_home)
    service = service_for(archive, tmp_path)
    try:
        first = service.recall({"start": "2024-06-14", "synthesize": False})
        assert first["has_more"] is True
        assert first["next_cursor"]
        pages = [first]
        while pages[-1]["has_more"]:
            assert pages[-1]["next_cursor"]
            assert len(pages) < 10
            pages.append(
                service.recall(
                    {
                        "start": "2024-06-14",
                        "cursor": pages[-1]["next_cursor"],
                        "synthesize": False,
                    }
                )
            )
        page_ids = [{row["id"] for row in page["sessions"]} for page in pages]
        assert all(
            left.isdisjoint(right)
            for index, left in enumerate(page_ids)
            for right in page_ids[index + 1 :]
        )
        assert len(set().union(*page_ids)) == 28
    finally:
        service.db.close()


def test_invalid_selector_does_not_touch_archive(archive: HistoryArchive, tmp_path: Path) -> None:
    class NoReadArchive:
        unavailable_reason = "should not be read"

        def available(self) -> bool:
            raise AssertionError("invalid input opened archive")

    service = service_for(archive, tmp_path)
    service.archive = NoReadArchive()
    try:
        invalid = service.recall({"start": "06-14", "synthesize": False})
        assert invalid["status"] == "invalid_request"
        assert invalid["warnings"]
        invalid_zone = service.recall(
            {"start": "2024-06-14", "timezone": "Not/AZone", "synthesize": False}
        )
        assert invalid_zone["status"] == "invalid_request"
    finally:
        service.db.close()


def test_range_resolution_handles_dst_and_fixed_clock() -> None:
    spring = resolve_range(
        relative="today",
        timezone_name="Pacific/Auckland",
        reference_now=datetime(2024, 9, 29, tzinfo=UTC),
    )
    fall = resolve_range(
        relative="today",
        timezone_name="America/New_York",
        reference_now=datetime(2024, 11, 3, 12, tzinfo=UTC),
    )
    assert (spring.end_utc - spring.start_utc).total_seconds() == 23 * 3600
    assert (fall.end_utc - fall.start_utc).total_seconds() == 25 * 3600
    with pytest.raises(ValueError):
        resolve_range(start="2024-06-15", end="2024-06-14")


def test_summary_queue_is_bounded_and_budgeted(tmp_path: Path) -> None:
    db = open_db(tmp_path / "remnant.db")
    try:
        for index in range(65):
            assert db.enqueue_history_summary(
                archive_key="archive",
                agent_id="owner",
                session_id=f"session-{index}",
            ) is (index < 64)
        assert (
            db.enqueue_history_summary(
                archive_key="archive", agent_id="owner", session_id="session-0"
            )
            is False
        )
        claimed = [db.claim_history_summary(agent_id="owner") for _ in range(20)]
        assert all(claimed)
        assert db.claim_history_summary(agent_id="owner") is None
        with db.read() as cur:
            calls = cur.execute(
                "SELECT value FROM dream_state WHERE owner=? AND key LIKE 'history_calls:%'",
                ("owner",),
            ).fetchone()
        assert int(json.loads(calls[0])) == HISTORY_SUMMARY_DAILY_CALL_CAP
    finally:
        db.close()


def test_summary_worker_validates_citations_and_stores_no_transcript(
    archive: HistoryArchive, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from remnant import history as history_module

    db = open_db(tmp_path / "remnant.db")
    config = RemnantConfig(history_summary_enabled=True)
    service = HistoryService(db, config, archive=archive)
    assert service.enqueue_session("day")
    calls: list[dict[str, object]] = []

    def fake_chat(**kwargs: object) -> str:
        calls.append(kwargs)
        reference = history_module._reference_from_message(
            {
                "id": 3,
                "session_id": "day",
                "role": "user",
                "content": "project alpha decision",
                "timestamp": epoch("2024-06-14T12:00:00"),
                "profile_name": None,
            }
        )
        return json.dumps(
            {
                "topics": ["project"],
                "statements": [
                    {
                        "topic": "project",
                        "text": "A project decision was discussed.",
                        "kind": "decision",
                        "sources": [reference],
                    }
                ],
            }
        )

    monkeypatch.setattr(history_module, "chat", fake_chat)
    try:
        assert service.process_one_summary() is True
        assert len(calls) == 1
        row = db.get_history_summary(
            archive_key=archive.archive_key, agent_id="default", session_id="day"
        )
        assert row and row["status"] == "ready"
        assert "project alpha decision" not in row["summary_json"]
        assert len(row["summary_json"].encode()) < 12_000
    finally:
        db.close()


def test_history_disabled_skips_summary_claim(archive: HistoryArchive, tmp_path: Path) -> None:
    db = open_db(tmp_path / "remnant.db")
    config = RemnantConfig(history_enabled=False)
    service = HistoryService(db, config, archive=archive)
    try:
        assert service.enqueue_session("day") is False
        assert service.process_one_summary() is False
        assert service.recall({"session_id": "day", "synthesize": False})["status"] == "disabled"
    finally:
        db.close()


def test_long_session_messages_continue_without_duplicates(
    archive: HistoryArchive, tmp_path: Path
) -> None:
    connection = sqlite3.connect(archive.path)
    try:
        connection.execute(
            "INSERT INTO sessions(id,source,started_at,ended_at,title,profile_name,"
            "rewind_count,hidden) VALUES(?,?,?,?,?,?,?,?)",
            ("long", "interactive", epoch("2024-06-14T00:00:00"), None, None, None, 0, 0),
        )
        connection.executemany(
            "INSERT INTO messages(id,session_id,role,content,timestamp,active,compacted,"
            "display_kind,_compressed_summary) VALUES(?,?,?,?,?,?,?,?,?)",
            [
                (
                    1000 + index,
                    "long",
                    "user" if index % 2 == 0 else "assistant",
                    f"long message {index}",
                    epoch("2024-06-14T03:00:00") + index,
                    1,
                    0,
                    None,
                    0,
                )
                for index in range(85)
            ],
        )
        connection.commit()
    finally:
        connection.close()
    service = service_for(archive, tmp_path)
    try:
        first = service.recall({"session_id": "long", "synthesize": False})
        assert first["has_more"] is True
        assert first["next_cursor"]
        pages = [first]
        while pages[-1]["has_more"]:
            assert pages[-1]["next_cursor"]
            assert len(pages) < 10
            pages.append(
                service.recall(
                    {
                        "session_id": "long",
                        "cursor": pages[-1]["next_cursor"],
                        "synthesize": False,
                    }
                )
            )
        evidence_ids = [item["source"]["message_id"] for page in pages for item in page["evidence"]]
        assert len(evidence_ids) == 85
        assert len(set(evidence_ids)) == len(evidence_ids)
        assert pages[-1]["has_more"] is False
    finally:
        service.db.close()


def _add_session_messages(path: Path, session_id: str, rows: list[tuple[int, str, float]]) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "INSERT INTO sessions(id,source,started_at,profile_name) VALUES(?,?,?,?)",
            (session_id, "interactive", rows[0][2] if rows else 0, None),
        )
        connection.executemany(
            "INSERT INTO messages(id,session_id,role,content,timestamp) VALUES(?,?,?,?,?)",
            [(mid, session_id, "user", content, timestamp) for mid, content, timestamp in rows],
        )
        connection.commit()
    finally:
        connection.close()


def test_review_regressions_keep_long_pages_and_wire_budget(
    archive: HistoryArchive, tmp_path: Path
) -> None:
    rows = [(1000 + index, "x" * 1900, epoch("2024-06-14T00:00:00") + index) for index in range(24)]
    _add_session_messages(archive.path, "large", rows)
    service = service_for(archive, tmp_path)
    try:
        args: dict[str, object] = {"session_id": "large", "synthesize": False}
        seen: list[int] = []
        while True:
            result = service.recall(args)
            seen.extend(item["source"]["message_id"] for item in result["evidence"])
            assert conservative_token_count(json.dumps(result, ensure_ascii=False)) <= 4000
            if not result["has_more"]:
                break
            assert result["next_cursor"]
            args = {"session_id": "large", "cursor": result["next_cursor"], "synthesize": False}
        assert seen == list(range(1000, 1024))
    finally:
        service.db.close()


def test_review_regressions_reject_unseen_and_invalidated_summary_sources(
    archive: HistoryArchive, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _add_session_messages(
        archive.path,
        "summary",
        [
            (2000 + index, f"message {index}", epoch("2024-06-14T00:00:00") + index)
            for index in range(60)
        ],
    )
    _add_session_messages(
        archive.path,
        "edited",
        [(3000, "We discussed options.", BASE), (3001, "Adopt option blue.", BASE + 1)],
    )
    service = service_for(archive, tmp_path, history_summary_enabled=True)
    try:
        unseen = _reference_from_message(service.archive.get_messages("summary")[29])
        service.enqueue_session("summary")
        calls: list[dict[str, object]] = []

        def fake_chat(**kwargs: object) -> str:
            calls.append(kwargs)
            return json.dumps(
                {
                    "topics": ["alpha"],
                    "statements": [
                        {"text": "Discussed alpha", "sources": [unseen]}
                    ],
                }
            )

        monkeypatch.setattr("remnant.history.chat", fake_chat)
        assert service.process_one_summary() is False
        assert calls and json.dumps(unseen, separators=(",", ":")) not in str(calls[0]["user"])
        cached = service.db.get_history_summary(
            archive_key=archive.archive_key, agent_id="default", session_id="summary"
        )
        assert cached and cached["status"] == "retry_wait"

        refs = [_reference_from_message(row) for row in service.archive.get_messages("edited")]
        service.enqueue_session("edited")
        claim = service.db.claim_history_summary(
            agent_id="default", archive_key=archive.archive_key
        )
        assert claim
        assert service.db.complete_history_summary(
            claim["id"],
            source_version=service.archive.source_version("edited") or "",
            summary={"statements": [{"text": "Adopt option blue.", "sources": refs}]},
            coverage={},
            claim_token=claim["claim_token"],
            archive_key=archive.archive_key,
        )
        connection = sqlite3.connect(archive.path)
        connection.execute(
            "UPDATE messages SET content=? WHERE id=3001",
            ("Actually adopt option red.",),
        )
        connection.commit()
        connection.close()
        result = service.recall({"session_id": "edited", "synthesize": False})
        assert result["coverage"]["summaries_used"] == 0
        assert all(not item.get("summary") for item in result["evidence"])
        assert any("Actually adopt option red." in item["excerpt"] for item in result["evidence"])
    finally:
        service.db.close()


def test_review_regressions_partition_summary_claims_and_retry_missing_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home_a = tmp_path / "a"
    home_b = tmp_path / "b"
    archive_a = HistoryArchive(make_archive(home_a), profile_home=home_a)
    archive_b = HistoryArchive(make_archive(home_b), profile_home=home_b)
    db = open_db(tmp_path / "shared.db")
    config = RemnantConfig(history_summary_enabled=True)
    first = HistoryService(db, config, archive=archive_a)
    second = HistoryService(db, config, archive=archive_b)
    try:
        assert first.enqueue_session("day")
        with monkeypatch.context() as patcher:
            patcher.setattr(
                "remnant.history.chat",
                lambda **_: (_ for _ in ()).throw(AssertionError()),
            )
            assert second.process_one_summary() is False
        row = db.get_history_summary(
            archive_key=archive_a.archive_key, agent_id="default", session_id="day"
        )
        assert row and row["status"] == "pending"
    finally:
        db.close()

    late_home = tmp_path / "late"
    late_db = open_db(tmp_path / "late-remnant.db")
    late = HistoryService(
        late_db,
        RemnantConfig(history_summary_enabled=False),
        profile_home=late_home,
    )
    try:
        assert late.recall({"query": "project", "synthesize": False})["status"] == "unavailable"
        make_archive(late_home)
        assert late.recall({"query": "project", "synthesize": False})["status"] != "unavailable"
    finally:
        late_db.close()


def test_review_regressions_fix_runtime_identity_topic_tail_and_relative_cursor(
    archive: HistoryArchive, tmp_path: Path
) -> None:
    _add_session_messages(archive.path, "tail", [(4000, "x" * 2200 + " quasar", BASE)])
    runtime_db = open_db(tmp_path / "runtime.db")
    runtime_config = RemnantConfig(
        agent_id="alice", runtime_identity_enabled=True, history_summary_enabled=False
    )
    runtime = HistoryService(runtime_db, runtime_config, archive=archive)
    try:
        runtime_db.insert_turn(
            session_id="day", agent_id="alice", user_text="owned", assistant_text=""
        )
        runtime_db.insert_turn(
            session_id="tail", agent_id="alice", user_text="owned", assistant_text=""
        )
        message = archive.get_messages("day")[0]
        assert archive.validate_reference(
            _reference_from_message(message),
            agent_id="alice",
            remnant_db=runtime_db,
            runtime_identity_enabled=True,
            trusted_session_ids=set(),
        )
        tail = runtime.recall({"query": "quasar", "synthesize": False})
        assert any(item["source"]["message_id"] == 4000 for item in tail["evidence"])
    finally:
        runtime_db.close()

    service = service_for(archive, tmp_path)
    try:
        _add_session_messages(
            archive.path,
            "relative-large",
            [
                (5000 + index, "relative", BASE + index)
                for index in range(85)
            ],
        )
        service._now = datetime(2024, 6, 15, tzinfo=UTC)
        first = service.recall({"relative": "yesterday", "synthesize": False})
        assert first["next_cursor"]
        service._now = datetime(2024, 6, 16, tzinfo=UTC)
        second = service.recall(
            {"relative": "yesterday", "cursor": first["next_cursor"], "synthesize": False}
        )
        assert second["resolved_range"] == first["resolved_range"]
        assert service.recall({"relative": "999999999999999999999d"})["status"] == "invalid_request"
        assert service.recall({"start": "9999-12-31"})["status"] == "invalid_request"
    finally:
        service.db.close()


def test_review_regressions_union_raw_and_summary_topic_matches(
    archive: HistoryArchive, tmp_path: Path
) -> None:
    _add_session_messages(archive.path, "raw", [(6000, "quasar raw discussion", BASE)])
    _add_session_messages(archive.path, "cached", [(6001, "a star discussion", BASE + 1)])
    service = service_for(archive, tmp_path)
    try:
        row = service.archive.get_messages("cached")[0]
        service.db.enqueue_history_summary(
            archive_key=archive.archive_key,
            agent_id="default",
            session_id="cached",
        )
        claim = service.db.claim_history_summary(
            agent_id="default", archive_key=archive.archive_key
        )
        assert claim
        assert service.db.complete_history_summary(
            claim["id"],
            source_version=service.archive.source_version("cached") or "",
            summary={
                "topics": ["quasar"],
                "statements": [
                    {
                        "text": "quasar discussion",
                        "sources": [_reference_from_message(row)],
                    }
                ],
            },
            coverage={},
            claim_token=claim["claim_token"],
            archive_key=archive.archive_key,
        )
        result = service.recall({"query": "quasar", "synthesize": False})
        ids = [item["source"]["message_id"] for item in result["evidence"]]
        assert ids == [6000, 6001]
    finally:
        service.db.close()


def test_review_regressions_close_archive_connections_and_honor_deadline(
    archive: HistoryArchive, tmp_path: Path
) -> None:
    gc.disable()
    try:
        before = len(list(Path("/proc/self/fd").iterdir()))
        for _ in range(25):
            archive.get_session("day")
        after = len(list(Path("/proc/self/fd").iterdir()))
        assert after <= before + 1
    finally:
        gc.enable()
        gc.collect()

    db = open_db(tmp_path / "deadline.db")
    service = HistoryService(
        db,
        RemnantConfig(
            agent_id="deadline",
            runtime_identity_enabled=True,
            history_summary_enabled=False,
        ),
        archive=archive,
    )
    try:
        db.insert_turn(
            session_id="day", agent_id="deadline", user_text="owned", assistant_text=""
        )
        locked = threading.Event()

        def hold_read_lock() -> None:
            with db.read():
                locked.set()
                time.sleep(3.2)

        thread = threading.Thread(target=hold_read_lock)
        thread.start()
        assert locked.wait(1)
        started = time.perf_counter()
        result = service.recall({"session_id": "day", "synthesize": False})
        elapsed = time.perf_counter() - started
        thread.join()
        assert elapsed < 3.2
        assert result["status"] == "partial"
        assert result["coverage"]["discovery_complete"] is False
    finally:
        db.close()
