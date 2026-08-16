"""Regression tests for the P0 privacy and durable-state fixes."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from remnant.config import RemnantConfig
from remnant.db import default_db_path, open_db
from remnant.dream import night_dream
from remnant.extract import ExtractionWorker
from remnant.llm import chat
from remnant.search import search
from remnant.vault import index_vault


def _db():
    return open_db(default_db_path())


def test_configured_profile_scope_cannot_be_disabled_by_empty_request():
    db = _db()
    try:
        db.insert_memory(
            content="alpha project note",
            source="vault",
            source_id="Projects/a.md",
            agent="agent",
            visibility="shared",
        )
        db.insert_memory(
            content="alpha private note",
            source="vault",
            source_id="Personal/diary.md",
            agent="agent",
            visibility="shared",
        )
        cfg = RemnantConfig(agent_id="agent", profile_scope=["Projects"])
        rows = search(db, cfg, "alpha", agent_id="agent", profile_scope=[])
        assert [row["source_id"] for row in rows] == ["Projects/a.md"]
    finally:
        db.close()


def test_private_memories_never_enter_night_dream_cloud_corpus(monkeypatch):
    db = _db()
    try:
        db.insert_memory(
            content="same durable private fact",
            source="manual",
            agent="alice",
            visibility="private",
            embedding=[1.0, 0.0],
            embed_model="test",
        )
        db.insert_memory(
            content="same durable private fact",
            source="manual",
            agent="bob",
            visibility="private",
            embedding=[1.0, 0.0],
            embed_model="test",
        )
        called = []

        def fake_judge(config, pairs, *, mode):
            called.extend(pairs)
            return []

        monkeypatch.setattr("remnant.dream._cloud_judge", fake_judge)
        result = night_dream(db, RemnantConfig(dream_night_budget=1), object())
        assert result["candidates"] == 0
        assert called == []
    finally:
        db.close()


def test_missing_vault_root_never_forgets_existing_documents(tmp_path: Path):
    db = _db()
    try:
        mid = db.insert_memory(
            content="alpha project note",
            source="vault",
            source_id="Projects/a.md",
            agent="agent",
            visibility="shared",
        )
        db.set_vault_hash("Projects/a.md", "hash", memory_id=mid)
        stats = index_vault(
            db,
            RemnantConfig(vault_path=str(tmp_path / "unmounted")),
            object(),
        )
        assert stats["forgotten"] == 0
        assert stats["failed"] == 1
        assert db.get_memory(mid)["status"] == "active"
    finally:
        db.close()


def test_successful_zero_fact_extraction_is_terminal():
    db = _db()
    try:
        turn_id = db.insert_turn_with_extraction(
            session_id="s",
            agent_id="agent",
            user_text="hello",
            assistant_text="hi",
        )
        job = db.claim_next_extraction("agent")
        assert job is not None
        db.complete_extraction(int(job["id"]), fact_count=0)
        assert db.get_unextracted_turns("agent") == []
        with db.read() as cur:
            cur.execute(
                "SELECT extraction_status, extraction_fact_count FROM turns WHERE id=?",
                (turn_id,),
            )
            row = cur.fetchone()
        assert row["extraction_status"] == "completed"
        assert row["extraction_fact_count"] == 0
    finally:
        db.close()


def test_failed_extraction_is_retryable_and_not_successful():
    db = _db()
    try:
        db.insert_turn_with_extraction(
            session_id="s",
            agent_id="agent",
            user_text="hello",
            assistant_text="hi",
        )
        job = db.claim_next_extraction("agent")
        assert job is not None
        db.fail_extraction(int(job["id"]), error="temporary endpoint failure")
        with db.read() as cur:
            cur.execute(
                "SELECT status, last_error FROM extraction_queue WHERE id=?",
                (int(job["id"]),),
            )
            queue_row = cur.fetchone()
            cur.execute("SELECT extraction_status FROM turns WHERE id=?", (int(job["turn_id"]),))
            turn_row = cur.fetchone()
        assert queue_row["status"] == "pending"
        assert queue_row["last_error"] == "temporary endpoint failure"
        assert turn_row["extraction_status"] == "retry_wait"
    finally:
        db.close()


def test_extraction_process_contract_propagates_failures(monkeypatch):
    worker = ExtractionWorker(object(), object(), RemnantConfig())
    monkeypatch.setattr(
        worker,
        "_extract",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("endpoint unavailable")),
    )
    try:
        with pytest.raises(RuntimeError, match="endpoint unavailable"):
            worker._process(
                {
                    "turn_id": 1,
                    "user_text": "hello",
                    "assistant_text": "hi",
                }
            )
    finally:
        worker.stop()


def test_llm_adapter_normalizes_native_and_openai_responses():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((str(request.url), request.read()))
        if "/api/chat" in str(request.url):
            return httpx.Response(200, json={"message": {"content": "native"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "openai"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        assert chat(
            url="http://llm/api/chat",
            model="model",
            system="system",
            user="user",
            timeout=1,
            client=client,
        ) == "native"
        assert chat(
            url="http://llm/v1/chat/completions",
            model="model",
            system="system",
            user="user",
            timeout=1,
            client=client,
        ) == "openai"
    finally:
        client.close()
    assert len(requests) == 2


def test_llm_adapter_adds_extraction_controls_per_protocol():
    payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append((str(request.url), json.loads(request.read())))
        if "/api/chat" in str(request.url):
            return httpx.Response(200, json={"message": {"content": '{"facts": []}'}})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"facts": []}'}}]})

    schema = {
        "type": "object",
        "properties": {"facts": {"type": "array"}},
        "required": ["facts"],
    }
    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        chat(
            url="http://llm/api/chat",
            model="model",
            system="system",
            user="user",
            timeout=1,
            temperature=0,
            max_tokens=1536,
            num_ctx=8192,
            think=False,
            response_schema=schema,
            client=client,
        )
        chat(
            url="http://llm/v1/chat/completions",
            model="model",
            system="system",
            user="user",
            timeout=1,
            temperature=0,
            max_tokens=1536,
            num_ctx=8192,
            think=False,
            response_schema=schema,
            client=client,
        )
    finally:
        client.close()

    native = payloads[0][1]
    assert native["think"] is False
    assert native["format"] == schema
    assert native["options"]["num_ctx"] == 8192
    assert native["options"]["num_predict"] == 1536

    openai = payloads[1][1]
    assert openai["response_format"]["type"] == "json_schema"
    assert openai["response_format"]["json_schema"]["schema"] == schema
    assert openai["reasoning_effort"] == "none"
    assert "options" not in openai
    assert "format" not in openai
    assert "num_ctx" not in openai
