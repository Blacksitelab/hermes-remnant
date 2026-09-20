from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from remnant.config import RemnantConfig
from remnant.db import open_db
from remnant.hygiene import (
    HygieneValidationError,
    apply_hygiene,
    create_approval_manifest,
    main,
    report_snapshot,
    write_report,
)
from remnant.lifecycle import MemoryLifecycle
from remnant.secrets import classify_text, memory_fingerprint, redact_text, redact_value

TOKEN = "sk-" + "proj-" + "H48SyntheticToken1234567890"


class _Embedder:
    _model = "test"

    @staticmethod
    def embed(_text: str) -> list[float]:
        return [1.0, 0.0]


def _seed(path: Path):
    db = open_db(path)
    owner = db.insert_memory(
        content=f"A note containing {TOKEN}",
        agent="owner",
        source_id=f"note-{TOKEN}",
        tags=["tag", TOKEN],
        metadata={"nested": {"token": TOKEN}, "pointer": "the API key is held elsewhere"},
    )
    db.create_claim(
        memory_id=owner,
        subject=f"subject-{TOKEN}",
        predicate="uses",
        object=f"object-{TOKEN}",
    )
    pointer = db.insert_memory(
        content="The API key is held by the credential owner.",
        agent="owner",
    )
    locked = db.insert_memory(
        content=f"locked {TOKEN}",
        agent="owner",
        metadata={"locked": True, "token": TOKEN},
    )
    foreign = db.insert_memory(content=f"foreign {TOKEN}", agent="other")
    return db, owner, pointer, locked, foreign


def _operation(report: dict, memory_id: str, action: str, **extra):
    row = next(row for row in report["rows"] if row["memory_id"] == memory_id)
    return {
        "memory_id": memory_id,
        "expected_fingerprint": row["fingerprint"],
        "action": action,
        **extra,
    }


def _report_manifest(report: dict, tmp_path: Path, name: str = "report") -> str:
    return write_report(report, tmp_path / f"{name}.json")["manifest"]


def _directory_state(path: Path) -> dict[str, tuple[int, int, str]]:
    return {
        item.name: (
            item.stat().st_size,
            item.stat().st_mtime_ns,
            hashlib.sha256(item.read_bytes()).hexdigest(),
        )
        for item in path.iterdir()
        if item.is_file()
    }


def test_pure_classifier_redacts_literals_but_keeps_pointer_class():
    findings = classify_text(f"api_key={TOKEN}; API key is stored in a vault")
    assert any(item.kind == "literal" for item in findings)
    assert any(item.kind == "pointer" for item in findings)
    redacted = redact_text(f"api_key={TOKEN}", field="metadata.api_key")
    assert TOKEN not in redacted
    assert "[SECRET-LIKE]" in redacted
    assert "field=metadata.api_key" in redacted


def test_structured_keys_and_executable_identities_do_not_leak(tmp_path: Path):
    structured = {TOKEN: TOKEN}
    redacted = redact_text(TOKEN, field=f"metadata.{TOKEN}") or ""
    encoded = json.dumps(redact_value(structured))
    assert TOKEN not in encoded
    assert TOKEN not in redacted
    assert "[REDACTED]" in redacted

    db = open_db(tmp_path / "identity.db")
    try:
        memory_id = db.insert_memory(content="ordinary", agent="owner", metadata={TOKEN: "value"})
        report = report_snapshot(db.path, "owner")
        rows = [row for row in report["rows"] if row["memory_id"] == memory_id]
        assert rows and all(TOKEN not in json.dumps(row) for row in rows)
        paths = write_report(report, tmp_path / "identity.json")
        assert all(TOKEN not in Path(path).read_text() for path in paths.values())
        with pytest.raises(HygieneValidationError):
            create_approval_manifest(
                report,
                approver="human",
                reason="safe approval",
                operations=[
                    {
                        "memory_id": memory_id,
                        "expected_fingerprint": rows[0]["fingerprint"],
                        "action": "keep",
                        "reason": TOKEN,
                    }
                ],
                report_manifest_path=paths["manifest"],
            )
        with pytest.raises(HygieneValidationError):
            report_snapshot(db.path, TOKEN)
    finally:
        db.close()


def test_unsafe_identity_diagnostics_are_generic(tmp_path: Path, capsys):
    output = tmp_path / "report.json"
    result = main(
        [
            "report",
            "--db",
            str(tmp_path / "missing.db"),
            "--agent",
            TOKEN,
            "--output",
            str(output),
        ]
    )
    captured = capsys.readouterr()
    assert result == 2
    assert TOKEN not in captured.out + captured.err
    assert captured.err.strip() == "hygiene: operation rejected"
    assert not output.exists()


def test_report_is_read_only_redacted_and_secure(tmp_path: Path):
    db, owner, pointer, locked, _ = _seed(tmp_path / "store.db")
    try:
        before_generation = db.memory_generation
        before_entries = {path.name for path in tmp_path.iterdir()}
        report = report_snapshot(db.path, "owner")
        after_entries = {path.name for path in tmp_path.iterdir()}
        assert before_generation == db.memory_generation
        assert before_entries == after_entries
        assert report["counts"]["locked_excluded"] == 1
        assert report["counts"]["eligible_memories"] == 2
        assert locked not in {row["memory_id"] for row in report["rows"]}
        assert any(
            row["memory_id"] == pointer and row["class"] == "review-pointer"
            for row in report["rows"]
        )
        encoded = json.dumps(report, sort_keys=True)
        assert TOKEN not in encoded

        csv_path = tmp_path / "report.csv"
        json_path = tmp_path / "report.json"
        paths = write_report(report, csv_path)
        write_report(report, json_path)
        assert Path(paths["report"]).stat().st_mode & 0o777 == 0o600
        assert Path(paths["manifest"]).stat().st_mode & 0o777 == 0o600
        assert tmp_path.stat().st_mode & 0o777 == 0o700
        assert TOKEN not in csv_path.read_text()
        assert TOKEN not in json_path.read_text()
    finally:
        db.close()


def test_report_snapshot_preserves_wal_delete_and_unsupported_sources(tmp_path: Path):
    wal_path = tmp_path / "wal.db"
    db = open_db(wal_path)
    db.insert_memory(content=f"password {TOKEN}", agent="owner")
    active_before = _directory_state(tmp_path)
    report = report_snapshot(wal_path, "owner")
    assert report["counts"]["literal_findings"] == 1
    assert _directory_state(tmp_path) == active_before
    db.close()

    raw = sqlite3.connect(wal_path)
    assert raw.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    raw.commit()
    raw.close()
    closed_wal_before = _directory_state(tmp_path)
    closed_wal_report = report_snapshot(wal_path, "owner")
    assert closed_wal_report["counts"]["literal_findings"] == 1
    assert _directory_state(tmp_path) == closed_wal_before

    raw = sqlite3.connect(wal_path)
    assert raw.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
    raw.close()
    delete_before = _directory_state(tmp_path)
    assert report_snapshot(wal_path, "owner")["counts"]["literal_findings"] == 1
    assert _directory_state(tmp_path) == delete_before

    missing = tmp_path / "missing.db"
    with pytest.raises(HygieneValidationError):
        report_snapshot(missing, "owner")
    assert not missing.exists()
    unsupported = tmp_path / "unsupported.db"
    raw = sqlite3.connect(unsupported)
    raw.execute("CREATE TABLE marker(value TEXT)")
    raw.commit()
    raw.close()
    unsupported_before = _directory_state(tmp_path)
    with pytest.raises(HygieneValidationError):
        report_snapshot(unsupported, "owner")
    assert _directory_state(tmp_path) == unsupported_before


def test_formula_cells_are_prefixed_in_csv(tmp_path: Path):
    db = open_db(tmp_path / "formula.db")
    try:
        memory_id = db.insert_memory(content="=SUM(1,2) password", agent="owner")
        report = report_snapshot(db.path, "owner")
        assert any(row["memory_id"] == memory_id for row in report["rows"])
        output = tmp_path / "formula.csv"
        write_report(report, output)
        assert "'=SUM" in output.read_text()
    finally:
        db.close()


def test_dry_run_and_audited_update_forget_merge(tmp_path: Path):
    db, owner, pointer, _, _ = _seed(tmp_path / "store.db")
    try:
        report = report_snapshot(db.path, "owner")
        report_manifest = _report_manifest(report, tmp_path, "update-report")
        update_manifest = create_approval_manifest(
            report,
            approver="human",
            reason="approved fixture update",
            operations=[_operation(report, owner, "update", content="safe replacement")],
            report_manifest_path=report_manifest,
        )
        before = (db.memory_generation, len(db.list_audit()))
        dry = apply_hygiene(db.path, update_manifest, agent_id="owner", dry_run=True)
        assert dry["status"] == "dry-run"
        assert (db.memory_generation, len(db.list_audit())) == before
        receipt_dir = tmp_path / "update-receipts"
        applied = apply_hygiene(
            db.path, update_manifest, agent_id="owner", receipt_dir=receipt_dir
        )
        assert applied["status"] == "applied"
        replacement = applied["receipts"][0]["replacement_id"]
        receipt_files = list(receipt_dir.glob("*.json"))
        assert receipt_files and all(TOKEN not in path.read_text() for path in receipt_files)
        assert db.get_memory(owner)["status"] == "superseded"
        assert db.get_memory(replacement)["content"] == "safe replacement"
        assert db.list_audit(action="update")[0]["id"] in applied["receipts"][0]["audit_ids"]

        report = report_snapshot(db.path, "owner")
        report_manifest = _report_manifest(report, tmp_path, "forget-report")
        forget_manifest = create_approval_manifest(
            report,
            approver="human",
            reason="approved fixture forget",
            operations=[_operation(report, pointer, "forget")],
            report_manifest_path=report_manifest,
        )
        forgotten = apply_hygiene(db.path, forget_manifest, agent_id="owner")
        assert forgotten["status"] == "applied"
        assert db.get_memory(pointer)["status"] == "forgotten"
        replayed_forget = apply_hygiene(db.path, forget_manifest, agent_id="owner")
        assert replayed_forget["receipts"][0]["status"] == "reconciled"
        assert len(db.list_audit(action="forget")) == 1
    finally:
        db.close()


def test_approved_merge_uses_existing_lifecycle_projections(tmp_path: Path):
    db = open_db(tmp_path / "merge.db")
    try:
        first = db.insert_memory(
            content="api_key=sk-" + "proj-MergeFixture1234567890", agent="owner"
        )
        second = db.insert_memory(
            content="token=tok-MergeFixture1234567890", agent="owner"
        )
        report = report_snapshot(db.path, "owner")
        rows = {
            row["memory_id"]: row
            for row in report["rows"]
            if row["memory_id"] in {first, second}
        }
        report_manifest = _report_manifest(report, tmp_path, "merge-report")
        manifest = create_approval_manifest(
            report,
            approver="human",
            reason="approved merge fixture",
            operations=[
                {
                    "action": "merge",
                    "memory_ids": [first, second],
                    "expected_fingerprints": {
                        first: rows[first]["fingerprint"],
                        second: rows[second]["fingerprint"],
                    },
                    "content": "safe merged replacement",
                }
            ],
            report_manifest_path=report_manifest,
        )
        result = apply_hygiene(db.path, manifest, agent_id="owner")
        assert result["status"] == "applied"
        replacement = db.get_memory(result["receipts"][0]["replacement_id"])
        assert replacement["content"] == "safe merged replacement"
        assert db.get_memory(first)["status"] == "superseded"
        assert db.get_memory(second)["status"] == "superseded"
        assert db.list_audit(action="merge")
    finally:
        db.close()


def test_stale_foreign_locked_and_concurrent_rows_stop_without_replay(tmp_path: Path):
    db, owner, _, locked, foreign = _seed(tmp_path / "store.db")
    try:
        report = report_snapshot(db.path, "owner")
        report_manifest = _report_manifest(report, tmp_path, "stale-report")
        stale_manifest = create_approval_manifest(
            report,
            approver="human",
            reason="stale fixture",
            operations=[_operation(report, owner, "forget")],
            report_manifest_path=report_manifest,
        )
        db.set_memory_field(owner, "tags", ["changed"], actor="test")
        stale = apply_hygiene(db.path, stale_manifest, agent_id="owner")
        assert stale["status"] == "partial"
        assert stale["applied"] == 0
        assert stale["conflicted"] == 1
        assert stale["no_op"] == 0
        assert stale["unperformed"] == 0
        assert db.get_memory(owner)["status"] == "active"

        current = report_snapshot(db.path, "owner")
        current_manifest = _report_manifest(current, tmp_path, "current-report")
        with pytest.raises(HygieneValidationError):
            create_approval_manifest(
                current,
                approver="human",
                reason="foreign fixture",
                operations=[
                    {
                        "memory_id": foreign,
                        "expected_fingerprint": memory_fingerprint(db.get_memory(foreign) or {}),
                        "action": "forget",
                    }
                ],
                report_manifest_path=current_manifest,
            )
        with pytest.raises(HygieneValidationError):
            create_approval_manifest(
                current,
                approver="human",
                reason="locked fixture",
                operations=[
                    {
                        "memory_id": locked,
                        "expected_fingerprint": memory_fingerprint(db.get_memory(locked) or {}),
                        "action": "forget",
                    }
                ],
                report_manifest_path=current_manifest,
            )
        assert db.get_memory(locked)["status"] == "active"

        race = open_db(db.path)
        try:
            race_report = report_snapshot(db.path, "owner")
            race_manifest_path = _report_manifest(race_report, tmp_path, "race-report")
            race_manifest = create_approval_manifest(
                race_report,
                approver="human",
                reason="concurrency fixture",
                operations=[_operation(race_report, owner, "forget")],
                report_manifest_path=race_manifest_path,
            )
            race.set_memory_field(owner, "metadata", {"changed": True}, actor="race")
            result = apply_hygiene(db.path, race_manifest, agent_id="owner")
            assert result["status"] == "partial"
            assert db.get_memory(owner)["status"] == "active"
        finally:
            race.close()
    finally:
        db.close()


def test_batch_counts_distinguish_noop_conflict_and_stopped_tail(tmp_path: Path):
    db = open_db(tmp_path / "counts.db")
    try:
        noop_id = db.insert_memory(content=f"noop {TOKEN}", agent="owner")
        applied_id = db.insert_memory(content=f"apply {TOKEN}", agent="owner")
        conflict_id = db.insert_memory(content=f"conflict {TOKEN}", agent="owner")
        tail_id = db.insert_memory(content=f"tail {TOKEN}", agent="owner")
        report = report_snapshot(db.path, "owner")
        report_manifest = _report_manifest(report, tmp_path, "counts-report")
        operations = [
            _operation(report, noop_id, "keep"),
            _operation(report, applied_id, "update", content="safe applied"),
            _operation(report, conflict_id, "forget"),
            _operation(report, tail_id, "forget"),
        ]
        manifest = create_approval_manifest(
            report,
            approver="human",
            reason="mixed count fixture",
            operations=operations,
            report_manifest_path=report_manifest,
        )
        db.set_memory_field(conflict_id, "tags", ["changed"], actor="test")
        result = apply_hygiene(db.path, manifest, agent_id="owner")
        assert result["status"] == "partial"
        assert result["applied"] == 1
        assert result["conflicted"] == 1
        assert result["no_op"] == 1
        assert result["unperformed"] == 1
        tail = db.get_memory(tail_id)
        assert tail is not None and tail["status"] == "active"
    finally:
        db.close()


def test_crash_after_commit_reconciles_audit_without_duplicate(tmp_path: Path, monkeypatch):
    db, owner, _, _, _ = _seed(tmp_path / "store.db")
    try:
        report = report_snapshot(db.path, "owner")
        report_manifest = _report_manifest(report, tmp_path, "crash-report")
        manifest = create_approval_manifest(
            report,
            approver="human",
            reason="crash replay fixture",
            operations=[_operation(report, owner, "update", content="replayed safely")],
            report_manifest_path=report_manifest,
        )
        receipt_dir = tmp_path / "receipts"
        import remnant.hygiene as hygiene

        original = hygiene._write_receipt
        monkeypatch.setattr(
            hygiene,
            "_write_receipt",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("crash")),
        )
        with pytest.raises(RuntimeError, match="crash"):
            apply_hygiene(db.path, manifest, agent_id="owner", receipt_dir=receipt_dir)
        monkeypatch.setattr(hygiene, "_write_receipt", original)
        replay = apply_hygiene(db.path, manifest, agent_id="owner", receipt_dir=receipt_dir)
        assert replay["status"] == "applied"
        assert replay["receipts"][0]["status"] == "reconciled"
        assert len(db.list_audit(action="update")) == 1
    finally:
        db.close()


def test_commit_before_receipt_reconciles_forget_update_and_merge(
    tmp_path: Path, monkeypatch
):
    db = open_db(tmp_path / "crash-all.db")
    try:
        forget_id = db.insert_memory(content=f"forget {TOKEN}", agent="owner")
        update_id = db.insert_memory(content=f"update {TOKEN}", agent="owner")
        merge_a = db.insert_memory(content=f"merge-a {TOKEN}", agent="owner")
        merge_b = db.insert_memory(content=f"merge-b {TOKEN}", agent="owner")
        report = report_snapshot(db.path, "owner")
        report_manifest = _report_manifest(report, tmp_path, "crash-all-report")
        rows = {row["memory_id"]: row for row in report["rows"]}
        operations = [
            ("forget", _operation(report, forget_id, "forget")),
            ("update", _operation(report, update_id, "update", content="safe update")),
            (
                "merge",
                {
                    "action": "merge",
                    "memory_ids": [merge_a, merge_b],
                    "expected_fingerprints": {
                        merge_a: rows[merge_a]["fingerprint"],
                        merge_b: rows[merge_b]["fingerprint"],
                    },
                    "content": "safe merge",
                },
            ),
        ]
        import remnant.hygiene as hygiene

        original = hygiene._write_receipt
        for name, operation in operations:
            manifest = create_approval_manifest(
                report,
                approver="human",
                reason=f"crash {name}",
                operations=[operation],
                report_manifest_path=report_manifest,
            )
            monkeypatch.setattr(
                hygiene,
                "_write_receipt",
                lambda *_args: (_ for _ in ()).throw(RuntimeError("crash")),
            )
            with pytest.raises(RuntimeError, match="crash"):
                apply_hygiene(
                    db.path,
                    manifest,
                    agent_id="owner",
                    receipt_dir=tmp_path / f"receipts-{name}",
                )
            monkeypatch.setattr(hygiene, "_write_receipt", original)
            replay = apply_hygiene(
                db.path,
                manifest,
                agent_id="owner",
                receipt_dir=tmp_path / f"receipts-{name}",
            )
            assert replay["receipts"][0]["status"] == "reconciled"
            assert len(db.list_audit(action=name)) == 1
    finally:
        db.close()


def test_audit_failure_rolls_back_hygiene_lifecycle(tmp_path: Path, monkeypatch):
    db = open_db(tmp_path / "rollback.db")
    try:
        memory_id = db.insert_memory(content="old", agent="owner")
        fingerprint = memory_fingerprint(db.get_memory(memory_id))

        def fail(*_args, **_kwargs):
            raise RuntimeError("audit failure")

        monkeypatch.setattr(db, "_write_audit", fail)
        with pytest.raises(RuntimeError, match="audit failure"):
            MemoryLifecycle(db, RemnantConfig(agent_id="owner"), _Embedder()).forget(
                memory_id,
                actor="human",
                agent_id="owner",
                expected_fingerprint=fingerprint,
                operation_id="rollback-test",
            )
        assert db.get_memory(memory_id)["status"] == "active"
        assert db.list_audit() == []
    finally:
        db.close()


def test_approval_manifest_binds_report_and_rejects_ambiguous_operations(tmp_path: Path):
    db, owner, _, _, _ = _seed(tmp_path / "binding.db")
    try:
        report = report_snapshot(db.path, "owner")
        report_paths = write_report(report, tmp_path / "report.csv")
        operation = _operation(report, owner, "forget")
        manifest = create_approval_manifest(
            report,
            approver="human",
            reason="binding fixture",
            operations=[operation],
            report_manifest_path=report_paths["manifest"],
        )
        result = apply_hygiene(db.path, manifest, agent_id="owner", dry_run=True)
        assert result["status"] == "dry-run"

        tampered = dict(manifest, kind="other")
        with pytest.raises(HygieneValidationError):
            apply_hygiene(db.path, tampered, agent_id="owner", dry_run=True)

        ambiguous = dict(manifest)
        ambiguous["operations"] = [
            {
                **operation,
                "memory_ids": [owner, owner],
                "expected_fingerprints": {owner: operation["expected_fingerprint"]},
            }
        ]
        with pytest.raises(HygieneValidationError):
            apply_hygiene(db.path, ambiguous, agent_id="owner", dry_run=True)
    finally:
        db.close()


def test_report_binding_rejects_missing_tampered_and_unreported_evidence(
    tmp_path: Path, monkeypatch
):
    db, owner, _, _, _ = _seed(tmp_path / "binding-strict.db")
    try:
        report = report_snapshot(db.path, "owner")
        paths = write_report(report, tmp_path / "source.json")
        operation = _operation(report, owner, "forget")
        with pytest.raises(HygieneValidationError):
            create_approval_manifest(
                report, approver="human", reason="missing binding", operations=[operation]
            )
        manifest = create_approval_manifest(
            report,
            approver="human",
            reason="strict binding",
            operations=[operation],
            report_manifest_path=paths["manifest"],
        )

        report_path = Path(paths["report"])
        original_report = report_path.read_text()
        report_path.write_text(original_report + "\n")
        with pytest.raises(HygieneValidationError):
            apply_hygiene(db.path, manifest, agent_id="owner", dry_run=True)
        report_path.write_text(original_report)

        fabricated = dict(manifest, report_hash="0" * 64)
        with pytest.raises(HygieneValidationError):
            apply_hygiene(db.path, fabricated, agent_id="owner")

        unreported = db.insert_memory(content=f"unreported {TOKEN}", agent="owner")
        unreported_manifest = dict(manifest)
        unreported_manifest["operations"] = [
            {
                "memory_id": unreported,
                "expected_fingerprint": memory_fingerprint(db.get_memory(unreported) or {}),
                "action": "forget",
            }
        ]
        with pytest.raises(HygieneValidationError):
            apply_hygiene(db.path, unreported_manifest, agent_id="owner")

        import remnant.hygiene as hygiene

        class _UnexpectedWritableOpen:
            def __init__(self, *_args, **_kwargs):
                raise AssertionError("writable DB opened before report verification")

        monkeypatch.setattr(hygiene, "RemnantDB", _UnexpectedWritableOpen)
        with pytest.raises(HygieneValidationError):
            apply_hygiene(db.path, fabricated, agent_id="owner")
    finally:
        db.close()
