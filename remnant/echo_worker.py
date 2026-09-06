"""Crash-safe, budgeted background processing for Echo jobs."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from .config import RemnantConfig
from .echo_policy import signal_spec, viewer_key_hash
from .echo_store import EchoRepository
from .echo_types import EchoSignalInput


class EchoWorker:
    """Process sampled receipts off the Hermes response path."""

    def __init__(
        self,
        service: Any,
        config: RemnantConfig,
        *,
        evaluator: Callable[[dict[str, Any], dict[str, Any]], list[dict[str, Any]]] | None = None,
        model_busy: Callable[[], bool] | None = None,
    ) -> None:
        self.service = service
        self.config = config
        self.store: EchoRepository = service.store
        self.evaluator = evaluator
        self.model_busy = model_busy
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="remnant-echo", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def wake(self) -> None:
        self._wake.set()

    def run_once(self) -> bool:
        """Process one job; return whether work was claimed."""
        self.store.reclaim_stale_jobs()
        if self.model_busy is not None and self.config.echo_pause_when_model_busy:
            try:
                if self.model_busy():
                    return False
            except Exception:
                pass
        status = self.store.daily_budget_status(agent_id=self.config.agent_id)
        if status["jobs"] >= self.config.echo_max_jobs_per_day:
            return False
        if status["evaluator_seconds"] >= self.config.echo_max_evaluator_seconds_per_day:
            return False
        job = self.store.claim_job()
        if job is None:
            return False
        payload = self.store.job_payload(job)
        if payload is None:
            self.store.complete_job(int(job["id"]), status="skipped", error="payload unavailable")
            return True
        if self.evaluator is None:
            self.store.complete_job(int(job["id"]), status="skipped", error="evaluator unavailable")
            return True
        started = time.perf_counter()
        try:
            signals = self.evaluator(job, payload)
            self._persist_signals(job, payload, signals)
        except Exception as exc:
            self.store.fail_job(int(job["id"]), error=str(exc), retry=True)
            return True
        elapsed = time.perf_counter() - started
        self.store.record_daily_metric(
            agent_id=str(payload.get("agent_id") or self.config.agent_id),
            metric="evaluator_seconds",
            total=elapsed,
            maximum=elapsed,
        )
        self.store.complete_job(int(job["id"]), status="done")
        self.service.aggregate(limit=100)
        return True

    def _persist_signals(
        self,
        job: dict[str, Any],
        payload: dict[str, Any],
        signals: list[dict[str, Any]],
    ) -> None:
        target_ids = {str(item) for item in payload.get("target_ids") or []}
        agent_id = str(payload.get("agent_id") or self.config.agent_id)
        viewer_hash = str(payload.get("viewer_key_hash") or viewer_key_hash(agent_id))
        archetype = str(payload.get("query_archetype") or "unknown")
        version = str(job.get("evaluator_version") or self.config.echo_policy_version)
        for raw in signals[: len(target_ids)]:
            if not isinstance(raw, dict):
                continue
            memory_id = str(raw.get("memory_id") or "")
            signal_type = str(raw.get("signal_type") or "")
            spec = signal_spec(signal_type)
            if memory_id not in target_ids or spec is None:
                continue
            direction, weight, source = spec
            paired_memory_id = None
            if str(job.get("job_type")) == "pair" and len(target_ids) == 2:
                paired_memory_id = next(
                    (item for item in target_ids if item != memory_id), None
                )
            try:
                confidence = max(0.0, min(1.0, float(raw.get("confidence", 1.0))))
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence <= 0:
                continue
            self.store.record_signal(
                EchoSignalInput(
                    memory_id=memory_id,
                    agent_id=agent_id,
                    viewer_key_hash=viewer_hash,
                    query_archetype=archetype,
                    signal_type=signal_type,
                    direction=direction,
                    weight=weight * confidence,
                    source=source,
                    receipt_id=str(payload.get("id") or job.get("receipt_id")),
                    paired_memory_id=paired_memory_id,
                    evaluator_version=version,
                )
            )

    def _run(self) -> None:
        next_compaction = time.monotonic() + 300.0
        while not self._stop.is_set():
            did_work = False
            try:
                did_work = self.run_once()
                if time.monotonic() >= next_compaction:
                    next_compaction = time.monotonic() + 300.0
                    self.service.compact()
            except Exception:
                # A worker failure must never take down the provider.
                did_work = False
            if did_work:
                continue
            self._wake.wait(timeout=max(0.1, float(self.config.echo_worker_poll_interval_s)))
            self._wake.clear()


__all__ = ["EchoWorker"]
