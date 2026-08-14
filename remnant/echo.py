"""Remnant Echo orchestration and baseline-safe ranking integration."""

from __future__ import annotations

import hashlib
import time
import uuid
from collections import defaultdict
from typing import Any

from .config import RemnantConfig
from .echo_policy import (
    GLOBAL_ARCHETYPE,
    activation_key,
    classify_query,
    combine_utility_rows,
    fingerprint,
    profile_scope_hash,
    signal_spec,
    viewer_key_hash,
)
from .echo_store import EchoRepository
from .echo_types import EchoDiagnostics, EchoReceiptDraft, EchoSignalInput


class EchoService:
    """Keep Echo optional, bounded, and independent from claim truth."""

    def __init__(self, db: Any, config: RemnantConfig):
        self.db = db
        self.config = config
        self.store = EchoRepository(db, config)
        self.viewer_key = config.agent_id

    @property
    def policy_version(self) -> str:
        return str(self.config.echo_policy_version)

    def build_receipt_draft(
        self,
        *,
        query: str,
        session_id: str,
        agent_id: str,
        viewer_key: str | None,
        profile_scope: list[str] | None,
        memory_generation: int,
        context: Any,
    ) -> EchoReceiptDraft | None:
        if not self.config.echo_enabled or context is None or not context.text:
            return None
        if not context.items:
            return None
        viewer_hash = viewer_key_hash(viewer_key or agent_id)
        query_hash = fingerprint(query)
        context_hash = hashlib.sha256(context.text.encode("utf-8")).hexdigest()
        scope_hash = profile_scope_hash(profile_scope)
        return EchoReceiptDraft(
            activation_key=activation_key(
                viewer_hash=viewer_hash,
                session_id=session_id,
                query_hash=query_hash,
                context_hash=context_hash,
                generation=memory_generation,
                policy_version=self.policy_version,
            ),
            session_id=session_id,
            agent_id=agent_id,
            viewer_key_hash=viewer_hash,
            profile_scope_hash=scope_hash,
            query_fingerprint=query_hash,
            query_archetype=classify_query(query),
            context_hash=context_hash,
            memory_generation=memory_generation,
            token_count=context.token_count,
            policy_version=self.policy_version,
            items=tuple(context.items),
        )

    def activate_receipt(self, draft: EchoReceiptDraft) -> str | None:
        if draft is None or not self.config.echo_enabled:
            return None
        receipt_id = str(uuid.uuid4())
        try:
            inserted = self.store.activate_receipt(draft, receipt_id=receipt_id)
        except Exception:
            return None
        if not inserted:
            # Look up the existing receipt by activation key for observability.
            return self.receipt_id_for_activation(draft.activation_key)
        self._maybe_queue_jobs(draft, receipt_id)
        return receipt_id

    def receipt_id_for_activation(self, activation_key_value: str) -> str | None:
        with self.db.read() as cur:
            cur.execute(
                "SELECT id FROM echo_receipts WHERE activation_key=?",
                (activation_key_value,),
            )
            row = cur.fetchone()
        return str(row["id"]) if row else None

    def close_receipt(
        self,
        *,
        session_id: str,
        viewer_key: str | None,
        query: str,
        turn_id: int,
    ) -> str | None:
        if not self.config.echo_enabled:
            return None
        try:
            return self.store.close_receipt(
                session_id=session_id,
                viewer_key_hash=viewer_key_hash(viewer_key),
                query_fingerprint=fingerprint(query),
                turn_id=turn_id,
            )
        except Exception:
            return None

    def adjust_results(
        self,
        results: list[dict[str, Any]],
        *,
        query: str,
        agent_id: str,
        viewer_key: str | None,
    ) -> tuple[list[dict[str, Any]], EchoDiagnostics]:
        archetype = classify_query(query)
        influence = 0.0 if self.config.echo_shadow_mode else self.config.echo_rank_influence
        if not self.config.echo_enabled or not results:
            return results, EchoDiagnostics(archetype, self.policy_version, 0.0)
        ids = [str(row.get("id")) for row in results if row.get("id") and not row.get("pending")]
        started = time.perf_counter()
        if not ids:
            return results, EchoDiagnostics(
                archetype,
                self.policy_version,
                influence,
                budget_bypassed=True,
            )
        try:
            rows = self.store.utility_rows(
                memory_ids=ids,
                agent_id=agent_id,
                viewer_key_hash=viewer_key_hash(viewer_key or agent_id),
                query_archetype=archetype,
                policy_version=self.policy_version,
            )
        except Exception:
            return results, EchoDiagnostics(archetype, self.policy_version, influence)
        lookup_ms = (time.perf_counter() - started) * 1000.0
        if (
            self.config.echo_disable_on_budget_exceeded
            and lookup_ms > self.config.echo_hot_path_budget_ms
        ):
            return results, EchoDiagnostics(
                archetype,
                self.policy_version,
                influence,
                budget_bypassed=True,
                details={"utility_lookup_ms": round(lookup_ms, 3)},
            )
        by_memory: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_memory[str(row["memory_id"])].append(row)
        adjusted: list[dict[str, Any]] = []
        changed = 0
        details: dict[str, Any] = {}
        shadow_scores: dict[str, float] = {}
        for base_rank, row in enumerate(results):
            item = dict(row)
            memory_id = str(item.get("id") or "")
            view = combine_utility_rows(
                by_memory.get(memory_id, []),
                memory_id=memory_id,
                archetype=archetype,
                config=self.config,
            )
            if view is None:
                adjusted.append(item)
                continue
            applied = view.adjustment * influence
            base_score = float(item.get("score") or 0.0)
            shadow_scores[memory_id] = base_score + view.adjustment
            echo_detail = {
                "archetype": archetype,
                "utility_mean": round(view.utility_mean, 6),
                "harm_risk": round(view.harm_risk, 6),
                "confidence": round(view.confidence, 6),
                "observations": round(view.observations, 6),
                "base_adjustment": round(view.adjustment, 6),
                "applied_adjustment": round(applied, 6),
                "shadow_score": round(base_score + view.adjustment, 6),
                "policy": self.policy_version,
            }
            details[memory_id] = echo_detail
            if influence:
                item["base_score"] = float(item.get("score") or 0.0)
                item["score"] = float(item.get("score") or 0.0) + applied
                ranking = dict(item.get("ranking") or {})
                ranking["echo"] = echo_detail
                item["ranking"] = ranking
            if abs(applied) > 0:
                changed += 1
            adjusted.append(item)
        shadow_order = sorted(
            shadow_scores,
            key=lambda memory_id: (-shadow_scores[memory_id], memory_id),
        )
        for shadow_rank, memory_id in enumerate(shadow_order):
            if memory_id in details:
                details[memory_id]["shadow_rank"] = shadow_rank
        if influence:
            adjusted.sort(
                key=lambda item: (
                    -float(item.get("score") or 0.0),
                    str(item.get("id") or ""),
                )
            )
        return adjusted, EchoDiagnostics(
            archetype,
            self.policy_version,
            influence,
            changed_count=changed,
            utility_hits=len(rows),
            details={"utility_lookup_ms": round(lookup_ms, 3), **details},
        )

    def record_feedback(
        self,
        *,
        memory_id: str,
        feedback: str,
        agent_id: str,
        viewer_key: str | None,
        query: str | None = None,
        receipt_id: str | None = None,
    ) -> int:
        normalized = str(feedback or "").strip().casefold()
        signal_type = "explicit_useful" if normalized == "useful" else "explicit_wrong"
        spec = signal_spec(signal_type)
        if spec is None:
            return 0
        direction, weight, source = spec
        archetypes = [GLOBAL_ARCHETYPE]
        if query:
            archetypes.append(classify_query(query))
        count = 0
        for archetype in dict.fromkeys(archetypes):
            count += self.store.record_signal(
                EchoSignalInput(
                    memory_id=str(memory_id),
                    agent_id=agent_id,
                    viewer_key_hash=viewer_key_hash(viewer_key or agent_id),
                    query_archetype=archetype,
                    signal_type=signal_type,
                    direction=direction,
                    weight=weight,
                    source=source,
                    receipt_id=receipt_id,
                    evaluator_version=self.policy_version,
                )
            )
        return count

    def aggregate(self, *, limit: int = 100) -> int:
        try:
            return self.store.aggregate_pending(limit=limit)
        except Exception:
            return 0

    def compact(self) -> dict[str, int]:
        try:
            return self.store.compact()
        except Exception:
            return {}

    def health(self) -> dict[str, Any]:
        try:
            return self.store.health()
        except Exception:
            return {"available": False}

    def _maybe_queue_jobs(self, draft: EchoReceiptDraft, receipt_id: str) -> None:
        """Queue deterministic single-item jobs; worker availability is optional."""
        if not self.config.echo_enabled or not draft.items:
            return
        memory_ids = [item.memory_id for item in draft.items if item.item_kind == "memory"]
        if not memory_ids:
            return
        digest = int(hashlib.sha256(draft.activation_key.encode("utf-8")).hexdigest(), 16)
        sample_rate = self.config.echo_initial_sample_rate
        try:
            existing = self.store.utility_rows(
                memory_ids=memory_ids,
                agent_id=draft.agent_id,
                viewer_key_hash=draft.viewer_key_hash,
                query_archetype=draft.query_archetype,
                policy_version=draft.policy_version,
            )
        except Exception:
            existing = []
        if any(
            float(row.get("effective_observations") or 0.0)
            >= self.config.echo_mature_observations
            for row in existing
        ):
            sample_rate = self.config.echo_mature_sample_rate
        if digest / float(1 << 256) >= sample_rate:
            return
        for memory_id in memory_ids[: max(1, self.config.echo_max_pairs_per_receipt)]:
            try:
                self.store.enqueue_job(
                    receipt_id=receipt_id,
                    job_type="single",
                    target_ids=[memory_id],
                    priority=1.0,
                    evaluator_version=self.policy_version,
                )
            except Exception:
                continue
        if self.config.echo_pair_attribution_enabled and len(memory_ids) >= 2:
            pair_limit = min(
                self.config.echo_max_pairs_per_receipt,
                len(memory_ids) - 1,
            )
            for index in range(pair_limit):
                try:
                    self.store.enqueue_job(
                        receipt_id=receipt_id,
                        job_type="pair",
                        target_ids=memory_ids[index : index + 2],
                        priority=0.5,
                        evaluator_version=self.policy_version,
                    )
                except Exception:
                    continue


__all__ = ["EchoService"]
