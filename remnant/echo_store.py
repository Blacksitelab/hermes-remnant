"""SQLite repository for Remnant Echo.

This module intentionally keeps Echo persistence outside the large legacy DB
CRUD class while reusing its single transaction/locking boundary.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from typing import Any

from .config import RemnantConfig
from .db import RemnantDB
from .echo_policy import GLOBAL_ARCHETYPE, canonical_pair
from .echo_types import EchoReceiptDraft, EchoSignalInput


def _decay(value: float, age_s: float, half_life_days: float) -> float:
    if value <= 0:
        return 0.0
    return value * 0.5 ** (max(0.0, age_s) / (half_life_days * 86400.0))


class EchoRepository:
    """Bounded Echo storage using RemnantDB's serialized transactions."""

    def __init__(self, db: RemnantDB, config: RemnantConfig):
        self.db = db
        self.config = config

    def activate_receipt(self, draft: EchoReceiptDraft, *, receipt_id: str) -> bool:
        now = time.time()
        with self.db.transaction() as cur:
            cur.execute(
                """INSERT OR IGNORE INTO echo_receipts(
                    id, activation_key, session_id, agent_id, viewer_key_hash,
                    profile_scope_hash, query_fingerprint, query_archetype,
                    context_hash, memory_generation, rendered_count, token_count,
                    policy_version, status, created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    receipt_id,
                    draft.activation_key,
                    draft.session_id,
                    draft.agent_id,
                    draft.viewer_key_hash,
                    draft.profile_scope_hash,
                    draft.query_fingerprint,
                    draft.query_archetype,
                    draft.context_hash,
                    draft.memory_generation,
                    len(draft.items),
                    draft.token_count,
                    draft.policy_version,
                    "open",
                    now,
                ),
            )
            inserted = cur.rowcount > 0
            if inserted:
                for item in draft.items:
                    cur.execute(
                        """INSERT INTO echo_receipt_items(
                            receipt_id, memory_id, ordinal, item_kind, source_turn_id,
                            evidence_class, score_lane, base_score, base_rank,
                            rendered_tokens, rendered_hash, claim_status
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            receipt_id,
                            item.memory_id,
                            item.ordinal,
                            item.item_kind,
                            item.source_turn_id,
                            item.evidence_class,
                            item.score_lane,
                            item.base_score,
                            item.base_rank,
                            item.rendered_tokens,
                            item.rendered_hash,
                            item.claim_status,
                        ),
                    )
        return inserted

    def close_receipt(
        self,
        *,
        session_id: str,
        viewer_key_hash: str,
        query_fingerprint: str,
        turn_id: int,
        max_age_s: float = 300.0,
    ) -> str | None:
        cutoff = time.time() - max(1.0, float(max_age_s))
        now = time.time()
        with self.db.transaction() as cur:
            cur.execute(
                """SELECT id FROM echo_receipts
                   WHERE session_id=? AND viewer_key_hash=?
                     AND query_fingerprint=? AND status='open' AND created_at>=?
                   ORDER BY created_at DESC LIMIT 1""",
                (session_id, viewer_key_hash, query_fingerprint, cutoff),
            )
            row = cur.fetchone()
            if row is None:
                return None
            receipt_id = str(row["id"])
            cur.execute(
                "UPDATE echo_receipts SET turn_id=?, status='closed', outcome=?, closed_at=? "
                "WHERE id=? AND status='open'",
                (turn_id, "turn_persisted", now, receipt_id),
            )
            return receipt_id

    def expire_open_receipts(self, *, max_age_s: float = 300.0) -> int:
        cutoff = time.time() - max(1.0, float(max_age_s))
        with self.db.transaction() as cur:
            cur.execute(
                "UPDATE echo_receipts SET status='expired', outcome='unmatched' "
                "WHERE status='open' AND created_at<?",
                (cutoff,),
            )
            expired = max(0, int(cur.rowcount))
            cur.execute(
                "UPDATE echo_jobs SET status='skipped', completed_at=?, "
                "last_error='receipt expired before turn persisted' "
                "WHERE status='pending' AND receipt_id IN "
                "(SELECT id FROM echo_receipts WHERE status='expired')",
                (time.time(),),
            )
            return expired

    def get_receipt(self, receipt_id: str) -> dict[str, Any] | None:
        with self.db.read() as cur:
            cur.execute("SELECT * FROM echo_receipts WHERE id=?", (receipt_id,))
            row = cur.fetchone()
            if row is None:
                return None
            result = dict(row)
            cur.execute(
                "SELECT * FROM echo_receipt_items WHERE receipt_id=? ORDER BY ordinal",
                (receipt_id,),
            )
            result["items"] = [dict(item) for item in cur.fetchall()]
            return result

    def record_signal(self, signal: EchoSignalInput) -> int:
        direction = 1 if signal.direction >= 0 else -1
        weight = max(0.0001, min(1.0, float(signal.weight)))
        with self.db.transaction() as cur:
            cur.execute(
                """INSERT INTO echo_signals(
                    receipt_id, memory_id, paired_memory_id, agent_id,
                    viewer_key_hash, query_archetype, signal_type, direction,
                    weight, source, evaluator_version, created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    signal.receipt_id,
                    signal.memory_id,
                    signal.paired_memory_id,
                    signal.agent_id,
                    signal.viewer_key_hash,
                    signal.query_archetype,
                    signal.signal_type,
                    direction,
                    weight,
                    signal.source,
                    signal.evaluator_version,
                    time.time(),
                ),
            )
            return int(cur.lastrowid)

    def utility_rows(
        self,
        *,
        memory_ids: list[str],
        agent_id: str,
        viewer_key_hash: str,
        query_archetype: str,
        policy_version: str,
    ) -> list[dict[str, Any]]:
        ids = [str(item) for item in memory_ids if str(item)]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        archetypes = [query_archetype]
        if query_archetype != GLOBAL_ARCHETYPE:
            archetypes.append(GLOBAL_ARCHETYPE)
        arch_placeholders = ",".join("?" for _ in archetypes)
        params: list[Any] = [agent_id, viewer_key_hash, policy_version, *archetypes, *ids]
        with self.db.read() as cur:
            cur.execute(
                f"""SELECT * FROM echo_utility
                    WHERE agent_id=? AND viewer_key_hash=? AND policy_version=?
                      AND query_archetype IN ({arch_placeholders})
                      AND memory_id IN ({placeholders})""",
                params,
            )
            return [dict(row) for row in cur.fetchall()]

    def enqueue_job(
        self,
        *,
        receipt_id: str,
        job_type: str,
        target_ids: list[str],
        priority: float,
        evaluator_version: str,
    ) -> int | None:
        if job_type not in {"single", "pair"} or len(target_ids) not in {1, 2}:
            return None
        if job_type == "pair" and len(target_ids) != 2:
            return None
        encoded = json.dumps([str(item) for item in target_ids], separators=(",", ":"))
        with self.db.transaction() as cur:
            cur.execute(
                """SELECT id FROM echo_jobs
                   WHERE receipt_id=? AND job_type=? AND target_ids=?
                     AND evaluator_version=? AND status IN ('pending','running')
                   LIMIT 1""",
                (receipt_id, job_type, encoded, evaluator_version),
            )
            existing = cur.fetchone()
            if existing is not None:
                return int(existing["id"])
            cur.execute(
                """INSERT INTO echo_jobs(
                    receipt_id, job_type, target_ids, priority, evaluator_version,
                    created_at
                ) VALUES(?,?,?,?,?,?)""",
                (receipt_id, job_type, encoded, float(priority), evaluator_version, time.time()),
            )
            return int(cur.lastrowid)

    def claim_job(self) -> dict[str, Any] | None:
        now = time.time()
        with self.db.transaction() as cur:
            cur.execute(
                """SELECT echo_jobs.* FROM echo_jobs
                   JOIN echo_receipts r ON r.id=echo_jobs.receipt_id
                   WHERE echo_jobs.status='pending' AND echo_jobs.next_attempt_at<=?
                     AND r.status='closed' AND r.turn_id IS NOT NULL AND r.agent_id=?
                   ORDER BY echo_jobs.priority DESC, echo_jobs.id LIMIT 1""",
                (now, self.config.agent_id),
            )
            row = cur.fetchone()
            if row is None:
                return None
            job_id = int(row["id"])
            cur.execute(
                "UPDATE echo_jobs SET status='running', attempts=attempts+1, started_at=? "
                "WHERE id=? AND status='pending'",
                (now, job_id),
            )
            result = dict(row)
            result["attempts"] = int(row["attempts"] or 0) + 1
            return result

    def reclaim_stale_jobs(self, *, now: float | None = None) -> int:
        """Return abandoned running jobs to the queue after a crash."""
        now = time.time() if now is None else float(now)
        cutoff = now - max(1.0, float(self.config.echo_job_stale_after_s))
        with self.db.transaction() as cur:
            cur.execute(
                "UPDATE echo_jobs SET status='pending', next_attempt_at=?, "
                "last_error=? WHERE status='running' AND started_at<? "
                "AND attempts<?",
                (
                    now,
                    "reclaimed after stale worker lease",
                    cutoff,
                    self.config.echo_job_max_attempts,
                ),
            )
            reclaimed = max(0, int(cur.rowcount))
            cur.execute(
                "UPDATE echo_jobs SET status='failed', completed_at=?, "
                "last_error=? WHERE status='running' AND started_at<? AND attempts>=?",
                (
                    now,
                    "stale worker lease exceeded retry limit",
                    cutoff,
                    self.config.echo_job_max_attempts,
                ),
            )
            return reclaimed + max(0, int(cur.rowcount))

    def job_payload(self, job: dict[str, Any]) -> dict[str, Any] | None:
        """Load the bounded, non-secret inputs required by an evaluator."""
        try:
            target_ids = json.loads(str(job.get("target_ids") or "[]"))
        except json.JSONDecodeError:
            return None
        if not isinstance(target_ids, list) or not target_ids:
            return None
        target_ids = [str(item) for item in target_ids if str(item)]
        if len(target_ids) > 2:
            return None
        with self.db.read() as cur:
            cur.execute(
                """SELECT r.*, t.user_text, t.assistant_text
                   FROM echo_receipts r LEFT JOIN turns t ON t.id=r.turn_id
                   WHERE r.id=? AND r.agent_id=? AND t.agent_id=r.agent_id""",
                (str(job.get("receipt_id") or ""), self.config.agent_id),
            )
            receipt = cur.fetchone()
            if (
                receipt is None
                or receipt["status"] != "closed"
                or receipt["turn_id"] is None
            ):
                return None
            placeholders = ",".join("?" for _ in target_ids)
            cur.execute(
                f"SELECT id, content, type, status, confidence, trust_score "
                f"FROM memories WHERE id IN ({placeholders}) AND agent=?",
                [*target_ids, self.config.agent_id],
            )
            memories = {str(row["id"]): dict(row) for row in cur.fetchall()}
        if any(
            memory_id not in memories or memories[memory_id].get("status") != "active"
            for memory_id in target_ids
        ):
            return None
        result = dict(receipt)
        result["target_ids"] = target_ids
        result["memories"] = [memories[memory_id] for memory_id in target_ids]
        return result

    def daily_budget_status(self, *, agent_id: str, now: float | None = None) -> dict[str, float]:
        """Return today's bounded Echo job/time usage."""
        now = time.time() if now is None else float(now)
        day_start = now - (now % 86400.0)
        with self.db.read() as cur:
            cur.execute(
                "SELECT COUNT(*) AS count FROM echo_jobs "
                "WHERE created_at>=? "
                "AND receipt_id IN (SELECT id FROM echo_receipts WHERE agent_id=?)",
                (day_start, agent_id),
            )
            jobs = float(cur.fetchone()["count"])
            cur.execute(
                "SELECT total FROM echo_daily_metrics WHERE day=? AND agent_id=? "
                "AND metric='evaluator_seconds'",
                (time.strftime("%Y-%m-%d", time.gmtime(now)), agent_id),
            )
            row = cur.fetchone()
            seconds = float(row["total"] or 0.0) if row else 0.0
        return {"jobs": jobs, "evaluator_seconds": seconds}

    def record_daily_metric(
        self,
        *,
        agent_id: str,
        metric: str,
        total: float = 0.0,
        count: int = 1,
        maximum: float = 0.0,
        now: float | None = None,
    ) -> None:
        now = time.time() if now is None else float(now)
        day = time.strftime("%Y-%m-%d", time.gmtime(now))
        with self.db.transaction() as cur:
            cur.execute(
                """INSERT INTO echo_daily_metrics(day, agent_id, metric, count, total, maximum)
                   VALUES(?,?,?,?,?,?)
                   ON CONFLICT(day, agent_id, metric) DO UPDATE SET
                     count=count+excluded.count,
                     total=total+excluded.total,
                     maximum=MAX(maximum, excluded.maximum)""",
                (day, agent_id, metric, int(count), float(total), float(maximum)),
            )

    def complete_job(self, job_id: int, *, status: str = "done", error: str | None = None) -> None:
        if status not in {"done", "skipped", "failed"}:
            status = "failed"
        with self.db.transaction() as cur:
            cur.execute(
                "UPDATE echo_jobs SET status=?, last_error=?, completed_at=? WHERE id=?",
                (status, error, time.time(), int(job_id)),
            )

    def fail_job(self, job_id: int, *, error: str, retry: bool) -> None:
        now = time.time()
        with self.db.transaction() as cur:
            cur.execute("SELECT attempts FROM echo_jobs WHERE id=?", (int(job_id),))
            row = cur.fetchone()
            attempts = int(row["attempts"] or 0) if row else self.config.echo_job_max_attempts
            if retry and attempts < self.config.echo_job_max_attempts:
                delay = min(300.0, 2.0 ** max(0, attempts - 1))
                cur.execute(
                    "UPDATE echo_jobs SET status='pending', next_attempt_at=?, "
                    "last_error=? WHERE id=?",
                    (now + delay, str(error)[:500], int(job_id)),
                )
            else:
                cur.execute(
                    "UPDATE echo_jobs SET status='failed', last_error=?, completed_at=? WHERE id=?",
                    (str(error)[:500], now, int(job_id)),
                )

    def aggregate_pending(self, *, limit: int = 100) -> int:
        now = time.time()
        with self.db.transaction() as cur:
            cur.execute(
                "SELECT * FROM echo_signals WHERE aggregated_at IS NULL ORDER BY id LIMIT ?",
                (max(1, int(limit)),),
            )
            signals = [dict(row) for row in cur.fetchall()]
            if not signals:
                return 0
            grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
            for signal in signals:
                pair = canonical_pair(signal.get("memory_id"), signal.get("paired_memory_id"))
                key = (
                    signal["agent_id"],
                    signal["viewer_key_hash"],
                    signal["query_archetype"],
                    signal.get("evaluator_version") or self.config.echo_policy_version,
                    pair,
                )
                grouped[key].append(signal)
            for (agent_id, viewer_hash, archetype, policy, pair), rows in grouped.items():
                if pair is not None:
                    self._aggregate_pair(
                        cur, agent_id, viewer_hash, archetype, policy, pair, rows, now
                    )
                else:
                    self._aggregate_memory(cur, agent_id, viewer_hash, archetype, policy, rows, now)
            ids = [int(signal["id"]) for signal in signals]
            placeholders = ",".join("?" for _ in ids)
            cur.execute(
                f"UPDATE echo_signals SET aggregated_at=? WHERE id IN ({placeholders})",
                [now, *ids],
            )
            return len(signals)

    def _aggregate_memory(
        self,
        cur: Any,
        agent_id: str,
        viewer_hash: str,
        archetype: str,
        policy: str,
        rows: list[dict[str, Any]],
        now: float,
    ) -> None:
        memory_id = str(rows[0]["memory_id"])
        cur.execute(
            "SELECT * FROM echo_utility WHERE agent_id=? AND viewer_key_hash=? "
            "AND memory_id=? AND query_archetype=? AND policy_version=?",
            (agent_id, viewer_hash, memory_id, archetype, policy),
        )
        current = dict(cur.fetchone() or {})
        last = float(current.get("last_signal_at") or now)
        age_s = max(0.0, now - last)
        ep = _decay(
            float(current.get("explicit_positive_mass") or 0),
            age_s,
            self.config.echo_explicit_feedback_half_life_days,
        )
        en = _decay(
            float(current.get("explicit_negative_mass") or 0),
            age_s,
            self.config.echo_explicit_feedback_half_life_days,
        )
        ip = _decay(
            float(current.get("inferred_positive_mass") or 0),
            age_s,
            self.config.echo_utility_half_life_days,
        )
        inn = _decay(
            float(current.get("inferred_negative_mass") or 0),
            age_s,
            self.config.echo_utility_half_life_days,
        )
        explicit_positive = int(current.get("explicit_positive") or 0)
        explicit_negative = int(current.get("explicit_negative") or 0)
        evaluator_samples = int(current.get("evaluator_samples") or 0)
        for row in rows:
            mass = float(row["weight"] or 0)
            positive = int(row["direction"]) > 0
            if row.get("source") == "explicit":
                ep = ep + mass if positive else ep
                en = en + mass if not positive else en
                explicit_positive += int(positive)
                explicit_negative += int(not positive)
            else:
                ip = ip + mass if positive else ip
                inn = inn + mass if not positive else inn
                evaluator_samples += 1
        observations = ep + en + ip + inn
        mean = (2.0 + ep + ip) / (4.0 + observations)
        confidence = 1.0 - math.exp(-observations / 10.0)
        harm = (en + inn) / max(1e-9, observations)
        values = (
            agent_id,
            viewer_hash,
            memory_id,
            archetype,
            policy,
            ep,
            en,
            ip,
            inn,
            explicit_positive,
            explicit_negative,
            evaluator_samples,
            observations,
            mean,
            harm,
            confidence,
            now,
            now,
        )
        cur.execute(
            """INSERT OR REPLACE INTO echo_utility(
                agent_id, viewer_key_hash, memory_id, query_archetype, policy_version,
                explicit_positive_mass, explicit_negative_mass, inferred_positive_mass,
                inferred_negative_mass, explicit_positive, explicit_negative,
                evaluator_samples, effective_observations, utility_mean, harm_risk,
                confidence, last_signal_at, updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            values,
        )

    def _aggregate_pair(
        self,
        cur: Any,
        agent_id: str,
        viewer_hash: str,
        archetype: str,
        policy: str,
        pair: tuple[str, str],
        rows: list[dict[str, Any]],
        now: float,
    ) -> None:
        first, second = pair
        cur.execute(
            "SELECT * FROM echo_pair_utility WHERE agent_id=? AND viewer_key_hash=? "
            "AND first_memory_id=? AND second_memory_id=? AND query_archetype=? "
            "AND policy_version=?",
            (agent_id, viewer_hash, first, second, archetype, policy),
        )
        current = dict(cur.fetchone() or {})
        age_s = max(0.0, now - float(current.get("last_signal_at") or now))
        positive = _decay(
            float(current.get("positive_mass") or 0),
            age_s,
            self.config.echo_pair_half_life_days,
        )
        negative = _decay(
            float(current.get("negative_mass") or 0),
            age_s,
            self.config.echo_pair_half_life_days,
        )
        samples = int(current.get("sample_count") or 0)
        for row in rows:
            mass = float(row["weight"] or 0)
            positive = positive + mass if int(row["direction"]) > 0 else positive
            negative = negative + mass if int(row["direction"]) < 0 else negative
            samples += 1
        total = positive + negative
        synergy = (positive - negative) / max(1.0, total)
        confidence = 1.0 - math.exp(-total / 6.0)
        cur.execute(
            """INSERT OR REPLACE INTO echo_pair_utility(
                agent_id, viewer_key_hash, first_memory_id, second_memory_id,
                query_archetype, policy_version, positive_mass, negative_mass,
                sample_count, synergy_score, confidence, last_signal_at, updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                agent_id,
                viewer_hash,
                first,
                second,
                archetype,
                policy,
                positive,
                negative,
                samples,
                synergy,
                confidence,
                now,
                now,
            ),
        )

    def compact(self, *, now: float | None = None) -> dict[str, int]:
        now = time.time() if now is None else float(now)
        receipt_cutoff = now - self.config.echo_receipt_retention_days * 86400
        signal_cutoff = now - self.config.echo_signal_retention_days * 86400
        with self.db.transaction() as cur:
            cur.execute(
                "UPDATE echo_receipts SET status='expired', outcome='unmatched' "
                "WHERE status='open' AND created_at<?",
                (now - 300.0,),
            )
            cur.execute(
                "UPDATE echo_jobs SET status='skipped', completed_at=?, "
                "last_error='receipt expired before turn persisted' "
                "WHERE status='pending' AND receipt_id IN "
                "(SELECT id FROM echo_receipts WHERE status='expired')",
                (now,),
            )
            cur.execute(
                "DELETE FROM echo_signals WHERE aggregated_at IS NOT NULL AND created_at<?",
                (signal_cutoff,),
            )
            signals_deleted = max(0, int(cur.rowcount))
            cur.execute(
                "DELETE FROM echo_receipts WHERE status!='open' AND created_at<?",
                (receipt_cutoff,),
            )
            receipts_deleted = max(0, int(cur.rowcount))
            cur.execute(
                "DELETE FROM echo_jobs WHERE status IN ('done','skipped','failed') "
                "AND COALESCE(completed_at, created_at)<?",
                (now - 7 * 86400,),
            )
            jobs_deleted = max(0, int(cur.rowcount))
            pair_cutoff = now - self.config.echo_pair_half_life_days * 86400
            cur.execute(
                "DELETE FROM echo_pair_utility WHERE updated_at<? "
                "OR (sample_count<2 AND updated_at<?)",
                (pair_cutoff, signal_cutoff),
            )
            pairs_deleted = max(0, int(cur.rowcount))
            cur.execute(
                "DELETE FROM prefetch_stats WHERE id <= COALESCE((SELECT MAX(id) - 10000 "
                "FROM prefetch_stats), 0) OR created_at < ?",
                (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(signal_cutoff)),),
            )
            prefetch_deleted = max(0, int(cur.rowcount))
        return {
            "signals_deleted": signals_deleted,
            "receipts_deleted": receipts_deleted,
            "jobs_deleted": jobs_deleted,
            "pairs_deleted": pairs_deleted,
            "prefetch_deleted": prefetch_deleted,
        }

    def health(self) -> dict[str, Any]:
        with self.db.read() as cur:
            result: dict[str, Any] = {}
            for table in (
                "echo_receipts",
                "echo_receipt_items",
                "echo_signals",
                "echo_utility",
                "echo_pair_utility",
                "echo_jobs",
            ):
                cur.execute(f"SELECT COUNT(*) AS count FROM {table}")
                result[table] = int(cur.fetchone()["count"])
            cur.execute("SELECT status, COUNT(*) AS count FROM echo_jobs GROUP BY status")
            result["job_status"] = {str(row["status"]): int(row["count"]) for row in cur.fetchall()}
            return result


__all__ = ["EchoRepository"]
