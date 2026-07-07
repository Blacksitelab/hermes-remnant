"""Trust calibration: set meaningful trust scores based on memory quality signals.

Decay alone produces a flat distribution (everything converges to the floor).
This script recalibrates trust scores using multiple signals:

1. **Source quality**: vault (1.0 confidence) > import (0.9) > manual (0.5) 
   > conversation (0.5) > hindsight (0.5)
2. **Confidence**: higher confidence → higher baseline trust
3. **Verified status**: verified memories get +0.1
4. **Engagement**: seen_count > 1 → boost (retrieved and not forgotten = useful)
5. **Recency**: recent memories get a small boost over old ones

The formula:
  base = confidence  (already set per-source on ingestion)
  if verified: base += 0.1
  if seen_count > 1: base += min(0.05 * (seen_count - 1), 0.15)
  recency_boost = min(age_factor * 0.05, 0.05)  # up to +0.05 for very recent
  trust = min(base + recency_boost, 0.95)

Then apply decay on top so old memories still trend down.

Usage:
    python -m remnant.calibrate_trust --dry-run   # preview
    python -m remnant.calibrate_trust --yes        # apply
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from collections import Counter
from datetime import datetime

log = logging.getLogger("remnant.calibrate_trust")


def _parse_iso(ts: str | None) -> float:
    if not ts:
        return 0.0
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return 0.0


def calibrate_trust(db_path: str, dry_run: bool = True) -> dict:
    """Recalibrate trust scores for all active memories."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("""
        SELECT id, source, confidence, trust_score, verified, seen_count, 
               created_at, updated_at
        FROM memories 
        WHERE status = 'active'
    """)
    rows = c.fetchall()

    log.info("Calibrating %d active memories...", len(rows))

    updates: list[tuple[float, str]] = []
    before_buckets: Counter = Counter()
    after_buckets: Counter = Counter()

    import time
    now = time.time()

    for row in rows:
        mid, source, confidence, current_trust, verified, seen_count, created_at, updated_at = row
        confidence = confidence or 0.5
        current_trust = current_trust or 0.5
        verified = verified or 0
        seen_count = seen_count or 1

        # Base trust from confidence
        base = confidence

        # Source quality adjustment — some sources are inherently more trustworthy
        # vault: curated, human-written notes → boost
        # import: explicitly stored by user/agent → boost
        # manual: user explicitly created → boost
        # conversation: extracted from chat → neutral
        # hindsight: bulk imported, lower quality → penalty
        source_boost = {
            "vault": 0.15,
            "import": 0.05,
            "manual": 0.10,
            "conversation": 0.0,
            "hindsight": -0.05,
        }
        base += source_boost.get(source, 0.0)

        # Verified boost
        if verified:
            base += 0.1

        # Engagement boost — retrieved multiple times means useful
        if seen_count > 1:
            base += min(0.05 * (seen_count - 1), 0.15)

        # Recency boost — memories updated in the last 7 days get a small bump
        updated_ts = _parse_iso(updated_at or created_at)
        if updated_ts > 0:
            age_days = (now - updated_ts) / 86400.0
            if age_days < 1:
                base += 0.05
            elif age_days < 7:
                base += 0.03
            elif age_days < 30:
                base += 0.01

        # Source quality adjustment
        # Vault notes have confidence=1.0 already, so they get base=1.0 → capped at 0.95
        # Conversation memories have confidence=0.5 → base=0.5
        # Import (memory_store) has confidence=0.9 → base=0.9
        # Hindsight has confidence=0.5 → base=0.5

        # Cap at 0.95
        new_trust = min(base, 0.95)

        # Round to 2 decimal places for clean distribution
        new_trust = round(new_trust, 2)

        before_buckets[round(current_trust, 1)] += 1
        after_buckets[round(new_trust, 1)] += 1

        if abs(new_trust - current_trust) > 0.001:
            updates.append((new_trust, mid))

    log.info("Before (bucketed by 0.1):")
    for bucket in sorted(before_buckets):
        log.info("  %.1f: %d", bucket, before_buckets[bucket])

    log.info("After (bucketed by 0.1):")
    for bucket in sorted(after_buckets):
        log.info("  %.1f: %d", bucket, after_buckets[bucket])

    log.info("Would update %d memories", len(updates))

    if not dry_run:
        # Apply in batches
        for new_trust, mid in updates:
            c.execute(
                "UPDATE memories SET trust_score = ?, updated_at = updated_at WHERE id = ?",
                (new_trust, mid),
            )
        conn.commit()
        log.info("Applied %d trust score updates", len(updates))
    else:
        log.info("DRY RUN — no changes applied")

    conn.close()
    return {
        "total": len(rows),
        "updated": len(updates),
        "before": dict(before_buckets),
        "after": dict(after_buckets),
        "dry_run": dry_run,
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate trust scores based on quality signals")
    parser.add_argument("--dry-run", action="store_true", help="Preview without applying")
    parser.add_argument("--yes", action="store_true", help="Apply changes")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    db_path = "/home/jd/.hermes/remnant/remnant.db"
    result = calibrate_trust(db_path, dry_run=not args.yes)
    print(f"\n{'DRY RUN' if result['dry_run'] else 'APPLIED'}: {result['updated']}/{result['total']} memories updated")


if __name__ == "__main__":
    main()