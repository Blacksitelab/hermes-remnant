"""Reproducible scale-envelope measurements for exact retrieval.

The benchmark is deliberately separate from ordinary evaluation and CI.  It
creates synthetic, disposable SQLite stores, measures the exact-vector oracle
and the full claim-aware recall path, and records enough environment metadata
to compare runs before deciding whether an ANN index is justified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sqlite3
import tempfile
import time
import tracemalloc
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import RemnantConfig
from ..db import _pack_embedding, open_db
from ..recall import RecallRequest, RecallService
from ..search import search


@dataclass(frozen=True)
class ScaleBenchmarkConfig:
    """Inputs that make a scale run reproducible."""

    sizes: tuple[int, ...] = (5_000, 25_000, 100_000, 1_000_000)
    dimensions: int = 64
    probes: int = 5
    seed: int = 0
    embedding_gap_every: int = 10


class SyntheticEmbedder:
    """Hash embedder shared by generated rows and deterministic probe queries."""

    _model = "scale-hash-v1"

    def __init__(self, dimensions: int):
        self.dimensions = max(2, int(dimensions))

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in str(text or "").casefold().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:2], "big") % self.dimensions] += 1.0
        norm = sum(value * value for value in vector) ** 0.5
        return [value / norm for value in vector] if norm else vector


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction) - 1))
    return round(ordered[index], 3)


def _path_size(path: Path) -> int:
    return sum(
        candidate.stat().st_size
        for candidate in path.parent.glob(f"{path.name}*")
        if candidate.is_file()
    )


def _seed_store(
    db: Any, size: int, *, dimensions: int, seed: int, gap_every: int
) -> dict[str, Any]:
    """Populate a store in one transaction, including FTS and claims."""
    started = time.perf_counter()
    namespace = uuid.uuid5(uuid.NAMESPACE_URL, f"https://remnant-scale/{seed}/{size}")
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    embedder = SyntheticEmbedder(dimensions)
    memories: list[tuple[Any, ...]] = []
    embeddings: list[tuple[Any, ...]] = []
    claims: list[tuple[Any, ...]] = []
    embedded = 0
    claim_count = 0
    for index in range(size):
        memory_id = str(uuid.uuid5(namespace, f"memory-{index}"))
        agent = f"agent-{index % 8}"
        visibility = ("private", "shared", "fleet")[index % 3]
        source = "vault" if index % 11 == 0 else "conversation"
        content = (
            f"target-{index:07d} topic-{index % 257} {agent} "
            f"project-{index % 41} preference evidence revision-{index % 19}"
        )
        memories.append(
            (
                memory_id, "document" if source == "vault" else "fact", content, source,
                f"scale/{index % 97}/note-{index}.md" if source == "vault" else str(index),
                agent, visibility, now, 0.8, 0.7, 0, None, "active", '["scale"]',
                json.dumps({"synthetic": True, "seed": seed}),
                hashlib.sha256(content.encode("utf-8")).hexdigest(), 1, now, now,
            )
        )
        if gap_every <= 0 or index % gap_every:
            embeddings.append((memory_id, embedder._model, _pack_embedding(embedder.embed(content)),
                               dimensions, now))
            embedded += 1
        if index % 5 == 0:
            claim_id = str(uuid.uuid5(namespace, f"claim-{index}"))
            status = "superseded" if index % 97 == 0 and index else "active"
            resolution = "superseded" if status == "superseded" else "active"
            claims.append(
                (
                    claim_id, memory_id, agent, "prefers", f"project-{index % 41}", None,
                    0.8, status, None, None, now, now, "scope", agent, "asserted", None,
                    resolution, "scale-v1", None, now, now,
                )
            )
            claim_count += 1
    with db.transaction() as cur:
        cur.executemany(
            "INSERT INTO memories(id,type,content,source,source_id,agent,visibility,"
            "timestamp,confidence,trust_score,verified,superseded_by,status,tags,metadata,"
            "content_hash,seen_count,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            memories,
        )
        if embeddings:
            cur.executemany(
                "INSERT INTO embeddings(memory_id,model,embedding,dimensions,created_at) "
                "VALUES(?,?,?,?,?)",
                embeddings,
            )
        if claims:
            cur.executemany(
                "INSERT INTO claims(id,memory_id,subject,predicate,object,qualifiers,"
                "confidence,status,valid_from,valid_to,observed_at,event_at,scope_type,"
                "scope_value,modality,conflict_type,resolution_status,extractor_version,"
                "source_turn_id,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                claims,
            )
    elapsed = (time.perf_counter() - started) * 1000.0
    return {
        "memories": size,
        "embedding_rows": embedded,
        "embedding_gap_ratio": round(1.0 - (embedded / size if size else 0.0), 4),
        "claim_rows": claim_count,
        "ingest_ms": round(elapsed, 3),
        "ingest_per_second": round(size / (elapsed / 1000.0), 2) if elapsed else 0.0,
    }


def _probe_indices(size: int, probes: int, gap_every: int) -> list[int]:
    values: list[int] = []
    for index in range(max(0, size)):
        if index % 8 == 0 and (gap_every <= 0 or index % gap_every):
            values.append(index)
        if len(values) >= max(1, probes):
            break
    return values


def _measure_store(path: Path, size: int, config: ScaleBenchmarkConfig) -> dict[str, Any]:
    db = open_db(path)
    try:
        seed_report = _seed_store(
            db,
            size,
            dimensions=config.dimensions,
            seed=config.seed,
            gap_every=config.embedding_gap_every,
        )
        embedder = SyntheticEmbedder(config.dimensions)
        remnant_config = RemnantConfig(
            agent_id="agent-0",
            default_search_strategy="semantic",
            search_limit=5,
            semantic_scan_limit=max(1, size),
            min_semantic_score=0.0,
            injection_token_budget=2_000,
        )
        indices = _probe_indices(size, config.probes, config.embedding_gap_every)
        exact_times: list[float] = []
        prefetch_times: list[float] = []
        relevant = 0
        ranks: list[int] = []
        tracemalloc.start()
        for index in indices:
            query = f"target-{index:07d}"
            started = time.perf_counter()
            rows = search(
                db, remnant_config, query, agent_id="agent-0", limit=5,
                strategy="semantic", embedder=embedder,
            )
            exact_times.append((time.perf_counter() - started) * 1000.0)
            returned = [str(row.get("id")) for row in rows]
            target = str(uuid.uuid5(uuid.uuid5(
                uuid.NAMESPACE_URL, f"https://remnant-scale/{config.seed}/{size}"
            ), f"memory-{index}"))
            if target in returned:
                relevant += 1
                ranks.append(returned.index(target) + 1)
            started = time.perf_counter()
            RecallService(db, remnant_config).recall(
                RecallRequest(
                    query=query, agent_id="agent-0", strategy="auto", limit=5,
                    token_budget=2_000, output_mode="context",
                ),
                embedder=embedder,
            )
            prefetch_times.append((time.perf_counter() - started) * 1000.0)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # Reopen once to expose cold-cache behavior without mixing it into the
        # warm p50/p95 measurements.
        db.close()
        cold_db = open_db(path)
        try:
            cold_index = indices[0] if indices else 0
            started = time.perf_counter()
            search(
                cold_db, remnant_config, f"target-{cold_index:07d}",
                agent_id="agent-0", limit=5, strategy="semantic", embedder=embedder,
            )
            cold_ms = (time.perf_counter() - started) * 1000.0
        finally:
            cold_db.close()
        return {
            "size": size,
            **seed_report,
            "database_bytes": _path_size(path),
            "exact_vector_ms": {
                "p50": _percentile(exact_times, 0.50),
                "p95": _percentile(exact_times, 0.95),
                "cache_state": "warm",
            },
            "full_prefetch_ms": {
                "p50": _percentile(prefetch_times, 0.50),
                "p95": _percentile(prefetch_times, 0.95),
                "cache_state": "warm",
            },
            "cold_exact_vector_ms": round(cold_ms, 3),
            "peak_memory_mb": round(peak / (1024 * 1024), 3),
            "recall_at_5": round(relevant / len(indices), 4) if indices else 0.0,
            "relevant_rank_p50": _percentile([float(rank) for rank in ranks], 0.50),
            "probes": len(indices),
        }
    finally:
        # The cold probe closes its own handle; this is idempotent for the warm
        # handle after the explicit close above.
        try:
            db.close()
        except Exception:
            pass


def benchmark_scale(
    *,
    sizes: Sequence[int] | None = None,
    work_dir: str | Path | None = None,
    dimensions: int = 64,
    probes: int = 5,
    seed: int = 0,
    embedding_gap_every: int = 10,
) -> dict[str, Any]:
    """Run the scale envelope; never touches the configured production DB."""
    config = ScaleBenchmarkConfig(
        sizes=tuple(int(size) for size in (sizes or ScaleBenchmarkConfig.sizes)),
        dimensions=dimensions,
        probes=probes,
        seed=seed,
        embedding_gap_every=embedding_gap_every,
    )
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if work_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="remnant-scale-")
        root = Path(temporary.name)
    else:
        root = Path(work_dir)
        root.mkdir(parents=True, exist_ok=True)
    try:
        stores = []
        for size in config.sizes:
            if size <= 0:
                raise ValueError("scale sizes must be positive")
            path = root / f"scale-{size}.db"
            if path.exists():
                raise FileExistsError(f"refusing to overwrite benchmark store: {path}")
            stores.append(_measure_store(path, size, config))
        return {
            "schema_version": 1,
            "benchmark": "remnant-scale-envelope-v1",
            "configuration": {
                "sizes": list(config.sizes),
                "embedding_dimensions": config.dimensions,
                "embedding_model": SyntheticEmbedder._model,
                "embedding_gap_every": config.embedding_gap_every,
                "probes": config.probes,
                "seed": config.seed,
                "cache_states": ["warm", "cold_probe"],
                "python": platform.python_version(),
                "platform": platform.platform(),
                "sqlite": sqlite3.sqlite_version,
                "commit_sha": os.environ.get("GITHUB_SHA", "local"),
            },
            "stores": stores,
        }
    finally:
        if temporary is not None:
            temporary.cleanup()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure Remnant's exact retrieval scale envelope."
    )
    parser.add_argument("--sizes", default="5000,25000,100000,1000000")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--dimensions", type=int, default=64)
    parser.add_argument("--probes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embedding-gap-every", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = benchmark_scale(
        sizes=tuple(int(value) for value in args.sizes.split(",") if value.strip()),
        work_dir=args.work_dir,
        dimensions=args.dimensions,
        probes=args.probes,
        seed=args.seed,
        embedding_gap_every=args.embedding_gap_every,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["ScaleBenchmarkConfig", "SyntheticEmbedder", "benchmark_scale"]
