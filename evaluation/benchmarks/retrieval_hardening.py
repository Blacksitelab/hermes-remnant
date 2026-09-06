import gc
import json
import statistics
import tempfile
import time
from pathlib import Path

from remnant import RemnantMemoryProvider
from remnant.config import RemnantConfig
from remnant.db import _pack_embedding, open_db
from remnant.search import _semantic_rank


class Embed:
    _model = "audit-model"

    def embed(self, *a, **kw):
        return [((j * 73) % 199 + 1) / 199 for j in range(768)]


with tempfile.TemporaryDirectory() as t:
    db = open_db(Path(t) / "benchmark.db")
    with db.transaction() as cur:
        cur.executemany(
            "INSERT INTO memories(id,type,content,source,agent,timestamp,created_at,updated_at) "
            "VALUES(?,'fact',?,'manual','audit','2026-09-01','2026-09-01','2026-09-01')",
            ((f"m-{i}", f"Audit configuration entry {i}") for i in range(5000)),
        )
        cur.executemany(
            "INSERT INTO embeddings(memory_id,model,embedding,dimensions,created_at) "
            "VALUES(?,'audit-model',?,768,'2026-09-01')",
            (
                (
                    f"m-{i}",
                    _pack_embedding([(((j * 73 + i * 37) % 997) + 1) / 997 for j in range(768)]),
                )
                for i in range(5000)
            ),
        )
    cfg = RemnantConfig(agent_id="audit", embed_model="audit-model", echo_enabled=False)
    emb = Embed()
    p = RemnantMemoryProvider()
    p._db = db
    p._config = cfg
    p._embedder = emb

    def semantic():
        return _semantic_rank(db, cfg, "configuration", agent_id="audit", embedder=emb)[:100]

    result = semantic()
    times = []
    full = []
    delivered = []
    for i in range(7):
        start = time.perf_counter()
        semantic()
        times.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        ctx = p.prefetch("configuration", session_id=f"s-{i}")
        full.append((time.perf_counter() - start) * 1000)
        delivered.append(bool(ctx))
    gc.collect()
    import tracemalloc

    tracemalloc.start()
    semantic()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    out = {
        "rows": 5000,
        "dimensions": 768,
        "embedding": "instant deterministic stub",
        "semantic_ms": round(statistics.median(times), 3),
        "provider_prefetch_ms": round(statistics.median(full), 3),
        "provider_delivered": sum(delivered),
        "probes": 7,
        "python_scoring_peak_mib": round(peak / 1024**2, 3),
        "top100": [r["id"] for r in result],
        "scores": [r["score"] for r in result],
    }
    db.close()
    print(json.dumps(out, indent=2))
