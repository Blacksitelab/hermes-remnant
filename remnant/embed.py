"""Ollama embedding client + cosine helper.

- nomic-embed-text (768-dim) via the BSL1 Ollama `/api/embeddings` endpoint.
- SQLite-backed cache keyed on (model, sha256(text)) so repeated facts never
  re-hit the network.
- `embed()` returns ``None`` on failure (never an empty list): callers must
  treat ``None`` as "no embedding" and skip semantic comparison / store no row.
- `cosine()` for dedup comparison.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time

import httpx

from .config import RemnantConfig
from .db import RemnantDB

log = logging.getLogger("remnant.embed")


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


class Embedder:
    """Embedding client with SQLite-backed cache."""

    def __init__(self, db: RemnantDB, config: RemnantConfig):
        self._db = db
        self._model = config.embed_model
        self._url = config.embed_url
        self._timeout = config.embed_timeout
        self._keep_alive = getattr(config, "embed_keep_alive", "10m")
        self._client = httpx.Client(timeout=self._timeout)

    def embed(self, text: str, *, timeout: float | None = None) -> list[float] | None:
        """Return the embedding for `text`, hitting the cache when possible.

        Returns ``None`` when the remote embedding call fails (and nothing is
        cached). Callers must treat ``None`` as "no embedding available": skip
        semantic comparison and store no embedding row, rather than treating an
        empty vector as a usable zero vector.
        """
        # Truncate to stay within the embed model's context window.
        # nomic-embed-text on BSL1 has ~3000 char context limit (empirically tested).
        # Use 2500 as a safe ceiling.
        if len(text) > 2500:
            text = text[:2500]
        text_hash = _hash(text)
        cached = self._db.get_cached_embedding(self._model, text_hash)
        if cached is not None:
            self._db.record_operation(
                "embedding", "cache_hit", input_units=len(text), output_units=len(cached)
            )
            return cached
        started = time.perf_counter()
        vec = self._embed_remote(text, timeout=timeout)
        if vec is None:
            self._db.record_operation(
                "embedding",
                "failure",
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                input_units=len(text),
            )
            return None
        self._db.record_operation(
            "embedding",
            "remote_success",
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            input_units=len(text),
            output_units=len(vec),
        )
        self._db.put_cached_embedding(self._model, text_hash, vec)
        return vec

    def _embed_remote(self, text: str, *, timeout: float | None = None) -> list[float] | None:
        try:
            # A prefetch call supplies a much smaller per-request timeout than
            # the general client timeout.  Passing it to the request is
            # essential: a client-level 30s timeout defeats prefetch's 500ms
            # deadline when Ollama accepts a connection but queues the work.
            request_timeout = self._timeout if timeout is None else max(0.001, float(timeout))
            resp = self._client.post(
                self._url,
                timeout=request_timeout,
                json={
                    "model": self._model,
                    "prompt": text,
                    "keep_alive": getattr(self, "_keep_alive", "10m"),
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return [float(x) for x in data["embedding"]]
        except (httpx.HTTPError, KeyError, ValueError) as e:
            log.warning("embedding request failed: %s", e)
            return None

    def close(self) -> None:
        self._client.close()


__all__ = ["Embedder", "cosine"]
