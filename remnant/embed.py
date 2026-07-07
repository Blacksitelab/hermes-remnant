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
        self._client = httpx.Client(timeout=self._timeout)

    def embed(self, text: str) -> list[float] | None:
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
            return cached
        vec = self._embed_remote(text)
        if vec is None:
            return None
        self._db.put_cached_embedding(self._model, text_hash, vec)
        return vec

    def _embed_remote(self, text: str) -> list[float] | None:
        try:
            resp = self._client.post(
                self._url,
                json={"model": self._model, "prompt": text, "keep_alive": -1},
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
