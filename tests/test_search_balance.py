"""Regression tests for search balance issues #30, #31, #32.

These run without a live Ollama: the Embedder uses the same deterministic
word-bag vectors as ``test_phase2.py``.
"""

from __future__ import annotations

import hashlib

import pytest

from remnant.config import RemnantConfig
from remnant.db import default_db_path, open_db
from remnant.embed import Embedder
from remnant.ingest import store_memory
from remnant.search import search


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fake_embed(db, config, dim=8):
    emb = Embedder.__new__(Embedder)
    emb._db = db
    emb._model = config.embed_model
    emb._url = config.embed_url
    emb._timeout = config.embed_timeout
    emb._client = None

    def embed(text: str) -> list[float]:
        cached = db.get_cached_embedding(emb._model, _hash(text))
        if cached is not None:
            return cached
        vec = [float((ord(c) * (i + 1)) % 97 / 97.0) for i, c in enumerate(text[:dim])]
        while len(vec) < dim:
            vec.append(0.0)
        db.put_cached_embedding(emb._model, _hash(text), vec)
        return vec

    emb.embed = embed
    emb.close = lambda: None
    return emb


@pytest.fixture(autouse=True)
def patch_extract(monkeypatch):
    from remnant import extract as extract_mod

    monkeypatch.setattr(extract_mod.ExtractionWorker, "_extract", lambda self, u, a: [])
    yield


def _store(db, emb, cfg, fact, *, agent="default", source=None, visibility="private"):
    return store_memory(
        db, emb, cfg,
        fact=fact,
        entity="general",
        session_id="s",
        agent_id=agent,
        visibility=visibility,
        source=source,
    )


def test_issue_30_default_min_semantic_score_is_03():
    """Issue #30: default min_semantic_score lowered from 0.5 to 0.3."""
    cfg = RemnantConfig()
    assert cfg.min_semantic_score == 0.3


def test_issue_31_auto_falls_back_to_bm25_when_semantic_weak():
    """Issue #31: auto strategy must return BM25 results when semantic signal
    is below threshold, not []."""
    db = open_db(default_db_path())
    cfg = RemnantConfig(default_search_strategy="auto")
    emb = _fake_embed(db, cfg)
    try:
        # Seed a memory with keyword overlap so BM25 finds it, but the fake
        # embedder gives it a weak cosine score (partial token overlap).
        mid = _store(
            db, emb, cfg,
            "Kris Hastings Hawke's Bay job at Pak-Line",
            source="conversation",
        )
        assert mid is not None

        # Set an aggressive threshold so the weak semantic score is rejected.
        cfg.min_semantic_score = 0.95
        res = search(db, cfg, "Kris Hastings Hawke's Bay", agent_id="default", embedder=emb)
        # Before the fix this returned [].
        assert res, "auto should fall back to BM25 when semantic is weak"
        assert any(r["id"] == mid for r in res)
    finally:
        db.close()


def test_issue_31_semantic_still_returns_empty_below_threshold():
    """Issue #31: pure semantic strategy correctly returns [] below threshold."""
    db = open_db(default_db_path())
    cfg = RemnantConfig(default_search_strategy="semantic")
    emb = _fake_embed(db, cfg)
    try:
        # Use a fact and query that share no tokens so BM25 does not pre-filter
        # it into the candidate list.
        _store(db, emb, cfg, "Sven prefers dark mode for all editors", source="conversation")
        cfg.min_semantic_score = 0.95
        res = search(
            db, cfg, "completely unrelated query",
            agent_id="default", strategy="semantic", embedder=emb,
        )
        assert res == []
    finally:
        db.close()


def test_issue_32_conversation_fact_outranks_vault_document():
    """Issue #32: small conversation facts should not be buried by large vault
    documents in auto RRF results."""
    db = open_db(default_db_path())
    cfg = RemnantConfig(default_search_strategy="auto")
    emb = _fake_embed(db, cfg)
    try:
        # Conversation fact that shares query tokens.
        conv_mid = _store(
            db, emb, cfg,
            "Kris Hastings Hawke's Bay job at Pak-Line",
            source="conversation",
        )
        # Vault document that also shares the same tokens, but is much longer.
        vault_body = "\n".join([
            "# Operation Find A New Job",
            "This is a log of job searching.",
        ] + ["Kris Hastings Hawke's Bay job"] * 50)
        vault_mid = _store(
            db, emb, cfg, vault_body, source="vault", visibility="shared",
        )
        assert conv_mid is not None
        assert vault_mid is not None

        res = search(
            db, cfg, "Kris Hastings Hawke's Bay",
            agent_id="default", strategy="auto", embedder=emb,
        )
        ids = [r["id"] for r in res]
        assert conv_mid in ids
        assert vault_mid in ids
        # The conversation fact should rank at or above the vault document.
        conv_idx = ids.index(conv_mid)
        vault_idx = ids.index(vault_mid)
        assert conv_idx <= vault_idx, "conversation fact should not be buried by vault"
    finally:
        db.close()


def test_issue_32_vault_still_ranks_first_when_genuinely_better():
    """Issue #32: source weighting should be mild — a precise vault match can
    still win when it is the only relevant result."""
    db = open_db(default_db_path())
    cfg = RemnantConfig(default_search_strategy="auto")
    emb = _fake_embed(db, cfg)
    try:
        vault_mid = _store(
            db, emb, cfg,
            "The BlacksiteLab homelab architecture overview",
            source="vault",
        )
        assert vault_mid is not None
        res = search(
            db, cfg, "BlacksiteLab homelab architecture",
            agent_id="default", strategy="auto", embedder=emb,
        )
        assert any(r["id"] == vault_mid for r in res)
    finally:
        db.close()
