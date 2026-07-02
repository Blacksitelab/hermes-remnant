"""Memory reflection: synthesize an answer across the top-N memories via a
local LLM (gemma4:12b on BSL1).

Input is bounded (top 20 memories) and output capped at max_tokens 512 to keep
the call cheap and predictable. Source memory ids are returned for attribution.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .config import REFLECT_MAX_TOKENS, REFLECT_TOP_N, RemnantConfig
from .db import RemnantDB
from .embed import Embedder
from .search import search as hybrid_search

log = logging.getLogger("remnant.reflect")

_REFLECT_PROMPT = (
    "You are a memory reflection engine. Given a question and a set of stored "
    "memories, synthesize a concise answer that is grounded ONLY in the provided "
    "memories. Cite memory ids like [mem:<id-short>] when using a fact.\n"
    "If the memories do not answer the question, say so explicitly.\n"
    "Keep the answer under 150 words."
)


def memory_reflect(
    question: str,
    db: RemnantDB,
    config: RemnantConfig,
    embedder: Embedder,
    agent_id: str,
) -> dict[str, Any]:
    """Retrieve top-N memories for `question` and synthesize via local LLM."""
    question = (question or "").strip()
    if not question:
        return {"error": "question is required"}

    results = hybrid_search(
        db, config, question,
        agent_id=agent_id, limit=REFLECT_TOP_N, strategy="auto",
        embedder=embedder,
    )
    if not results:
        return {"synthesis": "", "source_ids": [], "count": 0}

    top = results[:REFLECT_TOP_N]
    memory_block = "\n".join(
        f"[mem:{m['id'][:8]}] {m.get('content', '')}" for m in top
    )
    user_content = f"Question: {question}\n\nMemories:\n{memory_block}"

    synthesis = _call_llm(config, user_content)
    return {
        "synthesis": synthesis,
        "source_ids": [m["id"] for m in top],
        "count": len(top),
    }


def _call_llm(config: RemnantConfig, user_content: str) -> str:
    """Call the reflect endpoint. Returns empty string on any failure."""
    try:
        with httpx.Client(timeout=config.reflect_timeout) as client:
            resp = client.post(
                config.reflect_url,
                json={
                    "model": config.reflect_model,
                    "messages": [
                        {"role": "system", "content": _REFLECT_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0.2,
                    "max_tokens": REFLECT_MAX_TOKENS,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return str(data["choices"][0]["message"]["content"]).strip()
    except (httpx.HTTPError, KeyError, ValueError) as e:
        log.warning("reflect LLM call failed: %s", e)
        return ""


__all__ = ["memory_reflect"]
