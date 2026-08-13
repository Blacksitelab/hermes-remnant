"""Memory reflection: synthesize an answer across the top-N memories via a
local LLM (gemma4:12b on BSL1).

Input is bounded (top 20 memories) and output capped at max_tokens 512 to keep
the call cheap and predictable. Source memory ids are returned for attribution.
"""

from __future__ import annotations

import logging
from typing import Any

from .config import REFLECT_MAX_TOKENS, REFLECT_TOP_N, RemnantConfig
from .db import RemnantDB
from .embed import Embedder
from .llm import chat
from .recall import RecallRequest, RecallService

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
    session_id: str = "default",
) -> dict[str, Any]:
    """Retrieve top-N memories for `question` and synthesize via local LLM."""
    question = (question or "").strip()
    if not question:
        return {"error": "question is required"}

    response = RecallService(db, config).recall(
        RecallRequest(
            query=question,
            agent_id=agent_id,
            session_id=session_id or "default",
            strategy="auto",
            limit=REFLECT_TOP_N,
            include_pending=bool(getattr(config, "recent_turn_overlay_enabled", False)),
        ),
        embedder=embedder,
    )
    results = response.results
    if not results:
        return {"synthesis": "", "source_ids": [], "count": 0}

    top = results[:REFLECT_TOP_N]
    blocks: list[str] = []
    for memory in top:
        line = f"[mem:{memory['id'][:8]}] {memory.get('content', '')}"
        if memory.get("claim_status") in {"unresolved", "contradicted"}:
            line += f" [status={memory['claim_status']}]"
        group = memory.get("claim_group") or []
        alternatives = [
            str(row.get("content") or "")
            for row in group[1:]
            if row.get("content")
        ]
        if alternatives:
            line += " [competing evidence: " + " | ".join(alternatives[:2]) + "]"
        blocks.append(line)
    memory_block = "\n".join(blocks)
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
        return chat(
            url=config.reflect_url,
            model=config.reflect_model,
            system=_REFLECT_PROMPT,
            user=user_content,
            timeout=config.reflect_timeout,
            protocol=getattr(config, "llm_protocol", None),
            temperature=0.2,
            max_tokens=REFLECT_MAX_TOKENS,
        )
    except Exception as e:
        log.warning("reflect LLM call failed: %s", e)
        return ""


__all__ = ["memory_reflect"]
