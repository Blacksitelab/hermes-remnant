"""Dream loop (Phase 5).

Two callable entry points — ``day_dream()`` and ``night_dream()`` — designed
for cron/systemd timers. They are *not* a daemon.

Token efficiency is the central constraint: the loop NEVER feeds the full
memory DB to the cloud model. It pre-filters candidates locally using cosine
similarity over stored embeddings, then sends only a bounded list of candidate
pairs (max ``DREAM_MAX_CANDIDATE_PAIRS``, default 30) to the cloud model for
judgment.

Pipeline (both modes):
  1. Load last run timestamp + per-day/night impulse counter from
     ``dream_state``. Enforce budget + per-topic cooldown.
  2. Select recent memories (day: last 30 min; night: since the last night
     run). For each, find top-K similar active memories locally using cosine
     over stored embeddings — no LLM, no network.
  3. Also find cross-agent duplicate candidates where cosine > 0.7.
  4. Build a bounded candidate list (max 30 pairs).
  5. Call the cloud model with a JSON prompt asking for a per-pair judgment:
     ``connection`` | ``same_fact`` | ``noise``, plus a reason and an optional
     ``thread_title``. Two-stage: the first call generates observations, the
     second self-evaluates usefulness.
  6. Act on results:
     - ``same_fact`` across agents → merge into a shared memory via
       ``memory_edit(action='merge', actor='system')``; originals superseded.
     - ``connection`` → append a first-person reflection to ``DREAMS.md`` and
       optionally create a thread.
     - ``noise`` → discarded but logged in the diary.
  7. Persist the new run timestamp + counter back to ``dream_state``.

The diary at ``~/.hermes/remnant/DREAMS.md`` is first-person, never indexed by
Remnant, and intended only for human review.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from .config import (
    DREAM_CONNECT_THRESHOLD,
    DREAM_DEDUP_THRESHOLD,
    DREAM_MAX_CANDIDATE_PAIRS,
    DREAM_TOP_K,
    RemnantConfig,
)
from .db import RemnantDB, _unpack_embedding
from .edit import memory_edit
from .embed import Embedder, cosine

log = logging.getLogger("remnant.dream")

# Day window: memories created in the last 30 minutes.
DAY_WINDOW_S = 30 * 60

# dream_state keys.
_K_DAY_RUN_TS = "day_run_ts"
_K_NIGHT_RUN_TS = "night_run_ts"
_K_DAY_COUNTER = "day_counter"
_K_NIGHT_COUNTER = "night_counter"
_K_DAY_COUNTER_DATE = "day_counter_date"
_K_NIGHT_COUNTER_DATE = "night_counter_date"
_K_RECENT_TOPICS = "recent_topics"  # {topic_key: ts} for cooldown

_SYSTEM_PROMPT = (
    "You are a memory consolidation engine. You receive a small list of memory "
    "pairs and judge each pair's relationship. Return STRICT JSON only:\n"
    '{"judgments": [{"pair_ids": ["id_a","id_b"], '
    '"judgment": "connection|same_fact|noise", "reason": "...", '
    '"thread_title": "..."}]}\n'
    "Rules:\n"
    "- same_fact: two memories assert the same underlying fact (possibly from "
    "different agents / wording).\n"
    "- connection: a non-obvious but real thematic link worth recording.\n"
    "- noise: superficial word overlap with no real semantic link.\n"
    "- thread_title only for genuine connections that warrant a topic thread.\n"
)

_EVAL_PROMPT = (
    "You are a strict self-reviewer. Given the observations above and the "
    "candidate pairs, drop any judgment that is noise, redundant, or not "
    "useful. Return the SAME JSON schema with only the surviving judgments. "
    "If none survive, return {\"judgments\": []}."
)


def day_dream(
    db: RemnantDB,
    config: RemnantConfig,
    embedder: Embedder,
    *,
    actor: str = "system",
) -> dict[str, Any]:
    """Run the day dream loop. Callable from a cron/timer.

    Considers memories created in the last 30 minutes.
    """
    return _run_dream(db, config, embedder, mode="day", actor=actor)


def night_dream(
    db: RemnantDB,
    config: RemnantConfig,
    embedder: Embedder,
    *,
    actor: str = "system",
) -> dict[str, Any]:
    """Run the night dream loop. Callable from a cron/timer.

    Considers memories created since the last night run (or the whole active
    set on the first run).
    """
    return _run_dream(db, config, embedder, mode="night", actor=actor)


def _run_dream(
    db: RemnantDB,
    config: RemnantConfig,
    embedder: Embedder,
    *,
    mode: str,
    actor: str,
) -> dict[str, Any]:
    if mode not in ("day", "night"):
        raise ValueError(f"unknown dream mode: {mode}")
    now = time.time()
    budget = config.dream_day_budget if mode == "day" else config.dream_night_budget
    if budget <= 0:
        return {"mode": mode, "skipped": "budget_zero"}

    # 1. Cooldown: per-mode timestamp gate (min interval between runs).
    ts_key = _K_DAY_RUN_TS if mode == "day" else _K_NIGHT_RUN_TS
    last_run = db.get_state(ts_key)
    if last_run is not None:
        cooldown_s = config.dream_cooldown_minutes * 60
        # Cooldown applies per run for the day mode; for night mode the
        # cooldown is also enforced but typically longer in practice.
        if now - float(last_run) < cooldown_s and mode == "day":
            return {"mode": mode, "skipped": "cooldown"}

    # 2. Budget counter (resets per calendar day).
    counter_key = _K_DAY_COUNTER if mode == "day" else _K_NIGHT_COUNTER
    date_key = _K_DAY_COUNTER_DATE if mode == "day" else _K_NIGHT_COUNTER_DATE
    today = time.strftime("%Y-%m-%d", time.gmtime(now))
    counter_date = db.get_state(date_key)
    counter = int(db.get_state(counter_key, 0) or 0)
    if counter_date != today:
        counter = 0
        db.set_state(date_key, today)
        db.set_state(counter_key, 0)
    if counter >= budget:
        return {"mode": mode, "skipped": "budget_exhausted", "counter": counter}

    # 3. Select recent memories.
    since_ts = _window_start(db, mode, now)
    recent = db.get_recent_memories(since_ts=since_ts, limit=200)
    if not recent:
        # Nothing new; still record the run so cooldown resets.
        db.set_state(ts_key, now)
        return {"mode": mode, "candidates": 0, "actions": 0}

    # 4. Local cosine pre-filter. Build a bounded candidate pair list.
    pairs = _select_candidate_pairs(db, recent, mode=mode)
    if not pairs:
        db.set_state(ts_key, now)
        return {"mode": mode, "candidates": 0, "actions": 0}

    # 5. Two-stage cloud judgment.
    judgments = _cloud_judge(config, pairs, mode=mode)
    if not judgments:
        db.set_state(ts_key, now)
        return {"mode": mode, "candidates": len(pairs), "actions": 0}

    # 6. Enforce per-topic cooldown on surviving judgments.
    recent_topics: dict[str, float] = db.get_state(_K_RECENT_TOPICS, {}) or {}
    cooldown_s = config.dream_cooldown_minutes * 60
    kept: list[dict[str, Any]] = []
    for j in judgments:
        topic_key = _topic_key(j)
        last = recent_topics.get(topic_key)
        if last is not None and (now - last) < cooldown_s:
            continue
        kept.append(j)
        recent_topics[topic_key] = now
    db.set_state(_K_RECENT_TOPICS, recent_topics)

    # 7. Act on results (bounded by remaining budget).
    remaining = budget - counter
    actions = 0
    diary_lines: list[str] = []
    for j in kept[:remaining]:
        kind = (j.get("judgment") or "").strip().lower()
        pair_ids = j.get("pair_ids") or []
        reason = (j.get("reason") or "").strip()
        if kind == "same_fact" and len(pair_ids) >= 2:
            n = _merge_same_fact(db, config, embedder, pair_ids, actor=actor)
            actions += n
            if n:
                diary_lines.append(
                    f"I merged {n} cross-agent duplicate(s): {reason}"
                )
        elif kind == "connection":
            _append_diary(config, mode, reason)
            actions += 1
            title = (j.get("thread_title") or "").strip()
            if title:
                try:
                    from .threads import create_thread

                    create_thread(
                        db,
                        title=title,
                        topic=title[:120],
                        importance=0.6,
                        source="dream",
                        added_by=actor,
                    )
                except Exception as e:
                    log.warning("thread creation failed: %s", e)
        elif kind == "noise":
            # Discarded but logged in the diary for human review.
            _append_diary(config, mode, f"(noise, discarded) {reason}")
        else:
            log.debug("ignoring unknown judgment kind: %s", kind)

    # Batch-write any merge-driven diary lines that weren't per-judgment.
    for line in diary_lines:
        _append_diary(config, mode, line)

    # 8. Persist run state.
    db.set_state(ts_key, now)
    db.set_state(counter_key, counter + actions)

    return {
        "mode": mode,
        "candidates": len(pairs),
        "judgments": len(judgments),
        "kept": len(kept),
        "actions": actions,
        "remaining_budget": max(0, budget - (counter + actions)),
    }


# --- candidate selection (local, no LLM) -----------------------------------


def _select_candidate_pairs(
    db: RemnantDB,
    recent: list[dict[str, Any]],
    *,
    mode: str,
) -> list[dict[str, Any]]:
    """Build a bounded candidate pair list using local cosine similarity only.

    Returns a list of ``{"id_a","id_b","content_a","content_b","sim","kind"}``
    dicts. ``kind`` is ``"connection"`` for same-agent top-K neighbors and
    ``"cross_agent"`` for cross-agent duplicate candidates (sim > 0.7).

    Never sends anything to the network. Cap is ``DREAM_MAX_CANDIDATE_PAIRS``.
    """
    if not recent:
        return []

    # Load embeddings for the recent memories.
    recent_ids = [m["id"] for m in recent]
    recent_vecs = _load_vectors(db, recent_ids)
    if not any(recent_vecs.values()):
        # No embeddings available — cannot do cosine. Bail out cheaply.
        return []

    # Active corpus to compare against (bounded by recency). This is the full
    # active set (per the spec: "find top-5 similar active memories"); we
    # dedupe against the seed memory via `seen_pairs` rather than excluding
    # recent memories outright, so two newly-stored facts can still connect.
    # For night mode we include cross-agent shared/fleet memories for dedup;
    # for day mode we look across the shared corpus.
    corpus = db.get_memories_for_agent_scope(
        agent_id=None if mode == "night" else None,
        visibility=None if mode == "night" else "shared",
        limit=500,
    )
    corpus_by_id = {m["id"]: m for m in corpus}
    corpus_ids = list(corpus_by_id.keys())
    if not corpus_ids:
        return []
    corpus_vecs = _load_vectors(db, corpus_ids)

    pairs: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for rm in recent:
        rvec = recent_vecs.get(rm["id"])
        if not rvec:
            continue
        # Score against the whole active corpus, keep top-K by cosine.
        scored: list[tuple[float, dict[str, Any]]] = []
        for cid in corpus_ids:
            if cid == rm["id"]:
                continue  # don't pair a memory with itself
            cvec = corpus_vecs.get(cid)
            if not cvec:
                continue
            sim = cosine(rvec, cvec)
            if sim < DREAM_CONNECT_THRESHOLD:
                continue
            scored.append((sim, corpus_by_id[cid]))
        scored.sort(key=lambda x: x[0], reverse=True)
        for sim, other in scored[:DREAM_TOP_K]:
            key = tuple(sorted((rm["id"], other["id"])))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            cross = (other.get("agent") != rm.get("agent"))
            kind = "cross_agent" if (cross and sim >= DREAM_DEDUP_THRESHOLD) else "connection"
            pairs.append(_pair(rm, other, sim, kind))
            if len(pairs) >= DREAM_MAX_CANDIDATE_PAIRS:
                return pairs

    # Also look for cross-agent duplicates among the recent memories
    # themselves (two agents recently stored the same fact).
    for i in range(len(recent)):
        for j in range(i + 1, len(recent)):
            a, b = recent[i], recent[j]
            if a.get("agent") == b.get("agent"):
                continue
            av, bv = recent_vecs.get(a["id"]), recent_vecs.get(b["id"])
            if not av or not bv:
                continue
            sim = cosine(av, bv)
            if sim >= DREAM_DEDUP_THRESHOLD:
                key = tuple(sorted((a["id"], b["id"])))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                pairs.append(_pair(a, b, sim, "cross_agent"))
                if len(pairs) >= DREAM_MAX_CANDIDATE_PAIRS:
                    return pairs

    return pairs


def _pair(
    a: dict[str, Any], b: dict[str, Any], sim: float, kind: str
) -> dict[str, Any]:
    return {
        "id_a": a["id"],
        "id_b": b["id"],
        "content_a": a.get("content", ""),
        "content_b": b.get("content", ""),
        "agent_a": a.get("agent"),
        "agent_b": b.get("agent"),
        "sim": round(float(sim), 4),
        "kind": kind,
    }


def _load_vectors(db: RemnantDB, memory_ids: list[str]) -> dict[str, list[float]]:
    """Load embeddings for a bounded id list. Returns {id: vector}."""
    if not memory_ids:
        return {}
    out: dict[str, list[float]] = {}
    placeholders = ",".join("?" for _ in memory_ids)
    sql = (
        "SELECT memory_id, embedding FROM embeddings "
        f"WHERE memory_id IN ({placeholders})"
    )
    with db.read() as cur:
        cur.execute(sql, memory_ids)
        for r in cur.fetchall():
            out[r["memory_id"]] = _unpack_embedding(r["embedding"])
    return out


# --- cloud judgment (two-stage) --------------------------------------------


def _cloud_judge(
    config: RemnantConfig,
    pairs: list[dict[str, Any]],
    *,
    mode: str,
) -> list[dict[str, Any]]:
    """Call the cloud model in two stages: generate, then self-evaluate.

    On any network/parse failure returns an empty list (the loop stays safe).
    """
    if not pairs:
        return []
    url = config.dream_day_url if mode == "day" else config.dream_night_url
    model = config.dream_day_model if mode == "day" else config.dream_night_model
    timeout = config.dream_day_timeout if mode == "day" else config.dream_night_timeout
    user_content = _build_prompt(pairs)
    try:
        with httpx.Client(timeout=timeout) as client:
            first = _call(client, url, model, _SYSTEM_PROMPT, user_content)
            judgments = _parse_judgments(first, pairs)
            if not judgments:
                return []
            eval_content = _build_eval_prompt(pairs, judgments)
            second = _call(client, url, model, _EVAL_PROMPT, eval_content)
            refined = _parse_judgments(second, pairs)
            return refined if refined is not None else judgments
    except (httpx.HTTPError, KeyError, ValueError) as e:
        log.warning("dream cloud call failed: %s", e)
        return []


def _call(
    client: httpx.Client,
    url: str,
    model: str,
    system: str,
    user_content: str,
) -> str:
    resp = client.post(
        url,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.1,
            "max_tokens": 768,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    return str(data["choices"][0]["message"]["content"])


def _build_prompt(pairs: list[dict[str, Any]]) -> str:
    lines = ["Candidate memory pairs (ids are short tags):"]
    for i, p in enumerate(pairs):
        lines.append(
            f"[{i}] pair_ids=[{p['id_a'][:8]},{p['id_b'][:8]}] "
            f"sim={p['sim']} kind={p['kind']}"
        )
        lines.append(f"    A: {p['content_a']}")
        lines.append(f"    B: {p['content_b']}")
    lines.append("\nJudge each pair. Return STRICT JSON only.")
    return "\n".join(lines)


def _build_eval_prompt(
    pairs: list[dict[str, Any]], judgments: list[dict[str, Any]]
) -> str:
    lines = ["Original candidate pairs:"]
    for i, p in enumerate(pairs):
        lines.append(f"[{i}] pair_ids=[{p['id_a'][:8]},{p['id_b'][:8]}]")
        lines.append(f"    A: {p['content_a']}")
        lines.append(f"    B: {p['content_b']}")
    lines.append("\nProposed observations:")
    for j in judgments:
        lines.append(
            f"- pair_ids={j.get('pair_ids')} "
            f"judgment={j.get('judgment')} reason={j.get('reason')}"
        )
    lines.append("\nDrop anything that is noise, redundant, or not useful.")
    return "\n".join(lines)


def _parse_judgments(
    text: str, pairs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Parse the cloud JSON response and re-bind short ids to full ids."""
    obj = _extract_json(text)
    if obj is None:
        return []
    raw = obj.get("judgments") or []
    if not isinstance(raw, list):
        return []
    # Map short id prefixes back to full ids.
    id_index = {}
    for p in pairs:
        id_index[p["id_a"][:8]] = p["id_a"]
        id_index[p["id_b"][:8]] = p["id_b"]
    out: list[dict[str, Any]] = []
    for j in raw:
        if not isinstance(j, dict):
            continue
        kind = (j.get("judgment") or "").strip().lower()
        if kind not in ("connection", "same_fact", "noise"):
            continue
        short_ids = j.get("pair_ids") or []
        full_ids: list[str] = []
        for sid in short_ids:
            sid = str(sid)
            full_ids.append(id_index.get(sid[:8], sid))
        out.append(
            {
                "pair_ids": full_ids,
                "judgment": kind,
                "reason": str(j.get("reason", "")).strip(),
                "thread_title": str(j.get("thread_title", "")).strip(),
            }
        )
    return out


def _extract_json(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


# --- actions ---------------------------------------------------------------


def _merge_same_fact(
    db: RemnantDB,
    config: RemnantConfig,
    embedder: Embedder,
    pair_ids: list[str],
    *,
    actor: str,
) -> int:
    """Merge cross-agent same_fact memories into one shared memory.

    Uses ``memory_edit(action='merge', actor='system')`` with combined content.
    Returns 1 on success, 0 on failure (e.g. fewer than two valid memories).
    """
    ids = [mid for mid in pair_ids if db.get_memory(mid) is not None]
    if len(ids) < 2:
        return 0
    parts: list[str] = []
    for mid in ids:
        m = db.get_memory(mid)
        if m and m.get("content"):
            parts.append(m["content"].strip())
    if not parts:
        return 0
    combined = " | ".join(dict.fromkeys(parts))
    try:
        res = memory_edit(
            db,
            config,
            embedder,
            action="merge",
            actor=actor,
            memory_ids=ids,
            content=combined,
            visibility="shared",
        )
        return 1 if res.get("memory_id") else 0
    except Exception as e:
        log.warning("merge_same_fact failed: %s", e)
        return 0


def _append_diary(config: RemnantConfig, mode: str, line: str) -> None:
    """Append a first-person reflection to DREAMS.md (never indexed)."""
    path = Path(_expand_diary_path(config.diary_path))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        header = time.strftime(f"## %Y-%m-%d %H:%M ({mode})\n", time.gmtime())
        block = f"{header}\nI noticed that {line}\n\n---\n"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(block)
    except OSError as e:
        log.warning("failed to append diary: %s", e)


def _expand_diary_path(p: str) -> str:
    from os.path import expanduser

    return expanduser(p)


def _window_start(db: RemnantDB, mode: str, now: float) -> float:
    if mode == "day":
        return now - DAY_WINDOW_S
    last_night = db.get_state(_K_NIGHT_RUN_TS)
    if last_night is not None:
        return float(last_night)
    # First night run: look back 24h to keep the candidate set bounded.
    return now - 86400.0


def _topic_key(judgment: dict[str, Any]) -> str:
    """Stable key for cooldown. Uses the judgment kind + the first 8 chars of
    each pair id so repeated suggestions on the same pair are suppressed."""
    ids = judgment.get("pair_ids") or []
    return (judgment.get("judgment") or "x") + ":" + ",".join(
        str(i)[:8] for i in ids
    )


__all__ = ["day_dream", "night_dream"]
