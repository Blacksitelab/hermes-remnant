"""Bounded local evaluator for Echo counterfactual jobs."""

from __future__ import annotations

import ipaddress
import json
from typing import Any
from urllib.parse import urlparse

from .config import RemnantConfig
from .echo_policy import signal_spec
from .llm import chat


def is_local_endpoint(url: str) -> bool:
    """Allow the default evaluator only on loopback/private LAN endpoints."""
    host = (urlparse(str(url or "")).hostname or "").casefold().rstrip(".")
    if host.startswith("your-"):
        return False
    if host in {"localhost", "localhost.localdomain", "127.0.0.1", "::1"}:
        return True
    if host.endswith(".local") or host.endswith(".lan"):
        return True
    try:
        return ipaddress.ip_address(host).is_private
    except ValueError:
        return False


def _parse_signals(text: str, target_ids: set[str]) -> list[dict[str, Any]]:
    """Parse a strict JSON response and keep only known bounded signals."""
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return []
        try:
            value = json.loads(text[start : end + 1])
        except (TypeError, ValueError):
            return []
    raw_signals = value.get("signals") if isinstance(value, dict) else value
    if not isinstance(raw_signals, list):
        return []
    result: list[dict[str, Any]] = []
    for raw in raw_signals:
        if not isinstance(raw, dict):
            continue
        memory_id = str(raw.get("memory_id") or "")
        signal_type = str(raw.get("signal_type") or "")
        if memory_id not in target_ids or signal_spec(signal_type) is None:
            continue
        direction, weight, _ = signal_spec(signal_type)  # type: ignore[misc]
        try:
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 1.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence <= 0:
            continue
        result.append(
            {
                "memory_id": memory_id,
                "signal_type": signal_type,
                "direction": direction,
                "weight": min(weight, weight * confidence),
                "source": "inferred",
                "reason": str(raw.get("reason") or "")[:300],
            }
        )
    return result


class EchoEvaluator:
    """Use the configured local chat model, never the foreground model path."""

    def __init__(self, config: RemnantConfig):
        self.config = config

    @property
    def available(self) -> bool:
        return bool(
            self.config.echo_allow_remote_evaluator
            or is_local_endpoint(self.config.extract_url)
        )

    def __call__(self, job: dict[str, Any], payload: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.available:
            return []
        memories = payload.get("memories") or []
        target_ids = {str(item.get("id")) for item in memories if item.get("id")}
        if not target_ids:
            return []
        user_text = str(payload.get("user_text") or "")[-4_000:]
        assistant_text = str(payload.get("assistant_text") or "")[-6_000:]
        memory_text = "\n".join(
            f"MEMORY {item['id']}: {str(item.get('content') or '')[:2_000]}"
            for item in memories
        )
        system = (
            "You are a conservative memory utility evaluator. Return JSON only: "
            '{"signals":[{"memory_id":"...","signal_type":"...",'
            '"confidence":0.0,"reason":"..."}]}. '
            "Choose at most one signal per memory. Use counterfactual_support when a "
            "memory clearly helped answer the user; counterfactual_harm when it was "
            "misleading or contradicted. For a pair job, judge the combination and "
            "return signals for both IDs. Otherwise return an empty list. Never invent IDs."
        )
        prompt = (
            f"User request:\n{user_text}\n\nAssistant answer:\n{assistant_text}\n\n"
            f"Retrieved memories:\n{memory_text}"
        )
        text = chat(
            url=self.config.extract_url,
            model=self.config.extract_model,
            system=system,
            user=prompt,
            timeout=min(float(self.config.extract_timeout), 30.0),
            protocol=self.config.llm_protocol,
            temperature=0.0,
            max_tokens=256,
            keep_alive=self.config.extract_keep_alive,
        )
        return _parse_signals(text, target_ids)


__all__ = ["EchoEvaluator", "is_local_endpoint"]
