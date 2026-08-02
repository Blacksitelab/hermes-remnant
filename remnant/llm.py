"""Small adapter for the two chat API shapes Remnant supports."""

from __future__ import annotations

from typing import Any

import httpx


class LLMResponseError(ValueError):
    """The endpoint responded successfully but not with usable chat content."""


def _protocol_for(url: str, protocol: str | None = None) -> str:
    selected = (protocol or "auto").strip().lower()
    if selected in {"ollama", "ollama_native", "native"}:
        return "ollama_native"
    if selected in {"openai", "openai_compatible", "openai-compatible"}:
        return "openai_compatible"
    return "ollama_native" if "/api/chat" in (url or "").lower() else "openai_compatible"


def _content_from_response(data: Any, protocol: str) -> str:
    if not isinstance(data, dict):
        raise LLMResponseError("chat response was not an object")
    if protocol == "ollama_native":
        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
    else:
        choices = data.get("choices")
        first = choices[0] if isinstance(choices, list) and choices else None
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise LLMResponseError("chat response did not contain message content")
    return content.strip()


def chat(
    *,
    url: str,
    model: str,
    system: str,
    user: str,
    timeout: float,
    protocol: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    client: httpx.Client | None = None,
) -> str:
    """Call a configured chat endpoint and return normalized text.

    Transport and response-shape errors deliberately propagate. Callers that
    persist work must distinguish those failures from a valid empty response.
    """
    selected = _protocol_for(url, protocol)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if selected == "ollama_native":
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "keep_alive": -1,
        }
    else:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    owned_client = client is None
    active_client = client or httpx.Client(timeout=timeout)
    try:
        response = active_client.post(url, json=payload)
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMResponseError("chat response was not valid JSON") from exc
    finally:
        if owned_client:
            active_client.close()
    return _content_from_response(data, selected)


__all__ = ["LLMResponseError", "chat"]
