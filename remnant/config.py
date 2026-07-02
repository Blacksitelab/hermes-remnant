"""Configuration loading for the Remnant memory plugin.

Reads `config.yaml` keys under `memory.remnant.*` and falls back to sensible
defaults so the plugin works out-of-the-box in a dev environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

DEFAULT_DB_PATH = Path.home() / ".hermes" / "remnant" / "remnant.db"

DEFAULT_EMBED_URL = "http://192.168.0.11:11434/api/embeddings"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768

DEFAULT_EXTRACT_URL = "http://192.168.0.11:11434/v1/chat/completions"
DEFAULT_EXTRACT_MODEL = "gemma4:12b"

# Cosine similarity above this => duplicate memory
DEDUP_COSINE_THRESHOLD = 0.92
# BM25 candidate count when checking duplicates
DEDUP_CANDIDATES = 8
# Keyword search default limit
SEARCH_LIMIT = 10


def _coerce_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class RemnantConfig:
    db_path: str = str(DEFAULT_DB_PATH)
    embed_url: str = DEFAULT_EMBED_URL
    embed_model: str = DEFAULT_EMBED_MODEL
    embed_dim: int = EMBED_DIM
    embed_timeout: float = 30.0
    extract_url: str = DEFAULT_EXTRACT_URL
    extract_model: str = DEFAULT_EXTRACT_MODEL
    extract_timeout: float = 60.0
    extract_enabled: bool = True
    extract_workers: int = 1
    dedup_cosine_threshold: float = DEDUP_COSINE_THRESHOLD
    dedup_candidates: int = DEDUP_CANDIDATES
    search_limit: int = SEARCH_LIMIT
    # Default visibility for auto-extracted memories
    default_visibility: str = "private"
    agent_id: str = "default"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RemnantConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()} - {"extra"}
        kwargs: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for k, v in (d or {}).items():
            if k in known:
                kwargs[k] = v
            else:
                extra[k] = v
        kwargs["extra"] = extra
        return cls(**{k: v for k, v in kwargs.items() if v is not None})


def _extract_section(data: dict[str, Any]) -> dict[str, Any]:
    """Pull the `memory.remnant` sub-section out of a flat config dict."""
    if not data:
        return {}
    memory = data.get("memory", data)
    if isinstance(memory, dict):
        remnant = memory.get("remnant")
        if isinstance(remnant, dict):
            return remnant
    # Allow top-level remnant key
    remnant = data.get("remnant")
    if isinstance(remnant, dict):
        return remnant
    return {}


def load_config(path: str | os.PathLike[str] | None = None) -> RemnantConfig:
    """Load configuration from a YAML file, falling back to defaults.

    Resolution order: explicit `path` arg -> $HERMES_CONFIG -> ./config.yaml
    -> ~/.hermes/config.yaml. Missing files are silently ignored.
    """
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    env_path = os.environ.get("HERMES_CONFIG")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path.cwd() / "config.yaml")
    candidates.append(Path.home() / ".hermes" / "config.yaml")

    data: dict[str, Any] = {}
    for cand in candidates:
        try:
            if cand.is_file():
                with open(cand, "r", encoding="utf-8") as fh:
                    loaded = yaml.safe_load(fh)
                if isinstance(loaded, dict):
                    data = loaded
                    break
        except OSError:
            continue

    section = _extract_section(data)
    return RemnantConfig.from_dict(section)