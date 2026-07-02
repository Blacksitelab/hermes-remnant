"""Configuration for the Remnant memory provider.

Config lives at `<hermes_home>/remnant.json` and is profile-scoped. All paths
in the plugin derive from `hermes_home`, never from a hardcoded `~/.hermes`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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
# Max turns queued for extraction before backpressure
QUEUE_MAX = 256

CONFIG_FILENAME = "remnant.json"


@dataclass
class RemnantConfig:
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
    default_visibility: str = "private"
    agent_id: str = "default"
    queue_max: int = QUEUE_MAX
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("extra", None)
        d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> RemnantConfig:
        d = d or {}
        known = set(cls.__dataclass_fields__.keys()) - {"extra"}
        kwargs: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for k, v in d.items():
            if k in known:
                kwargs[k] = v
            else:
                extra[k] = v
        kwargs["extra"] = extra
        return cls(**{k: v for k, v in kwargs.items() if v is not None})


def config_path(hermes_home: str | Path) -> Path:
    return Path(hermes_home) / CONFIG_FILENAME


def load_config(hermes_home: str | Path) -> RemnantConfig:
    """Load config from `<hermes_home>/remnant.json`, falling back to defaults."""
    p = config_path(hermes_home)
    if p.is_file():
        try:
            with open(p, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return RemnantConfig.from_dict(data)
        except (OSError, json.JSONDecodeError):
            pass
    return RemnantConfig()


def save_config(values: dict[str, Any], hermes_home: str | Path) -> Path:
    """Write non-secret config to `<hermes_home>/remnant.json`. Returns the path."""
    p = config_path(hermes_home)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Merge onto defaults so partial config still has sensible values.
    merged = RemnantConfig.from_dict(values).to_dict()
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)
    return p
