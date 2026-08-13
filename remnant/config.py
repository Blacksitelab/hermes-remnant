"""Configuration for the Remnant memory provider.

Config lives at `<hermes_home>/remnant.json` and is profile-scoped. All paths
in the plugin derive from `hermes_home`, never from a hardcoded `~/.hermes`.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_EMBED_URL = "http://your-ollama-host.local:11434/api/embeddings"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768
DEFAULT_EMBED_KEEP_ALIVE = "10m"

DEFAULT_EXTRACT_URL = "http://your-ollama-host.local:11434/api/chat"
DEFAULT_EXTRACT_MODEL = "gemma4:12b"
DEFAULT_EXTRACT_KEEP_ALIVE = "2m"

# Phase 4: Obsidian vault indexing. The vault is the single source of truth for
# notes; we index it as type='document' memories. Excluded folders hold agent
# scratch/workspace trees that must never be ingested. The vault path default
# can be overridden via the REMNANT_VAULT_PATH env var (useful for tests and
# deployments that store the vault elsewhere than the hardcoded BSL path).
DEFAULT_VAULT_PATH = os.environ.get(
    "REMNANT_VAULT_PATH", "/path/to/your/obsidian-vault"
)
DEFAULT_VAULT_EXCLUDE = ["90_", "91_", "92_", "93_", "94_", "95_", "99_ARCHIVE"]
DEFAULT_VAULT_REINDEX_INTERVAL_S = 600
DEFAULT_VAULT_PASSAGE_CHARS = 1_200
DEFAULT_VAULT_PASSAGE_OVERLAP = 150

# Reflection reuses the extraction endpoint/model by default (gemma4:12b on BSL1).
DEFAULT_REFLECT_URL = DEFAULT_EXTRACT_URL
DEFAULT_REFLECT_MODEL = DEFAULT_EXTRACT_MODEL

# Phase 5: Dream loop. A bounded candidate list is pre-filtered locally with
# cosine similarity, then sent to a cloud model for judgment. Day/night have
# independent endpoints, budgets, and cooldowns. The dream loop is invoked by
# an external cron/systemd timer (day_dream() / night_dream()), never a daemon.
# The dream cloud model is reached via the local Ollama proxy (localhost), distinct from
# the embedding/extraction host so day/night prompts can be routed to a different model.
DEFAULT_DREAM_DAY_URL = "http://localhost:11434/v1/chat/completions"
DEFAULT_DREAM_DAY_MODEL = "deepseek-v4-flash:cloud"
DEFAULT_DREAM_NIGHT_URL = "http://localhost:11434/v1/chat/completions"
DEFAULT_DREAM_NIGHT_MODEL = "deepseek-v4-flash:cloud"
DEFAULT_DREAM_DAY_BUDGET = 3
DEFAULT_DREAM_NIGHT_BUDGET = 5
DEFAULT_DREAM_COOLDOWN_MINUTES = 120  # 2 hours per topic
DEFAULT_DREAM_DIARY_PATH = "~/.hermes/remnant/DREAMS.md"
# Bounded candidate list cap. The cloud model is only ever sent up to this many
# memory pairs; the rest of the corpus stays local.
DREAM_MAX_CANDIDATE_PAIRS = 30
# Top-K similar active memories per recent memory (local pre-filter).
DREAM_TOP_K = 5
# Cosine thresholds. >0.6 => candidate connection; >0.7 => cross-agent dedup.
DREAM_CONNECT_THRESHOLD = 0.6
DREAM_DEDUP_THRESHOLD = 0.7

# Cosine similarity above this => duplicate memory
DEDUP_COSINE_THRESHOLD = 0.85
# BM25 candidate count when checking duplicates
DEDUP_CANDIDATES = 8
# Keyword search default limit
SEARCH_LIMIT = 10
# Trust score time decay (issue #16). Half-life in days; floor is the lowest
# trust a memory can decay to through staleness. Set enabled=False to disable.
TRUST_DECAY_ENABLED = True
TRUST_DECAY_HALF_LIFE_DAYS = 30.0
TRUST_DECAY_FLOOR = 0.3
# Max turns queued for extraction before backpressure
QUEUE_MAX = 256

# Phase 2: proactive injection + semantic search
DEFAULT_INJECTION_TOKEN_BUDGET = 2000
DEFAULT_INJECTION_PREFETCH_DEADLINE_MS = 500
DEFAULT_PREFETCH_ENABLED = True
# Maximum time reserved for the one remote query-embedding call inside
# prefetch().  The remainder of the 500ms prefetch budget is kept for local
# search, formatting, and SQLite diagnostics.
DEFAULT_PREFETCH_EMBEDDING_TIMEOUT_MS = 250
# Upper bound for the local exact-vector scan. The default covers a personal
# or small fleet store while keeping retrieval predictable; use an ANN index
# once the corpus grows beyond this operational ceiling.
SEMANTIC_SCAN_LIMIT = 5_000
# Minimum cosine similarity for semantic/auto results. Top semantic score below
# this => no strong matches; for ``auto`` strategy we still fall back to BM25.
MIN_SEMANTIC_SCORE = 0.3
# RRF constant for hybrid fusion.
RRF_K = 60
# Reflection input cap (top-N memories) and output cap.
REFLECT_TOP_N = 20
REFLECT_MAX_TOKENS = 512

CONFIG_FILENAME = "remnant.json"


@dataclass
class RemnantConfig:
    embed_url: str = DEFAULT_EMBED_URL
    embed_model: str = DEFAULT_EMBED_MODEL
    embed_dim: int = EMBED_DIM
    embed_timeout: float = 30.0
    # Ollama model residency must be finite: extraction and embedding share a
    # host and an indefinitely pinned model can starve the other workload.
    embed_keep_alive: str | int | float = DEFAULT_EMBED_KEEP_ALIVE
    extract_url: str = DEFAULT_EXTRACT_URL
    extract_model: str = DEFAULT_EXTRACT_MODEL
    extract_timeout: float = 120.0
    extract_enabled: bool = True
    extract_workers: int = 1
    # ``auto`` infers Ollama-native vs OpenAI-compatible chat behavior from the
    # endpoint path. Deployments can pin the protocol during migration.
    llm_protocol: str = "auto"
    extract_keep_alive: str | int | float = DEFAULT_EXTRACT_KEEP_ALIVE
    reflect_url: str = DEFAULT_REFLECT_URL
    reflect_model: str = DEFAULT_REFLECT_MODEL
    reflect_timeout: float = 60.0
    # Phase 5: dream loop (callable from cron/systemd; not a daemon).
    dream_day_url: str = DEFAULT_DREAM_DAY_URL
    dream_day_model: str = DEFAULT_DREAM_DAY_MODEL
    dream_night_url: str = DEFAULT_DREAM_NIGHT_URL
    dream_night_model: str = DEFAULT_DREAM_NIGHT_MODEL
    dream_day_timeout: float = 90.0
    dream_night_timeout: float = 120.0
    dream_day_budget: int = DEFAULT_DREAM_DAY_BUDGET
    dream_night_budget: int = DEFAULT_DREAM_NIGHT_BUDGET
    dream_cooldown_minutes: int = DEFAULT_DREAM_COOLDOWN_MINUTES
    diary_path: str = DEFAULT_DREAM_DIARY_PATH
    injection_token_budget: int = DEFAULT_INJECTION_TOKEN_BUDGET
    injection_prefetch_deadline_ms: int = DEFAULT_INJECTION_PREFETCH_DEADLINE_MS
    prefetch_embedding_timeout_ms: int = DEFAULT_PREFETCH_EMBEDDING_TIMEOUT_MS
    prefetch_enabled: bool = DEFAULT_PREFETCH_ENABLED
    dedup_cosine_threshold: float = DEDUP_COSINE_THRESHOLD
    dedup_candidates: int = DEDUP_CANDIDATES
    search_limit: int = SEARCH_LIMIT
    min_semantic_score: float = MIN_SEMANTIC_SCORE
    default_search_strategy: str = "auto"
    semantic_scan_limit: int = SEMANTIC_SCAN_LIMIT
    default_visibility: str = "private"
    agent_id: str = "default"
    queue_max: int = QUEUE_MAX
    # Trust score time decay (issue #16).
    trust_decay_enabled: bool = TRUST_DECAY_ENABLED
    trust_decay_half_life_days: float = TRUST_DECAY_HALF_LIFE_DAYS
    trust_decay_floor: float = TRUST_DECAY_FLOOR
    # Phase 4: vault indexing + profile-scoped search.
    vault_path: str = DEFAULT_VAULT_PATH
    vault_exclude: list[str] = field(default_factory=lambda: list(DEFAULT_VAULT_EXCLUDE))
    profile_scope: list[str] = field(default_factory=list)
    vault_reindex_interval_s: int = DEFAULT_VAULT_REINDEX_INTERVAL_S
    # Heading-aware vault passages keep long notes precise at retrieval time.
    # Set passage_chars to 0 to retain legacy whole-note indexing.
    vault_passage_chars: int = DEFAULT_VAULT_PASSAGE_CHARS
    vault_passage_overlap: int = DEFAULT_VAULT_PASSAGE_OVERLAP
    # Entity extraction tuning (issue #5). Newly regex-extracted entities must
    # be sighted in at least this many distinct memories before being
    # persisted/linked. The LLM typed-entity path bypasses this threshold (the
    # extraction model already curates entities). Defaults to 2 so one-off
    # capitalized phrases (dates, places, generic nouns) do not pollute the
    # graph; set to 1 to restore the old always-link behaviour.
    entity_min_memories: int = 2
    # Issue #21/#22: max entities linked/related per memory. Caps the entity
    # graph to avoid complete-graph relation explosions and over-extraction.
    entity_max_entities: int = 15
    # Release-track memory correctness features.  They are independently
    # switchable so an upgraded installation can shadow and roll back each
    # behavior without changing the underlying evidence rows.
    structured_claim_extraction_v2: bool = False
    claim_reconciliation_enabled: bool = False
    claim_aware_ranking_enabled: bool = False
    resolved_context_enabled: bool = False
    recent_turn_overlay_enabled: bool = False
    relation_evidence_enabled: bool = False
    runtime_identity_enabled: bool = False
    ranking_profile: str = "legacy"
    recent_turn_overlay_limit: int = 3
    recent_turn_overlay_max_age_s: int = 900
    recent_turn_overlay_max_chars: int = 4000
    prefetch_cache_ttl_s: int = 60
    prefetch_cache_max_entries: int = 32
    runtime_user_aliases: dict[str, str] = field(default_factory=dict)
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
    existing: dict[str, Any] = {}
    if p.is_file():
        try:
            with open(p, encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                existing = loaded
        except (OSError, json.JSONDecodeError):
            existing = {}
    # Preserve settings not exposed by the current dashboard schema. This is
    # important for hand-tuned timeouts, budgets, and rollout flags, including
    # finite keep-alive values.
    merged = RemnantConfig.from_dict({**existing, **(values or {})}).to_dict()
    fd, temp_name = tempfile.mkstemp(prefix=".remnant-", suffix=".json", dir=p.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(merged, fh, indent=2)
            fh.write("\n")
        os.replace(temp_name, p)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return p
