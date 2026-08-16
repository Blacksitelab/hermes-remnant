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
DEFAULT_EXTRACT_NUM_CTX = 8_192
DEFAULT_EXTRACT_MAX_INPUT_TOKENS = 5_500
DEFAULT_EXTRACT_MAX_OUTPUT_TOKENS = 1_536
DEFAULT_EXTRACT_MAX_FACTS = 8
DEFAULT_EXTRACT_THINK = False
DEFAULT_EXTRACT_STRUCTURED_OUTPUT = True

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

# Echo: outcome-aware, shadow-first memory utility. These defaults keep the
# foreground path local and bounded; background evaluation is separately capped.
ECHO_ENABLED = True
ECHO_SHADOW_MODE = True
ECHO_POLICY_VERSION = "echo-v1"
ECHO_RANK_INFLUENCE = 0.0
ECHO_RECEIPT_RETENTION_DAYS = 30
ECHO_SIGNAL_RETENTION_DAYS = 30
ECHO_INITIAL_SAMPLE_RATE = 0.05
ECHO_MATURE_SAMPLE_RATE = 0.005
ECHO_MATURE_OBSERVATIONS = 20
ECHO_MAX_JOBS_PER_DAY = 20
ECHO_MAX_EVALUATOR_SECONDS_PER_DAY = 300
ECHO_WORKER_POLL_INTERVAL_S = 5
ECHO_JOB_STALE_AFTER_S = 900
ECHO_JOB_MAX_ATTEMPTS = 3
ECHO_MIN_OBSERVATIONS = 10
ECHO_MAX_RANK_ADJUSTMENT = 0.10
ECHO_UTILITY_HALF_LIFE_DAYS = 90.0
ECHO_EXPLICIT_FEEDBACK_HALF_LIFE_DAYS = 365.0
ECHO_PAIR_ATTRIBUTION_ENABLED = True
ECHO_MAX_PAIRS_PER_RECEIPT = 3
ECHO_MAX_PAIRS_PER_MEMORY_ARCHETYPE = 20
ECHO_PAIR_HALF_LIFE_DAYS = 60.0
ECHO_HOT_PATH_BUDGET_MS = 3
ECHO_DISABLE_ON_BUDGET_EXCEEDED = True
ECHO_PAUSE_WHEN_MODEL_BUSY = True
ECHO_ALLOW_REMOTE_EVALUATOR = False
# Reflection input cap (top-N memories) and output cap.
REFLECT_TOP_N = 20
REFLECT_MAX_TOKENS = 512

CONFIG_FILENAME = "remnant.json"

CONFIG_PRESETS: dict[str, dict[str, Any]] = {
    "claim_aware": {
        "structured_claim_extraction_v2": True,
        "claim_reconciliation_enabled": True,
        "claim_aware_ranking_enabled": True,
        "resolved_context_enabled": True,
        "recent_turn_overlay_enabled": True,
        "relation_evidence_enabled": True,
        "ranking_profile": "claims-v1",
        "runtime_identity_enabled": False,
    },
    "legacy": {
        "structured_claim_extraction_v2": False,
        "claim_reconciliation_enabled": False,
        "claim_aware_ranking_enabled": False,
        "resolved_context_enabled": False,
        "recent_turn_overlay_enabled": False,
        "relation_evidence_enabled": False,
        "ranking_profile": "legacy",
        "runtime_identity_enabled": False,
    },
    "claim_aware_shadow": {
        "structured_claim_extraction_v2": True,
        "claim_reconciliation_enabled": True,
        "claim_aware_ranking_enabled": False,
        "resolved_context_enabled": False,
        "recent_turn_overlay_enabled": True,
        "relation_evidence_enabled": True,
        "ranking_profile": "claims-v1",
        "runtime_identity_enabled": False,
    },
}


def apply_config_preset(values: dict[str, Any] | None, name: str) -> dict[str, Any]:
    """Apply a named behavior profile while preserving unrelated settings."""
    preset = CONFIG_PRESETS.get(str(name or "").strip().casefold())
    if preset is None:
        raise ValueError(f"unknown Remnant config preset: {name}")
    merged = dict(values or {})
    merged.update(preset)
    return merged


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
    # Extraction is a bounded background task: a single turn does not need
    # the model's full deployment context window or a long completion.
    extract_num_ctx: int = DEFAULT_EXTRACT_NUM_CTX
    extract_max_input_tokens: int = DEFAULT_EXTRACT_MAX_INPUT_TOKENS
    extract_max_output_tokens: int = DEFAULT_EXTRACT_MAX_OUTPUT_TOKENS
    extract_max_facts: int = DEFAULT_EXTRACT_MAX_FACTS
    extract_think: bool = DEFAULT_EXTRACT_THINK
    extract_structured_output: bool = DEFAULT_EXTRACT_STRUCTURED_OUTPUT
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
    # Claim-aware memory is the recommended operating profile.  Every switch
    # remains independently overridable so an installation can shadow or roll
    # back a behavior without discarding its underlying evidence rows.
    structured_claim_extraction_v2: bool = True
    claim_reconciliation_enabled: bool = True
    claim_aware_ranking_enabled: bool = True
    resolved_context_enabled: bool = True
    recent_turn_overlay_enabled: bool = True
    relation_evidence_enabled: bool = True
    # Keep runtime identity opt-in until Hermes supplies a stable user/platform
    # identity.  With no stable identity the fail-closed fallback is session
    # scoped, which would otherwise prevent useful cross-session recall.
    runtime_identity_enabled: bool = False
    ranking_profile: str = "claims-v1"
    recent_turn_overlay_limit: int = 3
    recent_turn_overlay_max_age_s: int = 900
    recent_turn_overlay_max_chars: int = 4000
    prefetch_cache_ttl_s: int = 60
    prefetch_cache_max_entries: int = 32
    # Echo outcome-aware utility; shadow mode is safe by default.
    echo_enabled: bool = ECHO_ENABLED
    echo_shadow_mode: bool = ECHO_SHADOW_MODE
    echo_policy_version: str = ECHO_POLICY_VERSION
    echo_rank_influence: float = ECHO_RANK_INFLUENCE
    echo_receipt_retention_days: int = ECHO_RECEIPT_RETENTION_DAYS
    echo_signal_retention_days: int = ECHO_SIGNAL_RETENTION_DAYS
    echo_initial_sample_rate: float = ECHO_INITIAL_SAMPLE_RATE
    echo_mature_sample_rate: float = ECHO_MATURE_SAMPLE_RATE
    echo_mature_observations: int = ECHO_MATURE_OBSERVATIONS
    echo_max_jobs_per_day: int = ECHO_MAX_JOBS_PER_DAY
    echo_max_evaluator_seconds_per_day: int = ECHO_MAX_EVALUATOR_SECONDS_PER_DAY
    echo_worker_poll_interval_s: int = ECHO_WORKER_POLL_INTERVAL_S
    echo_job_stale_after_s: int = ECHO_JOB_STALE_AFTER_S
    echo_job_max_attempts: int = ECHO_JOB_MAX_ATTEMPTS
    echo_min_observations: int = ECHO_MIN_OBSERVATIONS
    echo_max_rank_adjustment: float = ECHO_MAX_RANK_ADJUSTMENT
    echo_utility_half_life_days: float = ECHO_UTILITY_HALF_LIFE_DAYS
    echo_explicit_feedback_half_life_days: float = ECHO_EXPLICIT_FEEDBACK_HALF_LIFE_DAYS
    echo_pair_attribution_enabled: bool = ECHO_PAIR_ATTRIBUTION_ENABLED
    echo_max_pairs_per_receipt: int = ECHO_MAX_PAIRS_PER_RECEIPT
    echo_max_pairs_per_memory_archetype: int = ECHO_MAX_PAIRS_PER_MEMORY_ARCHETYPE
    echo_pair_half_life_days: float = ECHO_PAIR_HALF_LIFE_DAYS
    echo_hot_path_budget_ms: int = ECHO_HOT_PATH_BUDGET_MS
    echo_disable_on_budget_exceeded: bool = ECHO_DISABLE_ON_BUDGET_EXCEEDED
    echo_pause_when_model_busy: bool = ECHO_PAUSE_WHEN_MODEL_BUSY
    echo_allow_remote_evaluator: bool = ECHO_ALLOW_REMOTE_EVALUATOR
    runtime_user_aliases: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("extra", None)
        d.update(self.extra)
        return d

    def validate(self) -> RemnantConfig:
        """Normalize and validate Echo controls at one configuration boundary."""
        for name in (
            "extract_num_ctx",
            "extract_max_input_tokens",
            "extract_max_output_tokens",
            "extract_max_facts",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            setattr(self, name, value)
        if (
            self.extract_max_input_tokens + self.extract_max_output_tokens
            >= self.extract_num_ctx
        ):
            raise ValueError(
                "extract_max_input_tokens + extract_max_output_tokens must be "
                "less than extract_num_ctx"
            )
        self.echo_rank_influence = float(self.echo_rank_influence)
        self.echo_initial_sample_rate = float(self.echo_initial_sample_rate)
        self.echo_mature_sample_rate = float(self.echo_mature_sample_rate)
        self.echo_max_rank_adjustment = float(self.echo_max_rank_adjustment)
        self.echo_utility_half_life_days = float(self.echo_utility_half_life_days)
        self.echo_explicit_feedback_half_life_days = float(
            self.echo_explicit_feedback_half_life_days
        )
        self.echo_pair_half_life_days = float(self.echo_pair_half_life_days)
        for name in (
            "echo_receipt_retention_days",
            "echo_signal_retention_days",
            "echo_mature_observations",
            "echo_max_jobs_per_day",
            "echo_max_evaluator_seconds_per_day",
            "echo_worker_poll_interval_s",
            "echo_job_stale_after_s",
            "echo_job_max_attempts",
            "echo_min_observations",
            "echo_max_pairs_per_receipt",
            "echo_max_pairs_per_memory_archetype",
            "echo_hot_path_budget_ms",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            setattr(self, name, value)
        for name in (
            "echo_rank_influence",
            "echo_initial_sample_rate",
            "echo_mature_sample_rate",
            "echo_max_rank_adjustment",
        ):
            value = float(getattr(self, name))
            upper = 1.0 if name != "echo_max_rank_adjustment" else 0.25
            if not 0.0 <= value <= upper:
                raise ValueError(f"{name} must be between 0 and {upper}")
            setattr(self, name, value)
        for name in (
            "echo_utility_half_life_days",
            "echo_explicit_feedback_half_life_days",
            "echo_pair_half_life_days",
        ):
            value = float(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            setattr(self, name, value)
        if self.echo_max_pairs_per_receipt > 5:
            raise ValueError("echo_max_pairs_per_receipt must be <= 5")
        if not str(self.echo_policy_version).strip():
            raise ValueError("echo_policy_version must not be empty")
        return self

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
        return cls(**{k: v for k, v in kwargs.items() if v is not None}).validate()


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
