"""
config.py — Centralized configuration for the HR Agent System.

All runtime parameters are defined here as typed fields with defaults.
Import `settings` from this module throughout the codebase.
Never call os.getenv() or load_dotenv() in any other file.
"""

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Single source of truth for all system configuration.
    Values are loaded from environment variables and the .env file.
    Field names correspond directly to environment variable names (case-insensitive).
    """

    # ── LLM ──────────────────────────────────────────────────────────────────
    openrouter_api_key: str = ""
    model_name: str = "openai/gpt-4o-mini"

    # ── Intent Classifier ────────────────────────────────────────────────────
    classifier_confidence_threshold: float = 0.6
    classifier_timeout_seconds: int = 10
    classifier_max_retries: int = 3
    classifier_retry_delay: float = 1.0

    # ── Short-Term Memory (STM) ──────────────────────────────────────────────
    stm_max_entries: int = 10
    stm_ttl_hours: int = 24
    stm_context_injection_limit: int = 3  # entries injected into LLM prompt

    # ── Long-Term Memory (LTM) ───────────────────────────────────────────────
    ltm_significance_threshold: float = 0.6
    ltm_max_entries_per_user: int = 100
    ltm_retrieval_limit: int = 5

    # ── Sub-Agents ───────────────────────────────────────────────────────────
    agent_timeout_seconds: int = 30
    agent_max_retries: int = 3
    agent_retry_delay: float = 1.0

    # ── API / Endpoints ──────────────────────────────────────────────────────
    max_message_length: int = 2000
    audit_page_limit_max: int = 200
    cors_origins: list[str] = ["*"]

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = "sqlite:///./hr_system.db"
    audit_fallback_file: str = "audit_fallback.jsonl"

    # ── Application ──────────────────────────────────────────────────────────
    log_level: str = "INFO"
    app_version: str = "1.0.0"

    @field_validator("classifier_confidence_threshold", "ltm_significance_threshold")
    @classmethod
    def must_be_probability(cls, v: float) -> float:
        """Ensure threshold values are valid probabilities between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("classifier_timeout_seconds", "agent_timeout_seconds")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        """Ensure timeout values are positive integers."""
        if v <= 0:
            raise ValueError(f"Timeout must be positive, got {v}")
        return v

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Module-level singleton — import this object everywhere.
# Example: from config import settings
settings = Settings()
