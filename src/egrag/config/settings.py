"""Typed, validated application settings.

Settings are read from the environment (prefix ``EGRAG_``) and, optionally, an
env file. There is no hidden global state: callers construct or load a settings
instance explicitly and pass it where needed.
"""

from __future__ import annotations

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from egrag.domain.errors import ConfigurationError


class EGRagSettings(BaseSettings):
    """Runtime configuration for an EG-RAG pipeline run."""

    model_config = SettingsConfigDict(
        env_prefix="EGRAG_",
        extra="forbid",
        frozen=True,
        case_sensitive=False,
    )

    # Reproducibility
    seed: int = Field(default=0, ge=0)
    deterministic: bool = True

    # Retrieval / selection budgets
    top_k: int = Field(default=5, ge=1)
    selection_budget: int = Field(default=5, ge=1)

    # Chunking / reranking
    chunk_size: int = Field(default=512, ge=1)
    chunk_overlap: int = Field(default=64, ge=0)
    rerank_top_n: int = Field(default=5, ge=1)

    # Generation
    max_answer_tokens: int = Field(default=256, ge=1)

    # Caching
    cache_enabled: bool = False

    # Observability
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _check_overlap(self) -> EGRagSettings:
        if self.chunk_overlap >= self.chunk_size:
            msg = (
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})"
            )
            raise ValueError(msg)
        return self


def load_settings(env_file: str | None = None) -> EGRagSettings:
    """Load settings from the environment and an optional env file.

    Raises:
        ConfigurationError: if the resolved configuration is invalid, with an
            actionable message describing the offending field(s).
    """

    try:
        if env_file is not None:
            return EGRagSettings(_env_file=env_file)
        return EGRagSettings()
    except ValueError as exc:
        raise ConfigurationError(f"invalid EG-RAG configuration: {exc}") from exc


__all__ = ["EGRagSettings", "load_settings"]
