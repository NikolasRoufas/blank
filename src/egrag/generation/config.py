"""Generation configuration (provider-independent).

Evaluation mode defaults to deterministic decoding. Configuration is validated
against adapter capabilities before any expensive processing.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GenerationConfig(BaseModel):
    """Decoding and transport configuration, independent of any provider."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    deterministic: bool = True
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, allow_inf_nan=False)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, allow_inf_nan=False)
    max_new_tokens: int = Field(default=512, ge=1)
    seed: int | None = Field(default=0, ge=0)
    stop: tuple[str, ...] = ()
    timeout_s: float = Field(default=30.0, gt=0.0, allow_inf_nan=False)
    max_retries: int = Field(default=2, ge=0)
    stream: bool = False

    @classmethod
    def evaluation(cls) -> GenerationConfig:
        """Return a deterministic configuration suitable for evaluation."""

        return cls(deterministic=True, temperature=0.0, top_p=1.0, seed=0, stream=False)


__all__ = ["GenerationConfig"]
