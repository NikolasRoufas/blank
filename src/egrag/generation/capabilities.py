"""Generator capabilities and configuration validation.

Capabilities are declared by each adapter as a provider-independent model. The
configuration is validated against them before any expensive processing: hard
incompatibilities raise an actionable :class:`GenerationError`; soft ones return
warnings.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from egrag.domain.errors import GenerationError
from egrag.generation.config import GenerationConfig


class GeneratorCapabilities(BaseModel):
    """What a generator adapter supports."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chat_template: bool = False
    context_limit: int = Field(default=4096, ge=1)
    has_tokenizer: bool = False
    structured_output: bool = False
    streaming: bool = False
    deterministic_decoding: bool = True
    seed_support: bool = True


def validate_config(config: GenerationConfig, capabilities: GeneratorCapabilities) -> list[str]:
    """Validate a config against capabilities.

    Raises:
        GenerationError: for unsupported features that cannot be honored
            (streaming, or a required deterministic mode the adapter can't give).

    Returns:
        A list of non-fatal warnings (e.g. a seed that will be ignored).
    """

    if config.stream and not capabilities.streaming:
        raise GenerationError("the selected generator adapter does not support streaming")
    warnings: list[str] = []
    if config.deterministic and not capabilities.deterministic_decoding:
        warnings.append(
            "deterministic decoding was requested but this adapter cannot guarantee it; "
            "results may vary between runs"
        )
    if config.seed is not None and not capabilities.seed_support:
        warnings.append("seed is not supported by this adapter and will be ignored")
    return warnings


__all__ = ["GeneratorCapabilities", "validate_config"]
