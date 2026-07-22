"""Interfaces and value types for the claim-extraction subsystem.

The structured extractor depends on a narrow :class:`StructuredModel` protocol
(raw text in, raw text out) rather than the general ``Generator`` port, which is
coupled to answer generation. This keeps extraction model-agnostic.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from egrag.domain.models import AtomicClaim


class ExtractionConfig(BaseModel):
    """Configuration controlling extraction behavior.

    Pronoun resolution is opt-in and governed by a documented rule: a leading
    pronoun is resolved only when ``resolve_clear_pronouns`` is true *and*
    exactly one distinct named entity precedes it in the passage.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    split_conjunctions: bool = True
    resolve_clear_pronouns: bool = False
    deduplicate: bool = True
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    max_claims: int | None = Field(default=None, ge=0)
    default_confidence: float = Field(default=0.9, ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    """The outcome of extracting claims from one passage.

    ``warnings`` capture run-level diagnostics (e.g. a dropped ungrounded claim)
    that are not attached to any single surviving claim.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    claims: tuple[AtomicClaim, ...] = ()
    warnings: tuple[str, ...] = ()


@runtime_checkable
class StructuredModel(Protocol):
    """A minimal structured-generation backend: a prompt in, raw text out.

    Implementations must not be invoked or downloaded at import time. The raw
    string is parsed and validated by the extractor, never trusted blindly.
    """

    def complete(self, prompt: str, *, seed: int = 0, deterministic: bool = True) -> str: ...


__all__ = ["ExtractionConfig", "ExtractionResult", "StructuredModel"]
