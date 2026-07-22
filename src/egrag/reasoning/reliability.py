"""Source-reliability strategies.

Reliability priors are **configurable assumptions**, not measured truth. None of
these strategies infer reliability from recency, repetition, ranking, or
popularity — doing so would conflate provenance diversity and temporal signals
with trust. All implement the :class:`egrag.domain.ports.SourceReliabilityScorer`
protocol.
"""

from __future__ import annotations

from collections.abc import Mapping

from egrag.domain.models import SourceMetadata


class UniformReliability:
    """Assigns the same configured reliability to every source."""

    def __init__(self, value: float = 0.5) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"reliability must be in [0, 1], got {value}")
        self._value = value

    def score(self, source: SourceMetadata) -> float:
        return self._value


class ConfiguredPriorReliability:
    """Looks up a per-source prior; falls back to a configured default."""

    def __init__(self, priors: Mapping[str, float], *, default: float = 0.5) -> None:
        for source_id, value in priors.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"prior for {source_id!r} must be in [0, 1], got {value}")
        if not 0.0 <= default <= 1.0:
            raise ValueError(f"default reliability must be in [0, 1], got {default}")
        self._priors = dict(priors)
        self._default = default

    def score(self, source: SourceMetadata) -> float:
        return self._priors.get(source.source_id, self._default)


class MetadataReliability:
    """Uses a source's declared ``reliability_prior`` metadata when present.

    The prior is a configurable assumption attached to the source's metadata; in
    its absence the configured default is used. No inference from recency,
    repetition, ranking, or popularity is performed.
    """

    def __init__(self, *, default: float = 0.5) -> None:
        if not 0.0 <= default <= 1.0:
            raise ValueError(f"default reliability must be in [0, 1], got {default}")
        self._default = default

    def score(self, source: SourceMetadata) -> float:
        if source.reliability_prior is not None:
            return source.reliability_prior
        return self._default


__all__ = ["ConfiguredPriorReliability", "MetadataReliability", "UniformReliability"]
