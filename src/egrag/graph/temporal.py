"""Temporal resolution for SUPERSEDES edges.

A SUPERSEDES edge is created only when two claims concern the same
update-sensitive proposition (same subject and predicate, with a differing
value), there is a sufficient temporal order, and the update confidence clears a
threshold. Unknown timestamps yield no supersession. Newer is never assumed
true — both claims are preserved and belief is untouched.

Event time (``valid_from``/``asserted_at``) is preferred over publication time
(``observed_at``/``source.published_at``); unknown time is represented as
``None``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from egrag.domain.models import AtomicClaim
from egrag.graph.types import TemporalConfig


@dataclass(frozen=True)
class TemporalInfo:
    """Resolved times for a claim; ``None`` means explicitly unknown."""

    event_time: datetime | None
    publication_time: datetime | None

    @property
    def effective_time(self) -> datetime | None:
        return self.event_time if self.event_time is not None else self.publication_time


@dataclass(frozen=True)
class SupersessionEdge:
    """A resolved supersession: ``newer_id`` supersedes ``older_id``."""

    newer_id: str
    older_id: str
    confidence: float
    explanation: str


def resolve_temporal(claim: AtomicClaim) -> TemporalInfo:
    """Resolve a claim's event and publication times (either may be unknown)."""

    event_time = claim.valid_from if claim.valid_from is not None else claim.asserted_at
    publication_time = (
        claim.provenance.observed_at
        if claim.provenance.observed_at is not None
        else claim.provenance.source.published_at
    )
    return TemporalInfo(event_time=event_time, publication_time=publication_time)


def _update_confidence(a: AtomicClaim, b: AtomicClaim) -> float:
    """Confidence that a and b concern the same update-sensitive proposition."""

    if a.semantics is None or b.semantics is None:
        return 0.0
    subject_a = (a.semantics.subject or "").casefold()
    subject_b = (b.semantics.subject or "").casefold()
    if not subject_a or subject_a != subject_b:
        return 0.0  # different/absent subject -> not the same proposition

    predicate_a = (a.semantics.predicate or "").casefold()
    predicate_b = (b.semantics.predicate or "").casefold()
    confidence = 0.9 if predicate_a and predicate_a == predicate_b else 0.5

    # Require an actual change in value; identical value is a duplicate, not an update.
    object_a = (a.semantics.object or "").casefold()
    object_b = (b.semantics.object or "").casefold()
    differs = (
        object_a != object_b
        or a.semantics.quantities != b.semantics.quantities
        or a.semantics.temporal_expressions != b.semantics.temporal_expressions
    )
    return confidence if differs else 0.0


class SupersessionResolver:
    """Derives SUPERSEDES edges from temporal order and update confidence."""

    def __init__(self, config: TemporalConfig | None = None) -> None:
        self._config = config or TemporalConfig()

    def resolve(self, claims: Sequence[AtomicClaim]) -> list[SupersessionEdge]:
        if not self._config.enabled:
            return []
        ordered = sorted(claims, key=lambda c: c.claim_id)
        edges: list[SupersessionEdge] = []
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a, b = ordered[i], ordered[j]
                confidence = _update_confidence(a, b)
                if confidence < self._config.min_update_confidence:
                    continue
                time_a = resolve_temporal(a).effective_time
                time_b = resolve_temporal(b).effective_time
                if time_a is None or time_b is None or time_a == time_b:
                    continue  # insufficient temporal order
                newer, older = (a, b) if time_a > time_b else (b, a)
                edges.append(
                    SupersessionEdge(
                        newer_id=newer.claim_id,
                        older_id=older.claim_id,
                        confidence=round(confidence, 6),
                        explanation="same subject/predicate with a newer, differing value",
                    )
                )
        edges.sort(key=lambda e: (e.newer_id, e.older_id))
        return edges


__all__ = ["SupersessionEdge", "SupersessionResolver", "TemporalInfo", "resolve_temporal"]
