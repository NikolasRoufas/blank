"""Value types, configuration, and the pair-classifier protocol for graph build.

These types are provider-agnostic and operate on domain models. The graph
package never imports NetworkX, HTTP clients, or model libraries; concrete
classifiers live behind the :class:`PairClassifier` protocol.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from egrag.domain.models import AtomicClaim


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CandidateStrategy(StrEnum):
    """How candidate pairs are produced."""

    BRUTE_FORCE = "brute_force"
    PRUNED = "pruned"


class ClaimPair(_Frozen):
    """An ordered pair of claims to classify (``source`` = premise)."""

    source: AtomicClaim
    target: AtomicClaim
    # Signals that caused this pair to be generated (for auditability).
    reasons: tuple[str, ...] = ()


class RelationProbabilities(_Frozen):
    """NLI-style probabilities for an ordered (premise → hypothesis) pair."""

    entailment: float = Field(ge=0.0, le=1.0)
    contradiction: float = Field(ge=0.0, le=1.0)
    neutral: float = Field(ge=0.0, le=1.0)


class CandidateConfig(_Frozen):
    """Configuration for candidate-pair generation."""

    strategy: CandidateStrategy = CandidateStrategy.PRUNED
    use_shared_entities: bool = True
    use_lexical_overlap: bool = True
    use_subject_predicate: bool = True
    use_temporal_overlap: bool = False
    use_query_relevance: bool = False
    use_source_diversity: bool = False
    use_claim_type: bool = False
    shared_entities_min: int = Field(default=1, ge=1)
    lexical_overlap_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    query_relevance_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    max_pairs: int | None = Field(default=None, ge=0)


class ClassificationConfig(_Frozen):
    """Thresholds turning classifier probabilities into stored edges."""

    entailment_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    contradiction_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    # Mutual entailment above this is treated as a semantic duplicate.
    duplicate_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    store_neutral: bool = False  # debugging only


class DuplicateConfig(_Frozen):
    """Configuration for lexical duplicate detection."""

    detect_exact: bool = True
    detect_normalized: bool = True


class TemporalConfig(_Frozen):
    """Configuration for temporal supersession."""

    enabled: bool = True
    min_update_confidence: float = Field(default=0.6, ge=0.0, le=1.0)


class CandidateStats(_Frozen):
    """Statistics describing candidate-pair generation/pruning."""

    num_claims: int = Field(default=0, ge=0)
    possible_pairs: int = Field(default=0, ge=0)
    generated_pairs: int = Field(default=0, ge=0)
    pruned_pairs: int = Field(default=0, ge=0)
    budget_truncated: int = Field(default=0, ge=0)


class GraphConstructionMetrics(_Frozen):
    """Metrics describing one graph-construction run."""

    num_claims: int = Field(default=0, ge=0)
    possible_pairs: int = Field(default=0, ge=0)
    generated_candidate_pairs: int = Field(default=0, ge=0)
    classified_pairs: int = Field(default=0, ge=0)
    pruned_pairs: int = Field(default=0, ge=0)
    edges_by_type: dict[str, int] = Field(default_factory=dict)
    nli_batch_count: int = Field(default=0, ge=0)
    construction_ms: float = Field(default=0.0, ge=0.0)


class GraphSummary(_Frozen):
    """A compact human-facing summary of a graph's contents."""

    num_nodes: int = Field(default=0, ge=0)
    num_edges: int = Field(default=0, ge=0)
    edges_by_type: dict[str, int] = Field(default_factory=dict)
    num_components: int = Field(default=0, ge=0)
    num_sources: int = Field(default=0, ge=0)


@runtime_checkable
class PairClassifier(Protocol):
    """Classifies ordered claim pairs into relation probabilities (batched).

    Implementations must return one :class:`RelationProbabilities` per input
    pair, in the same order. They expose identity/version metadata for
    provenance and must not load or download models at import time.
    """

    classifier_id: str
    classifier_version: str
    model_revision: str | None

    def classify(self, pairs: list[ClaimPair]) -> list[RelationProbabilities]: ...


__all__ = [
    "CandidateConfig",
    "CandidateStats",
    "CandidateStrategy",
    "ClaimPair",
    "ClassificationConfig",
    "DuplicateConfig",
    "GraphConstructionMetrics",
    "GraphSummary",
    "PairClassifier",
    "RelationProbabilities",
    "TemporalConfig",
]
