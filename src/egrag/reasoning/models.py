"""Configuration and result models for the reasoning subsystem.

Every quantity is kept as a distinct field — extraction confidence, relation
confidence, source reliability, initial belief, propagated belief, temporal
validity, provenance diversity, query utility, selection contribution, and final
selection score are never collapsed into one another. All weights/config are
validated; NaN/inf are rejected (``allow_inf_nan=False``).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Probability = float


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --- Initial scoring ---------------------------------------------------------


class ScoreWeights(_Frozen):
    """Weights for the documented weighted-sum initial-belief baseline.

    Initial belief is a convex combination ``sum(w_i * signal_i) / sum(w_i)`` of
    six normalized signals, which is numerically stable and bounded in [0, 1]
    (no products of many probabilities). Weights are configurable assumptions.
    """

    retrieval: float = Field(default=0.5, ge=0.0, allow_inf_nan=False)
    query_relevance: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    extraction: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    source_reliability: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    temporal_validity: float = Field(default=0.5, ge=0.0, allow_inf_nan=False)
    independent_support: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _positive_total(self) -> ScoreWeights:
        if self.total() <= 0.0:
            raise ValueError("score weights must sum to a positive value")
        return self

    def total(self) -> float:
        return (
            self.retrieval
            + self.query_relevance
            + self.extraction
            + self.source_reliability
            + self.temporal_validity
            + self.independent_support
        )


class ScoreComponents(_Frozen):
    """The six raw normalized signals feeding initial belief (each in [0, 1])."""

    retrieval: Probability = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    query_relevance: Probability = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    extraction: Probability = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    source_reliability: Probability = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    temporal_validity: Probability = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    independent_support: Probability = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class ClaimScore(_Frozen):
    """All distinct score quantities for one claim."""

    claim_id: str
    components: ScoreComponents
    initial_belief: Probability = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    propagated_belief: Probability | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    query_utility: Probability = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    provenance_diversity: int = Field(ge=0)
    # Per-signal weighted contributions to initial belief (explanation).
    contributions: dict[str, float] = Field(default_factory=dict)

    def with_propagated(self, value: float) -> ClaimScore:
        return self.model_copy(update={"propagated_belief": value})


class ScoreBoard(_Frozen):
    """A serializable collection of claim scores with id lookup."""

    scores: tuple[ClaimScore, ...] = ()

    def by_id(self) -> dict[str, ClaimScore]:
        return {score.claim_id: score for score in self.scores}


# --- Propagation -------------------------------------------------------------


class PropagationConfig(_Frozen):
    """Configuration for signed belief propagation (a configurable baseline)."""

    damping: float = Field(default=0.5, ge=0.0, lt=1.0, allow_inf_nan=False)
    tolerance: float = Field(default=1e-4, gt=0.0, allow_inf_nan=False)
    max_iterations: int = Field(default=50, ge=1)
    support_weight: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    contradiction_weight: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    # Discount applied to repeated supporters sharing a source/lineage (k-th gets
    # discount**(k-1)); 0 means only the first independent supporter counts.
    duplicate_discount: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)
    lineage_discount: float = Field(default=0.25, ge=0.0, le=1.0, allow_inf_nan=False)
    logit_clamp: float = Field(default=12.0, gt=0.0, allow_inf_nan=False)
    on_nonconvergence: Literal["raise", "return"] = "raise"


class IterationDiagnostic(_Frozen):
    """Per-iteration diagnostic (debug mode)."""

    iteration: int = Field(ge=0)
    max_delta: float = Field(ge=0.0, allow_inf_nan=False)


class PropagationResult(_Frozen):
    """The outcome of belief propagation."""

    beliefs: dict[str, float] = Field(default_factory=dict)
    iterations: int = Field(ge=0)
    converged: bool = True
    diagnostics: tuple[IterationDiagnostic, ...] = ()


# --- Selection ---------------------------------------------------------------


class SelectionStrategy(StrEnum):
    """Available subgraph-selection strategies."""

    TOP_CLAIMS = "top_claims"
    GREEDY_CONNECTED = "greedy_connected"
    BEAM = "beam"


class SelectionConfig(_Frozen):
    """Objective weights, penalties, and budget for subgraph selection."""

    strategy: SelectionStrategy = SelectionStrategy.GREEDY_CONNECTED
    # Rewards.
    utility_weight: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    belief_weight: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    entity_coverage_weight: float = Field(default=0.5, ge=0.0, allow_inf_nan=False)
    support_coherence_weight: float = Field(default=0.3, ge=0.0, allow_inf_nan=False)
    independence_weight: float = Field(default=0.3, ge=0.0, allow_inf_nan=False)
    uncertainty_weight: float = Field(default=0.2, ge=0.0, allow_inf_nan=False)
    # Penalties.
    redundancy_penalty: float = Field(default=0.5, ge=0.0, allow_inf_nan=False)
    repeated_lineage_penalty: float = Field(default=0.5, ge=0.0, allow_inf_nan=False)
    disconnected_penalty: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    unresolved_conflict_penalty: float = Field(default=0.3, ge=0.0, allow_inf_nan=False)
    # Search.
    beam_width: int = Field(default=4, ge=1)
    allow_disconnected_fallback: bool = True


class SelectionEntry(_Frozen):
    """Explainability record for one candidate claim (selected or rejected)."""

    claim_id: str
    initial_belief: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    propagated_belief: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    query_utility: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    components: ScoreComponents
    selection_contribution: float = Field(allow_inf_nan=False)
    final_selection_score: float = Field(allow_inf_nan=False)
    selected: bool
    reason: str
    supporting_neighbors: tuple[str, ...] = ()
    contradicting_neighbors: tuple[str, ...] = ()
    duplicate_cluster: tuple[str, ...] = ()
    source_id: str
    tokens: int = Field(ge=0)


class TokenBudget(_Frozen):
    """A deterministic token budget with a reserved output allocation."""

    total: int = Field(ge=0)
    reserved_output: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check(self) -> TokenBudget:
        if self.reserved_output > self.total:
            raise ValueError("reserved_output must not exceed the total budget")
        return self

    @property
    def available(self) -> int:
        return self.total - self.reserved_output


class SelectionResult(_Frozen):
    """The selected reasoning subgraph plus full per-claim explanations."""

    subgraph_id: str
    strategy: SelectionStrategy
    selected_claim_ids: tuple[str, ...] = ()
    relation_ids: tuple[str, ...] = ()
    entries: tuple[SelectionEntry, ...] = ()
    total_tokens: int = Field(ge=0)
    budget: TokenBudget
    connected: bool = True


__all__ = [
    "ClaimScore",
    "IterationDiagnostic",
    "PropagationConfig",
    "PropagationResult",
    "ScoreBoard",
    "ScoreComponents",
    "ScoreWeights",
    "SelectionConfig",
    "SelectionEntry",
    "SelectionResult",
    "SelectionStrategy",
    "TokenBudget",
]
