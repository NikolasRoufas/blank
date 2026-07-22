"""Reasoning subsystem: initial scoring, belief propagation, conflict resolution,
and reasoning-subgraph selection.

All equations and weights are configurable baselines, not universal truth (see
``docs/reasoning.md``). This package imports no infrastructure or model
libraries; optional backends (e.g. tokenizer) load lazily.
"""

from __future__ import annotations

from egrag.reasoning.conflict import ConflictSetResolver
from egrag.reasoning.models import (
    ClaimScore,
    IterationDiagnostic,
    PropagationConfig,
    PropagationResult,
    ScoreBoard,
    ScoreComponents,
    ScoreWeights,
    SelectionConfig,
    SelectionEntry,
    SelectionResult,
    SelectionStrategy,
    TokenBudget,
)
from egrag.reasoning.propagation import NoPropagationBaseline, SignedBeliefPropagator
from egrag.reasoning.reliability import (
    ConfiguredPriorReliability,
    MetadataReliability,
    UniformReliability,
)
from egrag.reasoning.scoring import BaselineInitialScorer
from egrag.reasoning.selection import (
    BeamSearchSelector,
    GreedyConnectedSelector,
    TopClaimsSelector,
    query_entity_coverage,
    to_reasoning_subgraph,
)
from egrag.reasoning.tokens import (
    CharacterTokenCounter,
    HuggingFaceTokenCounter,
    WhitespaceTokenCounter,
)

__all__ = [
    "BaselineInitialScorer",
    "BeamSearchSelector",
    "CharacterTokenCounter",
    "ClaimScore",
    "ConfiguredPriorReliability",
    "ConflictSetResolver",
    "GreedyConnectedSelector",
    "HuggingFaceTokenCounter",
    "IterationDiagnostic",
    "MetadataReliability",
    "NoPropagationBaseline",
    "PropagationConfig",
    "PropagationResult",
    "ScoreBoard",
    "ScoreComponents",
    "ScoreWeights",
    "SelectionConfig",
    "SelectionEntry",
    "SelectionResult",
    "SelectionStrategy",
    "SignedBeliefPropagator",
    "TokenBudget",
    "TopClaimsSelector",
    "UniformReliability",
    "WhitespaceTokenCounter",
    "query_entity_coverage",
    "to_reasoning_subgraph",
]
