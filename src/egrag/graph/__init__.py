"""Evidence-graph construction over domain models.

This package imports no infrastructure libraries (no NetworkX, HTTP, or model
backends). Concrete classifiers sit behind the ``PairClassifier`` protocol and
load any optional dependency lazily. Optional GraphML export lives in
``egrag.adapters.graph`` behind the ``graph`` extra.
"""

from __future__ import annotations

from egrag.graph.api import EvidenceGraph
from egrag.graph.bridges import (
    BridgeDecision,
    BridgeDetector,
    DeterministicBridgeDetector,
    claim_entities,
    detect_bridges,
    extract_entities,
    query_subgoals,
)
from egrag.graph.candidates import CandidateResult, generate_candidates
from egrag.graph.classification import HuggingFaceNLIClassifier, LexicalPairClassifier
from egrag.graph.construction import GraphBuilder, GraphConstructionResult
from egrag.graph.duplicates import DuplicatePair, detect_lexical_duplicates
from egrag.graph.nli import (
    LABEL_VALIDATION_CASES,
    NLILabelMappingError,
    RelationDecision,
    StructuralContradictionGate,
    classify_directional,
    decide_relation,
    structural_contradiction_ok,
    validate_label_mapping,
)
from egrag.graph.serialization import GraphSerializer
from egrag.graph.temporal import SupersessionResolver, resolve_temporal
from egrag.graph.types import (
    CandidateConfig,
    CandidateStats,
    CandidateStrategy,
    ClaimPair,
    ClassificationConfig,
    DuplicateConfig,
    GraphConstructionMetrics,
    GraphSummary,
    PairClassifier,
    RelationProbabilities,
    TemporalConfig,
)
from egrag.graph.validation import validate_components, validate_snapshot

__all__ = [
    "LABEL_VALIDATION_CASES",
    "BridgeDecision",
    "BridgeDetector",
    "CandidateConfig",
    "CandidateResult",
    "CandidateStats",
    "CandidateStrategy",
    "ClaimPair",
    "ClassificationConfig",
    "DeterministicBridgeDetector",
    "DuplicateConfig",
    "DuplicatePair",
    "EvidenceGraph",
    "GraphBuilder",
    "GraphConstructionMetrics",
    "GraphConstructionResult",
    "GraphSerializer",
    "GraphSummary",
    "HuggingFaceNLIClassifier",
    "LexicalPairClassifier",
    "NLILabelMappingError",
    "PairClassifier",
    "RelationDecision",
    "RelationProbabilities",
    "StructuralContradictionGate",
    "SupersessionResolver",
    "TemporalConfig",
    "claim_entities",
    "classify_directional",
    "decide_relation",
    "detect_bridges",
    "detect_lexical_duplicates",
    "extract_entities",
    "generate_candidates",
    "query_subgoals",
    "resolve_temporal",
    "structural_contradiction_ok",
    "validate_components",
    "validate_label_mapping",
    "validate_snapshot",
]
