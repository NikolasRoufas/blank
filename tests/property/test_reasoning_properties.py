"""Property-based tests for propagation bounds, determinism, and stability."""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from egrag.domain.models import (
    AtomicClaim,
    ClaimProvenance,
    ClaimSemantics,
    EvidenceGraphSnapshot,
    EvidenceRelation,
    RelationDirection,
    RelationType,
    SourceMetadata,
    SourceSpan,
)
from egrag.graph.api import EvidenceGraph
from egrag.reasoning import ScoreBoard, ScoreComponents, SignedBeliefPropagator
from egrag.reasoning.models import ClaimScore

prob = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


def _claim(cid: str, source_id: str) -> AtomicClaim:
    text = f"claim {cid}"
    return AtomicClaim(
        claim_id=cid,
        text=text,
        provenance=ClaimProvenance(
            source=SourceMetadata(source_id=source_id),
            spans=(SourceSpan(source_id=source_id, start=0, end=len(text), text=text),),
        ),
        extraction_confidence=0.8,
        semantics=ClaimSemantics(),
    )


def _score(cid: str, belief: float) -> ClaimScore:
    return ClaimScore(
        claim_id=cid,
        components=ScoreComponents(
            retrieval=0.5,
            query_relevance=0.5,
            extraction=0.5,
            source_reliability=0.5,
            temporal_validity=1.0,
            independent_support=0.0,
        ),
        initial_belief=belief,
        query_utility=belief,
        provenance_diversity=0,
    )


def _chain_graph() -> EvidenceGraph:
    claims = [_claim("a", "s1"), _claim("b", "s2"), _claim("c", "s3")]
    relations = [
        EvidenceRelation(
            relation_id="r1",
            source_claim_id="a",
            target_claim_id="b",
            relation_type=RelationType.SUPPORT,
            relation_confidence=0.8,
            direction=RelationDirection.DIRECTED,
        ),
        EvidenceRelation(
            relation_id="r2",
            source_claim_id="b",
            target_claim_id="c",
            relation_type=RelationType.CONTRADICTION,
            relation_confidence=0.8,
            direction=RelationDirection.SYMMETRIC,
        ),
    ]
    return EvidenceGraph(
        EvidenceGraphSnapshot(snapshot_id="g", claims=tuple(claims), relations=tuple(relations))
    )


@pytest.mark.property
@given(a=prob, b=prob, c=prob)
def test_propagated_beliefs_bounded_and_finite(a: float, b: float, c: float) -> None:
    graph = _chain_graph()
    board = ScoreBoard(scores=(_score("a", a), _score("b", b), _score("c", c)))
    result = SignedBeliefPropagator().propagate(graph, board)
    for value in result.beliefs.values():
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0


@pytest.mark.property
@given(a=prob, b=prob, c=prob)
def test_propagation_is_deterministic(a: float, b: float, c: float) -> None:
    graph = _chain_graph()
    board = ScoreBoard(scores=(_score("a", a), _score("b", b), _score("c", c)))
    first = SignedBeliefPropagator().propagate(graph, board)
    second = SignedBeliefPropagator().propagate(graph, board)
    assert first.beliefs == second.beliefs
    assert first.iterations == second.iterations
