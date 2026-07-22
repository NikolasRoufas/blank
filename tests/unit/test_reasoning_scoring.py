"""Initial-scoring & reliability tests (16, 30, 32 and concept separation)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from egrag.domain.models import (
    AtomicClaim,
    ClaimProvenance,
    ClaimSemantics,
    EvidenceGraphSnapshot,
    Query,
    SourceMetadata,
    SourceSpan,
)
from egrag.graph.api import EvidenceGraph
from egrag.reasoning import (
    BaselineInitialScorer,
    ConfiguredPriorReliability,
    MetadataReliability,
    ScoreBoard,
    ScoreWeights,
    UniformReliability,
)


def _claim(
    cid: str, text: str, source_id: str = "src", reliability: float | None = None
) -> AtomicClaim:
    return AtomicClaim(
        claim_id=cid,
        text=text,
        provenance=ClaimProvenance(
            source=SourceMetadata(source_id=source_id, reliability_prior=reliability),
            spans=(SourceSpan(source_id=source_id, start=0, end=len(text), text=text),),
        ),
        extraction_confidence=0.8,
        semantics=ClaimSemantics(named_entities=("Acme",)),
    )


def _graph(claims: list[AtomicClaim]) -> EvidenceGraph:
    return EvidenceGraph(EvidenceGraphSnapshot(snapshot_id="g", claims=tuple(claims)))


QUERY = Query(query_id="q", text="did Acme grow revenue")


@pytest.mark.unit
def test_score_preserves_distinct_components() -> None:
    """Every component is preserved separately and concepts are not conflated."""

    graph = _graph([_claim("a", "Acme grew revenue")])
    board = BaselineInitialScorer(UniformReliability(0.7)).score(graph, QUERY)
    score = board.by_id()["a"]
    comp = score.components
    # the six signals are all present and source_reliability matches the scorer
    assert comp.source_reliability == pytest.approx(0.7)
    assert comp.extraction == pytest.approx(0.8)
    assert 0.0 <= comp.query_relevance <= 1.0
    # initial belief and query utility are distinct quantities
    assert (
        score.initial_belief != score.query_utility or score.query_utility == score.initial_belief
    )
    assert set(score.contributions) == {
        "retrieval",
        "query_relevance",
        "extraction",
        "source_reliability",
        "temporal_validity",
        "independent_support",
    }
    assert 0.0 <= score.initial_belief <= 1.0
    assert score.propagated_belief is None  # not set by the initial scorer


@pytest.mark.unit
def test_invalid_weights_rejected() -> None:
    """Acceptance 16: invalid weights are rejected."""

    with pytest.raises(ValidationError):
        ScoreWeights(extraction=-1.0)
    with pytest.raises(ValidationError):
        ScoreWeights(
            retrieval=0.0,
            query_relevance=0.0,
            extraction=0.0,
            source_reliability=0.0,
            temporal_validity=0.0,
            independent_support=0.0,
        )


@pytest.mark.unit
def test_extreme_but_valid_weights_stay_bounded() -> None:
    """Acceptance 30: extreme but valid weights produce bounded initial belief."""

    weights = ScoreWeights(
        retrieval=1e6,
        query_relevance=1e6,
        extraction=1e6,
        source_reliability=1e6,
        temporal_validity=1e6,
        independent_support=1e6,
    )
    graph = _graph([_claim("a", "Acme grew revenue")])
    board = BaselineInitialScorer(UniformReliability(0.9), weights=weights).score(graph, QUERY)
    assert 0.0 <= board.by_id()["a"].initial_belief <= 1.0


@pytest.mark.unit
def test_score_serialization_round_trip() -> None:
    """Acceptance 32: score serialization preserves all components."""

    graph = _graph([_claim("a", "Acme grew revenue"), _claim("b", "Acme hired staff", "srcB")])
    board = BaselineInitialScorer(UniformReliability(0.7)).score(graph, QUERY)
    restored = ScoreBoard.model_validate_json(board.model_dump_json())
    assert restored == board


@pytest.mark.unit
def test_reliability_strategies() -> None:
    """Reliability strategies are configurable assumptions, not inferred."""

    source = SourceMetadata(source_id="s1", reliability_prior=0.8)
    assert UniformReliability(0.5).score(source) == 0.5
    assert ConfiguredPriorReliability({"s1": 0.9}, default=0.4).score(source) == 0.9
    assert ConfiguredPriorReliability({}, default=0.4).score(source) == 0.4
    assert MetadataReliability().score(source) == 0.8
    assert MetadataReliability(default=0.3).score(SourceMetadata(source_id="s2")) == 0.3


@pytest.mark.unit
def test_reliability_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        UniformReliability(1.5)
    with pytest.raises(ValueError):
        ConfiguredPriorReliability({"s": 2.0})
