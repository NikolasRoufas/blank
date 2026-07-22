"""Classification & directionality acceptance tests (cases 1-5, 25)."""

from __future__ import annotations

import pytest

from egrag.domain.models import (
    AtomicClaim,
    ClaimProvenance,
    ClaimSemantics,
    RelationDirection,
    RelationType,
    SourceMetadata,
    SourceSpan,
)
from egrag.fakes import FakePairClassifier
from egrag.graph import CandidateConfig, GraphBuilder, LexicalPairClassifier
from egrag.graph.candidates import generate_candidates
from egrag.graph.types import CandidateStrategy, RelationProbabilities

BRUTE = CandidateConfig(strategy=CandidateStrategy.BRUTE_FORCE)


def _claim(claim_id: str, text: str, source_id: str = "src") -> AtomicClaim:
    return AtomicClaim(
        claim_id=claim_id,
        text=text,
        provenance=ClaimProvenance(
            source=SourceMetadata(source_id=source_id),
            spans=(SourceSpan(source_id=source_id, start=0, end=len(text), text=text),),
        ),
        extraction_confidence=0.9,
        semantics=ClaimSemantics(named_entities=("Acme",)),
    )


A = _claim("a", "Acme grew revenue", "srcA")
B = _claim("b", "Acme increased revenue", "srcB")


@pytest.mark.unit
def test_support_direction_is_preserved() -> None:
    """Acceptance 1 & 2: directional entailment yields a one-way SUPPORT edge."""

    probs = {
        ("a", "b"): RelationProbabilities(entailment=0.9, contradiction=0.0, neutral=0.1),
        ("b", "a"): RelationProbabilities(entailment=0.2, contradiction=0.0, neutral=0.8),
    }
    result = GraphBuilder(FakePairClassifier(probs), candidate_config=BRUTE).build([A, B])
    support = [e for e in result.graph.edges() if e.relation_type is RelationType.SUPPORT]
    assert len(support) == 1
    edge = support[0]
    assert (edge.source_claim_id, edge.target_claim_id) == ("a", "b")
    assert edge.direction is RelationDirection.DIRECTED
    assert [c.claim_id for c in result.graph.supporting_evidence("b")] == ["a"]
    assert result.graph.supporting_evidence("a") == ()


@pytest.mark.unit
def test_contradiction_follows_symmetric_policy() -> None:
    """Acceptance 3: contradiction is stored as one canonical symmetric edge."""

    c = _claim("c", "The deal closed", "srcA")
    d = _claim("d", "The deal did not close", "srcB")
    probs = {
        ("c", "d"): RelationProbabilities(entailment=0.0, contradiction=0.85, neutral=0.15),
        ("d", "c"): RelationProbabilities(entailment=0.0, contradiction=0.80, neutral=0.20),
    }
    result = GraphBuilder(FakePairClassifier(probs), candidate_config=BRUTE).build([c, d])
    contradictions = [
        e for e in result.graph.edges() if e.relation_type is RelationType.CONTRADICTION
    ]
    assert len(contradictions) == 1  # single canonical edge, not two
    edge = contradictions[0]
    assert (edge.source_claim_id, edge.target_claim_id) == ("c", "d")  # canonical order
    assert edge.direction is RelationDirection.SYMMETRIC
    assert edge.relation_confidence == 0.85  # max over both directions
    # treated bidirectionally by the API
    assert [x.claim_id for x in result.graph.contradicting_evidence("c")] == ["d"]
    assert [x.claim_id for x in result.graph.contradicting_evidence("d")] == ["c"]


@pytest.mark.unit
def test_neutral_creates_no_edge() -> None:
    """Acceptance 4: neutral classifications create no stored edge."""

    result = GraphBuilder(FakePairClassifier({}), candidate_config=BRUTE).build([A, B])
    assert result.graph.edges() == ()


@pytest.mark.unit
def test_below_threshold_creates_no_edge() -> None:
    """Acceptance 5: confidence below threshold creates no edge."""

    probs = {
        ("a", "b"): RelationProbabilities(entailment=0.4, contradiction=0.0, neutral=0.6),
        ("b", "a"): RelationProbabilities(entailment=0.4, contradiction=0.0, neutral=0.6),
    }
    result = GraphBuilder(FakePairClassifier(probs), candidate_config=BRUTE).build([A, B])
    assert result.graph.edges() == ()


@pytest.mark.unit
def test_batching_preserves_classification_order() -> None:
    """Acceptance 25: classifier returns one result per pair, in input order."""

    claims = [_claim(f"c{i}", f"claim number {i}") for i in range(6)]
    pairs = generate_candidates(claims, BRUTE).pairs
    results = LexicalPairClassifier().classify(pairs)
    assert len(results) == len(pairs)
    # Re-classifying yields identical aligned results (deterministic, ordered).
    assert results == LexicalPairClassifier().classify(pairs)


@pytest.mark.unit
def test_store_neutral_only_in_debug_mode() -> None:
    """Neutral edges appear only when explicitly requested (debug)."""

    probs = {
        ("a", "b"): RelationProbabilities(entailment=0.1, contradiction=0.1, neutral=0.8),
        ("b", "a"): RelationProbabilities(entailment=0.1, contradiction=0.1, neutral=0.8),
    }
    debug = GraphBuilder(
        FakePairClassifier(probs), candidate_config=BRUTE, store_neutral=True
    ).build([A, B])
    assert any(e.relation_type is RelationType.NEUTRAL for e in debug.graph.edges())
