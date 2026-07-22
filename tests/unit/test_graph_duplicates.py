"""Duplicate-handling acceptance tests (cases 6-10)."""

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
from egrag.graph import CandidateConfig, GraphBuilder
from egrag.graph.duplicates import detect_lexical_duplicates
from egrag.graph.types import CandidateStrategy, RelationProbabilities


def _claim(claim_id: str, text: str, source_id: str = "src") -> AtomicClaim:
    return AtomicClaim(
        claim_id=claim_id,
        text=text,
        provenance=ClaimProvenance(
            source=SourceMetadata(source_id=source_id),
            spans=(SourceSpan(source_id=source_id, start=0, end=len(text), text=text),),
        ),
        extraction_confidence=0.9,
        semantics=ClaimSemantics(),
    )


@pytest.mark.unit
def test_exact_duplicates_detected() -> None:
    """Acceptance 6: exact duplicates are detected."""

    pairs = detect_lexical_duplicates([_claim("a", "Same text."), _claim("b", "Same text.")])
    assert len(pairs) == 1
    assert pairs[0].kind == "exact"


@pytest.mark.unit
def test_normalized_duplicates_detected() -> None:
    """Acceptance 7: normalized duplicates are detected."""

    pairs = detect_lexical_duplicates(
        [_claim("a", "Same   text here."), _claim("b", "same text here.")]
    )
    assert len(pairs) == 1
    assert pairs[0].kind == "normalized"


@pytest.mark.unit
def test_duplicate_nodes_preserve_independent_provenance() -> None:
    """Acceptance 8: duplicate claims keep both nodes and their provenance."""

    a = _claim("a", "Same text.", "srcA")
    b = _claim("b", "Same text.", "srcB")
    result = GraphBuilder(FakePairClassifier({})).build([a, b])
    # both nodes survive (not merged/erased)
    assert {n.claim_id for n in result.graph.nodes()} == {"a", "b"}
    dup_edges = [e for e in result.graph.edges() if e.relation_type is RelationType.DUPLICATE]
    assert len(dup_edges) == 1
    assert dup_edges[0].direction is RelationDirection.SYMMETRIC
    # provenance of each node is intact
    assert result.graph.node("a").provenance.source.source_id == "srcA"
    assert result.graph.node("b").provenance.source.source_id == "srcB"


@pytest.mark.unit
def test_same_source_paraphrases_not_independent_corroboration() -> None:
    """Acceptance 9: same-source duplicates do not inflate corroboration."""

    # Two same-source claims both support a third claim from a different source.
    s1 = _claim("s1", "Acme raised one billion dollars", "srcA")
    s2 = _claim("s2", "Acme raised 1 billion dollars", "srcA")  # same source paraphrase
    target = _claim("t", "Acme secured major funding", "srcT")
    probs = {
        ("s1", "t"): RelationProbabilities(entailment=0.9, contradiction=0.0, neutral=0.1),
        ("s2", "t"): RelationProbabilities(entailment=0.9, contradiction=0.0, neutral=0.1),
    }
    result = GraphBuilder(
        FakePairClassifier(probs),
        candidate_config=CandidateConfig(strategy=CandidateStrategy.BRUTE_FORCE),
    ).build([s1, s2, target])
    # Two supporters but only ONE distinct corroborating source.
    assert len(result.graph.supporting_evidence("t")) == 2
    assert result.graph.corroborating_sources("t") == ("srcA",)


@pytest.mark.unit
def test_cross_source_duplicates_preserve_separate_sources() -> None:
    """Acceptance 10: cross-source duplicates keep separate sources."""

    a = _claim("a", "Same text.", "srcA")
    b = _claim("b", "Same text.", "srcB")
    result = GraphBuilder(FakePairClassifier({})).build([a, b])
    sources = {n.provenance.source.source_id for n in result.graph.nodes()}
    assert sources == {"srcA", "srcB"}
