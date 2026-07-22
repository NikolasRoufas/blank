"""Graph API, validation, serialization, and metrics tests (15-20, 28)."""

from __future__ import annotations

import pytest

from egrag.domain.errors import GraphValidationError, SerializationError
from egrag.domain.models import (
    AtomicClaim,
    ClaimProvenance,
    ClaimSemantics,
    EvidenceRelation,
    RelationType,
    SourceMetadata,
    SourceSpan,
)
from egrag.fakes import FakePairClassifier
from egrag.graph import (
    CandidateConfig,
    GraphBuilder,
    GraphSerializer,
    validate_components,
)
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
PROBS = {
    ("a", "b"): RelationProbabilities(entailment=0.9, contradiction=0.0, neutral=0.1),
    ("b", "a"): RelationProbabilities(entailment=0.2, contradiction=0.0, neutral=0.8),
}


def _built() -> object:
    return GraphBuilder(FakePairClassifier(PROBS), candidate_config=BRUTE).build([A, B])


@pytest.mark.unit
def test_every_node_has_a_source_span() -> None:
    """Acceptance 15: every node retains at least one valid source span."""

    graph = _built().graph  # type: ignore[attr-defined]
    for node in graph.nodes():
        assert len(node.provenance.spans) >= 1


@pytest.mark.unit
def test_every_inferred_edge_has_classifier_provenance() -> None:
    """Acceptance 16: every inferred edge retains classifier provenance."""

    graph = _built().graph  # type: ignore[attr-defined]
    assert graph.edges()  # at least one edge
    for edge in graph.edges():
        assert edge.metadata is not None
        assert edge.metadata.classifier_id
        assert edge.metadata.classifier_version


@pytest.mark.unit
def test_serialization_round_trip_preserves_everything() -> None:
    """Acceptance 17: serialization preserves every node, edge, score, provenance."""

    snapshot = _built().graph.snapshot()  # type: ignore[attr-defined]
    serializer = GraphSerializer()
    restored = serializer.deserialize(serializer.serialize(snapshot))
    assert restored == snapshot


@pytest.mark.unit
@pytest.mark.parametrize("payload", ["", "{ not json", "{}", "[1,2,3]"])
def test_malformed_graph_data_rejected_safely(payload: str) -> None:
    with pytest.raises(SerializationError):
        GraphSerializer().deserialize(payload)


@pytest.mark.unit
def test_dangling_edges_rejected() -> None:
    """Acceptance 18: dangling edges are rejected."""

    rel = EvidenceRelation(
        relation_id="r1",
        source_claim_id="a",
        target_claim_id="missing",
        relation_type=RelationType.SUPPORT,
        relation_confidence=0.9,
    )
    with pytest.raises(GraphValidationError):
        validate_components([A], [rel])


@pytest.mark.unit
def test_duplicate_node_ids_rejected() -> None:
    """Acceptance 19: duplicate node IDs are rejected."""

    with pytest.raises(GraphValidationError):
        validate_components([A, _claim("a", "different text")], [])


@pytest.mark.unit
def test_invalid_self_edges_rejected() -> None:
    """Acceptance 20: invalid self-edges are rejected."""

    with pytest.raises(ValueError):  # EvidenceRelation forbids self-edges at construction
        EvidenceRelation(
            relation_id="r1",
            source_claim_id="a",
            target_claim_id="a",
            relation_type=RelationType.SUPPORT,
            relation_confidence=0.9,
        )


@pytest.mark.unit
def test_neutral_edge_rejected_unless_allowed() -> None:
    neutral = EvidenceRelation(
        relation_id="r1",
        source_claim_id="a",
        target_claim_id="b",
        relation_type=RelationType.NEUTRAL,
        relation_confidence=0.5,
    )
    with pytest.raises(GraphValidationError):
        validate_components([A, B], [neutral])
    validate_components([A, B], [neutral], allow_neutral=True)  # allowed in debug


@pytest.mark.unit
def test_metrics_match_graph_contents() -> None:
    """Acceptance 28: graph metrics match actual graph contents."""

    result = _built()
    graph = result.graph  # type: ignore[attr-defined]
    metrics = result.metrics  # type: ignore[attr-defined]
    assert metrics.num_claims == len(graph.nodes())
    actual_by_type: dict[str, int] = {}
    for edge in graph.edges():
        actual_by_type[edge.relation_type.value] = (
            actual_by_type.get(edge.relation_type.value, 0) + 1
        )
    assert metrics.edges_by_type == actual_by_type
    # two claims -> one possible unordered pair, classified in both directions
    assert metrics.possible_pairs == 1
    assert metrics.classified_pairs == 2
