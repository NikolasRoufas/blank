"""Candidate-generation acceptance tests (cases 21-24, 26, 27)."""

from __future__ import annotations

import pytest

from egrag.domain.models import (
    AtomicClaim,
    ClaimProvenance,
    ClaimSemantics,
    RelationType,
    SourceMetadata,
    SourceSpan,
)
from egrag.fakes import FakePairClassifier
from egrag.graph import CandidateConfig, GraphBuilder
from egrag.graph.candidates import generate_candidates
from egrag.graph.types import CandidateStrategy, RelationProbabilities

BRUTE = CandidateConfig(strategy=CandidateStrategy.BRUTE_FORCE)
PRUNED = CandidateConfig(strategy=CandidateStrategy.PRUNED)


def _claim(claim_id: str, text: str, entity: str, source_id: str = "src") -> AtomicClaim:
    return AtomicClaim(
        claim_id=claim_id,
        text=text,
        provenance=ClaimProvenance(
            source=SourceMetadata(source_id=source_id),
            spans=(SourceSpan(source_id=source_id, start=0, end=len(text), text=text),),
        ),
        extraction_confidence=0.9,
        semantics=ClaimSemantics(named_entities=(entity,)),
    )


# a and b share entity "Acme"; c is unrelated ("Zeta").
A = _claim("a", "Acme grew fast", "Acme")
B = _claim("b", "Acme expanded operations", "Acme")
C = _claim("c", "Zeta declined sharply", "Zeta")


@pytest.mark.unit
def test_candidate_ordering_is_deterministic() -> None:
    """Acceptance 21: candidate-pair ordering is deterministic."""

    first = generate_candidates([C, A, B], BRUTE).pairs
    second = generate_candidates([A, B, C], BRUTE).pairs
    keys = [(p.source.claim_id, p.target.claim_id) for p in first]
    assert keys == [(p.source.claim_id, p.target.claim_id) for p in second]
    assert keys == sorted(keys)  # sorted by (source, target)


@pytest.mark.unit
def test_pair_budget_enforced_deterministically() -> None:
    """Acceptance 22: pair budgets are enforced deterministically."""

    config = CandidateConfig(strategy=CandidateStrategy.BRUTE_FORCE, max_pairs=1)
    result = generate_candidates([A, B, C], config)
    assert result.stats.generated_pairs == 1
    assert result.stats.budget_truncated == 2  # 3 possible - 1 kept
    # deterministic across runs
    again = generate_candidates([A, B, C], config)
    assert [(p.source.claim_id, p.target.claim_id) for p in result.pairs] == [
        (p.source.claim_id, p.target.claim_id) for p in again.pairs
    ]


@pytest.mark.unit
def test_pruning_keeps_known_relevant_pair() -> None:
    """Acceptance 24: pruning does not omit a known relevant pair."""

    pairs = generate_candidates([A, B, C], PRUNED).pairs
    unordered = {tuple(sorted((p.source.claim_id, p.target.claim_id))) for p in pairs}
    assert ("a", "b") in unordered  # shared entity -> kept
    assert ("a", "c") not in unordered  # unrelated -> pruned


@pytest.mark.unit
def test_brute_and_pruned_produce_equivalent_required_edges() -> None:
    """Acceptance 23: brute-force and pruned yield the same required edges."""

    probs = {
        ("a", "b"): RelationProbabilities(entailment=0.7, contradiction=0.0, neutral=0.3),
        ("b", "a"): RelationProbabilities(entailment=0.2, contradiction=0.0, neutral=0.8),
    }

    def edges_for(config: CandidateConfig) -> set[tuple[str, str, str]]:
        result = GraphBuilder(FakePairClassifier(probs), candidate_config=config).build([A, B, C])
        return {
            (e.source_claim_id, e.target_claim_id, e.relation_type.value)
            for e in result.graph.edges()
        }

    brute_edges = edges_for(BRUTE)
    pruned_edges = edges_for(PRUNED)
    assert brute_edges == pruned_edges
    assert ("a", "b", RelationType.SUPPORT.value) in pruned_edges


@pytest.mark.unit
def test_empty_claim_set_creates_valid_empty_graph() -> None:
    """Acceptance 26: empty claim sets create a valid empty graph."""

    result = GraphBuilder(FakePairClassifier({})).build([])
    result.graph.validate()
    assert result.graph.nodes() == ()
    assert result.graph.edges() == ()
    assert result.metrics.num_claims == 0
    assert result.metrics.possible_pairs == 0


@pytest.mark.unit
def test_single_claim_graph_is_valid() -> None:
    """Acceptance 27: single-claim graphs are valid."""

    result = GraphBuilder(FakePairClassifier({})).build([A])
    result.graph.validate()
    assert len(result.graph.nodes()) == 1
    assert result.graph.edges() == ()
    assert result.graph.connected_components() == [("a",)]
