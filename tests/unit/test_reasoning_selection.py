"""Subgraph-selection acceptance tests (21-27, 29)."""

from __future__ import annotations

import pytest

from egrag.domain.models import (
    AtomicClaim,
    ClaimProvenance,
    ClaimSemantics,
    EvidenceGraphSnapshot,
    EvidenceRelation,
    Query,
    RelationDirection,
    RelationType,
    SourceMetadata,
    SourceSpan,
)
from egrag.graph.api import EvidenceGraph
from egrag.reasoning import (
    BeamSearchSelector,
    ConflictSetResolver,
    GreedyConnectedSelector,
    ScoreBoard,
    ScoreComponents,
    TopClaimsSelector,
    WhitespaceTokenCounter,
    query_entity_coverage,
)
from egrag.reasoning.models import ClaimScore, SelectionConfig, TokenBudget
from egrag.reasoning.tokens import CharacterTokenCounter

QUERY = Query(query_id="q", text="acme revenue growth")


def _claim(cid: str, text: str, source_id: str) -> AtomicClaim:
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


def _rel(rid: str, s: str, t: str, kind: RelationType) -> EvidenceRelation:
    direction = (
        RelationDirection.SYMMETRIC
        if kind in (RelationType.CONTRADICTION, RelationType.DUPLICATE)
        else RelationDirection.DIRECTED
    )
    return EvidenceRelation(
        relation_id=rid,
        source_claim_id=s,
        target_claim_id=t,
        relation_type=kind,
        relation_confidence=0.8,
        direction=direction,
    )


def _score(cid: str, belief: float, util: float) -> ClaimScore:
    return ClaimScore(
        claim_id=cid,
        components=ScoreComponents(
            retrieval=0.5,
            query_relevance=util,
            extraction=0.8,
            source_reliability=0.5,
            temporal_validity=1.0,
            independent_support=0.0,
        ),
        initial_belief=belief,
        propagated_belief=belief,
        query_utility=util,
        provenance_diversity=0,
    )


@pytest.mark.unit
def test_selection_obeys_token_budget_and_reserved_output() -> None:
    """Acceptance 21 & 22: selection respects the budget and reserved output."""

    claims = [_claim(f"c{i}", f"acme revenue claim number {i}", f"src{i}") for i in range(5)]
    rels = [_rel(f"r{i}", "c0", f"c{i}", RelationType.SUPPORT) for i in range(1, 5)]
    graph = EvidenceGraph(
        EvidenceGraphSnapshot(snapshot_id="g", claims=tuple(claims), relations=tuple(rels))
    )
    board = ScoreBoard(scores=tuple(_score(f"c{i}", 0.7, 0.6) for i in range(5)))
    # Each claim is 5 whitespace tokens; total 30, reserved 10 -> available 20 -> at most 4 claims.
    budget = TokenBudget(total=30, reserved_output=10)
    result = GreedyConnectedSelector().select(
        graph, QUERY, board, token_budget=budget, token_counter=WhitespaceTokenCounter()
    )
    assert result.total_tokens <= budget.available
    assert result.total_tokens <= 20


@pytest.mark.unit
def test_greedy_selection_is_connected() -> None:
    """Acceptance 23: the greedy subgraph is connected."""

    claims = [
        _claim("a", "acme revenue rose", "s1"),
        _claim("b", "acme revenue growth strong", "s2"),
        _claim("c", "acme revenue up", "s3"),
    ]
    rels = [_rel("r1", "a", "b", RelationType.SUPPORT), _rel("r2", "b", "c", RelationType.SUPPORT)]
    graph = EvidenceGraph(
        EvidenceGraphSnapshot(snapshot_id="g", claims=tuple(claims), relations=tuple(rels))
    )
    board = ScoreBoard(scores=(_score("a", 0.8, 0.7), _score("b", 0.7, 0.6), _score("c", 0.6, 0.5)))
    result = GreedyConnectedSelector().select(
        graph,
        QUERY,
        board,
        token_budget=TokenBudget(total=100),
        token_counter=CharacterTokenCounter(),
    )
    assert result.connected
    selected = set(result.selected_claim_ids)
    # every selected claim (beyond the seed) is adjacent to another selected claim
    for cid in selected:
        neighbors = set(graph.neighbors(cid))
        assert len(selected) == 1 or neighbors & (selected - {cid})


@pytest.mark.unit
def test_unresolved_contradiction_evidence_is_retained() -> None:
    """Acceptance 24: contradictory evidence is retained to explain uncertainty."""

    a = _claim("a", "acme revenue grew", "s1")
    b = _claim("b", "acme revenue did not grow", "s2")
    graph = EvidenceGraph(
        EvidenceGraphSnapshot(
            snapshot_id="g",
            claims=(a, b),
            relations=(_rel("r", "a", "b", RelationType.CONTRADICTION),),
        )
    )
    board = ScoreBoard(scores=(_score("a", 0.55, 0.6), _score("b", 0.52, 0.6)))
    conflicts = ConflictSetResolver(margin=0.1).resolve(graph, board)
    result = GreedyConnectedSelector().select(
        graph,
        QUERY,
        board,
        token_budget=TokenBudget(total=200),
        token_counter=CharacterTokenCounter(),
        conflicts=conflicts,
    )
    assert {"a", "b"} <= set(result.selected_claim_ids)  # both sides retained


@pytest.mark.unit
def test_repeated_source_lineage_is_penalized() -> None:
    """Acceptance 25: repeated source lineage is penalized in selection."""

    seed = _claim("seed", "acme revenue overview here", "srcA")
    same = _claim("same", "acme revenue more detail", "srcA")  # same source as seed
    diff = _claim("diff", "acme revenue other detail", "srcB")  # different source
    rels = [
        _rel("r1", "same", "seed", RelationType.SUPPORT),
        _rel("r2", "diff", "seed", RelationType.SUPPORT),
    ]
    graph = EvidenceGraph(
        EvidenceGraphSnapshot(snapshot_id="g", claims=(seed, same, diff), relations=tuple(rels))
    )
    board = ScoreBoard(
        scores=(_score("seed", 0.8, 0.7), _score("same", 0.7, 0.6), _score("diff", 0.7, 0.6))
    )
    # Budget for seed + exactly one more (each claim is 4 whitespace tokens).
    budget = TokenBudget(total=8)
    result = GreedyConnectedSelector().select(
        graph, QUERY, board, token_budget=budget, token_counter=WhitespaceTokenCounter()
    )
    assert "diff" in result.selected_claim_ids  # different source preferred
    assert "same" not in result.selected_claim_ids  # same-source lineage penalized out


@pytest.mark.unit
def test_query_entity_coverage_measured() -> None:
    """Acceptance 26: query entity coverage is measured correctly."""

    # query terms = {acme, revenue, growth}; "acme revenue" covers 2/3.
    assert query_entity_coverage(["acme revenue rose"], QUERY) == pytest.approx(2 / 3)
    assert query_entity_coverage([], QUERY) == 0.0
    assert query_entity_coverage(["acme revenue growth"], QUERY) == pytest.approx(1.0)


@pytest.mark.unit
def test_top_claims_baseline_available() -> None:
    """Acceptance 27: a top-claims baseline is available for ablation."""

    claims = [_claim("a", "acme revenue rose", "s1"), _claim("b", "unrelated text here", "s2")]
    graph = EvidenceGraph(EvidenceGraphSnapshot(snapshot_id="g", claims=tuple(claims)))
    board = ScoreBoard(scores=(_score("a", 0.9, 0.8), _score("b", 0.2, 0.1)))
    result = TopClaimsSelector().select(
        graph,
        QUERY,
        board,
        token_budget=TokenBudget(total=100),
        token_counter=CharacterTokenCounter(),
    )
    assert "a" in result.selected_claim_ids
    assert result.selected_claim_ids[0] == "a"  # highest base value first


@pytest.mark.unit
def test_selector_output_contains_explanation_fields() -> None:
    """Acceptance 29: selector output contains explanation fields for every claim."""

    claims = [_claim("a", "acme revenue rose", "s1"), _claim("b", "acme revenue growth", "s2")]
    rels = [_rel("r", "a", "b", RelationType.SUPPORT)]
    graph = EvidenceGraph(
        EvidenceGraphSnapshot(snapshot_id="g", claims=tuple(claims), relations=tuple(rels))
    )
    board = ScoreBoard(scores=(_score("a", 0.8, 0.7), _score("b", 0.7, 0.6)))
    result = GreedyConnectedSelector().select(
        graph,
        QUERY,
        board,
        token_budget=TokenBudget(total=100),
        token_counter=CharacterTokenCounter(),
    )
    assert len(result.entries) == 2  # every claim explained, selected or not
    for entry in result.entries:
        assert entry.reason
        assert entry.source_id
        assert entry.tokens >= 0
        assert 0.0 <= entry.propagated_belief <= 1.0
        # explanation includes neighbors and contributions
        assert isinstance(entry.supporting_neighbors, tuple)
        assert isinstance(entry.final_selection_score, float)


@pytest.mark.unit
def test_beam_search_selector_runs_within_budget() -> None:
    claims = [_claim(f"c{i}", f"acme revenue point {i}", f"s{i}") for i in range(4)]
    rels = [_rel(f"r{i}", "c0", f"c{i}", RelationType.SUPPORT) for i in range(1, 4)]
    graph = EvidenceGraph(
        EvidenceGraphSnapshot(snapshot_id="g", claims=tuple(claims), relations=tuple(rels))
    )
    board = ScoreBoard(scores=tuple(_score(f"c{i}", 0.7, 0.6) for i in range(4)))
    result = BeamSearchSelector(SelectionConfig(beam_width=2)).select(
        graph,
        QUERY,
        board,
        token_budget=TokenBudget(total=100),
        token_counter=CharacterTokenCounter(),
    )
    assert result.selected_claim_ids
    assert result.total_tokens <= 100
