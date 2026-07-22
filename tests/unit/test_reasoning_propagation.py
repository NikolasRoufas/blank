"""Belief-propagation acceptance tests (1-7, 12-20, 28, 30, 31)."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from egrag.domain.errors import ConvergenceError
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
from egrag.reasoning import (
    NoPropagationBaseline,
    PropagationConfig,
    ScoreBoard,
    ScoreComponents,
    SignedBeliefPropagator,
)
from egrag.reasoning.models import ClaimScore


def _claim(cid: str, source_id: str = "src", text: str | None = None) -> AtomicClaim:
    text = text or f"claim {cid}"
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


def _rel(rid: str, s: str, t: str, kind: RelationType, conf: float = 0.9) -> EvidenceRelation:
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
        relation_confidence=conf,
        direction=direction,
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


def _graph(claims: list[AtomicClaim], relations: list[EvidenceRelation]) -> EvidenceGraph:
    return EvidenceGraph(
        EvidenceGraphSnapshot(snapshot_id="g", claims=tuple(claims), relations=tuple(relations))
    )


@pytest.mark.unit
def test_support_chain_increases_belief() -> None:
    """Acceptance 1: a support chain increases belief."""

    g = _graph([_claim("a"), _claim("b", "srcB")], [_rel("r", "a", "b", RelationType.SUPPORT)])
    board = ScoreBoard(scores=(_score("a", 0.8), _score("b", 0.5)))
    result = SignedBeliefPropagator().propagate(g, board)
    assert result.beliefs["b"] > 0.5
    assert result.converged


@pytest.mark.unit
def test_contradiction_decreases_without_forcing_zero() -> None:
    """Acceptance 2: contradiction decreases belief but not to zero."""

    g = _graph(
        [_claim("a"), _claim("b", "srcB")],
        [_rel("r", "a", "b", RelationType.CONTRADICTION, 0.9)],
    )
    board = ScoreBoard(scores=(_score("a", 0.6), _score("b", 0.6)))
    result = SignedBeliefPropagator().propagate(g, board)
    assert result.beliefs["a"] < 0.6
    assert result.beliefs["a"] > 0.0
    assert result.beliefs["b"] > 0.0


@pytest.mark.unit
def test_two_node_cycle_converges_with_damping() -> None:
    """Acceptance 3: a symmetric two-node support cycle converges."""

    g = _graph(
        [_claim("a"), _claim("b", "srcB")],
        [_rel("r1", "a", "b", RelationType.SUPPORT), _rel("r2", "b", "a", RelationType.SUPPORT)],
    )
    board = ScoreBoard(scores=(_score("a", 0.7), _score("b", 0.6)))
    result = SignedBeliefPropagator().propagate(g, board)
    assert result.converged


@pytest.mark.unit
def test_longer_cycle_converges_or_raises_typed_error() -> None:
    """Acceptance 4 & 31: a cyclic graph converges, or raises a typed error."""

    claims = [_claim("a"), _claim("b", "srcB"), _claim("c", "srcC")]
    relations = [
        _rel("r1", "a", "b", RelationType.SUPPORT),
        _rel("r2", "b", "c", RelationType.SUPPORT),
        _rel("r3", "c", "a", RelationType.SUPPORT),
    ]
    g = _graph(claims, relations)
    board = ScoreBoard(scores=(_score("a", 0.7), _score("b", 0.6), _score("c", 0.55)))
    assert SignedBeliefPropagator().propagate(g, board).converged

    strict = SignedBeliefPropagator(
        PropagationConfig(max_iterations=1, tolerance=1e-9, damping=0.0, on_nonconvergence="raise")
    )
    with pytest.raises(ConvergenceError):
        strict.propagate(g, board)

    lenient = SignedBeliefPropagator(
        PropagationConfig(max_iterations=1, tolerance=1e-9, damping=0.0, on_nonconvergence="return")
    )
    result = lenient.propagate(g, board)
    assert result.converged is False  # explainable result


@pytest.mark.unit
def test_duplicate_same_source_does_not_inflate() -> None:
    """Acceptance 5: duplicate claims from one source do not inflate belief."""

    def belief_for(supporters: list[tuple[str, str]], dup: bool) -> float:
        claims = [_claim("t", "srcT")]
        relations = []
        for i, (sid, src) in enumerate(supporters):
            claims.append(_claim(sid, src))
            relations.append(_rel(f"r{i}", sid, "t", RelationType.SUPPORT))
        if dup and len(supporters) == 2:
            relations.append(
                _rel("rd", supporters[0][0], supporters[1][0], RelationType.DUPLICATE, 0.99)
            )
        g = _graph(claims, relations)
        scores = [_score("t", 0.5)] + [_score(sid, 0.85) for sid, _ in supporters]
        return SignedBeliefPropagator().propagate(g, ScoreBoard(scores=tuple(scores))).beliefs["t"]

    single = belief_for([("s1", "srcA")], dup=False)
    dup_same = belief_for([("s1", "srcA"), ("s2", "srcA")], dup=True)
    assert abs(dup_same - single) < 0.02  # second duplicate adds no independent support


@pytest.mark.unit
def test_independent_sources_increase_support() -> None:
    """Acceptance 6: independent sources can increase support."""

    def belief_for(sources: list[str]) -> float:
        claims = [_claim("t", "srcT")]
        relations = []
        for i, src in enumerate(sources):
            claims.append(_claim(f"s{i}", src))
            relations.append(_rel(f"r{i}", f"s{i}", "t", RelationType.SUPPORT))
        g = _graph(claims, relations)
        scores = [_score("t", 0.5)] + [_score(f"s{i}", 0.85) for i in range(len(sources))]
        return SignedBeliefPropagator().propagate(g, ScoreBoard(scores=tuple(scores))).beliefs["t"]

    assert belief_for(["srcA", "srcB"]) > belief_for(["srcA"])


@pytest.mark.unit
def test_source_copying_receives_dependency_discount() -> None:
    """Acceptance 7: copied sources receive the configured lineage discount."""

    def belief_for(sources: list[str]) -> float:
        claims = [_claim("t", "srcT")]
        relations = []
        for i, src in enumerate(sources):
            claims.append(_claim(f"s{i}", src))
            relations.append(_rel(f"r{i}", f"s{i}", "t", RelationType.SUPPORT))
        g = _graph(claims, relations)
        scores = [_score("t", 0.5)] + [_score(f"s{i}", 0.85) for i in range(len(sources))]
        return SignedBeliefPropagator().propagate(g, ScoreBoard(scores=tuple(scores))).beliefs["t"]

    copied = belief_for(["srcA", "srcA"])  # two supporters, same source
    independent = belief_for(["srcA", "srcB"])
    single = belief_for(["srcA"])
    assert copied < independent  # dependency discount applied
    assert copied >= single  # but the discounted copy still adds a little


@pytest.mark.unit
def test_no_propagation_preserves_initial() -> None:
    """Acceptance 12 & 28: the no-propagation ablation preserves initial beliefs."""

    g = _graph([_claim("a"), _claim("b", "srcB")], [_rel("r", "a", "b", RelationType.SUPPORT)])
    board = ScoreBoard(scores=(_score("a", 0.8), _score("b", 0.5)))
    result = NoPropagationBaseline().propagate(g, board)
    assert result.beliefs == {"a": 0.8, "b": 0.5}


@pytest.mark.unit
def test_beliefs_finite_and_in_range_with_extreme_weights() -> None:
    """Acceptance 13, 14 & 30: beliefs stay finite and in [0, 1] under extreme weights."""

    claims = [_claim("t", "srcT")] + [_claim(f"s{i}", f"src{i}") for i in range(8)]
    relations = [_rel(f"r{i}", f"s{i}", "t", RelationType.SUPPORT) for i in range(8)]
    g = _graph(claims, relations)
    scores = [_score("t", 0.5)] + [_score(f"s{i}", 0.99) for i in range(8)]
    cfg = PropagationConfig(support_weight=1000.0, contradiction_weight=1000.0)
    result = SignedBeliefPropagator(cfg).propagate(g, ScoreBoard(scores=tuple(scores)))
    for value in result.beliefs.values():
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0


@pytest.mark.unit
def test_nan_input_rejected() -> None:
    """Acceptance 15: NaN input is rejected at the model boundary."""

    with pytest.raises(ValidationError):
        _score("a", float("nan"))


@pytest.mark.unit
def test_invalid_config_rejected() -> None:
    """Acceptance 16: invalid weights/config are rejected."""

    with pytest.raises(ValidationError):
        PropagationConfig(support_weight=-1.0)
    with pytest.raises(ValidationError):
        PropagationConfig(damping=1.0)  # must be < 1
    with pytest.raises(ValidationError):
        PropagationConfig(tolerance=0.0)  # must be > 0


@pytest.mark.unit
def test_deterministic_runs_identical() -> None:
    """Acceptance 17: deterministic runs produce identical results."""

    claims = [_claim("a"), _claim("b", "srcB"), _claim("c", "srcC")]
    relations = [
        _rel("r1", "a", "b", RelationType.SUPPORT),
        _rel("r2", "c", "b", RelationType.SUPPORT),
    ]
    g = _graph(claims, relations)
    board = ScoreBoard(scores=(_score("a", 0.7), _score("b", 0.5), _score("c", 0.6)))
    first = SignedBeliefPropagator().propagate(g, board)
    second = SignedBeliefPropagator().propagate(g, board)
    assert first.beliefs == second.beliefs
    assert first.iterations == second.iterations


@pytest.mark.unit
def test_empty_and_single_node_graphs() -> None:
    """Acceptance 18 & 19: empty and single-node graphs are handled."""

    empty = SignedBeliefPropagator().propagate(_graph([], []), ScoreBoard())
    assert empty.beliefs == {}
    assert empty.converged

    single = SignedBeliefPropagator().propagate(
        _graph([_claim("a")], []), ScoreBoard(scores=(_score("a", 0.7),))
    )
    assert single.beliefs == {"a": 0.7}
    assert single.converged


@pytest.mark.unit
def test_disconnected_graph_components_propagate_independently() -> None:
    """Acceptance 20: disconnected components are handled (each evolves alone)."""

    claims = [_claim("a"), _claim("b", "srcB"), _claim("x"), _claim("y", "srcY")]
    relations = [
        _rel("r1", "a", "b", RelationType.SUPPORT),  # component 1
        _rel("r2", "x", "y", RelationType.SUPPORT),  # component 2
    ]
    g = _graph(claims, relations)
    board = ScoreBoard(
        scores=(_score("a", 0.8), _score("b", 0.5), _score("x", 0.2), _score("y", 0.5))
    )
    result = SignedBeliefPropagator().propagate(g, board)
    assert result.converged
    assert result.beliefs["b"] > 0.5  # supported by high a
    assert result.beliefs["y"] < 0.5  # supported by low x
