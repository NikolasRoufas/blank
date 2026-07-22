"""Conflict-resolution acceptance tests (8-11)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from egrag.domain.models import (
    AtomicClaim,
    ClaimProvenance,
    ClaimSemantics,
    ConflictOutcome,
    EvidenceGraphSnapshot,
    EvidenceRelation,
    RelationDirection,
    RelationType,
    SourceMetadata,
    SourceSpan,
)
from egrag.graph.api import EvidenceGraph
from egrag.reasoning import ConflictSetResolver, ScoreBoard, ScoreComponents
from egrag.reasoning.models import ClaimScore

JAN = datetime(2024, 1, 1, tzinfo=UTC)
DEC = datetime(2024, 12, 1, tzinfo=UTC)


def _claim(cid: str, source_id: str, reliability: float, when: datetime) -> AtomicClaim:
    text = f"claim {cid}"
    return AtomicClaim(
        claim_id=cid,
        text=text,
        provenance=ClaimProvenance(
            source=SourceMetadata(source_id=source_id, reliability_prior=reliability),
            spans=(SourceSpan(source_id=source_id, start=0, end=len(text), text=text),),
        ),
        extraction_confidence=0.8,
        semantics=ClaimSemantics(),
        asserted_at=when,
    )


def _score(cid: str, *, reliability: float, belief: float, support: int) -> ClaimScore:
    return ClaimScore(
        claim_id=cid,
        components=ScoreComponents(
            retrieval=0.5,
            query_relevance=0.5,
            extraction=0.8,
            source_reliability=reliability,
            temporal_validity=1.0,
            independent_support=support / (support + 1),
        ),
        initial_belief=belief,
        propagated_belief=belief,
        query_utility=0.6,
        provenance_diversity=support,
    )


def _conflict(claims: list[AtomicClaim]) -> EvidenceGraph:
    rel = EvidenceRelation(
        relation_id="r",
        source_claim_id=claims[0].claim_id,
        target_claim_id=claims[1].claim_id,
        relation_type=RelationType.CONTRADICTION,
        relation_confidence=0.8,
        direction=RelationDirection.SYMMETRIC,
    )
    return EvidenceGraph(
        EvidenceGraphSnapshot(snapshot_id="g", claims=tuple(claims), relations=(rel,))
    )


@pytest.mark.unit
def test_newer_unreliable_does_not_defeat_older_corroborated() -> None:
    """Acceptance 8: newer but unreliable evidence does not defeat older corroborated."""

    older = _claim("old", "srcA", 0.9, JAN)
    newer = _claim("new", "srcB", 0.3, DEC)  # later timestamp, weaker evidence
    graph = _conflict([older, newer])
    board = ScoreBoard(
        scores=(
            _score("old", reliability=0.9, belief=0.85, support=2),
            _score("new", reliability=0.3, belief=0.30, support=0),
        )
    )
    conflicts = ConflictSetResolver().resolve(graph, board)
    assert len(conflicts) == 1
    assert conflicts[0].outcome is ConflictOutcome.PREFERRED
    assert conflicts[0].preferred_claim_id == "old"  # newer did NOT win on recency


@pytest.mark.unit
def test_older_corroborated_remains_preferred() -> None:
    """Acceptance 9: older independently corroborated evidence can remain preferred."""

    older = _claim("old", "srcA", 0.8, JAN)
    newer = _claim("new", "srcB", 0.8, DEC)
    graph = _conflict([older, newer])
    board = ScoreBoard(
        scores=(
            _score("old", reliability=0.8, belief=0.80, support=3),
            _score("new", reliability=0.8, belief=0.55, support=0),
        )
    )
    conflicts = ConflictSetResolver().resolve(graph, board)
    assert conflicts[0].preferred_claim_id == "old"
    assert conflicts[0].outcome is ConflictOutcome.PREFERRED


@pytest.mark.unit
def test_close_conflict_remains_unresolved() -> None:
    """Acceptance 10: a close conflict (within margin) remains unresolved."""

    a = _claim("a", "srcA", 0.7, JAN)
    b = _claim("b", "srcB", 0.7, DEC)
    graph = _conflict([a, b])
    board = ScoreBoard(
        scores=(
            _score("a", reliability=0.7, belief=0.56, support=1),
            _score("b", reliability=0.7, belief=0.54, support=1),
        )
    )
    conflicts = ConflictSetResolver(margin=0.1).resolve(graph, board)
    assert conflicts[0].outcome is ConflictOutcome.UNRESOLVED
    assert conflicts[0].resolved is False
    assert conflicts[0].preferred_claim_id is None


@pytest.mark.unit
def test_clear_winner_is_preferred() -> None:
    """Acceptance 11: a clear conflict winner can be preferred."""

    a = _claim("a", "srcA", 0.9, JAN)
    b = _claim("b", "srcB", 0.4, DEC)
    graph = _conflict([a, b])
    board = ScoreBoard(
        scores=(
            _score("a", reliability=0.9, belief=0.82, support=2),
            _score("b", reliability=0.4, belief=0.40, support=0),
        )
    )
    conflicts = ConflictSetResolver(margin=0.1).resolve(graph, board)
    assert conflicts[0].outcome is ConflictOutcome.PREFERRED
    assert conflicts[0].preferred_claim_id == "a"


@pytest.mark.unit
def test_conflict_members_preserve_signals() -> None:
    """Conflict members keep distinct signals (belief, reliability, support, time)."""

    a = _claim("a", "srcA", 0.9, JAN)
    b = _claim("b", "srcB", 0.4, DEC)
    board = ScoreBoard(
        scores=(
            _score("a", reliability=0.9, belief=0.82, support=2),
            _score("b", reliability=0.4, belief=0.40, support=0),
        )
    )
    conflict = ConflictSetResolver().resolve(_conflict([a, b]), board)[0]
    members = {m.claim_id: m for m in conflict.members}
    assert members["a"].source_reliability == pytest.approx(0.9)
    assert members["a"].independent_support == 2
    assert members["a"].timestamp == JAN
    assert members["b"].propagated_belief == pytest.approx(0.40)
