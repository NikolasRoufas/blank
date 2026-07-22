"""Temporal supersession acceptance tests (cases 11-14)."""

from __future__ import annotations

from datetime import UTC, datetime

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
from egrag.graph import GraphBuilder


def _claim(
    claim_id: str,
    text: str,
    *,
    subject: str | None = None,
    predicate: str | None = None,
    obj: str | None = None,
    asserted: datetime | None = None,
    source_id: str = "src",
) -> AtomicClaim:
    return AtomicClaim(
        claim_id=claim_id,
        text=text,
        provenance=ClaimProvenance(
            source=SourceMetadata(source_id=source_id),
            spans=(SourceSpan(source_id=source_id, start=0, end=len(text), text=text),),
        ),
        extraction_confidence=0.9,
        semantics=ClaimSemantics(subject=subject, predicate=predicate, object=obj),
        asserted_at=asserted,
    )


def _supersessions(result: object) -> list:
    graph = result.graph  # type: ignore[attr-defined]
    return [e for e in graph.edges() if e.relation_type is RelationType.SUPERSESSION]


JAN = datetime(2024, 1, 1, tzinfo=UTC)
FEB = datetime(2024, 2, 1, tzinfo=UTC)


@pytest.mark.unit
def test_same_proposition_update_creates_supersedes() -> None:
    """Acceptance 11: newer evidence on the same proposition creates SUPERSEDES."""

    old = _claim(
        "old",
        "deadline is July 10",
        subject="deadline",
        predicate="is",
        obj="July 10",
        asserted=JAN,
    )
    new = _claim(
        "new",
        "deadline is July 20",
        subject="deadline",
        predicate="is",
        obj="July 20",
        asserted=FEB,
    )
    edges = _supersessions(GraphBuilder(FakePairClassifier({})).build([old, new]))
    assert len(edges) == 1
    assert (edges[0].source_claim_id, edges[0].target_claim_id) == ("new", "old")


@pytest.mark.unit
def test_unrelated_newer_claim_does_not_supersede() -> None:
    """Acceptance 12: a newer unrelated claim does not supersede an older claim."""

    old = _claim(
        "old",
        "deadline is July 10",
        subject="deadline",
        predicate="is",
        obj="July 10",
        asserted=JAN,
    )
    other = _claim(
        "other",
        "budget is one million",
        subject="budget",
        predicate="is",
        obj="one million",
        asserted=FEB,
    )
    assert _supersessions(GraphBuilder(FakePairClassifier({})).build([old, other])) == []


@pytest.mark.unit
def test_unknown_timestamps_create_no_supersession() -> None:
    """Acceptance 13: unknown timestamps do not create unsupported supersession."""

    old = _claim(
        "old",
        "deadline is July 10",
        subject="deadline",
        predicate="is",
        obj="July 10",
        asserted=JAN,
    )
    new = _claim(
        "new",
        "deadline is July 20",
        subject="deadline",
        predicate="is",
        obj="July 20",
        asserted=None,
    )
    assert _supersessions(GraphBuilder(FakePairClassifier({})).build([old, new])) == []


@pytest.mark.unit
def test_low_confidence_update_distinguishable_from_confirmed() -> None:
    """Acceptance 14: a low-confidence update is not stored as SUPERSEDES."""

    # Low confidence: same subject but no predicate -> update confidence 0.5 < 0.6.
    old_low = _claim("old", "deadline July 10", subject="deadline", obj="July 10", asserted=JAN)
    new_low = _claim("new", "deadline July 20", subject="deadline", obj="July 20", asserted=FEB)
    assert _supersessions(GraphBuilder(FakePairClassifier({})).build([old_low, new_low])) == []

    # Confirmed: subject AND predicate match -> confidence 0.9 >= 0.6.
    old_hi = _claim(
        "old",
        "deadline is July 10",
        subject="deadline",
        predicate="is",
        obj="July 10",
        asserted=JAN,
    )
    new_hi = _claim(
        "new",
        "deadline is July 20",
        subject="deadline",
        predicate="is",
        obj="July 20",
        asserted=FEB,
    )
    assert len(_supersessions(GraphBuilder(FakePairClassifier({})).build([old_hi, new_hi]))) == 1
