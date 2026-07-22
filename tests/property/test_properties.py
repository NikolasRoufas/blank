"""Property-based tests (Hypothesis) for invariants and round-trips."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from egrag.domain.models import (
    AtomicClaim,
    ClaimProvenance,
    EvidenceGraphSnapshot,
    Query,
    SourceMetadata,
    SourceSpan,
)
from egrag.fakes import FakeBeliefPropagator, FakeInitialClaimScorer

# Non-empty text that survives whitespace stripping.
text_strategy = st.text(min_size=1, max_size=80).map(str.strip).filter(lambda s: len(s) > 0)
prob_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


@pytest.mark.property
@given(start=st.integers(min_value=0, max_value=10_000), length=st.integers(1, 1_000))
def test_source_span_round_trip(start: int, length: int) -> None:
    span = SourceSpan(source_id="s1", start=start, end=start + length, text="content")
    restored = SourceSpan.model_validate_json(span.model_dump_json())
    assert restored == span


@pytest.mark.property
@given(claim_text=text_strategy, query_text=text_strategy, reliability=prob_strategy)
def test_initial_scores_are_bounded(claim_text: str, query_text: str, reliability: float) -> None:
    span = SourceSpan(source_id="s1", start=0, end=3, text="abc")
    claim = AtomicClaim(
        claim_id="c1",
        text=claim_text,
        provenance=ClaimProvenance(source=SourceMetadata(source_id="s1"), spans=(span,)),
        extraction_confidence=0.9,
        source_reliability=reliability,
    )
    scored = FakeInitialClaimScorer().score(claim, Query(query_id="q1", text=query_text))
    assert scored.belief is not None and 0.0 <= scored.belief <= 1.0
    assert scored.query_utility is not None and 0.0 <= scored.query_utility <= 1.0


@pytest.mark.property
@given(initial=prob_strategy)
def test_propagated_belief_stays_bounded(initial: float) -> None:
    span = SourceSpan(source_id="s1", start=0, end=3, text="abc")
    claim = AtomicClaim(
        claim_id="c1",
        text="a claim",
        provenance=ClaimProvenance(source=SourceMetadata(source_id="s1"), spans=(span,)),
        extraction_confidence=0.9,
        belief=initial,
    )
    snapshot = EvidenceGraphSnapshot(snapshot_id="g1", claims=(claim,))
    propagated = FakeBeliefPropagator().propagate(snapshot)
    value = propagated.claims[0].belief
    assert value is not None and 0.0 <= value <= 1.0
