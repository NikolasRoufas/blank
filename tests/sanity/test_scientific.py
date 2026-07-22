"""Scientific-integrity invariant tests (the contract's non-negotiables)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from egrag.application.pipeline import EGRagPipeline
from egrag.domain.models import (
    AtomicClaim,
    ClaimProvenance,
    EvidenceGraphSnapshot,
    EvidenceRelation,
    Query,
    RelationType,
    SourceMetadata,
    SourceSpan,
)
from egrag.fakes import (
    FakeBeliefPropagator,
    FakeConflictResolver,
    FakeInitialClaimScorer,
    build_demo_components,
)


def _claim(
    claim_id: str,
    source_id: str,
    *,
    text: str = "shared content here",
    reliability: float = 0.8,
    asserted_at: datetime | None = None,
) -> AtomicClaim:
    span = SourceSpan(source_id=source_id, start=0, end=3, text="abc")
    provenance = ClaimProvenance(
        source=SourceMetadata(source_id=source_id, reliability_prior=reliability),
        spans=(span,),
    )
    return AtomicClaim(
        claim_id=claim_id,
        text=text,
        provenance=provenance,
        extraction_confidence=0.9,
        source_reliability=reliability,
        belief=0.5,
        asserted_at=asserted_at,
    )


def _support(rid: str, src: str, dst: str) -> EvidenceRelation:
    return EvidenceRelation(
        relation_id=rid,
        source_claim_id=src,
        target_claim_id=dst,
        relation_type=RelationType.SUPPORT,
        relation_confidence=0.8,
    )


@pytest.mark.sanity
def test_contradictions_are_never_silently_discarded() -> None:
    """A contradiction is retained in a conflict set, not dropped."""

    c1 = _claim("c1", "srcA", text="EG-RAG is generator agnostic")
    c2 = _claim("c2", "srcB", text="EG-RAG is not generator agnostic")
    contradiction = EvidenceRelation(
        relation_id="r1",
        source_claim_id="c1",
        target_claim_id="c2",
        relation_type=RelationType.CONTRADICTION,
        relation_confidence=0.7,
    )
    snapshot = EvidenceGraphSnapshot(snapshot_id="g1", claims=(c1, c2), relations=(contradiction,))

    conflicts = FakeConflictResolver().resolve(snapshot)

    assert len(conflicts) == 1
    assert set(conflicts[0].claim_ids) == {"c1", "c2"}
    assert conflicts[0].resolved is False  # retained, not silently resolved away


@pytest.mark.sanity
def test_pipeline_surfaces_conflicts_in_package() -> None:
    """The demo pipeline carries its contradiction through to the package."""

    result = EGRagPipeline(build_demo_components()).run(
        Query(query_id="q1", text="Is EG-RAG generator agnostic?")
    )
    assert result.metrics.num_conflicts >= 1
    assert result.package.conflicts


@pytest.mark.sanity
def test_recency_does_not_increase_belief() -> None:
    """Newer evidence is not automatically more truthful."""

    scorer = FakeInitialClaimScorer()
    query = Query(query_id="q1", text="shared content")
    older = _claim("c1", "srcA", asserted_at=datetime(2000, 1, 1, tzinfo=UTC))
    newer = _claim("c2", "srcA", asserted_at=datetime(2025, 1, 1, tzinfo=UTC))

    scored_older = scorer.score(older, query)
    scored_newer = scorer.score(newer, query)

    assert scored_older.belief == scored_newer.belief


@pytest.mark.sanity
def test_single_source_repetition_is_not_corroboration() -> None:
    """Repetition from one source must not raise belief; a distinct source may."""

    propagator = FakeBeliefPropagator()
    target = _claim("t", "srcT")

    # One supporter from srcB.
    one = EvidenceGraphSnapshot(
        snapshot_id="g1",
        claims=(target, _claim("s1", "srcB")),
        relations=(_support("r1", "s1", "t"),),
    )
    # Two supporters, both still from srcB (mere repetition).
    repeated = EvidenceGraphSnapshot(
        snapshot_id="g2",
        claims=(target, _claim("s1", "srcB"), _claim("s2", "srcB")),
        relations=(_support("r1", "s1", "t"), _support("r2", "s2", "t")),
    )
    # Two supporters from two distinct sources (genuine corroboration).
    distinct = EvidenceGraphSnapshot(
        snapshot_id="g3",
        claims=(target, _claim("s1", "srcB"), _claim("s2", "srcC")),
        relations=(_support("r1", "s1", "t"), _support("r2", "s2", "t")),
    )

    def belief_of(snapshot: EvidenceGraphSnapshot) -> float:
        propagated = propagator.propagate(snapshot)
        value = next(c.belief for c in propagated.claims if c.claim_id == "t")
        assert value is not None
        return value

    assert belief_of(repeated) == belief_of(one)
    assert belief_of(distinct) > belief_of(one)


@pytest.mark.sanity
def test_score_concepts_remain_distinct() -> None:
    """The five evidentiary quantities are independent fields, never conflated."""

    claim = AtomicClaim(
        claim_id="c1",
        text="distinct scores",
        provenance=ClaimProvenance(
            source=SourceMetadata(source_id="s1"),
            spans=(SourceSpan(source_id="s1", start=0, end=3, text="abc"),),
        ),
        extraction_confidence=0.30,
        source_reliability=0.40,
        belief=0.50,
        query_utility=0.60,
    )
    relation = EvidenceRelation(
        relation_id="r1",
        source_claim_id="c1",
        target_claim_id="c2",
        relation_type=RelationType.SUPPORT,
        relation_confidence=0.70,
    )
    values = {
        claim.extraction_confidence,
        claim.source_reliability,
        claim.belief,
        claim.query_utility,
        relation.relation_confidence,
    }
    assert len(values) == 5  # all five are independently set and distinct


@pytest.mark.sanity
def test_every_extracted_claim_has_provenance() -> None:
    """Provenance is total: each claim retains at least one source span."""

    components = build_demo_components()
    result = EGRagPipeline(components).run(Query(query_id="q1", text="evidence graph"))
    assert result.package.claims
    for claim in result.package.claims:
        assert len(claim.provenance.spans) >= 1
