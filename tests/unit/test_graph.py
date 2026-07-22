"""Unit tests for the pure domain graph algorithms."""

from __future__ import annotations

import pytest

from egrag.domain.graph import (
    build_adjacency,
    connected_components,
    distinct_supporting_sources,
    induced_relation_ids,
    neighbors,
)
from egrag.domain.models import (
    AtomicClaim,
    ClaimProvenance,
    EvidenceGraphSnapshot,
    EvidenceRelation,
    RelationType,
    SourceMetadata,
    SourceSpan,
)


def _claim(claim_id: str, source_id: str) -> AtomicClaim:
    span = SourceSpan(source_id=source_id, start=0, end=3, text="abc")
    provenance = ClaimProvenance(source=SourceMetadata(source_id=source_id), spans=(span,))
    return AtomicClaim(
        claim_id=claim_id, text="text here", provenance=provenance, extraction_confidence=0.9
    )


def _rel(rid: str, src: str, dst: str, kind: RelationType) -> EvidenceRelation:
    return EvidenceRelation(
        relation_id=rid,
        source_claim_id=src,
        target_claim_id=dst,
        relation_type=kind,
        relation_confidence=0.8,
    )


def _snapshot() -> EvidenceGraphSnapshot:
    claims = (
        _claim("c1", "srcA"),
        _claim("c2", "srcB"),
        _claim("c3", "srcA"),  # same source as c1
        _claim("c4", "srcC"),
    )
    relations = (
        _rel("r1", "c2", "c1", RelationType.SUPPORT),
        _rel("r2", "c3", "c1", RelationType.SUPPORT),  # same source as c1 -> no corroboration
        _rel("r3", "c4", "c1", RelationType.SUPPORT),
        _rel("r4", "c1", "c2", RelationType.CONTRADICTION),
    )
    return EvidenceGraphSnapshot(snapshot_id="g1", claims=claims, relations=relations)


@pytest.mark.unit
def test_build_adjacency_includes_all_claims() -> None:
    adjacency = build_adjacency(_snapshot())
    assert set(adjacency) == {"c1", "c2", "c3", "c4"}
    assert [r.relation_id for r in adjacency["c2"]] == ["r1"]


@pytest.mark.unit
def test_neighbors_are_undirected_and_sorted() -> None:
    snapshot = _snapshot()
    assert neighbors(snapshot, "c1") == ("c2", "c3", "c4")
    assert neighbors(snapshot, "c1", relation_types={RelationType.CONTRADICTION}) == ("c2",)


@pytest.mark.unit
def test_connected_components_over_contradictions() -> None:
    components = connected_components(_snapshot(), relation_types={RelationType.CONTRADICTION})
    assert components == [("c1", "c2")]


@pytest.mark.unit
def test_distinct_supporting_sources_excludes_same_source() -> None:
    """c1 is supported by c2 (srcB), c3 (srcA, same as c1), c4 (srcC).

    Only distinct sources other than c1's own (srcA) count: srcB and srcC -> 2.
    """

    snapshot = _snapshot()
    source_of = {c.claim_id: c.provenance.source.source_id for c in snapshot.claims}
    assert distinct_supporting_sources(snapshot, "c1", source_of) == 2


@pytest.mark.unit
def test_induced_relation_ids() -> None:
    snapshot = _snapshot()
    assert set(induced_relation_ids(snapshot, ["c1", "c2"])) == {"r1", "r4"}
    assert induced_relation_ids(snapshot, ["c1"]) == ()
