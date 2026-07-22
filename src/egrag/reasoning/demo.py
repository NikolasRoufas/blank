"""A small deterministic synthetic example for the reasoning demonstration trace.

Four claims about the same proposition exercise support, contradiction, and a
same-source duplicate:

* ``c1`` (srcA): "Acme revenue grew in 2023"
* ``c2`` (srcB): "Acme revenue increased in 2023"  -- supports c1 (independent)
* ``c3`` (srcC): "Acme revenue fell in 2023"        -- contradicts c1
* ``c4`` (srcA): "Acme revenue grew in 2023"        -- duplicate of c1 (same source)
"""

from __future__ import annotations

from egrag.domain.models import (
    AtomicClaim,
    ClaimProvenance,
    ClaimSemantics,
    EvidenceGraphSnapshot,
    EvidenceRelation,
    Query,
    RelationDirection,
    SourceMetadata,
    SourceSpan,
)
from egrag.domain.models import RelationType as RT
from egrag.graph.api import EvidenceGraph


def _claim(cid: str, text: str, source_id: str, reliability: float) -> AtomicClaim:
    return AtomicClaim(
        claim_id=cid,
        text=text,
        provenance=ClaimProvenance(
            source=SourceMetadata(source_id=source_id, reliability_prior=reliability),
            spans=(SourceSpan(source_id=source_id, start=0, end=len(text), text=text),),
        ),
        extraction_confidence=0.85,
        semantics=ClaimSemantics(named_entities=("Acme",)),
    )


def demo_query() -> Query:
    return Query(query_id="demo", text="did Acme revenue grow in 2023")


def build_demo_reasoning_graph() -> EvidenceGraph:
    """Build the fixed synthetic reasoning graph."""

    claims = [
        _claim("c1", "Acme revenue grew in 2023", "srcA", 0.8),
        _claim("c2", "Acme revenue increased in 2023", "srcB", 0.7),
        _claim("c3", "Acme revenue fell in 2023", "srcC", 0.6),
        _claim("c4", "Acme revenue grew in 2023", "srcA", 0.8),
    ]
    relations = [
        EvidenceRelation(
            relation_id="e1",
            source_claim_id="c2",
            target_claim_id="c1",
            relation_type=RT.SUPPORT,
            relation_confidence=0.85,
            direction=RelationDirection.DIRECTED,
        ),
        EvidenceRelation(
            relation_id="e2",
            source_claim_id="c1",
            target_claim_id="c3",
            relation_type=RT.CONTRADICTION,
            relation_confidence=0.8,
            direction=RelationDirection.SYMMETRIC,
        ),
        EvidenceRelation(
            relation_id="e3",
            source_claim_id="c1",
            target_claim_id="c4",
            relation_type=RT.DUPLICATE,
            relation_confidence=0.99,
            direction=RelationDirection.SYMMETRIC,
        ),
    ]
    return EvidenceGraph(
        EvidenceGraphSnapshot(snapshot_id="demo", claims=tuple(claims), relations=tuple(relations))
    )


__all__ = ["build_demo_reasoning_graph", "demo_query"]
