"""Structural validation for evidence graphs.

Rejects dangling edges, duplicate node IDs, duplicate relation IDs, invalid
relation types, invalid confidence values, unsupported self-edges, stored
neutral edges (unless explicitly allowed), and malformed provenance references.
"""

from __future__ import annotations

from collections.abc import Sequence

from egrag.domain.errors import GraphValidationError
from egrag.domain.models import AtomicClaim, EvidenceGraphSnapshot, EvidenceRelation, RelationType


def validate_components(
    claims: Sequence[AtomicClaim],
    relations: Sequence[EvidenceRelation],
    *,
    allow_self_edges: bool = False,
    allow_neutral: bool = False,
) -> None:
    """Validate claims and relations; raise :class:`GraphValidationError` on any issue."""

    claim_ids = [claim.claim_id for claim in claims]
    if len(claim_ids) != len(set(claim_ids)):
        raise GraphValidationError("duplicate node IDs are not allowed")
    known = set(claim_ids)

    for claim in claims:
        if len(claim.provenance.spans) < 1:
            raise GraphValidationError(
                f"claim {claim.claim_id!r} has malformed provenance (no source span)"
            )

    relation_ids: list[str] = []
    for relation in relations:
        relation_ids.append(relation.relation_id)
        missing = {relation.source_claim_id, relation.target_claim_id} - known
        if missing:
            raise GraphValidationError(
                f"dangling edge {relation.relation_id!r} references unknown claim(s): "
                f"{sorted(missing)}"
            )
        if not allow_self_edges and relation.source_claim_id == relation.target_claim_id:
            raise GraphValidationError(f"self-edge {relation.relation_id!r} is not supported")
        if not allow_neutral and relation.relation_type is RelationType.NEUTRAL:
            raise GraphValidationError(
                f"neutral edge {relation.relation_id!r} must not be stored outside debug mode"
            )
        if not 0.0 <= relation.relation_confidence <= 1.0:
            raise GraphValidationError(
                f"edge {relation.relation_id!r} has invalid confidence "
                f"{relation.relation_confidence}"
            )
    if len(relation_ids) != len(set(relation_ids)):
        raise GraphValidationError("duplicate relation IDs are not allowed")


def validate_snapshot(
    snapshot: EvidenceGraphSnapshot,
    *,
    allow_self_edges: bool = False,
    allow_neutral: bool = False,
) -> None:
    """Validate a built snapshot."""

    validate_components(
        snapshot.claims,
        snapshot.relations,
        allow_self_edges=allow_self_edges,
        allow_neutral=allow_neutral,
    )


__all__ = ["validate_components", "validate_snapshot"]
