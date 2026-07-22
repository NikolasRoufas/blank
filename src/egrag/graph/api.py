"""Read API over an immutable evidence-graph snapshot.

Wraps :class:`EvidenceGraphSnapshot` (the source of truth) and exposes node,
edge, neighbor, evidence, duplicate, temporal, lineage, component, cluster,
summary, validation, and serialization operations — all without NetworkX.
Symmetric relations (CONTRADICTION, DUPLICATE) are treated bidirectionally.
"""

from __future__ import annotations

from collections.abc import Collection

from egrag.domain.graph import connected_components
from egrag.domain.models import (
    AtomicClaim,
    EvidenceGraphSnapshot,
    EvidenceRelation,
    RelationType,
)
from egrag.graph.types import GraphSummary
from egrag.graph.validation import validate_snapshot


class EvidenceGraph:
    """An immutable, queryable view over an evidence-graph snapshot."""

    def __init__(self, snapshot: EvidenceGraphSnapshot) -> None:
        self._snapshot = snapshot
        self._claims: dict[str, AtomicClaim] = {c.claim_id: c for c in snapshot.claims}

    # --- nodes & edges -------------------------------------------------------

    def nodes(self) -> tuple[AtomicClaim, ...]:
        return self._snapshot.claims

    def node(self, claim_id: str) -> AtomicClaim | None:
        return self._claims.get(claim_id)

    def edges(self) -> tuple[EvidenceRelation, ...]:
        return self._snapshot.relations

    def edges_of_type(self, relation_type: RelationType) -> tuple[EvidenceRelation, ...]:
        return tuple(r for r in self._snapshot.relations if r.relation_type is relation_type)

    def neighbors(
        self, claim_id: str, *, relation_types: Collection[RelationType] | None = None
    ) -> tuple[str, ...]:
        """Return neighbor claim ids (edges treated undirected for neighborhoods)."""

        found: set[str] = set()
        for relation in self._snapshot.relations:
            if relation_types is not None and relation.relation_type not in relation_types:
                continue
            if relation.source_claim_id == claim_id:
                found.add(relation.target_claim_id)
            elif relation.target_claim_id == claim_id:
                found.add(relation.source_claim_id)
        return tuple(sorted(found))

    # --- evidence queries ----------------------------------------------------

    def supporting_evidence(self, claim_id: str) -> tuple[AtomicClaim, ...]:
        """Claims that directionally SUPPORT ``claim_id`` (edge source → claim_id)."""

        ids = sorted(
            r.source_claim_id
            for r in self._snapshot.relations
            if r.relation_type is RelationType.SUPPORT and r.target_claim_id == claim_id
        )
        return tuple(self._claims[i] for i in ids if i in self._claims)

    def contradicting_evidence(self, claim_id: str) -> tuple[AtomicClaim, ...]:
        """Claims connected to ``claim_id`` by a (symmetric) CONTRADICTION edge."""

        ids = sorted(self.neighbors(claim_id, relation_types={RelationType.CONTRADICTION}))
        return tuple(self._claims[i] for i in ids if i in self._claims)

    def duplicates(self, claim_id: str) -> tuple[AtomicClaim, ...]:
        """Claims connected to ``claim_id`` by a (symmetric) DUPLICATE edge."""

        ids = sorted(self.neighbors(claim_id, relation_types={RelationType.DUPLICATE}))
        return tuple(self._claims[i] for i in ids if i in self._claims)

    def superseded_by(self, claim_id: str) -> tuple[AtomicClaim, ...]:
        """Newer claims that SUPERSEDE ``claim_id`` (edge newer → claim_id)."""

        ids = sorted(
            r.source_claim_id
            for r in self._snapshot.relations
            if r.relation_type is RelationType.SUPERSESSION and r.target_claim_id == claim_id
        )
        return tuple(self._claims[i] for i in ids if i in self._claims)

    def supersedes(self, claim_id: str) -> tuple[AtomicClaim, ...]:
        """Older claims that ``claim_id`` SUPERSEDES (edge claim_id → older)."""

        ids = sorted(
            r.target_claim_id
            for r in self._snapshot.relations
            if r.relation_type is RelationType.SUPERSESSION and r.source_claim_id == claim_id
        )
        return tuple(self._claims[i] for i in ids if i in self._claims)

    def corroborating_sources(self, claim_id: str) -> tuple[str, ...]:
        """Distinct source ids among supporters, excluding the claim's own source.

        Counting distinct sources (not distinct supporters) ensures same-source
        paraphrases are not treated as independent corroboration.
        """

        own = (
            self._claims[claim_id].provenance.source.source_id if claim_id in self._claims else None
        )
        sources = {
            supporter.provenance.source.source_id
            for supporter in self.supporting_evidence(claim_id)
            if supporter.provenance.source.source_id != own
        }
        return tuple(sorted(sources))

    def source_lineage(self, claim_id: str) -> tuple[str, ...]:
        """Return the chain of claim ids from ``claim_id`` through what it supersedes."""

        lineage: list[str] = []
        seen: set[str] = set()
        current: str | None = claim_id
        while current is not None and current not in seen:
            seen.add(current)
            lineage.append(current)
            older = self.supersedes(current)
            current = older[0].claim_id if older else None
        return tuple(lineage)

    # --- structure -----------------------------------------------------------

    def connected_components(self) -> list[tuple[str, ...]]:
        components = connected_components(self._snapshot)
        connected = {cid for component in components for cid in component}
        singletons = [(c.claim_id,) for c in self._snapshot.claims if c.claim_id not in connected]
        return sorted([*components, *singletons], key=lambda comp: comp[0])

    def clusters(self) -> list[tuple[str, ...]]:
        """Claim clusters: connected components over all stored relations."""

        return self.connected_components()

    # --- snapshot, validation, summary --------------------------------------

    def snapshot(self) -> EvidenceGraphSnapshot:
        return self._snapshot

    def validate(self, *, allow_self_edges: bool = False, allow_neutral: bool = False) -> None:
        validate_snapshot(
            self._snapshot, allow_self_edges=allow_self_edges, allow_neutral=allow_neutral
        )

    def summary(self) -> GraphSummary:
        edges_by_type: dict[str, int] = {}
        for relation in self._snapshot.relations:
            key = relation.relation_type.value
            edges_by_type[key] = edges_by_type.get(key, 0) + 1
        sources = {c.provenance.source.source_id for c in self._snapshot.claims}
        return GraphSummary(
            num_nodes=len(self._snapshot.claims),
            num_edges=len(self._snapshot.relations),
            edges_by_type=edges_by_type,
            num_components=len(self.connected_components()),
            num_sources=len(sources),
        )


__all__ = ["EvidenceGraph"]
