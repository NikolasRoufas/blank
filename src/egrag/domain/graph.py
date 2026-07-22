"""Pure graph algorithms over :class:`EvidenceGraphSnapshot`.

These functions operate directly on domain models and depend on no third-party
graph library. They are deliberately small and deterministic; richer graph
reasoning (belief propagation, conflict resolution, subgraph selection) lives
behind ports and, for non-trivial methods, in the ``experimental`` package.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Iterable

from egrag.domain.models import EvidenceGraphSnapshot, EvidenceRelation, RelationType


def claim_ids(snapshot: EvidenceGraphSnapshot) -> tuple[str, ...]:
    """Return the claim identifiers in the snapshot, in declaration order."""

    return tuple(claim.claim_id for claim in snapshot.claims)


def build_adjacency(
    snapshot: EvidenceGraphSnapshot,
    *,
    relation_types: Collection[RelationType] | None = None,
) -> dict[str, list[EvidenceRelation]]:
    """Build an outgoing-edge adjacency map keyed by source claim id.

    When ``relation_types`` is given, only relations of those types are
    included. Every claim id appears as a key, even with no outgoing edges.
    """

    adjacency: dict[str, list[EvidenceRelation]] = {cid: [] for cid in claim_ids(snapshot)}
    for relation in snapshot.relations:
        if relation_types is not None and relation.relation_type not in relation_types:
            continue
        adjacency[relation.source_claim_id].append(relation)
    return adjacency


def neighbors(
    snapshot: EvidenceGraphSnapshot,
    claim_id: str,
    *,
    relation_types: Collection[RelationType] | None = None,
) -> tuple[str, ...]:
    """Return claim ids adjacent to ``claim_id`` over undirected edges.

    Edges are treated as undirected for neighborhood queries; ordering is
    deterministic (sorted) for reproducibility.
    """

    found: set[str] = set()
    for relation in snapshot.relations:
        if relation_types is not None and relation.relation_type not in relation_types:
            continue
        if relation.source_claim_id == claim_id:
            found.add(relation.target_claim_id)
        elif relation.target_claim_id == claim_id:
            found.add(relation.source_claim_id)
    return tuple(sorted(found))


def connected_components(
    snapshot: EvidenceGraphSnapshot,
    *,
    relation_types: Collection[RelationType] | None = None,
) -> list[tuple[str, ...]]:
    """Return connected components over the (undirected) selected relations.

    Only claims touched by at least one selected relation are returned. Each
    component is a sorted tuple; components are ordered by their smallest member
    for determinism.
    """

    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for relation in snapshot.relations:
        if relation_types is not None and relation.relation_type not in relation_types:
            continue
        union(relation.source_claim_id, relation.target_claim_id)

    groups: dict[str, set[str]] = defaultdict(set)
    for node in list(parent):
        groups[find(node)].add(node)

    components = [tuple(sorted(members)) for members in groups.values()]
    components.sort(key=lambda comp: comp[0])
    return components


def distinct_supporting_sources(
    snapshot: EvidenceGraphSnapshot,
    claim_id: str,
    claim_source: dict[str, str],
) -> int:
    """Count the distinct sources that support ``claim_id``.

    Repetition from a single source must not be counted as independent
    corroboration, so this counts *distinct* source ids among supporting
    neighbors (excluding the claim's own source).
    """

    own_source = claim_source.get(claim_id)
    supporters: set[str] = set()
    for relation in snapshot.relations:
        if relation.relation_type is not RelationType.SUPPORT:
            continue
        if relation.target_claim_id != claim_id:
            continue
        source = claim_source.get(relation.source_claim_id)
        if source is not None and source != own_source:
            supporters.add(source)
    return len(supporters)


def induced_relation_ids(
    snapshot: EvidenceGraphSnapshot,
    selected_claim_ids: Iterable[str],
) -> tuple[str, ...]:
    """Return ids of relations whose endpoints are both in the selection."""

    selected = set(selected_claim_ids)
    return tuple(
        relation.relation_id
        for relation in snapshot.relations
        if relation.source_claim_id in selected and relation.target_claim_id in selected
    )


__all__ = [
    "build_adjacency",
    "claim_ids",
    "connected_components",
    "distinct_supporting_sources",
    "induced_relation_ids",
    "neighbors",
]
