"""GraphML export via NetworkX — the only NetworkX usage in the codebase.

Requires the ``graph`` extra. NetworkX is imported lazily on first use, so
importing this module pulls in no optional dependency. The domain remains the
source of truth; NetworkX is used purely as a serialization format here.
"""

from __future__ import annotations

import importlib
from typing import Any

from egrag.domain.models import EvidenceGraphSnapshot


def to_graphml(snapshot: EvidenceGraphSnapshot) -> str:
    """Return a GraphML string for the snapshot (requires the ``graph`` extra)."""

    try:
        nx: Any = importlib.import_module("networkx")
    except ModuleNotFoundError as exc:
        raise RuntimeError("networkx is not installed; install the 'graph' extra") from exc

    graph = nx.MultiDiGraph()
    for claim in snapshot.claims:
        graph.add_node(
            claim.claim_id,
            text=claim.text,
            source_id=claim.provenance.source.source_id,
            extraction_confidence=claim.extraction_confidence,
        )
    for relation in snapshot.relations:
        graph.add_edge(
            relation.source_claim_id,
            relation.target_claim_id,
            key=relation.relation_id,
            relation_type=relation.relation_type.value,
            direction=relation.direction.value,
            confidence=relation.relation_confidence,
        )
    lines: list[bytes] = list(nx.generate_graphml(graph))
    return "\n".join(line.decode("utf-8") if isinstance(line, bytes) else line for line in lines)


__all__ = ["to_graphml"]
