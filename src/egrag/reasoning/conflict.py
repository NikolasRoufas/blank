"""Conflict-set detection and resolution (a configurable baseline).

Detects *groups* of competing claims (connected components over contradiction
edges), not just isolated edges. Each set preserves the signals used to judge it
(propagated belief, source reliability, independent support, relation
confidence, timestamps). Resolution ranks members by propagated belief, then
independent support, then reliability — **recency is never a tiebreaker**. Close
competitors (within a configurable margin) are left UNRESOLVED rather than
forced to a winner; contradictory evidence is always retained.
"""

from __future__ import annotations

from egrag.domain.graph import connected_components
from egrag.domain.models import (
    ConflictMember,
    ConflictOutcome,
    ConflictSet,
    EvidenceRelation,
    RelationType,
)
from egrag.graph.api import EvidenceGraph
from egrag.graph.temporal import resolve_temporal
from egrag.reasoning.models import ClaimScore, ScoreBoard


class ConflictSetResolver:
    """Resolves contradiction components into explained conflict sets."""

    def __init__(
        self,
        *,
        margin: float = 0.1,
        low_evidence_threshold: float = 0.15,
        irrelevant_utility_threshold: float = 0.0,
    ) -> None:
        for name, value in (
            ("margin", margin),
            ("low_evidence_threshold", low_evidence_threshold),
            ("irrelevant_utility_threshold", irrelevant_utility_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        self._margin = margin
        self._low_evidence = low_evidence_threshold
        self._irrelevant = irrelevant_utility_threshold

    def resolve(self, graph: EvidenceGraph, board: ScoreBoard) -> list[ConflictSet]:
        snapshot = graph.snapshot()
        scores = board.by_id()
        components = connected_components(snapshot, relation_types={RelationType.CONTRADICTION})
        contradiction_edges = [
            r for r in snapshot.relations if r.relation_type is RelationType.CONTRADICTION
        ]

        conflicts: list[ConflictSet] = []
        for index, member_ids in enumerate(components):
            if len(member_ids) < 2:
                continue
            member_set = set(member_ids)
            relation_ids = tuple(
                sorted(
                    r.relation_id
                    for r in contradiction_edges
                    if r.source_claim_id in member_set and r.target_claim_id in member_set
                )
            )
            members = self._build_members(member_ids, contradiction_edges, graph, scores)
            outcome, preferred, rationale = self._resolve_members(
                member_ids, members, scores, graph
            )
            conflicts.append(
                ConflictSet(
                    conflict_id=f"conflict:{index}",
                    claim_ids=tuple(member_ids),
                    relation_ids=relation_ids,
                    members=members,
                    outcome=outcome,
                    preferred_claim_id=preferred,
                    resolved=outcome is not ConflictOutcome.UNRESOLVED,
                    resolution_note=rationale,
                )
            )
        return conflicts

    def _build_members(
        self,
        member_ids: tuple[str, ...],
        contradiction_edges: list[EvidenceRelation],
        graph: EvidenceGraph,
        scores: dict[str, ClaimScore],
    ) -> tuple[ConflictMember, ...]:
        members: list[ConflictMember] = []
        for cid in member_ids:
            score = scores.get(cid)
            belief = (
                score.propagated_belief
                if score is not None and score.propagated_belief is not None
                else (score.initial_belief if score is not None else 0.5)
            )
            reliability = score.components.source_reliability if score is not None else 0.5
            independent = score.provenance_diversity if score is not None else 0
            rel_conf = max(
                (
                    r.relation_confidence
                    for r in contradiction_edges
                    if cid in (r.source_claim_id, r.target_claim_id)
                ),
                default=None,
            )
            claim = graph.node(cid)
            timestamp = resolve_temporal(claim).effective_time if claim is not None else None
            members.append(
                ConflictMember(
                    claim_id=cid,
                    propagated_belief=round(belief, 6),
                    source_reliability=round(reliability, 6),
                    independent_support=independent,
                    relation_confidence=rel_conf,
                    timestamp=timestamp,
                )
            )
        return tuple(members)

    def _resolve_members(
        self,
        member_ids: tuple[str, ...],
        members: tuple[ConflictMember, ...],
        scores: dict[str, ClaimScore],
        graph: EvidenceGraph,
    ) -> tuple[ConflictOutcome, str | None, str]:
        utilities = [scores[m.claim_id].query_utility for m in members if m.claim_id in scores]
        if utilities and all(u <= self._irrelevant for u in utilities):
            return (
                ConflictOutcome.EXCLUDED_IRRELEVANT,
                None,
                "all competing claims fall below the relevance threshold",
            )

        # Rank by belief, then independent support, then reliability (NOT recency).
        ranked = sorted(
            members,
            key=lambda m: (-m.propagated_belief, -m.independent_support, -m.source_reliability),
        )
        top = ranked[0]
        if top.propagated_belief < self._low_evidence:
            return (
                ConflictOutcome.REJECTED_LOW_EVIDENCE,
                None,
                f"strongest competitor belief {top.propagated_belief:.3f} below "
                f"low-evidence threshold {self._low_evidence:.3f}",
            )

        runner_up = ranked[1]
        if top.propagated_belief - runner_up.propagated_belief < self._margin:
            return (
                ConflictOutcome.UNRESOLVED,
                None,
                f"top two competitors within margin {self._margin:.3f} "
                f"({top.propagated_belief:.3f} vs {runner_up.propagated_belief:.3f})",
            )

        if self._supersedes_any(top.claim_id, member_ids, graph):
            return (
                ConflictOutcome.SUPERSEDED,
                top.claim_id,
                f"{top.claim_id!r} supersedes a competing claim and leads on belief",
            )
        return (
            ConflictOutcome.PREFERRED,
            top.claim_id,
            f"{top.claim_id!r} leads on belief by at least the margin {self._margin:.3f}",
        )

    @staticmethod
    def _supersedes_any(claim_id: str, member_ids: tuple[str, ...], graph: EvidenceGraph) -> bool:
        superseded = {c.claim_id for c in graph.supersedes(claim_id)}
        return bool(superseded & set(member_ids))


__all__ = ["ConflictSetResolver"]
