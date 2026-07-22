"""Assemble a versioned, model-independent EvidencePackage from reasoning output.

Status per selected claim is derived from the graph and conflicts (superseded,
rejected/disputed via conflicts, else accepted). The token budget is enforced
here, **before** generation: claims that do not fit are dropped from the
selection (not from the full claim set) and recorded in the package budget, so
truncation is never silent.
"""

from __future__ import annotations

from egrag.domain.graph import connected_components
from egrag.domain.models import (
    ConflictOutcome,
    ConflictSet,
    EvidencePackage,
    EvidenceStatus,
    GenerationPolicy,
    PackageBudget,
    Query,
    RelationType,
    SelectedEvidence,
)
from egrag.domain.ports import TokenCounter
from egrag.graph.api import EvidenceGraph
from egrag.reasoning.models import ScoreBoard, SelectionResult, TokenBudget


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _status_map(
    graph: EvidenceGraph, conflicts: tuple[ConflictSet, ...], selected_ids: set[str]
) -> dict[str, EvidenceStatus]:
    superseded = {cid for cid in selected_ids if graph.superseded_by(cid)}
    unresolved: set[str] = set()
    rejected: set[str] = set()
    for conflict in conflicts:
        if conflict.outcome is ConflictOutcome.UNRESOLVED:
            unresolved.update(conflict.claim_ids)
        elif conflict.outcome is ConflictOutcome.PREFERRED and conflict.preferred_claim_id:
            rejected.update(set(conflict.claim_ids) - {conflict.preferred_claim_id})
        elif conflict.outcome is ConflictOutcome.REJECTED_LOW_EVIDENCE:
            rejected.update(conflict.claim_ids)

    status: dict[str, EvidenceStatus] = {}
    for cid in selected_ids:
        if cid in superseded:
            status[cid] = EvidenceStatus.SUPERSEDED
        elif cid in rejected:
            status[cid] = EvidenceStatus.REJECTED
        elif cid in unresolved:
            status[cid] = EvidenceStatus.DISPUTED
        else:
            status[cid] = EvidenceStatus.ACCEPTED
    return status


def build_evidence_package(
    query: Query,
    graph: EvidenceGraph,
    board: ScoreBoard,
    conflicts: list[ConflictSet],
    selection: SelectionResult,
    *,
    budget: TokenBudget,
    token_counter: TokenCounter,
    policy: GenerationPolicy | None = None,
    package_id: str = "package",
) -> EvidencePackage:
    """Build the evidence package, enforcing the token budget before generation."""

    snapshot = graph.snapshot()
    scores = board.by_id()
    entries = {entry.claim_id: entry for entry in selection.entries}
    selected_ids = list(selection.selected_claim_ids)
    status = _status_map(graph, tuple(conflicts), set(selected_ids))
    text_by_id = {claim.claim_id: claim.text for claim in snapshot.claims}

    kept: list[SelectedEvidence] = []
    truncated: list[str] = []
    used = 0
    for rank, cid in enumerate(selected_ids):
        cost = token_counter.count(text_by_id.get(cid, ""))
        if used + cost > budget.available:
            truncated.append(cid)
            continue
        used += cost
        score = scores.get(cid)
        entry = entries.get(cid)
        belief = (
            score.propagated_belief
            if score is not None and score.propagated_belief is not None
            else (score.initial_belief if score is not None else None)
        )
        kept.append(
            SelectedEvidence(
                claim_id=cid,
                selection_score=_clamp01(entry.final_selection_score) if entry else 0.0,
                rank=rank,
                status=status.get(cid, EvidenceStatus.ACCEPTED),
                belief=belief,
                utility=score.query_utility if score is not None else None,
                reason=entry.reason if entry else None,
            )
        )

    duplicate_groups = tuple(
        tuple(group)
        for group in connected_components(snapshot, relation_types={RelationType.DUPLICATE})
        if len(group) >= 2
    )

    return EvidencePackage(
        package_id=package_id,
        query=query,
        claims=snapshot.claims,
        relations=snapshot.relations,
        conflicts=tuple(conflicts),
        selected=tuple(kept),
        subgraph=None,
        duplicate_groups=duplicate_groups,
        generation_policy=policy or GenerationPolicy(),
        budget=PackageBudget(
            total=budget.total,
            reserved_output=budget.reserved_output,
            used=used,
            truncated_claim_ids=tuple(truncated),
        ),
    )


__all__ = ["build_evidence_package"]
