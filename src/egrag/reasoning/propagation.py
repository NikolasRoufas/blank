"""Deterministic signed belief propagation (a configurable baseline).

Belief updates happen in logit space and are squashed back through a sigmoid, so
values stay bounded and no probabilities are multiplied. Support edges send
positive messages; contradiction edges send negative messages. Repeated
supporters that share a source (lineage) or a duplicate cluster are discounted,
so duplicated evidence and copied sources do not inflate belief. Damping plus a
convergence tolerance and iteration cap guard against oscillation and runaway
scaling; logit clamping guards against overflow. SUPERSEDES is intentionally
*not* used here to erase evidence (it affects temporal validity during scoring).

See ``docs/reasoning.md`` for the equations and assumptions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from egrag.domain.errors import ConvergenceError
from egrag.domain.models import EvidenceGraphSnapshot, RelationType
from egrag.graph.api import EvidenceGraph
from egrag.reasoning.models import (
    ClaimScore,
    IterationDiagnostic,
    PropagationConfig,
    PropagationResult,
    ScoreBoard,
)

_EPS = 1e-6


def _logit(p: float, clamp: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return max(-clamp, min(clamp, math.log(p / (1.0 - p))))


def _sigmoid(x: float, clamp: float) -> float:
    x = max(-clamp, min(clamp, x))
    return 1.0 / (1.0 + math.exp(-x))


def _connected_components(nodes: Sequence[str], edges: list[tuple[str, str]]) -> dict[str, str]:
    """Union-find component id (smallest member) per node, over undirected edges."""

    parent = {node: node for node in nodes}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in edges:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    return {node: find(node) for node in nodes}


class SignedBeliefPropagator:
    """Iterative signed message passing over support/contradiction edges."""

    def __init__(self, config: PropagationConfig | None = None) -> None:
        self._config = config or PropagationConfig()

    def propagate(self, graph: EvidenceGraph, board: ScoreBoard) -> PropagationResult:
        cfg = self._config
        scores = board.by_id()
        snapshot = graph.snapshot()
        node_ids = [claim.claim_id for claim in snapshot.claims]

        beliefs: dict[str, float] = {}
        for cid in node_ids:
            score = scores.get(cid)
            value = score.initial_belief if score is not None else 0.5
            if not math.isfinite(value):
                raise ValueError(f"initial belief for {cid!r} is not finite: {value}")
            beliefs[cid] = min(max(value, 0.0), 1.0)

        if not node_ids:
            return PropagationResult(beliefs={}, iterations=0, converged=True)

        support_in, contra_adj = self._build_adjacency(snapshot, node_ids)
        source_of = {c.claim_id: c.provenance.source.source_id for c in snapshot.claims}
        dup_cluster = _connected_components(
            node_ids,
            [
                (r.source_claim_id, r.target_claim_id)
                for r in snapshot.relations
                if r.relation_type is RelationType.DUPLICATE
            ],
        )
        base_logit = {cid: _logit(beliefs[cid], cfg.logit_clamp) for cid in node_ids}

        diagnostics: list[IterationDiagnostic] = []
        converged = False
        iterations = 0
        for iteration in range(1, cfg.max_iterations + 1):
            iterations = iteration
            updated: dict[str, float] = {}
            for cid in node_ids:
                net = base_logit[cid]
                net += self._support_message(cid, support_in, beliefs, source_of, dup_cluster, cfg)
                net += self._contradiction_message(cid, contra_adj, beliefs, source_of, cfg)
                target = _sigmoid(net, cfg.logit_clamp)
                updated[cid] = cfg.damping * beliefs[cid] + (1.0 - cfg.damping) * target
            max_delta = max(abs(updated[cid] - beliefs[cid]) for cid in node_ids)
            beliefs = updated
            diagnostics.append(IterationDiagnostic(iteration=iteration, max_delta=max_delta))
            if max_delta < cfg.tolerance:
                converged = True
                break

        if not converged and cfg.on_nonconvergence == "raise":
            raise ConvergenceError(
                f"belief propagation did not converge within {cfg.max_iterations} iterations "
                f"(last max_delta={diagnostics[-1].max_delta:.3e})"
            )

        bounded = {cid: min(max(beliefs[cid], 0.0), 1.0) for cid in node_ids}
        for value in bounded.values():
            if not math.isfinite(value):  # pragma: no cover - clamps prevent this
                raise ValueError("propagated belief became non-finite")
        return PropagationResult(
            beliefs=bounded,
            iterations=iterations,
            converged=converged,
            diagnostics=tuple(diagnostics),
        )

    def apply(self, board: ScoreBoard, result: PropagationResult) -> ScoreBoard:
        """Return a new ScoreBoard with propagated beliefs filled in."""

        updated: list[ClaimScore] = []
        for score in board.scores:
            value = result.beliefs.get(score.claim_id, score.initial_belief)
            updated.append(score.with_propagated(round(value, 6)))
        return ScoreBoard(scores=tuple(updated))

    # --- helpers -------------------------------------------------------------

    @staticmethod
    def _build_adjacency(
        snapshot: EvidenceGraphSnapshot, node_ids: list[str]
    ) -> tuple[dict[str, list[tuple[str, float]]], dict[str, list[tuple[str, float]]]]:
        support_in: dict[str, list[tuple[str, float]]] = {cid: [] for cid in node_ids}
        contra_adj: dict[str, list[tuple[str, float]]] = {cid: [] for cid in node_ids}
        for relation in snapshot.relations:
            src, tgt = relation.source_claim_id, relation.target_claim_id
            if src == tgt:
                continue  # guard against self-support
            if relation.relation_type is RelationType.SUPPORT:
                support_in[tgt].append((src, relation.relation_confidence))
            elif relation.relation_type is RelationType.CONTRADICTION:
                contra_adj[src].append((tgt, relation.relation_confidence))
                contra_adj[tgt].append((src, relation.relation_confidence))
        return support_in, contra_adj

    @staticmethod
    def _discounted(
        neighbors: list[tuple[str, float]],
        beliefs: dict[str, float],
        source_of: dict[str, str],
        dup_cluster: dict[str, str] | None,
        cfg: PropagationConfig,
    ) -> float:
        """Sum signed strengths with lineage/duplicate discounting (deterministic)."""

        source_rank: dict[str, int] = {}
        cluster_rank: dict[str, int] = {}
        total = 0.0
        for neighbor_id, rel_conf in sorted(neighbors, key=lambda item: item[0]):
            source_key = source_of.get(neighbor_id, neighbor_id)
            s_rank = source_rank.get(source_key, 0)
            source_rank[source_key] = s_rank + 1
            factor = cfg.lineage_discount**s_rank
            if dup_cluster is not None:
                cluster_key = dup_cluster.get(neighbor_id, neighbor_id)
                c_rank = cluster_rank.get(cluster_key, 0)
                cluster_rank[cluster_key] = c_rank + 1
                factor *= cfg.duplicate_discount**c_rank
            signed = 2.0 * (beliefs[neighbor_id] - 0.5)  # in [-1, 1]
            total += rel_conf * signed * factor
        return total

    def _support_message(
        self,
        cid: str,
        support_in: dict[str, list[tuple[str, float]]],
        beliefs: dict[str, float],
        source_of: dict[str, str],
        dup_cluster: dict[str, str],
        cfg: PropagationConfig,
    ) -> float:
        return cfg.support_weight * self._discounted(
            support_in[cid], beliefs, source_of, dup_cluster, cfg
        )

    def _contradiction_message(
        self,
        cid: str,
        contra_adj: dict[str, list[tuple[str, float]]],
        beliefs: dict[str, float],
        source_of: dict[str, str],
        cfg: PropagationConfig,
    ) -> float:
        # Contradiction discounts by source lineage but not by duplicate cluster.
        return -cfg.contradiction_weight * self._discounted(
            contra_adj[cid], beliefs, source_of, None, cfg
        )


class NoPropagationBaseline:
    """Ablation baseline: returns initial beliefs unchanged (no message passing)."""

    def propagate(self, graph: EvidenceGraph, board: ScoreBoard) -> PropagationResult:
        beliefs: dict[str, float] = {}
        for score in board.scores:
            if not math.isfinite(score.initial_belief):
                raise ValueError(f"initial belief for {score.claim_id!r} is not finite")
            beliefs[score.claim_id] = score.initial_belief
        return PropagationResult(beliefs=beliefs, iterations=0, converged=True)

    def apply(self, board: ScoreBoard, result: PropagationResult) -> ScoreBoard:
        return SignedBeliefPropagator().apply(board, result)


__all__ = ["NoPropagationBaseline", "SignedBeliefPropagator"]
