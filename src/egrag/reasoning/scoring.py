"""Baseline initial claim scoring (a configurable weighted-sum baseline).

Initial belief is a convex combination of six normalized signals — retrieval,
query relevance, extraction confidence, source reliability, temporal validity,
and independent support — which is numerically stable and avoids multiplying
many probabilities. Each component is preserved separately, along with its
weighted contribution, so the score is fully explainable. Query utility is
computed independently of belief.
"""

from __future__ import annotations

from collections.abc import Mapping

from egrag.domain.models import Query
from egrag.domain.ports import SourceReliabilityScorer
from egrag.graph.api import EvidenceGraph
from egrag.graph.text_utils import jaccard, tokens
from egrag.reasoning.models import ClaimScore, ScoreBoard, ScoreComponents, ScoreWeights


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class BaselineInitialScorer:
    """Computes per-claim :class:`ClaimScore` records from a graph and a query."""

    def __init__(
        self,
        reliability: SourceReliabilityScorer,
        *,
        weights: ScoreWeights | None = None,
        superseded_validity: float = 0.5,
        default_retrieval: float = 0.5,
    ) -> None:
        if not 0.0 <= superseded_validity <= 1.0:
            raise ValueError("superseded_validity must be in [0, 1]")
        if not 0.0 <= default_retrieval <= 1.0:
            raise ValueError("default_retrieval must be in [0, 1]")
        self._reliability = reliability
        self._weights = weights or ScoreWeights()
        self._superseded_validity = superseded_validity
        self._default_retrieval = default_retrieval

    def score(
        self,
        graph: EvidenceGraph,
        query: Query,
        *,
        retrieval_scores: Mapping[str, float] | None = None,
    ) -> ScoreBoard:
        retrieval_map = dict(retrieval_scores or {})
        query_tokens = tokens(query.text)
        weights = self._weights
        total_weight = weights.total()

        scores: list[ClaimScore] = []
        for claim in graph.nodes():
            cid = claim.claim_id
            retrieval = _clamp01(retrieval_map.get(cid, self._default_retrieval))
            query_relevance = jaccard(tokens(claim.text), query_tokens)
            extraction = _clamp01(claim.extraction_confidence)
            source_reliability = _clamp01(self._reliability.score(claim.provenance.source))
            is_superseded = bool(graph.superseded_by(cid))
            temporal_validity = self._superseded_validity if is_superseded else 1.0
            distinct_sources = len(graph.corroborating_sources(cid))
            independent_support = distinct_sources / (distinct_sources + 1)

            components = ScoreComponents(
                retrieval=retrieval,
                query_relevance=round(query_relevance, 6),
                extraction=extraction,
                source_reliability=source_reliability,
                temporal_validity=temporal_validity,
                independent_support=round(independent_support, 6),
            )
            signal_pairs = {
                "retrieval": (weights.retrieval, retrieval),
                "query_relevance": (weights.query_relevance, query_relevance),
                "extraction": (weights.extraction, extraction),
                "source_reliability": (weights.source_reliability, source_reliability),
                "temporal_validity": (weights.temporal_validity, temporal_validity),
                "independent_support": (weights.independent_support, independent_support),
            }
            contributions = {
                name: round(weight * value / total_weight, 6)
                for name, (weight, value) in signal_pairs.items()
            }
            initial_belief = _clamp01(round(sum(contributions.values()), 6))
            query_utility = _clamp01(round((query_relevance + retrieval) / 2.0, 6))

            scores.append(
                ClaimScore(
                    claim_id=cid,
                    components=components,
                    initial_belief=initial_belief,
                    propagated_belief=None,
                    query_utility=query_utility,
                    provenance_diversity=distinct_sources,
                    contributions=contributions,
                )
            )
        return ScoreBoard(scores=tuple(scores))


__all__ = ["BaselineInitialScorer"]
