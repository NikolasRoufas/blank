"""Hybrid retrieval via score fusion of multiple component retrievers.

Two strategies are supported:

* **weighted** — each component's candidate scores are min-max normalized to
  [0, 1], then combined as a weighted sum. The fused ``score`` and
  ``normalized_score`` are in [0, 1].
* **rrf** — reciprocal rank fusion: each component contributes
  ``weight / (k + rank)``; ``score`` is the (un-normalized) RRF sum and
  ``normalized_score`` is ``None``.

Weights are normalized to sum to 1 (documented behavior). Invalid weights or an
invalid strategy are rejected. Candidates are deduplicated by stable passage id,
so identical passages (same id) merge while passages from different sources
(different ids) are both preserved. Raw component scores are retained for
inspection, kept distinct from the normalized fused score. Ties break
deterministically by passage id.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from egrag.adapters.retrieval.base import (
    BaseRetriever,
    FusionStrategy,
    RankStats,
    ScoredPassage,
    min_max_normalize,
)
from egrag.domain.models import Passage, Query


class HybridRetriever(BaseRetriever):
    """Fuses several component retrievers into one ranking."""

    name = "hybrid"

    def __init__(
        self,
        components: Mapping[str, BaseRetriever],
        *,
        strategy: FusionStrategy | str = FusionStrategy.WEIGHTED,
        weights: Mapping[str, float] | None = None,
        rrf_k: int = 60,
    ) -> None:
        if not components:
            raise ValueError("hybrid retriever requires at least one component")
        if rrf_k <= 0:
            raise ValueError(f"rrf_k must be > 0, got {rrf_k}")
        self._components = dict(components)
        self._strategy = FusionStrategy(strategy)  # invalid strategy -> ValueError
        self._rrf_k = rrf_k
        self._weights = self._normalize_weights(weights)

    def _normalize_weights(self, weights: Mapping[str, float] | None) -> dict[str, float]:
        names = list(self._components)
        if weights is None:
            resolved = dict.fromkeys(names, 1.0)
        else:
            unknown = set(weights) - set(names)
            if unknown:
                raise ValueError(f"weights reference unknown components: {sorted(unknown)}")
            resolved = {name: float(weights.get(name, 0.0)) for name in names}
        if any(value < 0.0 for value in resolved.values()):
            raise ValueError("weights must be non-negative")
        total = sum(resolved.values())
        if total <= 0.0:
            raise ValueError("weights must sum to a positive value")
        return {name: value / total for name, value in resolved.items()}

    def _rank(self, query: Query, top_k: int) -> tuple[list[ScoredPassage], RankStats]:
        per_component: dict[str, list[ScoredPassage]] = {}
        cache_hits = 0
        cache_misses = 0
        embedding_ms = 0.0
        for name, component in self._components.items():
            report = component.search_report(query, top_k)
            per_component[name] = list(report.results)
            cache_hits += report.stats.cache_hits
            cache_misses += report.stats.cache_misses
            embedding_ms += report.stats.embedding_ms

        passages: dict[str, Passage] = {}
        raw_scores: dict[str, dict[str, float]] = defaultdict(dict)
        ranks: dict[str, dict[str, int]] = defaultdict(dict)
        for name, results in per_component.items():
            for item in results:
                passage_id = item.passage.passage_id
                passages[passage_id] = item.passage
                raw_scores[passage_id][name] = item.score
                ranks[passage_id][name] = item.rank

        if self._strategy is FusionStrategy.WEIGHTED:
            fused = self._fuse_weighted(passages, per_component, raw_scores)
        else:
            fused = self._fuse_rrf(passages, ranks, raw_scores)

        fused.sort(key=lambda item: (-item.score, item.passage.passage_id))
        stats = RankStats(
            cache_hits=cache_hits, cache_misses=cache_misses, embedding_ms=embedding_ms
        )
        return fused, stats

    def _fuse_weighted(
        self,
        passages: dict[str, Passage],
        per_component: dict[str, list[ScoredPassage]],
        raw_scores: dict[str, dict[str, float]],
    ) -> list[ScoredPassage]:
        normalized: dict[str, dict[str, float]] = {}
        for name, results in per_component.items():
            normalized[name] = min_max_normalize(
                {item.passage.passage_id: item.score for item in results}
            )
        fused: list[ScoredPassage] = []
        for passage_id, passage in passages.items():
            total = 0.0
            for name in self._components:
                value = normalized.get(name, {}).get(passage_id)
                if value is None:
                    continue
                total += self._weights[name] * value
            fused.append(
                ScoredPassage(
                    passage=passage,
                    score=round(total, 6),
                    normalized_score=round(total, 6),
                    components=dict(raw_scores[passage_id]),
                    retriever=self.name,
                )
            )
        return fused

    def _fuse_rrf(
        self,
        passages: dict[str, Passage],
        ranks: dict[str, dict[str, int]],
        raw_scores: dict[str, dict[str, float]],
    ) -> list[ScoredPassage]:
        fused: list[ScoredPassage] = []
        for passage_id, passage in passages.items():
            total = 0.0
            for name in self._components:
                rank = ranks[passage_id].get(name)
                if rank is None:
                    continue
                total += self._weights[name] * (1.0 / (self._rrf_k + rank + 1))
            fused.append(
                ScoredPassage(
                    passage=passage,
                    score=round(total, 8),
                    normalized_score=None,
                    components=dict(raw_scores[passage_id]),
                    retriever=self.name,
                )
            )
        return fused


__all__ = ["HybridRetriever"]
