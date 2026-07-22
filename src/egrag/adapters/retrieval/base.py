"""Shared retrieval types, scoring helpers, and the retriever base class.

The types here are provider-agnostic: ``ScoredPassage`` carries a passage plus
its final score, raw per-component scores (kept distinct from any normalized
score), and its rank. No BM25- or embedding-specific object ever leaves this
layer, so the application layer only ever sees domain ``Passage`` objects via
the :class:`egrag.domain.ports.Retriever` protocol.
"""

from __future__ import annotations

import abc
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field

from egrag.domain.models import Passage, Query
from egrag.domain.ports import Embedding


class ScoredPassage(BaseModel):
    """A passage with its ranking score and inspectable component scores."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    passage: Passage
    score: float
    rank: int = Field(default=0, ge=0)
    # Raw, un-normalized per-retriever scores, keyed by retriever name.
    components: dict[str, float] = Field(default_factory=dict)
    # Normalized fused score in [0, 1] when the strategy defines one; else None.
    normalized_score: float | None = None
    retriever: str | None = None


class RetrievalStats(BaseModel):
    """Observability counters and timings for a single retrieval call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    num_candidates: int = Field(default=0, ge=0)
    num_results: int = Field(default=0, ge=0)
    cache_hits: int = Field(default=0, ge=0)
    cache_misses: int = Field(default=0, ge=0)
    retrieval_ms: float = Field(default=0.0, ge=0.0)
    embedding_ms: float = Field(default=0.0, ge=0.0)


class RetrievalReport(BaseModel):
    """Retrieval results paired with their statistics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    results: tuple[ScoredPassage, ...] = ()
    stats: RetrievalStats = Field(default_factory=RetrievalStats)


class FusionStrategy(StrEnum):
    """Supported hybrid fusion strategies."""

    WEIGHTED = "weighted"
    RECIPROCAL_RANK = "rrf"


@dataclass(frozen=True)
class RankStats:
    """Internal per-rank statistics reported by concrete retrievers."""

    cache_hits: int = 0
    cache_misses: int = 0
    embedding_ms: float = 0.0


def validate_top_k(top_k: int) -> None:
    """Validate the documented ``top_k`` contract.

    ``top_k == 0`` is valid and yields an empty result; negative values are
    rejected with :class:`ValueError`.
    """

    if top_k < 0:
        raise ValueError(f"top_k must be >= 0, got {top_k}")


def l2_norm(vector: Sequence[float]) -> float:
    """Return the Euclidean norm of ``vector``."""

    return math.sqrt(sum(component * component for component in vector))


def cosine_similarity(a: Embedding, b: Embedding) -> float:
    """Return the cosine similarity of two vectors; 0.0 if either is zero.

    Cosine similarity normalizes by vector length, so embeddings need not be
    pre-normalized; the normalization is explicit and happens here.
    """

    if len(a) != len(b):
        raise ValueError(f"embedding dimensions differ: {len(a)} != {len(b)}")
    norm_a = l2_norm(a)
    norm_b = l2_norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (norm_a * norm_b)


def min_max_normalize(scores: Mapping[str, float]) -> dict[str, float]:
    """Min-max normalize scores to [0, 1].

    When every score is equal (including a single item), all map to 1.0, since
    they are equally relevant within that component's candidate set.
    """

    if not scores:
        return {}
    values = list(scores.values())
    low, high = min(values), max(values)
    if high == low:
        return dict.fromkeys(scores, 1.0)
    span = high - low
    return {key: (value - low) / span for key, value in scores.items()}


class BaseRetriever(abc.ABC):
    """Base class implementing the :class:`Retriever` protocol via ranking.

    Subclasses implement :meth:`_rank`, returning *all* scored candidates in
    deterministic descending order. The base class handles ``top_k`` validation,
    slicing, rank assignment, and statistics so behavior is uniform.
    """

    name: str = "retriever"

    @abc.abstractmethod
    def _rank(self, query: Query, top_k: int) -> tuple[list[ScoredPassage], RankStats]:
        """Return all scored candidates (sorted desc, deterministic tie-break)."""

    def _run(self, query: Query, top_k: int) -> tuple[list[ScoredPassage], RetrievalStats]:
        validate_top_k(top_k)
        start = perf_counter()
        if top_k == 0:
            elapsed = (perf_counter() - start) * 1000.0
            return [], RetrievalStats(num_candidates=0, num_results=0, retrieval_ms=elapsed)
        ranked, rank_stats = self._rank(query, top_k)
        sliced = ranked[:top_k]
        results = [item.model_copy(update={"rank": index}) for index, item in enumerate(sliced)]
        elapsed = (perf_counter() - start) * 1000.0
        stats = RetrievalStats(
            num_candidates=len(ranked),
            num_results=len(results),
            cache_hits=rank_stats.cache_hits,
            cache_misses=rank_stats.cache_misses,
            retrieval_ms=elapsed,
            embedding_ms=rank_stats.embedding_ms,
        )
        return results, stats

    def search(self, query: Query, top_k: int) -> list[ScoredPassage]:
        """Return the top-``top_k`` scored passages."""

        return self._run(query, top_k)[0]

    def search_report(self, query: Query, top_k: int) -> RetrievalReport:
        """Return the top-``top_k`` results together with retrieval statistics."""

        results, stats = self._run(query, top_k)
        return RetrievalReport(results=tuple(results), stats=stats)

    def retrieve(self, query: Query, top_k: int) -> list[Passage]:
        """Implement the :class:`Retriever` protocol, returning domain passages."""

        return [
            item.passage.model_copy(update={"retrieval_score": item.score, "rank": item.rank})
            for item in self.search(query, top_k)
        ]


__all__ = [
    "BaseRetriever",
    "FusionStrategy",
    "RankStats",
    "RetrievalReport",
    "RetrievalStats",
    "ScoredPassage",
    "cosine_similarity",
    "l2_norm",
    "min_max_normalize",
    "validate_top_k",
]
