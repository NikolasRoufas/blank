"""Score-based baseline reranker.

Reorders candidate passages by their existing retrieval score (descending), with
deterministic tie-breaking by passage id. Passages are returned unchanged except
for an updated rank, so stable passage identity and provenance are preserved.
Passages missing a score are ordered last.
"""

from __future__ import annotations

from collections.abc import Sequence

from egrag.domain.models import Passage, Query


class ScoreReranker:
    """Deterministic reranker that sorts by existing retrieval score."""

    name = "score-reranker"

    def rerank(self, query: Query, passages: Sequence[Passage]) -> list[Passage]:
        ordered = sorted(
            passages,
            key=lambda passage: (
                -(
                    passage.retrieval_score
                    if passage.retrieval_score is not None
                    else float("-inf")
                ),
                passage.passage_id,
            ),
        )
        return [passage.model_copy(update={"rank": index}) for index, passage in enumerate(ordered)]


__all__ = ["ScoreReranker"]
