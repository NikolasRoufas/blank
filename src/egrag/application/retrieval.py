"""Retrieve-then-rerank orchestration over the domain ports.

This helper depends only on the :class:`Retriever` and :class:`Reranker`
protocols, so it works with any concrete adapter or fake. It retrieves the top
``top_k`` candidates, reranks them, and returns the top ``top_n``.
"""

from __future__ import annotations

from egrag.domain.models import Passage, Query
from egrag.domain.ports import Reranker, Retriever


def retrieve_and_rerank(
    retriever: Retriever,
    reranker: Reranker,
    query: Query,
    *,
    top_k: int,
    top_n: int,
) -> list[Passage]:
    """Retrieve ``top_k`` candidates, rerank, and return the top ``top_n``.

    ``top_n == 0`` returns an empty list; negative ``top_n`` is rejected.
    """

    if top_n < 0:
        raise ValueError(f"top_n must be >= 0, got {top_n}")
    candidates = retriever.retrieve(query, top_k)
    reranked = reranker.rerank(query, candidates)
    if top_n == 0:
        return []
    return reranked[:top_n]


__all__ = ["retrieve_and_rerank"]
