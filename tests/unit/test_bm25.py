"""Unit tests for BM25 sparse retrieval (acceptance 1, 7, 10-14)."""

from __future__ import annotations

import pytest

from egrag.adapters.retrieval.bm25 import BM25Retriever
from egrag.domain.models import Passage, Query, SourceSpan


def _passage(passage_id: str, text: str, source_id: str = "src") -> Passage:
    return Passage(
        passage_id=passage_id,
        document_id="doc",
        text=text,
        span=SourceSpan(source_id=source_id, start=0, end=len(text), text=text),
    )


CORPUS = [
    _passage("p-elephant", "the elephant roams the savanna grassland"),
    _passage("p-spaceship", "the spaceship orbits a distant planet"),
    _passage("p-mixed", "an elephant and a spaceship appear together"),
]


@pytest.mark.unit
def test_exact_term_passage_ranks_above_unrelated() -> None:
    """Acceptance 1: a passage with the query term outranks unrelated passages."""

    retriever = BM25Retriever(CORPUS)
    results = retriever.search(Query(query_id="q", text="elephant"), top_k=10)
    ids = [item.passage.passage_id for item in results]
    assert "p-elephant" in ids
    assert "p-spaceship" not in ids  # no shared term -> excluded
    # the dedicated elephant passage ranks above the mixed passage
    assert ids.index("p-elephant") < ids.index("p-mixed")


@pytest.mark.unit
def test_ties_are_deterministic() -> None:
    """Acceptance 7: identical-content passages tie-break by passage id, stably."""

    corpus = [
        _passage("p-b", "duplicate content token"),
        _passage("p-a", "duplicate content token"),
    ]
    retriever = BM25Retriever(corpus)
    query = Query(query_id="q", text="duplicate content")
    first = [item.passage.passage_id for item in retriever.search(query, top_k=10)]
    second = [item.passage.passage_id for item in retriever.search(query, top_k=10)]
    assert first == second == ["p-a", "p-b"]


@pytest.mark.unit
def test_empty_corpus_returns_empty() -> None:
    """Acceptance 10: an empty corpus returns an empty result without crashing."""

    retriever = BM25Retriever([])
    assert retriever.search(Query(query_id="q", text="anything"), top_k=5) == []


@pytest.mark.unit
def test_empty_query_returns_empty() -> None:
    """Acceptance 11: a query with no tokens returns no results (documented)."""

    retriever = BM25Retriever(CORPUS)
    # Non-empty text, but no alphanumeric tokens.
    assert retriever.search(Query(query_id="q", text="?!?-"), top_k=5) == []


@pytest.mark.unit
def test_top_k_zero_returns_empty() -> None:
    """Acceptance 12: top_k=0 returns an empty result (documented)."""

    retriever = BM25Retriever(CORPUS)
    assert retriever.search(Query(query_id="q", text="elephant"), top_k=0) == []


@pytest.mark.unit
def test_negative_top_k_rejected() -> None:
    """Acceptance 13: negative top_k is rejected."""

    retriever = BM25Retriever(CORPUS)
    with pytest.raises(ValueError):
        retriever.search(Query(query_id="q", text="elephant"), top_k=-1)


@pytest.mark.unit
def test_more_results_than_corpus_is_safe() -> None:
    """Acceptance 14: requesting more results than the corpus size is safe."""

    retriever = BM25Retriever(CORPUS)
    results = retriever.search(Query(query_id="q", text="elephant spaceship"), top_k=100)
    assert len(results) <= len(CORPUS)


@pytest.mark.unit
def test_retrieve_sets_passage_score_and_rank() -> None:
    """The Retriever-protocol path returns domain passages with score and rank."""

    retriever = BM25Retriever(CORPUS)
    passages = retriever.retrieve(Query(query_id="q", text="elephant"), top_k=5)
    assert passages[0].retrieval_score is not None
    assert passages[0].rank == 0
    for passage in passages:
        assert isinstance(passage, Passage)


@pytest.mark.unit
def test_invalid_bm25_parameters_rejected() -> None:
    with pytest.raises(ValueError):
        BM25Retriever(CORPUS, k1=-1.0)
    with pytest.raises(ValueError):
        BM25Retriever(CORPUS, b=1.5)
