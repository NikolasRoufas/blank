"""Unit tests for reranking and retrieve-then-rerank (acceptance 25)."""

from __future__ import annotations

import pytest

from egrag.adapters.reranking.score import ScoreReranker
from egrag.adapters.retrieval.bm25 import BM25Retriever
from egrag.application.retrieval import retrieve_and_rerank
from egrag.domain.models import ClaimProvenance, Passage, Query, SourceMetadata, SourceSpan


def _passage(passage_id: str, text: str, score: float | None) -> Passage:
    return Passage(
        passage_id=passage_id,
        document_id="doc",
        text=text,
        span=SourceSpan(source_id="src", start=0, end=len(text), text=text),
        retrieval_score=score,
    )


@pytest.mark.unit
def test_score_reranker_orders_by_score_preserving_identity() -> None:
    """Acceptance 25: reranking preserves passage identity and provenance."""

    passages = [
        _passage("p-low", "low relevance", 0.1),
        _passage("p-high", "high relevance", 0.9),
        _passage("p-mid", "mid relevance", 0.5),
    ]
    reranked = ScoreReranker().rerank(Query(query_id="q", text="relevance"), passages)
    assert [p.passage_id for p in reranked] == ["p-high", "p-mid", "p-low"]
    # identity and provenance preserved
    assert {p.passage_id for p in reranked} == {p.passage_id for p in passages}
    by_id = {p.passage_id: p for p in passages}
    for result in reranked:
        # same span/provenance survive the rerank (only rank changes)
        original = by_id[result.passage_id]
        assert result.span == original.span
        assert result.document_id == original.document_id


@pytest.mark.unit
def test_score_reranker_assigns_ranks() -> None:
    passages = [_passage("p1", "a", 0.2), _passage("p2", "b", 0.8)]
    reranked = ScoreReranker().rerank(Query(query_id="q", text="x"), passages)
    assert [p.rank for p in reranked] == [0, 1]


@pytest.mark.unit
def test_score_reranker_ties_deterministic() -> None:
    passages = [_passage("p-b", "x", 0.5), _passage("p-a", "x", 0.5)]
    reranked = ScoreReranker().rerank(Query(query_id="q", text="x"), passages)
    assert [p.passage_id for p in reranked] == ["p-a", "p-b"]


@pytest.mark.unit
def test_retrieve_and_rerank_applies_top_k_then_top_n() -> None:
    # Only one passage contains the query term, so it is the unambiguous winner
    # regardless of BM25 length normalization.
    corpus = [
        _passage("p-elephant", "the elephant", None),
        _passage("p-mouse", "the mouse", None),
        _passage("p-bird", "the bird", None),
    ]
    retriever = BM25Retriever(corpus)
    result = retrieve_and_rerank(
        retriever, ScoreReranker(), Query(query_id="q", text="elephant"), top_k=5, top_n=1
    )
    assert len(result) == 1
    assert result[0].passage_id == "p-elephant"


@pytest.mark.unit
def test_retrieve_and_rerank_top_n_zero_and_negative() -> None:
    corpus = [_passage("p1", "elephant", None)]
    retriever = BM25Retriever(corpus)
    query = Query(query_id="q", text="elephant")
    assert retrieve_and_rerank(retriever, ScoreReranker(), query, top_k=5, top_n=0) == []
    with pytest.raises(ValueError):
        retrieve_and_rerank(retriever, ScoreReranker(), query, top_k=5, top_n=-1)


@pytest.mark.unit
def test_provenance_fields_survive_rerank() -> None:
    """An explicit provenance check: span offsets are intact after reranking."""

    span = SourceSpan(source_id="src", start=4, end=12, text="evidence")
    provenance = ClaimProvenance(source=SourceMetadata(source_id="src"), spans=(span,))
    assert provenance.spans[0].start == 4  # sanity on the fixture
    passage = Passage(
        passage_id="p1",
        document_id="doc",
        text="an evidence span",
        span=SourceSpan(source_id="src", start=0, end=16, text="an evidence span"),
        retrieval_score=0.3,
    )
    reranked = ScoreReranker().rerank(Query(query_id="q", text="evidence"), [passage])
    assert reranked[0].span == passage.span
