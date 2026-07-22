"""Property-based tests for chunking offsets, score bounds, and determinism."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from egrag.adapters.retrieval.bm25 import BM25Retriever
from egrag.adapters.retrieval.chunking import SentenceAwareChunker
from egrag.domain.models import Document, Passage, Query, SourceMetadata, SourceSpan

# Text drawn from letters, spaces, and sentence terminators; non-empty after strip.
_doc_text = (
    st.text(alphabet=st.sampled_from("abcdefg .!?"), min_size=1, max_size=200)
    .map(str.strip)
    .filter(lambda value: len(value) > 0)
)


def _passage(passage_id: str, text: str) -> Passage:
    return Passage(
        passage_id=passage_id,
        document_id="doc",
        text=text,
        span=SourceSpan(source_id="src", start=0, end=len(text), text=text),
    )


@pytest.mark.property
@given(
    text=_doc_text,
    chunk_size=st.integers(min_value=1, max_value=64),
    overlap=st.integers(min_value=0, max_value=63),
)
def test_chunk_offsets_are_always_valid(text: str, chunk_size: int, overlap: int) -> None:
    if overlap >= chunk_size:
        overlap = chunk_size - 1
    document = Document(document_id="doc", text=text, source=SourceMetadata(source_id="src"))
    passages = SentenceAwareChunker(chunk_size=chunk_size, overlap=overlap).chunk(document)
    assert passages
    for passage in passages:
        assert passage.text  # never empty
        assert 0 <= passage.span.start < passage.span.end <= len(document.text)
        assert passage.text == document.text[passage.span.start : passage.span.end]


@pytest.mark.property
@given(
    query_text=st.text(alphabet=st.sampled_from("elephant mouse "), min_size=1, max_size=20)
    .map(str.strip)
    .filter(lambda value: len(value) > 0)
)
def test_bm25_scores_are_non_negative_and_deterministic(query_text: str) -> None:
    corpus = [
        _passage("p1", "the elephant is large"),
        _passage("p2", "the mouse is small"),
        _passage("p3", "an elephant and a mouse"),
    ]
    retriever = BM25Retriever(corpus)
    query = Query(query_id="q", text=query_text)
    first = retriever.search(query, top_k=10)
    second = retriever.search(query, top_k=10)

    assert all(item.score > 0.0 for item in first)  # only positive matches returned
    assert [i.passage.passage_id for i in first] == [i.passage.passage_id for i in second]
    assert [i.score for i in first] == [i.score for i in second]
