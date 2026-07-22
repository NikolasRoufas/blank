"""Unit tests for chunking and passage preparation (acceptance 15-18)."""

from __future__ import annotations

import pytest

from egrag.adapters.retrieval.chunking import (
    SentenceAwareChunker,
    WholeDocumentChunker,
    prepare_passages,
)
from egrag.domain.models import Document, SourceMetadata

FOUR_SENTENCES = "Alpha is first. Beta is second. Gamma is third. Delta is fourth."


def _doc(text: str, doc_id: str = "doc-1", source_id: str = "src-1") -> Document:
    return Document(
        document_id=doc_id,
        text=text,
        source=SourceMetadata(source_id=source_id),
    )


def _assert_offsets_match(document: Document, passages: list) -> None:
    for passage in passages:
        assert passage.text  # never empty
        assert passage.text == document.text[passage.span.start : passage.span.end]
        assert passage.span.text == passage.text
        assert passage.span.source_id == document.source.source_id


@pytest.mark.unit
def test_whole_document_chunker_preserves_prechunked_input() -> None:
    """Acceptance 18: pre-chunked input is preserved as a single passage."""

    document = _doc("A pre-chunked passage of text.")
    passages = WholeDocumentChunker().chunk(document)
    assert len(passages) == 1
    assert passages[0].text == document.text
    assert passages[0].span.start == 0
    assert passages[0].span.end == len(document.text)
    _assert_offsets_match(document, passages)


@pytest.mark.unit
def test_chunking_never_produces_empty_chunks() -> None:
    """Acceptance 16: chunking never produces empty chunks."""

    document = _doc(FOUR_SENTENCES)
    passages = SentenceAwareChunker(chunk_size=20, overlap=0).chunk(document)
    assert passages
    for passage in passages:
        assert passage.text.strip()
        assert passage.span.end > passage.span.start


@pytest.mark.unit
def test_large_chunk_size_avoids_unnecessary_splits() -> None:
    """Acceptance 17: a chunk size larger than the document yields one chunk."""

    document = _doc(FOUR_SENTENCES)
    passages = SentenceAwareChunker(chunk_size=1000, overlap=0).chunk(document)
    assert len(passages) == 1
    assert passages[0].text == document.text


@pytest.mark.unit
def test_sentence_aware_chunking_does_not_split_sentences() -> None:
    """Acceptance 17: chunk boundaries fall on sentence boundaries."""

    document = _doc(FOUR_SENTENCES)
    passages = SentenceAwareChunker(chunk_size=20, overlap=0).chunk(document)
    assert len(passages) > 1  # the document was split
    # Every sentence ends with '.', so every chunk must end with '.' — proving
    # no chunk boundary fell in the middle of a sentence.
    for passage in passages:
        assert passage.text.rstrip().endswith(".")
    _assert_offsets_match(document, passages)


@pytest.mark.unit
def test_chunk_overlap_preserves_offsets_and_overlaps() -> None:
    """Acceptance 15: overlapping chunks keep correct offsets and actually overlap."""

    document = _doc(FOUR_SENTENCES)
    passages = SentenceAwareChunker(chunk_size=32, overlap=16).chunk(document)
    _assert_offsets_match(document, passages)
    assert len(passages) > 1
    # At least one consecutive pair overlaps in source-character space.
    overlaps = [passages[i + 1].span.start < passages[i].span.end for i in range(len(passages) - 1)]
    assert any(overlaps)


@pytest.mark.unit
def test_chunking_deduplicates_identical_spans() -> None:
    """Duplicate chunk spans are not emitted twice."""

    document = _doc(FOUR_SENTENCES)
    passages = SentenceAwareChunker(chunk_size=40, overlap=20).chunk(document)
    spans = [(p.span.start, p.span.end) for p in passages]
    assert len(spans) == len(set(spans))
    ids = [p.passage_id for p in passages]
    assert len(ids) == len(set(ids))


@pytest.mark.unit
def test_document_without_sentence_boundary_yields_one_chunk() -> None:
    """A document with no terminator still yields exactly one non-empty chunk."""

    document = _doc("no terminator here")
    passages = SentenceAwareChunker(chunk_size=5, overlap=0).chunk(document)
    assert len(passages) == 1
    assert passages[0].text == document.text


@pytest.mark.unit
def test_prepare_passages_over_empty_corpus() -> None:
    """Acceptance 10 (preparation): an empty document set yields no passages."""

    assert prepare_passages([], SentenceAwareChunker()) == []


@pytest.mark.unit
def test_prepare_passages_assigns_unique_ids_across_documents() -> None:
    documents = [_doc(FOUR_SENTENCES, "doc-a", "src-a"), _doc(FOUR_SENTENCES, "doc-b", "src-b")]
    passages = prepare_passages(documents, SentenceAwareChunker(chunk_size=20, overlap=0))
    ids = [p.passage_id for p in passages]
    assert len(ids) == len(set(ids))


@pytest.mark.unit
def test_invalid_chunker_configuration_rejected() -> None:
    with pytest.raises(ValueError):
        SentenceAwareChunker(chunk_size=0)
    with pytest.raises(ValueError):
        SentenceAwareChunker(chunk_size=10, overlap=10)  # overlap must be < chunk_size
    with pytest.raises(ValueError):
        SentenceAwareChunker(chunk_size=10, overlap=-1)


@pytest.mark.unit
def test_input_document_not_mutated() -> None:
    document = _doc(FOUR_SENTENCES)
    original = document.model_copy(deep=True)
    SentenceAwareChunker(chunk_size=20, overlap=0).chunk(document)
    assert document == original
