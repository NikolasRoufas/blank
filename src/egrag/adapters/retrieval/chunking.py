"""Document chunkers producing passages with accurate source offsets.

Chunkers never normalize or mutate input: each passage's text is exactly the
original document substring ``document.text[span.start:span.end]``, and input
``Document`` objects are not modified. Chunkers never emit empty passages and
prevent duplicate chunks. Sentence-aware chunking avoids splitting in the middle
of a sentence whenever a reliable sentence boundary is available.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from egrag.domain.models import Document, Passage, SourceSpan

# A sentence is a run of non-terminator characters optionally followed by
# terminators, or a final run with no terminator at end-of-string.
_SENTENCE_RE = re.compile(r"[^.!?]*[.!?]+|[^.!?]+\Z")


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) ranges of sentence content, trimmed of whitespace."""

    spans: list[tuple[int, int]] = []
    for match in _SENTENCE_RE.finditer(text):
        raw = match.group()
        if not raw:
            continue
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw) - len(raw.rstrip())
        start = match.start() + leading
        end = match.end() - trailing
        if end > start:
            spans.append((start, end))
    return spans


def _make_passage(document: Document, index: int, start: int, end: int) -> Passage:
    text = document.text[start:end]
    span = SourceSpan(
        source_id=document.source.source_id,
        start=start,
        end=end,
        text=text,
    )
    return Passage(
        passage_id=f"{document.document_id}::p{index}",
        document_id=document.document_id,
        text=text,
        span=span,
    )


class WholeDocumentChunker:
    """Emits the entire document as a single passage (pre-chunked input).

    Use this when documents are already chunked: the passage preserves the full
    text and the offsets ``[0, len(text)]``.
    """

    def chunk(self, document: Document) -> list[Passage]:
        return [_make_passage(document, 0, 0, len(document.text))]


class SentenceAwareChunker:
    """Groups whole sentences into chunks of a configurable size and overlap.

    ``chunk_size`` and ``overlap`` are measured in characters of the source
    text. A sentence is never split; an oversized single sentence becomes its
    own chunk. ``overlap`` re-includes trailing sentences of the previous chunk,
    and progress is always guaranteed.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {overlap}")
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(self, document: Document) -> list[Passage]:
        sentences = _sentence_spans(document.text)
        if not sentences:
            # No reliable sentence boundary: keep the whole document as one chunk.
            return [_make_passage(document, 0, 0, len(document.text))]

        passages: list[Passage] = []
        seen: set[tuple[int, int]] = set()
        n = len(sentences)
        index = 0
        i = 0
        while i < n:
            first_start = sentences[i][0]
            j = i
            while j + 1 < n and (sentences[j + 1][1] - first_start) <= self._chunk_size:
                j += 1
            last_end = sentences[j][1]

            key = (first_start, last_end)
            if key not in seen:
                seen.add(key)
                passages.append(_make_passage(document, index, first_start, last_end))
                index += 1

            if j + 1 >= n:
                break
            i = self._next_start(sentences, i, j)
        return passages

    def _next_start(self, sentences: Sequence[tuple[int, int]], i: int, j: int) -> int:
        """Pick the next starting sentence, honoring overlap and guaranteeing progress."""

        if self._overlap == 0:
            return j + 1
        last_end = sentences[j][1]
        candidate = j
        while candidate > i and (last_end - sentences[candidate][0]) <= self._overlap:
            candidate -= 1
        candidate += 1
        return max(candidate, i + 1)


def prepare_passages(
    documents: Iterable[Document],
    chunker: SentenceAwareChunker | WholeDocumentChunker,
) -> list[Passage]:
    """Chunk every document into passages, preserving order.

    An empty document iterable yields an empty list. Passage identifiers are
    unique as long as document identifiers are unique.
    """

    passages: list[Passage] = []
    for document in documents:
        passages.extend(chunker.chunk(document))
    return passages


__all__ = [
    "SentenceAwareChunker",
    "WholeDocumentChunker",
    "prepare_passages",
]
