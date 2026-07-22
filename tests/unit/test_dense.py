"""Unit tests for dense retrieval and embedding caching (acceptance 2, 3, 10, 19, 20)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from egrag.adapters.retrieval.dense import DenseRetriever, EmbeddingCache
from egrag.domain.models import Passage, Query, SourceSpan
from egrag.domain.ports import Embedding
from egrag.fakes import FakeEmbeddingProvider

VOCAB = ["alpha", "beta", "gamma", "delta"]


def _passage(passage_id: str, text: str) -> Passage:
    return Passage(
        passage_id=passage_id,
        document_id="doc",
        text=text,
        span=SourceSpan(source_id="src", start=0, end=len(text), text=text),
    )


class CountingEmbedder:
    """Wraps an embedder and records how many texts it actually embeds."""

    def __init__(self, inner: FakeEmbeddingProvider) -> None:
        self._inner = inner
        self.embedded: list[str] = []
        self.name = inner.name

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        self.embedded.extend(texts)
        return self._inner.embed(texts)


@pytest.mark.unit
def test_dense_known_order_with_deterministic_embeddings() -> None:
    """Acceptance 2: dense retrieval produces a known order with fake embeddings."""

    embedder = FakeEmbeddingProvider(vocabulary=VOCAB)
    corpus = [
        _passage("p-exact", "alpha beta"),
        _passage("p-partial", "alpha gamma"),
        _passage("p-none", "delta"),
    ]
    retriever = DenseRetriever(corpus, embedder)
    results = retriever.search(Query(query_id="q", text="alpha beta"), top_k=3)
    ids = [item.passage.passage_id for item in results]
    assert ids == ["p-exact", "p-partial", "p-none"]
    assert results[0].score == pytest.approx(1.0)
    assert results[2].score == pytest.approx(0.0)


@pytest.mark.unit
def test_dense_retrieval_is_repeatable() -> None:
    """Acceptance 3: repeated deterministic dense retrieval is identical."""

    embedder = FakeEmbeddingProvider(vocabulary=VOCAB)
    corpus = [_passage("p1", "alpha beta"), _passage("p2", "gamma delta")]
    retriever = DenseRetriever(corpus, embedder)
    query = Query(query_id="q", text="alpha gamma")
    first = retriever.search(query, top_k=2)
    second = retriever.search(query, top_k=2)
    assert first == second


@pytest.mark.unit
def test_dense_empty_corpus_returns_empty() -> None:
    """Acceptance 10: an empty corpus returns an empty result."""

    retriever = DenseRetriever([], FakeEmbeddingProvider(vocabulary=VOCAB))
    assert retriever.search(Query(query_id="q", text="alpha"), top_k=5) == []


@pytest.mark.unit
def test_repeated_retrieval_does_not_recompute_corpus() -> None:
    """Acceptance 19: a second retrieval re-embeds nothing (full cache hit)."""

    embedder = CountingEmbedder(FakeEmbeddingProvider(vocabulary=VOCAB))
    corpus = [_passage("p1", "alpha beta"), _passage("p2", "gamma delta")]
    retriever = DenseRetriever(corpus, embedder)
    query = Query(query_id="q", text="alpha")

    retriever.search(query, top_k=2)
    embedded_after_first = len(embedder.embedded)
    retriever.search(query, top_k=2)
    assert len(embedder.embedded) == embedded_after_first  # nothing re-embedded


@pytest.mark.unit
def test_changed_corpus_invalidates_only_changed_entry() -> None:
    """Acceptance 20: changing one passage re-embeds only that passage."""

    embedder = CountingEmbedder(FakeEmbeddingProvider(vocabulary=VOCAB))
    cache = EmbeddingCache()
    corpus = [_passage("p1", "alpha beta"), _passage("p2", "gamma delta")]
    query = Query(query_id="q", text="alpha")

    DenseRetriever(corpus, embedder, cache=cache).search(query, top_k=2)
    embedded_after_first = len(embedder.embedded)

    changed = [_passage("p1", "alpha beta"), _passage("p2", "gamma epsilon")]
    report = DenseRetriever(changed, embedder, cache=cache).search_report(query, top_k=2)

    # Exactly one new text ("gamma epsilon") was embedded; the rest were cached.
    assert len(embedder.embedded) - embedded_after_first == 1
    assert report.stats.cache_misses == 1


@pytest.mark.unit
def test_cache_reports_hits_and_misses() -> None:
    embedder = FakeEmbeddingProvider(vocabulary=VOCAB)
    corpus = [_passage("p1", "alpha"), _passage("p2", "beta")]
    cache = EmbeddingCache()
    retriever = DenseRetriever(corpus, embedder, cache=cache)
    # Query text is distinct from both passages, so it is its own cache entry.
    query = Query(query_id="q", text="gamma")

    first = retriever.search_report(query, top_k=2)
    assert first.stats.cache_misses == 3  # two passages + query
    assert first.stats.cache_hits == 0

    second = retriever.search_report(query, top_k=2)
    assert second.stats.cache_hits == 3
    assert second.stats.cache_misses == 0
