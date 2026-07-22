"""Dense retrieval: cosine search over an in-memory embedding index.

The retriever is generic over any :class:`EmbeddingProvider`. Corpus embeddings
are cached by content hash so unchanged text is never re-embedded, and a changed
passage invalidates only its own cache entry. Embeddings are produced in a single
batched call per uncached set. Ranking is deterministic (descending cosine, then
ascending passage id). No model is loaded or downloaded at import time.
"""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Sequence
from time import perf_counter
from typing import Any

from egrag.adapters.retrieval.base import (
    BaseRetriever,
    RankStats,
    ScoredPassage,
    cosine_similarity,
)
from egrag.domain.models import Passage, Query
from egrag.domain.ports import Embedding, EmbeddingProvider


def _provider_name(provider: EmbeddingProvider) -> str:
    return str(getattr(provider, "name", type(provider).__name__))


class EmbeddingCache:
    """Content-addressed embedding cache keyed by provider name and text hash.

    The cache holds no global state; each instance owns its store. Hits and
    misses are reported per call so callers can record cache effectiveness.
    """

    def __init__(self) -> None:
        self._store: dict[str, Embedding] = {}

    @staticmethod
    def _key(provider_name: str, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{provider_name}:{digest}"

    def embed(
        self, texts: Sequence[str], provider: EmbeddingProvider
    ) -> tuple[list[Embedding], int, int]:
        """Return embeddings for ``texts`` plus (hits, misses).

        Cached entries are reused; only uncached texts are embedded, in one
        batched call to the provider.
        """

        provider_name = _provider_name(provider)
        keys = [self._key(provider_name, text) for text in texts]
        result: list[Embedding | None] = [None] * len(texts)
        missing_texts: list[str] = []
        missing_positions: list[int] = []
        hits = 0
        for position, key in enumerate(keys):
            cached = self._store.get(key)
            if cached is not None:
                result[position] = cached
                hits += 1
            else:
                missing_texts.append(texts[position])
                missing_positions.append(position)
        misses = len(missing_texts)
        if missing_texts:
            embedded = provider.embed(missing_texts)
            if len(embedded) != len(missing_texts):
                raise ValueError("embedding provider returned a mismatched number of vectors")
            for offset, position in enumerate(missing_positions):
                vector = embedded[offset]
                self._store[keys[position]] = vector
                result[position] = vector
        return [vector for vector in result if vector is not None], hits, misses


class DenseRetriever(BaseRetriever):
    """Cosine-similarity dense retriever over an in-memory corpus."""

    name = "dense"

    def __init__(
        self,
        corpus: Sequence[Passage],
        embedder: EmbeddingProvider,
        *,
        cache: EmbeddingCache | None = None,
    ) -> None:
        self._passages = list(corpus)
        self._embedder = embedder
        self._cache = cache if cache is not None else EmbeddingCache()

    def _rank(self, query: Query, top_k: int) -> tuple[list[ScoredPassage], RankStats]:
        start = perf_counter()
        corpus_texts = [passage.text for passage in self._passages]
        corpus_vectors, corpus_hits, corpus_misses = self._cache.embed(corpus_texts, self._embedder)
        query_vectors, query_hits, query_misses = self._cache.embed([query.text], self._embedder)
        embedding_ms = (perf_counter() - start) * 1000.0
        query_vector = query_vectors[0]

        scored: list[ScoredPassage] = []
        for passage, vector in zip(self._passages, corpus_vectors, strict=True):
            similarity = cosine_similarity(query_vector, vector)
            scored.append(
                ScoredPassage(
                    passage=passage,
                    score=round(similarity, 6),
                    components={self.name: round(similarity, 6)},
                    retriever=self.name,
                )
            )
        scored.sort(key=lambda item: (-item.score, item.passage.passage_id))
        stats = RankStats(
            cache_hits=corpus_hits + query_hits,
            cache_misses=corpus_misses + query_misses,
            embedding_ms=embedding_ms,
        )
        return scored, stats


class SentenceTransformerEmbeddingProvider:
    """Optional embedding provider backed by sentence-transformers.

    Requires the ``dense`` extra. The model is loaded lazily on first use — never
    at import time — so importing this module pulls in no heavy dependency.
    """

    def __init__(self, model_name: str, *, batch_size: int = 32) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: Any = None
        self.name = f"sentence-transformers:{model_name}"

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                module: Any = importlib.import_module("sentence_transformers")
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed; install the 'dense' extra"
                ) from exc
            self._model = module.SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        model = self._ensure_model()
        vectors = model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=False,
            convert_to_numpy=True,
        )
        return [tuple(float(value) for value in row) for row in vectors]


__all__ = [
    "DenseRetriever",
    "EmbeddingCache",
    "SentenceTransformerEmbeddingProvider",
]
