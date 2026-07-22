"""Pure-Python Okapi BM25 sparse retriever.

This retriever requires no third-party library and works in a core-only install.
Score semantics are explicit: the score is the raw Okapi BM25 score (higher is
more relevant, not normalized to any range). Only passages matching at least one
query term are returned; a query that produces no tokens returns no results.
Ordering is deterministic — descending by score, then ascending by passage id.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from egrag.adapters.retrieval.base import BaseRetriever, RankStats, ScoredPassage
from egrag.adapters.retrieval.tokenization import Tokenizer, default_tokenizer
from egrag.domain.models import Passage, Query


class BM25Retriever(BaseRetriever):
    """In-memory Okapi BM25 retriever behind the ``Retriever`` protocol."""

    name = "bm25"

    def __init__(
        self,
        corpus: Sequence[Passage],
        *,
        tokenizer: Tokenizer = default_tokenizer,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 < 0:
            raise ValueError(f"k1 must be >= 0, got {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b must be in [0, 1], got {b}")
        self._passages = list(corpus)
        self._tokenizer = tokenizer
        self._k1 = k1
        self._b = b

        doc_tokens = [tokenizer(passage.text) for passage in self._passages]
        self._term_freqs = [Counter(tokens) for tokens in doc_tokens]
        self._doc_len = [len(tokens) for tokens in doc_tokens]
        self._avgdl = (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0

        document_freq: Counter[str] = Counter()
        for tokens in doc_tokens:
            document_freq.update(set(tokens))
        n_docs = len(self._passages)
        self._idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in document_freq.items()
        }

    def _score(self, query_terms: set[str], index: int) -> float:
        term_freq = self._term_freqs[index]
        doc_len = self._doc_len[index]
        score = 0.0
        for term in query_terms:
            freq = term_freq.get(term, 0)
            if freq == 0:
                continue
            idf = self._idf.get(term, 0.0)
            if self._avgdl > 0.0:
                denom = freq + self._k1 * (1.0 - self._b + self._b * doc_len / self._avgdl)
            else:
                denom = freq + self._k1
            score += idf * (freq * (self._k1 + 1.0)) / denom
        return score

    def _rank(self, query: Query, top_k: int) -> tuple[list[ScoredPassage], RankStats]:
        query_terms = set(self._tokenizer(query.text))
        if not query_terms:
            return [], RankStats()
        scored: list[ScoredPassage] = []
        for index, passage in enumerate(self._passages):
            value = self._score(query_terms, index)
            if value <= 0.0:
                continue
            scored.append(
                ScoredPassage(
                    passage=passage,
                    score=value,
                    components={self.name: value},
                    retriever=self.name,
                )
            )
        scored.sort(key=lambda item: (-item.score, item.passage.passage_id))
        return scored, RankStats()


__all__ = ["BM25Retriever"]
