"""Retrieval adapters: chunking, sparse (BM25), dense, and hybrid retrieval.

Importing this package pulls in no optional dependency. The dense module's
sentence-transformers provider loads its backend lazily on first use.
"""

from __future__ import annotations

from egrag.adapters.retrieval.base import (
    BaseRetriever,
    FusionStrategy,
    RetrievalReport,
    RetrievalStats,
    ScoredPassage,
)
from egrag.adapters.retrieval.bm25 import BM25Retriever
from egrag.adapters.retrieval.chunking import (
    SentenceAwareChunker,
    WholeDocumentChunker,
    prepare_passages,
)
from egrag.adapters.retrieval.dense import (
    DenseRetriever,
    EmbeddingCache,
    SentenceTransformerEmbeddingProvider,
)
from egrag.adapters.retrieval.hybrid import HybridRetriever
from egrag.adapters.retrieval.tokenization import Tokenizer, default_tokenizer

__all__ = [
    "BM25Retriever",
    "BaseRetriever",
    "DenseRetriever",
    "EmbeddingCache",
    "FusionStrategy",
    "HybridRetriever",
    "RetrievalReport",
    "RetrievalStats",
    "ScoredPassage",
    "SentenceAwareChunker",
    "SentenceTransformerEmbeddingProvider",
    "Tokenizer",
    "WholeDocumentChunker",
    "default_tokenizer",
    "prepare_passages",
]
