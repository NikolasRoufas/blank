"""Reranking adapters behind the :class:`egrag.domain.ports.Reranker` protocol.

Importing this package pulls in no optional dependency; the cross-encoder adapter
loads its backend lazily on first use.
"""

from __future__ import annotations

from egrag.adapters.reranking.cross_encoder import CrossEncoderReranker
from egrag.adapters.reranking.score import ScoreReranker

__all__ = ["CrossEncoderReranker", "ScoreReranker"]
