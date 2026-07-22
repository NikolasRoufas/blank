"""Optional cross-encoder reranker (sentence-transformers ``CrossEncoder``).

Requires the ``dense`` extra. The model is loaded lazily on first use — never at
import time. Scoring is batched into a single model call over all (query,
passage) pairs. Ordering is deterministic (descending score, then ascending
passage id); passage identity and provenance are preserved.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

from egrag.domain.models import Passage, Query


class CrossEncoderReranker:
    """Reranks passages using a cross-encoder relevance model."""

    def __init__(self, model_name: str, *, batch_size: int = 32) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: Any = None
        self.name = f"cross-encoder:{model_name}"

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                module: Any = importlib.import_module("sentence_transformers")
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed; install the 'dense' extra"
                ) from exc
            self._model = module.CrossEncoder(self._model_name)
        return self._model

    def rerank(self, query: Query, passages: Sequence[Passage]) -> list[Passage]:
        if not passages:
            return []
        model = self._ensure_model()
        pairs = [(query.text, passage.text) for passage in passages]
        scores = model.predict(pairs, batch_size=self._batch_size)
        ordered = sorted(
            zip(passages, (float(score) for score in scores), strict=True),
            key=lambda pair: (-pair[1], pair[0].passage_id),
        )
        return [
            passage.model_copy(update={"retrieval_score": score, "rank": index})
            for index, (passage, score) in enumerate(ordered)
        ]


__all__ = ["CrossEncoderReranker"]
