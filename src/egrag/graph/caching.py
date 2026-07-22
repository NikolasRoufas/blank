"""Persistent caching wrapper for a pair classifier (NLI).

Wraps any :class:`PairClassifier` and memoizes per-pair relation probabilities
behind :func:`egrag.caching.build_nli_cache_key`, which incorporates the premise
and hypothesis content, the model id, model/tokenizer revisions, the label
mapping version, truncation/length, the NLI thresholds, and the schema version —
so a change to any of these is a cache miss. The cached value is an explicit,
versioned JSON encoding of the three probabilities, so a cold run and a warm run
return identical results.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from egrag.caching.keys import build_nli_cache_key
from egrag.domain.ports import CacheBackend
from egrag.graph.types import ClaimPair, PairClassifier, RelationProbabilities

_VALUE_VERSION = 1


def _encode(prob: RelationProbabilities) -> str:
    return json.dumps(
        {
            "v": _VALUE_VERSION,
            "entailment": prob.entailment,
            "contradiction": prob.contradiction,
            "neutral": prob.neutral,
        }
    )


def _decode(value: str) -> RelationProbabilities | None:
    try:
        data = json.loads(value)
        if data.get("v") != _VALUE_VERSION:
            return None
        return RelationProbabilities(
            entailment=data["entailment"],
            contradiction=data["contradiction"],
            neutral=data["neutral"],
        )
    except (ValueError, KeyError, TypeError):
        return None


class CachedPairClassifier:
    """A :class:`PairClassifier` that memoizes per-pair probabilities."""

    def __init__(
        self,
        inner: PairClassifier,
        cache: CacheBackend,
        *,
        model: str,
        entailment_threshold: float,
        contradiction_threshold: float,
        duplicate_threshold: float,
        model_revision: str | None = None,
        tokenizer_revision: str | None = None,
        label_mapping_version: str = "1",
        max_length: int | None = None,
        truncation: bool = True,
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._model = model
        self._model_revision = model_revision
        self._tokenizer_revision = tokenizer_revision
        self._label_mapping_version = label_mapping_version
        self._max_length = max_length
        self._truncation = truncation
        self._entail = entailment_threshold
        self._contra = contradiction_threshold
        self._dup = duplicate_threshold
        # PairClassifier protocol metadata (provenance).
        self.classifier_id = f"cached:{getattr(inner, 'classifier_id', 'pair-classifier')}"
        self.classifier_version = getattr(inner, "classifier_version", model)
        self.model_revision = model_revision

    def _key(self, pair: ClaimPair) -> str:
        return build_nli_cache_key(
            pair.source.text,
            pair.target.text,
            model=self._model,
            model_revision=self._model_revision,
            tokenizer_revision=self._tokenizer_revision,
            label_mapping_version=self._label_mapping_version,
            max_length=self._max_length,
            truncation=self._truncation,
            entailment_threshold=self._entail,
            contradiction_threshold=self._contra,
            duplicate_threshold=self._dup,
        )

    def classify(self, pairs: Sequence[ClaimPair]) -> list[RelationProbabilities]:
        results: list[RelationProbabilities | None] = [None] * len(pairs)
        misses: list[ClaimPair] = []
        miss_slots: list[tuple[int, str]] = []
        for i, pair in enumerate(pairs):
            key = self._key(pair)
            cached = self._cache.get(key)
            decoded = _decode(cached) if cached is not None else None
            if decoded is not None:
                results[i] = decoded
            else:
                misses.append(pair)
                miss_slots.append((i, key))
        if misses:
            fresh = self._inner.classify(misses)
            for (i, key), prob in zip(miss_slots, fresh, strict=True):
                results[i] = prob
                self._cache.set(key, _encode(prob))
        return [r for r in results if r is not None]


__all__ = ["CachedPairClassifier"]
