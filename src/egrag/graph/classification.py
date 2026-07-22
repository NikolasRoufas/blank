"""Relation classifiers returning entailment/contradiction/neutral probabilities.

* :class:`LexicalPairClassifier` — a deterministic, dependency-free rule-based
  baseline (token overlap + negation parity).
* :class:`HuggingFaceNLIClassifier` — an optional NLI adapter loaded lazily
  (``local-models`` extra); never imported or downloaded at import time.

Both return one :class:`RelationProbabilities` per input pair, in order, and
expose model/revision metadata. The deterministic fake lives in ``egrag.fakes``.
"""

from __future__ import annotations

import importlib
from typing import Any

from egrag.graph.text_utils import jaccard, tokens
from egrag.graph.types import ClaimPair, RelationProbabilities

_NEGATION = frozenset({"not", "no", "never", "cannot", "without", "neither", "nor", "none"})


def _negated(pair_text: str, semantics_negation: bool | None) -> bool:
    if semantics_negation:
        return True
    return bool(tokens(pair_text) & _NEGATION)


class LexicalPairClassifier:
    """Rule-based baseline: token overlap with negation-parity contradiction.

    Note: lexical overlap is symmetric, so this baseline is not truly
    directional; use the deterministic fake classifier for directional tests.
    """

    classifier_id = "lexical-nli"
    classifier_version = "1.0.0"
    model_revision: str | None = None

    def __init__(self, *, min_overlap: float = 0.2) -> None:
        self._min_overlap = min_overlap

    def classify(self, pairs: list[ClaimPair]) -> list[RelationProbabilities]:
        results: list[RelationProbabilities] = []
        for pair in pairs:
            ta, tb = tokens(pair.source.text), tokens(pair.target.text)
            content_a, content_b = ta - _NEGATION, tb - _NEGATION
            overlap = jaccard(content_a, content_b)
            neg_a = _negated(
                pair.source.text, pair.source.semantics.negation if pair.source.semantics else None
            )
            neg_b = _negated(
                pair.target.text, pair.target.semantics.negation if pair.target.semantics else None
            )
            if overlap < self._min_overlap:
                results.append(
                    RelationProbabilities(entailment=0.0, contradiction=0.0, neutral=1.0)
                )
            elif neg_a != neg_b:
                results.append(
                    RelationProbabilities(
                        entailment=0.0,
                        contradiction=round(overlap, 6),
                        neutral=round(1.0 - overlap, 6),
                    )
                )
            else:
                results.append(
                    RelationProbabilities(
                        entailment=round(overlap, 6),
                        contradiction=0.0,
                        neutral=round(1.0 - overlap, 6),
                    )
                )
        return results


class HuggingFaceNLIClassifier:
    """Optional NLI classifier backed by a Transformers text-classification model.

    Requires the ``local-models`` extra. The model is loaded lazily on first use.
    Inference is batched and results preserve input order.
    """

    classifier_id = "huggingface-nli"

    def __init__(
        self,
        model_name: str,
        *,
        model_revision: str | None = None,
        tokenizer_revision: str | None = None,
        batch_size: int = 16,
        device: str | int | None = None,
        max_length: int = 256,
        truncation: bool = True,
        label_mapping_version: str = "1",
    ) -> None:
        self._model_name = model_name
        self.classifier_version = model_name
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision
        self._batch_size = batch_size
        self._device = device
        self._max_length = max_length
        self._truncation = truncation
        self.label_mapping_version = label_mapping_version
        self._pipeline: Any = None

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is None:
            try:
                transformers: Any = importlib.import_module("transformers")
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "transformers is not installed; install the 'local-models' extra"
                ) from exc
            kwargs: dict[str, Any] = {"revision": self.model_revision, "top_k": None}
            if self._device is not None:
                kwargs["device"] = self._device
            self._pipeline = transformers.pipeline(
                "text-classification", model=self._model_name, **kwargs
            )
        return self._pipeline

    def classify(self, pairs: list[ClaimPair]) -> list[RelationProbabilities]:
        if not pairs:
            return []
        pipeline = self._ensure_pipeline()
        inputs = [{"text": p.source.text, "text_pair": p.target.text} for p in pairs]
        results: list[RelationProbabilities] = []
        for start in range(0, len(inputs), self._batch_size):
            batch = inputs[start : start + self._batch_size]
            for scored in pipeline(
                batch,
                batch_size=self._batch_size,
                truncation=self._truncation,
                max_length=self._max_length,
            ):
                label_scores = {entry["label"].lower(): float(entry["score"]) for entry in scored}
                results.append(
                    RelationProbabilities(
                        entailment=label_scores.get("entailment", 0.0),
                        contradiction=label_scores.get("contradiction", 0.0),
                        neutral=label_scores.get("neutral", 0.0),
                    )
                )
        return results


__all__ = ["HuggingFaceNLIClassifier", "LexicalPairClassifier"]
