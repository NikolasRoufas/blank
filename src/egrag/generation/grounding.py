"""Grounding verifiers.

``BaselineGroundingVerifier`` runs deterministic, dependency-free checks and
appends warnings to the answer (it never raises for soft issues). The optional
``NLIGroundingVerifier`` adds answer-to-evidence entailment checks behind the
``local-models`` extra.

**Automated NLI does not prove factual correctness** — it only flags sentences
that an NLI model does not find entailed by the cited evidence.
"""

from __future__ import annotations

import importlib
import re
from typing import Any

from egrag.domain.models import EvidencePackage, EvidenceStatus, GeneratedAnswer
from egrag.generation.rendering import unresolved_conflict_ids

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
_CITATION_RE = re.compile(r"\[[^\]]+\]")
_IMPORTANT_MIN_WORDS = 8


class BaselineGroundingVerifier:
    """Deterministic grounding checks; implements the ``GroundingVerifier`` port."""

    def verify(self, answer: GeneratedAnswer, package: EvidencePackage) -> GeneratedAnswer:
        warnings: list[str] = list(answer.unsupported_warnings)
        known = {claim.claim_id for claim in package.claims}
        status_by_id = {item.claim_id: item.status for item in package.selected}

        for cid in answer.cited_claim_ids:
            if cid not in known:
                warnings.append(f"citation to unknown claim {cid!r}")

        if package.selected and not answer.cited_claim_ids and not answer.abstained:
            warnings.append("answer provides no citations despite available evidence")

        superseded = [
            cid
            for cid in answer.cited_claim_ids
            if status_by_id.get(cid) is EvidenceStatus.SUPERSEDED
        ]
        if superseded and not answer.uncertainty.strip():
            warnings.append(
                f"superseded evidence {sorted(superseded)} cited without temporal context"
            )

        if unresolved_conflict_ids(package) and not answer.uncertainty.strip():
            warnings.append(
                "unresolved conflicts are present but the answer expresses no uncertainty"
            )

        for sentence in self._important_sentences(answer.text):
            if not _CITATION_RE.search(sentence):
                warnings.append(f"uncited claim-like sentence: {sentence.strip()[:60]!r}")

        return answer.model_copy(update={"unsupported_warnings": tuple(warnings)})

    @staticmethod
    def _important_sentences(text: str) -> list[str]:
        sentences = [m.group().strip() for m in _SENTENCE_RE.finditer(text)]
        return [s for s in sentences if len(s.split()) >= _IMPORTANT_MIN_WORDS]


class NLIGroundingVerifier:
    """Optional answer-to-evidence NLI verifier (``local-models`` extra, lazy).

    Flags answer sentences not entailed by their cited claims. This does not
    prove factual correctness; it is an additional, fallible signal.
    """

    def __init__(self, model_name: str, *, entailment_threshold: float = 0.5) -> None:
        self._model_name = model_name
        self._threshold = entailment_threshold
        self._pipeline: Any = None

    def _ensure(self) -> Any:
        if self._pipeline is None:
            try:
                transformers: Any = importlib.import_module("transformers")
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "transformers is not installed; install the 'local-models' extra"
                ) from exc
            self._pipeline = transformers.pipeline(
                "text-classification", model=self._model_name, top_k=None
            )
        return self._pipeline

    def verify(self, answer: GeneratedAnswer, package: EvidencePackage) -> GeneratedAnswer:
        pipeline = self._ensure()
        claim_text = {c.claim_id: c.text for c in package.claims}
        premise = " ".join(claim_text.get(cid, "") for cid in answer.cited_claim_ids).strip()
        if not premise:
            return answer
        warnings = list(answer.unsupported_warnings)
        for sentence in BaselineGroundingVerifier._important_sentences(answer.text):
            scored = pipeline({"text": premise, "text_pair": sentence})
            labels = {entry["label"].lower(): float(entry["score"]) for entry in scored}
            if labels.get("entailment", 0.0) < self._threshold:
                warnings.append(f"NLI did not find entailment for: {sentence.strip()[:60]!r}")
        return answer.model_copy(update={"unsupported_warnings": tuple(warnings)})


__all__ = ["BaselineGroundingVerifier", "NLIGroundingVerifier"]
