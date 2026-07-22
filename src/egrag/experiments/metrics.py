"""Evaluation metrics, grouped by kind with documented limitations.

Three kinds are kept explicitly separate (see :data:`METRIC_KINDS`):

* ``deterministic`` — exact functions of strings/IDs (EM, token-F1, citation and
  evidence precision/recall, counts, latency). Reproducible and unambiguous.
* ``heuristic`` — lexical approximations (entailment/contradiction/coherence by
  word overlap). Cheap and deterministic but **not** semantically reliable.
* ``model_based`` — would use a learned model (e.g. NLI). Interfaces are provided
  but no model is bundled; results must be labeled and never treated as ground
  truth of correctness.

No metric uses gold labels during system generation; metrics are computed by the
runner *after* a prediction exists.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol, runtime_checkable

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_WS = re.compile(r"\s+")


class MetricKind(StrEnum):
    DETERMINISTIC = "deterministic"
    HEURISTIC = "heuristic"
    MODEL_BASED = "model_based"


# --- Text normalization & answer metrics (deterministic) ---------------------


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation/articles/extra ws."""

    lowered = text.lower()
    no_punct = "".join(ch for ch in lowered if ch not in string.punctuation)
    no_articles = _ARTICLES.sub(" ", no_punct)
    return _WS.sub(" ", no_articles).strip()


def exact_match(prediction: str, golds: Sequence[str]) -> float:
    """1.0 if the raw prediction equals any gold exactly, else 0.0."""

    return float(any(prediction == gold for gold in golds))


def normalized_exact_match(prediction: str, golds: Sequence[str]) -> float:
    """1.0 if the normalized prediction equals any normalized gold, else 0.0."""

    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(gold) for gold in golds))


def token_f1(prediction: str, golds: Sequence[str]) -> float:
    """Max token-level F1 over golds.

    Edge cases (documented): if a gold and the prediction are both empty after
    normalization, F1 is 1.0; if exactly one is empty, F1 is 0.0.
    """

    if not golds:
        return 0.0
    pred_tokens = normalize_answer(prediction).split()
    best = 0.0
    for gold in golds:
        gold_tokens = normalize_answer(gold).split()
        if not pred_tokens and not gold_tokens:
            best = max(best, 1.0)
            continue
        if not pred_tokens or not gold_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def answer_accuracy(prediction: str, golds: Sequence[str]) -> float:
    """Alias for normalized exact match (the harness's answer-correctness signal)."""

    return normalized_exact_match(prediction, golds)


def is_empty_prediction(prediction: str) -> bool:
    """A prediction is empty if it has no non-whitespace content."""

    return not prediction.strip()


# --- Set-based citation / evidence metrics (deterministic) -------------------


def _prf(predicted: set[str], gold: set[str]) -> tuple[float, float]:
    if not predicted:
        precision = 1.0 if not gold else 0.0
    else:
        precision = len(predicted & gold) / len(predicted)
    recall = 1.0 if not gold else len(predicted & gold) / len(gold)
    return precision, recall


def citation_precision(predicted: Sequence[str], gold: Sequence[str]) -> float:
    return _prf(set(predicted), set(gold))[0]


def citation_recall(predicted: Sequence[str], gold: Sequence[str]) -> float:
    return _prf(set(predicted), set(gold))[1]


def citation_completeness(predicted: Sequence[str], gold: Sequence[str]) -> float:
    """1.0 iff every gold source is cited (recall == 1.0 over non-empty gold)."""

    if not gold:
        return 1.0
    return float(set(gold) <= set(predicted))


def evidence_precision(predicted: Sequence[str], gold: Sequence[str]) -> float:
    return _prf(set(predicted), set(gold))[0]


def evidence_recall(predicted: Sequence[str], gold: Sequence[str]) -> float:
    return _prf(set(predicted), set(gold))[1]


def invalid_citation_count(citations: Sequence[str], known_claim_ids: Sequence[str]) -> int:
    """Number of cited IDs that are not present in the package's claims."""

    known = set(known_claim_ids)
    return sum(1 for c in citations if c not in known)


# --- Heuristic semantic metrics (NOT ground truth) ---------------------------


def _tokens(text: str) -> set[str]:
    return set(normalize_answer(text).split())


def lexical_entailment(hypothesis: str, premises: Sequence[str]) -> float:
    """Heuristic: fraction of hypothesis tokens covered by the premises.

    Limitation: pure lexical overlap. It cannot detect paraphrase, negation, or
    logical entailment, and must never be reported as semantic entailment.
    """

    hyp = _tokens(hypothesis)
    if not hyp:
        return 1.0
    covered = set().union(*[_tokens(p) for p in premises]) if premises else set()
    return len(hyp & covered) / len(hyp)


def heuristic_contradiction_rate(answer: str, premises: Sequence[str]) -> float:
    """Heuristic: 1.0 if the answer negates premise content by surface cues.

    Limitation: detects only explicit negation tokens near shared content; a
    crude proxy, not a real contradiction detector.
    """

    negation = {"not", "no", "never", "cannot", "false", "incorrect"}
    answer_tokens = _tokens(answer)
    if answer_tokens & negation:
        shared = answer_tokens & (
            set().union(*[_tokens(p) for p in premises]) if premises else set()
        )
        return 1.0 if shared else 0.0
    return 0.0


def unsupported_claim_rate(answer_sentences: Sequence[str], premises: Sequence[str]) -> float:
    """Heuristic: fraction of answer sentences not lexically supported (<0.5)."""

    if not answer_sentences:
        return 0.0
    unsupported = sum(1 for s in answer_sentences if lexical_entailment(s, premises) < 0.5)
    return unsupported / len(answer_sentences)


@runtime_checkable
class EntailmentMetric(Protocol):
    """Model-based answer-evidence entailment interface (no model bundled)."""

    kind: MetricKind

    def score(self, hypothesis: str, premises: Sequence[str]) -> float: ...


class HeuristicEntailmentMetric:
    """Default lexical stand-in for :class:`EntailmentMetric` (labeled heuristic)."""

    kind: MetricKind = MetricKind.HEURISTIC

    def score(self, hypothesis: str, premises: Sequence[str]) -> float:
        return lexical_entailment(hypothesis, premises)


# --- Registry of kinds & limitations -----------------------------------------

METRIC_KINDS: dict[str, MetricKind] = {
    "exact_match": MetricKind.DETERMINISTIC,
    "normalized_exact_match": MetricKind.DETERMINISTIC,
    "token_f1": MetricKind.DETERMINISTIC,
    "answer_accuracy": MetricKind.DETERMINISTIC,
    "citation_precision": MetricKind.DETERMINISTIC,
    "citation_recall": MetricKind.DETERMINISTIC,
    "citation_completeness": MetricKind.DETERMINISTIC,
    "evidence_precision": MetricKind.DETERMINISTIC,
    "evidence_recall": MetricKind.DETERMINISTIC,
    "invalid_citations": MetricKind.DETERMINISTIC,
    "empty_prediction": MetricKind.DETERMINISTIC,
    "latency_ms": MetricKind.DETERMINISTIC,
    "num_passages": MetricKind.DETERMINISTIC,
    "num_claims": MetricKind.DETERMINISTIC,
    "num_graph_nodes": MetricKind.DETERMINISTIC,
    "num_graph_edges": MetricKind.DETERMINISTIC,
    "num_selected": MetricKind.DETERMINISTIC,
    "token_estimate": MetricKind.DETERMINISTIC,
    "answer_evidence_entailment": MetricKind.HEURISTIC,
    "contradiction_rate": MetricKind.HEURISTIC,
    "unsupported_claim_rate": MetricKind.HEURISTIC,
    "conflict_resolution_accuracy": MetricKind.HEURISTIC,
    "reasoning_subgraph_coherence": MetricKind.HEURISTIC,
}

METRIC_LIMITATIONS: dict[str, str] = {
    "exact_match": "Surface string equality; sensitive to formatting and aliases.",
    "token_f1": "Bag-of-tokens overlap; ignores order, negation, and meaning.",
    "citation_recall": "Compares cited source IDs to gold; requires gold evidence.",
    "answer_evidence_entailment": "Lexical proxy; not real NLI, cannot prove support.",
    "contradiction_rate": "Surface negation cue; not a contradiction detector.",
    "unsupported_claim_rate": "Lexical support proxy at the sentence level.",
    "conflict_resolution_accuracy": "Compares chosen side to gold stance heuristically.",
    "reasoning_subgraph_coherence": "Lexical connectivity proxy, not logical validity.",
}


def metric_kind(name: str) -> MetricKind:
    """Return the kind of a metric, defaulting to heuristic for safety."""

    return METRIC_KINDS.get(name, MetricKind.HEURISTIC)


__all__ = [
    "METRIC_KINDS",
    "METRIC_LIMITATIONS",
    "EntailmentMetric",
    "HeuristicEntailmentMetric",
    "MetricKind",
    "answer_accuracy",
    "citation_completeness",
    "citation_precision",
    "citation_recall",
    "evidence_precision",
    "evidence_recall",
    "exact_match",
    "heuristic_contradiction_rate",
    "invalid_citation_count",
    "is_empty_prediction",
    "lexical_entailment",
    "metric_kind",
    "normalize_answer",
    "normalized_exact_match",
    "token_f1",
    "unsupported_claim_rate",
]
