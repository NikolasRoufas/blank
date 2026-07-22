"""Benchmark-specific metrics (HotpotQA, FEVER).

Pure functions operating on already-extracted predictions and gold values; no
model or network. Kept separate from graph-mechanism metrics. HotpotQA answer
normalization follows the official SQuAD/HotpotQA convention (lowercase, strip
punctuation/articles, collapse whitespace).
"""

from __future__ import annotations

import re
import string
from collections import Counter

# --- answer normalization (HotpotQA / SQuAD) --------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Official-style normalization: lowercase, drop punctuation/articles/extra ws."""

    text = text.lower()
    text = text.translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, golds: tuple[str, ...]) -> float:
    norm = normalize_answer(prediction)
    return 1.0 if any(norm == normalize_answer(g) for g in golds) else 0.0


def token_f1(prediction: str, golds: tuple[str, ...]) -> float:
    """Max token-level F1 over gold aliases (official HotpotQA/SQuAD F1)."""

    best = 0.0
    pred_tokens = normalize_answer(prediction).split()
    for gold in golds:
        gold_tokens = normalize_answer(gold).split()
        # yes/no/noanswer must match exactly under official rules
        if pred_tokens == [] or gold_tokens == []:
            best = max(best, 1.0 if pred_tokens == gold_tokens else 0.0)
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        same = sum(common.values())
        if same == 0:
            continue
        precision = same / len(pred_tokens)
        recall = same / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def _prf(pred: set[object], gold: set[object]) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred or not gold:
        return 0.0, 0.0, 0.0
    tp = len(pred & gold)
    precision = tp / len(pred)
    recall = tp / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


# --- HotpotQA supporting facts / joint --------------------------------------


def supporting_fact_prf(
    pred_sp: set[tuple[str, int]], gold_sp: set[tuple[str, int]]
) -> tuple[float, float, float]:
    """Precision/recall/F1 over (title, sentence_index) supporting-fact pairs."""

    return _prf(set(pred_sp), set(gold_sp))


def hotpot_joint(
    answer_em: float,
    answer_f1: float,
    sp_prf: tuple[float, float, float],
) -> tuple[float, float]:
    """Joint EM and F1 (answer AND supporting facts), per the official metric."""

    _sp_p, _sp_r, sp_f1 = sp_prf
    joint_em = answer_em * (1.0 if sp_f1 == 1.0 else 0.0)
    # joint precision/recall multiply the answer correctness by SP precision/recall
    joint_f1 = answer_f1 * sp_f1
    return joint_em, joint_f1


# --- FEVER ------------------------------------------------------------------

FEVER_LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")


def fever_label_accuracy(prediction: str, gold: str) -> float:
    return 1.0 if prediction.strip().upper() == gold.strip().upper() else 0.0


def fever_evidence_prf(pred_pages: set[str], gold_pages: set[str]) -> tuple[float, float, float]:
    return _prf(set(pred_pages), set(gold_pages))


def evidence_set_recovered(
    pred_pages: set[str], gold_evidence_sets: tuple[frozenset[str], ...]
) -> bool:
    """True if at least one complete gold evidence set is covered (FEVER rule)."""

    if not gold_evidence_sets:
        return True  # NOT ENOUGH INFO: no evidence required
    return any(gold_set <= pred_pages for gold_set in gold_evidence_sets)


def fever_score(
    prediction: str,
    gold: str,
    pred_pages: set[str],
    gold_evidence_sets: tuple[frozenset[str], ...],
) -> float:
    """Official FEVER score: label correct AND (for S/R) a complete evidence set found.

    For NOT ENOUGH INFO only the label must be correct.
    """

    label_ok = fever_label_accuracy(prediction, gold) == 1.0
    if not label_ok:
        return 0.0
    if gold.strip().upper() == "NOT ENOUGH INFO":
        return 1.0
    return 1.0 if evidence_set_recovered(pred_pages, gold_evidence_sets) else 0.0


__all__ = [
    "FEVER_LABELS",
    "evidence_set_recovered",
    "exact_match",
    "fever_evidence_prf",
    "fever_label_accuracy",
    "fever_score",
    "hotpot_joint",
    "normalize_answer",
    "supporting_fact_prf",
    "token_f1",
]
