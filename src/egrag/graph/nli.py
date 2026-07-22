"""NLI support utilities: label-mapping validation, directional classification,
and the documented relation precedence policy.

These are classifier-agnostic (they operate through the ``PairClassifier``
protocol), so they are testable offline with a deterministic fake and require no
real model. The real :class:`HuggingFaceNLIClassifier` plugs in unchanged.

Relation precedence (premise → hypothesis NLI scores, given thresholds):

1. **DUPLICATE_OF** — mutual entailment in both directions ≥ ``duplicate_threshold``
   (near-identical/paraphrase). Duplicates preserve provenance and do NOT count as
   independent support.
2. **SUPPORTS** — entailment in at least one direction ≥ ``entailment_threshold``
   (and not a duplicate). Direction points from the entailing premise to the
   entailed hypothesis.
3. **CONTRADICTS** — contradiction ≥ ``contradiction_threshold`` (symmetric, per
   the graph's contradiction policy).
4. **NEUTRAL** — otherwise; no canonical edge is stored.

This mirrors the thresholds in :class:`egrag.graph.types.ClassificationConfig`
(``duplicate_threshold`` defaults to 0.8); it does not change them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from egrag.domain.models import (
    AtomicClaim,
    ClaimProvenance,
    SourceMetadata,
    SourceSpan,
)
from egrag.graph.types import ClaimPair, PairClassifier, RelationProbabilities

LABEL_MAPPING_VERSION = "1"

RelationName = Literal["duplicate", "supports", "contradicts", "neutral"]


class NLILabelMappingError(RuntimeError):
    """Raised when a classifier's label mapping fails controlled validation."""


@dataclass(frozen=True)
class ControlledCase:
    premise: str
    hypothesis: str
    expected: Literal["entailment", "contradiction", "neutral"]


# Controlled examples used to validate a classifier's label mapping at startup.
LABEL_VALIDATION_CASES: tuple[ControlledCase, ...] = (
    ControlledCase(
        "Google acquired DeepMind in 2014.",
        "DeepMind was acquired by Google in 2014.",
        "entailment",
    ),
    ControlledCase(
        "The project deadline is June 30.",
        "The project deadline is not June 30.",
        "contradiction",
    ),
    ControlledCase(
        "The project deadline is June 30.",
        "The project manager works in London.",
        "neutral",
    ),
)


def _case_claim(text: str, cid: str) -> AtomicClaim:
    return AtomicClaim(
        claim_id=cid,
        text=text,
        provenance=ClaimProvenance(
            source=SourceMetadata(source_id=cid),
            spans=(SourceSpan(source_id=cid, start=0, end=len(text), text=text),),
        ),
        extraction_confidence=1.0,
    )


def _argmax_label(prob: RelationProbabilities) -> str:
    scores = {
        "entailment": prob.entailment,
        "contradiction": prob.contradiction,
        "neutral": prob.neutral,
    }
    return max(scores, key=lambda k: scores[k])


def validate_label_mapping(classifier: PairClassifier, *, raise_on_error: bool = True) -> list[str]:
    """Validate a classifier's label mapping with controlled examples.

    Runs the entailment/contradiction/neutral cases and checks that the predicted
    argmax label matches the expected one. Returns a list of human-readable issues
    (empty when correct). With ``raise_on_error`` (default) any mismatch raises
    :class:`NLILabelMappingError` — a hard startup guard against a mis-ordered or
    swapped label mapping. Does not rely on config field ordering.
    """

    pairs = [
        ClaimPair(source=_case_claim(c.premise, f"p{i}"), target=_case_claim(c.hypothesis, f"h{i}"))
        for i, c in enumerate(LABEL_VALIDATION_CASES)
    ]
    probs = classifier.classify(pairs)
    issues: list[str] = []
    if len(probs) != len(pairs):
        issues.append(f"classifier returned {len(probs)} results for {len(pairs)} pairs")
    else:
        for case, prob in zip(LABEL_VALIDATION_CASES, probs, strict=True):
            predicted = _argmax_label(prob)
            if predicted != case.expected:
                issues.append(
                    f"label mapping mismatch: premise={case.premise!r} "
                    f"hypothesis={case.hypothesis!r} expected={case.expected} got={predicted}"
                )
    if issues and raise_on_error:
        raise NLILabelMappingError("; ".join(issues))
    return issues


@dataclass(frozen=True)
class RelationDecision:
    """The relation chosen for a pair plus the directional NLI scores behind it."""

    relation: RelationName
    # direction: which claim is the premise (source) for a SUPPORTS edge; None if
    # symmetric (duplicate/contradicts) or neutral.
    source_first: bool | None
    entailment_ab: float
    entailment_ba: float
    contradiction: float


def decide_relation(
    entailment_ab: float,
    entailment_ba: float,
    contradiction: float,
    *,
    entailment_threshold: float = 0.5,
    contradiction_threshold: float = 0.5,
    duplicate_threshold: float = 0.8,
) -> RelationDecision:
    """Apply the documented precedence policy to directional NLI scores."""

    if min(entailment_ab, entailment_ba) >= duplicate_threshold:
        relation: RelationName = "duplicate"
        source_first: bool | None = None
    elif max(entailment_ab, entailment_ba) >= entailment_threshold:
        relation = "supports"
        source_first = entailment_ab >= entailment_ba
    elif contradiction >= contradiction_threshold:
        relation = "contradicts"
        source_first = None
    else:
        relation = "neutral"
        source_first = None
    return RelationDecision(
        relation=relation,
        source_first=source_first,
        entailment_ab=entailment_ab,
        entailment_ba=entailment_ba,
        contradiction=contradiction,
    )


@dataclass(frozen=True)
class ContradictionGateDecision:
    """Trace for a structural contradiction acceptance check."""

    accepted: bool
    nli_contradiction: float
    shared_entity: bool
    shared_subject: bool
    rejection_reason: str | None = None


def _claim_entities(claim: AtomicClaim) -> set[str]:
    if claim.semantics and claim.semantics.named_entities:
        return {e.casefold() for e in claim.semantics.named_entities}
    return set()


def structural_contradiction_ok(a: AtomicClaim, b: AtomicClaim) -> ContradictionGateDecision:
    """Require shared proposition alignment before accepting a contradiction.

    A real-NLI contradiction is accepted only when the two claims share an entity
    or the same subject (a shared proposition to disagree about). Unrelated claims
    with a high NLI contradiction score (a common false positive on short text) are
    rejected structurally — without changing any NLI threshold.
    """

    shared_entity = bool(_claim_entities(a) & _claim_entities(b))
    sa = (a.semantics.subject or "").casefold() if a.semantics else ""
    sb = (b.semantics.subject or "").casefold() if b.semantics else ""
    shared_subject = bool(sa) and sa == sb
    accepted = shared_entity or shared_subject
    return ContradictionGateDecision(
        accepted=accepted,
        nli_contradiction=0.0,
        shared_entity=shared_entity,
        shared_subject=shared_subject,
        rejection_reason=None if accepted else "no shared entity or subject (unrelated claims)",
    )


class StructuralContradictionGate:
    """Wraps a classifier; demotes structurally-unjustified contradictions to neutral.

    Reduces NLI contradiction false positives on unrelated claims **without**
    altering NLI thresholds. Implements the ``PairClassifier`` protocol.
    """

    classifier_id = "structural-contradiction-gate"
    classifier_version = "1.0.0"
    model_revision: str | None = None

    def __init__(self, base: PairClassifier, *, contradiction_threshold: float = 0.5) -> None:
        self._base = base
        self._threshold = contradiction_threshold

    def classify(self, pairs: Sequence[ClaimPair]) -> list[RelationProbabilities]:
        out: list[RelationProbabilities] = []
        for pair, pr in zip(pairs, self._base.classify(list(pairs)), strict=True):
            if (
                pr.contradiction >= self._threshold
                and not structural_contradiction_ok(pair.source, pair.target).accepted
            ):
                # demote to neutral; preserve total mass
                out.append(
                    RelationProbabilities(
                        entailment=pr.entailment,
                        contradiction=0.0,
                        neutral=round(pr.neutral + pr.contradiction, 6),
                    )
                )
            else:
                out.append(pr)
        return out


def classify_directional(
    classifier: PairClassifier,
    claim_a: AtomicClaim,
    claim_b: AtomicClaim,
    *,
    entailment_threshold: float = 0.5,
    contradiction_threshold: float = 0.5,
    duplicate_threshold: float = 0.8,
) -> RelationDecision:
    """Classify both directions (a→b, b→a) and apply the precedence policy.

    NLI entailment is directional, so both orderings are evaluated; contradiction
    is taken as the stronger of the two directions (symmetric graph policy). A
    single batched ``classify`` call covers both ordered pairs.
    """

    probs = classifier.classify(
        [ClaimPair(source=claim_a, target=claim_b), ClaimPair(source=claim_b, target=claim_a)]
    )
    ab, ba = probs[0], probs[1]
    return decide_relation(
        ab.entailment,
        ba.entailment,
        max(ab.contradiction, ba.contradiction),
        entailment_threshold=entailment_threshold,
        contradiction_threshold=contradiction_threshold,
        duplicate_threshold=duplicate_threshold,
    )


__all__ = [
    "LABEL_MAPPING_VERSION",
    "LABEL_VALIDATION_CASES",
    "ContradictionGateDecision",
    "ControlledCase",
    "NLILabelMappingError",
    "RelationDecision",
    "RelationName",
    "StructuralContradictionGate",
    "classify_directional",
    "decide_relation",
    "structural_contradiction_ok",
    "validate_label_mapping",
]
