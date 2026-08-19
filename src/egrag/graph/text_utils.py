"""Deterministic text helpers for candidate generation and duplicate detection."""

from __future__ import annotations

import re

from egrag.domain.models import AtomicClaim

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_WHITESPACE_RE = re.compile(r"\s+")


def tokens(text: str) -> set[str]:
    """Return the set of lowercased alphanumeric tokens in ``text``."""

    return set(_TOKEN_RE.findall(text.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    """Return the Jaccard overlap of two token sets (0.0 when both empty)."""

    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def normalize(text: str) -> str:
    """Collapse whitespace, strip, and casefold for normalized comparison."""

    return _WHITESPACE_RE.sub(" ", text).strip().casefold()


def claim_entities(claim: AtomicClaim) -> set[str]:
    """Return a claim's named entities (casefolded), or token fallback."""

    if claim.semantics is not None and claim.semantics.named_entities:
        return {entity.casefold() for entity in claim.semantics.named_entities}
    return set()


def subject_predicate_compatible(a: AtomicClaim, b: AtomicClaim) -> bool:
    """True when both claims have the same non-empty extracted subject.

    Shared by candidate-pair generation (a pairing signal) and, optionally,
    contradiction-edge materialization (a precondition): two claims that only
    share an entity mention -- not the same subject -- are not necessarily
    about the same proposition, and an NLI "contradiction" between them is not
    evidence they disagree about anything.
    """

    if a.semantics is None or b.semantics is None:
        return False
    sa = (a.semantics.subject or "").casefold()
    sb = (b.semantics.subject or "").casefold()
    return bool(sa) and sa == sb


__all__ = ["claim_entities", "jaccard", "normalize", "subject_predicate_compatible", "tokens"]
