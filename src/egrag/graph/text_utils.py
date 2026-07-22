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


__all__ = ["claim_entities", "jaccard", "normalize", "tokens"]
