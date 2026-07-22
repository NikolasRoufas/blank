"""Deterministic post-processing helpers shared by extractors.

Covers whitespace normalization, deterministic claim IDs, source-span discovery
and verification, sentence segmentation, and same-source deduplication. None of
these helpers mutate their inputs.
"""

from __future__ import annotations

import hashlib
import re

from egrag.domain.models import AtomicClaim, Passage, SourceSpan

_WHITESPACE_RE = re.compile(r"\s+")
# A sentence boundary is a run of terminators followed by whitespace or the end
# of the text. Requiring a following space (or EOS) means a period inside a
# decimal such as "2.5" is NOT treated as a boundary.
_BOUNDARY_RE = re.compile(r"[.!?]+(?=\s|$)")


def normalize_text(text: str) -> str:
    """Collapse runs of whitespace to single spaces and strip the ends."""

    return _WHITESPACE_RE.sub(" ", text).strip()


def claim_id(passage_id: str, normalized_text: str, start: int) -> str:
    """Return a stable, deterministic claim identifier.

    The ID is a function of the passage, the normalized claim text, and the span
    start offset, so identical inputs always yield the same ID.
    """

    digest = hashlib.sha256(f"{passage_id}\x00{normalized_text}\x00{start}".encode())
    return f"clm-{digest.hexdigest()[:16]}"


def _trimmed_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    segment = text[start:end]
    leading = len(segment) - len(segment.lstrip())
    trailing = len(segment) - len(segment.rstrip())
    new_start = start + leading
    new_end = end - trailing
    return (new_start, new_end) if new_end > new_start else None


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) offsets of sentence content, trimmed of whitespace.

    Sentences are split on terminators (``.!?``) that are followed by whitespace
    or end-of-string, so decimals and similar in-token periods are preserved.
    """

    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _BOUNDARY_RE.finditer(text):
        trimmed = _trimmed_span(text, cursor, match.end())
        if trimmed is not None:
            spans.append(trimmed)
        cursor = match.end()
    if cursor < len(text):
        trimmed = _trimmed_span(text, cursor, len(text))
        if trimmed is not None:
            spans.append(trimmed)
    return spans


def find_span(passage: Passage, span_text: str) -> SourceSpan | None:
    """Locate ``span_text`` verbatim in the passage and build a SourceSpan.

    Returns ``None`` when the text does not occur in the passage (an ungrounded
    claim), so callers can reject it rather than fabricate provenance.
    """

    if not span_text:
        return None
    index = passage.text.find(span_text)
    if index < 0:
        return None
    start = passage.span.start + index
    return SourceSpan(
        source_id=passage.span.source_id,
        start=start,
        end=start + len(span_text),
        text=span_text,
    )


def verify_span(passage: Passage, span: SourceSpan) -> bool:
    """Return True if the span lies within the passage and its text matches."""

    local_start = span.start - passage.span.start
    local_end = span.end - passage.span.start
    if local_start < 0 or local_end > len(passage.text):
        return False
    return passage.text[local_start:local_end] == span.text


def deduplicate_same_source(
    claims: list[AtomicClaim],
) -> tuple[list[AtomicClaim], list[str]]:
    """Remove exact and normalized duplicates that share a source.

    Two claims are duplicates when they have the same source id and the same
    normalized claim text. Cross-source duplicates are kept so provenance-aware
    graph handling can decide later. Returns (kept_claims, warnings).
    """

    seen: set[tuple[str, str]] = set()
    kept: list[AtomicClaim] = []
    warnings: list[str] = []
    for claim in claims:
        key = (claim.provenance.source.source_id, normalize_text(claim.text).casefold())
        if key in seen:
            warnings.append(f"dropped duplicate same-source claim: {claim.text!r}")
            continue
        seen.add(key)
        kept.append(claim)
    return kept, warnings


__all__ = [
    "claim_id",
    "deduplicate_same_source",
    "find_span",
    "normalize_text",
    "sentence_spans",
    "verify_span",
]
