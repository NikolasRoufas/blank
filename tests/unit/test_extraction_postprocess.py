"""Post-processing acceptance tests (cases 17, 18, 23) and helper unit tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from egrag.adapters.extraction import SentenceClaimExtractor
from egrag.adapters.extraction.postprocess import (
    claim_id,
    deduplicate_same_source,
    find_span,
    normalize_text,
    verify_span,
)
from egrag.domain.models import Passage, SourceMetadata, SourceSpan


def _passage(text: str, passage_id: str = "p1", source_id: str = "src1") -> Passage:
    return Passage(
        passage_id=passage_id,
        document_id="doc",
        text=text,
        span=SourceSpan(source_id=source_id, start=0, end=len(text), text=text),
    )


EX = SentenceClaimExtractor()


@pytest.mark.unit
def test_same_source_duplicates_deduplicated() -> None:
    """Acceptance 17: duplicate same-source claims are deduplicated."""

    result = EX.extract_result(_passage("Alice founded X. Alice founded X."))
    assert len(result.claims) == 1
    assert any("duplicate" in w for w in result.warnings)


@pytest.mark.unit
def test_cross_source_duplicates_preserve_provenance() -> None:
    """Acceptance 18: cross-source duplicates keep separate provenance."""

    from_a = EX.extract(
        _passage("Alice founded X.", passage_id="pa", source_id="src-a"),
        source=SourceMetadata(source_id="src-a"),
    )
    from_b = EX.extract(
        _passage("Alice founded X.", passage_id="pb", source_id="src-b"),
        source=SourceMetadata(source_id="src-b"),
    )
    kept, warnings = deduplicate_same_source([*from_a, *from_b])
    assert len(kept) == 2
    assert {c.provenance.source.source_id for c in kept} == {"src-a", "src-b"}
    assert warnings == []


@pytest.mark.unit
def test_invalid_source_offsets_rejected() -> None:
    """Acceptance 23: invalid source offsets are rejected."""

    # The schema rejects non-ordered offsets.
    with pytest.raises(ValidationError):
        SourceSpan(source_id="s", start=5, end=3, text="abc")

    passage = _passage("Alice founded Acme.")
    # An out-of-range span fails verification.
    out_of_range = SourceSpan(source_id="src1", start=0, end=100, text="x" * 100)
    assert verify_span(passage, out_of_range) is False
    # Text absent from the passage yields no span (no fabricated provenance).
    assert find_span(passage, "Bob founded Beta") is None


@pytest.mark.unit
def test_find_and_verify_span_roundtrip() -> None:
    passage = _passage("Alice founded Acme.")
    span = find_span(passage, "founded Acme")
    assert span is not None
    assert verify_span(passage, span) is True
    assert passage.text[span.start : span.end] == "founded Acme"


@pytest.mark.unit
def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("  Alice   founded\n X  ") == "Alice founded X"


@pytest.mark.unit
def test_claim_id_is_deterministic_and_changes_with_inputs() -> None:
    a = claim_id("p1", "alice founded x", 0)
    b = claim_id("p1", "alice founded x", 0)
    c = claim_id("p1", "alice founded x", 5)
    assert a == b
    assert a != c
