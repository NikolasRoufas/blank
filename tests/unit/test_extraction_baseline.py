"""Baseline claim-extraction acceptance tests (cases 1-13, 19, 20, 22)."""

from __future__ import annotations

import pytest

from egrag.adapters.extraction import ExtractionConfig, SentenceClaimExtractor
from egrag.domain.models import Passage, Query, SourceMetadata, SourceSpan


def _passage(text: str, passage_id: str = "p1", source_id: str = "src1") -> Passage:
    return Passage(
        passage_id=passage_id,
        document_id="doc",
        text=text,
        span=SourceSpan(source_id=source_id, start=0, end=len(text), text=text),
    )


EX = SentenceClaimExtractor()


@pytest.mark.unit
def test_conjunction_splits_into_two_claims() -> None:
    """Acceptance 1: a two-proposition conjunction becomes two claims."""

    claims = EX.extract(_passage("Alice founded X and Bob founded Y."))
    assert len(claims) == 2
    texts = [c.text for c in claims]
    assert any("Alice founded X" in t for t in texts)
    assert any("Bob founded Y" in t for t in texts)


@pytest.mark.unit
def test_conjunction_not_split_when_parts_not_meaningful() -> None:
    """A short conjunct list is not split into fragments."""

    claims = EX.extract(_passage("The flag is red and white."))
    assert len(claims) == 1


@pytest.mark.unit
def test_negation_is_preserved() -> None:
    """Acceptance 2: explicit negation is preserved."""

    claim = EX.extract(_passage("Alice did not found X."))[0]
    assert claim.semantics is not None
    assert claim.semantics.negation is True
    assert "not" in claim.text


@pytest.mark.unit
def test_modality_is_preserved() -> None:
    """Acceptance 3: modality/uncertainty is preserved."""

    claim = EX.extract(_passage("The deadline may be July 15."))[0]
    assert claim.semantics is not None
    assert "may" in claim.semantics.modality
    assert "may" in claim.text


@pytest.mark.unit
def test_attribution_is_preserved() -> None:
    """Acceptance 4: reported statements preserve attribution."""

    claim = EX.extract(_passage("According to Bob, the deadline is July 15."))[0]
    assert claim.semantics is not None
    assert claim.semantics.attribution == "Bob"


@pytest.mark.unit
def test_dates_preserved_exactly() -> None:
    """Acceptance 5: dates are preserved exactly."""

    claim = EX.extract(_passage("The launch is on July 15, 2024."))[0]
    assert claim.semantics is not None
    assert "July 15, 2024" in claim.semantics.temporal_expressions


@pytest.mark.unit
def test_quantities_preserved_exactly() -> None:
    """Acceptance 6: quantities are preserved exactly."""

    claim = EX.extract(_passage("The budget is $2.5 million."))[0]
    assert claim.semantics is not None
    assert "$2.5 million" in claim.semantics.quantities


@pytest.mark.unit
def test_names_preserved_exactly() -> None:
    """Acceptance 7: names are preserved exactly."""

    claims = EX.extract(_passage("Alice founded X and Bob founded Y."))
    entities = {e for c in claims if c.semantics for e in c.semantics.named_entities}
    assert "Alice" in entities
    assert "Bob" in entities


@pytest.mark.unit
def test_opinion_is_not_marked_verified_fact() -> None:
    """Acceptance 8 & 19: an extractor assigns no truth belief, even to opinions."""

    claim = EX.extract(_passage("Alice is the best founder ever."))[0]
    assert claim.belief is None
    assert claim.source_reliability is None
    assert claim.query_utility is None


@pytest.mark.unit
def test_ambiguous_pronoun_left_unresolved() -> None:
    """Acceptance 9: an ambiguous pronoun is not resolved (default config)."""

    claim = EX.extract(_passage("It is very large indeed."))[0]
    assert claim.text.startswith("It")
    assert claim.extraction is not None
    assert any("ambiguous pronoun" in w for w in claim.extraction.warnings)


@pytest.mark.unit
def test_clear_pronoun_resolved_only_under_rule() -> None:
    """Acceptance 10: a clear pronoun resolves only under the documented rule."""

    passage = _passage("Alice arrived early. She left late.")
    # Default: not resolved.
    default_claims = EX.extract(passage)
    she_claim = next(c for c in default_claims if "left late" in c.text)
    assert she_claim.text.startswith("She")

    # Rule enabled + exactly one preceding entity (Alice): resolved.
    config = ExtractionConfig(resolve_clear_pronouns=True)
    resolved_claims = EX.extract(passage, config=config)
    resolved = next(c for c in resolved_claims if "left late" in c.text)
    assert resolved.text.startswith("Alice")
    assert resolved.extraction is not None
    assert any("resolved pronoun" in w for w in resolved.extraction.warnings)


@pytest.mark.unit
def test_passage_without_factual_content_returns_zero_claims() -> None:
    """Acceptance 11 & 22: a non-factual passage returns zero claims."""

    assert EX.extract(_passage("What is the plan?")) == []
    assert EX.extract(_passage("...")) == []


@pytest.mark.unit
def test_every_claim_maps_to_valid_source_span() -> None:
    """Acceptance 12 & 13: each claim maps to a span whose text matches the passage."""

    passage = _passage("Alice founded X and Bob founded Y.")
    for claim in EX.extract(passage):
        span = claim.provenance.spans[0]
        local_start = span.start - passage.span.start
        local_end = span.end - passage.span.start
        assert passage.text[local_start:local_end] == span.text


@pytest.mark.unit
def test_deterministic_claim_ids() -> None:
    """Acceptance 20: deterministic inputs produce deterministic claim IDs."""

    passage = _passage("Alice founded X and Bob founded Y.")
    first = [c.claim_id for c in EX.extract(passage)]
    second = [c.claim_id for c in EX.extract(passage)]
    assert first == second
    assert all(cid.startswith("clm-") for cid in first)


@pytest.mark.unit
def test_source_author_metadata_preserved() -> None:
    """Source author metadata is preserved when provided."""

    source = SourceMetadata(source_id="src1", author="J. Doe", title="Report")
    claim = EX.extract(_passage("Alice founded X."), source=source)[0]
    assert claim.provenance.source.author == "J. Doe"


@pytest.mark.unit
def test_query_context_is_accepted() -> None:
    """The extractor accepts optional query context without changing grounding."""

    query = Query(query_id="q", text="who founded X?")
    claims = EX.extract(_passage("Alice founded X."), query=query)
    assert len(claims) == 1
