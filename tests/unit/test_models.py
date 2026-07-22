"""Unit tests for domain-model validation (acceptance cases 1-6)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, ValidationError

from egrag.domain.models import (
    AtomicClaim,
    ClaimProvenance,
    ConflictSet,
    Document,
    EvidenceGraphSnapshot,
    EvidencePackage,
    EvidenceRelation,
    GeneratedAnswer,
    Passage,
    Query,
    ReasoningSubgraph,
    RelationType,
    RunManifest,
    SelectedEvidence,
    SourceMetadata,
    SourceSpan,
)

NOW = datetime(2024, 1, 1, tzinfo=UTC)


def _span() -> SourceSpan:
    return SourceSpan(source_id="s1", start=0, end=3, text="abc")


def _provenance() -> ClaimProvenance:
    return ClaimProvenance(source=SourceMetadata(source_id="s1"), spans=(_span(),))


def _claim(claim_id: str = "c1", text: str = "hello world") -> AtomicClaim:
    return AtomicClaim(
        claim_id=claim_id,
        text=text,
        provenance=_provenance(),
        extraction_confidence=0.9,
    )


def _query() -> Query:
    return Query(query_id="q1", text="hello")


# (factory, field-name) pairs covering required non-empty text/identifier fields.
EMPTY_TEXT_CASES: list[tuple[Callable[[str], BaseModel], str]] = [
    (lambda v: Query(query_id=v, text="hi"), "query_id"),
    (lambda v: Query(query_id="q1", text=v), "text"),
    (lambda v: SourceMetadata(source_id=v), "source_id"),
    (lambda v: SourceSpan(source_id=v, start=0, end=3, text="abc"), "source_id"),
    (lambda v: SourceSpan(source_id="s1", start=0, end=3, text=v), "text"),
    (lambda v: Document(document_id=v, text="t", source=SourceMetadata(source_id="s1")), "doc_id"),
    (lambda v: Document(document_id="d1", text=v, source=SourceMetadata(source_id="s1")), "text"),
    (lambda v: Passage(passage_id=v, document_id="d1", text="t", span=_span()), "passage_id"),
    (lambda v: Passage(passage_id="p1", document_id=v, text="t", span=_span()), "document_id"),
    (lambda v: Passage(passage_id="p1", document_id="d1", text=v, span=_span()), "text"),
    (lambda v: _claim(claim_id=v), "claim_id"),
    (lambda v: _claim(text=v), "text"),
    (
        lambda v: EvidenceRelation(
            relation_id=v,
            source_claim_id="c1",
            target_claim_id="c2",
            relation_type=RelationType.SUPPORT,
            relation_confidence=0.5,
        ),
        "relation_id",
    ),
    (lambda v: EvidenceGraphSnapshot(snapshot_id=v), "snapshot_id"),
    (lambda v: ConflictSet(conflict_id=v, claim_ids=("c1", "c2")), "conflict_id"),
    (lambda v: ReasoningSubgraph(subgraph_id=v), "subgraph_id"),
    (lambda v: SelectedEvidence(claim_id=v, selection_score=0.5, rank=0), "claim_id"),
    (lambda v: EvidencePackage(package_id=v, query=_query()), "package_id"),
    (lambda v: GeneratedAnswer(text=v), "text"),
    (lambda v: RunManifest(egrag_version=v, seed=0, created_at=NOW), "egrag_version"),
]


@pytest.mark.unit
@pytest.mark.parametrize("factory, field", EMPTY_TEXT_CASES)
def test_required_text_rejects_empty(factory: Callable[[str], BaseModel], field: str) -> None:
    """Acceptance 1: required text/identifier fields reject empty strings."""

    with pytest.raises(ValidationError):
        factory("")
    with pytest.raises(ValidationError):
        factory("   ")  # whitespace-only is also empty after stripping


@pytest.mark.unit
@pytest.mark.parametrize("value", [-0.1, 1.1, 2.0, -1.0])
def test_probability_out_of_range_rejected(value: float) -> None:
    """Acceptance 2: probability/confidence values outside [0, 1] are rejected."""

    with pytest.raises(ValidationError):
        AtomicClaim(
            claim_id="c1",
            text="hi",
            provenance=_provenance(),
            extraction_confidence=value,
        )
    with pytest.raises(ValidationError):
        EvidenceRelation(
            relation_id="r1",
            source_claim_id="c1",
            target_claim_id="c2",
            relation_type=RelationType.SUPPORT,
            relation_confidence=value,
        )
    with pytest.raises(ValidationError):
        SelectedEvidence(claim_id="c1", selection_score=value, rank=0)
    with pytest.raises(ValidationError):
        SourceMetadata(source_id="s1", reliability_prior=value)


@pytest.mark.unit
@pytest.mark.parametrize(
    "start, end",
    [(-1, 3), (3, 3), (5, 2), (0, 0), (0, -2)],
)
def test_invalid_source_offsets_rejected(start: int, end: int) -> None:
    """Acceptance 3: negative or non-ordered source offsets are rejected."""

    with pytest.raises(ValidationError):
        SourceSpan(source_id="s1", start=start, end=end, text="abc")


@pytest.mark.unit
def test_invalid_timestamps_rejected() -> None:
    """Acceptance 4: invalid/non-ordered timestamps are rejected."""

    # valid_to precedes valid_from
    with pytest.raises(ValidationError):
        AtomicClaim(
            claim_id="c1",
            text="hi",
            provenance=_provenance(),
            extraction_confidence=0.5,
            valid_from=datetime(2025, 1, 1, tzinfo=UTC),
            valid_to=datetime(2024, 1, 1, tzinfo=UTC),
        )
    # unparseable datetime value
    with pytest.raises(ValidationError):
        Query(query_id="q1", text="hi", created_at="not-a-date")  # type: ignore[arg-type]


@pytest.mark.unit
def test_duplicate_identifiers_rejected() -> None:
    """Acceptance 5: duplicate identifiers are rejected where uniqueness is required."""

    c1 = _claim("c1")
    c1b = _claim("c1")  # same id
    with pytest.raises(ValidationError):
        EvidenceGraphSnapshot(snapshot_id="g1", claims=(c1, c1b))

    rel = EvidenceRelation(
        relation_id="r1",
        source_claim_id="c1",
        target_claim_id="c2",
        relation_type=RelationType.SUPPORT,
        relation_confidence=0.5,
    )
    with pytest.raises(ValidationError):
        EvidenceGraphSnapshot(snapshot_id="g1", claims=(_claim("c1"),), relations=(rel, rel))

    with pytest.raises(ValidationError):
        ConflictSet(conflict_id="cf1", claim_ids=("c1", "c1"))

    with pytest.raises(ValidationError):
        GeneratedAnswer(text="answer", cited_claim_ids=("c1", "c1"))


@pytest.mark.unit
def test_relation_must_reference_existing_claims() -> None:
    """Acceptance 5 (references): relations must point to existing claims."""

    rel = EvidenceRelation(
        relation_id="r1",
        source_claim_id="c1",
        target_claim_id="missing",
        relation_type=RelationType.SUPPORT,
        relation_confidence=0.5,
    )
    with pytest.raises(ValidationError):
        EvidenceGraphSnapshot(snapshot_id="g1", claims=(_claim("c1"),), relations=(rel,))


@pytest.mark.unit
def test_invalid_relation_type_rejected() -> None:
    """Acceptance 6: invalid relation types are rejected."""

    with pytest.raises(ValidationError):
        EvidenceRelation(
            relation_id="r1",
            source_claim_id="c1",
            target_claim_id="c2",
            relation_type="not-a-real-type",  # type: ignore[arg-type]
            relation_confidence=0.5,
        )


@pytest.mark.unit
def test_models_are_frozen() -> None:
    """Immutability: domain models reject attribute mutation."""

    claim = _claim()
    with pytest.raises(ValidationError):
        claim.text = "changed"  # type: ignore[misc]


@pytest.mark.unit
def test_self_relation_rejected() -> None:
    """A relation may not link a claim to itself."""

    with pytest.raises(ValidationError):
        EvidenceRelation(
            relation_id="r1",
            source_claim_id="c1",
            target_claim_id="c1",
            relation_type=RelationType.DUPLICATE,
            relation_confidence=0.5,
        )
