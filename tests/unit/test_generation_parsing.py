"""Parsing, attribution-validation, and grounding tests (6-13)."""

from __future__ import annotations

import json

import pytest

from egrag.domain.errors import GenerationError
from egrag.domain.models import (
    AtomicClaim,
    ClaimProvenance,
    ConflictOutcome,
    ConflictSet,
    EvidencePackage,
    EvidenceStatus,
    GeneratedAnswer,
    Query,
    SelectedEvidence,
    SourceMetadata,
    SourceSpan,
)
from egrag.generation import (
    BaselineGroundingVerifier,
    PlainTextEvidenceRenderer,
    parse_generation,
    validate_attribution,
)


def _claim(cid: str, text: str = "a fact") -> AtomicClaim:
    return AtomicClaim(
        claim_id=cid,
        text=text,
        provenance=ClaimProvenance(
            source=SourceMetadata(source_id="s"),
            spans=(SourceSpan(source_id="s", start=0, end=len(text), text=text),),
        ),
        extraction_confidence=0.8,
    )


def _package(selected: list[SelectedEvidence], *, conflicts: tuple = ()) -> EvidencePackage:
    claims = [_claim(item.claim_id) for item in selected]
    return EvidencePackage(
        package_id="pkg",
        query=Query(query_id="q", text="what happened"),
        claims=tuple(claims),
        selected=tuple(selected),
        conflicts=conflicts,
    )


def _out(answer: str, citations: list[str], uncertainty: str = "") -> str:
    return json.dumps({"answer": answer, "citations": citations, "uncertainty": uncertainty})


@pytest.mark.unit
def test_unknown_citation_is_rejected() -> None:
    """Acceptance 6: unknown claim citations are rejected."""

    package = _package([SelectedEvidence(claim_id="c1", selection_score=0.5, rank=0)])
    parsed = parse_generation(_out("answer [c9]", ["c9"]))
    with pytest.raises(GenerationError):
        validate_attribution(parsed, package)


@pytest.mark.unit
def test_missing_citations_detected() -> None:
    """Acceptance 7: missing required citations are detected (flagged)."""

    package = _package([SelectedEvidence(claim_id="c1", selection_score=0.5, rank=0)])
    answer = GeneratedAnswer(text="An answer with no citations at all here.", abstained=False)
    verified = BaselineGroundingVerifier().verify(answer, package)
    assert any("no citations" in w for w in verified.unsupported_warnings)


@pytest.mark.unit
def test_rejected_claims_cannot_be_cited_as_accepted() -> None:
    """Acceptance 8: rejected claims cannot be cited as accepted evidence."""

    package = _package(
        [
            SelectedEvidence(
                claim_id="c1", selection_score=0.5, rank=0, status=EvidenceStatus.REJECTED
            )
        ]
    )
    parsed = parse_generation(_out("answer [c1]", ["c1"]))
    with pytest.raises(GenerationError):
        validate_attribution(parsed, package)


@pytest.mark.unit
def test_superseded_claims_require_context() -> None:
    """Acceptance 9: superseded claims cited without context are flagged."""

    package = _package(
        [
            SelectedEvidence(
                claim_id="c1", selection_score=0.5, rank=0, status=EvidenceStatus.SUPERSEDED
            )
        ]
    )
    answer = GeneratedAnswer(text="It is [c1].", cited_claim_ids=("c1",), uncertainty="")
    verified = BaselineGroundingVerifier().verify(answer, package)
    assert any("superseded" in w for w in verified.unsupported_warnings)


@pytest.mark.unit
def test_unresolved_conflicts_produce_uncertainty_instruction() -> None:
    """Acceptance 10: unresolved conflicts produce an uncertainty instruction."""

    conflict = ConflictSet(
        conflict_id="cf",
        claim_ids=("c1", "c2"),
        outcome=ConflictOutcome.UNRESOLVED,
    )
    package = _package(
        [
            SelectedEvidence(claim_id="c1", selection_score=0.5, rank=0),
            SelectedEvidence(claim_id="c2", selection_score=0.5, rank=1),
        ],
        conflicts=(conflict,),
    )
    prompt = PlainTextEvidenceRenderer().render(package)
    assert "UNRESOLVED conflict" in prompt
    assert "express uncertainty" in prompt.lower()


@pytest.mark.unit
def test_unsupported_certainty_flagged() -> None:
    """Acceptance 11: unsupported certainty (no uncertainty for unresolved) is flagged."""

    conflict = ConflictSet(
        conflict_id="cf", claim_ids=("c1", "c2"), outcome=ConflictOutcome.UNRESOLVED
    )
    package = _package(
        [
            SelectedEvidence(claim_id="c1", selection_score=0.5, rank=0),
            SelectedEvidence(claim_id="c2", selection_score=0.5, rank=1),
        ],
        conflicts=(conflict,),
    )
    answer = GeneratedAnswer(text="It is definitely [c1].", cited_claim_ids=("c1",), uncertainty="")
    verified = BaselineGroundingVerifier().verify(answer, package)
    assert any("no uncertainty" in w for w in verified.unsupported_warnings)


@pytest.mark.unit
def test_empty_output_raises_typed_error() -> None:
    """Acceptance 12: empty model output produces a typed error."""

    with pytest.raises(GenerationError):
        parse_generation("")
    with pytest.raises(GenerationError):
        parse_generation(json.dumps({"answer": "", "citations": []}))


@pytest.mark.unit
def test_malformed_output_raises_typed_error() -> None:
    """Acceptance 13: malformed structured output produces a typed error."""

    with pytest.raises(GenerationError):
        parse_generation("not json {")
    with pytest.raises(GenerationError):
        parse_generation(json.dumps(["not", "an", "object"]))
    with pytest.raises(GenerationError):
        parse_generation(json.dumps({"answer": "ok", "citations": "c1"}))  # citations not a list
