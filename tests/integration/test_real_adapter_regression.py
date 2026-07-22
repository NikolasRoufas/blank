"""End-to-end regression for the previously-failing real-adapter behaviors
(real-adapter-repair §8). All use deterministic fakes — no model download.

Reproduces and fixes:
* extractor output of valid JSON wrapped in prose (formerly 0/4 valid);
* generator output of valid JSON followed by trailing prose (formerly 0/6);
* hallucinated/confident answer when evidence is insufficient (must not be accepted);
* unknown citation (rejected);
* citing rejected evidence (rejected);
* unresolved conflict (must surface uncertainty).
"""

from __future__ import annotations

import json

import pytest

from egrag.adapters.extraction.structured import StructuredClaimExtractor
from egrag.adapters.retrieval import SentenceAwareChunker, prepare_passages
from egrag.domain.errors import GenerationError
from egrag.domain.models import (
    AtomicClaim,
    ClaimProvenance,
    ConflictOutcome,
    ConflictSet,
    Document,
    EvidencePackage,
    EvidenceStatus,
    Query,
    SelectedEvidence,
    SourceMetadata,
    SourceSpan,
)
from egrag.generation import FakeTextGenerator, GenerationConfig, GenerationService
from egrag.generation.parsing import parse_generation

# --- helpers ----------------------------------------------------------------


def _claim(cid: str, text: str = "Tokyo is the capital of Japan.") -> AtomicClaim:
    return AtomicClaim(
        claim_id=cid,
        text=text,
        provenance=ClaimProvenance(
            source=SourceMetadata(source_id="s"),
            spans=(SourceSpan(source_id="s", start=0, end=len(text), text=text),),
        ),
        extraction_confidence=0.9,
    )


class _StaticModel:
    """A StructuredModel that returns a fixed raw string."""

    def __init__(self, raw: str) -> None:
        self._raw = raw

    def complete(self, prompt: str, *, seed: int = 0, deterministic: bool = True) -> str:
        return self._raw


class _StaticGenerator:
    def __init__(self, raw: str) -> None:
        self._raw = raw

    def capabilities(self):
        from egrag.generation import GeneratorCapabilities

        return GeneratorCapabilities(chat_template=False, context_limit=8192)

    def complete(self, prompt: str, config: GenerationConfig) -> str:
        return self._raw


# --- extractor: JSON wrapped in prose is recovered --------------------------


@pytest.mark.integration
def test_extractor_recovers_json_from_prose() -> None:
    passage = prepare_passages(
        [
            Document(
                document_id="d",
                text="Marie Curie was a Polish physicist.",
                source=SourceMetadata(source_id="Marie_Curie"),
            )
        ],
        SentenceAwareChunker(),
    )[0]
    raw = (
        "Sure, here is the JSON you asked for: "
        + json.dumps(
            {
                "claims": [
                    {
                        "claim_text": "Marie Curie was a Polish physicist.",
                        "source_span_text": "Marie Curie was a Polish physicist.",
                        "confidence": 0.9,
                    }
                ]
            }
        )
        + " Hope this helps!"
    )
    extractor = StructuredClaimExtractor(_StaticModel(raw))
    result = extractor.extract_result(passage)
    assert len(result.claims) == 1  # formerly 0/4
    assert result.claims[0].text == "Marie Curie was a Polish physicist."
    assert any("JSON recovery" in w for w in result.warnings)  # recovery surfaced, not hidden


@pytest.mark.integration
def test_extractor_still_rejects_pure_prose() -> None:
    passage = prepare_passages(
        [Document(document_id="d", text="x.", source=SourceMetadata(source_id="s"))],
        SentenceAwareChunker(),
    )[0]
    from egrag.domain.errors import ExtractionError

    with pytest.raises(ExtractionError):
        StructuredClaimExtractor(_StaticModel("I cannot do that.")).extract_result(passage)


# --- generator: JSON + trailing prose is recovered --------------------------


def _package(selected: list[SelectedEvidence], claims: list[AtomicClaim]) -> EvidencePackage:
    return EvidencePackage(
        package_id="pkg",
        query=Query(query_id="q", text="capital of Japan?"),
        claims=tuple(claims),
        selected=tuple(selected),
    )


@pytest.mark.integration
def test_generator_json_plus_prose_is_recovered() -> None:
    raw = '{"answer": "Tokyo", "citations": ["c1"], "uncertainty": ""}, that is the answer.'
    pkg = _package([SelectedEvidence(claim_id="c1", selection_score=0.5, rank=0)], [_claim("c1")])
    answer = GenerationService().generate(pkg, _StaticGenerator(raw), GenerationConfig())
    assert answer.text == "Tokyo"
    assert answer.cited_claim_ids == ("c1",)


@pytest.mark.integration
def test_parser_marks_recovery_flag() -> None:
    parsed = parse_generation('{"answer":"a","citations":[]} trailing')
    assert parsed.recovered is True
    strict = parse_generation('{"answer":"a","citations":[]}')
    assert strict.recovered is False


# --- insufficient evidence: no confident unsupported answer -----------------


@pytest.mark.integration
def test_insufficient_evidence_does_not_yield_confident_answer() -> None:
    # No selected evidence -> the service refuses to invoke the generator at all,
    # so a hallucinated confident answer can never be returned.
    pkg = _package([], [_claim("c1")])
    hallucinator = _StaticGenerator('{"answer": "It happened in 2013.", "citations": []}')
    with pytest.raises(GenerationError):
        GenerationService().generate(pkg, hallucinator, GenerationConfig())


# --- attribution: unknown and rejected citations are rejected ---------------


@pytest.mark.integration
def test_unknown_citation_rejected() -> None:
    pkg = _package([SelectedEvidence(claim_id="c1", selection_score=0.5, rank=0)], [_claim("c1")])
    gen = _StaticGenerator('{"answer": "x", "citations": ["c999"], "uncertainty": ""}')
    with pytest.raises(GenerationError, match="unknown claim"):
        GenerationService().generate(pkg, gen, GenerationConfig())


@pytest.mark.integration
def test_rejected_evidence_citation_rejected() -> None:
    sel = SelectedEvidence(
        claim_id="c1", selection_score=0.5, rank=0, status=EvidenceStatus.REJECTED
    )
    pkg = _package([sel], [_claim("c1")])
    gen = _StaticGenerator('{"answer": "x", "citations": ["c1"], "uncertainty": ""}')
    with pytest.raises(GenerationError, match="rejected"):
        GenerationService().generate(pkg, gen, GenerationConfig())


# --- unresolved conflict surfaces uncertainty -------------------------------


@pytest.mark.integration
def test_unresolved_conflict_surfaces_uncertainty() -> None:
    claims = [_claim("c1", "The deadline is June 30."), _claim("c2", "The deadline is July 1.")]
    selected = [
        SelectedEvidence(claim_id="c1", selection_score=0.5, rank=0),
        SelectedEvidence(claim_id="c2", selection_score=0.5, rank=1),
    ]
    pkg = EvidencePackage(
        package_id="pkg",
        query=Query(query_id="q", text="deadline?"),
        claims=tuple(claims),
        selected=tuple(selected),
        conflicts=(
            ConflictSet(
                conflict_id="cf1", claim_ids=("c1", "c2"), outcome=ConflictOutcome.UNRESOLVED
            ),
        ),
    )
    answer = GenerationService().generate(pkg, FakeTextGenerator(), GenerationConfig())
    assert answer.uncertainty  # the unresolved conflict is surfaced as uncertainty
