"""Generation-service tests (14, 15, 16, 26)."""

from __future__ import annotations

import pytest

from egrag.domain.errors import GenerationError
from egrag.domain.models import (
    AtomicClaim,
    ClaimProvenance,
    EvidencePackage,
    Query,
    SelectedEvidence,
    SourceMetadata,
    SourceSpan,
)
from egrag.generation import (
    FakeTextGenerator,
    GenerationConfig,
    GenerationService,
    GeneratorCapabilities,
)


def _claim(cid: str, text: str = "a grounded fact about acme") -> AtomicClaim:
    return AtomicClaim(
        claim_id=cid,
        text=text,
        provenance=ClaimProvenance(
            source=SourceMetadata(source_id="s"),
            spans=(SourceSpan(source_id="s", start=0, end=len(text), text=text),),
        ),
        extraction_confidence=0.8,
    )


def _package(
    selected: list[SelectedEvidence], *, claims: list[AtomicClaim] | None = None
) -> EvidencePackage:
    claims = claims if claims is not None else [_claim(s.claim_id) for s in selected]
    return EvidencePackage(
        package_id="pkg",
        query=Query(query_id="q", text="what happened"),
        claims=tuple(claims),
        selected=tuple(selected),
    )


class SpyGenerator:
    """Wraps the fake generator and counts how many times it is invoked."""

    def __init__(self, *, context_limit: int = 8192, chat_template: bool = False) -> None:
        self.calls = 0
        self._context_limit = context_limit
        self._chat_template = chat_template

    def capabilities(self) -> GeneratorCapabilities:
        return GeneratorCapabilities(
            context_limit=self._context_limit,
            chat_template=self._chat_template,
            deterministic_decoding=True,
            seed_support=True,
            streaming=False,
        )

    def complete(self, prompt: str, config: GenerationConfig) -> str:
        self.calls += 1
        return FakeTextGenerator().complete(prompt, config)


@pytest.mark.unit
def test_generator_not_called_when_package_invalid() -> None:
    """Acceptance 15: the generator is not called when the package is invalid."""

    package = _package([], claims=[_claim("c1")])  # claims present but nothing selected
    spy = SpyGenerator()
    with pytest.raises(GenerationError):
        GenerationService().generate(package, spy)
    assert spy.calls == 0


@pytest.mark.unit
def test_context_limit_validated_before_inference() -> None:
    """Acceptance 14 & 16: oversized evidence is handled before the generator runs."""

    package = _package([SelectedEvidence(claim_id="c1", selection_score=0.5, rank=0)])
    spy = SpyGenerator(context_limit=5)  # far smaller than the rendered prompt
    with pytest.raises(GenerationError):
        GenerationService().generate(package, spy, GenerationConfig(max_new_tokens=10))
    assert spy.calls == 0


@pytest.mark.unit
def test_causal_and_chat_adapters_follow_same_grounding_policy() -> None:
    """Acceptance 26: causal and chat adapters follow the same grounding policy."""

    package = _package([SelectedEvidence(claim_id="c1", selection_score=0.5, rank=0)])
    causal = GenerationService().generate(package, FakeTextGenerator(chat_template=False))
    chat = GenerationService().generate(package, FakeTextGenerator(chat_template=True))
    assert causal.cited_claim_ids == chat.cited_claim_ids == ("c1",)
    assert causal.unsupported_warnings == chat.unsupported_warnings


@pytest.mark.unit
def test_service_produces_grounded_answer() -> None:
    package = _package([SelectedEvidence(claim_id="c1", selection_score=0.5, rank=0)])
    answer = GenerationService().generate(package, FakeTextGenerator())
    assert answer.cited_claim_ids == ("c1",)
    assert "[c1]" in answer.text
