"""Capability-aware renderer selection (real-adapter-repair §4, §8).

A chat-capable generator must receive ``ChatEvidenceRenderer`` messages with
instructions in the system role and untrusted evidence in the user role; a plain
generator receives the plain-text prompt. No model is loaded (fake generators).
"""

from __future__ import annotations

import json
import re

import pytest

from egrag.domain.models import (
    AtomicClaim,
    ChatMessage,
    ClaimProvenance,
    EvidencePackage,
    Query,
    SelectedEvidence,
    SourceMetadata,
    SourceSpan,
)
from egrag.generation import GenerationConfig, GenerationService, GeneratorCapabilities

_HEADER = re.compile(r"^\[([^\]]+)\] source=", re.MULTILINE)


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


def _package() -> EvidencePackage:
    sel = SelectedEvidence(claim_id="c1", selection_score=0.5, rank=0)
    return EvidencePackage(
        package_id="pkg",
        query=Query(query_id="q", text="what happened"),
        claims=(_claim("c1"),),
        selected=(sel,),
    )


class FakeChatGenerator:
    """Chat-capable generator that records the messages it receives."""

    def __init__(self) -> None:
        self.messages: tuple[ChatMessage, ...] | None = None
        self.plain_called = False

    def capabilities(self) -> GeneratorCapabilities:
        return GeneratorCapabilities(
            chat_template=True,
            context_limit=8192,
            has_tokenizer=True,
            deterministic_decoding=True,
            seed_support=True,
        )

    def complete(self, prompt: str, config: GenerationConfig) -> str:  # pragma: no cover
        self.plain_called = True
        return json.dumps({"answer": "x", "citations": [], "uncertainty": ""})

    def complete_chat(self, messages, config: GenerationConfig) -> str:
        self.messages = tuple(messages)
        user = "\n".join(m.content for m in messages if m.role == "user")
        cited = _HEADER.findall(user)
        return json.dumps({"answer": "grounded", "citations": cited, "uncertainty": ""})


@pytest.mark.unit
def test_chat_generator_receives_system_and_user_roles() -> None:
    gen = FakeChatGenerator()
    answer = GenerationService().generate(_package(), gen, GenerationConfig())
    assert gen.plain_called is False  # chat path taken, not the plain path
    assert gen.messages is not None
    roles = [m.role for m in gen.messages]
    assert "system" in roles and "user" in roles
    system = "\n".join(m.content for m in gen.messages if m.role == "system")
    user = "\n".join(m.content for m in gen.messages if m.role == "user")
    # Instructions in the system role; untrusted evidence + claim ids in the user role.
    assert "evidence" in system.lower()
    assert "[c1]" in user
    # The model cited the evidence it was given and attribution validated.
    assert answer.cited_claim_ids == ("c1",)


@pytest.mark.unit
def test_plain_generator_uses_plain_path() -> None:
    from egrag.generation import FakeTextGenerator

    answer = GenerationService().generate(_package(), FakeTextGenerator(), GenerationConfig())
    assert answer.cited_claim_ids == ("c1",)
