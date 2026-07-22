"""Generation orchestration: validate → render → generate → parse → verify.

The generator is a thin component; all evidence processing happens before it is
invoked. The generator is **not called** when the package is invalid or the
rendered prompt would exceed the adapter's context limit — those raise typed
errors first. Invalid attribution is rejected; grounding warnings are attached
to the returned answer.
"""

from __future__ import annotations

from egrag.domain.errors import GenerationError
from egrag.domain.models import EvidencePackage, GeneratedAnswer
from egrag.domain.ports import TokenCounter
from egrag.generation.capabilities import validate_config
from egrag.generation.config import GenerationConfig
from egrag.generation.grounding import BaselineGroundingVerifier
from egrag.generation.interfaces import ChatTextGenerator, TextGenerator
from egrag.generation.parsing import parse_generation
from egrag.generation.rendering import (
    ChatEvidenceRenderer,
    PlainTextEvidenceRenderer,
    evidence_warnings,
)
from egrag.generation.validation import validate_attribution
from egrag.reasoning.tokens import CharacterTokenCounter


class GenerationService:
    """Coordinates rendering, generation, parsing, validation, and grounding.

    The renderer is chosen by generator capability: a chat-capable generator (one
    that reports ``chat_template`` and implements ``complete_chat``) receives
    :class:`ChatEvidenceRenderer` messages (instructions in the system role,
    untrusted evidence in the user role); a plain causal generator receives the
    :class:`PlainTextEvidenceRenderer` string.
    """

    def __init__(
        self,
        *,
        renderer: PlainTextEvidenceRenderer | None = None,
        chat_renderer: ChatEvidenceRenderer | None = None,
        verifier: BaselineGroundingVerifier | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._renderer = renderer or PlainTextEvidenceRenderer()
        self._chat_renderer = chat_renderer or ChatEvidenceRenderer()
        self._verifier = verifier or BaselineGroundingVerifier()
        self._token_counter = token_counter or CharacterTokenCounter()

    def generate(
        self,
        package: EvidencePackage,
        generator: TextGenerator,
        config: GenerationConfig | None = None,
    ) -> GeneratedAnswer:
        cfg = config or GenerationConfig()
        capabilities = generator.capabilities()
        # 1. Validate config against capabilities (raises on hard incompatibility).
        cap_warnings = validate_config(cfg, capabilities)

        # 2. Refuse to invoke the generator on an invalid package.
        if not package.claims or not package.selected:
            raise GenerationError(
                "evidence package has no usable claims; not invoking the generator"
            )

        # 3. Choose rendering by capability: chat messages for a chat-capable
        #    generator, a plain prompt otherwise. Enforce the context limit on the
        #    rendered text BEFORE invoking the generator.
        chat_gen = (
            generator
            if capabilities.chat_template and isinstance(generator, ChatTextGenerator)
            else None
        )
        if chat_gen is not None:
            messages = self._chat_renderer.render_messages(package)
            rendered = "\n\n".join(m.content for m in messages)
        else:
            rendered = self._renderer.render(package)
        prompt_tokens = self._token_counter.count(rendered)
        if prompt_tokens + cfg.max_new_tokens > capabilities.context_limit:
            raise GenerationError(
                f"rendered evidence ({prompt_tokens} tokens) plus output "
                f"({cfg.max_new_tokens}) exceeds the adapter context limit "
                f"({capabilities.context_limit}); reduce the evidence budget"
            )

        # 4. Invoke the generator and parse/validate its output.
        if chat_gen is not None:
            raw = chat_gen.complete_chat(messages, cfg)
        else:
            raw = generator.complete(rendered, cfg)
        parsed = parse_generation(raw)
        validate_attribution(parsed, package)

        answer = GeneratedAnswer(
            text=parsed.answer,
            cited_claim_ids=parsed.cited_claim_ids,
            uncertainty=parsed.uncertainty,
            abstained=not parsed.cited_claim_ids,
            unsupported_warnings=tuple(cap_warnings) + tuple(evidence_warnings(package)),
        )
        return self._verifier.verify(answer, package)


__all__ = ["GenerationService"]
