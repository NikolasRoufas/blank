"""Model-agnostic evidence serialization, generation, and grounding validation.

Importing this package pulls in no optional dependency and initializes no model;
the Hugging Face, httpx, and NLI backends load lazily on first use.
"""

from __future__ import annotations

from egrag.generation.adapters import (
    CachedTextGenerator,
    FakeTextGenerator,
    HttpxTransport,
    HuggingFaceGenerator,
    OpenAICompatibleGenerator,
)
from egrag.generation.assembly import build_evidence_package
from egrag.generation.capabilities import GeneratorCapabilities, validate_config
from egrag.generation.config import GenerationConfig
from egrag.generation.grounding import BaselineGroundingVerifier, NLIGroundingVerifier
from egrag.generation.interfaces import (
    ChatTextGenerator,
    HttpRequestError,
    HttpResponse,
    HttpTimeoutError,
    HttpTransport,
    TextGenerator,
)
from egrag.generation.parsing import ParsedAnswer, parse_generation
from egrag.generation.rendering import (
    ChatEvidenceRenderer,
    MarkdownEvidenceRenderer,
    PlainTextEvidenceRenderer,
    evidence_warnings,
    instruction_lines,
    unresolved_conflict_ids,
)
from egrag.generation.service import GenerationService
from egrag.generation.validation import validate_attribution

__all__ = [
    "BaselineGroundingVerifier",
    "CachedTextGenerator",
    "ChatEvidenceRenderer",
    "ChatTextGenerator",
    "FakeTextGenerator",
    "GenerationConfig",
    "GenerationService",
    "GeneratorCapabilities",
    "HttpRequestError",
    "HttpResponse",
    "HttpTimeoutError",
    "HttpTransport",
    "HttpxTransport",
    "HuggingFaceGenerator",
    "MarkdownEvidenceRenderer",
    "NLIGroundingVerifier",
    "OpenAICompatibleGenerator",
    "ParsedAnswer",
    "PlainTextEvidenceRenderer",
    "TextGenerator",
    "build_evidence_package",
    "evidence_warnings",
    "instruction_lines",
    "parse_generation",
    "unresolved_conflict_ids",
    "validate_attribution",
    "validate_config",
]
