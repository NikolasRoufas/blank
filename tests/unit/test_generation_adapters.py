"""Generator-adapter tests (17-25): fake determinism, capabilities, HTTP, isolation."""

from __future__ import annotations

import importlib
import json
import sys
from typing import Any

import pytest

from egrag.domain.errors import GenerationError
from egrag.generation import (
    FakeTextGenerator,
    GenerationConfig,
    GeneratorCapabilities,
    HttpResponse,
    HttpTimeoutError,
    OpenAICompatibleGenerator,
    validate_config,
)

PROMPT = (
    "# EVIDENCE\n## ACCEPTED EVIDENCE\n[c1] source=s belief=0.50 utility=0.50\n"
    "<<<EVIDENCE_BEGIN>>>\nAcme grew.\n<<<EVIDENCE_END>>>\n"
)


class FakeTransport:
    """A deterministic in-memory HTTP transport (no network, no httpx)."""

    def __init__(
        self, *, responses: list[HttpResponse] | None = None, raise_timeout: bool = False
    ) -> None:
        self.calls = 0
        self._responses = responses or []
        self._raise_timeout = raise_timeout

    def post(self, url: str, payload: dict[str, Any], timeout: float) -> HttpResponse:
        self.calls += 1
        if self._raise_timeout:
            raise HttpTimeoutError("simulated timeout")
        index = min(self.calls - 1, len(self._responses) - 1)
        return self._responses[index]


def _chat_response(content: str) -> HttpResponse:
    return HttpResponse(
        status_code=200,
        body=json.dumps({"choices": [{"message": {"content": content}}]}),
    )


@pytest.mark.unit
def test_fake_generation_is_repeatable() -> None:
    """Acceptance 17: deterministic fake generation produces repeatable output."""

    generator = FakeTextGenerator()
    config = GenerationConfig()
    first = generator.complete(PROMPT, config)
    second = generator.complete(PROMPT, config)
    assert first == second
    assert json.loads(first)["citations"] == ["c1"]


@pytest.mark.unit
def test_unsupported_capabilities_produce_errors_or_warnings() -> None:
    """Acceptance 18: unsupported adapter capabilities produce errors or warnings."""

    caps = GeneratorCapabilities(streaming=False, seed_support=False, deterministic_decoding=True)
    with pytest.raises(GenerationError):
        validate_config(GenerationConfig(stream=True), caps)
    warnings = validate_config(GenerationConfig(seed=7), caps)
    assert any("seed" in w for w in warnings)
    # deterministic-incompatibility is a warning (see acceptance test 28), not a raise
    no_det = GeneratorCapabilities(deterministic_decoding=False)
    det_warnings = validate_config(GenerationConfig(deterministic=True), no_det)
    assert any("deterministic" in w for w in det_warnings)


@pytest.mark.unit
def test_http_timeout_produces_typed_error() -> None:
    """Acceptance 19: HTTP timeouts produce typed errors."""

    transport = FakeTransport(raise_timeout=True)
    generator = OpenAICompatibleGenerator("http://local", "m", transport=transport)
    with pytest.raises(GenerationError):
        generator.complete(PROMPT, GenerationConfig(max_retries=1))


@pytest.mark.unit
def test_malformed_http_response_produces_typed_error() -> None:
    """Acceptance 20: malformed HTTP responses produce typed errors."""

    bad_json = FakeTransport(responses=[HttpResponse(status_code=200, body="not json {")])
    gen1 = OpenAICompatibleGenerator("http://local", "m", transport=bad_json)
    with pytest.raises(GenerationError):
        gen1.complete(PROMPT, GenerationConfig(max_retries=0))

    missing = FakeTransport(responses=[HttpResponse(status_code=200, body=json.dumps({"x": 1}))])
    gen2 = OpenAICompatibleGenerator("http://local", "m", transport=missing)
    with pytest.raises(GenerationError):
        gen2.complete(PROMPT, GenerationConfig(max_retries=0))


@pytest.mark.unit
def test_retry_limit_is_enforced() -> None:
    """Acceptance 21 & 22: safe transient retry limits are enforced (mocked transport)."""

    transport = FakeTransport(raise_timeout=True)
    generator = OpenAICompatibleGenerator("http://local", "m", transport=transport)
    with pytest.raises(GenerationError):
        generator.complete(PROMPT, GenerationConfig(max_retries=2))
    assert transport.calls == 3  # initial attempt + 2 retries


@pytest.mark.unit
def test_openai_adapter_success_path() -> None:
    transport = FakeTransport(responses=[_chat_response('{"answer": "ok", "citations": []}')])
    generator = OpenAICompatibleGenerator("http://local", "m", transport=transport)
    out = generator.complete(PROMPT, GenerationConfig(max_retries=0))
    assert json.loads(out)["answer"] == "ok"
    assert transport.calls == 1


@pytest.mark.unit
def test_import_initializes_no_generator() -> None:
    """Acceptance 25: importing the package initializes no generator/model."""

    for name in [
        m for m in sys.modules if m == "egrag.generation" or m.startswith("egrag.generation.")
    ]:
        del sys.modules[name]
    sys.modules.pop("transformers", None)
    sys.modules.pop("httpx", None)
    importlib.import_module("egrag.generation")
    assert "transformers" not in sys.modules
    assert "httpx" not in sys.modules
