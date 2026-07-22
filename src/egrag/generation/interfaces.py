"""Provider-independent generation interfaces.

A :class:`TextGenerator` is the thin, interchangeable model component: a string
prompt in, a string out, plus declared capabilities. The HTTP transport is also
abstracted so the OpenAI-compatible adapter can be tested without httpx or the
network. No provider-specific type appears in these signatures.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from egrag.domain.models import ChatMessage
from egrag.generation.capabilities import GeneratorCapabilities
from egrag.generation.config import GenerationConfig


@runtime_checkable
class TextGenerator(Protocol):
    """A model adapter that completes a prompt into text."""

    def capabilities(self) -> GeneratorCapabilities: ...

    def complete(self, prompt: str, config: GenerationConfig) -> str: ...


@runtime_checkable
class ChatTextGenerator(TextGenerator, Protocol):
    """A generator that also consumes chat messages (instructions in the system
    role, untrusted evidence in the user role) and applies the model's chat
    template inside the adapter boundary."""

    def complete_chat(self, messages: Sequence[ChatMessage], config: GenerationConfig) -> str: ...


class HttpTimeoutError(Exception):
    """Raised by a transport when a request exceeds its timeout."""


class HttpRequestError(Exception):
    """Raised by a transport for a non-timeout request failure."""


@dataclass(frozen=True)
class HttpResponse:
    """A minimal HTTP response (status + raw body)."""

    status_code: int
    body: str

    def json(self) -> Any:
        import json

        return json.loads(self.body)


@runtime_checkable
class HttpTransport(Protocol):
    """A minimal POST transport; raises :class:`HttpTimeoutError` on timeout."""

    def post(self, url: str, payload: dict[str, Any], timeout: float) -> HttpResponse: ...


__all__ = [
    "ChatTextGenerator",
    "HttpRequestError",
    "HttpResponse",
    "HttpTimeoutError",
    "HttpTransport",
    "TextGenerator",
]
