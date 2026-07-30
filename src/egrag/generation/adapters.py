"""Generator adapters: deterministic fake, Hugging Face, OpenAI-compatible HTTP.

Provider-specific types (transformers, httpx) are confined to these adapters and
imported lazily — never at import time, and never leaked into application or
domain code. Each adapter declares its capabilities.
"""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Sequence
from typing import Any

from egrag.caching.keys import build_cache_key
from egrag.domain.errors import GenerationError
from egrag.domain.models import ChatMessage
from egrag.domain.ports import CacheBackend
from egrag.generation.capabilities import GeneratorCapabilities
from egrag.generation.config import GenerationConfig
from egrag.generation.interfaces import (
    ChatTextGenerator,
    HttpRequestError,
    HttpResponse,
    HttpTimeoutError,
    HttpTransport,
    TextGenerator,
)
from egrag.hf_runtime import HFTextPipeline

_CLAIM_HEADER_RE = re.compile(r"^\[([^\]]+)\] source=")


class FakeTextGenerator:
    """Deterministic generator that cites the citable claims found in the prompt.

    It reads claim IDs from the rendered evidence (ignoring REJECTED/SUPERSEDED
    sections), emits the JSON output contract, and expresses uncertainty when the
    prompt declares an unresolved conflict. Output is fully repeatable.
    """

    def __init__(
        self,
        *,
        chat_template: bool = False,
        context_limit: int = 8192,
        streaming: bool = False,
    ) -> None:
        self._chat_template = chat_template
        self._context_limit = context_limit
        self._streaming = streaming

    def capabilities(self) -> GeneratorCapabilities:
        return GeneratorCapabilities(
            chat_template=self._chat_template,
            context_limit=self._context_limit,
            has_tokenizer=False,
            structured_output=True,
            streaming=self._streaming,
            deterministic_decoding=True,
            seed_support=True,
        )

    def complete(self, prompt: str, config: GenerationConfig) -> str:
        citable: list[str] = []
        citable_section = True
        for line in prompt.splitlines():
            if line.startswith("#"):
                upper = line.upper()
                citable_section = not ("REJECTED" in upper or "SUPERSEDED" in upper)
                continue
            match = _CLAIM_HEADER_RE.match(line)
            if match is not None and citable_section:
                citable.append(match.group(1))
        uncertainty = (
            "Some evidence is in unresolved conflict; the answer is uncertain on those points."
            if "UNRESOLVED" in prompt
            else ""
        )
        if citable:
            cited = " ".join(f"[{cid}]" for cid in citable)
            answer = f"Based on the supplied evidence {cited}, here is the grounded answer."
        else:
            answer = "No supported evidence is available to answer the query."
        return json.dumps({"answer": answer, "citations": citable, "uncertainty": uncertainty})


class HuggingFaceGenerator:
    """Local generation via a Transformers text-generation pipeline (``local-models``).

    The model/tokenizer load lazily on first use (never at import time). When the
    tokenizer provides a chat template and ``apply_chat_template`` is enabled, the
    adapter consumes :class:`ChatMessage` turns via :meth:`complete_chat` and
    applies the template; this is what ``capabilities().chat_template`` reports.
    Plain causal models still work through :meth:`complete`. Supports
    cpu/mps/cuda/integer devices and an optional dtype; decoding is deterministic
    by default.
    """

    def __init__(
        self,
        model_name: str,
        *,
        context_limit: int = 4096,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        device: str | int | None = None,
        dtype: str | None = None,
        quantization: str | None = None,
        require_cuda: bool = False,
        apply_chat_template: bool = True,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._model_name = model_name
        self._context_limit = context_limit
        self._apply_chat_template = apply_chat_template
        self._runtime = HFTextPipeline(
            model_name,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            device=device,
            dtype=dtype,
            quantization=quantization,
            require_cuda=require_cuda,
            chat_template_kwargs=chat_template_kwargs,
        )

    def resolved_runtime_info(self) -> dict[str, str]:
        """Best-effort resolved model/device/dtype/quantization, for manifests."""

        return self._runtime.resolved_revisions()

    def capabilities(self) -> GeneratorCapabilities:
        return GeneratorCapabilities(
            chat_template=self._apply_chat_template,
            context_limit=self._context_limit,
            has_tokenizer=True,
            structured_output=False,
            streaming=False,
            deterministic_decoding=True,
            seed_support=True,
        )

    def complete(self, prompt: str, config: GenerationConfig) -> str:
        return self._runtime.generate(
            prompt=prompt,
            max_new_tokens=config.max_new_tokens,
            deterministic=config.deterministic,
            temperature=config.temperature,
            top_p=config.top_p,
            seed=config.seed,
            stop=config.stop,
        )

    def complete_chat(self, messages: tuple[ChatMessage, ...], config: GenerationConfig) -> str:
        """Generate from chat messages, applying the model's chat template.

        Falls back to a plain role-concatenated prompt if the tokenizer has no
        chat template. Provider message formatting stays inside this adapter.
        """

        payload = [{"role": m.role, "content": m.content} for m in messages]
        return self._runtime.generate(
            messages=payload,
            max_new_tokens=config.max_new_tokens,
            deterministic=config.deterministic,
            temperature=config.temperature,
            top_p=config.top_p,
            seed=config.seed,
            stop=config.stop,
        )


class CachedTextGenerator:
    """Wraps a generator and memoizes raw output in a cache backend.

    The cached value is the raw model string, so cold and warm runs are
    identical. The key includes the model id, model revision, prompt version,
    schema version, and decoding controls (max tokens, temperature, top-p, seed,
    determinism, stop sequences) — so output is never reused across a different
    model/revision/prompt/decoding. ``complete_chat`` is delegated only when the
    wrapped generator is chat-capable (guarded by ``capabilities().chat_template``
    in the service).
    """

    def __init__(
        self,
        inner: TextGenerator,
        cache: CacheBackend,
        *,
        model: str,
        model_revision: str | None = None,
        prompt_version: str | None = None,
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._model = model
        self._model_revision = model_revision
        self._prompt_version = prompt_version

    def capabilities(self) -> GeneratorCapabilities:
        return self._inner.capabilities()

    def _cfg(self, config: GenerationConfig) -> dict[str, Any]:
        return {
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "seed": config.seed,
            "deterministic": config.deterministic,
            "stop": list(config.stop),
        }

    def _key(self, namespace: str, algorithm: str, content: str, config: GenerationConfig) -> str:
        return build_cache_key(
            namespace=namespace,
            content=content,
            algorithm=algorithm,
            model=self._model,
            model_revision=self._model_revision,
            prompt_version=self._prompt_version,
            config=self._cfg(config),
        )

    def complete(self, prompt: str, config: GenerationConfig) -> str:
        key = self._key("generation", "text-generation", prompt, config)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        output = self._inner.complete(prompt, config)
        self._cache.set(key, output)
        return output

    def complete_chat(self, messages: Sequence[ChatMessage], config: GenerationConfig) -> str:
        content = json.dumps([{"role": m.role, "content": m.content} for m in messages])
        key = self._key("generation-chat", "chat-generation", content, config)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if not isinstance(self._inner, ChatTextGenerator):
            raise GenerationError("wrapped generator does not support chat messages")
        output = self._inner.complete_chat(messages, config)
        self._cache.set(key, output)
        return output


class HttpxTransport:
    """Default :class:`HttpTransport` backed by httpx (``http-models`` extra)."""

    def __init__(self, *, headers: dict[str, str] | None = None) -> None:
        self._headers = headers or {}

    def post(self, url: str, payload: dict[str, Any], timeout: float) -> HttpResponse:
        try:
            httpx: Any = importlib.import_module("httpx")
        except ModuleNotFoundError as exc:
            raise RuntimeError("httpx is not installed; install the 'http-models' extra") from exc
        try:
            response = httpx.post(url, json=payload, timeout=timeout, headers=self._headers)
        except httpx.TimeoutException as exc:
            raise HttpTimeoutError(str(exc)) from exc
        except httpx.HTTPError as exc:  # pragma: no cover - exercised via fake transport
            raise HttpRequestError(str(exc)) from exc
        return HttpResponse(status_code=response.status_code, body=response.text)


class OpenAICompatibleGenerator:
    """Generation via an OpenAI-compatible HTTP server (e.g. a local inference server).

    The HTTP transport is injectable so the adapter is fully testable without
    httpx or the network. Timeouts and malformed responses raise typed errors;
    transient failures are retried up to ``config.max_retries`` times.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        transport: HttpTransport | None = None,
        context_limit: int = 8192,
        streaming: bool = False,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._transport = transport if transport is not None else HttpxTransport()
        self._context_limit = context_limit
        self._streaming = streaming

    def capabilities(self) -> GeneratorCapabilities:
        return GeneratorCapabilities(
            chat_template=True,
            context_limit=self._context_limit,
            has_tokenizer=False,
            structured_output=False,
            streaming=self._streaming,
            deterministic_decoding=True,
            seed_support=True,
        )

    def complete(self, prompt: str, config: GenerationConfig) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0 if config.deterministic else config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_new_tokens,
        }
        if config.seed is not None:
            payload["seed"] = config.seed
        if config.stop:
            payload["stop"] = list(config.stop)

        url = f"{self._base_url}/v1/chat/completions"
        last_error: Exception | None = None
        for _attempt in range(config.max_retries + 1):
            try:
                response = self._transport.post(url, payload, config.timeout_s)
            except HttpTimeoutError as exc:
                last_error = exc
                continue  # transient: retry
            except HttpRequestError as exc:
                last_error = exc
                continue
            if response.status_code >= 500:
                last_error = HttpRequestError(f"server error {response.status_code}")
                continue
            if response.status_code != 200:
                raise GenerationError(f"generation HTTP error {response.status_code}")
            return self._parse_choice(response)
        raise GenerationError(
            f"generation request failed after {config.max_retries + 1} attempt(s): {last_error}"
        )

    @staticmethod
    def _parse_choice(response: HttpResponse) -> str:
        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise GenerationError(f"malformed generation response: {exc}") from exc
        if not isinstance(content, str):
            raise GenerationError("malformed generation response: content is not a string")
        return content


__all__ = [
    "CachedTextGenerator",
    "FakeTextGenerator",
    "HttpxTransport",
    "HuggingFaceGenerator",
    "OpenAICompatibleGenerator",
]
