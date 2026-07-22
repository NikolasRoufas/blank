"""Optional Hugging Face structured-generation backend for claim extraction.

Requires the ``local-models`` extra. The model/tokenizer are loaded lazily on
first use (never at import time, no download at import). When the tokenizer
provides a chat template the extraction prompt is sent as a system + user chat
conversation (so instruction-tuned models follow the JSON contract); otherwise
the raw prompt is used. Decoding is deterministic by default. Supports
cpu/mps/cuda/integer devices and an optional dtype.
"""

from __future__ import annotations

from egrag.hf_runtime import HFTextPipeline

_SYSTEM_INSTRUCTION = (
    "You are a precise information-extraction engine. Read the user's instructions "
    "and passage, then respond with ONLY the requested JSON object and nothing else. "
    "Do not add explanations, prose, or markdown fences."
)


class HuggingFaceStructuredModel:
    """A :class:`StructuredModel` backed by a Transformers text-generation pipeline."""

    def __init__(
        self,
        model_name: str,
        *,
        max_new_tokens: int = 512,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        device: str | int | None = None,
        dtype: str | None = None,
        apply_chat_template: bool = True,
    ) -> None:
        self._max_new_tokens = max_new_tokens
        self._apply_chat_template = apply_chat_template
        self.name = f"huggingface:{model_name}"
        self._runtime = HFTextPipeline(
            model_name,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            device=device,
            dtype=dtype,
        )

    def complete(self, prompt: str, *, seed: int = 0, deterministic: bool = True) -> str:
        messages = None
        if self._apply_chat_template:
            messages = [
                {"role": "system", "content": _SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ]
        return self._runtime.generate(
            prompt=None if messages is not None else prompt,
            messages=messages,
            max_new_tokens=self._max_new_tokens,
            deterministic=deterministic,
            seed=seed,
        )


__all__ = ["HuggingFaceStructuredModel"]
