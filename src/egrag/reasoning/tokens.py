"""Replaceable token counters for deterministic budgeting.

The selector depends only on the :class:`egrag.domain.ports.TokenCounter`
protocol — never on a specific generator's tokenizer. A conservative
character-based fallback is always available; a tokenizer-based counter can be
plugged in when one is present.
"""

from __future__ import annotations

import importlib
import math
from typing import Any


class CharacterTokenCounter:
    """Conservative deterministic fallback: ``ceil(len(text) / chars_per_token)``.

    Defaults to ~4 characters per token. Being an over-estimate-friendly,
    tokenizer-independent heuristic, it keeps budgeting deterministic and safe.
    """

    def __init__(self, chars_per_token: float = 4.0) -> None:
        if chars_per_token <= 0.0:
            raise ValueError("chars_per_token must be positive")
        self._chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        if not text:
            return 0
        return math.ceil(len(text) / self._chars_per_token)


class WhitespaceTokenCounter:
    """Counts whitespace-delimited words (deterministic, dependency-free)."""

    def count(self, text: str) -> int:
        return len(text.split())


class HuggingFaceTokenCounter:
    """Optional tokenizer-based counter (``local-models`` extra), loaded lazily."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._tokenizer: Any = None

    def _ensure(self) -> Any:
        if self._tokenizer is None:
            try:
                transformers: Any = importlib.import_module("transformers")
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "transformers is not installed; install the 'local-models' extra"
                ) from exc
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(self._model_name)
        return self._tokenizer

    def count(self, text: str) -> int:
        tokenizer = self._ensure()
        return len(tokenizer.encode(text, add_special_tokens=False))


__all__ = ["CharacterTokenCounter", "HuggingFaceTokenCounter", "WhitespaceTokenCounter"]
