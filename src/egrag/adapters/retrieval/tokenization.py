"""Tokenization for sparse retrieval.

Tokenization is the explicit normalization step for BM25: it is configurable and
documented. The default lowercases text and extracts alphanumeric runs. Passage
text itself is never normalized — only the tokens derived from it for scoring.
"""

from __future__ import annotations

import re
from collections.abc import Callable

Tokenizer = Callable[[str], list[str]]
"""A function mapping text to an ordered list of tokens."""

_DEFAULT_TOKEN_RE = re.compile(r"[a-z0-9]+")


def default_tokenizer(text: str) -> list[str]:
    """Lowercase and split into alphanumeric tokens (the default normalization)."""

    return _DEFAULT_TOKEN_RE.findall(text.lower())


__all__ = ["Tokenizer", "default_tokenizer"]
