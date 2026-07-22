"""Loader for versioned extraction prompt templates.

Prompts live in versioned files under ``prompts/`` and are read at call time
(never at import time). Keeping prompts in files — not scattered string literals
— makes their version explicit and auditable.
"""

from __future__ import annotations

from importlib import resources

_PACKAGE = "egrag.adapters.extraction.prompts"


def load_prompt(version: str) -> str:
    """Return the text of the prompt template for ``version`` (e.g. ``extraction_v1``)."""

    resource = resources.files(_PACKAGE).joinpath(f"{version}.md")
    if not resource.is_file():
        raise FileNotFoundError(f"unknown extraction prompt version: {version!r}")
    return resource.read_text(encoding="utf-8")


__all__ = ["load_prompt"]
