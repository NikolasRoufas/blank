"""Narrow, testable recovery of a single top-level JSON object from model output.

Model output is *untrusted*. This module never "repairs" arbitrary text into
success: it accepts output only when it contains **exactly one** unambiguous
top-level JSON object (optionally surrounded by whitespace or prose). It rejects
empty output, truncated objects, non-object JSON (arrays/scalars), and output
containing two or more competing top-level JSON objects.

The scanner is brace-aware *and* string-aware (it ignores braces inside JSON
string literals and respects backslash escapes), so a naive ``\\{.*\\}`` regex is
deliberately avoided.

Callers run their own schema validation on :attr:`JsonRecovery.data`; this module
only guarantees "one well-formed JSON object". :attr:`JsonRecovery.recovered`
records whether recovery beyond a strict whole-string parse was required, so
unfaithful formatting is surfaced as a diagnostic rather than hidden.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


class StructuredOutputError(ValueError):
    """Raised when output does not contain exactly one valid top-level JSON object."""


@dataclass(frozen=True)
class JsonRecovery:
    """The outcome of recovering one JSON object from raw model output."""

    data: dict[str, Any]
    recovered: bool
    raw: str


def _top_level_object_spans(text: str) -> tuple[list[tuple[int, int]], bool]:
    """Return (spans, had_unterminated).

    ``spans`` are ``(start, end)`` half-open indices of each balanced top-level
    ``{...}`` group, tracking JSON string state so braces inside strings are
    ignored. ``had_unterminated`` is True if a top-level ``{`` opened but never
    closed (a truncation signal).
    """

    spans: list[tuple[int, int]] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    had_unterminated = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append((start, i + 1))
                start = -1
    if depth > 0:
        had_unterminated = True
    return spans, had_unterminated


def recover_json_object(raw: str) -> JsonRecovery:
    """Recover exactly one top-level JSON object from ``raw``.

    Raises:
        StructuredOutputError: if the output is empty, not a JSON object, is
            truncated, or contains two or more competing top-level JSON objects.
    """

    text = raw.strip()
    if not text:
        raise StructuredOutputError("empty model output")

    # 1. Strict whole-string parse (the faithful, no-recovery path).
    try:
        whole = json.loads(text)
    except json.JSONDecodeError:
        whole = _SENTINEL
    if whole is not _SENTINEL:
        if isinstance(whole, dict):
            return JsonRecovery(data=whole, recovered=False, raw=raw)
        raise StructuredOutputError(f"top-level JSON is a {type(whole).__name__}, not an object")

    # 2. Recovery: find balanced top-level {...} groups that each parse as an object.
    spans, had_unterminated = _top_level_object_spans(text)
    candidates: list[dict[str, Any]] = []
    for s, e in spans:
        try:
            value = json.loads(text[s:e])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)

    if len(candidates) == 1:
        return JsonRecovery(data=candidates[0], recovered=True, raw=raw)
    if len(candidates) > 1:
        raise StructuredOutputError(
            f"output contains {len(candidates)} competing top-level JSON objects; ambiguous"
        )
    if had_unterminated:
        raise StructuredOutputError("truncated or unterminated JSON object")
    raise StructuredOutputError("no valid top-level JSON object found in output")


_SENTINEL: Any = object()

__all__ = ["JsonRecovery", "StructuredOutputError", "recover_json_object"]
