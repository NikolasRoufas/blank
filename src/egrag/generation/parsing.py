"""Parse raw generation output into a structured, validated answer.

The output contract is JSON: ``{"answer": str, "citations": [str...],
"uncertainty": str}``. Empty or malformed output raises a typed
:class:`GenerationError`; invalid attribution is never silently accepted.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from egrag.domain.errors import GenerationError
from egrag.structured_json import StructuredOutputError, recover_json_object


class ParsedAnswer(BaseModel):
    """A parsed generation result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    answer: str
    cited_claim_ids: tuple[str, ...] = ()
    uncertainty: str = ""
    structured: dict[str, Any] | None = None
    recovered: bool = False
    """True if the JSON object had to be recovered from surrounding prose
    (valid output but unfaithful formatting) rather than being the whole output."""


def parse_generation(raw: str) -> ParsedAnswer:
    """Parse the JSON output contract into a :class:`ParsedAnswer`.

    Accepts exactly one unambiguous top-level JSON object (optionally surrounded
    by prose); rejects empty, truncated, non-object, multiple-object, or
    schema-invalid output. Recovery is surfaced via :attr:`ParsedAnswer.recovered`.
    """

    try:
        recovery = recover_json_object(raw)
    except StructuredOutputError as exc:
        raise GenerationError(f"malformed structured output: {exc}") from exc
    data = recovery.data

    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise GenerationError("model output has an empty or missing 'answer'")

    citations = data.get("citations", [])
    if not isinstance(citations, list) or not all(isinstance(c, str) for c in citations):
        raise GenerationError("model output 'citations' must be a list of strings")

    uncertainty = data.get("uncertainty", "")
    if not isinstance(uncertainty, str):
        raise GenerationError("model output 'uncertainty' must be a string")

    # Normalize each citation to a bare claim id, then deduplicate preserving order.
    # Evidence claims are labelled ``[id]`` in the prompt, so a bracketed citation
    # ``"[id]"`` refers unambiguously to claim ``id`` — strip one surrounding pair
    # of square brackets. This maps the model's inline-citation style to the id; it
    # does not fabricate a citation (an unknown id is still rejected downstream).
    seen: set[str] = set()
    unique: list[str] = []
    for raw_cid in citations:
        cid = raw_cid.strip()
        if len(cid) >= 2 and cid.startswith("[") and cid.endswith("]"):
            cid = cid[1:-1].strip()
        if cid and cid not in seen:
            seen.add(cid)
            unique.append(cid)

    return ParsedAnswer(
        answer=answer.strip(),
        cited_claim_ids=tuple(unique),
        uncertainty=uncertainty.strip(),
        structured=data,
        recovered=recovery.recovered,
    )


__all__ = ["ParsedAnswer", "parse_generation"]
