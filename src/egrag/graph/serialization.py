"""Versioned JSON serialization for evidence-graph snapshots.

Round-trips losslessly (every node, edge, score, and provenance field is
preserved). Malformed input raises a domain :class:`SerializationError` rather
than leaking a parser/validation exception.
"""

from __future__ import annotations

from pydantic import ValidationError as PydanticValidationError

from egrag.domain.errors import SerializationError
from egrag.domain.models import EvidenceGraphSnapshot


class GraphSerializer:
    """Serializes evidence-graph snapshots to and from JSON text."""

    def __init__(self, *, indent: int | None = None) -> None:
        self._indent = indent

    def serialize(self, snapshot: EvidenceGraphSnapshot) -> str:
        return snapshot.model_dump_json(indent=self._indent)

    def deserialize(self, data: str) -> EvidenceGraphSnapshot:
        try:
            return EvidenceGraphSnapshot.model_validate_json(data)
        except PydanticValidationError as exc:
            msg = f"malformed evidence graph: {exc.error_count()} validation error(s)"
            raise SerializationError(msg) from exc
        except ValueError as exc:
            raise SerializationError(f"invalid JSON for evidence graph: {exc}") from exc


__all__ = ["GraphSerializer"]
