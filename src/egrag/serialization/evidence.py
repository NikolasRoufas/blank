"""JSON serialization for :class:`EvidencePackage`.

Serialization is versioned (the schema version travels inside the package) and
round-trips losslessly. Deserialization validates input and raises a domain
:class:`SerializationError` on malformed data rather than leaking a raw parser
or validation exception.
"""

from __future__ import annotations

from pydantic import ValidationError as PydanticValidationError

from egrag.domain.errors import SerializationError
from egrag.domain.models import EvidencePackage


class JsonEvidenceSerializer:
    """Serializes evidence packages to and from JSON text.

    Implements the :class:`egrag.domain.ports.EvidenceSerializer` protocol.
    """

    def __init__(self, *, indent: int | None = None) -> None:
        self._indent = indent

    def serialize(self, package: EvidencePackage) -> str:
        """Return a JSON string representing ``package``."""

        return package.model_dump_json(indent=self._indent)

    def deserialize(self, data: str) -> EvidencePackage:
        """Parse and validate a JSON string into an :class:`EvidencePackage`.

        Raises:
            SerializationError: if ``data`` is not valid JSON or does not
                satisfy the evidence-package schema.
        """

        try:
            return EvidencePackage.model_validate_json(data)
        except PydanticValidationError as exc:
            msg = f"malformed evidence package: {exc.error_count()} validation error(s)"
            raise SerializationError(msg) from exc
        except ValueError as exc:  # invalid JSON syntax
            raise SerializationError(f"invalid JSON for evidence package: {exc}") from exc


__all__ = ["JsonEvidenceSerializer"]
