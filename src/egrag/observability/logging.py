"""Structured logging configuration.

Importing this module has no side effects: logging is configured only when
:func:`configure_logging` is called explicitly (e.g. by the CLI). Structured
context is attached via the ``extra`` mechanism of the standard library.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from egrag.security import redact_secrets

_LOGGER_NAMESPACE = "egrag"


def configure_logging(level: str = "INFO") -> None:
    """Configure the ``egrag`` logger namespace at the given level.

    Idempotent: repeated calls adjust the level without adding duplicate
    handlers. Does not touch the root logger's existing handlers.
    """

    logger = logging.getLogger(_LOGGER_NAMESPACE)
    numeric_level = logging.getLevelNamesMapping().get(level.upper())
    if numeric_level is None:
        raise ValueError(f"unknown log level: {level!r}")
    logger.setLevel(numeric_level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False


def get_logger(name: str, **context: Any) -> logging.LoggerAdapter[logging.Logger]:
    """Return a logger under the ``egrag`` namespace with bound context."""

    base = logging.getLogger(f"{_LOGGER_NAMESPACE}.{name}")
    bound: Mapping[str, Any] = dict(context)
    return logging.LoggerAdapter(base, dict(bound))


def log_event(
    logger: logging.Logger | logging.LoggerAdapter[Any], event: str, **fields: Any
) -> str:
    """Log a structured event, redacting secrets from all fields.

    Returns the rendered message (useful for tests). Callers must pass counts/IDs
    rather than raw document content (see :func:`document_log_fields`).
    """

    safe_fields = redact_secrets(fields)
    message = f"{event} {json.dumps(safe_fields, sort_keys=True, default=str)}"
    logger.info(message)
    return message


def document_log_fields(
    text: str, *, log_documents: bool = False, preview: int = 80
) -> dict[str, Any]:
    """Return safe log fields for a document.

    By default only the character count is included — never the content. When
    ``log_documents`` is explicitly enabled, a short preview is added.
    """

    fields: dict[str, Any] = {"chars": len(text)}
    if log_documents:
        fields["preview"] = text[:preview]
    return fields


__all__ = ["configure_logging", "document_log_fields", "get_logger", "log_event"]
