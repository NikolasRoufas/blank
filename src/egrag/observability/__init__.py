"""Observability: structured logging, metrics, and private-content-safe helpers.

Importing this package has no side effects (no logging configured, no I/O).
"""

from __future__ import annotations

from egrag.observability.logging import (
    configure_logging,
    document_log_fields,
    get_logger,
    log_event,
)
from egrag.observability.metrics import RunObservation, StageTimer

__all__ = [
    "RunObservation",
    "StageTimer",
    "configure_logging",
    "document_log_fields",
    "get_logger",
    "log_event",
]
