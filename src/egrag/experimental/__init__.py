"""Experimental research algorithms — unstable API.

This package holds research-grade algorithms (belief propagation, conflict
resolution, subgraph selection) whose methodology is not yet validated. Code
here is reachable from the application only through the stable ports in
:mod:`egrag.domain.ports` and is excluded from semantic-versioning guarantees.

Concrete experimental algorithms are introduced in later milestones.
"""

from __future__ import annotations

EXPERIMENTAL_NOTICE: str = (
    "Experimental EG-RAG algorithm: methodology is not yet validated and the "
    "API may change without notice."
)

__all__ = ["EXPERIMENTAL_NOTICE"]
