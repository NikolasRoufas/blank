"""Concrete adapters implementing domain ports over third-party libraries.

Each adapter subpackage wraps an optional dependency installed via an extra.
Importing this package must not import any optional dependency; concrete
adapters import their backing library lazily inside their own modules.

Real retrieval and model-inference adapters are introduced in later milestones.
"""

from __future__ import annotations

__all__: list[str] = []
