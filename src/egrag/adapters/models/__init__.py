"""Model-inference adapters.

Local inference (Transformers, ``local-models`` extra) and OpenAI-compatible
HTTP inference (``http-models`` extra) adapters will live here and implement the
:class:`egrag.domain.ports.Generator` protocol. They are introduced in a later
milestone; no real model inference is implemented yet.
"""

from __future__ import annotations

__all__: list[str] = []
