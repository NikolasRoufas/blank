"""Domain layer: pure models, ports, and graph algorithms.

This layer must remain free of infrastructure dependencies. It imports only the
standard library and Pydantic. It must never import NetworkX, Transformers,
sentence-transformers, HTTP clients, vector databases, or storage libraries.
"""

from __future__ import annotations

__all__: list[str] = []
