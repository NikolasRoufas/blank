"""EG-RAG: Evidence Graph Retrieval-Augmented Generation.

This top-level package performs no expensive work at import time: it does not
load models, open network connections, read a corpus, or import optional
dependencies. Submodules are imported explicitly by callers.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
