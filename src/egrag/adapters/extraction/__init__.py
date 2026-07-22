"""Atomic claim extraction adapters.

Importing this package pulls in no optional dependency; the Hugging Face backend
loads its model lazily on first use.
"""

from __future__ import annotations

from egrag.adapters.extraction.baseline import SentenceClaimExtractor
from egrag.adapters.extraction.huggingface import HuggingFaceStructuredModel
from egrag.adapters.extraction.interfaces import (
    ExtractionConfig,
    ExtractionResult,
    StructuredModel,
)
from egrag.adapters.extraction.prompts_loader import load_prompt
from egrag.adapters.extraction.structured import (
    RawExtractedClaim,
    StructuredClaimExtractor,
    StructuredExtractionOutput,
)

__all__ = [
    "ExtractionConfig",
    "ExtractionResult",
    "HuggingFaceStructuredModel",
    "RawExtractedClaim",
    "SentenceClaimExtractor",
    "StructuredClaimExtractor",
    "StructuredExtractionOutput",
    "StructuredModel",
    "load_prompt",
]
