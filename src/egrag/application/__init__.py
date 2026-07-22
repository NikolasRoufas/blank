"""Application layer: pipeline orchestration over domain ports.

This layer depends only on domain models and ports — never on concrete adapter
or fake implementations, and never on infrastructure libraries.
"""

from __future__ import annotations

from egrag.application.pipeline import EGRagPipeline, PipelineComponents
from egrag.application.retrieval import retrieve_and_rerank

__all__ = ["EGRagPipeline", "PipelineComponents", "retrieve_and_rerank"]
