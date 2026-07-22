"""Structured run metrics (counts, durations, cache effectiveness)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from pydantic import BaseModel, ConfigDict, Field


class RunObservation(BaseModel):
    """Aggregated metrics for one pipeline run (no document content)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage_durations_ms: dict[str, float] = Field(default_factory=dict)
    num_passages: int = Field(default=0, ge=0)
    num_claims: int = Field(default=0, ge=0)
    num_candidate_pairs: int = Field(default=0, ge=0)
    num_graph_nodes: int = Field(default=0, ge=0)
    num_graph_edges: int = Field(default=0, ge=0)
    propagation_iterations: int = Field(default=0, ge=0)
    num_conflicts: int = Field(default=0, ge=0)
    num_selected: int = Field(default=0, ge=0)
    token_estimate: int = Field(default=0, ge=0)
    generation_latency_ms: float = Field(default=0.0, ge=0.0)
    validation_warnings: int = Field(default=0, ge=0)
    cache_hits: int = Field(default=0, ge=0)
    cache_misses: int = Field(default=0, ge=0)


class StageTimer:
    """Accumulates per-stage wall-clock durations in milliseconds."""

    def __init__(self) -> None:
        self.durations_ms: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.durations_ms[name] = (time.perf_counter() - start) * 1000.0


__all__ = ["RunObservation", "StageTimer"]
