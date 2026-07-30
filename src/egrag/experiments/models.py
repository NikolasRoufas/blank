"""Typed abstractions for the evaluation harness.

This layer is **separate from the production pipeline**: it imports and reuses
production components but adds no evaluation logic to them. Gold answers and gold
evidence live only in :class:`DatasetExample`/:class:`GoldEvidence` and are never
passed to a system variant during generation (see :mod:`egrag.experiments.variants`).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from egrag.domain.models import Document
from egrag.domain.version import SCHEMA_VERSION

Family = Literal["passage", "claim", "graph"]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GoldEvidence(_Model):
    """Reference evidence for an example. ``available=False`` means unknown gold."""

    available: bool = True
    source_ids: tuple[str, ...] = ()
    facts: tuple[str, ...] = ()
    stance: Literal["supports", "refutes", "not_enough_info"] | None = None

    @property
    def has_evidence(self) -> bool:
        return self.available and bool(self.source_ids)


class DatasetExample(_Model):
    """One evaluation example with gold answers and gold evidence."""

    example_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    documents: tuple[Document, ...]
    gold_answers: tuple[str, ...] = ()
    gold_evidence: GoldEvidence = GoldEvidence()
    split: str = "test"
    metadata: dict[str, str] = Field(default_factory=dict)


class SystemVariant(_Model):
    """A controlled system configuration that isolates an intended component.

    ``generator_override``/``top_k_override``/``evidence_budget_override`` exist
    only so fairness checks can *detect* an unfair difference; fair experiments
    leave them ``None`` so every variant shares the experiment-level settings.
    """

    name: str = Field(min_length=1)
    description: str = ""
    family: Family = "graph"
    rerank: bool = False
    propagation: bool = True
    selection: Literal["top", "greedy"] = "greedy"
    temporal: bool = True
    contradiction: bool = True
    isolates: tuple[str, ...] = ()
    generator_override: str | None = None
    top_k_override: int | None = None
    evidence_budget_override: int | None = None


class ExperimentConfig(_Model):
    """Configuration for an experiment run (configuration-driven, inspectable).

    ``generator`` selects the adapter: ``"fake"`` (deterministic, default) or
    ``"huggingface"`` (a real local model, e.g. Qwen). The ``generator_*`` fields
    below apply only to the ``"huggingface"`` adapter and are otherwise ignored;
    they are recorded verbatim in :class:`ExperimentManifest` either way so a run
    that used the fake generator is never confused with one that used a real
    model.
    """

    name: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    dataset_path: str | None = None
    variants: tuple[str, ...] = Field(min_length=1)
    seeds: tuple[int, ...] = (0,)
    output_dir: str = Field(min_length=1)
    generator: Literal["fake", "huggingface"] = "fake"
    generator_model: str | None = None
    generator_revision: str | None = None
    generator_device: str = "auto"
    generator_dtype: str | None = None
    generator_quantization: Literal["none", "4bit", "8bit"] = "none"
    generator_disable_thinking: bool = False
    require_cuda: bool = False
    top_k: int = Field(default=4, ge=1)
    evidence_token_budget: int = Field(default=256, ge=1)
    reserved_output_tokens: int = Field(default=64, ge=0)
    max_new_tokens: int = Field(default=512, ge=1)
    chunk_size: int = Field(default=512, ge=1)
    chunk_overlap: int = Field(default=64, ge=0)
    limit: int | None = Field(default=None, ge=1)
    enforce_fairness: bool = True
    bootstrap_samples: int = Field(default=1000, ge=1)
    bootstrap_seed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _huggingface_needs_model(self) -> ExperimentConfig:
        if self.generator == "huggingface" and not self.generator_model:
            raise ValueError("generator_model is required when generator is 'huggingface'")
        if self.generator_quantization != "none" and self.generator != "huggingface":
            raise ValueError("generator_quantization requires generator: huggingface")
        return self


class ExampleResult(_Model):
    """The result of running one variant on one example at one seed."""

    example_id: str
    variant: str
    seed: int
    answer: str = ""
    citations: tuple[str, ...] = ()
    evidence_source_ids: tuple[str, ...] = ()
    metrics: dict[str, float] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    latency_ms: float = 0.0
    peak_memory_kb: int | None = None
    failed: bool = False
    error: str | None = None
    warnings: tuple[str, ...] = ()


class AggregateMetrics(_Model):
    """Aggregated metrics for one variant at one seed; preserves failure counts."""

    variant: str
    seed: int
    num_examples: int = Field(ge=0)
    num_failed: int = Field(ge=0)
    metrics: dict[str, float] = Field(default_factory=dict)
    metric_kinds: dict[str, str] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()


class ExperimentManifest(_Model):
    """Reproducibility metadata for an experiment run."""

    experiment_name: str
    schema_version: str = SCHEMA_VERSION
    egrag_version: str
    dataset: str
    dataset_fingerprint: str
    num_examples: int = Field(ge=0)
    variants: tuple[SystemVariant, ...]
    seeds: tuple[int, ...]
    generator: str
    generator_model: str | None = None
    generator_revision: str | None = None
    generator_device: str | None = None
    generator_dtype: str | None = None
    generator_quantization: str = "none"
    generator_disable_thinking: bool = False
    generator_resolved: dict[str, str] = Field(default_factory=dict)
    top_k: int
    evidence_token_budget: int
    reserved_output_tokens: int
    fairness_enforced: bool
    started_at: datetime
    ended_at: datetime
    git_commit: str | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    resolved_config: dict[str, Any] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()


class ComparisonResult(_Model):
    """A paired bootstrap comparison of two variants on one metric."""

    metric: str
    variant_a: str
    variant_b: str
    n_paired: int = Field(ge=0)
    mean_a: float
    mean_b: float
    mean_delta: float
    ci_low: float
    ci_high: float
    bootstrap_samples: int
    seed: int
    method: str = "paired percentile bootstrap (no significance claim)"


__all__ = [
    "AggregateMetrics",
    "ComparisonResult",
    "DatasetExample",
    "ExampleResult",
    "ExperimentConfig",
    "ExperimentManifest",
    "Family",
    "GoldEvidence",
    "SystemVariant",
]
