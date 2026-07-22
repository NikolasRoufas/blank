"""Hierarchical, strict, validated configuration loaded from safe YAML.

Every section is a frozen Pydantic model that **forbids unknown fields** (the
documented strictness policy): a typo or stray key fails fast rather than being
silently ignored. Cross-section consistency is validated up front so invalid
settings fail before any expensive processing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from egrag.domain.errors import ConfigurationError
from egrag.security import DEFAULT_MAX_FILE_BYTES, safe_load_yaml


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CorpusConfig(_Section):
    source: Literal["demo", "directory"] = "demo"
    path: str | None = None
    max_file_bytes: int = Field(default=DEFAULT_MAX_FILE_BYTES, ge=1)


class ChunkingConfig(_Section):
    strategy: Literal["sentence", "whole_document"] = "sentence"
    chunk_size: int = Field(default=512, ge=1)
    overlap: int = Field(default=64, ge=0)

    @model_validator(mode="after")
    def _overlap(self) -> ChunkingConfig:
        if self.overlap >= self.chunk_size:
            raise ValueError("chunking.overlap must be smaller than chunking.chunk_size")
        return self


class RetrievalConfig(_Section):
    strategy: Literal["bm25", "dense", "hybrid"] = "bm25"
    top_k: int = Field(default=4, ge=1)
    bm25_k1: float = Field(default=1.5, ge=0.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)
    dense_model: str | None = None
    fusion: Literal["weighted", "rrf"] = "weighted"


class RerankingConfig(_Section):
    enabled: bool = False
    strategy: Literal["score", "cross_encoder"] = "score"
    top_n: int = Field(default=4, ge=1)
    model: str | None = None


class ExtractionConfig(_Section):
    method: Literal["sentence_baseline", "structured"] = "sentence_baseline"
    prompt_version: str = "extraction_v1"
    model: str | None = None
    deduplicate: bool = True
    resolve_clear_pronouns: bool = False


class GraphConstructionConfig(_Section):
    enabled: bool = True
    store_neutral: bool = False


class CandidateGenerationConfig(_Section):
    strategy: Literal["pruned", "brute_force"] = "pruned"
    max_pairs: int | None = Field(default=None, ge=0)
    lexical_overlap_threshold: float = Field(default=0.2, ge=0.0, le=1.0)


class NLIConfig(_Section):
    classifier: Literal["lexical", "huggingface", "fake"] = "lexical"
    model: str | None = None
    model_revision: str | None = None
    entailment_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    contradiction_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    duplicate_threshold: float = Field(default=0.8, ge=0.0, le=1.0)


class TemporalReasoningConfig(_Section):
    enabled: bool = True
    min_update_confidence: float = Field(default=0.6, ge=0.0, le=1.0)


class SourceReliabilityConfig(_Section):
    strategy: Literal["uniform", "configured", "metadata"] = "metadata"
    default: float = Field(default=0.5, ge=0.0, le=1.0)
    priors: dict[str, float] = Field(default_factory=dict)


class InitialScoringConfig(_Section):
    retrieval_weight: float = Field(default=0.5, ge=0.0)
    query_relevance_weight: float = Field(default=1.0, ge=0.0)
    extraction_weight: float = Field(default=1.0, ge=0.0)
    source_reliability_weight: float = Field(default=1.0, ge=0.0)
    temporal_validity_weight: float = Field(default=0.5, ge=0.0)
    independent_support_weight: float = Field(default=1.0, ge=0.0)
    superseded_validity: float = Field(default=0.5, ge=0.0, le=1.0)
    default_retrieval: float = Field(default=0.5, ge=0.0, le=1.0)


class PropagationConfig(_Section):
    enabled: bool = True
    damping: float = Field(default=0.5, ge=0.0, lt=1.0)
    tolerance: float = Field(default=1e-4, gt=0.0)
    max_iterations: int = Field(default=50, ge=1)
    support_weight: float = Field(default=1.0, ge=0.0)
    contradiction_weight: float = Field(default=1.0, ge=0.0)
    duplicate_discount: float = Field(default=0.0, ge=0.0, le=1.0)
    lineage_discount: float = Field(default=0.25, ge=0.0, le=1.0)
    on_nonconvergence: Literal["raise", "return"] = "raise"


class ConflictResolutionConfig(_Section):
    margin: float = Field(default=0.1, ge=0.0, le=1.0)
    low_evidence_threshold: float = Field(default=0.15, ge=0.0, le=1.0)


class SubgraphSelectionConfig(_Section):
    strategy: Literal["top_claims", "greedy_connected", "beam"] = "greedy_connected"
    beam_width: int = Field(default=4, ge=1)
    utility_weight: float = Field(default=1.0, ge=0.0)
    belief_weight: float = Field(default=1.0, ge=0.0)


class SerializationConfig(_Section):
    renderer: Literal["plain", "markdown", "chat", "json"] = "plain"
    round_scores: bool = True
    score_decimals: int = Field(default=2, ge=0, le=6)


class GenerationConfig(_Section):
    adapter: Literal["fake", "huggingface", "openai"] = "fake"
    model: str | None = None
    base_url: str | None = None
    context_limit: int = Field(default=8192, ge=1)
    evidence_token_budget: int = Field(default=512, ge=1)
    reserved_output_tokens: int = Field(default=128, ge=0)
    deterministic: bool = True
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_new_tokens: int = Field(default=512, ge=1)
    timeout_s: float = Field(default=30.0, gt=0.0)
    max_retries: int = Field(default=2, ge=0)
    stream: bool = False

    @model_validator(mode="after")
    def _context_fits(self) -> GenerationConfig:
        if self.evidence_token_budget + self.max_new_tokens > self.context_limit:
            raise ValueError(
                "generation.evidence_token_budget + max_new_tokens exceeds context_limit"
            )
        if self.reserved_output_tokens > self.evidence_token_budget:
            raise ValueError(
                "generation.reserved_output_tokens must not exceed evidence_token_budget"
            )
        return self


class GroundingVerificationConfig(_Section):
    enabled: bool = True
    use_nli: bool = False
    nli_model: str | None = None


class CachingConfig(_Section):
    enabled: bool = False
    backend: Literal["memory", "disk", "null"] = "memory"
    directory: str | None = None

    @model_validator(mode="after")
    def _disk_needs_directory(self) -> CachingConfig:
        if self.enabled and self.backend == "disk" and not self.directory:
            raise ValueError("caching.directory is required when backend is 'disk'")
        return self


class LoggingConfig(_Section):
    level: str = "INFO"
    log_documents: bool = False


class SecurityConfig(_Section):
    max_file_bytes: int = Field(default=DEFAULT_MAX_FILE_BYTES, ge=1)
    allowed_url_schemes: tuple[str, ...] = ("http", "https")
    allow_remote_model_code: bool = False
    artifact_base: str = "."


class ReproducibilityConfig(_Section):
    seed: int = Field(default=0, ge=0)
    deterministic: bool = True
    write_manifest: bool = True
    manifest_path: str | None = None


class ExperimentsConfig(_Section):
    name: str = "baseline"
    tags: tuple[str, ...] = ()
    ablation: (
        Literal["passage_rag", "claim_only", "no_graph", "no_propagation", "top_claims"] | None
    ) = None


class EGRagConfig(BaseModel):
    """The full, validated EG-RAG configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    corpus: CorpusConfig = CorpusConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    reranking: RerankingConfig = RerankingConfig()
    extraction: ExtractionConfig = ExtractionConfig()
    graph_construction: GraphConstructionConfig = GraphConstructionConfig()
    candidate_generation: CandidateGenerationConfig = CandidateGenerationConfig()
    nli: NLIConfig = NLIConfig()
    temporal_reasoning: TemporalReasoningConfig = TemporalReasoningConfig()
    source_reliability: SourceReliabilityConfig = SourceReliabilityConfig()
    initial_scoring: InitialScoringConfig = InitialScoringConfig()
    propagation: PropagationConfig = PropagationConfig()
    conflict_resolution: ConflictResolutionConfig = ConflictResolutionConfig()
    subgraph_selection: SubgraphSelectionConfig = SubgraphSelectionConfig()
    serialization: SerializationConfig = SerializationConfig()
    generation: GenerationConfig = GenerationConfig()
    grounding_verification: GroundingVerificationConfig = GroundingVerificationConfig()
    caching: CachingConfig = CachingConfig()
    logging: LoggingConfig = LoggingConfig()
    security: SecurityConfig = SecurityConfig()
    reproducibility: ReproducibilityConfig = ReproducibilityConfig()
    experiments: ExperimentsConfig = ExperimentsConfig()

    @model_validator(mode="after")
    def _cross_section(self) -> EGRagConfig:
        if self.reranking.enabled and self.reranking.top_n > self.retrieval.top_k:
            raise ValueError("reranking.top_n cannot exceed retrieval.top_k")
        return self


def config_from_yaml(text: str) -> EGRagConfig:
    """Parse and validate configuration from a YAML string."""

    data = safe_load_yaml(text)
    try:
        return EGRagConfig.model_validate(data)
    except ValueError as exc:
        raise ConfigurationError(f"invalid configuration: {exc}") from exc


def load_config(path: str | Path) -> EGRagConfig:
    """Load and validate configuration from a YAML file."""

    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigurationError(f"configuration file not found: {file_path}")
    return config_from_yaml(file_path.read_text(encoding="utf-8"))


__all__ = [
    "CachingConfig",
    "CandidateGenerationConfig",
    "ChunkingConfig",
    "ConflictResolutionConfig",
    "CorpusConfig",
    "EGRagConfig",
    "ExperimentsConfig",
    "ExtractionConfig",
    "GenerationConfig",
    "GraphConstructionConfig",
    "GroundingVerificationConfig",
    "InitialScoringConfig",
    "LoggingConfig",
    "NLIConfig",
    "PropagationConfig",
    "ReproducibilityConfig",
    "RerankingConfig",
    "RetrievalConfig",
    "SecurityConfig",
    "SerializationConfig",
    "SourceReliabilityConfig",
    "SubgraphSelectionConfig",
    "TemporalReasoningConfig",
    "config_from_yaml",
    "load_config",
]
