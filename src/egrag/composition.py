"""The single composition root: build and run the pipeline from a config.

All wiring lives here (no scattered factories). Before any expensive
processing, :func:`validate_runtime` checks optional dependencies, model paths,
endpoint configuration, context limits, evidence budgets, deterministic-mode
capabilities, adapter compatibility, and cache configuration — failing fast with
a typed error, or returning non-fatal warnings (e.g. determinism).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from egrag.answering import answer_query
from egrag.config.schema import EGRagConfig
from egrag.domain.errors import (
    ArtifactWriteError,
    ConfigurationError,
    MissingDependencyError,
    ModelLoadError,
)
from egrag.domain.models import Document, PipelineResult, Query, SourceMetadata
from egrag.fakes import build_demo_documents
from egrag.generation import (
    FakeTextGenerator,
    GenerationConfig,
    HuggingFaceGenerator,
    OpenAICompatibleGenerator,
    TextGenerator,
    validate_config,
)
from egrag.hf_runtime import ensure_device_available
from egrag.reproducibility import build_run_manifest, now
from egrag.security import check_file_size, safe_artifact_path, validate_url, validate_within_base


def _require(module: str, extra: str) -> None:
    if importlib.util.find_spec(module) is None:
        raise MissingDependencyError(module, extra)


def _is_local_path(value: str) -> bool:
    return value.startswith(("/", "./", "../", "~", "\\")) or Path(value).is_absolute()


def to_generation_config(config: EGRagConfig) -> GenerationConfig:
    """Map the config's generation section to a runtime GenerationConfig."""

    gen = config.generation
    return GenerationConfig(
        deterministic=gen.deterministic,
        temperature=gen.temperature,
        top_p=gen.top_p,
        max_new_tokens=gen.max_new_tokens,
        seed=config.reproducibility.seed if gen.deterministic else None,
        timeout_s=gen.timeout_s,
        max_retries=gen.max_retries,
        stream=gen.stream,
    )


def build_generator(config: EGRagConfig) -> TextGenerator:
    """Construct the configured generator adapter (no model load at this point)."""

    gen = config.generation
    if gen.adapter == "fake":
        return FakeTextGenerator(context_limit=gen.context_limit)
    if gen.adapter == "openai":
        if not gen.base_url:
            raise ConfigurationError("generation.base_url is required for the openai adapter")
        validate_url(gen.base_url, allowed_schemes=frozenset(config.security.allowed_url_schemes))
        if not gen.model:
            raise ConfigurationError("generation.model is required for the openai adapter")
        _require("httpx", "http-models")
        return OpenAICompatibleGenerator(gen.base_url, gen.model, context_limit=gen.context_limit)
    if not gen.model:
        raise ConfigurationError("generation.model is required for the huggingface adapter")
    _require("transformers", "local-models")
    if gen.quantization != "none":
        _require("bitsandbytes", "quantization")
    if gen.require_cuda:
        ensure_device_available("cuda")
    return HuggingFaceGenerator(
        gen.model,
        context_limit=gen.context_limit,
        revision=gen.model_revision,
        device=gen.device,
        dtype=gen.dtype,
        quantization=gen.quantization,
        require_cuda=gen.require_cuda,
    )


def validate_runtime(config: EGRagConfig) -> list[str]:
    """Validate runtime prerequisites; raise typed errors or return warnings."""

    gen = config.generation
    # Model paths must exist when they look local (checked before dependencies).
    if gen.model and _is_local_path(gen.model) and not Path(gen.model).expanduser().exists():
        raise ModelLoadError(f"configured model path does not exist: {gen.model}")

    # Optional-dependency requirements for the chosen components.
    if config.retrieval.strategy in ("dense", "hybrid"):
        _require("sentence_transformers", "dense")
    if config.reranking.enabled and config.reranking.strategy == "cross_encoder":
        _require("sentence_transformers", "dense")
    if config.nli.classifier == "huggingface" or config.grounding_verification.use_nli:
        _require("transformers", "local-models")

    # Cache configuration.
    if config.caching.enabled and config.caching.backend == "disk":
        assert config.caching.directory is not None  # guaranteed by config validation
        safe_artifact_path(config.caching.directory, config.security.artifact_base)

    # Build the generator (cheap; no model load) and validate config vs capabilities.
    generator = build_generator(config)
    warnings = validate_config(to_generation_config(config), generator.capabilities())
    return warnings


def load_corpus(config: EGRagConfig) -> list[Document]:
    """Load the configured corpus (demo, or a validated local directory of .txt)."""

    if config.corpus.source == "demo":
        return build_demo_documents()
    if not config.corpus.path:
        raise ConfigurationError("corpus.path is required when corpus.source is 'directory'")
    base = Path(config.security.artifact_base).resolve()
    directory = validate_within_base(config.corpus.path, base)
    if not directory.is_dir():
        raise ConfigurationError(f"corpus directory does not exist: {directory}")

    documents: list[Document] = []
    for index, file in enumerate(sorted(directory.glob("*.txt"))):
        check_file_size(file, max_bytes=config.corpus.max_file_bytes)
        documents.append(
            Document(
                document_id=f"doc-{index}",
                text=file.read_text(encoding="utf-8"),
                source=SourceMetadata(source_id=file.stem, title=file.name),
            )
        )
    return documents


def run_pipeline(
    config: EGRagConfig,
    query: Query,
    *,
    artifact_dir: str | None = None,
) -> PipelineResult:
    """Validate, build, and run the full pipeline for ``query`` from ``config``."""

    started = now()
    warnings = validate_runtime(config)
    generator = build_generator(config)
    documents = load_corpus(config)

    result = answer_query(
        query,
        documents,
        generator=generator,
        config=to_generation_config(config),
        top_k=config.retrieval.top_k,
        evidence_token_budget=config.generation.evidence_token_budget,
        reserved_output_tokens=config.generation.reserved_output_tokens,
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.overlap,
    )

    artifact_paths: list[str] = []
    if artifact_dir is not None:
        base = Path(config.security.artifact_base).resolve()
        target_dir = safe_artifact_path(artifact_dir, base)
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            package_path = target_dir / "evidence_package.json"
            package_path.write_text(result.package.model_dump_json(indent=2), encoding="utf-8")
            artifact_paths.append(str(package_path))
        except OSError as exc:
            raise ArtifactWriteError(f"failed to write artifacts: {exc}") from exc

    ended = now()
    manifest = build_run_manifest(
        config,
        component_identities={
            "generator": type(generator).__name__,
            "retriever": "BM25Retriever",
            "extractor": "SentenceClaimExtractor",
        },
        model_identifiers={"generator": config.generation.model or config.generation.adapter},
        model_revisions=(
            {"generator": config.generation.model_revision}
            if config.generation.model_revision
            else {}
        ),
        prompt_versions={"extraction": config.extraction.prompt_version},
        corpus_texts=[doc.text for doc in documents],
        started_at=started,
        ended_at=ended,
        artifact_paths=artifact_paths,
        warnings=warnings,
        deterministic_capability=generator.capabilities().deterministic_decoding,
    )
    return result.model_copy(update={"manifest": manifest})


__all__ = [
    "build_generator",
    "load_corpus",
    "run_pipeline",
    "to_generation_config",
    "validate_runtime",
]
