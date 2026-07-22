"""Composition-root acceptance tests (4, 5, 6, 28, 29)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from egrag.composition import run_pipeline, validate_runtime
from egrag.config import config_from_yaml
from egrag.domain.errors import (
    ConfigurationError,
    MissingDependencyError,
    ModelLoadError,
    SecurityError,
)
from egrag.domain.models import Query
from egrag.generation import GenerationConfig, GeneratorCapabilities, validate_config


@pytest.mark.unit
def test_missing_optional_dependency_is_actionable() -> None:
    """Acceptance 4: missing optional dependencies give install instructions."""

    if importlib.util.find_spec("transformers") is not None:
        pytest.skip("transformers is installed; the missing-dependency path is not exercised")
    config = config_from_yaml("nli:\n  classifier: huggingface")
    with pytest.raises(MissingDependencyError) as excinfo:
        validate_runtime(config)
    message = str(excinfo.value)
    assert "egrag[local-models]" in message  # actionable install command


@pytest.mark.unit
def test_unavailable_model_path_fails_early() -> None:
    """Acceptance 5: unavailable model paths fail early."""

    config = config_from_yaml(
        "generation:\n  adapter: huggingface\n  model: /definitely/not/a/real/model/path"
    )
    with pytest.raises(ModelLoadError):
        validate_runtime(config)


@pytest.mark.unit
def test_openai_endpoint_requires_base_url() -> None:
    """Acceptance 6 (endpoint config): the openai adapter needs a valid base URL."""

    with pytest.raises(ConfigurationError):
        validate_runtime(config_from_yaml("generation:\n  adapter: openai\n  model: m"))


@pytest.mark.unit
def test_deterministic_mode_warns_when_unsupported() -> None:
    """Acceptance 28: deterministic mode warns when an adapter cannot guarantee it."""

    caps = GeneratorCapabilities(deterministic_decoding=False)
    warnings = validate_config(GenerationConfig(deterministic=True), caps)
    assert any("deterministic" in w.lower() for w in warnings)


@pytest.mark.unit
def test_artifacts_written_only_to_validated_paths(tmp_path: Path) -> None:
    """Acceptance 29: artifacts are written only to validated paths."""

    # artifact_base defaults to "."; an escaping artifact_dir must be rejected.
    config = config_from_yaml("generation:\n  adapter: fake")
    query = Query(query_id="q", text="how does dense retrieval rank passages")
    with pytest.raises(SecurityError):
        run_pipeline(config, query, artifact_dir="../escape")


@pytest.mark.unit
def test_runtime_validation_passes_for_default_config() -> None:
    config = config_from_yaml("generation:\n  adapter: fake")
    assert validate_runtime(config) == []
