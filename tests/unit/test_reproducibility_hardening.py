"""Reproducibility-manifest acceptance tests (19, 20)."""

from __future__ import annotations

import pytest

from egrag.config import config_from_yaml
from egrag.reproducibility import build_run_manifest, now


def _manifest():
    config = config_from_yaml("experiments:\n  name: t\ngeneration:\n  adapter: fake")
    started = now()
    return build_run_manifest(
        config,
        component_identities={"generator": "FakeTextGenerator"},
        model_identifiers={"generator": "fake"},
        prompt_versions={"extraction": "extraction_v1"},
        corpus_texts=["doc one", "doc two"],
        started_at=started,
        ended_at=now(),
        artifact_paths=["out/evidence_package.json"],
        warnings=["a warning"],
        deterministic_capability=True,
    )


@pytest.mark.unit
def test_manifest_includes_required_fields() -> None:
    """Acceptance 19: run manifests include required reproducibility fields."""

    manifest = _manifest()
    assert manifest.egrag_version
    assert manifest.schema_version
    assert manifest.environment["python"]
    assert manifest.started_at is not None and manifest.ended_at is not None
    assert manifest.corpus_fingerprint
    assert manifest.component_identities["generator"] == "FakeTextGenerator"
    assert manifest.prompt_versions["extraction"] == "extraction_v1"
    assert manifest.artifact_paths == ("out/evidence_package.json",)
    assert manifest.warnings == ("a warning",)
    assert manifest.deterministic_capability is True
    # git commit is optional ("when available")
    assert manifest.git_commit is None or isinstance(manifest.git_commit, str)


@pytest.mark.unit
def test_manifest_preserves_resolved_config() -> None:
    """Acceptance 20: run manifests preserve the resolved configuration."""

    manifest = _manifest()
    assert manifest.resolved_config["experiments"]["name"] == "t"
    assert manifest.resolved_config["generation"]["adapter"] == "fake"
    # round-trips through JSON (manifest is serializable)
    from egrag.domain.models import RunManifest

    assert RunManifest.model_validate_json(manifest.model_dump_json()) == manifest
