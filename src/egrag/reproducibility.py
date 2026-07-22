"""Run-manifest construction for reproducibility.

Captures everything needed to reproduce a run: framework + schema versions, git
commit (when available), resolved configuration, seeds, component
implementations, model identifiers/revisions, prompt versions, corpus
fingerprint, timestamps, environment, artifact paths, warnings, and the
deterministic-capability status. Gathering this never touches the network and
never records secrets.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from egrag import __version__
from egrag.config.schema import EGRagConfig
from egrag.domain.models import RunManifest
from egrag.domain.version import SCHEMA_VERSION
from egrag.security import redact_secrets


def git_commit() -> str | None:
    """Return the current git commit hash, or ``None`` when unavailable."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment dependent
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def environment_info() -> dict[str, str]:
    """Return non-secret environment details for the manifest."""

    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }


def corpus_fingerprint(texts: Sequence[str]) -> str:
    """Return a stable fingerprint over an ordered collection of texts."""

    digest = hashlib.sha256()
    for text in texts:
        digest.update(hashlib.sha256(text.encode("utf-8")).digest())
    return digest.hexdigest()


def build_run_manifest(
    config: EGRagConfig,
    *,
    component_identities: Mapping[str, str],
    model_identifiers: Mapping[str, str] | None = None,
    model_revisions: Mapping[str, str] | None = None,
    prompt_versions: Mapping[str, str] | None = None,
    corpus_texts: Sequence[str] | None = None,
    started_at: datetime,
    ended_at: datetime,
    artifact_paths: Sequence[str] = (),
    warnings: Sequence[str] = (),
    deterministic_capability: bool = True,
) -> RunManifest:
    """Build a fully populated :class:`RunManifest`."""

    resolved = redact_secrets(config.model_dump(mode="json"))
    return RunManifest(
        egrag_version=__version__,
        schema_version=SCHEMA_VERSION,
        seed=config.reproducibility.seed,
        deterministic=config.reproducibility.deterministic,
        created_at=started_at,
        started_at=started_at,
        ended_at=ended_at,
        component_identities=dict(component_identities),
        git_commit=git_commit(),
        environment=environment_info(),
        model_identifiers=dict(model_identifiers or {}),
        model_revisions=dict(model_revisions or {}),
        prompt_versions=dict(prompt_versions or {}),
        corpus_fingerprint=corpus_fingerprint(corpus_texts) if corpus_texts else None,
        artifact_paths=tuple(artifact_paths),
        warnings=tuple(warnings),
        deterministic_capability=deterministic_capability,
        resolved_config=resolved,
    )


def now() -> datetime:
    """Return the current UTC time (helper for explicit timestamps)."""

    return datetime.now(UTC)


__all__ = [
    "build_run_manifest",
    "corpus_fingerprint",
    "environment_info",
    "git_commit",
    "now",
]
