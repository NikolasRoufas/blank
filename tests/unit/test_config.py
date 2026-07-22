"""Unit tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from egrag.config import EGRagSettings, load_settings
from egrag.domain.errors import ConfigurationError


@pytest.mark.unit
def test_defaults_are_valid() -> None:
    settings = EGRagSettings()
    assert settings.seed == 0
    assert settings.top_k >= 1
    assert settings.deterministic is True


@pytest.mark.unit
def test_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EGRAG_SEED", "11")
    monkeypatch.setenv("EGRAG_TOP_K", "3")
    settings = load_settings()
    assert settings.seed == 11
    assert settings.top_k == 3


@pytest.mark.unit
def test_invalid_configuration_raises_domain_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EGRAG_TOP_K", "0")  # below the minimum of 1
    with pytest.raises(ConfigurationError):
        load_settings()


@pytest.mark.unit
def test_env_file_is_loaded(tmp_path: Path) -> None:
    env_file = tmp_path / "custom.env"
    env_file.write_text("EGRAG_SEED=42\nEGRAG_SELECTION_BUDGET=2\n", encoding="utf-8")
    settings = load_settings(str(env_file))
    assert settings.seed == 42
    assert settings.selection_budget == 2
