"""Configuration acceptance tests (1, 2, 3, 6)."""

from __future__ import annotations

import glob

import pytest

from egrag.config import config_from_yaml, load_config
from egrag.domain.errors import ConfigurationError


@pytest.mark.unit
def test_invalid_yaml_fails_safely() -> None:
    """Acceptance 1: invalid YAML fails safely (typed error, no execution)."""

    with pytest.raises(ConfigurationError):
        config_from_yaml("retrieval: [unclosed")
    with pytest.raises(ConfigurationError):
        config_from_yaml("- just\n- a\n- list")  # root not a mapping


@pytest.mark.unit
def test_unknown_fields_are_forbidden() -> None:
    """Acceptance 2: unknown fields follow the documented (strict) policy."""

    with pytest.raises(ConfigurationError):
        config_from_yaml("not_a_section: 1")
    with pytest.raises(ConfigurationError):
        config_from_yaml("retrieval:\n  unknown_key: 1")


@pytest.mark.unit
def test_inconsistent_settings_fail_before_work() -> None:
    """Acceptance 3: inconsistent settings fail fast during validation."""

    with pytest.raises(ConfigurationError):
        config_from_yaml("chunking:\n  chunk_size: 10\n  overlap: 20")
    with pytest.raises(ConfigurationError):
        config_from_yaml("reranking:\n  enabled: true\n  top_n: 99\nretrieval:\n  top_k: 4")


@pytest.mark.unit
def test_incompatible_context_limits_fail_early() -> None:
    """Acceptance 6: incompatible context limits fail during config validation."""

    with pytest.raises(ConfigurationError):
        config_from_yaml(
            "generation:\n  evidence_token_budget: 9000\n  max_new_tokens: 100\n"
            "  context_limit: 1000"
        )


@pytest.mark.unit
def test_shipped_configs_load() -> None:
    """Every shipped config file is valid."""

    files = glob.glob("configs/*.yaml")
    assert files
    for path in files:
        load_config(path)
