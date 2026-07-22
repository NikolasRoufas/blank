"""CLI acceptance tests (26, 27): documented commands run; failures are non-zero."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from egrag.cli.main import app

runner = CliRunner()

# Commands documented in README.md / CONTRIBUTING.md (offline, no downloads).
DOCUMENTED_COMMANDS = [
    ["doctor"],
    ["inspect-config"],
    ["run", "--query", "how does dense retrieval rank passages"],
    ["search", "--query", "dense retrieval"],
    ["extract", "--query", "evidence graph"],
    ["graph", "--query", "evidence graph"],
    ["reason"],
]


@pytest.mark.e2e
@pytest.mark.parametrize("command", DOCUMENTED_COMMANDS)
def test_documented_commands_execute(command: list[str]) -> None:
    """Acceptance 26: documented CLI commands run in a clean environment."""

    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output


@pytest.mark.e2e
def test_run_with_config_file() -> None:
    result = runner.invoke(
        app, ["run", "--query", "evidence graph", "--config", "configs/cpu_demo.yaml"]
    )
    assert result.exit_code == 0
    assert "citations:" in result.stdout


@pytest.mark.e2e
def test_cli_failures_return_nonzero() -> None:
    """Acceptance 27: CLI failures return a non-zero status."""

    assert runner.invoke(app, ["run", "--query", "   "]).exit_code != 0  # empty query
    assert runner.invoke(app, ["run", "--query", "x", "--config", "nope.yaml"]).exit_code != 0


@pytest.mark.e2e
def test_run_with_invalid_config_is_nonzero(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("retrieval:\n  bogus_field: 1\n", encoding="utf-8")
    result = runner.invoke(app, ["run", "--query", "x", "--config", str(bad)])
    assert result.exit_code != 0
    assert "configuration" in result.output.lower()
