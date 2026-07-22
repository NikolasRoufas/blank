"""End-to-end CLI tests (acceptance cases 13, 14)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from egrag.cli.main import app

runner = CliRunner()


@pytest.mark.e2e
def test_run_succeeds_and_prints_answer() -> None:
    result = runner.invoke(app, ["run", "--query", "Is EG-RAG generator agnostic?"])
    assert result.exit_code == 0
    assert "evidence" in result.stdout.lower()
    assert "citations:" in result.stdout


@pytest.mark.e2e
def test_run_json_output_is_valid() -> None:
    import json

    result = runner.invoke(app, ["run", "--query", "hello", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["answer"]["text"]


@pytest.mark.e2e
def test_run_rejects_empty_query_with_actionable_message() -> None:
    """Acceptance 13 & 14: invalid input gives a message and a non-zero exit."""

    result = runner.invoke(app, ["run", "--query", "   "])
    assert result.exit_code != 0
    assert "non-empty" in result.output


@pytest.mark.e2e
def test_invalid_configuration_reports_actionable_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance 13: invalid configuration is reported with guidance."""

    monkeypatch.setenv("EGRAG_TOP_K", "0")
    result = runner.invoke(app, ["run", "--query", "hello"])
    assert result.exit_code != 0
    assert "configuration" in result.output.lower()


@pytest.mark.e2e
def test_inspect_config_outputs_json() -> None:
    result = runner.invoke(app, ["inspect-config"])
    assert result.exit_code == 0
    assert '"seed"' in result.stdout


@pytest.mark.e2e
def test_doctor_runs_and_reports_dependencies() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "core dependencies:" in result.stdout
    assert "optional dependencies:" in result.stdout
