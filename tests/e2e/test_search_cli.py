"""End-to-end tests for the `egrag search` retrieval command."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from egrag.cli.main import app

runner = CliRunner()


@pytest.mark.e2e
def test_search_prints_ranked_passages() -> None:
    result = runner.invoke(app, ["search", "--query", "dense retrieval cosine"])
    assert result.exit_code == 0
    assert "score=" in result.stdout
    assert "candidates=" in result.stdout


@pytest.mark.e2e
def test_search_show_components() -> None:
    result = runner.invoke(app, ["search", "--query", "bm25 query terms", "--show-components"])
    assert result.exit_code == 0
    assert "components:" in result.stdout
    assert "bm25=" in result.stdout


@pytest.mark.e2e
def test_search_json_output_is_valid() -> None:
    result = runner.invoke(app, ["search", "--query", "evidence graph", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "results" in payload
    assert "stats" in payload
    assert payload["stats"]["num_results"] >= 0


@pytest.mark.e2e
def test_search_rejects_empty_query() -> None:
    result = runner.invoke(app, ["search", "--query", "   "])
    assert result.exit_code != 0
    assert "non-empty" in result.output


@pytest.mark.e2e
def test_search_rejects_negative_top_k() -> None:
    result = runner.invoke(app, ["search", "--query", "evidence", "--top-k", "-1"])
    assert result.exit_code != 0
    assert "invalid retrieval request" in result.output


@pytest.mark.e2e
def test_search_does_not_run_claim_extraction() -> None:
    """The search output is purely retrieval — no answer/citation fields."""

    result = runner.invoke(app, ["search", "--query", "evidence graph"])
    assert result.exit_code == 0
    assert "citations:" not in result.stdout
    assert "abstained:" not in result.stdout
