"""End-to-end tests for the `egrag extract` command."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from egrag.cli.main import app

runner = CliRunner()


@pytest.mark.e2e
def test_extract_prints_claims_and_provenance() -> None:
    result = runner.invoke(app, ["extract", "--query", "dense retrieval cosine", "--top-k", "2"])
    assert result.exit_code == 0
    assert "source=" in result.stdout
    assert "passage=" in result.stdout
    assert "claims=" in result.stdout


@pytest.mark.e2e
def test_extract_json_output_is_valid() -> None:
    result = runner.invoke(app, ["extract", "--query", "evidence graph", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "claims" in payload
    for claim in payload["claims"]:
        assert claim["provenance"]["spans"]
        assert claim["belief"] is None  # extractor never sets belief


@pytest.mark.e2e
def test_extract_stops_before_graph_construction() -> None:
    """The extract output is purely claims — no graph/answer fields."""

    result = runner.invoke(app, ["extract", "--query", "evidence graph"])
    assert result.exit_code == 0
    assert "conflicts:" not in result.stdout
    assert "citations:" not in result.stdout


@pytest.mark.e2e
def test_extract_rejects_empty_query() -> None:
    result = runner.invoke(app, ["extract", "--query", "   "])
    assert result.exit_code != 0
    assert "non-empty" in result.output
