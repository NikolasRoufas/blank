"""End-to-end tests for the `egrag graph` command and NetworkX isolation."""

from __future__ import annotations

import importlib
import json
import sys

import pytest
from typer.testing import CliRunner

from egrag.cli.main import app

runner = CliRunner()


@pytest.mark.e2e
def test_graph_summary_output() -> None:
    result = runner.invoke(app, ["graph", "--query", "evidence graph retrieval", "--top-k", "3"])
    assert result.exit_code == 0
    assert "nodes=" in result.stdout
    assert "candidate pruning:" in result.stdout
    assert "components=" in result.stdout


@pytest.mark.e2e
def test_graph_json_output_is_valid() -> None:
    result = runner.invoke(app, ["graph", "--query", "evidence graph", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "snapshot" in payload and "metrics" in payload
    assert "claims" in payload["snapshot"]
    assert payload["metrics"]["num_claims"] >= 0


@pytest.mark.e2e
def test_graph_rejects_empty_query() -> None:
    result = runner.invoke(app, ["graph", "--query", "   "])
    assert result.exit_code != 0
    assert "non-empty" in result.output


@pytest.mark.e2e
def test_graphml_without_extra_reports_actionable_error() -> None:
    """GraphML export needs the optional 'graph' extra (NetworkX)."""

    if importlib.util.find_spec("networkx") is not None:
        pytest.skip("networkx is installed; the missing-extra path is not exercised")
    result = runner.invoke(app, ["graph", "--query", "evidence graph", "--graphml"])
    assert result.exit_code != 0
    assert "graph" in result.output.lower()


@pytest.mark.unit
def test_graph_package_does_not_import_networkx() -> None:
    """The graph package must not pull in NetworkX (it stays behind the adapter)."""

    for name in [m for m in sys.modules if m == "egrag.graph" or m.startswith("egrag.graph.")]:
        del sys.modules[name]
    sys.modules.pop("networkx", None)
    importlib.import_module("egrag.graph")
    importlib.import_module("egrag.graph.construction")
    assert "networkx" not in sys.modules
