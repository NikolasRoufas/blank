"""End-to-end tests for the `egrag reason` demonstration trace."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from egrag.cli.main import app

runner = CliRunner()


@pytest.mark.e2e
def test_reason_trace_human_readable() -> None:
    result = runner.invoke(app, ["reason"])
    assert result.exit_code == 0
    out = result.stdout
    assert "initial scores:" in out
    assert "propagation: converged=True" in out
    assert "conflicts:" in out
    assert "selected subgraph:" in out
    # the support target rises and the contradicted claim falls
    assert "c1:" in out and "c3:" in out


@pytest.mark.e2e
def test_reason_trace_json() -> None:
    result = runner.invoke(app, ["reason", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert {"scores", "diagnostics", "conflicts", "selection"} <= set(payload)
    assert payload["converged"] is True
    # conflict between c1 and c3 is present and resolved to a preferred claim
    assert any(c["preferred_claim_id"] == "c1" for c in payload["conflicts"])


@pytest.mark.e2e
def test_reason_is_deterministic() -> None:
    first = runner.invoke(app, ["reason", "--json"]).stdout
    second = runner.invoke(app, ["reason", "--json"]).stdout
    assert first == second
