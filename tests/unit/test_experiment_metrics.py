"""Metric and statistics acceptance tests (14, 16, 17, 18)."""

from __future__ import annotations

import pytest

from egrag.experiments.metrics import (
    METRIC_KINDS,
    MetricKind,
    citation_completeness,
    citation_precision,
    citation_recall,
    exact_match,
    metric_kind,
    normalize_answer,
    normalized_exact_match,
    token_f1,
)
from egrag.experiments.stats import aggregate_seeds, bootstrap_ci, mean, percentile, stdev


@pytest.mark.unit
def test_exact_match_normalization_behaves_as_documented() -> None:
    """Acceptance 16: EM normalization strips articles/punctuation/case/space."""

    assert normalize_answer("The  Cat.") == "cat"
    assert exact_match("The Cat.", ["The Cat."]) == 1.0
    assert exact_match("the cat", ["The Cat."]) == 0.0  # raw EM is strict
    assert normalized_exact_match("the cat", ["The Cat."]) == 1.0
    assert normalized_exact_match("a dog", ["the dog"]) == 1.0  # articles dropped


@pytest.mark.unit
def test_token_f1_handles_empty_cases() -> None:
    """Acceptance 17: token-F1 empty/empty -> 1.0; empty/non-empty -> 0.0."""

    assert token_f1("", [""]) == 1.0
    assert token_f1("", ["something"]) == 0.0
    assert token_f1("something", [""]) == 0.0
    assert token_f1("the quick brown fox", ["quick brown fox"]) > 0.8


@pytest.mark.unit
def test_citation_metrics() -> None:
    assert citation_precision(["a", "b"], ["a"]) == 0.5
    assert citation_recall(["a"], ["a", "b"]) == 0.5
    assert citation_completeness(["a", "b"], ["a", "b"]) == 1.0
    assert citation_completeness(["a"], ["a", "b"]) == 0.0


@pytest.mark.unit
def test_model_based_metrics_labeled_separately() -> None:
    """Acceptance 18: metric kinds are explicit and heuristic != deterministic."""

    assert metric_kind("exact_match") is MetricKind.DETERMINISTIC
    assert metric_kind("answer_evidence_entailment") is MetricKind.HEURISTIC
    assert METRIC_KINDS["token_f1"] is MetricKind.DETERMINISTIC
    deterministic = {k for k, v in METRIC_KINDS.items() if v is MetricKind.DETERMINISTIC}
    heuristic = {k for k, v in METRIC_KINDS.items() if v is MetricKind.HEURISTIC}
    assert deterministic.isdisjoint(heuristic)


@pytest.mark.unit
def test_bootstrap_is_deterministic_under_fixed_seed() -> None:
    """Acceptance 14: bootstrap intervals are deterministic under a fixed seed."""

    values = [0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8]
    first = bootstrap_ci(values, samples=500, seed=123)
    second = bootstrap_ci(values, samples=500, seed=123)
    assert first == second
    assert bootstrap_ci(values, samples=500, seed=999) != first
    assert first[0] <= mean(values) <= first[1]


@pytest.mark.unit
def test_descriptive_stats() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert mean(values) == 2.5
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 4.0
    assert stdev([2.0]) == 0.0


@pytest.mark.unit
def test_seed_aggregation_aligns_compatible_runs() -> None:
    """Acceptance 15: random-seed aggregation aligns compatible metric keys."""

    per_seed = [{"f1": 0.4, "em": 0.0}, {"f1": 0.6, "em": 1.0}]
    agg = aggregate_seeds(per_seed)
    assert agg["f1"]["mean"] == pytest.approx(0.5)
    assert agg["em"]["mean"] == pytest.approx(0.5)
    # incompatible keys (present in only one seed) are dropped from aggregation
    agg2 = aggregate_seeds([{"f1": 0.4}, {"f1": 0.6, "extra": 1.0}])
    assert set(agg2) == {"f1"}
