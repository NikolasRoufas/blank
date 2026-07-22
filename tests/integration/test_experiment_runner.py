"""Experiment-runner acceptance tests (1-7, 11-13, 23-30)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from egrag.experiments import (
    ExperimentConfig,
    ExperimentRunner,
    SyntheticGraphDataset,
    compare,
    get_variant,
    inspect_example,
    summarize,
)
from egrag.experiments.artifacts import RESULTS_FILE, read_results
from egrag.experiments.models import ExampleResult
from egrag.experiments.runner import _aggregate

VARIANTS = ("passage_rag", "claim_only_rag", "full_egrag")
_OPTIONAL_LIBS = {"torch", "transformers", "httpx", "sentence_transformers", "networkx"}


def _config(output_dir: Path, **kw: object) -> ExperimentConfig:
    base: dict[str, object] = {
        "name": "t",
        "dataset": "synthetic_graph",
        "variants": VARIANTS,
        "seeds": (0,),
        "output_dir": str(output_dir),
        "top_k": 3,
        "evidence_token_budget": 256,
        "reserved_output_tokens": 64,
    }
    base.update(kw)
    return ExperimentConfig(**base)  # type: ignore[arg-type]


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    out = tmp_path / "run"
    ExperimentRunner(_config(out)).run()
    return out


@pytest.mark.integration
def test_compared_systems_use_identical_example_ids(run_dir: Path) -> None:
    """Acceptance 1."""

    results, _ = read_results(run_dir / RESULTS_FILE)
    by_variant: dict[str, set[str]] = {}
    for r in results:
        by_variant.setdefault(r.variant, set()).add(r.example_id)
    id_sets = list(by_variant.values())
    assert all(s == id_sets[0] for s in id_sets)


@pytest.mark.integration
def test_paired_comparison_aligns_by_example_id(run_dir: Path) -> None:
    """Acceptance 2."""

    result = compare(run_dir, "full_egrag", "passage_rag", "citation_recall", samples=200)
    assert result.n_paired == len(SyntheticGraphDataset().load())
    assert result.ci_low <= result.mean_delta <= result.ci_high


@pytest.mark.integration
def test_fairness_equal_budgets_and_generator(run_dir: Path) -> None:
    """Acceptance 3, 4, 5: shared retrieval/evidence budgets and generator."""

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["top_k"] == 3
    assert manifest["evidence_token_budget"] == 256
    assert manifest["generator"] == "fake"
    assert manifest["fairness_enforced"] is True
    # variants carry no per-variant overrides
    for variant in manifest["variants"]:
        assert variant["generator_override"] is None
        assert variant["top_k_override"] is None
        assert variant["evidence_budget_override"] is None


@pytest.mark.integration
def test_resume_does_not_duplicate_and_preserves(run_dir: Path, tmp_path: Path) -> None:
    """Acceptance 6, 7, 24: resume adds nothing and preserves completed results."""

    before, _ = read_results(run_dir / RESULTS_FILE)
    ExperimentRunner(_config(run_dir)).run(resume=True)
    after, _ = read_results(run_dir / RESULTS_FILE)
    assert len(after) == len(before)
    before_keys = {(r.variant, r.seed, r.example_id): r.answer for r in before}
    after_keys = {(r.variant, r.seed, r.example_id): r.answer for r in after}
    assert before_keys == after_keys


@pytest.mark.integration
def test_corrupted_partial_results_handled_safely(run_dir: Path) -> None:
    """Acceptance 25: a corrupted trailing line is skipped and recomputed."""

    results_path = run_dir / RESULTS_FILE
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write('{"example_id": "syn-1", "variant": "full_egrag"  CORRUPT\n')
    parsed, skipped = read_results(results_path)
    assert skipped == 1
    # resume tolerates the corruption and does not crash
    ExperimentRunner(_config(run_dir)).run(resume=True)
    reparsed, _ = read_results(results_path)
    assert len(reparsed) >= len(parsed)


@pytest.mark.integration
def test_failed_examples_visible_and_counted() -> None:
    """Acceptance 13 & 26: failures stay visible and preserve failure counts."""

    results = [
        ExampleResult(example_id="a", variant="v", seed=0, metrics={"token_f1": 0.5}),
        ExampleResult(example_id="b", variant="v", seed=0, failed=True, error="boom"),
    ]
    agg = _aggregate(results, "v", 0)
    assert agg.num_examples == 2
    assert agg.num_failed == 1
    assert agg.metrics["token_f1"] == 0.5  # mean over successful only, failure not dropped


@pytest.mark.integration
def test_empty_and_invalid_citation_counts() -> None:
    """Acceptance 11 & 12."""

    from egrag.experiments.metrics import invalid_citation_count, is_empty_prediction

    assert is_empty_prediction("   ") is True
    assert is_empty_prediction("answer") is False
    assert invalid_citation_count(["c1", "c9"], ["c1"]) == 1


@pytest.mark.integration
def test_per_example_artifacts_round_trip(run_dir: Path) -> None:
    """Acceptance 23."""

    results, _ = read_results(run_dir / RESULTS_FILE)
    assert results
    for r in results:
        restored = ExampleResult.model_validate_json(r.model_dump_json())
        assert restored == r


@pytest.mark.integration
def test_variant_config_recorded_in_manifest(run_dir: Path) -> None:
    """Acceptance 27."""

    manifest = json.loads((run_dir / "manifest.json").read_text())
    names = {v["name"] for v in manifest["variants"]}
    assert names == set(VARIANTS)
    full = next(v for v in manifest["variants"] if v["name"] == "full_egrag")
    assert full["propagation"] is True and full["family"] == "graph"


@pytest.mark.integration
def test_system_variants_isolate_intended_component() -> None:
    """Acceptance 28: variants differ only in their intended field(s)."""

    full = get_variant("full_egrag")
    no_prop = get_variant("graph_no_propagation")
    differing = {f for f in full.model_fields if getattr(full, f) != getattr(no_prop, f)}
    assert differing <= {"name", "description", "propagation", "isolates"}

    passage = get_variant("passage_rag")
    reranked = get_variant("reranked_passage_rag")
    differing2 = {f for f in passage.model_fields if getattr(passage, f) != getattr(reranked, f)}
    assert differing2 <= {"name", "description", "rerank", "isolates"}


@pytest.mark.integration
def test_artifacts_written(run_dir: Path) -> None:
    expected = {
        "manifest.json",
        "resolved_config.json",
        "aggregate.json",
        "results.jsonl",
        "failures.log",
        "timing.json",
        "metric_warnings.json",
        "reproducibility.json",
        "packages",
        "graphs",
    }
    assert expected <= {p.name for p in run_dir.iterdir()}
    summary = summarize(run_dir)
    assert summary
    assert inspect_example(run_dir, "syn-1")


@pytest.mark.integration
def test_relative_output_dir_writes_to_that_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a relative output_dir must not be double-joined onto its base.

    Previously ``safe_artifact_path`` re-resolved the raw relative path against
    the parent base, writing results to ``out/out/run`` instead of ``out/run``.
    """

    monkeypatch.chdir(tmp_path)
    cfg = ExperimentConfig(
        name="rel",
        dataset="synthetic_graph",
        variants=("passage_rag",),
        seeds=(42,),
        output_dir="out/run",
        top_k=3,
        evidence_token_budget=256,
        reserved_output_tokens=64,
    )
    ExperimentRunner(cfg).run()
    assert (tmp_path / "out" / "run" / RESULTS_FILE).is_file()
    assert not (tmp_path / "out" / "out").exists()  # no doubly-nested directory


@pytest.mark.integration
def test_uses_only_synthetic_and_no_optional_libs(run_dir: Path) -> None:
    """Acceptance 29 & 30: synthetic fixtures only; no model libs imported."""

    config = json.loads((run_dir / "resolved_config.json").read_text())
    assert config["dataset"] == "synthetic_graph"
    assert _OPTIONAL_LIBS.isdisjoint(set(sys.modules))
