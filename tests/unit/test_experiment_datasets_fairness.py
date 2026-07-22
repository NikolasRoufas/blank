"""Dataset-integrity and fairness/leakage acceptance tests (8,9,10,21,22,19,20)."""

from __future__ import annotations

import inspect

import pytest

from egrag.domain.models import Document, SourceMetadata
from egrag.experiments.datasets import SyntheticGraphDataset, check_dataset_integrity
from egrag.experiments.fairness import (
    align_by_example_id,
    assert_no_gold_parameters,
    check_fairness,
    same_example_ids,
)
from egrag.experiments.models import (
    DatasetExample,
    ExampleResult,
    ExperimentConfig,
    GoldEvidence,
    SystemVariant,
)
from egrag.experiments.variants import run_system


def _doc(sid: str, text: str) -> Document:
    return Document(document_id=sid, text=text, source=SourceMetadata(source_id=sid))


def _example(example_id: str, *, split: str = "test", gold: bool = True) -> DatasetExample:
    return DatasetExample(
        example_id=example_id,
        question="what happened in the report?",
        documents=(_doc("d1", "The report described strong results."),),
        gold_answers=("strong results",),
        gold_evidence=GoldEvidence(source_ids=("d1",)) if gold else GoldEvidence(available=False),
        split=split,
    )


@pytest.mark.unit
def test_duplicate_dataset_ids_detected() -> None:
    """Acceptance 8."""

    issues = check_dataset_integrity([_example("x"), _example("x")])
    assert any("duplicate dataset id" in i for i in issues)


@pytest.mark.unit
def test_duplicate_examples_detected() -> None:
    """Acceptance 9."""

    issues = check_dataset_integrity([_example("a"), _example("b")])
    assert any("duplicate example content" in i for i in issues)


@pytest.mark.unit
def test_missing_gold_evidence_reported() -> None:
    """Acceptance 10: missing gold evidence is reported, not silently scored."""

    issues = check_dataset_integrity([_example("a", gold=False)])
    assert any("missing gold evidence" in i for i in issues)


@pytest.mark.unit
def test_train_test_overlap_detected() -> None:
    issues = check_dataset_integrity([_example("a", split="train"), _example("b", split="test")])
    assert any("train/test overlap" in i for i in issues)


@pytest.mark.unit
def test_unfair_generator_mismatch_detected() -> None:
    """Acceptance 21."""

    config = ExperimentConfig(
        name="x", dataset="synthetic_graph", variants=("a", "b"), output_dir="out"
    )
    variants = [
        SystemVariant(name="a"),
        SystemVariant(name="b", generator_override="other-llm"),
    ]
    issues = check_fairness(config, variants)
    assert any("generator mismatch" in i for i in issues)


@pytest.mark.unit
def test_unfair_retrieval_budget_mismatch_detected() -> None:
    """Acceptance 22."""

    config = ExperimentConfig(
        name="x", dataset="synthetic_graph", variants=("a", "b"), output_dir="out"
    )
    variants = [SystemVariant(name="a"), SystemVariant(name="b", top_k_override=99)]
    assert any("retrieval-budget mismatch" in i for i in check_fairness(config, variants))


@pytest.mark.unit
def test_same_example_ids_and_alignment() -> None:
    a = [ExampleResult(example_id="1", variant="a", seed=0)]
    b = [ExampleResult(example_id="1", variant="b", seed=0)]
    assert same_example_ids({"a": a, "b": b}) == []
    paired, issues = align_by_example_id(a, b)
    assert len(paired) == 1 and issues == []
    mismatched = [ExampleResult(example_id="2", variant="b", seed=0)]
    assert same_example_ids({"a": a, "b": mismatched})


@pytest.mark.unit
def test_no_gold_parameters_in_system_callable() -> None:
    """Acceptance 19 & 20: gold labels are not parameters of system generation."""

    assert_no_gold_parameters(run_system)
    params = set(inspect.signature(run_system).parameters)
    assert "gold_answers" not in params and "gold_evidence" not in params


@pytest.mark.unit
def test_synthetic_dataset_loads() -> None:
    examples = SyntheticGraphDataset().load()
    assert len(examples) >= 2
    assert all(ex.documents for ex in examples)
