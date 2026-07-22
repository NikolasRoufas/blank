"""Benchmark-pipeline tests: FEVER adapter, HotpotQA blocker, metrics, integrity.

All offline and deterministic. FEVER tests read a tiny inline JSONL fixture (no
dependency on the machine's HF cache); the real cached dataset is exercised only
in a cache-gated test.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from egrag.domain.errors import InvalidInputError, MissingDependencyError
from egrag.experiments.benchmark_metrics import (
    evidence_set_recovered,
    exact_match,
    fever_evidence_prf,
    fever_label_accuracy,
    fever_score,
    normalize_answer,
    supporting_fact_prf,
    token_f1,
)
from egrag.experiments.benchmarks import (
    FeverGoldEvidenceDataset,
    HotpotQADataset,
    dataset_fingerprint,
    validate_benchmark,
)

_FEVER_LINES = [
    {
        "claim": "Paris is the capital of France.",
        "label": "SUPPORTS",
        "evidence": [["Paris", "0", "Paris is the capital and most populous city of France."]],
        "id": "ex1",
        "verifiable": "VERIFIABLE",
        "original_id": 1,
    },
    {
        "claim": "The Moon is made of cheese.",
        "label": "REFUTES",
        "evidence": [["Moon", "2", "The Moon is a rocky astronomical body."]],
        "id": "ex2",
        "verifiable": "VERIFIABLE",
        "original_id": 2,
    },
    {
        "claim": "An unknown person did something unverifiable.",
        "label": "NOT ENOUGH INFO",
        "evidence": [],
        "id": "ex3",
        "verifiable": "NOT VERIFIABLE",
        "original_id": 3,
    },
]


@pytest.fixture
def fever_file(tmp_path: Path) -> Path:
    p = tmp_path / "valid.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in _FEVER_LINES) + "\n", encoding="utf-8")
    return p


@pytest.mark.integration
def test_fever_parsing(fever_file: Path) -> None:
    """Required test 2."""

    exs = FeverGoldEvidenceDataset(split="valid", path=fever_file).load()
    assert [e.example_id for e in exs] == ["ex1", "ex2", "ex3"]
    assert exs[0].gold_evidence.stance == "supports"
    assert exs[1].gold_evidence.stance == "refutes"
    assert exs[2].gold_evidence.stance == "not_enough_info"
    assert exs[2].gold_evidence.available is False
    assert exs[0].gold_evidence.source_ids == ("Paris",)


@pytest.mark.integration
def test_evidence_id_normalization(fever_file: Path) -> None:
    """Required test 4: page id preserved exactly; title is human-readable."""

    ex = FeverGoldEvidenceDataset(path=fever_file).load()[0]
    assert ex.documents[0].source.source_id == "Paris"
    assert ex.documents[0].document_id.startswith("Paris#0#")


@pytest.mark.integration
def test_dataset_fingerprint_stability(fever_file: Path) -> None:
    """Required test 3."""

    a = FeverGoldEvidenceDataset(path=fever_file).load()
    b = FeverGoldEvidenceDataset(path=fever_file).load()
    assert dataset_fingerprint(a) == dataset_fingerprint(b)
    assert dataset_fingerprint(a) != dataset_fingerprint(a[:2])


@pytest.mark.integration
def test_gold_never_enters_pipeline_inputs(fever_file: Path) -> None:
    """Required test 5: pipeline-visible fields exclude gold label/evidence stance.

    The pipeline consumes ``question`` and ``documents`` only; gold answer and the
    gold stance live on separate fields and never appear in document text.
    """

    ex = FeverGoldEvidenceDataset(path=fever_file).load()[0]
    pipeline_text = ex.question + " " + " ".join(d.text for d in ex.documents)
    assert "SUPPORTS" not in pipeline_text
    assert ex.gold_answers == ("SUPPORTS",)  # gold kept aside


@pytest.mark.integration
def test_malformed_dataset_rejected(tmp_path: Path) -> None:
    """Required test 6."""

    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"claim": "x", "label": "BOGUS", "evidence": [], "id": "b"}) + "\n")
    with pytest.raises(InvalidInputError):
        FeverGoldEvidenceDataset(path=p).load()


@pytest.mark.integration
def test_dataset_validation_flags_issues(fever_file: Path) -> None:
    exs = FeverGoldEvidenceDataset(path=fever_file).load()
    assert validate_benchmark(exs) == []
    # duplicate id -> flagged
    assert any("duplicate" in m for m in validate_benchmark(exs + exs[:1]))


@pytest.mark.integration
def test_hotpotqa_blocked_cleanly() -> None:
    """HotpotQA is cached only as parquet; without the ``benchmarks`` extra
    (pyarrow) loading must raise a typed error. When the extra is installed the
    load succeeds instead (covered by the gated test in
    ``test_benchmark_calibration``), so this assertion is skipped."""

    import importlib.util

    if importlib.util.find_spec("pyarrow") is not None:
        pytest.skip("pyarrow installed; blocking no longer applies (see gated load test)")
    with pytest.raises(MissingDependencyError):
        HotpotQADataset(limit=1).load()


@pytest.mark.integration
def test_normalize_and_answer_metrics() -> None:
    """Required test 13: HotpotQA metric edge cases."""

    assert normalize_answer("The  United States.") == "united states"
    assert exact_match("the United States", ("United States",)) == 1.0
    assert exact_match("France", ("United States",)) == 0.0
    assert token_f1("New York City", ("New York",)) == pytest.approx(
        2 * 1.0 * (2 / 3) / (1 + 2 / 3)
    )
    # yes/no exact handling
    assert token_f1("yes", ("yes",)) == 1.0
    assert token_f1("yes", ("no",)) == 0.0


@pytest.mark.integration
def test_supporting_fact_prf() -> None:
    gold = {("A", 0), ("B", 1)}
    assert supporting_fact_prf({("A", 0), ("B", 1)}, gold)[2] == pytest.approx(1.0)
    p, r, _ = supporting_fact_prf({("A", 0)}, gold)
    assert p == 1.0 and r == 0.5


@pytest.mark.integration
def test_fever_metrics_and_alternative_evidence_sets() -> None:
    """Required tests 14 & 15."""

    assert fever_label_accuracy("supports", "SUPPORTS") == 1.0
    assert fever_label_accuracy("REFUTES", "SUPPORTS") == 0.0
    _, _, f1 = fever_evidence_prf({"P1"}, {"P1"})
    assert f1 == 1.0
    # alternative valid evidence sets: covering ANY complete set suffices
    gold_sets = (frozenset({"P1", "P2"}), frozenset({"P3"}))
    assert evidence_set_recovered({"P3"}, gold_sets) is True
    assert evidence_set_recovered({"P1"}, gold_sets) is False
    # NEI: no evidence required
    assert evidence_set_recovered(set(), ()) is True
    # official FEVER score: label + complete evidence for S/R; label only for NEI
    assert fever_score("SUPPORTS", "SUPPORTS", {"P3"}, gold_sets) == 1.0
    assert fever_score("SUPPORTS", "SUPPORTS", {"P1"}, gold_sets) == 0.0
    assert fever_score("NOT ENOUGH INFO", "NOT ENOUGH INFO", set(), ()) == 1.0
    assert fever_score("REFUTES", "SUPPORTS", {"P3"}, gold_sets) == 0.0


@pytest.mark.integration
def test_pilot_sample_reproducible(fever_file: Path) -> None:
    """Required test 20: a fixed-seed stratified sample is reproducible."""

    import random

    exs = FeverGoldEvidenceDataset(path=fever_file).load()
    s1 = random.Random(7).sample(exs, k=2)
    s2 = random.Random(7).sample(exs, k=2)
    assert [e.example_id for e in s1] == [e.example_id for e in s2]


@pytest.mark.integration
def test_no_network_in_unit_tests(fever_file: Path) -> None:
    """Required test 23: loading/validating performs no network I/O.

    The session-wide socket block (conftest) raises on any connection, so this
    succeeding confirms the adapter path is fully offline.
    """

    exs = FeverGoldEvidenceDataset(path=fever_file).load()
    assert validate_benchmark(exs) == []


@pytest.mark.requires_local_models
@pytest.mark.skipif(
    os.environ.get("EGRAG_RUN_LOCAL_MODELS") != "1",
    reason="cached-dataset test; set EGRAG_RUN_LOCAL_MODELS=1",
)
def test_real_cached_fever_loads() -> None:
    """Cache-gated: the real cached FEVER validation split loads and validates."""

    exs = FeverGoldEvidenceDataset(split="valid", limit=50).load()
    assert len(exs) == 50
    assert validate_benchmark(exs) == []
    stances = {e.gold_evidence.stance for e in exs}
    assert stances <= {"supports", "refutes", "not_enough_info"}
