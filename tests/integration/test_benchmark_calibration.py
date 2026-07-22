"""Benchmark-calibration tests: benchmarks extra isolation, HotpotQA mapping,
FEVER empty-evidence handling, sample reproducibility, frozen-config checksums.

All offline and deterministic. The HotpotQA ``_parse`` mapping is exercised with
synthetic rows (no parquet reader needed); the real parquet load is gated behind
the ``benchmarks`` extra and skips cleanly when ``pyarrow`` is absent.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from egrag.domain.errors import MissingDependencyError
from egrag.experiments.benchmarks import (
    FeverGoldEvidenceDataset,
    HotpotQADataset,
    dataset_fingerprint,
)

# --- optional-dependency isolation (required tests 1, 3) --------------------


@pytest.mark.integration
def test_core_import_does_not_import_pyarrow() -> None:
    """Required test 1: importing the benchmark module must not import pyarrow.

    pyarrow is imported lazily inside ``HotpotQADataset.load`` only.
    """

    importlib.import_module("egrag.experiments.benchmarks")
    assert "pyarrow" not in sys.modules


@pytest.mark.integration
def test_hotpotqa_missing_parquet_error_names_benchmarks_extra() -> None:
    """Required test 3: typed, actionable error naming the ``benchmarks`` extra."""

    if importlib.util.find_spec("pyarrow") is not None:
        pytest.skip("pyarrow installed; covered by the gated load test")
    with pytest.raises(MissingDependencyError) as ei:
        HotpotQADataset(limit=1).load()
    assert ei.value.extra == "benchmarks"
    assert "egrag[benchmarks]" in str(ei.value)


# --- HotpotQA _parse mapping (required tests 4: synthetic rows) -------------


def _hotpot_row(**over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": "hp1",
        "question": "Are Scott Derrickson and Ed Wood of the same nationality?",
        "answer": "yes",
        "type": "comparison",
        "level": "hard",
        "context": {
            "title": ["Scott Derrickson", "Ed Wood"],
            "sentences": [
                ["Scott Derrickson is an American director.", "He was born in 1966."],
                ["Edward Davis Wood Jr. was an American filmmaker.", ""],
            ],
        },
        "supporting_facts": {
            "title": ["Scott Derrickson", "Ed Wood"],
            "sent_id": [0, 0],
        },
    }
    row.update(over)
    return row


@pytest.mark.integration
def test_hotpotqa_parse_supporting_facts_and_documents() -> None:
    """Required test 4: titles, sentence identity, and gold pages preserved."""

    ex = HotpotQADataset._parse(_hotpot_row())
    assert ex.example_id == "hp1"
    # one Document per non-empty sentence; empty trailing sentence skipped
    ids = [d.document_id for d in ex.documents]
    assert "Scott Derrickson#0" in ids
    assert "Ed Wood#0" in ids
    assert all(d.text.strip() for d in ex.documents)  # no empty docs
    # gold evidence pages preserved (deduped, order-stable)
    assert ex.gold_evidence.source_ids == ("Scott Derrickson", "Ed Wood")
    sp = json.loads(ex.metadata["supporting_facts"])
    assert sp == [["Scott Derrickson", 0], ["Ed Wood", 0]]
    assert ex.metadata["type"] == "comparison"


@pytest.mark.integration
def test_hotpotqa_parse_duplicate_titles() -> None:
    """Required test 4: duplicate supporting-fact titles collapse to one gold page."""

    row = _hotpot_row(supporting_facts={"title": ["Ed Wood", "Ed Wood"], "sent_id": [0, 1]})
    ex = HotpotQADataset._parse(row)
    assert ex.gold_evidence.source_ids == ("Ed Wood",)


@pytest.mark.integration
def test_hotpotqa_parse_yes_no_answer() -> None:
    """Required test 4: yes/no answers preserved verbatim as the gold answer."""

    ex = HotpotQADataset._parse(_hotpot_row(answer="no"))
    assert ex.gold_answers == ("no",)


@pytest.mark.integration
def test_hotpotqa_parse_malformed_row_raises() -> None:
    """Required test 4: a malformed row (missing context) raises, not silently parses."""

    bad = _hotpot_row()
    del bad["context"]
    with pytest.raises((KeyError, TypeError)):
        HotpotQADataset._parse(bad)


@pytest.mark.requires_benchmarks
@pytest.mark.skipif(
    importlib.util.find_spec("pyarrow") is None,
    reason="needs the 'benchmarks' extra (pyarrow) installed",
)
def test_hotpotqa_parquet_loads_when_extra_present() -> None:
    """Required test 2: real cached parquet loads when pyarrow is available."""

    exs = HotpotQADataset(limit=3).load()
    assert len(exs) == 3
    assert all(e.example_id for e in exs)
    assert all(e.gold_answers for e in exs)


# --- FEVER empty-evidence regression (this milestone's bug fix) -------------


@pytest.mark.integration
def test_fever_skips_empty_evidence_sentences(tmp_path: Path) -> None:
    """Regression: an evidence row with an empty sentence is skipped, not fatal.

    Real FEVER (copenlu/fever_gold_evidence valid) ships 62 such spans; the
    adapter must drop the empty span and record the count rather than raising.
    """

    line = {
        "claim": "Some claim.",
        "label": "SUPPORTS",
        "evidence": [
            ["Page_A", "0", ""],  # empty sentence -> skipped
            ["Page_A", "1", "A non-empty supporting sentence."],
        ],
        "id": "evx",
        "verifiable": "VERIFIABLE",
        "original_id": 9,
    }
    p = tmp_path / "valid.jsonl"
    p.write_text(json.dumps(line) + "\n", encoding="utf-8")
    ex = FeverGoldEvidenceDataset(path=p).load()[0]
    assert len(ex.documents) == 1
    assert ex.documents[0].text == "A non-empty supporting sentence."
    assert ex.metadata["num_evidence_skipped_empty"] == "1"
    assert ex.gold_evidence.source_ids == ("Page_A",)


# --- sample-manifest reproducibility (required test 6/20) -------------------

_SAMPLES = Path("artifacts/benchmark-calibration/samples")


@pytest.mark.integration
@pytest.mark.skipif(
    not (_SAMPLES / "fever-dev-100.json").is_file(),
    reason="calibration sample manifest not present",
)
def test_fever_sample_manifest_is_balanced_and_stable() -> None:
    """Required test 6: the frozen dev sample is balanced and uniquely identified."""

    data = json.loads((_SAMPLES / "fever-dev-100.json").read_text(encoding="utf-8"))
    ids = data["example_ids"]
    assert len(ids) == len(set(ids)) == data["size"] == 100
    dist = data["label_distribution"]
    assert min(dist.values()) >= 33  # balanced across the three labels
    assert data["dataset_fingerprint_sha256"]


# --- frozen-config checksum stability (required test 17) --------------------


def _config_checksum(mapping: dict[str, Any]) -> str:
    """Canonical, order-independent checksum used for frozen configs."""

    canonical = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@pytest.mark.integration
def test_frozen_config_checksum_stable_and_sensitive() -> None:
    """Required test 17: checksum is stable under key order, sensitive to values."""

    a = {"top_k": 5, "nli": {"entail": 0.4, "contra": 0.7}}
    b = {"nli": {"contra": 0.7, "entail": 0.4}, "top_k": 5}  # reordered
    c = {"top_k": 8, "nli": {"entail": 0.4, "contra": 0.7}}  # changed value
    assert _config_checksum(a) == _config_checksum(b)
    assert _config_checksum(a) != _config_checksum(c)


@pytest.mark.integration
def test_dataset_fingerprint_distinguishes_samples(tmp_path: Path) -> None:
    """Fingerprint changes when the example set changes."""

    def write(ids: list[str]) -> Path:
        p = tmp_path / f"{'_'.join(ids)}.jsonl"
        p.write_text(
            "\n".join(
                json.dumps({"claim": f"c{i}", "label": "SUPPORTS", "evidence": [], "id": i})
                for i in ids
            )
            + "\n",
            encoding="utf-8",
        )
        return p

    a = FeverGoldEvidenceDataset(path=write(["a", "b"])).load()
    b = FeverGoldEvidenceDataset(path=write(["a", "c"])).load()
    assert dataset_fingerprint(a) != dataset_fingerprint(b)
