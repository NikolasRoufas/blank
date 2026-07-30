#!/usr/bin/env python
"""Convert the frozen FEVER/HotpotQA smoke-25 samples into the generic JSONL
format ``egrag.experiments.datasets.JsonlDataset`` (and hence ``ExperimentRunner``)
consumes, using the real ``egrag.experiments.benchmarks`` adapters for parsing
so gold answers/evidence are computed exactly as the repository already defines
them, then re-serialized without any transformation of their content.

Requires the raw dataset files already downloaded (see docs/reproduction.md)
and the corresponding sample manifest under
artifacts/benchmark-calibration/samples/.

Run:
    uv run python scripts/prepare_benchmark_samples.py
"""

from __future__ import annotations

import json
from pathlib import Path

from egrag.experiments.benchmarks import FeverGoldEvidenceDataset, HotpotQADataset
from egrag.experiments.models import DatasetExample

RAW = Path("artifacts/benchmark-matrix/_raw_data")
SAMPLES = Path("artifacts/benchmark-calibration/samples")
OUT = RAW / "filtered"


def _to_jsonl_row(ex: DatasetExample) -> dict:
    row: dict = {
        "id": ex.example_id,
        "question": ex.question,
        "documents": [
            {"source_id": d.source.source_id, "text": d.text, "title": d.source.title}
            for d in ex.documents
        ],
        "answers": list(ex.gold_answers),
        "gold_source_ids": list(ex.gold_evidence.source_ids),
        "split": ex.split,
    }
    if "label" in ex.metadata:
        row["label"] = ex.metadata["label"]
    return row


def convert(examples: list[DatasetExample], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(_to_jsonl_row(ex)) + "\n")
    print(f"wrote {len(examples)} examples -> {out_path}")


def main() -> None:
    fever = FeverGoldEvidenceDataset(path=OUT / "fever-smoke-25.jsonl")
    fever_examples = fever.load()
    assert len(fever_examples) == 25, f"expected 25 FEVER examples, got {len(fever_examples)}"
    convert(fever_examples, OUT / "fever-smoke-25.runner.jsonl")

    hotpot = HotpotQADataset(path=OUT / "hotpot-smoke-25.parquet")
    hotpot_examples = hotpot.load()
    assert len(hotpot_examples) == 25, f"expected 25 HotpotQA examples, got {len(hotpot_examples)}"
    convert(hotpot_examples, OUT / "hotpot-smoke-25.runner.jsonl")


if __name__ == "__main__":
    main()
