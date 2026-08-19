#!/usr/bin/env python
"""Filter the full FEVER/HotpotQA validation sets down to the pre-committed,
frozen dev-100 example-ID manifests, and convert to the generic runner JSONL
format -- mirroring scripts/prepare_benchmark_samples.py's smoke-25 approach,
but for the dev-100 samples used in the bottleneck-investigation run.

The example IDs come only from the frozen manifests
(artifacts/benchmark-calibration/samples/{fever,hotpot}-dev-100.json), selected
before any model ran; nothing here is chosen by model success.

Run:
    uv run python scripts/_prepare_dev100.py
"""

from __future__ import annotations

import json
from pathlib import Path

from egrag.experiments.benchmarks import FeverGoldEvidenceDataset, HotpotQADataset
from egrag.experiments.models import DatasetExample

SAMPLES = Path("artifacts/benchmark-calibration/samples")
OUT = Path("artifacts/dev100-bottleneck/_raw_data/filtered")


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
    fever_manifest = json.loads((SAMPLES / "fever-dev-100.json").read_text())
    fever_ids = set(fever_manifest["example_ids"])
    fever_all = FeverGoldEvidenceDataset(split="valid").load()
    fever_by_id = {e.example_id: e for e in fever_all}
    fever_examples = [fever_by_id[i] for i in fever_manifest["example_ids"] if i in fever_by_id]
    missing = fever_ids - {e.example_id for e in fever_examples}
    if missing:
        print(f"WARNING: {len(missing)} FEVER dev-100 ids not found in valid split: {missing}")
    assert len(fever_examples) == fever_manifest["size"], (
        f"expected {fever_manifest['size']} FEVER examples, got {len(fever_examples)}"
    )
    convert(fever_examples, OUT / "fever-dev-100.runner.jsonl")

    hotpot_manifest = json.loads((SAMPLES / "hotpot-dev-100.json").read_text())
    hotpot_ids = set(hotpot_manifest["example_ids"])
    hotpot_all = HotpotQADataset(split="validation").load()
    hotpot_by_id = {e.example_id: e for e in hotpot_all}
    hotpot_examples = [hotpot_by_id[i] for i in hotpot_manifest["example_ids"] if i in hotpot_by_id]
    missing_h = hotpot_ids - {e.example_id for e in hotpot_examples}
    if missing_h:
        print(f"WARNING: {len(missing_h)} HotpotQA dev-100 ids not found in validation split: {missing_h}")
    assert len(hotpot_examples) == hotpot_manifest["size"], (
        f"expected {hotpot_manifest['size']} HotpotQA examples, got {len(hotpot_examples)}"
    )
    convert(hotpot_examples, OUT / "hotpot-dev-100.runner.jsonl")


if __name__ == "__main__":
    main()
