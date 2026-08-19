#!/usr/bin/env python
"""Phase 2: re-run ONLY full_egrag on the same dev-100 samples, with ONE new
classification-config field flipped (contradiction_requires_shared_subject=
True) and everything else identical to the Phase-1 run (retrieval, chunking,
extraction, NLI model/thresholds, propagation, temporal, selection strategy,
generator/model/decoding, evidence budget, seed).

This is the single-variable follow-up to Phase 1's finding that ~78-80% of
full_egrag's HotpotQA graph edges are CONTRADICTION edges between claims that
share an entity but not a subject/proposition (H-GRAPH). Output goes to a new
directory (qwen2.5-7b-instruct_gated) -- artifacts/dev100-bottleneck/{fever,
hotpotqa}/qwen2.5-7b-instruct/ (Phase 1's full_egrag/claim_only_rag reference)
is untouched.

Run:
    uv run python scripts/run_dev100_phase2_gated.py --benchmark hotpotqa
    uv run python scripts/run_dev100_phase2_gated.py --benchmark fever
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from egrag.experiments.models import ExperimentConfig
from egrag.experiments.runner import ExperimentRunner
from egrag.experiments.variants import RunComponents
from egrag.graph import ClassificationConfig, HuggingFaceNLIClassifier

NLI_MODEL = "roberta-large-mnli"
NLI_REVISION = "2a8f12d27941090092df78e4ba6f0928eb5eac98"
# The ONLY difference from Phase 1's CLASSIFICATION_CONFIG:
# contradiction_requires_shared_subject=True (all thresholds unchanged).
GATED_CLASSIFICATION_CONFIG = ClassificationConfig(
    entailment_threshold=0.4,
    contradiction_threshold=0.7,
    duplicate_threshold=0.8,
    contradiction_requires_shared_subject=True,
)

MODEL_KEY = "qwen2.5-7b-instruct_gated"
MODEL_SPEC = {"model": "Qwen/Qwen2.5-7B-Instruct", "disable_thinking": False}

DATASET_PATHS = {
    "fever": "artifacts/dev100-bottleneck/_raw_data/filtered/fever-dev-100.runner.jsonl",
    "hotpotqa": "artifacts/dev100-bottleneck/_raw_data/filtered/hotpot-dev-100.runner.jsonl",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["fever", "hotpotqa"], required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    dataset_path = DATASET_PATHS[args.benchmark]
    if not Path(dataset_path).is_file():
        raise SystemExit(f"{dataset_path} not found")

    print(f"Loading real NLI classifier ({NLI_MODEL}, revision {NLI_REVISION[:12]})...")
    nli = HuggingFaceNLIClassifier(NLI_MODEL, model_revision=NLI_REVISION, device="cuda")
    components = RunComponents(classifier=nli, classification_config=GATED_CLASSIFICATION_CONFIG)

    out_dir = Path("artifacts/dev100-bottleneck") / args.benchmark / MODEL_KEY
    config = ExperimentConfig(
        name=f"{args.benchmark}_{MODEL_KEY}_dev100",
        dataset=args.benchmark,
        dataset_path=dataset_path,
        variants=("full_egrag",),  # the only variant this experiment needs
        seeds=(0,),
        output_dir=str(out_dir),
        generator="huggingface",
        generator_model=MODEL_SPEC["model"],
        generator_device="cuda",
        generator_dtype="bfloat16",
        generator_disable_thinking=MODEL_SPEC["disable_thinking"],
        require_cuda=True,
        top_k=5,
        evidence_token_budget=256,
        reserved_output_tokens=min(16 if args.max_new_tokens <= 64 else 64, 256 - 1),
        max_new_tokens=args.max_new_tokens,
        chunk_size=256,
        chunk_overlap=0,
        limit=args.limit,
        enforce_fairness=True,
    )
    print(f"\n=== {args.benchmark} / {MODEL_KEY} (dev-100, gated contradiction) ===", flush=True)
    t0 = time.time()
    manifest = ExperimentRunner(config, components=components).run(resume=args.resume)
    elapsed = time.time() - t0
    print(
        f"done in {elapsed:.0f}s: examples={manifest.num_examples} "
        f"variants={len(manifest.variants)} warnings={len(manifest.warnings)}"
    )
    summary = {
        "benchmark": args.benchmark,
        "model": MODEL_KEY,
        "examples": manifest.num_examples,
        "seconds": round(elapsed, 1),
        "output_dir": str(out_dir),
    }
    summary_path = Path("artifacts/dev100-bottleneck") / f"{args.benchmark}_phase2_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
