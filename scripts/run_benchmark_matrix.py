#!/usr/bin/env python
"""Run the real Qwen generator matrix (3B / 7B / Qwen3.5-9B) over a real
benchmark sample (FEVER or HotpotQA smoke-25), reproducing the pre-committed
dev-frozen protocol in
``artifacts/benchmark-calibration/frozen-configs/{fever,hotpotqa}.yaml``
exactly: BM25 top_k=5, sentence-aware chunking (256/0), deterministic
sentence-claim extraction (a real extractor failed structured-output
validation in the prior calibration work), real NLI (roberta-large-mnli,
pinned revision, thresholds 0.4/0.7/0.8), evidence budget 256 (16 reserved),
deterministic decoding, max_new_tokens=64, seed=[0]. Only the generator model
varies across the three runs. All settings above come directly from the frozen
config file, not from this script.

Requires scripts/prepare_benchmark_samples.py to have been run first.

Run:
    uv run python scripts/run_benchmark_matrix.py --benchmark fever
    uv run python scripts/run_benchmark_matrix.py --benchmark hotpotqa
    uv run python scripts/run_benchmark_matrix.py --benchmark fever \\
        --models qwen2.5-3b-instruct --limit 3
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
# From the frozen configs (fever.yaml / hotpotqa.yaml): entailment 0.4,
# contradiction 0.7, duplicate 0.8 -- NOT the repo's ClassificationConfig()
# defaults (0.5/0.5/0.8); reproducing the calibrated protocol exactly.
CLASSIFICATION_CONFIG = ClassificationConfig(
    entailment_threshold=0.4, contradiction_threshold=0.7, duplicate_threshold=0.8
)

VARIANTS = (
    "passage_rag,reranked_passage_rag,claim_only_rag,graph_no_propagation,"
    "graph_top_claim,graph_coherent_subgraph,graph_no_temporal,"
    "graph_no_contradiction,graph_with_propagation,full_egrag"
)

MODELS: dict[str, dict] = {
    "qwen2.5-3b-instruct": {"model": "Qwen/Qwen2.5-3B-Instruct", "disable_thinking": False},
    "qwen2.5-7b-instruct": {"model": "Qwen/Qwen2.5-7B-Instruct", "disable_thinking": False},
    "qwen3.5-9b": {"model": "Qwen/Qwen3.5-9B", "disable_thinking": True},
}

DATASET_PATHS = {
    "fever": "artifacts/benchmark-matrix/_raw_data/filtered/fever-smoke-25.runner.jsonl",
    "hotpotqa": "artifacts/benchmark-matrix/_raw_data/filtered/hotpot-smoke-25.runner.jsonl",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["fever", "hotpotqa"], required=True)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--limit", type=int, default=None, help="cap examples (debug only)")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="64 matches the frozen config exactly; a larger value is a labeled deviation",
    )
    parser.add_argument("--tag", default="", help="output-dir suffix, e.g. '-mnt256'")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    dataset_path = DATASET_PATHS[args.benchmark]
    if not Path(dataset_path).is_file():
        raise SystemExit(
            f"{dataset_path} not found; run scripts/prepare_benchmark_samples.py first"
        )

    print(f"Loading real NLI classifier ({NLI_MODEL}, revision {NLI_REVISION[:12]})...")
    nli = HuggingFaceNLIClassifier(NLI_MODEL, model_revision=NLI_REVISION, device="cuda")
    components = RunComponents(classifier=nli, classification_config=CLASSIFICATION_CONFIG)

    model_keys = args.models.split(",")
    summary = []
    for model_key in model_keys:
        spec = MODELS[model_key]
        out_dir = Path("artifacts/benchmark-matrix") / args.benchmark / f"{model_key}{args.tag}"
        config = ExperimentConfig(
            name=f"{args.benchmark}_{model_key}",
            dataset=args.benchmark,
            dataset_path=dataset_path,
            variants=tuple(VARIANTS.split(",")),
            seeds=(0,),
            output_dir=str(out_dir),
            generator="huggingface",
            generator_model=spec["model"],
            generator_device="cuda",
            generator_dtype="bfloat16",
            generator_disable_thinking=spec["disable_thinking"],
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
        print(f"\n=== {args.benchmark} / {model_key} ===", flush=True)
        t0 = time.time()
        manifest = ExperimentRunner(config, components=components).run(resume=args.resume)
        elapsed = time.time() - t0
        print(
            f"done in {elapsed:.0f}s: examples={manifest.num_examples} "
            f"variants={len(manifest.variants)} warnings={len(manifest.warnings)}"
        )
        summary.append(
            {
                "benchmark": args.benchmark,
                "model": model_key,
                "examples": manifest.num_examples,
                "seconds": round(elapsed, 1),
                "output_dir": str(out_dir),
            }
        )

    summary_path = (
        Path("artifacts/benchmark-matrix") / f"{args.benchmark}_run_summary{args.tag}.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
