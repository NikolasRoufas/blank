#!/usr/bin/env python
"""Bottleneck-investigation run: dev-100 FEVER + HotpotQA, Qwen2.5-7B-Instruct
only, real NLI (roberta-large-mnli), testing H-GRAPH (relation-classification
miscalibration, esp. false-positive CONTRADICTION edges) as the primary
hypothesis for why full_egrag underperforms claim_only_rag on real data.

Adapted from scripts/run_benchmark_matrix.py (same frozen protocol: BM25
top_k=5, sentence-aware chunking 256/0, deterministic sentence-claim
extraction, real NLI thresholds 0.4/0.7/0.8 from the frozen configs, evidence
budget 256, deterministic decoding). Differences from that script, each a
deliberate, labeled deviation for this investigation:

  * dataset  -> the frozen dev-100 samples (not smoke-25)
  * model    -> Qwen2.5-7B-Instruct only (per the model rule: no scale change
                yet; the earlier best model on HotpotQA anyway)
  * variants -> passage_rag, claim_only_rag, graph_top_claim,
                graph_no_contradiction, full_egrag only (not all 10) --
                this is the minimal set that isolates the H-GRAPH test
                (full_egrag vs graph_no_contradiction, extraction- and
                selection-identical) plus the two necessary references
                (claim_only_rag: no-graph reference; graph_top_claim:
                selection-only variable, to separate a selection-strategy
                explanation from a relation-classification explanation;
                passage_rag: the nominal Simple RAG baseline, kept for
                completeness even though it is known to crash on most
                HotpotQA examples from a pre-existing, unrelated bug).
  * output   -> artifacts/dev100-bottleneck/ (a new directory; nothing in
                artifacts/benchmark-matrix/ is touched or overwritten).

Run:
    uv run python scripts/run_dev100_bottleneck.py --benchmark fever
    uv run python scripts/run_dev100_bottleneck.py --benchmark hotpotqa
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
CLASSIFICATION_CONFIG = ClassificationConfig(
    entailment_threshold=0.4, contradiction_threshold=0.7, duplicate_threshold=0.8
)

VARIANTS = "passage_rag,claim_only_rag,graph_top_claim,graph_no_contradiction,full_egrag"

MODEL_KEY = "qwen2.5-7b-instruct"
MODEL_SPEC = {"model": "Qwen/Qwen2.5-7B-Instruct", "disable_thinking": False}

DATASET_PATHS = {
    "fever": "artifacts/dev100-bottleneck/_raw_data/filtered/fever-dev-100.runner.jsonl",
    "hotpotqa": "artifacts/dev100-bottleneck/_raw_data/filtered/hotpot-dev-100.runner.jsonl",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["fever", "hotpotqa"], required=True)
    parser.add_argument("--limit", type=int, default=None, help="cap examples (debug only)")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                         help="256, not the frozen 64 -- the earlier smoke run found 64 caused "
                              "heavy truncation on real generators; using 256 throughout so this "
                              "run isn't confounded by the same truncation issue")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    dataset_path = DATASET_PATHS[args.benchmark]
    if not Path(dataset_path).is_file():
        raise SystemExit(f"{dataset_path} not found; run scripts/_prepare_dev100.py first")

    print(f"Loading real NLI classifier ({NLI_MODEL}, revision {NLI_REVISION[:12]})...")
    nli = HuggingFaceNLIClassifier(NLI_MODEL, model_revision=NLI_REVISION, device="cuda")
    components = RunComponents(classifier=nli, classification_config=CLASSIFICATION_CONFIG)

    out_dir = Path("artifacts/dev100-bottleneck") / args.benchmark / MODEL_KEY
    config = ExperimentConfig(
        name=f"{args.benchmark}_{MODEL_KEY}_dev100",
        dataset=args.benchmark,
        dataset_path=dataset_path,
        variants=tuple(VARIANTS.split(",")),
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
    print(f"\n=== {args.benchmark} / {MODEL_KEY} (dev-100) ===", flush=True)
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
    summary_path = Path("artifacts/dev100-bottleneck") / f"{args.benchmark}_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
