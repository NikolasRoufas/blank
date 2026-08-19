#!/usr/bin/env python
"""Phase 3 addition: the missing `passage_rag` (Simple RAG) condition for the
two new Qwen scales, using the exact same frozen dev-100 pipeline as every
other Phase 1/2/3 run (BM25 top_k=5, sentence-aware chunking 256/0, evidence
budget 256, max_new_tokens=256, seed=0, deterministic decoding). passage_rag
never touches the extractor, NLI classifier, or graph -- it retrieves passages
and hands them straight to the generator -- so no NLI classifier is loaded
here (nothing would call it).

Output: artifacts/dev100-bottleneck/PHASE3/hotpotqa/{model_key}/A_passage_rag/
-- a new condition directory alongside the existing A_claim_only_rag/,
B_full_egrag_ungated/, C_full_egrag_gated/, D_graph_no_contradiction/ (all
untouched). Qwen2.5-7B-Instruct's passage_rag already exists from Phase 1
(artifacts/dev100-bottleneck/hotpotqa/qwen2.5-7b-instruct/, variant=passage_rag)
and is reused, not re-run.

Run:
    uv run python scripts/run_phase3_passage_rag.py --model-key qwen3.5-9b --benchmark hotpotqa
    uv run python scripts/run_phase3_passage_rag.py --model-key qwen2.5-3b-instruct --benchmark hotpotqa
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from egrag.experiments.models import ExperimentConfig
from egrag.experiments.runner import ExperimentRunner

MODELS: dict[str, dict] = {
    "qwen2.5-3b-instruct": {"model": "Qwen/Qwen2.5-3B-Instruct", "disable_thinking": False},
    "qwen3.5-9b": {"model": "Qwen/Qwen3.5-9B", "disable_thinking": True},
}

DATASET_PATHS = {
    "fever": "artifacts/dev100-bottleneck/_raw_data/filtered/fever-dev-100.runner.jsonl",
    "hotpotqa": "artifacts/dev100-bottleneck/_raw_data/filtered/hotpot-dev-100.runner.jsonl",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=list(MODELS), required=True)
    parser.add_argument("--benchmark", choices=["fever", "hotpotqa"], required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    dataset_path = DATASET_PATHS[args.benchmark]
    if not Path(dataset_path).is_file():
        raise SystemExit(f"{dataset_path} not found")
    spec = MODELS[args.model_key]

    out_dir = Path("artifacts/dev100-bottleneck/PHASE3") / args.benchmark / args.model_key / "A_passage_rag"
    config = ExperimentConfig(
        name=f"phase3_{args.benchmark}_{args.model_key}_A_passage_rag",
        dataset=args.benchmark,
        dataset_path=dataset_path,
        variants=("passage_rag",),
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
    print(f"\n=== {args.benchmark} / {args.model_key} / A_passage_rag ===", flush=True)
    t0 = time.time()
    manifest = ExperimentRunner(config).run(resume=args.resume)
    elapsed = time.time() - t0
    print(
        f"done in {elapsed:.0f}s: examples={manifest.num_examples} "
        f"variants={len(manifest.variants)} warnings={len(manifest.warnings)}"
    )


if __name__ == "__main__":
    main()
