#!/usr/bin/env python
"""Phase 3: cross-model generality of the Phase 2 contradiction-gating fix.

Reuses exactly the Phase 1/2 pipeline (ExperimentConfig/ExperimentRunner/
RunComponents, same frozen protocol: BM25 top_k=5, sentence-aware chunking
256/0, deterministic sentence-claim extraction, real NLI roberta-large-mnli
thresholds 0.4/0.7/0.8, evidence budget 256, max_new_tokens=256, seed=0,
deterministic decoding) on the SAME dev-100 filtered sample Phase 1/2 used
(artifacts/dev100-bottleneck/_raw_data/filtered/*-dev-100.runner.jsonl --
not regenerated). Loads the model + NLI classifier ONCE and runs all four
Phase 3 conditions back-to-back (more efficient than Phase 1/2's split-script
pattern, which reloaded the model per script invocation):

  A: claim_only_rag                                  (no-graph reference)
  B: full_egrag, contradiction_requires_shared_subject=False  (current/ungated)
  C: full_egrag, contradiction_requires_shared_subject=True   (Phase 2 gate)
  D: graph_no_contradiction                            (Phase 1 ablation)

Only the models tested are new (Qwen2.5-3B-Instruct, Qwen3.5-9B -- the other
two points in the repo's established A/B/C Qwen matrix, see
scripts/run_benchmark_matrix.py's MODELS dict / docs/reproduction.md). Every
other condition is frozen and identical to Phase 1/2.

Output: artifacts/dev100-bottleneck/PHASE3/{benchmark}/{model_key}/{condition}/
-- a NEW directory tree; nothing under artifacts/dev100-bottleneck/{fever,
hotpotqa}/qwen2.5-7b-instruct{,_gated}/ (Phase 1/2) is touched.

Run:
    uv run python scripts/run_phase3.py --model-key qwen2.5-3b-instruct --benchmark hotpotqa
    uv run python scripts/run_phase3.py --model-key qwen3.5-9b --benchmark hotpotqa
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
BASE_THRESHOLDS = {"entailment_threshold": 0.4, "contradiction_threshold": 0.7, "duplicate_threshold": 0.8}

# Same MODELS dict as scripts/run_benchmark_matrix.py -- the repo's established
# A/B/C Qwen scale matrix. 7B is the already-completed Phase 1/2 reference and
# is not re-run here.
MODELS: dict[str, dict] = {
    "qwen2.5-3b-instruct": {"model": "Qwen/Qwen2.5-3B-Instruct", "disable_thinking": False},
    "qwen3.5-9b": {"model": "Qwen/Qwen3.5-9B", "disable_thinking": True},
}

# (condition_dir_name, variant_name, contradiction_requires_shared_subject)
CONDITIONS = [
    ("A_claim_only_rag", "claim_only_rag", False),
    ("B_full_egrag_ungated", "full_egrag", False),
    ("C_full_egrag_gated", "full_egrag", True),
    ("D_graph_no_contradiction", "graph_no_contradiction", False),
]

DATASET_PATHS = {
    "fever": "artifacts/dev100-bottleneck/_raw_data/filtered/fever-dev-100.runner.jsonl",
    "hotpotqa": "artifacts/dev100-bottleneck/_raw_data/filtered/hotpot-dev-100.runner.jsonl",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=list(MODELS), required=True)
    parser.add_argument("--benchmark", choices=["fever", "hotpotqa"], required=True)
    parser.add_argument("--limit", type=int, default=None, help="cap examples (debug only)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    dataset_path = DATASET_PATHS[args.benchmark]
    if not Path(dataset_path).is_file():
        raise SystemExit(f"{dataset_path} not found -- Phase 1's scripts/_prepare_dev100.py must have run")

    spec = MODELS[args.model_key]

    print(f"Loading real NLI classifier ({NLI_MODEL}, revision {NLI_REVISION[:12]})...")
    nli = HuggingFaceNLIClassifier(NLI_MODEL, model_revision=NLI_REVISION, device="cuda")

    summary = []
    for condition_dir, variant_name, gate_on in CONDITIONS:
        classification_config = ClassificationConfig(
            **BASE_THRESHOLDS, contradiction_requires_shared_subject=gate_on
        )
        components = RunComponents(classifier=nli, classification_config=classification_config)

        out_dir = Path("artifacts/dev100-bottleneck/PHASE3") / args.benchmark / args.model_key / condition_dir
        config = ExperimentConfig(
            name=f"phase3_{args.benchmark}_{args.model_key}_{condition_dir}",
            dataset=args.benchmark,
            dataset_path=dataset_path,
            variants=(variant_name,),
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
        print(f"\n=== {args.benchmark} / {args.model_key} / {condition_dir} "
              f"(variant={variant_name}, gate={gate_on}) ===", flush=True)
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
                "model_key": args.model_key,
                "condition": condition_dir,
                "variant": variant_name,
                "contradiction_requires_shared_subject": gate_on,
                "examples": manifest.num_examples,
                "seconds": round(elapsed, 1),
                "output_dir": str(out_dir),
            }
        )

    summary_path = (
        Path("artifacts/dev100-bottleneck/PHASE3") / f"{args.benchmark}_{args.model_key}_run_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
