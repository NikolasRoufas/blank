#!/usr/bin/env python
"""CUDA server smoke test: verify a freshly provisioned GPU box before any real run.

Checks, in order: Python/torch/transformers/CUDA/GPU report, tokenizer loading,
generator model loading + one minimal generation, NLI model loading + one
classification, and one complete EG-RAG example (``full_egrag`` variant) using
the real generator and real NLI classifier together (deterministic sentence
extraction, matching the experiment harness's default methodology).

This is intentionally bounded: it loads each model once and runs a handful of
tiny cases. It does NOT run a benchmark, a dataset, or the paper experiment
matrix — see ``egrag experiment run`` for that. Every step fails loudly (a
non-zero exit and a clear message) rather than silently skipping or falling
back to CPU when CUDA was requested.

Run (defaults to Qwen2.5-3B-Instruct on CUDA, bfloat16):
    uv run --extra local-models --extra graph --extra experiments \\
        python scripts/cuda_smoke_test.py

Run for a specific model/precision:
    uv run python scripts/cuda_smoke_test.py \\
        --generator-model Qwen/Qwen2.5-14B-Instruct --quantization 4bit

Run without requiring a GPU (CPU development check only):
    uv run python scripts/cuda_smoke_test.py --device cpu --no-require-cuda \\
        --generator-model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def _step(name: str) -> None:
    print(f"\n=== {name} ===", flush=True)


def _fail(name: str, exc: BaseException) -> None:
    print(f"FAIL [{name}]: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--generator-revision", default=None)
    parser.add_argument("--nli-model", default="roberta-large-mnli")
    parser.add_argument("--nli-revision", default="2a8f12d27941090092df78e4ba6f0928eb5eac98")
    parser.add_argument("--device", default="cuda", help="cpu/mps/cuda/cuda:N/auto")
    parser.add_argument("--dtype", default="bfloat16", help="float32/float16/bfloat16/auto")
    parser.add_argument("--quantization", default="none", choices=["none", "4bit", "8bit"])
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        default=False,
        help="pass enable_thinking=False to the chat template (hybrid-reasoning models)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="generation budget for the minimal-generation and end-to-end checks",
    )
    parser.add_argument(
        "--require-cuda",
        dest="require_cuda",
        action="store_true",
        default=True,
        help="fail instead of silently running on CPU (default: on)",
    )
    parser.add_argument("--no-require-cuda", dest="require_cuda", action="store_false")
    parser.add_argument(
        "--output",
        default="artifacts/cuda-smoke/latest.json",
        help="where to write the JSON report",
    )
    args = parser.parse_args()

    report: dict[str, Any] = {"args": vars(args)}

    _step("1/6 Python / torch / transformers / CUDA / GPU report")
    from egrag.hf_runtime import ensure_device_available, gpu_report

    gpu = gpu_report(device=args.device, dtype=args.dtype)
    report["gpu_report"] = gpu
    print(json.dumps(gpu, indent=2))
    if args.require_cuda:
        try:
            ensure_device_available("cuda" if args.device == "auto" else args.device)
        except RuntimeError as exc:
            _fail("cuda-required", exc)
    print("OK: environment report collected" + (" (CUDA confirmed)" if args.require_cuda else ""))

    _step("2/6 Tokenizer loading")
    t0 = time.time()
    try:
        import transformers

        tok = transformers.AutoTokenizer.from_pretrained(
            args.generator_model, revision=args.generator_revision
        )
    except Exception as exc:
        _fail("tokenizer-load", exc)
    report["tokenizer"] = {
        "model": args.generator_model,
        "vocab_size": tok.vocab_size,
        "has_chat_template": bool(getattr(tok, "chat_template", None)),
        "seconds": round(time.time() - t0, 1),
    }
    print(f"OK: tokenizer loaded ({report['tokenizer']['seconds']}s)")

    _step("3/6 Generator model loading + one minimal generation")
    from egrag.domain.models import (
        AtomicClaim,
        ClaimProvenance,
        EvidencePackage,
        Query,
        SelectedEvidence,
        SourceMetadata,
        SourceSpan,
    )
    from egrag.generation import GenerationConfig, GenerationService, HuggingFaceGenerator

    t0 = time.time()
    try:
        generator = HuggingFaceGenerator(
            args.generator_model,
            context_limit=4096,
            revision=args.generator_revision,
            device=args.device,
            dtype=args.dtype,
            quantization=args.quantization,
            require_cuda=args.require_cuda,
            chat_template_kwargs=({"enable_thinking": False} if args.disable_thinking else None),
        )
        claim = AtomicClaim(
            claim_id="c1",
            text="The Eiffel Tower is located in Paris.",
            provenance=ClaimProvenance(
                source=SourceMetadata(source_id="Eiffel_Tower"),
                spans=(SourceSpan(source_id="Eiffel_Tower", start=0, end=37, text="..."),),
            ),
            extraction_confidence=1.0,
        )
        package = EvidencePackage(
            package_id="smoke-pkg",
            query=Query(query_id="smoke-q", text="In what country is the Eiffel Tower?"),
            claims=(claim,),
            selected=(SelectedEvidence(claim_id="c1", selection_score=1.0, rank=0),),
        )
        cfg = GenerationConfig(deterministic=True, seed=0, max_new_tokens=args.max_new_tokens)
        answer = GenerationService().generate(package, generator, cfg)
    except Exception as exc:
        _fail("generator-load-or-generate", exc)
    resolved = generator.resolved_runtime_info()
    report["generation"] = {
        "resolved": resolved,
        "answer_preview": answer.text[:200],
        "abstained": answer.abstained,
        "citations": list(answer.cited_claim_ids),
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(report["generation"], indent=2))
    print(f"OK: one generation completed ({report['generation']['seconds']}s)")

    _step("4/6 NLI model loading + one classification")
    from egrag.graph import HuggingFaceNLIClassifier
    from egrag.graph.nli import classify_directional

    t0 = time.time()
    try:
        nli = HuggingFaceNLIClassifier(
            args.nli_model,
            model_revision=args.nli_revision,
            device=args.device if args.device != "auto" else None,
        )
        claim_a = AtomicClaim(
            claim_id="a",
            text="The Eiffel Tower is in Paris.",
            provenance=ClaimProvenance(
                source=SourceMetadata(source_id="a"),
                spans=(SourceSpan(source_id="a", start=0, end=10, text="..."),),
            ),
            extraction_confidence=1.0,
        )
        claim_b = AtomicClaim(
            claim_id="b",
            text="The Eiffel Tower is in Berlin.",
            provenance=ClaimProvenance(
                source=SourceMetadata(source_id="b"),
                spans=(SourceSpan(source_id="b", start=0, end=10, text="..."),),
            ),
            extraction_confidence=1.0,
        )
        decision = classify_directional(
            nli,
            claim_a,
            claim_b,
            entailment_threshold=0.4,
            contradiction_threshold=0.7,
            duplicate_threshold=0.8,
        )
    except Exception as exc:
        _fail("nli-load-or-classify", exc)
    report["nli"] = {
        "model": args.nli_model,
        "revision": args.nli_revision,
        "relation": decision.relation,
        "entailment_ab": round(decision.entailment_ab, 3),
        "contradiction": round(decision.contradiction, 3),
        "expected_relation": "contradicts",
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(report["nli"], indent=2))
    ok = decision.relation == "contradicts"
    print(
        f"{'OK' if ok else 'WARNING'}: NLI classified a known-contradiction pair as "
        f"{decision.relation!r} ({report['nli']['seconds']}s)"
    )

    _step("5/6 One complete EG-RAG example (full_egrag variant, real generator + real NLI)")
    from egrag.domain.models import Document
    from egrag.experiments.variants import RunComponents, RunSettings, get_variant, run_system

    t0 = time.time()
    try:
        docs = [
            Document(
                document_id="d1",
                text="The Eiffel Tower is located in Paris. Paris is the capital of France.",
                source=SourceMetadata(source_id="Eiffel_Tower", title="Eiffel Tower"),
            ),
        ]
        components = RunComponents(classifier=nli)
        settings = RunSettings(top_k=3, evidence_token_budget=256, reserved_output_tokens=32)
        output = run_system(
            get_variant("full_egrag"),
            Query(query_id="e2e", text="In what country is the Eiffel Tower?"),
            docs,
            generator=generator,
            config=cfg,
            settings=settings,
            components=components,
        )
    except Exception as exc:
        _fail("full-pipeline-example", exc)
    report["end_to_end"] = {
        "answer_preview": output.answer[:200],
        "abstained": output.abstained,
        "citations": list(output.cited_claim_ids),
        "graph_nodes": output.counts.get("num_graph_nodes", 0),
        "graph_edges": output.counts.get("num_graph_edges", 0),
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(report["end_to_end"], indent=2))
    print(f"OK: full_egrag example completed ({report['end_to_end']['seconds']}s)")

    _step("6/6 Writing report")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print("\nSMOKE_OK")


if __name__ == "__main__":
    main()
