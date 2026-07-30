"""`egrag experiment` CLI subcommands (run, resume, compare, summarize, inspect)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NoReturn

import typer
import yaml

from egrag.domain.errors import ConfigurationError, EGRagError
from egrag.experiments import (
    ExperimentConfig,
    ExperimentRunner,
    compare,
    inspect_example,
    summarize,
)

experiment_app = typer.Typer(
    name="experiment",
    help="Run and analyze evaluation experiments (offline, deterministic fakes).",
    no_args_is_help=True,
    add_completion=False,
)

_EXIT_CONFIG = 2
_EXIT_FAILURE = 1


def _fail(message: str, code: int) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code)


def _build_config(
    name: str,
    dataset: str,
    variants: str,
    seeds: str,
    output_dir: str,
    generator: str,
    top_k: int,
    evidence_budget: int,
    limit: int | None,
    dataset_path: str | None,
    enforce_fairness: bool,
    generator_model: str | None = None,
    generator_revision: str | None = None,
    generator_device: str = "auto",
    generator_dtype: str | None = None,
    generator_quantization: str = "none",
    generator_disable_thinking: bool = False,
    require_cuda: bool = False,
) -> ExperimentConfig:
    try:
        return ExperimentConfig(
            name=name,
            dataset=dataset,
            dataset_path=dataset_path,
            variants=tuple(v.strip() for v in variants.split(",") if v.strip()),
            seeds=tuple(int(s) for s in seeds.split(",") if s.strip()),
            output_dir=output_dir,
            generator=generator,  # type: ignore[arg-type]
            generator_model=generator_model,
            generator_revision=generator_revision,
            generator_device=generator_device,
            generator_dtype=generator_dtype,
            generator_quantization=generator_quantization,  # type: ignore[arg-type]
            generator_disable_thinking=generator_disable_thinking,
            require_cuda=require_cuda,
            top_k=top_k,
            evidence_token_budget=evidence_budget,
            limit=limit,
            enforce_fairness=enforce_fairness,
        )
    except (ValueError, ConfigurationError) as exc:
        _fail(f"invalid experiment configuration: {exc}", _EXIT_CONFIG)


def _execute(config: ExperimentConfig, *, resume: bool) -> None:
    try:
        manifest = ExperimentRunner(config).run(resume=resume)
    except ConfigurationError as exc:
        _fail(str(exc), _EXIT_CONFIG)
    except EGRagError as exc:
        _fail(f"experiment failed: {exc}", _EXIT_FAILURE)
    typer.echo(f"experiment '{manifest.experiment_name}' wrote artifacts to {config.output_dir}")
    typer.echo(
        f"examples={manifest.num_examples} variants={len(manifest.variants)} "
        f"seeds={list(manifest.seeds)}"
    )
    if manifest.warnings:
        typer.echo(f"warnings: {len(manifest.warnings)}")


@experiment_app.command("run")
def run(
    name: str = typer.Option(..., "--name"),
    dataset: str = typer.Option("synthetic_graph", "--dataset"),
    variants: str = typer.Option("passage_rag,claim_only_rag,full_egrag", "--variants"),
    seeds: str = typer.Option("0", "--seeds"),
    output_dir: str = typer.Option(..., "--output-dir"),
    generator: str = typer.Option("fake", "--generator", help="fake or huggingface"),
    generator_model: str | None = typer.Option(
        None, "--generator-model", help="HF model id, e.g. Qwen/Qwen2.5-7B-Instruct"
    ),
    generator_revision: str | None = typer.Option(None, "--generator-revision"),
    generator_device: str = typer.Option("auto", "--generator-device", help="cpu/mps/cuda/auto"),
    generator_dtype: str | None = typer.Option(
        None, "--generator-dtype", help="float32/float16/bfloat16/auto"
    ),
    generator_quantization: str = typer.Option(
        "none", "--generator-quantization", help="none/4bit/8bit"
    ),
    generator_disable_thinking: bool = typer.Option(
        False,
        "--generator-disable-thinking/--generator-allow-thinking",
        help="pass enable_thinking=False to the chat template (hybrid-reasoning models)",
    ),
    require_cuda: bool = typer.Option(
        False, "--require-cuda/--no-require-cuda", help="fail instead of silently using CPU"
    ),
    top_k: int = typer.Option(4, "--top-k"),
    evidence_budget: int = typer.Option(256, "--evidence-budget"),
    limit: int | None = typer.Option(None, "--limit"),
    dataset_path: str | None = typer.Option(None, "--dataset-path"),
    enforce_fairness: bool = typer.Option(True, "--enforce-fairness/--no-fairness"),
) -> None:
    """Run an experiment from a clean output directory."""

    config = _build_config(
        name,
        dataset,
        variants,
        seeds,
        output_dir,
        generator,
        top_k,
        evidence_budget,
        limit,
        dataset_path,
        enforce_fairness,
        generator_model,
        generator_revision,
        generator_device,
        generator_dtype,
        generator_quantization,
        generator_disable_thinking,
        require_cuda,
    )
    _execute(config, resume=False)


@experiment_app.command("resume")
def resume(
    name: str = typer.Option(..., "--name"),
    dataset: str = typer.Option("synthetic_graph", "--dataset"),
    variants: str = typer.Option("passage_rag,claim_only_rag,full_egrag", "--variants"),
    seeds: str = typer.Option("0", "--seeds"),
    output_dir: str = typer.Option(..., "--output-dir"),
    generator: str = typer.Option("fake", "--generator", help="fake or huggingface"),
    generator_model: str | None = typer.Option(
        None, "--generator-model", help="HF model id, e.g. Qwen/Qwen2.5-7B-Instruct"
    ),
    generator_revision: str | None = typer.Option(None, "--generator-revision"),
    generator_device: str = typer.Option("auto", "--generator-device", help="cpu/mps/cuda/auto"),
    generator_dtype: str | None = typer.Option(
        None, "--generator-dtype", help="float32/float16/bfloat16/auto"
    ),
    generator_quantization: str = typer.Option(
        "none", "--generator-quantization", help="none/4bit/8bit"
    ),
    generator_disable_thinking: bool = typer.Option(
        False,
        "--generator-disable-thinking/--generator-allow-thinking",
        help="pass enable_thinking=False to the chat template (hybrid-reasoning models)",
    ),
    require_cuda: bool = typer.Option(
        False, "--require-cuda/--no-require-cuda", help="fail instead of silently using CPU"
    ),
    top_k: int = typer.Option(4, "--top-k"),
    evidence_budget: int = typer.Option(256, "--evidence-budget"),
    limit: int | None = typer.Option(None, "--limit"),
    dataset_path: str | None = typer.Option(None, "--dataset-path"),
    enforce_fairness: bool = typer.Option(True, "--enforce-fairness/--no-fairness"),
) -> None:
    """Resume an interrupted experiment without duplicating completed examples."""

    config = _build_config(
        name,
        dataset,
        variants,
        seeds,
        output_dir,
        generator,
        top_k,
        evidence_budget,
        limit,
        dataset_path,
        enforce_fairness,
        generator_model,
        generator_revision,
        generator_device,
        generator_dtype,
        generator_quantization,
        generator_disable_thinking,
        require_cuda,
    )
    _execute(config, resume=True)


@experiment_app.command("compare")
def compare_cmd(
    output_dir: str = typer.Option(..., "--output-dir"),
    variant_a: str = typer.Option(..., "--variant-a"),
    variant_b: str = typer.Option(..., "--variant-b"),
    metric: str = typer.Option("token_f1", "--metric"),
    seed: int = typer.Option(0, "--seed"),
    samples: int = typer.Option(1000, "--samples"),
) -> None:
    """Paired bootstrap comparison of two variants on a metric (no significance claim)."""

    result = compare(output_dir, variant_a, variant_b, metric, seed=seed, samples=samples)
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))


@experiment_app.command("summarize")
def summarize_cmd(output_dir: str = typer.Option(..., "--output-dir")) -> None:
    """Print aggregate metrics (with failure counts) for a completed run."""

    summary = summarize(output_dir)
    payload = {key: agg.model_dump(mode="json") for key, agg in summary.items()}
    typer.echo(json.dumps(payload, indent=2))


@experiment_app.command("inspect-example")
def inspect_example_cmd(
    output_dir: str = typer.Option(..., "--output-dir"),
    example_id: str = typer.Option(..., "--example-id"),
) -> None:
    """Show all per-variant results for a single example."""

    results = inspect_example(output_dir, example_id)
    if not results:
        _fail(f"no results for example {example_id!r}", _EXIT_FAILURE)
    typer.echo(json.dumps([r.model_dump(mode="json") for r in results], indent=2))


# Rough per-item cost constants (CPU, measured in the real-adapter-repair smokes;
# GPU is far faster). Used only for dry-run estimates — clearly labelled rough.
_EST = {
    "claims_per_example": 12,
    "nli_pairs_per_example": 40,
    "extract_seconds_per_example": 12.0,
    "nli_seconds_per_pair": 0.06,
    "generation_seconds_per_call": 1.6,
    "bytes_per_cache_entry": 1024,
}
_GRAPH_FAMILY_VARIANTS = {
    "graph_no_propagation",
    "graph_with_propagation",
    "graph_top_claim",
    "graph_coherent_subgraph",
    "graph_no_temporal",
    "graph_no_contradiction",
    "full_egrag",
}


def _read_json(path: Path) -> dict[str, Any]:
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


@experiment_app.command("matrix")
def matrix(
    benchmark: str = typer.Option(..., "--benchmark", help="hotpotqa or fever"),
    output_dir: str = typer.Option(..., "--output-dir"),
    sample: str | None = typer.Option(None, "--sample", help="path to a sample manifest JSON"),
    frozen_config: str | None = typer.Option(
        None, "--frozen-config", help="path to a frozen-config YAML (defaults by benchmark)"
    ),
    split: str = typer.Option("validation", "--split"),
    variants: str = typer.Option(
        "passage_rag,reranked_passage_rag,claim_only_rag,graph_no_propagation,"
        "graph_no_contradiction,graph_top_claim,graph_coherent_subgraph,full_egrag",
        "--variants",
    ),
    seed: int = typer.Option(0, "--seed"),
    device: str = typer.Option("auto", "--device"),
    cache_dir: str = typer.Option(".egrag-cache", "--cache-dir"),
    resume: bool = typer.Option(False, "--resume/--no-resume"),
    max_examples: int | None = typer.Option(None, "--max-examples"),
    extractor_model: str = typer.Option("Qwen/Qwen2.5-0.5B-Instruct", "--extractor-model"),
    nli_model: str = typer.Option("roberta-large-mnli", "--nli-model"),
    generator_model: str = typer.Option("Qwen/Qwen2.5-0.5B-Instruct", "--generator-model"),
    dry_run: bool = typer.Option(True, "--dry-run/--execute"),
) -> None:
    """Plan (dry-run) the final benchmark matrix from a frozen config.

    In dry-run mode this loads the frozen config + sample manifest and prints the
    resolved plan (fingerprint, revisions, variants, estimates, cache plan, output
    paths) WITHOUT running any inference. Execution is intentionally disabled in
    this milestone.
    """

    if not dry_run:
        _fail(
            "matrix execution is disabled in the real-adapter-repair milestone; "
            "use --dry-run (run the matrix in the dedicated final-experiment milestone)",
            _EXIT_CONFIG,
        )

    variant_list = [v.strip() for v in variants.split(",") if v.strip()]
    fc_path = Path(
        frozen_config or f"artifacts/benchmark-calibration/frozen-configs/{benchmark}.yaml"
    )
    frozen: dict[str, Any] = {}
    if fc_path.is_file():
        loaded = yaml.safe_load(fc_path.read_text(encoding="utf-8"))
        frozen = loaded if isinstance(loaded, dict) else {}

    sample_ids: list[str] = []
    fingerprint = frozen.get("dataset_fingerprint_sha256_prefix", "unknown")
    if sample:
        sp = Path(sample)
        if sp.is_file():
            manifest = _read_json(sp)
            sample_ids = list(manifest.get("example_ids", []))
            fingerprint = manifest.get("dataset_fingerprint_sha256", fingerprint)

    n = len(sample_ids) if sample_ids else (max_examples or 0)
    if max_examples is not None:
        n = min(n, max_examples) if n else max_examples

    graph_variants = [v for v in variant_list if v in _GRAPH_FAMILY_VARIANTS]
    # Extraction + NLI are variant-independent: computed once per example/pair and
    # reused across variants via the persistent cache. Generation is per variant.
    total_nli_pairs = n * _EST["nli_pairs_per_example"] if graph_variants else 0
    extract_seconds = n * _EST["extract_seconds_per_example"]
    nli_seconds = total_nli_pairs * _EST["nli_seconds_per_pair"]
    gen_calls = n * len(variant_list)
    gen_seconds = gen_calls * _EST["generation_seconds_per_call"]
    total_seconds = extract_seconds + nli_seconds + gen_seconds
    cache_entries = n + total_nli_pairs + gen_calls
    est_storage_mb = round(cache_entries * _EST["bytes_per_cache_entry"] / (1024**2), 1)

    out_root = Path(output_dir) / benchmark
    plan = {
        "mode": "DRY-RUN (no inference executed)",
        "benchmark": benchmark,
        "split": split,
        "frozen_config_path": str(fc_path),
        "frozen_config_found": fc_path.is_file(),
        "dataset_fingerprint": fingerprint,
        "sample_manifest": sample,
        "sample_count": n,
        "seed": seed,
        "device": device,
        "resume": resume,
        "models": {
            "extractor": extractor_model,
            "nli": {"model": nli_model, "revision": frozen.get("nli", {}).get("revision")},
            "generator": generator_model,
        },
        "resolved_settings": {
            "top_k": frozen.get("top_k"),
            "evidence_token_budget": frozen.get("evidence_token_budget"),
            "nli_thresholds": {
                k: frozen.get("nli", {}).get(k)
                for k in ("entailment_threshold", "contradiction_threshold", "duplicate_threshold")
            },
        },
        "variants": variant_list,
        "graph_variants": graph_variants,
        "estimates_rough_cpu": {
            "note": "rough CPU estimates from smoke latencies; GPU is far faster",
            "nli_pairs_total": total_nli_pairs,
            "extract_seconds": round(extract_seconds),
            "nli_seconds": round(nli_seconds),
            "generation_calls": gen_calls,
            "generation_seconds": round(gen_seconds),
            "total_seconds": round(total_seconds),
            "total_hours": round(total_seconds / 3600, 2),
            "estimated_cache_storage_mb": est_storage_mb,
        },
        "cache_reuse_plan": {
            "cache_dir": cache_dir,
            "extraction": "computed once per example, reused across all variants",
            "nli": "computed once per claim pair, reused across all graph variants",
            "generation": "computed per variant (varies with the evidence package)",
        },
        "output_paths": {
            "root": str(out_root),
            "per_example_jsonl": str(out_root / "results.jsonl"),
            "manifest": str(out_root / "manifest.json"),
            "aggregates": str(out_root / "aggregates.json"),
        },
    }
    typer.echo(json.dumps(plan, indent=2))


__all__ = ["experiment_app"]
