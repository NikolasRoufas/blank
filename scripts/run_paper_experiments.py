#!/usr/bin/env python
"""Reproducible driver for the local EG-RAG paper experiments.

Runs only implemented variants/datasets with the deterministic fake generator,
preserves raw + aggregate results, and emits machine-readable tables, statistical
comparisons, figure scripts (+ metadata), qualitative cases, a validation report,
and collated failures under ``artifacts/paper-results/``. No network, no LaTeX,
no tuning, no fabrication. Run:

    PYTHONPATH=src .venv/bin/python scripts/run_paper_experiments.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

from egrag.experiments import ExperimentConfig, ExperimentRunner, compare, summarize
from egrag.experiments.artifacts import RESULTS_FILE, read_results
from egrag.experiments.fairness import same_example_ids
from egrag.experiments.metrics import METRIC_KINDS, metric_kind
from egrag.experiments.models import ExampleResult
from egrag.experiments.stats import bootstrap_ci, mean

ROOT = Path("artifacts/paper-results")
RUNS = ROOT / "runs"
MAIN_VARIANTS = [
    "passage_rag",
    "reranked_passage_rag",
    "claim_only_rag",
    "graph_no_propagation",
    "graph_top_claim",
    "graph_coherent_subgraph",
    "graph_no_temporal",
    "graph_no_contradiction",
    "graph_with_propagation",
    "full_egrag",
]
MAIN_DATASETS = ["synthetic_graph", "temporal_conflict"]
ABLATION_VARIANTS = [
    "full_egrag",
    "graph_no_propagation",
    "graph_no_temporal",
    "graph_no_contradiction",
    "graph_top_claim",
]
BUDGETS = [128, 256, 512]
BUDGET_VARIANTS = ["passage_rag", "claim_only_rag", "full_egrag"]
SEEDS = (42, 123, 2026)
TOP_K = 3
EVIDENCE_BUDGET = 256
RESERVED = 64
BOOTSTRAP = 10000
BOOTSTRAP_SEED = 12345

HIGHER_BETTER = {
    "exact_match",
    "normalized_exact_match",
    "token_f1",
    "answer_accuracy",
    "citation_precision",
    "citation_recall",
    "citation_completeness",
    "evidence_precision",
    "evidence_recall",
    "answer_evidence_entailment",
}
LOWER_BETTER = {
    "invalid_citations",
    "empty_prediction",
    "contradiction_rate",
    "unsupported_claim_rate",
    "latency_ms",
}
ANSWER_EVID_METRICS = [
    "exact_match",
    "normalized_exact_match",
    "token_f1",
    "answer_accuracy",
    "citation_precision",
    "citation_recall",
    "citation_completeness",
    "evidence_precision",
    "evidence_recall",
    "answer_evidence_entailment",
    "contradiction_rate",
    "unsupported_claim_rate",
    "invalid_citations",
    "empty_prediction",
]
COUNT_METRICS = [
    "num_passages",
    "num_claims",
    "num_graph_nodes",
    "num_graph_edges",
    "num_selected",
    "token_estimate",
    "latency_ms",
]


def run_experiment(name: str, dataset: str, variants: list[str], out: Path, budget: int) -> Path:
    cfg = ExperimentConfig(
        name=name,
        dataset=dataset,
        variants=tuple(variants),
        seeds=SEEDS,
        output_dir=str(out),
        generator="fake",
        top_k=TOP_K,
        evidence_token_budget=budget,
        reserved_output_tokens=min(RESERVED, budget),
        enforce_fairness=True,
        bootstrap_samples=BOOTSTRAP,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    (ROOT / "frozen-configs" / f"{name}.json").write_text(
        json.dumps(cfg.model_dump(mode="json"), indent=2, sort_keys=True)
    )
    ExperimentRunner(cfg).run()
    return out


def metric_values(
    results: list[ExampleResult], variant: str, metric: str, seed: int
) -> list[float]:
    return [
        r.metrics[metric]
        for r in results
        if r.variant == variant and r.seed == seed and not r.failed and metric in r.metrics
    ]


def collate_run(tag: str, run_dir: Path) -> list[ExampleResult]:
    shutil.copyfile(run_dir / RESULTS_FILE, ROOT / "per-example" / f"{tag}.jsonl")
    shutil.copyfile(run_dir / "aggregate.json", ROOT / "aggregates" / f"{tag}.json")
    shutil.copyfile(run_dir / "manifest.json", ROOT / "manifests" / f"{tag}.json")
    results, _ = read_results(run_dir / RESULTS_FILE)
    return results


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def md_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    head = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in columns) + " |" for r in rows]
    return "\n".join([head, sep, *body]) + "\n"


def fmt(value: float) -> float:
    return round(value, 4)


def main() -> None:
    for sub in [
        "frozen-configs",
        "per-example",
        "aggregates",
        "manifests",
        "comparisons",
        "tables",
        "figures",
        "qualitative",
        "logs",
    ]:
        (ROOT / sub).mkdir(parents=True, exist_ok=True)

    # ---- run main + budget experiments ----
    main_results: dict[str, list[ExampleResult]] = {}
    for dataset in MAIN_DATASETS:
        out = RUNS / "main" / dataset
        if out.exists():
            shutil.rmtree(out)
        run_experiment(f"main_{dataset}", dataset, MAIN_VARIANTS, out, EVIDENCE_BUDGET)
        main_results[dataset] = collate_run(f"main_{dataset}", out)

    budget_results: dict[int, list[ExampleResult]] = {}
    for budget in BUDGETS:
        out = RUNS / "budget" / str(budget)
        if out.exists():
            shutil.rmtree(out)
        run_experiment(f"budget_{budget}", "synthetic_graph", BUDGET_VARIANTS, out, budget)
        budget_results[budget] = collate_run(f"budget_{budget}", out)

    # ---- determinism check across seeds ----
    determinism: dict[str, bool] = {}
    for dataset, results in main_results.items():
        for variant in MAIN_VARIANTS:
            seed_sets = [
                tuple(
                    sorted(
                        (
                            r.example_id,
                            round(r.metrics.get("token_f1", -1), 6),
                            round(r.metrics.get("citation_recall", -1), 6),
                        )
                        for r in results
                        if r.variant == variant and r.seed == s and not r.failed
                    )
                )
                for s in SEEDS
            ]
            determinism[f"{dataset}|{variant}"] = all(s == seed_sets[0] for s in seed_sets)
    (ROOT / "logs" / "determinism.json").write_text(
        json.dumps(determinism, indent=2, sort_keys=True)
    )

    # ---- failed examples collation ----
    failed_path = ROOT / "failed-examples.jsonl"
    with failed_path.open("w", encoding="utf-8") as fh:
        for dataset, results in main_results.items():
            for r in results:
                if r.failed:
                    fh.write(
                        json.dumps(
                            {
                                "dataset": dataset,
                                "split": "test",
                                "system": r.variant,
                                "generator": "fake",
                                "seed": r.seed,
                                "example_id": r.example_id,
                                "failed_stage": "run_system",
                                "exception_type": (r.error or "").split(":")[0],
                                "message": r.error,
                                "retry_status": "not_retried (deterministic)",
                                "final_status": "failed",
                            }
                        )
                        + "\n"
                    )

    # ---- build tables (use seed=42; determinism verified separately) ----
    def rows_for(
        results: list[ExampleResult],
        dataset: str,
        variants: list[str],
        metrics: list[str],
        with_ci: bool,
    ) -> list[dict[str, object]]:
        rows = []
        for variant in variants:
            n = len({r.example_id for r in results if r.variant == variant and r.seed == 42})
            n_fail = len([r for r in results if r.variant == variant and r.seed == 42 and r.failed])
            row: dict[str, object] = {
                "dataset": dataset,
                "system": variant,
                "n_examples": n,
                "n_failed": n_fail,
            }
            for m in metrics:
                vals = metric_values(results, variant, m, 42)
                row[m] = fmt(mean(vals)) if vals else ""
                if with_ci and m in ANSWER_EVID_METRICS and vals:
                    lo, hi = bootstrap_ci(vals, samples=BOOTSTRAP, seed=BOOTSTRAP_SEED)
                    row[f"{m}_ci_low"] = fmt(lo)
                    row[f"{m}_ci_high"] = fmt(hi)
            rows.append(row)
        return rows

    # main-results.csv
    main_metrics = list(ANSWER_EVID_METRICS)
    main_cols = ["dataset", "system", "n_examples", "n_failed"]
    for m in main_metrics:
        main_cols.append(m)
        if m in ANSWER_EVID_METRICS:
            main_cols += [f"{m}_ci_low", f"{m}_ci_high"]
    main_rows: list[dict[str, object]] = []
    for dataset in MAIN_DATASETS:
        main_rows += rows_for(main_results[dataset], dataset, MAIN_VARIANTS, main_metrics, True)
    write_csv(ROOT / "main-results.csv", main_rows, main_cols)
    (ROOT / "tables" / "main-results.md").write_text(
        md_table(
            main_rows,
            [
                "dataset",
                "system",
                "n_examples",
                "n_failed",
                "token_f1",
                "citation_recall",
                "evidence_recall",
                "answer_evidence_entailment",
            ],
        )
    )

    # ablation-results.csv (synthetic_graph)
    abl_rows = rows_for(
        main_results["synthetic_graph"], "synthetic_graph", ABLATION_VARIANTS, main_metrics, True
    )
    write_csv(ROOT / "ablation-results.csv", abl_rows, main_cols)
    (ROOT / "tables" / "ablation-results.md").write_text(
        md_table(
            abl_rows, ["system", "token_f1", "citation_recall", "evidence_recall", "num_selected"]
        )
    )

    # robustness-results.csv (budget sweep)
    rob_cols = [
        "dataset",
        "evidence_token_budget",
        "system",
        "n_examples",
        "token_f1",
        "citation_recall",
        "evidence_recall",
        "num_selected",
        "latency_ms",
    ]
    rob_rows: list[dict[str, object]] = []
    for budget in BUDGETS:
        res = budget_results[budget]
        for variant in BUDGET_VARIANTS:
            row = {
                "dataset": "synthetic_graph",
                "evidence_token_budget": budget,
                "system": variant,
                "n_examples": len(
                    {r.example_id for r in res if r.variant == variant and r.seed == 42}
                ),
            }
            for m in [
                "token_f1",
                "citation_recall",
                "evidence_recall",
                "num_selected",
                "latency_ms",
            ]:
                vals = metric_values(res, variant, m, 42)
                row[m] = fmt(mean(vals)) if vals else ""
            rob_rows.append(row)
    write_csv(ROOT / "robustness-results.csv", rob_rows, rob_cols)
    (ROOT / "tables" / "robustness-results.md").write_text(md_table(rob_rows, rob_cols))

    # efficiency-results.csv
    eff_cols = ["dataset", "system", *COUNT_METRICS]
    eff_rows: list[dict[str, object]] = []
    for dataset in MAIN_DATASETS:
        eff_rows += rows_for(main_results[dataset], dataset, MAIN_VARIANTS, COUNT_METRICS, False)
    for row in eff_rows:
        for k in ["n_examples", "n_failed"]:
            row.pop(k, None)
    write_csv(ROOT / "efficiency-results.csv", eff_rows, eff_cols)
    (ROOT / "tables" / "efficiency-results.md").write_text(md_table(eff_rows, eff_cols))

    # ---- statistical comparisons: full_egrag vs strongest valid baseline ----
    comparisons = []
    for dataset in MAIN_DATASETS:
        run_dir = RUNS / "main" / dataset
        baselines = ["passage_rag", "reranked_passage_rag", "claim_only_rag"]
        # strongest baseline by mean token_f1 (a deterministic answer metric)
        best = max(
            baselines, key=lambda v: mean(metric_values(main_results[dataset], v, "token_f1", 42))
        )
        for metric in [
            "token_f1",
            "citation_recall",
            "evidence_recall",
            "answer_evidence_entailment",
        ]:
            cr = compare(
                run_dir,
                "full_egrag",
                best,
                metric,
                seed=42,
                samples=BOOTSTRAP,
                bootstrap_seed=BOOTSTRAP_SEED,
            )
            comparisons.append(
                {
                    "dataset": dataset,
                    "strongest_baseline_by_token_f1": best,
                    **cr.model_dump(mode="json"),
                }
            )
    (ROOT / "statistical-comparisons.json").write_text(json.dumps(comparisons, indent=2))

    # ---- result validation: recompute aggregates from per-example ----
    validation = {"checks": [], "ok": True}

    def check(name: str, ok: bool, detail: str = "") -> None:
        validation["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            validation["ok"] = False

    for dataset, results in main_results.items():
        run_dir = RUNS / "main" / dataset
        saved = summarize(run_dir)
        for variant in MAIN_VARIANTS:
            agg = saved.get(f"{variant}|42")
            if agg is None:
                check(f"aggregate-present:{dataset}:{variant}", False)
                continue
            for m, saved_val in agg.metrics.items():
                recomputed = mean(metric_values(results, variant, m, 42))
                close = abs(recomputed - saved_val) < 1e-9
                if not close:
                    check(
                        f"recompute:{dataset}:{variant}:{m}",
                        False,
                        f"saved={saved_val} recomputed={recomputed}",
                    )
        # duplicate / missing IDs
        ids = [r.example_id for r in results if r.variant == "full_egrag" and r.seed == 42]
        check(f"no-duplicate-ids:{dataset}", len(ids) == len(set(ids)))
        check(
            f"same-example-ids:{dataset}",
            same_example_ids(
                {v: [r for r in results if r.variant == v and r.seed == 42] for v in MAIN_VARIANTS}
            )
            == [],
        )
    check(
        "all-seeds-deterministic",
        all(determinism.values()),
        detail=json.dumps({k: v for k, v in determinism.items() if not v}),
    )
    (ROOT / "logs" / "validation.json").write_text(json.dumps(validation, indent=2))

    # ---- figure scripts + metadata (matplotlib absent -> not rendered) ----
    figures = {
        "quality_vs_budget": {
            "source": "robustness-results.csv",
            "x": "evidence_token_budget",
            "y": ["token_f1", "citation_recall", "evidence_recall"],
            "systems": BUDGET_VARIANTS,
        },
        "latency_by_variant": {
            "source": "efficiency-results.csv",
            "x": "system",
            "y": ["latency_ms"],
            "systems": MAIN_VARIANTS,
        },
    }
    for fig, meta in figures.items():
        (ROOT / "figures" / f"{fig}.metadata.json").write_text(
            json.dumps(
                {
                    **meta,
                    "rendering": "not rendered (matplotlib not installed; no network to install)",
                    "outputs_expected": [f"{fig}.pdf", f"{fig}.png"],
                    "generation_command": f"python artifacts/paper-results/figures/{fig}.py",
                },
                indent=2,
            )
        )
        (ROOT / "figures" / f"{fig}.py").write_text(
            f'''#!/usr/bin/env python
"""Render the {fig} figure from {meta["source"]}. Requires matplotlib (not bundled)."""
import csv, sys
from pathlib import Path
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib not installed; install it to render this figure")
src = Path(__file__).resolve().parents[1] / "{meta["source"]}"
rows = list(csv.DictReader(src.open()))
meta = {json.dumps(meta)}
fig, ax = plt.subplots()
for system in meta["systems"]:
    srows = [r for r in rows if r.get("system") == system]
    for y in meta["y"]:
        xs = [r[meta["x"]] for r in srows if r.get(y) not in (None, "")]
        ys = [float(r[y]) for r in srows if r.get(y) not in (None, "")]
        if ys:
            ax.plot(xs, ys, marker="o", label=f"{{system}}:{{y}}")
ax.set_xlabel(meta["x"]); ax.set_ylabel(",".join(meta["y"])); ax.legend(fontsize=6)
out = Path(__file__).with_suffix("")
fig.savefig(str(out) + ".pdf"); fig.savefig(str(out) + ".png", dpi=150)
print("wrote", out)
'''
        )

    # ---- qualitative cases: dump per-variant package/graph for both datasets ----
    qual = []
    for dataset in MAIN_DATASETS:
        run_dir = RUNS / "main" / dataset
        results = main_results[dataset]
        for r in [x for x in results if x.seed == 42]:
            pkg = run_dir / "packages" / f"{r.variant}__{r.example_id}.json"
            graph = run_dir / "graphs" / f"{r.variant}__{r.example_id}.json"
            rec = {
                "dataset": dataset,
                "system": r.variant,
                "example_id": r.example_id,
                "answer": r.answer,
                "citations": list(r.citations),
                "evidence_source_ids": list(r.evidence_source_ids),
                "metrics": r.metrics,
                "counts": r.counts,
                "warnings": list(r.warnings),
                "package_file": str(pkg) if pkg.exists() else None,
                "graph_file": str(graph) if graph.exists() else None,
                "failed": r.failed,
                "error": r.error,
            }
            qual.append(rec)
    (ROOT / "qualitative" / "all-cases.jsonl").write_text(
        "\n".join(json.dumps(q) for q in qual) + "\n"
    )

    # ---- frozen-config checksums ----
    checksums = {}
    for cfg_file in sorted((ROOT / "frozen-configs").glob("*.json")):
        checksums[cfg_file.name] = hashlib.sha256(cfg_file.read_bytes()).hexdigest()
    # also checksum the shipped pipeline configs for reference
    for cfg_file in sorted(Path("configs").glob("*.yaml")):
        checksums[f"configs/{cfg_file.name}"] = hashlib.sha256(cfg_file.read_bytes()).hexdigest()
    (ROOT / "frozen-configs" / "checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True)
    )

    # ---- metric kinds reference ----
    (ROOT / "logs" / "metric-kinds.json").write_text(
        json.dumps({m: str(metric_kind(m)) for m in sorted(set(METRIC_KINDS))}, indent=2)
    )

    print(
        f"DRIVER_OK datasets={len(MAIN_DATASETS)} budgets={len(BUDGETS)} "
        f"comparisons={len(comparisons)} validation_ok={validation['ok']}"
    )


if __name__ == "__main__":
    main()
