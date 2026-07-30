#!/usr/bin/env python
"""Aggregate every completed run under artifacts/{qwen-matrix,benchmark-matrix}/
into CSV/JSON summary tables: failure rate, token F1, citation recall, runtime,
GPU peak memory, and an automatic failure-category breakdown (categories are
discovered from the actual error strings, not assumed in advance). Also
computes paired bootstrap CIs (egrag.experiments.stats, already implemented;
no significance/p-values -- the project's own stats module deliberately does
not claim them) for full_egrag vs the strongest baseline, where both exist in
the same run.

Reads only what has actually completed; a run directory that doesn't exist yet
is skipped and reported as such, never fabricated. Every number here is
computed directly from results.jsonl/aggregate.json/manifest.json.

Run:
    uv run python scripts/analyze_benchmark_results.py
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

from egrag.experiments.stats import paired_comparison

ROOT = Path("artifacts")
OUT = ROOT / "benchmark-matrix" / "analysis"

RUN_GROUPS = {
    "synthetic": {
        "qwen2.5-3b-instruct": ROOT / "qwen-matrix" / "qwen2.5-3b-instruct",
        "qwen2.5-7b-instruct": ROOT / "qwen-matrix" / "qwen2.5-7b-instruct",
        "qwen3.5-9b": ROOT / "qwen-matrix" / "qwen3.5-9b",
    },
}
for benchmark in ("fever", "hotpotqa"):
    for mnt_tag, mnt_label in (("", "mnt64"), ("_mnt256", "mnt256")):
        for model in ("qwen2.5-3b-instruct", "qwen2.5-7b-instruct", "qwen3.5-9b"):
            group = f"{benchmark}_{mnt_label}"
            RUN_GROUPS.setdefault(group, {})[model] = (
                ROOT / "benchmark-matrix" / benchmark / f"{model}{mnt_tag}"
            )


def _find_datasets(model_root: Path) -> list[Path]:
    if not model_root.is_dir():
        return []
    if (model_root / "results.jsonl").is_file():
        return [model_root]
    return sorted(p for p in model_root.iterdir() if (p / "results.jsonl").is_file())


CID_RE = re.compile(r"unknown claim id\(s\): \[(.*)\]'?$")


def _citation_id_pattern(cited_id: str) -> str:
    if "[" in cited_id or "]" in cited_id:
        return "bracket_formatting_artifact"
    if "source=" in cited_id or "belief=" in cited_id:
        return "evidence_metadata_leaked_as_id"
    if re.fullmatch(r"c[0-9a-f]{16,}", cited_id):
        return "clm_prefix_truncated"
    if re.fullmatch(r"[0-9a-f]{16,}", cited_id):
        return "hash_prefix_dropped_entirely"
    return "short_or_compound_id_mismatch"


def _classify_failure(error: str) -> str:
    if error.startswith("GenerationError: answer cites unknown claim id"):
        return "hallucinated_citation_id"
    if "truncated or unterminated JSON" in error or "no valid top-level JSON" in error:
        return "generation_truncated_or_malformed_json"
    if error.startswith("GenerationError:"):
        return "other_generation_error"
    exc_type = error.split(":")[0]
    return f"other:{exc_type}"


def load_results(run_dir: Path) -> list[dict]:
    p = run_dir / "results.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def summarize_group(group_name: str, models: dict[str, Path]) -> dict:
    rows = {}
    for model_key, model_root in models.items():
        dataset_dirs = _find_datasets(model_root)
        if not dataset_dirs:
            rows[model_key] = {"status": "not_yet_run", "path": str(model_root)}
            continue
        all_results: list[dict] = []
        for d in dataset_dirs:
            all_results.extend(load_results(d))
        n = len(all_results)
        failed = [r for r in all_results if r["failed"]]
        ok = [r for r in all_results if not r["failed"]]
        tf1 = [r["metrics"]["token_f1"] for r in ok if "token_f1" in r.get("metrics", {})]
        cr = [
            r["metrics"]["citation_recall"] for r in ok if "citation_recall" in r.get("metrics", {})
        ]
        aa = [
            r["metrics"]["answer_accuracy"] for r in ok if "answer_accuracy" in r.get("metrics", {})
        ]
        latencies = [r["latency_ms"] for r in all_results]
        mem = [r["peak_memory_kb"] for r in all_results if r.get("peak_memory_kb") is not None]
        rows[model_key] = {
            "status": "complete",
            "n": n,
            "n_datasets": len(dataset_dirs),
            "failed": len(failed),
            "failure_rate": len(failed) / n if n else None,
            # "_mean": average over successful generations only -- NOT comparable
            # across conditions with different failure rates, since the set of
            # "successes" is a different (self-selected) subpopulation each time.
            "token_f1_mean": mean(tf1) if tf1 else None,
            "token_f1_n": len(tf1),
            "citation_recall_mean": mean(cr) if cr else None,
            "citation_recall_n": len(cr),
            "answer_accuracy_mean": mean(aa) if aa else None,
            "answer_accuracy_n": len(aa),
            # "_adjusted": sum over ALL n examples (failures scored 0) -- the
            # primary metric for comparing across conditions/models/token
            # budgets, since it isn't distorted by a shifting success population.
            "token_f1_adjusted": sum(tf1) / n if n else None,
            "citation_recall_adjusted": sum(cr) / n if n else None,
            "answer_accuracy_adjusted": sum(aa) / n if n else None,
            "latency_ms_mean": mean(latencies) if latencies else None,
            "latency_ms_stdev": stdev(latencies) if len(latencies) > 1 else None,
            "peak_memory_kb_mean": mean(mem) if mem else None,
            "peak_memory_kb_max": max(mem) if mem else None,
            "peak_memory_n": len(mem),
        }
    return rows


_BASELINE_CANDIDATES = ("passage_rag", "reranked_passage_rag", "claim_only_rag")


def bootstrap_comparisons(models: dict[str, Path]) -> dict:
    """Paired bootstrap CI, full_egrag vs the strongest baseline present, per
    model, per metric. No significance test / p-value: matches this project's
    existing stats.py, which deliberately reports only bootstrap CIs.
    """

    out: dict[str, list[dict]] = {}
    for model_key, model_root in models.items():
        dataset_dirs = _find_datasets(model_root)
        if not dataset_dirs:
            continue
        all_results = [r for d in dataset_dirs for r in load_results(d)]
        by_variant: dict[str, dict[str, dict]] = {}
        for r in all_results:
            if r["failed"]:
                continue
            by_variant.setdefault(r["variant"], {})[r["example_id"]] = r
        if "full_egrag" not in by_variant:
            continue
        baselines = [v for v in _BASELINE_CANDIDATES if v in by_variant]
        if not baselines:
            continue

        def _mean_metric(variant: str, metric: str, results_by_variant: dict = by_variant) -> float:
            vals = [
                r["metrics"][metric]
                for r in results_by_variant[variant].values()
                if metric in r["metrics"]
            ]
            return mean(vals) if vals else -1.0

        best_baseline = max(baselines, key=lambda v: _mean_metric(v, "token_f1"))
        comparisons = []
        for metric in ("token_f1", "citation_recall"):
            a_ids = set(by_variant["full_egrag"])
            b_ids = set(by_variant[best_baseline])
            common = sorted(a_ids & b_ids)
            paired = [
                (
                    by_variant["full_egrag"][i]["metrics"].get(metric, 0.0),
                    by_variant[best_baseline][i]["metrics"].get(metric, 0.0),
                )
                for i in common
            ]
            if not paired:
                continue
            result = paired_comparison(
                metric, "full_egrag", best_baseline, paired, samples=2000, seed=0
            )
            comparisons.append(result.model_dump(mode="json"))
        if comparisons:
            out[model_key] = comparisons
    return out


def failure_taxonomy(models: dict[str, Path]) -> dict:
    per_model_cat: dict[str, Counter] = {}
    per_model_id_pattern: dict[str, Counter] = {}
    total_cat: Counter = Counter()
    total_id_pattern: Counter = Counter()
    for model_key, model_root in models.items():
        dataset_dirs = _find_datasets(model_root)
        if not dataset_dirs:
            continue
        cat_counter: Counter = Counter()
        id_counter: Counter = Counter()
        for d in dataset_dirs:
            for r in load_results(d):
                if not r["failed"]:
                    continue
                cat = _classify_failure(r["error"])
                cat_counter[cat] += 1
                total_cat[cat] += 1
                m = CID_RE.search(r["error"])
                if m:
                    for cid in re.findall(r"'([^']*)'", m.group(1)):
                        id_counter[_citation_id_pattern(cid)] += 1
                        total_id_pattern[_citation_id_pattern(cid)] += 1
        if cat_counter:
            per_model_cat[model_key] = cat_counter
            per_model_id_pattern[model_key] = id_counter
    return {
        "per_model_category": {k: dict(v) for k, v in per_model_cat.items()},
        "per_model_citation_id_pattern": {k: dict(v) for k, v in per_model_id_pattern.items()},
        "overall_category": dict(total_cat),
        "overall_citation_id_pattern": dict(total_id_pattern),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_summaries = {}
    all_taxonomies = {}
    all_comparisons = {}
    for group_name, models in RUN_GROUPS.items():
        all_summaries[group_name] = summarize_group(group_name, models)
        all_taxonomies[group_name] = failure_taxonomy(models)
        all_comparisons[group_name] = bootstrap_comparisons(models)
    (OUT / "bootstrap_comparisons.json").write_text(json.dumps(all_comparisons, indent=2))

    (OUT / "metrics_summary.json").write_text(json.dumps(all_summaries, indent=2))
    (OUT / "failure_taxonomy.json").write_text(json.dumps(all_taxonomies, indent=2))

    # Flat CSV for easy inspection / spreadsheet import.
    csv_path = OUT / "metrics_summary.csv"
    fields = [
        "group",
        "model",
        "status",
        "n",
        "failed",
        "failure_rate",
        "token_f1_mean",
        "token_f1_adjusted",
        "citation_recall_mean",
        "citation_recall_adjusted",
        "answer_accuracy_mean",
        "answer_accuracy_adjusted",
        "latency_ms_mean",
        "peak_memory_kb_mean",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for group_name, rows in all_summaries.items():
            for model_key, row in rows.items():
                w.writerow(
                    {"group": group_name, "model": model_key, **{k: row.get(k) for k in fields[2:]}}
                )

    print(f"wrote {OUT / 'metrics_summary.json'}")
    print(f"wrote {OUT / 'metrics_summary.csv'}")
    print(f"wrote {OUT / 'failure_taxonomy.json'}")

    # Quick human-readable status printout.
    for group_name, rows in all_summaries.items():
        print(f"\n{group_name}:")
        for model_key, row in rows.items():
            if row["status"] != "complete":
                print(f"  {model_key}: NOT YET RUN")
                continue
            print(
                f"  {model_key}: n={row['n']} failed={row['failed']} "
                f"({row['failure_rate']:.0%}) token_f1={row['token_f1_mean']}"
            )


if __name__ == "__main__":
    main()
