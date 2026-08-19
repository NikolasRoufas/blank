#!/usr/bin/env python
"""Analyze the dev-100 bottleneck-investigation run: coverage-adjusted metrics
per variant, paired bootstrap comparisons (H-GRAPH test), and relation-type
composition of full_egrag graphs. Writes a JSON + Markdown report under
artifacts/dev100-bottleneck/. Reads only what run_dev100_bottleneck.py wrote;
does not touch artifacts/benchmark-matrix/.

Run:
    uv run python scripts/analyze_dev100_bottleneck.py
"""

from __future__ import annotations

import json
from pathlib import Path

from egrag.experiments.stats import paired_comparison

ROOT = Path("artifacts/dev100-bottleneck")
MODEL_KEY = "qwen2.5-7b-instruct"
VARIANTS = ("passage_rag", "claim_only_rag", "graph_top_claim", "graph_no_contradiction", "full_egrag")
COMPARISONS = [
    # (a, b) -- H-GRAPH primary test: does removing contradiction edges recover
    # the loss relative to full_egrag, and relative to the no-graph reference?
    ("graph_no_contradiction", "full_egrag"),
    ("full_egrag", "claim_only_rag"),
    ("graph_no_contradiction", "claim_only_rag"),
    ("graph_top_claim", "full_egrag"),
    ("graph_top_claim", "claim_only_rag"),
]
METRICS = ("token_f1", "citation_recall", "answer_accuracy")


def load_results(benchmark: str) -> list[dict]:
    path = ROOT / benchmark / MODEL_KEY / "results.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def by_variant(results: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {v: [] for v in VARIANTS}
    for r in results:
        out.setdefault(r["variant"], []).append(r)
    return out


def coverage_adjusted(rs: list[dict], metric: str) -> float:
    """sum(metric over successful examples) / n_total -- ACL_RESULTS.md's convention."""
    if not rs:
        return float("nan")
    total = sum(r["metrics"].get(metric, 0.0) for r in rs if not r["failed"])
    return total / len(rs)


def variant_summary(rs: list[dict]) -> dict:
    n = len(rs)
    failed = sum(1 for r in rs if r["failed"])
    ok = [r for r in rs if not r["failed"]]

    def raw_mean(metric: str) -> float | None:
        vals = [r["metrics"].get(metric) for r in ok if r["metrics"].get(metric) is not None]
        return sum(vals) / len(vals) if vals else None

    def avg_count(key: str) -> float | None:
        vals = [r["counts"].get(key) for r in ok if r["counts"].get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "n": n,
        "failed": failed,
        "failure_rate": round(failed / n, 3) if n else None,
        **{f"{m}_mean_successful": (round(raw_mean(m), 4) if raw_mean(m) is not None else None) for m in METRICS},
        **{f"{m}_adjusted": round(coverage_adjusted(rs, m), 4) for m in METRICS},
        "avg_num_claims": round(avg_count("num_claims"), 3) if avg_count("num_claims") is not None else None,
        "avg_num_graph_edges": round(avg_count("num_graph_edges"), 3) if avg_count("num_graph_edges") is not None else None,
        "avg_num_selected": round(avg_count("num_selected"), 3) if avg_count("num_selected") is not None else None,
    }


def paired(results_by_variant: dict[str, list[dict]], a: str, b: str, metric: str) -> dict:
    a_by_id = {r["example_id"]: r for r in results_by_variant.get(a, [])}
    b_by_id = {r["example_id"]: r for r in results_by_variant.get(b, [])}
    common = sorted(set(a_by_id) & set(b_by_id))
    pairs = []
    for eid in common:
        ra, rb = a_by_id[eid], b_by_id[eid]
        if ra["failed"] or rb["failed"]:
            continue
        va = ra["metrics"].get(metric)
        vb = rb["metrics"].get(metric)
        if va is None or vb is None:
            continue
        pairs.append((va, vb))
    if not pairs:
        return {"metric": metric, "variant_a": a, "variant_b": b, "n_paired": 0}
    result = paired_comparison(metric, a, b, pairs, samples=2000, seed=0)
    return {
        "metric": result.metric,
        "variant_a": result.variant_a,
        "variant_b": result.variant_b,
        "n_paired": result.n_paired,
        "mean_a": round(result.mean_a, 4),
        "mean_b": round(result.mean_b, 4),
        "mean_delta": round(result.mean_delta, 4),
        "ci_95": [round(result.ci_low, 4), round(result.ci_high, 4)],
        "ci_excludes_zero": result.ci_low > 0 or result.ci_high < 0,
    }


def relation_composition(benchmark: str) -> dict:
    """Count relation_type occurrences across all full_egrag packages for a benchmark."""
    pkg_dir = ROOT / benchmark / MODEL_KEY / "packages"
    counts: dict[str, int] = {}
    n_packages = 0
    n_examples_with_edges = 0
    if not pkg_dir.is_dir():
        return {"n_packages": 0, "relation_type_counts": {}}
    for f in sorted(pkg_dir.glob("full_egrag__*.json")):
        n_packages += 1
        d = json.loads(f.read_text())
        rels = d.get("relations", [])
        if rels:
            n_examples_with_edges += 1
        for rel in rels:
            kind = rel.get("relation_type", "unknown")
            counts[kind] = counts.get(kind, 0) + 1
    return {
        "n_packages": n_packages,
        "n_examples_with_edges": n_examples_with_edges,
        "relation_type_counts": counts,
    }


def main() -> None:
    report: dict = {}
    for benchmark in ("fever", "hotpotqa"):
        results = load_results(benchmark)
        if not results:
            report[benchmark] = {"status": "no results found"}
            continue
        rbv = by_variant(results)
        summary = {v: variant_summary(rbv.get(v, [])) for v in VARIANTS}
        comparisons = []
        for a, b in COMPARISONS:
            for metric in METRICS:
                comparisons.append(paired(rbv, a, b, metric))
        report[benchmark] = {
            "n_total_examples": len({r["example_id"] for r in results}),
            "variant_summary": summary,
            "paired_comparisons": comparisons,
            "full_egrag_relation_composition": relation_composition(benchmark),
        }

    out_json = ROOT / "analysis.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_json}")

    # Markdown summary
    lines = ["# Dev-100 Bottleneck Investigation -- Results\n"]
    lines.append(f"Model: Qwen2.5-7B-Instruct. Real NLI (roberta-large-mnli, thresholds 0.4/0.7/0.8).")
    lines.append("Evidence budget 256 (64 reserved for output), max_new_tokens=256, seed=0, deterministic decoding.\n")
    for benchmark, data in report.items():
        if data.get("status"):
            lines.append(f"## {benchmark}: {data['status']}\n")
            continue
        lines.append(f"## {benchmark} (n={data['n_total_examples']})\n")
        lines.append("| Variant | n | failed | tf1_adj | citR_adj | ansAcc_adj | avg_claims | avg_edges | avg_selected |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for v in VARIANTS:
            s = data["variant_summary"][v]
            lines.append(
                f"| {v} | {s['n']} | {s['failed']} | {s['token_f1_adjusted']} | "
                f"{s['citation_recall_adjusted']} | {s['answer_accuracy_adjusted']} | "
                f"{s['avg_num_claims']} | {s['avg_num_graph_edges']} | {s['avg_num_selected']} |"
            )
        lines.append("\n### Paired bootstrap comparisons (95% CI, 2000 resamples, both-succeeded only)\n")
        lines.append("| A | B | metric | n_paired | mean_A | mean_B | delta | 95% CI | excludes 0 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for c in data["paired_comparisons"]:
            if c.get("n_paired", 0) == 0:
                lines.append(f"| {c['variant_a']} | {c['variant_b']} | {c['metric']} | 0 | - | - | - | - | - |")
                continue
            lines.append(
                f"| {c['variant_a']} | {c['variant_b']} | {c['metric']} | {c['n_paired']} | "
                f"{c['mean_a']} | {c['mean_b']} | {c['mean_delta']} | {c['ci_95']} | {c['ci_excludes_zero']} |"
            )
        rc = data["full_egrag_relation_composition"]
        lines.append(
            f"\n### full_egrag relation-type composition ({rc['n_packages']} packages, "
            f"{rc['n_examples_with_edges']} with >=1 edge)\n"
        )
        lines.append("```json")
        lines.append(json.dumps(rc["relation_type_counts"], indent=2))
        lines.append("```\n")

    out_md = ROOT / "REPORT.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
