#!/usr/bin/env python
"""Analyze Phase 2 (gated contradiction) against the Phase 1 dev-100 baseline.
Paired by example_id against the already-written Phase 1 results.jsonl files
(read-only; nothing in Phase 1's output directories is modified). Writes
artifacts/dev100-bottleneck/phase2_analysis.json and PHASE2_REPORT.md.

Run:
    uv run python scripts/analyze_phase2.py
"""

from __future__ import annotations

import json
from pathlib import Path

from egrag.experiments.stats import paired_comparison

ROOT = Path("artifacts/dev100-bottleneck")
MODEL_KEY = "qwen2.5-7b-instruct"
GATED_KEY = "qwen2.5-7b-instruct_gated"
METRICS = ("token_f1", "citation_recall", "answer_accuracy")


def load(benchmark: str, model_key: str) -> list[dict]:
    path = ROOT / benchmark / model_key / "results.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def index_by_id(rows: list[dict], variant: str | None = None) -> dict[str, dict]:
    return {r["example_id"]: r for r in rows if variant is None or r["variant"] == variant}


def summary(rows: list[dict]) -> dict:
    n = len(rows)
    failed = sum(1 for r in rows if r["failed"])

    def adj(metric: str) -> float:
        return sum(r["metrics"].get(metric, 0.0) for r in rows if not r["failed"]) / n if n else float("nan")

    def avg(key: str) -> float | None:
        vals = [r["counts"].get(key) for r in rows if not r["failed"] and r["counts"].get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "n": n,
        "failed": failed,
        **{f"{m}_adjusted": round(adj(m), 4) for m in METRICS},
        "avg_num_claims": round(avg("num_claims"), 3) if avg("num_claims") is not None else None,
        "avg_num_graph_edges": round(avg("num_graph_edges"), 3) if avg("num_graph_edges") is not None else None,
        "avg_num_selected": round(avg("num_selected"), 3) if avg("num_selected") is not None else None,
    }


def paired(a_rows: dict[str, dict], b_rows: dict[str, dict], a_name: str, b_name: str, metric: str) -> dict:
    common = sorted(set(a_rows) & set(b_rows))
    pairs = []
    for eid in common:
        ra, rb = a_rows[eid], b_rows[eid]
        if ra["failed"] or rb["failed"]:
            continue
        va, vb = ra["metrics"].get(metric), rb["metrics"].get(metric)
        if va is None or vb is None:
            continue
        pairs.append((va, vb))
    if not pairs:
        return {"metric": metric, "variant_a": a_name, "variant_b": b_name, "n_paired": 0}
    r = paired_comparison(metric, a_name, b_name, pairs, samples=2000, seed=0)
    return {
        "metric": r.metric, "variant_a": r.variant_a, "variant_b": r.variant_b,
        "n_paired": r.n_paired, "mean_a": round(r.mean_a, 4), "mean_b": round(r.mean_b, 4),
        "mean_delta": round(r.mean_delta, 4), "ci_95": [round(r.ci_low, 4), round(r.ci_high, 4)],
        "ci_excludes_zero": r.ci_low > 0 or r.ci_high < 0,
    }


def relation_composition(benchmark: str, model_key: str) -> dict:
    pkg_dir = ROOT / benchmark / model_key / "packages"
    counts: dict[str, int] = {}
    n_packages = 0
    n_with_edges = 0
    if not pkg_dir.is_dir():
        return {"n_packages": 0, "relation_type_counts": {}}
    for f in sorted(pkg_dir.glob("full_egrag__*.json")):
        n_packages += 1
        d = json.loads(f.read_text())
        rels = d.get("relations", [])
        if rels:
            n_with_edges += 1
        for rel in rels:
            kind = rel.get("relation_type", "unknown")
            counts[kind] = counts.get(kind, 0) + 1
    return {"n_packages": n_packages, "n_examples_with_edges": n_with_edges, "relation_type_counts": counts}


def main() -> None:
    report: dict = {}
    for benchmark in ("hotpotqa", "fever"):
        phase1 = load(benchmark, MODEL_KEY)
        phase2 = load(benchmark, GATED_KEY)
        if not phase2:
            report[benchmark] = {"status": "phase 2 not run yet"}
            continue
        full_egrag_p1 = [r for r in phase1 if r["variant"] == "full_egrag"]
        claim_only_p1 = [r for r in phase1 if r["variant"] == "claim_only_rag"]
        no_contra_p1 = [r for r in phase1 if r["variant"] == "graph_no_contradiction"]
        gated_p2 = phase2  # single variant: full_egrag with the gate on

        gated_idx = index_by_id(gated_p2)
        comparisons = []
        for name, rows in (
            ("full_egrag_phase1", full_egrag_p1),
            ("claim_only_rag_phase1", claim_only_p1),
            ("graph_no_contradiction_phase1", no_contra_p1),
        ):
            idx = index_by_id(rows)
            for metric in METRICS:
                comparisons.append(paired(gated_idx, idx, "full_egrag_gated_phase2", name, metric))

        report[benchmark] = {
            "n_total_examples": len({r["example_id"] for r in phase2}),
            "full_egrag_gated_phase2_summary": summary(gated_p2),
            "full_egrag_phase1_summary": summary(full_egrag_p1),
            "claim_only_rag_phase1_summary": summary(claim_only_p1),
            "graph_no_contradiction_phase1_summary": summary(no_contra_p1),
            "paired_comparisons": comparisons,
            "gated_phase2_relation_composition": relation_composition(benchmark, GATED_KEY),
            "full_egrag_phase1_relation_composition": relation_composition(benchmark, MODEL_KEY),
        }

    (ROOT / "phase2_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {ROOT / 'phase2_analysis.json'}")

    lines = ["# Phase 2 -- Gated Contradiction Edges vs Phase 1 Baseline\n"]
    lines.append(
        "Single-variable change from Phase 1's full_egrag: `contradiction_requires_shared_subject=True` "
        "in ClassificationConfig (graph/types.py). Everything else -- retrieval, chunking, extraction, "
        "NLI model/thresholds, propagation, temporal edges, selection strategy, generator/model/decoding, "
        "evidence budget, seed, dataset -- is identical to Phase 1.\n"
    )
    for benchmark, data in report.items():
        if data.get("status"):
            lines.append(f"## {benchmark}: {data['status']}\n")
            continue
        lines.append(f"## {benchmark} (n={data['n_total_examples']})\n")
        lines.append("| Condition | n | failed | tf1_adj | citR_adj | ansAcc_adj | avg_edges | avg_selected |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for key, label in (
            ("full_egrag_gated_phase2_summary", "full_egrag + gated contradiction (Phase 2)"),
            ("full_egrag_phase1_summary", "full_egrag (Phase 1, ungated)"),
            ("claim_only_rag_phase1_summary", "claim_only_rag (no-graph reference)"),
            ("graph_no_contradiction_phase1_summary", "graph_no_contradiction (Phase 1 ablation)"),
        ):
            s = data[key]
            lines.append(
                f"| {label} | {s['n']} | {s['failed']} | {s['token_f1_adjusted']} | "
                f"{s['citation_recall_adjusted']} | {s['answer_accuracy_adjusted']} | "
                f"{s['avg_num_graph_edges']} | {s['avg_num_selected']} |"
            )
        lines.append("\n### Paired bootstrap comparisons (full_egrag_gated_phase2 vs ..., 95% CI, both-succeeded only)\n")
        lines.append("| vs | metric | n_paired | mean_gated | mean_other | delta | 95% CI | excludes 0 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for c in data["paired_comparisons"]:
            if c.get("n_paired", 0) == 0:
                lines.append(f"| {c['variant_b']} | {c['metric']} | 0 | - | - | - | - | - |")
                continue
            lines.append(
                f"| {c['variant_b']} | {c['metric']} | {c['n_paired']} | {c['mean_a']} | {c['mean_b']} | "
                f"{c['mean_delta']} | {c['ci_95']} | {c['ci_excludes_zero']} |"
            )
        rc2 = data["gated_phase2_relation_composition"]
        rc1 = data["full_egrag_phase1_relation_composition"]
        lines.append(f"\n### Relation-type composition: Phase 1 (ungated) vs Phase 2 (gated)\n")
        lines.append(f"Phase 1 ({rc1['n_packages']} pkgs, {rc1['n_examples_with_edges']} w/ edges): "
                      f"`{json.dumps(rc1['relation_type_counts'])}`\n")
        lines.append(f"Phase 2 ({rc2['n_packages']} pkgs, {rc2['n_examples_with_edges']} w/ edges): "
                      f"`{json.dumps(rc2['relation_type_counts'])}`\n")

    (ROOT / "PHASE2_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {ROOT / 'PHASE2_REPORT.md'}")


if __name__ == "__main__":
    main()
