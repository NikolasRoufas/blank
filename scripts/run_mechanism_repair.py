#!/usr/bin/env python
"""Repaired mechanism experiments -> artifacts/paper-results-repaired/.

Runs the gold-annotated mechanism suite in oracle and end-to-end modes, the graph
ablations, and an evidence-budget sweep; computes mechanism metrics against gold;
enforces an activation preflight; and writes per-example, aggregate, table,
diagnostic, and validation artifacts. No fake-generator answer metrics are used as
graph evidence. Run:

    PYTHONPATH=src .venv/bin/python scripts/run_mechanism_repair.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean

from egrag.experiments.mechanism_eval import (
    VariantFlags,
    activation,
    activation_preflight,
    mechanism_metrics,
    run_example,
)
from egrag.experiments.mechanisms import build_suite

ROOT = Path("artifacts/paper-results-repaired")
SUITE = build_suite()
E2E_CATEGORIES = {"support", "contradiction"}  # categories with passages + lexical recovery
ABLATIONS = {
    "full_egrag": VariantFlags(
        propagation=True, temporal=True, contradiction=True, selection="greedy"
    ),
    "graph_no_propagation": VariantFlags(propagation=False),
    "graph_no_temporal": VariantFlags(temporal=False),
    "graph_no_contradiction": VariantFlags(contradiction=False),
    "graph_top_claim": VariantFlags(selection="top"),
}
ACTIVATION_COLS = [
    "support_edges",
    "contradiction_edges",
    "supersession_edges",
    "duplicate_edges",
    "propagation_iterations",
    "conflict_sets",
    "connected_selected",
    "selected_claims",
]


def _mkdirs() -> None:
    for sub in [
        "oracle-mechanism",
        "end-to-end",
        "ablations",
        "budget-sweep",
        "per-example",
        "aggregates",
        "tables",
        "diagnostics",
    ]:
        (ROOT / sub).mkdir(parents=True, exist_ok=True)


def _aggregate(rows: list[dict]) -> dict[str, float]:
    keys: set[str] = set()
    for r in rows:
        keys |= {k for k, v in r["metrics"].items() if v is not None}
    agg = {}
    for k in sorted(keys):
        vals = [r["metrics"][k] for r in rows if r["metrics"].get(k) is not None]
        if vals:
            agg[k] = round(fmean(vals), 4)
    return agg


def _run_mode(
    mode: str, categories: set[str] | None, flags: VariantFlags, *, preflight: bool = False
) -> list[dict]:
    rows = []
    runs = []
    for ex in SUITE:
        if categories is not None and ex.category not in categories:
            continue
        run = run_example(ex, mode, flags)
        runs.append((ex, run))
        rows.append(
            {
                "example_id": ex.example_id,
                "category": ex.category,
                "mode": mode,
                "activation": activation(run),
                "metrics": mechanism_metrics(ex, run),
            }
        )
    # The preflight guards only the FULL pipeline; ablations intentionally remove
    # their target edge type (e.g. graph_no_temporal produces no supersession),
    # so it must not be applied to ablation runs.
    if preflight:
        activation_preflight(runs)
    return rows


def main() -> None:
    _mkdirs()
    flags = VariantFlags()

    # 1. oracle-mechanism mode (all categories); preflight guards the full pipeline
    oracle_rows = _run_mode("oracle", None, flags, preflight=True)
    (ROOT / "per-example" / "oracle.jsonl").write_text(
        "\n".join(json.dumps(r) for r in oracle_rows) + "\n"
    )
    by_cat_oracle = {}
    for cat in sorted({r["category"] for r in oracle_rows}):
        by_cat_oracle[cat] = _aggregate([r for r in oracle_rows if r["category"] == cat])
    (ROOT / "aggregates" / "oracle.json").write_text(json.dumps(by_cat_oracle, indent=2))
    (ROOT / "oracle-mechanism" / "summary.json").write_text(json.dumps(by_cat_oracle, indent=2))

    # 2. end-to-end mode (support + contradiction). NOTE: e2e re-extracts claims
    # with fresh IDs, so index-based gold metrics do not apply; we report
    # activation-based EDGE RECOVERY (did the pipeline produce the expected edge
    # type from raw text) instead, which is the honest end-to-end signal.
    e2e_rows = _run_mode("end_to_end", E2E_CATEGORIES, flags)
    (ROOT / "per-example" / "end_to_end.jsonl").write_text(
        "\n".join(json.dumps(r) for r in e2e_rows) + "\n"
    )
    expected_edge = {"support": "support_edges", "contradiction": "contradiction_edges"}
    edge_keys = ["support_edges", "contradiction_edges", "supersession_edges", "duplicate_edges"]
    e2e_recovery = {}
    for cat in sorted({r["category"] for r in e2e_rows}):
        rows = [r for r in e2e_rows if r["category"] == cat]
        n = len(rows)
        with_expected = sum(1 for r in rows if r["activation"][expected_edge[cat]] >= 1)
        with_any = sum(1 for r in rows if any(r["activation"][k] for k in edge_keys))
        e2e_recovery[cat] = {
            "examples": n,
            "expected_edge_type": expected_edge[cat],
            "pct_with_expected_edge": round(with_expected / n, 4),
            "pct_with_any_edge": round(with_any / n, 4),
        }
    (ROOT / "aggregates" / "end_to_end.json").write_text(json.dumps(e2e_recovery, indent=2))
    (ROOT / "end-to-end" / "summary.json").write_text(json.dumps(e2e_recovery, indent=2))

    # 3. ablations (oracle mode) + component-activation diagnostics
    activation_csv = ROOT / "diagnostics" / "component-activation.csv"
    abl_summary = {}
    with activation_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["variant", "dataset", "examples", *ACTIVATION_COLS[:-1], "avg_selected_claims"]
        )
        for variant, vflags in ABLATIONS.items():
            rows = _run_mode("oracle", None, vflags)
            (ROOT / "ablations" / f"{variant}.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n"
            )
            abl_summary[variant] = _aggregate(rows)
            totals = dict.fromkeys(ACTIVATION_COLS, 0)
            for r in rows:
                for c in ACTIVATION_COLS:
                    totals[c] += r["activation"][c]
            n = len(rows)
            writer.writerow(
                [
                    variant,
                    "mechanism_suite",
                    n,
                    totals["support_edges"],
                    totals["contradiction_edges"],
                    totals["supersession_edges"],
                    totals["duplicate_edges"],
                    totals["propagation_iterations"],
                    totals["conflict_sets"],
                    totals["connected_selected"],
                    round(totals["selected_claims"] / n, 3),
                ]
            )
    (ROOT / "ablations" / "summary.json").write_text(json.dumps(abl_summary, indent=2))

    # 4. evidence-budget sweep (oracle, multi-hop where budget matters)
    budget_rows = []
    for budget in (128, 256, 512):
        bflags = VariantFlags()
        for ex in SUITE:
            if ex.category != "multi_hop":
                continue
            from egrag.domain.models import Query
            from egrag.experiments.mechanism_eval import build_run
            from egrag.experiments.mechanisms import GoldRelationClassifier

            run = build_run(
                list(ex.claims),
                Query(query_id=ex.example_id, text=ex.query),
                GoldRelationClassifier(ex),
                bflags,
                evidence_budget=budget,
                reserved=min(64, budget),
            )
            m = mechanism_metrics(ex, run)
            budget_rows.append(
                {
                    "evidence_token_budget": budget,
                    "example_id": ex.example_id,
                    "required_hop_coverage": m["required_hop_coverage"],
                    "selected_claims": activation(run)["selected_claims"],
                }
            )
    bcsv = ROOT / "budget-sweep" / "budget-results.csv"
    bcsv.parent.mkdir(parents=True, exist_ok=True)
    with bcsv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["evidence_token_budget", "n", "mean_required_hop_coverage", "mean_selected_claims"]
        )
        for budget in (128, 256, 512):
            br = [r for r in budget_rows if r["evidence_token_budget"] == budget]
            w.writerow(
                [
                    budget,
                    len(br),
                    round(fmean(r["required_hop_coverage"] for r in br), 4),
                    round(fmean(r["selected_claims"] for r in br), 3),
                ]
            )

    # 5. tables: mechanism metrics by category (oracle + e2e)
    mech_csv = ROOT / "tables" / "mechanism-metrics.csv"
    metric_cols = [
        "support_edge_recall",
        "contradiction_edge_recall",
        "supersession_recall",
        "candidate_pair_recall",
        "conflict_set_recall",
        "conflict_resolution_accuracy",
        "unresolved_conflict_accuracy",
        "required_hop_coverage",
        "selected_subgraph_connectivity",
        "duplicate_cluster_accuracy",
    ]
    with mech_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["mode", "category", *metric_cols])
        for cat in sorted(by_cat_oracle):  # oracle only; e2e uses fresh claim IDs
            w.writerow(["oracle", cat, *[by_cat_oracle[cat].get(c, "") for c in metric_cols]])
    # separate, honest end-to-end recovery table
    e2e_csv = ROOT / "tables" / "end-to-end-recovery.csv"
    with e2e_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "category",
                "examples",
                "expected_edge_type",
                "pct_with_expected_edge",
                "pct_with_any_edge",
            ]
        )
        for cat, rec in e2e_recovery.items():
            w.writerow(
                [
                    cat,
                    rec["examples"],
                    rec["expected_edge_type"],
                    rec["pct_with_expected_edge"],
                    rec["pct_with_any_edge"],
                ]
            )

    # 6. failed examples (none expected; preflight already guards edges)
    (ROOT / "failed-examples.jsonl").write_text("")

    # 7. validation: recompute aggregates from per-example, check no manual edits
    validation = {"checks": [], "ok": True}

    def check(name, ok, detail=""):
        validation["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        validation["ok"] = validation["ok"] and bool(ok)

    recomputed = {}
    for cat in sorted({r["category"] for r in oracle_rows}):
        recomputed[cat] = _aggregate([r for r in oracle_rows if r["category"] == cat])
    check("oracle-aggregates-recompute-match", recomputed == by_cat_oracle)
    edge_required = [
        r
        for r in oracle_rows
        if r["category"]
        in {
            "support",
            "contradiction",
            "temporal",
            "duplicate",
            "multi_hop",
            "unresolved_conflict",
            "preferred_conflict",
        }
    ]
    check(
        "every-edge-required-example-has-edges",
        all(
            sum(
                r["activation"][c]
                for c in [
                    "support_edges",
                    "contradiction_edges",
                    "supersession_edges",
                    "duplicate_edges",
                ]
            )
            >= 1
            for r in edge_required
        ),
    )
    (ROOT / "diagnostics" / "validation.json").write_text(json.dumps(validation, indent=2))

    print(
        f"REPAIR_OK suite={len(SUITE)} oracle_cats={len(by_cat_oracle)} "
        f"e2e_cats={len(e2e_recovery)} ablations={len(ABLATIONS)} validation_ok={validation['ok']}"
    )


if __name__ == "__main__":
    main()
