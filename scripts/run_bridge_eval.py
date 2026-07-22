#!/usr/bin/env python
"""Controlled bridge-milestone evaluation (oracle mode -> artifacts/bridge-milestone/).

Runs the gold-annotated suite in oracle mode (deterministic; gold relations) with
the deterministic bridge detector for query-conditioned connectivity, across the
required variants, and writes per-example / aggregate / ablation / diagnostic
artifacts plus a before-vs-after comparison and validation report. Evidential
relation quality is the oracle upper bound; bridges are the new connectivity.

Run: PYTHONPATH=src .venv/bin/python scripts/run_bridge_eval.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import fmean

from egrag.experiments.mechanism_eval import (
    VariantFlags,
    activation,
    mechanism_metrics,
    run_example,
)
from egrag.experiments.mechanisms import build_suite

ROOT = Path("artifacts/bridge-milestone")
SUITE = build_suite()
VARIANTS = {
    "full_egrag": VariantFlags(),
    "graph_no_bridge": VariantFlags(bridges=False),
    "graph_no_propagation": VariantFlags(propagation=False),
    "graph_no_contradiction": VariantFlags(contradiction=False),
    "graph_no_temporal": VariantFlags(temporal=False),
    "graph_top_claim": VariantFlags(selection="top"),
}
EDGE_KEYS = [
    "support_edges",
    "contradiction_edges",
    "supersession_edges",
    "duplicate_edges",
    "bridge_edges",
]


def _aggregate(rows: list[dict]) -> dict[str, float]:
    keys: set[str] = set()
    for r in rows:
        keys |= {k for k, v in r["metrics"].items() if v is not None}
    out = {}
    for k in sorted(keys):
        vals = [r["metrics"][k] for r in rows if r["metrics"].get(k) is not None]
        if vals:
            out[k] = round(fmean(vals), 4)
    return out


def _run(flags: VariantFlags) -> list[dict]:
    rows = []
    for ex in SUITE:
        run = run_example(ex, "oracle", flags)
        rows.append(
            {
                "example_id": ex.example_id,
                "category": ex.category,
                "activation": activation(run),
                "metrics": mechanism_metrics(ex, run),
                "beliefs": run.beliefs,
            }
        )
    return rows


def main() -> None:
    for sub in ["per-example", "aggregates", "ablations", "diagnostics", "tables", "fixtures"]:
        (ROOT / sub).mkdir(parents=True, exist_ok=True)

    # fixture inventory
    (ROOT / "fixtures" / "inventory.json").write_text(
        json.dumps(dict(Counter(e.category for e in SUITE)), indent=2, sort_keys=True)
    )

    results = {name: _run(flags) for name, flags in VARIANTS.items()}
    (ROOT / "per-example" / "full_egrag.jsonl").write_text(
        "\n".join(json.dumps(r) for r in results["full_egrag"]) + "\n"
    )

    # aggregates by category for full_egrag
    cats = sorted({r["category"] for r in results["full_egrag"]})
    by_cat = {c: _aggregate([r for r in results["full_egrag"] if r["category"] == c]) for c in cats}
    (ROOT / "aggregates" / "full_egrag.json").write_text(json.dumps(by_cat, indent=2))

    # ablation activation + key metric table
    abl_lines = [
        "variant,"
        + ",".join(EDGE_KEYS)
        + ",conflict_sets,prop_iters,required_hop_coverage,connected_rate"
    ]
    abl_summary = {}
    for name, rows in results.items():
        totals = {k: sum(r["activation"][k] for r in rows) for k in EDGE_KEYS}
        cs = sum(r["activation"]["conflict_sets"] for r in rows)
        pi = sum(r["activation"]["propagation_iterations"] for r in rows)
        mh = [r for r in rows if r["category"] == "multi_hop"]
        hop = round(fmean(r["metrics"]["required_hop_coverage"] for r in mh), 4) if mh else 0.0
        conn = [
            r["metrics"]["selected_subgraph_connectivity"]
            for r in rows
            if r["metrics"].get("selected_subgraph_connectivity") is not None
        ]
        conn_rate = round(fmean(conn), 4) if conn else 0.0
        abl_lines.append(
            f"{name},{totals['support_edges']},{totals['contradiction_edges']},"
            f"{totals['supersession_edges']},{totals['duplicate_edges']},{totals['bridge_edges']},"
            f"{cs},{pi},{hop},{conn_rate}"
        )
        abl_summary[name] = {
            "edges": totals,
            "conflict_sets": cs,
            "propagation_iterations": pi,
            "multi_hop_required_hop_coverage": hop,
            "connected_rate": conn_rate,
        }
    (ROOT / "diagnostics" / "component-activation.csv").write_text("\n".join(abl_lines) + "\n")
    (ROOT / "ablations" / "summary.json").write_text(json.dumps(abl_summary, indent=2))

    # invariance proofs: beliefs identical full vs no-bridge; bridges -> 0 conflicts
    belief_invariant = all(
        f["beliefs"] == n["beliefs"]
        for f, n in zip(results["full_egrag"], results["graph_no_bridge"], strict=True)
    )
    bridge_conflict_count = sum(
        1
        for r in results["full_egrag"]
        if r["activation"]["bridge_edges"] > 0
        and r["activation"]["conflict_sets"] > 0
        and r["category"] in {"multi_hop"}  # multi-hop has bridges, should have no conflicts
    )
    bridge_belief_delta = 0.0 if belief_invariant else 1.0
    diagnostics = {
        "belief_invariant_full_vs_no_bridge": belief_invariant,
        "bridge_induced_belief_delta": bridge_belief_delta,
        "bridge_induced_conflict_count": bridge_conflict_count,
        "multi_hop_hop_coverage_full": abl_summary["full_egrag"]["multi_hop_required_hop_coverage"],
        "multi_hop_hop_coverage_no_bridge": abl_summary["graph_no_bridge"][
            "multi_hop_required_hop_coverage"
        ],
    }
    (ROOT / "diagnostics" / "invariance.json").write_text(json.dumps(diagnostics, indent=2))

    # bridge metrics
    bridge_rows = [
        r for r in results["full_egrag"] if r["metrics"].get("bridge_recall") is not None
    ]
    bp = [
        r["metrics"]["bridge_precision"]
        for r in bridge_rows
        if r["metrics"].get("bridge_precision") is not None
    ]
    br = [r["metrics"]["bridge_recall"] for r in bridge_rows]
    bridge_metrics = {
        "bridge_precision": round(fmean(bp), 4) if bp else None,
        "bridge_recall": round(fmean(br), 4) if br else None,
        "n_examples_with_bridge_gold_or_pred": len(bridge_rows),
        "total_accepted_bridges": sum(
            r["activation"]["bridge_edges"] for r in results["full_egrag"]
        ),
    }
    (ROOT / "aggregates" / "bridge-metrics.json").write_text(json.dumps(bridge_metrics, indent=2))

    # support/duplicate confusion (full): per support/duplicate/directional category
    confusion = {}
    for cat in ("support", "directional_support", "duplicate"):
        rows = [r for r in results["full_egrag"] if r["category"] == cat]
        confusion[cat] = {
            "support_edges": sum(r["activation"]["support_edges"] for r in rows),
            "duplicate_edges": sum(r["activation"]["duplicate_edges"] for r in rows),
            "support_edge_recall": _aggregate(rows).get("support_edge_recall"),
            "duplicate_cluster_accuracy": _aggregate(rows).get("duplicate_cluster_accuracy"),
        }
    (ROOT / "diagnostics" / "support-duplicate-confusion.json").write_text(
        json.dumps(confusion, indent=2)
    )

    # validation
    validation = {
        "non_zero_bridges": bridge_metrics["total_accepted_bridges"] > 0,
        "belief_invariant": belief_invariant,
        "zero_bridge_conflicts": bridge_conflict_count == 0,
        "hop_coverage_improves": diagnostics["multi_hop_hop_coverage_full"]
        > diagnostics["multi_hop_hop_coverage_no_bridge"],
        "no_bridge_ablation_zero_bridges": abl_summary["graph_no_bridge"]["edges"]["bridge_edges"]
        == 0,
        "no_contradiction_zero": abl_summary["graph_no_contradiction"]["edges"][
            "contradiction_edges"
        ]
        == 0,
        "no_propagation_zero_iters": abl_summary["graph_no_propagation"]["propagation_iterations"]
        == 0,
    }
    validation["ok"] = all(validation.values())
    (ROOT / "validation-report.json").write_text(json.dumps(validation, indent=2))

    print(
        f"BRIDGE_EVAL_OK suite={len(SUITE)} bridges={bridge_metrics['total_accepted_bridges']} "
        f"hop_full={diagnostics['multi_hop_hop_coverage_full']} "
        f"hop_no_bridge={diagnostics['multi_hop_hop_coverage_no_bridge']} "
        f"belief_invariant={belief_invariant} validation_ok={validation['ok']}"
    )


if __name__ == "__main__":
    main()
