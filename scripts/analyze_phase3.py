#!/usr/bin/env python
"""Analyze Phase 3 (cross-model generality) results and fold in the existing
Phase 1/2 Qwen2.5-7B-Instruct dev-100 numbers as the reference row. Reads only
already-written results.jsonl/package files under artifacts/dev100-bottleneck/
(Phase 1/2, read-only) and artifacts/dev100-bottleneck/PHASE3/ (Phase 3, this
script's own output tree). Nothing is modified or overwritten.

Run:
    uv run python scripts/analyze_phase3.py
"""

from __future__ import annotations

import json
from pathlib import Path

from egrag.experiments.stats import paired_comparison

ROOT = Path("artifacts/dev100-bottleneck")
PHASE3 = ROOT / "PHASE3"
METRICS = ("token_f1", "citation_recall", "answer_accuracy")

MODEL_KEYS_PHASE3 = ["qwen2.5-3b-instruct", "qwen3.5-9b"]
REFERENCE_MODEL_KEY = "qwen2.5-7b-instruct"  # Phase 1/2, not re-run here

CONDITIONS = [
    ("A_claim_only_rag", "claim_only_rag"),
    ("B_full_egrag_ungated", "full_egrag"),
    ("C_full_egrag_gated", "full_egrag"),
    ("D_graph_no_contradiction", "graph_no_contradiction"),
]

# Phase 3 comparisons (condition_dir_a, condition_dir_b) -- matches spec section 12.
PRIMARY_COMPARISONS = [
    ("B_full_egrag_ungated", "A_claim_only_rag"),
    ("C_full_egrag_gated", "A_claim_only_rag"),
    ("C_full_egrag_gated", "B_full_egrag_ungated"),
    ("D_graph_no_contradiction", "A_claim_only_rag"),
]


def load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
        "failure_rate": round(failed / n, 3) if n else None,
        **{f"{m}_adjusted": round(adj(m), 4) for m in METRICS},
        "avg_num_claims": round(avg("num_claims"), 3) if avg("num_claims") is not None else None,
        "avg_num_graph_edges": round(avg("num_graph_edges"), 3) if avg("num_graph_edges") is not None else None,
        "avg_num_selected": round(avg("num_selected"), 3) if avg("num_selected") is not None else None,
    }


def paired(a_rows: list[dict], b_rows: list[dict], a_name: str, b_name: str, metric: str) -> dict:
    a_idx = {r["example_id"]: r for r in a_rows}
    b_idx = {r["example_id"]: r for r in b_rows}
    common = sorted(set(a_idx) & set(b_idx))
    pairs = []
    for eid in common:
        ra, rb = a_idx[eid], b_idx[eid]
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


def relation_composition(pkg_dir: Path, variant_glob: str) -> dict:
    counts: dict[str, int] = {}
    n_packages = 0
    n_with_edges = 0
    if not pkg_dir.is_dir():
        return {"n_packages": 0, "relation_type_counts": {}}
    for f in sorted(pkg_dir.glob(variant_glob)):
        n_packages += 1
        d = json.loads(f.read_text())
        rels = d.get("relations", [])
        if rels:
            n_with_edges += 1
        for rel in rels:
            kind = rel.get("relation_type", "unknown")
            counts[kind] = counts.get(kind, 0) + 1
    total_edges = sum(counts.values())
    return {
        "n_packages": n_packages,
        "n_examples_with_edges": n_with_edges,
        "relation_type_counts": counts,
        "total_edges": total_edges,
        "contradiction_pct": round(100 * counts.get("contradiction", 0) / total_edges, 1) if total_edges else 0.0,
    }


def analyze_model(benchmark: str, model_key: str) -> dict | None:
    base = PHASE3 / benchmark / model_key
    condition_rows: dict[str, list[dict]] = {}
    for cond_dir, variant in CONDITIONS:
        rows = load(base / cond_dir / "results.jsonl")
        if not rows:
            return None
        condition_rows[cond_dir] = rows

    summaries = {cond: summary(rows) for cond, rows in condition_rows.items()}
    comparisons = []
    for a_dir, b_dir in PRIMARY_COMPARISONS:
        for metric in METRICS:
            comparisons.append(
                paired(condition_rows[a_dir], condition_rows[b_dir], a_dir, b_dir, metric)
            )

    relation_comp = {
        cond_dir: relation_composition(base / cond_dir / "packages", f"{variant}__*.json")
        for cond_dir, variant in CONDITIONS
        if variant in ("full_egrag", "graph_no_contradiction")
    }

    return {
        "n_total_examples": len({r["example_id"] for r in condition_rows["A_claim_only_rag"]}),
        "condition_summary": summaries,
        "paired_comparisons": comparisons,
        "relation_composition": relation_comp,
    }


def load_phase12_reference(benchmark: str) -> dict | None:
    """Pull the already-analyzed Phase 1 (ungated/claim_only/no_contra) and
    Phase 2 (gated) 7B numbers straight from their existing analysis files --
    does not recompute or re-touch Phase 1/2 result files."""
    p1_path = ROOT / "analysis.json"
    p2_path = ROOT / "phase2_analysis.json"
    if not p1_path.is_file() or not p2_path.is_file():
        return None
    p1 = json.loads(p1_path.read_text()).get(benchmark, {})
    p2 = json.loads(p2_path.read_text()).get(benchmark, {})
    if not p1 or not p2:
        return None
    return {
        "A_claim_only_rag": p1.get("variant_summary", {}).get("claim_only_rag"),
        "B_full_egrag_ungated": p1.get("variant_summary", {}).get("full_egrag"),
        "C_full_egrag_gated": p2.get("full_egrag_gated_phase2_summary"),
        "D_graph_no_contradiction": p1.get("variant_summary", {}).get("graph_no_contradiction"),
        "relation_composition_B": p1.get("full_egrag_relation_composition"),
        "relation_composition_C": p2.get("gated_phase2_relation_composition"),
    }


def main() -> None:
    report: dict = {}
    for benchmark in ("hotpotqa", "fever"):
        model_results = {}
        for model_key in MODEL_KEYS_PHASE3:
            result = analyze_model(benchmark, model_key)
            if result is not None:
                model_results[model_key] = result
        if not model_results:
            continue
        report[benchmark] = {
            "models": model_results,
            "reference_qwen2.5-7b-instruct_phase1_2": load_phase12_reference(benchmark),
        }

    (PHASE3 / "phase3_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {PHASE3 / 'phase3_analysis.json'}")

    lines = ["# Phase 3 -- Cross-Model Generality of the Contradiction-Gating Fix\n"]
    lines.append(
        "Same frozen dev-100 pipeline as Phase 1/2 (BM25 top_k=5, sentence-aware chunking 256/0, "
        "deterministic sentence-claim extraction, real NLI roberta-large-mnli thresholds "
        "0.4/0.7/0.8, evidence budget 256, max_new_tokens=256, seed=0, deterministic decoding). "
        "Only the generator model varies across A/B/C rows below; only "
        "`contradiction_requires_shared_subject` varies between conditions B and C.\n"
    )
    for benchmark, data in report.items():
        lines.append(f"## {benchmark}\n")
        ref = data.get("reference_qwen2.5-7b-instruct_phase1_2")
        model_order = (["qwen2.5-3b-instruct"] if "qwen2.5-3b-instruct" in data["models"] else []) + \
                      (["qwen2.5-7b-instruct"] if ref else []) + \
                      (["qwen3.5-9b"] if "qwen3.5-9b" in data["models"] else [])

        lines.append("### Model-by-model condition summary\n")
        lines.append("| Model | Condition | n | failed | tf1_adj | citR_adj | ansAcc_adj | avg_edges | avg_selected |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for model_key in model_order:
            if model_key == "qwen2.5-7b-instruct":
                for cond_dir, _ in CONDITIONS:
                    s = ref.get(cond_dir)
                    if not s:
                        continue
                    lines.append(
                        f"| {model_key} (Phase 1/2 ref) | {cond_dir} | {s['n']} | {s['failed']} | "
                        f"{s['token_f1_adjusted']} | {s['citation_recall_adjusted']} | "
                        f"{s['answer_accuracy_adjusted']} | {s.get('avg_num_graph_edges')} | "
                        f"{s.get('avg_num_selected')} |"
                    )
            else:
                m = data["models"][model_key]
                for cond_dir, _ in CONDITIONS:
                    s = m["condition_summary"][cond_dir]
                    lines.append(
                        f"| {model_key} | {cond_dir} | {s['n']} | {s['failed']} | "
                        f"{s['token_f1_adjusted']} | {s['citation_recall_adjusted']} | "
                        f"{s['answer_accuracy_adjusted']} | {s.get('avg_num_graph_edges')} | "
                        f"{s.get('avg_num_selected')} |"
                    )
        lines.append("")

        lines.append("### Paired bootstrap comparisons (95% CI, 2000 resamples, both-succeeded only)\n")
        for model_key in [k for k in model_order if k != "qwen2.5-7b-instruct"]:
            m = data["models"][model_key]
            lines.append(f"**{model_key}**\n")
            lines.append("| A vs B | metric | n_paired | mean_A | mean_B | delta | 95% CI | excludes 0 |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for c in m["paired_comparisons"]:
                if c.get("n_paired", 0) == 0:
                    lines.append(f"| {c['variant_a']} vs {c['variant_b']} | {c['metric']} | 0 | - | - | - | - | - |")
                    continue
                lines.append(
                    f"| {c['variant_a']} vs {c['variant_b']} | {c['metric']} | {c['n_paired']} | "
                    f"{c['mean_a']} | {c['mean_b']} | {c['mean_delta']} | {c['ci_95']} | {c['ci_excludes_zero']} |"
                )
            lines.append("")

        lines.append("### Relation-type composition (graph conditions B/C/D)\n")
        for model_key in [k for k in model_order if k != "qwen2.5-7b-instruct"]:
            m = data["models"][model_key]
            lines.append(f"**{model_key}**\n")
            for cond_dir, rc in m["relation_composition"].items():
                lines.append(
                    f"- {cond_dir}: {rc['n_packages']} pkgs, {rc['n_examples_with_edges']} w/ edges, "
                    f"{rc['total_edges']} total edges, {rc['contradiction_pct']}% contradiction -- "
                    f"`{json.dumps(rc['relation_type_counts'])}`"
                )
            lines.append("")
        if ref:
            lines.append("**qwen2.5-7b-instruct (Phase 1/2 reference)**\n")
            for key, label in (("relation_composition_B", "B_full_egrag_ungated"), ("relation_composition_C", "C_full_egrag_gated")):
                rc = ref.get(key)
                if rc:
                    lines.append(
                        f"- {label}: {rc['n_packages']} pkgs, {rc['n_examples_with_edges']} w/ edges -- "
                        f"`{json.dumps(rc['relation_type_counts'])}`"
                    )
            lines.append("")

    (PHASE3 / "PHASE3_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {PHASE3 / 'PHASE3_REPORT.md'}")


if __name__ == "__main__":
    main()
