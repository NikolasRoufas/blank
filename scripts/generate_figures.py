#!/usr/bin/env python
"""Publication-quality figures from artifacts/benchmark-matrix/analysis/*.json
(written by scripts/analyze_benchmark_results.py). Vector output (PDF + SVG).

Every figure reads real computed data; a group with no completed runs is
skipped and reported as skipped, not plotted with placeholder values.

Run:
    uv run python scripts/generate_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANALYSIS = Path("artifacts/benchmark-matrix/analysis")
FIGDIR = ANALYSIS / "figures"

MODEL_ORDER = ["qwen2.5-3b-instruct", "qwen2.5-7b-instruct", "qwen3.5-9b"]
MODEL_LABELS = {
    "qwen2.5-3b-instruct": "Qwen2.5-3B",
    "qwen2.5-7b-instruct": "Qwen2.5-7B",
    "qwen3.5-9b": "Qwen3.5-9B",
}
COLORS = {
    "qwen2.5-3b-instruct": "#4C72B0",
    "qwen2.5-7b-instruct": "#55A868",
    "qwen3.5-9b": "#C44E52",
}


def _save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(FIGDIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGDIR / name}.{{pdf,svg}}")


def _complete_rows(summary: dict, group: str) -> dict:
    rows = summary.get(group, {})
    return {m: r for m, r in rows.items() if r.get("status") == "complete"}


def fig_failure_rate_by_group(summary: dict) -> None:
    groups = [g for g in summary if _complete_rows(summary, g)]
    if not groups:
        print("skip fig_failure_rate_by_group: no completed groups")
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(groups)), 4))
    width = 0.25
    x = range(len(groups))
    for i, model in enumerate(MODEL_ORDER):
        vals = [
            summary[g][model]["failure_rate"] * 100
            if model in summary[g] and summary[g][model].get("status") == "complete"
            else None
            for g in groups
        ]
        xs = [xi + (i - 1) * width for xi in x]
        heights = [v if v is not None else 0 for v in vals]
        ax.bar(xs, heights, width=width, label=MODEL_LABELS[model], color=COLORS[model])
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel("Failure rate (%)")
    ax.set_title("Citation-validation failure rate by run")
    ax.legend()
    _save(fig, "failure_rate_by_group")


def fig_token_f1_scaling(summary: dict) -> None:
    groups = [g for g in summary if _complete_rows(summary, g)]
    if not groups:
        print("skip fig_token_f1_scaling: no completed groups")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for g in groups:
        xs, ys = [], []
        for i, model in enumerate(MODEL_ORDER):
            row = summary[g].get(model)
            if row and row.get("status") == "complete" and row.get("token_f1_mean") is not None:
                xs.append(i)
                ys.append(row["token_f1_mean"])
        if xs:
            ax.plot(xs, ys, marker="o", label=g)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_ylabel("Token F1 (successful generations only)")
    ax.set_title("Token F1 vs. generator scale")
    ax.legend(fontsize=7)
    _save(fig, "token_f1_scaling")


def fig_runtime_by_model(summary: dict) -> None:
    groups = [g for g in summary if _complete_rows(summary, g)]
    if not groups:
        print("skip fig_runtime_by_model: no completed groups")
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(groups)), 4))
    width = 0.25
    x = range(len(groups))
    for i, model in enumerate(MODEL_ORDER):
        vals = [
            summary[g][model]["latency_ms_mean"]
            if model in summary[g] and summary[g][model].get("status") == "complete"
            else 0
            for g in groups
        ]
        xs = [xi + (i - 1) * width for xi in x]
        ax.bar(xs, vals, width=width, label=MODEL_LABELS[model], color=COLORS[model])
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel("Mean latency per call (ms)")
    ax.set_title("Runtime by run")
    ax.legend()
    _save(fig, "runtime_by_group")


def fig_failure_taxonomy(taxonomy: dict) -> None:
    any_data = False
    fig, axes = plt.subplots(1, max(1, len(taxonomy)), figsize=(5 * max(1, len(taxonomy)), 4))
    if len(taxonomy) <= 1:
        axes = [axes]
    for ax, (group, data) in zip(axes, taxonomy.items(), strict=False):
        cat = data.get("overall_category", {})
        if not cat:
            ax.set_visible(False)
            continue
        any_data = True
        labels, values = zip(*sorted(cat.items(), key=lambda kv: -kv[1]), strict=False)
        ax.barh(labels, values, color="#C44E52")
        ax.set_title(group)
        ax.set_xlabel("count")
    if not any_data:
        print("skip fig_failure_taxonomy: no failure data yet")
        plt.close(fig)
        return
    fig.suptitle("Failure taxonomy (automatic categorization)")
    _save(fig, "failure_taxonomy")


def fig_citation_id_patterns(taxonomy: dict) -> None:
    overall_total: dict[str, int] = {}
    for data in taxonomy.values():
        for k, v in data.get("overall_citation_id_pattern", {}).items():
            overall_total[k] = overall_total.get(k, 0) + v
    if not overall_total:
        print("skip fig_citation_id_patterns: no citation-failure data yet")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    labels, values = zip(*sorted(overall_total.items(), key=lambda kv: -kv[1]), strict=False)
    ax.pie(values, labels=labels, autopct="%1.0f%%", textprops={"fontsize": 8})
    ax.set_title("Malformed citation-ID patterns (all runs combined)")
    _save(fig, "citation_id_patterns")


def main() -> None:
    summary_path = ANALYSIS / "metrics_summary.json"
    taxonomy_path = ANALYSIS / "failure_taxonomy.json"
    if not summary_path.is_file():
        raise SystemExit("run scripts/analyze_benchmark_results.py first")
    summary = json.loads(summary_path.read_text())
    taxonomy = json.loads(taxonomy_path.read_text()) if taxonomy_path.is_file() else {}

    fig_failure_rate_by_group(summary)
    fig_token_f1_scaling(summary)
    fig_runtime_by_model(summary)
    fig_failure_taxonomy(taxonomy)
    fig_citation_id_patterns(taxonomy)


if __name__ == "__main__":
    main()
