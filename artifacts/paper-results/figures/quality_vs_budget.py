#!/usr/bin/env python
"""Render the quality_vs_budget figure from robustness-results.csv. Requires matplotlib (not bundled)."""
import csv, sys
from pathlib import Path
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib not installed; install it to render this figure")
src = Path(__file__).resolve().parents[1] / "robustness-results.csv"
rows = list(csv.DictReader(src.open()))
meta = {"source": "robustness-results.csv", "x": "evidence_token_budget", "y": ["token_f1", "citation_recall", "evidence_recall"], "systems": ["passage_rag", "claim_only_rag", "full_egrag"]}
fig, ax = plt.subplots()
for system in meta["systems"]:
    srows = [r for r in rows if r.get("system") == system]
    for y in meta["y"]:
        xs = [r[meta["x"]] for r in srows if r.get(y) not in (None, "")]
        ys = [float(r[y]) for r in srows if r.get(y) not in (None, "")]
        if ys:
            ax.plot(xs, ys, marker="o", label=f"{system}:{y}")
ax.set_xlabel(meta["x"]); ax.set_ylabel(",".join(meta["y"])); ax.legend(fontsize=6)
out = Path(__file__).with_suffix("")
fig.savefig(str(out) + ".pdf"); fig.savefig(str(out) + ".png", dpi=150)
print("wrote", out)
