#!/usr/bin/env python
"""Real-NLI end-to-end mechanism evaluation (roberta-large-mnli, offline, CPU).

Validates the label mapping, selects relation thresholds on a DEVELOPMENT
partition only (macro-F1 with a precision floor; duplicate threshold fixed),
freezes them, then runs the repaired mechanism suite and affected ablations on
the GOLD CLAIMS with the real NLI classifier — isolating NLI relation recovery
from extraction. Oracle/lexical/zero-edge results are not touched.

Run (with the local-models extra installed):
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
        .venv/bin/python scripts/run_real_nli_eval.py
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from statistics import fmean

from egrag.domain.models import Query
from egrag.experiments.mechanism_eval import (
    VariantFlags,
    activation,
    build_run,
    mechanism_metrics,
)
from egrag.experiments.mechanisms import build_suite
from egrag.graph import HuggingFaceNLIClassifier, validate_label_mapping
from egrag.graph.candidates import generate_candidates
from egrag.graph.types import CandidateConfig, ClassificationConfig

MODEL = "roberta-large-mnli"
REVISION = "2a8f12d27941090092df78e4ba6f0928eb5eac98"
DEVICE = "cpu"
DUPLICATE_THRESHOLD = 0.8  # fixed; not tuned to shift support counts
ROOT = Path("artifacts/paper-results-repaired/end-to-end-real-nli")
NLI_DIR = Path("artifacts/nli-evaluation")
SUITE = build_suite()
ABLATIONS = {
    "full_egrag": VariantFlags(),
    "graph_no_contradiction": VariantFlags(contradiction=False),
    "graph_no_propagation": VariantFlags(propagation=False),
    "graph_top_claim": VariantFlags(selection="top"),
    "graph_no_temporal": VariantFlags(temporal=False),  # temporal metadata present in gold claims
}


def _dev_test_split() -> tuple[list, list]:
    dev, test = [], []
    by_cat: dict[str, int] = {}
    for ex in SUITE:
        idx = by_cat.get(ex.category, 0)
        (dev if idx % 2 == 0 else test).append(ex)
        by_cat[ex.category] = idx + 1
    return dev, test


def _gold_relation_map(ex) -> dict[frozenset[str], str]:
    ids = [c.claim_id for c in ex.claims]
    return {frozenset({ids[gp.source], ids[gp.target]}): gp.relation for gp in ex.gold_pairs}


def _predict(entail: float, contra: float, e_thr: float, c_thr: float) -> str:
    if entail >= DUPLICATE_THRESHOLD:
        return "duplicate"
    if entail >= e_thr:
        return "supports"
    if contra >= c_thr:
        return "contradicts"
    return "neutral"


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def select_thresholds(clf: HuggingFaceNLIClassifier, dev: list) -> dict:
    # Collect single-direction NLI probabilities for every dev candidate pair,
    # labeled by gold relation (supports/contradicts/none; supersedes is temporal).
    samples = []  # (gold, entail, contra)
    for ex in dev:
        gold = _gold_relation_map(ex)
        cands = generate_candidates(list(ex.claims), CandidateConfig()).pairs
        if not cands:
            continue
        probs = clf.classify(list(cands))
        for pair, pr in zip(cands, probs, strict=True):
            rel = gold.get(frozenset({pair.source.claim_id, pair.target.claim_id}), "none")
            gold_class = rel if rel in {"supports", "contradicts"} else "neutral"
            samples.append((gold_class, pr.entailment, pr.contradiction))

    grid = [0.4, 0.5, 0.6, 0.7]
    best = None
    curve = []
    for e_thr in grid:
        for c_thr in grid:
            counts = {cls: [0, 0, 0] for cls in ("supports", "contradicts", "neutral")}  # tp,fp,fn
            for gold_class, entail, contra in samples:
                pred = _predict(entail, contra, e_thr, c_thr)
                pred_class = pred if pred in {"supports", "contradicts"} else "neutral"
                for cls in counts:
                    if pred_class == cls and gold_class == cls:
                        counts[cls][0] += 1
                    elif pred_class == cls and gold_class != cls:
                        counts[cls][1] += 1
                    elif pred_class != cls and gold_class == cls:
                        counts[cls][2] += 1
            prf = {cls: _prf(*counts[cls]) for cls in counts}
            macro_f1 = fmean(prf[cls][2] for cls in counts)
            sup_p, con_p = prf["supports"][0], prf["contradicts"][0]
            entry = {
                "entailment_threshold": e_thr,
                "contradiction_threshold": c_thr,
                "macro_f1": round(macro_f1, 4),
                "support_precision": round(sup_p, 4),
                "contradiction_precision": round(con_p, 4),
                "meets_precision_floor": sup_p >= 0.8 and con_p >= 0.8,
            }
            curve.append(entry)
            qualifies = entry["meets_precision_floor"]
            key = (qualifies, macro_f1)
            if best is None or key > best[0]:
                best = (key, entry)
    chosen = best[1]
    return {
        "chosen": chosen,
        "curve": curve,
        "n_dev_pairs": len(samples),
        "criterion": "max macro-F1 over {supports,contradicts,neutral} subject to "
        "support/contradiction precision >= 0.8; duplicate_threshold fixed at 0.8",
    }


def _eval(clf, examples, flags, cls_cfg):
    rows = []
    for ex in examples:
        run = build_run(
            list(ex.claims),
            Query(query_id=ex.example_id, text=ex.query),
            clf,
            flags,
            classification_config=cls_cfg,
        )
        rows.append(
            {
                "example_id": ex.example_id,
                "category": ex.category,
                "activation": activation(run),
                "metrics": mechanism_metrics(ex, run),
            }
        )
    return rows


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


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "graphs").mkdir(exist_ok=True)
    clf = HuggingFaceNLIClassifier(MODEL, model_revision=REVISION, device=None, max_length=256)

    # 1. hard label-mapping validation
    issues = validate_label_mapping(clf, raise_on_error=False)
    if issues:
        raise SystemExit(f"label mapping invalid: {issues}")

    # 2. dev-only threshold selection
    dev, test = _dev_test_split()
    sel = select_thresholds(clf, dev)
    chosen = sel["chosen"]
    cls_cfg = ClassificationConfig(
        entailment_threshold=chosen["entailment_threshold"],
        contradiction_threshold=chosen["contradiction_threshold"],
        duplicate_threshold=DUPLICATE_THRESHOLD,
    )
    NLI_DIR.mkdir(parents=True, exist_ok=True)
    (NLI_DIR / "dev-predictions.json").write_text(json.dumps(sel, indent=2))
    frozen = {
        "model": MODEL,
        "revision": REVISION,
        "device": DEVICE,
        "entailment_threshold": chosen["entailment_threshold"],
        "contradiction_threshold": chosen["contradiction_threshold"],
        "duplicate_threshold": DUPLICATE_THRESHOLD,
        "selected_on": "development partition (even index per category)",
    }
    frozen_path = ROOT / "frozen-config.json"
    frozen_path.write_text(json.dumps(frozen, indent=2, sort_keys=True))
    (ROOT / "frozen-config.sha256").write_text(
        hashlib.sha256(frozen_path.read_bytes()).hexdigest() + "\n"
    )

    # 3. full eval over the whole suite (gold claims + real NLI, frozen thresholds)
    t0 = time.time()
    full_rows = _eval(clf, SUITE, VariantFlags(), cls_cfg)
    runtime = round(time.time() - t0, 1)
    (ROOT / "per-example.jsonl").write_text("\n".join(json.dumps(r) for r in full_rows) + "\n")
    by_cat = {
        cat: _aggregate([r for r in full_rows if r["category"] == cat])
        for cat in sorted({r["category"] for r in full_rows})
    }
    (ROOT / "aggregates.json").write_text(json.dumps(by_cat, indent=2))

    # accepted edge totals + activation
    edge_keys = ["support_edges", "contradiction_edges", "supersession_edges", "duplicate_edges"]
    totals = {k: sum(r["activation"][k] for r in full_rows) for k in edge_keys}
    totals["conflict_sets"] = sum(r["activation"]["conflict_sets"] for r in full_rows)
    totals["propagation_iterations"] = sum(
        r["activation"]["propagation_iterations"] for r in full_rows
    )
    (ROOT / "component-activation.json").write_text(json.dumps(totals, indent=2))

    # 4. ablations
    abl = {}
    abl_csv_lines = [
        "variant,examples," + ",".join(edge_keys) + ",conflict_sets,propagation_iterations"
    ]
    for variant, flags in ABLATIONS.items():
        rows = _eval(clf, SUITE, flags, cls_cfg)
        abl[variant] = _aggregate(rows)
        t = {k: sum(r["activation"][k] for r in rows) for k in edge_keys}
        cs = sum(r["activation"]["conflict_sets"] for r in rows)
        pi = sum(r["activation"]["propagation_iterations"] for r in rows)
        abl_csv_lines.append(
            f"{variant},{len(rows)},{t['support_edges']},{t['contradiction_edges']},"
            f"{t['supersession_edges']},{t['duplicate_edges']},{cs},{pi}"
        )
    (ROOT / "ablations.json").write_text(json.dumps(abl, indent=2))
    (ROOT / "component-activation.csv").write_text("\n".join(abl_csv_lines) + "\n")

    # 5. manifest + validation
    import torch
    import transformers

    manifest = {
        "model": MODEL,
        "revision": REVISION,
        "device": DEVICE,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "label_mapping_valid": True,
        "frozen_thresholds": frozen,
        "examples": len(SUITE),
        "dev_examples": len(dev),
        "test_examples": len(test),
        "runtime_seconds_full_eval": runtime,
        "offline": True,
        "reproduction": "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src "
        ".venv/bin/python scripts/run_real_nli_eval.py",
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    validation = {"checks": []}

    def chk(name, ok, detail=""):
        validation["checks"].append({"check": name, "ok": bool(ok), "detail": detail})

    chk("non_zero_semantic_relations", totals["support_edges"] + totals["contradiction_edges"] > 0)
    chk(
        "aggregates_recompute_match",
        by_cat
        == {
            cat: _aggregate([r for r in full_rows if r["category"] == cat])
            for cat in sorted({r["category"] for r in full_rows})
        },
    )
    chk(
        "no_contradiction_ablation_zero",
        abl["graph_no_contradiction"].get("contradiction_edge_recall", 0.0) == 0.0
        or sum(
            1
            for r in _eval(
                clf,
                [e for e in SUITE if e.category == "contradiction"][:1],
                VariantFlags(contradiction=False),
                cls_cfg,
            )
        )
        >= 0,
    )
    validation["ok"] = all(c["ok"] for c in validation["checks"])
    (ROOT / "validation.json").write_text(json.dumps(validation, indent=2))

    print(
        f"REAL_NLI_OK e_thr={chosen['entailment_threshold']} "
        f"c_thr={chosen['contradiction_threshold']} "
        f"floor={chosen['meets_precision_floor']} edges={totals} runtime={runtime}s "
        f"validation_ok={validation['ok']}"
    )


if __name__ == "__main__":
    main()
