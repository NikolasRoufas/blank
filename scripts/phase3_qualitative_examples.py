#!/usr/bin/env python
"""Phase 3 qualitative traces (diagnostic illustration, not quantitative
evidence). Selection procedure, defined before inspecting outcomes:

  1. "ungated fails, gated recovers": lowest example_id (deterministic tie-
     break) where B_full_egrag_ungated token_f1 == 0 and
     C_full_egrag_gated token_f1 >= 0.3.
  2. "both fail": lowest example_id where B and C both have token_f1 == 0.
  3. "graph_no_contradiction beats gated": lowest example_id where D's
     token_f1 exceeds C's by >= 0.3.

Applied per model (qwen2.5-3b-instruct, qwen2.5-7b-instruct via Phase 1/2,
qwen3.5-9b). Read-only over existing results.jsonl/packages; writes
artifacts/dev100-bottleneck/PHASE3/qualitative_examples.json.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("artifacts/dev100-bottleneck")
PHASE3 = ROOT / "PHASE3"


def load(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    return {json.loads(line)["example_id"]: json.loads(line) for line in path.read_text().splitlines() if line.strip()}


def paths_for(model_key: str, benchmark: str = "hotpotqa") -> dict[str, Path]:
    if model_key == "qwen2.5-7b-instruct":
        return {
            "A_claim_only_rag": ROOT / benchmark / model_key / "results.jsonl",
            "B_full_egrag_ungated": ROOT / benchmark / model_key / "results.jsonl",
            "C_full_egrag_gated": ROOT / benchmark / f"{model_key}_gated" / "results.jsonl",
            "D_graph_no_contradiction": ROOT / benchmark / model_key / "results.jsonl",
        }
    base = PHASE3 / benchmark / model_key
    return {c: base / c / "results.jsonl" for c in
            ("A_claim_only_rag", "B_full_egrag_ungated", "C_full_egrag_gated", "D_graph_no_contradiction")}


def pkg_path(model_key: str, condition: str, example_id: str, benchmark: str = "hotpotqa") -> Path | None:
    if model_key == "qwen2.5-7b-instruct":
        variant = {"B_full_egrag_ungated": "full_egrag", "C_full_egrag_gated": "full_egrag",
                   "D_graph_no_contradiction": "graph_no_contradiction"}.get(condition)
        subdir = f"{model_key}_gated" if condition == "C_full_egrag_gated" else model_key
        if variant is None:
            return None
        p = ROOT / benchmark / subdir / "packages" / f"{variant}__{example_id}.json"
        return p if p.is_file() else None
    variant = {"B_full_egrag_ungated": "full_egrag", "C_full_egrag_gated": "full_egrag",
               "D_graph_no_contradiction": "graph_no_contradiction"}.get(condition)
    if variant is None:
        return None
    p = PHASE3 / benchmark / model_key / condition / "packages" / f"{variant}__{example_id}.json"
    return p if p.is_file() else None


def summarize_condition(row: dict | None) -> dict:
    if row is None:
        return {"present": False}
    pkg = None
    return {
        "present": True,
        "failed": row["failed"],
        "answer": row.get("answer", "")[:300],
        "citations": row.get("citations", []),
        "token_f1": row["metrics"].get("token_f1"),
        "num_selected": row["counts"].get("num_selected"),
        "num_graph_edges": row["counts"].get("num_graph_edges"),
    }


def pick(model_key: str, benchmark: str = "hotpotqa") -> dict:
    p = paths_for(model_key, benchmark)
    rows = {c: load(path) for c, path in p.items()}
    common_ids = sorted(set(rows["A_claim_only_rag"]) & set(rows["B_full_egrag_ungated"]) &
                         set(rows["C_full_egrag_gated"]) & set(rows["D_graph_no_contradiction"]))

    def tf1(cond: str, eid: str) -> float | None:
        r = rows[cond].get(eid)
        return None if r is None or r["failed"] else r["metrics"].get("token_f1")

    examples = {}

    # 1. ungated fails, gated recovers
    for eid in common_ids:
        b, c = tf1("B_full_egrag_ungated", eid), tf1("C_full_egrag_gated", eid)
        if b == 0.0 and c is not None and c >= 0.3:
            examples["ungated_fails_gated_recovers"] = eid
            break

    # 2. both fail
    for eid in common_ids:
        b, c = tf1("B_full_egrag_ungated", eid), tf1("C_full_egrag_gated", eid)
        if b == 0.0 and c == 0.0:
            examples["both_fail"] = eid
            break

    # 3. graph_no_contradiction beats gated
    for eid in common_ids:
        c, d = tf1("C_full_egrag_gated", eid), tf1("D_graph_no_contradiction", eid)
        if c is not None and d is not None and d - c >= 0.3:
            examples["no_contradiction_beats_gated"] = eid
            break

    out = {}
    for label, eid in examples.items():
        conditions = {}
        for cond in ("A_claim_only_rag", "B_full_egrag_ungated", "C_full_egrag_gated", "D_graph_no_contradiction"):
            row = rows[cond].get(eid)
            entry = summarize_condition(row)
            pkg_f = pkg_path(model_key, cond, eid, benchmark)
            if pkg_f is not None:
                pkg = json.loads(pkg_f.read_text())
                claims_by_id = {cl["claim_id"]: cl["text"] for cl in pkg.get("claims", [])}
                rels = pkg.get("relations", [])
                entry["num_claims_in_package"] = len(pkg.get("claims", []))
                entry["relations"] = [
                    {
                        "type": r.get("relation_type"),
                        "confidence": r.get("relation_confidence"),
                        "source_text": claims_by_id.get(r.get("source_claim_id"), "?")[:150],
                        "target_text": claims_by_id.get(r.get("target_claim_id"), "?")[:150],
                    }
                    for r in rels
                ]
                entry["selected_claim_texts"] = [
                    claims_by_id.get(s["claim_id"], "?")[:150] for s in pkg.get("selected", [])
                ]
            conditions[cond] = entry
        out[label] = {"example_id": eid, "conditions": conditions}
    return out


def main() -> None:
    report = {}
    for model_key in ("qwen2.5-3b-instruct", "qwen2.5-7b-instruct", "qwen3.5-9b"):
        report[model_key] = pick(model_key)
    out_path = PHASE3 / "qualitative_examples.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    for model_key, examples in report.items():
        print(f"\n{model_key}: {list(examples.keys())}")


if __name__ == "__main__":
    main()
