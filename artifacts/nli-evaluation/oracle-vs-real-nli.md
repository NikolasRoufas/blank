# Oracle vs. Real-NLI — comparison

**Oracle** = gold relations (`GoldRelationClassifier`), an upper-bound diagnostic
(unchanged). **Real-NLI** = `roberta-large-mnli` (rev `2a8f12d2…`, CPU, offline)
over the **same gold claims** with dev-frozen thresholds (entailment 0.4,
contradiction 0.7, duplicate 0.8). Running on gold claims isolates **NLI relation
recovery** from extraction. Lexical/oracle/zero-edge results are untouched.

## Per-category metrics (mean; oracle → real-NLI)

| category | metric | oracle | real-NLI | gap |
|---|---|---|---|---|
| support | support_edge_recall | 1.00 | **0.00** | −1.00 |
| support | required_hop_coverage | 1.00 | 1.00 | 0 |
| contradiction | contradiction_edge_recall | 1.00 | **1.00** | 0 |
| contradiction | conflict_set_recall / unresolved_accuracy | 1.00 | 1.00 | 0 |
| temporal | supersession_recall | 1.00 | **1.00** | 0 |
| temporal | (NLI also adds contradiction edges) | — | contradiction_recall 1.00 | n/a |
| duplicate | duplicate_cluster_accuracy | 1.00 | **1.00** | 0 |
| duplicate | support_edge_recall | 1.00 | **0.00** | −1.00 |
| multi_hop | support_edge_recall | 1.00 | **0.00** | −1.00 |
| multi_hop | required_hop_coverage | 1.00 | **0.14** | −0.86 |
| unresolved_conflict | conflict_set_recall / unresolved_accuracy | 1.00 | 1.00 | 0 |
| preferred_conflict | contradiction_edge_recall / conflict_set_recall | 1.00 | 1.00 | 0 |
| preferred_conflict | support_edge_recall | 1.00 | **1.00** | 0 |
| preferred_conflict | conflict_resolution_accuracy | 1.00 | **0.18** | −0.82 |
| (all) | candidate_pair_recall | 1.00 | **1.00** | 0 |

Accepted edges (full suite, real-NLI): support 40, contradiction 135,
supersession 22, duplicate 88; conflict sets 110.

## Threshold selection (development only)

308 dev candidate pairs. Best config by macro-F1 with a support/contradiction
precision floor of 0.8: **entailment 0.4, contradiction 0.7** (duplicate fixed at
0.8). **The precision floor was NOT met** — dev contradiction precision = 0.47
(support precision = 1.0). roberta-large-mnli over-predicts CONTRADICTION on short,
unrelated synthetic claims; raising the contradiction threshold to 0.7 mitigates
but does not eliminate it. Thresholds were frozen and checksummed before the full
run; the duplicate threshold was **not** lowered to manufacture support edges.

## Error attribution (earliest failing stage)

`candidate generation → NLI classification → graph construction → propagation/conflict → selection`

- **Support failure** (recall 0.0): candidate recall is 1.0, so the failure is at
  **NLI classification** — near-identical support pairs score mutual entailment
  ≥ 0.8 and are (correctly, per policy) labeled **DUPLICATE**, not SUPPORT.
- **Multi-hop failure** (hop coverage 0.14): also **NLI classification** — the two
  bridge claims (e.g. "X's CEO is P" / "P was born in C") do not entail each
  other, so no SUPPORT edge connects them; NLI does not infer query-specific
  `DEPENDS_ON` dependency. Selection then cannot assemble a connected 2-hop
  subgraph.
- **Preferred-conflict resolution failure** (0.18): **NLI classification** emits
  spurious contradiction/duplicate edges on the short synthetic claims, distorting
  the belief that the conflict resolver uses to pick the preferred side.
- **Contradiction / temporal supersession / duplicate**: recovered (no gap).
  Supersession is timestamp-driven and NLI-independent.

## Reading

Real NLI **end-to-end recovers contradiction and (timestamped) supersession** on
controlled synthetic claims, matching the oracle. It **does not recover** the
support and multi-hop mechanisms on these fixtures: near-identical "support"
claims are duplicates, and multi-hop bridges are not entailments. This is an
honest characterization of NLI relation recovery, **not** a benchmark result.
