# Before vs. After — zero-edge repair

Not a benchmark comparison. These are mechanism-activation and gold-recovery
counts on synthetic, gold-annotated fixtures, showing that the graph mechanisms
moved from inactive (zero edges) to active and gold-aligned.

## Edge activation

| run | support edges | contradiction edges | supersession edges | duplicate edges | conflict sets | propagation iters |
|---|---|---|---|---|---|---|
| **previous zero-edge run** (all graph variants) | 0 | 0 | 0 | 0 | 0 (none recorded) | n/r (edgeless) |
| **repaired oracle — full_egrag** (154 examples) | 176 | 66 | 22 | 22 | 66 | 1997 |
| **repaired end-to-end** (support+contradiction, 44 examples) | see recovery below | | | | | |

## Repaired oracle mechanism recall (vs gold; per category, mean)

| category | key gold metric | value |
|---|---|---|
| support | support_edge_recall | 1.0 |
| contradiction | contradiction_edge_recall / unresolved_conflict_accuracy | 1.0 / 1.0 |
| temporal | supersession_recall | 1.0 |
| duplicate | duplicate_cluster_accuracy / support_edge_recall | 1.0 / 1.0 |
| multi_hop | required_hop_coverage / required_claim_recall | 1.0 / 1.0 |
| unresolved_conflict | conflict_set_recall / unresolved_conflict_accuracy | 1.0 / 1.0 |
| preferred_conflict | conflict_resolution_accuracy | 1.0 |
| (all) | candidate_pair_recall | 1.0 |

## End-to-end edge recovery (activation-based; honest)

End-to-end re-extracts claims from raw text with fresh IDs, so index-based gold
metrics do not apply; we report whether the pipeline produced the expected edge
type from text:

| category | expected edge | % with expected edge | % with any edge |
|---|---|---|---|
| contradiction | contradiction | 1.00 | 1.00 |
| support | support | 0.00 | 1.00 (recovered as a **duplicate** edge) |

**Finding:** the lexical pipeline recovers contradictions from text, but
near-identical corroborating sources are classified as **duplicates** rather than
**support** (lexical overlap > 0.8 → duplicate). This is a genuine end-to-end
limitation of the lexical classifier, surfaced rather than hidden.

## Ablation activation (proves each ablation isolates its component)

| variant | support | contradiction | supersession | duplicate | prop. iters | conflict sets | connected sel. |
|---|---|---|---|---|---|---|---|
| full_egrag | 176 | 66 | 22 | 22 | 1997 | 66 | 154 |
| graph_no_propagation | 176 | 66 | 22 | 22 | **0** | 66 | 154 |
| graph_no_temporal | 176 | 66 | **0** | 22 | 1997 | 66 | 154 |
| graph_no_contradiction | 176 | **0** | 22 | 22 | 1254 | **0** | 154 |
| graph_top_claim | 176 | 66 | 22 | 22 | 1997 | 66 | **110** |

## Affected metrics / remaining failures

- The previous run's graph-mechanism metrics were all uninformative (0 edges);
  they are invalidated.
- After repair, oracle mechanism metrics are saturated at 1.0 because the oracle
  classifier supplies gold relations — these validate *downstream reasoning*
  (propagation, conflict, temporal, selection, duplicate handling), **not** the
  upstream lexical classifier.
- End-to-end remains limited: support is recovered as duplicate; temporal /
  multi-hop / conflict end-to-end are not evaluated because the lexical extractor
  does not populate structured semantics/timestamps from text (oracle-only).
