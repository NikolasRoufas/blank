# Before vs. After — bridge milestone (controlled, oracle mode)

Not a benchmark comparison. Synthetic gold-annotated suite (202 examples);
evidential relations from the oracle (upper bound), connectivity from the
deterministic bridge detector.

## Multi-hop reasoning connectivity

| run | bridge edges | required-hop coverage (multi_hop) | connected rate |
|---|---|---|---|
| previous real-NLI full EG-RAG (no bridges) | 0 | 0.14 | — |
| **full EG-RAG + bridges** | **30** | **1.00** | 1.00 |
| graph_no_bridge ablation | 0 | **0.00** | 1.00 |
| graph_top_claim | 30 | 1.00 | **0.6436** (top-claim breaks connectivity) |

Bridges raise required-hop coverage from 0.0 (no-bridge) to 1.0; disabling them
removes exactly the connectivity (bridge edges → 0, coverage → 0).

## Component activation (each ablation isolates its target)

| variant | support | contradiction | supersession | duplicate | bridge | conflict sets | prop iters |
|---|---|---|---|---|---|---|---|
| full_egrag | 172 | 66 | 22 | 22 | 30 | 66 | 2018 |
| graph_no_bridge | 172 | 66 | 22 | 22 | **0** | 66 | 2018 |
| graph_no_propagation | 172 | 66 | 22 | 22 | 30 | 66 | **0** |
| graph_no_contradiction | 172 | **0** | 22 | 22 | 30 | **0** | 1275 |
| graph_no_temporal | 172 | 66 | **0** | 22 | 30 | 66 | 2018 |
| graph_top_claim | 172 | 66 | 22 | 22 | 30 | 66 | 2018 |

## Invariants (bridges do not affect evidence)

- **Belief invariance:** beliefs identical between full and no-bridge runs
  (`bridge_induced_belief_delta = 0.0`).
- **Conflict invariance:** `bridge_induced_conflict_count = 0`.
- Bridges are not corroboration: multi-hop bridged claims have 0 support edges.

## Bridge quality

bridge precision **1.0**, bridge recall **1.0**, bridge entity accuracy **1.0**;
30 accepted bridges (one per multi_hop fixture).

## Support vs. duplicate (corrected fixtures)

| category | support edges | duplicate edges | support recall | dup-cluster acc |
|---|---|---|---|---|
| support | 44 | 0 | 1.00 | — |
| directional_support | 40 | 0 | 1.00 | — |
| duplicate | 44 | 22 | 1.00 | 1.00 |

Directional support is recovered as **support**, not duplicate; paraphrases remain
**duplicates**. No support→duplicate confusion in oracle mode.

## Contradiction precision on hard negatives

The structural contradiction gate demotes contradictions lacking a shared
entity/subject, **without changing NLI thresholds**. Two honest measurements:

- **Guarantee (fake always-contradict source):** 20 contradiction edges ungated
  → **0** gated on the 20 hard-negative pairs (test-verified). The gate removes
  every structurally-unjustified contradiction.
- **Real model (roberta-large-mnli, offline):** **0** spurious contradictions
  ungated on these specific hard-negatives → **0** gated. The model was already
  clean here, so there was no real-model false-positive to remove on this fixture
  set.

**Reported limitation:** the designed hard-negatives are sufficiently unrelated
that roberta-large-mnli already labels them neutral; the earlier real-NLI run's
spurious contradictions arose on *other* categories (near-similar claims), not on
these. So the gate's value is a proven structural guarantee, not a measured
real-model precision gain on this set. Hard-negatives that the model actually
mis-contradicts are needed to measure a real-model improvement.
