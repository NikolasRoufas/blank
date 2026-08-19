# Phase 2 -- Gated Contradiction Edges vs Phase 1 Baseline

Single-variable change from Phase 1's full_egrag: `contradiction_requires_shared_subject=True` in ClassificationConfig (graph/types.py). Everything else -- retrieval, chunking, extraction, NLI model/thresholds, propagation, temporal edges, selection strategy, generator/model/decoding, evidence budget, seed, dataset -- is identical to Phase 1.

## hotpotqa (n=100)

| Condition | n | failed | tf1_adj | citR_adj | ansAcc_adj | avg_edges | avg_selected |
|---|---|---|---|---|---|---|---|
| full_egrag + gated contradiction (Phase 2) | 100 | 26 | 0.2796 | 0.175 | 0.25 | 1.243 | 1.811 |
| full_egrag (Phase 1, ungated) | 100 | 19 | 0.1482 | 0.19 | 0.11 | 3.494 | 2.84 |
| claim_only_rag (no-graph reference) | 100 | 20 | 0.3431 | 0.225 | 0.27 | 0.0 | 7.237 |
| graph_no_contradiction (Phase 1 ablation) | 100 | 30 | 0.3348 | 0.16 | 0.31 | 0.757 | 1.557 |

### Paired bootstrap comparisons (full_egrag_gated_phase2 vs ..., 95% CI, both-succeeded only)

| vs | metric | n_paired | mean_gated | mean_other | delta | 95% CI | excludes 0 |
|---|---|---|---|---|---|---|---|
| full_egrag_phase1 | token_f1 | 67 | 0.385 | 0.1867 | 0.1984 | [0.1023, 0.3011] | True |
| full_egrag_phase1 | citation_recall | 67 | 0.2463 | 0.2239 | 0.0224 | [-0.0075, 0.0522] | False |
| full_egrag_phase1 | answer_accuracy | 67 | 0.3433 | 0.1343 | 0.209 | [0.1045, 0.3134] | True |
| claim_only_rag_phase1 | token_f1 | 60 | 0.3861 | 0.4128 | -0.0267 | [-0.134, 0.0738] | False |
| claim_only_rag_phase1 | citation_recall | 60 | 0.2583 | 0.2833 | -0.025 | [-0.1167, 0.0583] | False |
| claim_only_rag_phase1 | answer_accuracy | 60 | 0.35 | 0.3167 | 0.0333 | [-0.0833, 0.15] | False |
| graph_no_contradiction_phase1 | token_f1 | 68 | 0.408 | 0.4777 | -0.0697 | [-0.1359, -0.0147] | True |
| graph_no_contradiction_phase1 | citation_recall | 68 | 0.2353 | 0.2353 | 0.0 | [-0.0221, 0.0221] | False |
| graph_no_contradiction_phase1 | answer_accuracy | 68 | 0.3676 | 0.4412 | -0.0735 | [-0.1471, -0.0147] | True |

### Relation-type composition: Phase 1 (ungated) vs Phase 2 (gated)

Phase 1 (81 pkgs, 70 w/ edges): `{"contradiction": 227, "support": 53, "duplicate": 3}`

Phase 2 (74 pkgs, 39 w/ edges): `{"support": 53, "duplicate": 2, "contradiction": 37}`

## fever (n=100)

| Condition | n | failed | tf1_adj | citR_adj | ansAcc_adj | avg_edges | avg_selected |
|---|---|---|---|---|---|---|---|
| full_egrag + gated contradiction (Phase 2) | 100 | 24 | 0.0228 | 0.595 | 0.0 | 0.013 | 0.974 |
| full_egrag (Phase 1, ungated) | 100 | 25 | 0.0212 | 0.575 | 0.0 | 0.093 | 1.027 |
| claim_only_rag (no-graph reference) | 100 | 15 | 0.0176 | 0.695 | 0.0 | 0.0 | 1.753 |
| graph_no_contradiction (Phase 1 ablation) | 100 | 24 | 0.0228 | 0.595 | 0.0 | 0.013 | 0.974 |

### Paired bootstrap comparisons (full_egrag_gated_phase2 vs ..., 95% CI, both-succeeded only)

| vs | metric | n_paired | mean_gated | mean_other | delta | 95% CI | excludes 0 |
|---|---|---|---|---|---|---|---|
| full_egrag_phase1 | token_f1 | 75 | 0.0305 | 0.0282 | 0.0022 | [0.0, 0.0067] | False |
| full_egrag_phase1 | citation_recall | 75 | 0.78 | 0.7667 | 0.0133 | [0.0, 0.04] | False |
| full_egrag_phase1 | answer_accuracy | 75 | 0.0 | 0.0 | 0.0 | [0.0, 0.0] | False |
| claim_only_rag_phase1 | token_f1 | 75 | 0.0305 | 0.0234 | 0.007 | [-0.0012, 0.0159] | False |
| claim_only_rag_phase1 | citation_recall | 75 | 0.78 | 0.7933 | -0.0133 | [-0.0667, 0.0267] | False |
| claim_only_rag_phase1 | answer_accuracy | 75 | 0.0 | 0.0 | 0.0 | [0.0, 0.0] | False |
| graph_no_contradiction_phase1 | token_f1 | 76 | 0.0301 | 0.0301 | 0.0 | [0.0, 0.0] | False |
| graph_no_contradiction_phase1 | citation_recall | 76 | 0.7829 | 0.7829 | 0.0 | [0.0, 0.0] | False |
| graph_no_contradiction_phase1 | answer_accuracy | 76 | 0.0 | 0.0 | 0.0 | [0.0, 0.0] | False |

### Relation-type composition: Phase 1 (ungated) vs Phase 2 (gated)

Phase 1 (75 pkgs, 7 w/ edges): `{"contradiction": 6, "support": 1}`

Phase 2 (76 pkgs, 1 w/ edges): `{"support": 1}`
