# Dev-100 Bottleneck Investigation -- Results

Model: Qwen2.5-7B-Instruct. Real NLI (roberta-large-mnli, thresholds 0.4/0.7/0.8).
Evidence budget 256 (64 reserved for output), max_new_tokens=256, seed=0, deterministic decoding.

## fever (n=100)

| Variant | n | failed | tf1_adj | citR_adj | ansAcc_adj | avg_claims | avg_edges | avg_selected |
|---|---|---|---|---|---|---|---|---|
| passage_rag | 100 | 65 | 0.0189 | 0.215 | 0.0 | 0.943 | 0.0 | 0.943 |
| claim_only_rag | 100 | 15 | 0.0176 | 0.695 | 0.0 | 1.753 | 0.0 | 1.753 |
| graph_top_claim | 100 | 16 | 0.0197 | 0.675 | 0.0 | 1.738 | 0.083 | 1.738 |
| graph_no_contradiction | 100 | 24 | 0.0228 | 0.595 | 0.0 | 1.697 | 0.013 | 0.974 |
| full_egrag | 100 | 25 | 0.0212 | 0.575 | 0.0 | 1.667 | 0.093 | 1.027 |

### Paired bootstrap comparisons (95% CI, 2000 resamples, both-succeeded only)

| A | B | metric | n_paired | mean_A | mean_B | delta | 95% CI | excludes 0 |
|---|---|---|---|---|---|---|---|---|
| graph_no_contradiction | full_egrag | token_f1 | 75 | 0.0305 | 0.0282 | 0.0022 | [0.0, 0.0067] | False |
| graph_no_contradiction | full_egrag | citation_recall | 75 | 0.78 | 0.7667 | 0.0133 | [0.0, 0.04] | False |
| graph_no_contradiction | full_egrag | answer_accuracy | 75 | 0.0 | 0.0 | 0.0 | [0.0, 0.0] | False |
| full_egrag | claim_only_rag | token_f1 | 74 | 0.0286 | 0.0238 | 0.0049 | [-0.0049, 0.0149] | False |
| full_egrag | claim_only_rag | citation_recall | 74 | 0.7635 | 0.7905 | -0.027 | [-0.0811, 0.0135] | False |
| full_egrag | claim_only_rag | answer_accuracy | 74 | 0.0 | 0.0 | 0.0 | [0.0, 0.0] | False |
| graph_no_contradiction | claim_only_rag | token_f1 | 75 | 0.0305 | 0.0234 | 0.007 | [-0.0012, 0.0159] | False |
| graph_no_contradiction | claim_only_rag | citation_recall | 75 | 0.78 | 0.7933 | -0.0133 | [-0.0667, 0.0267] | False |
| graph_no_contradiction | claim_only_rag | answer_accuracy | 75 | 0.0 | 0.0 | 0.0 | [0.0, 0.0] | False |
| graph_top_claim | full_egrag | token_f1 | 73 | 0.0227 | 0.029 | -0.0063 | [-0.0144, 0.0005] | False |
| graph_top_claim | full_egrag | citation_recall | 73 | 0.7877 | 0.774 | 0.0137 | [-0.0274, 0.0548] | False |
| graph_top_claim | full_egrag | answer_accuracy | 73 | 0.0 | 0.0 | 0.0 | [0.0, 0.0] | False |
| graph_top_claim | claim_only_rag | token_f1 | 79 | 0.0197 | 0.0223 | -0.0026 | [-0.0114, 0.0051] | False |
| graph_top_claim | claim_only_rag | citation_recall | 79 | 0.8165 | 0.8038 | 0.0127 | [-0.0253, 0.057] | False |
| graph_top_claim | claim_only_rag | answer_accuracy | 79 | 0.0 | 0.0 | 0.0 | [0.0, 0.0] | False |

### full_egrag relation-type composition (75 packages, 7 with >=1 edge)

```json
{
  "contradiction": 6,
  "support": 1
}
```

## hotpotqa (n=100)

| Variant | n | failed | tf1_adj | citR_adj | ansAcc_adj | avg_claims | avg_edges | avg_selected |
|---|---|---|---|---|---|---|---|---|
| passage_rag | 100 | 95 | 0.0147 | 0.01 | 0.0 | 5.0 | 0.0 | 4.6 |
| claim_only_rag | 100 | 20 | 0.3431 | 0.225 | 0.27 | 7.9 | 0.0 | 7.237 |
| graph_top_claim | 100 | 11 | 0.2298 | 0.27 | 0.14 | 7.82 | 3.27 | 7.124 |
| graph_no_contradiction | 100 | 30 | 0.3348 | 0.16 | 0.31 | 7.929 | 0.757 | 1.557 |
| full_egrag | 100 | 19 | 0.1482 | 0.19 | 0.11 | 7.815 | 3.494 | 2.84 |

### Paired bootstrap comparisons (95% CI, 2000 resamples, both-succeeded only)

| A | B | metric | n_paired | mean_A | mean_B | delta | 95% CI | excludes 0 |
|---|---|---|---|---|---|---|---|---|
| graph_no_contradiction | full_egrag | token_f1 | 63 | 0.4679 | 0.2103 | 0.2576 | [0.1375, 0.3785] | True |
| graph_no_contradiction | full_egrag | citation_recall | 63 | 0.2381 | 0.2222 | 0.0159 | [-0.0238, 0.0635] | False |
| graph_no_contradiction | full_egrag | answer_accuracy | 63 | 0.4286 | 0.1587 | 0.2698 | [0.1429, 0.381] | True |
| full_egrag | claim_only_rag | token_f1 | 67 | 0.1933 | 0.3895 | -0.1963 | [-0.299, -0.0962] | True |
| full_egrag | claim_only_rag | citation_recall | 67 | 0.2687 | 0.306 | -0.0373 | [-0.1343, 0.0597] | False |
| full_egrag | claim_only_rag | answer_accuracy | 67 | 0.1493 | 0.2985 | -0.1493 | [-0.2537, -0.0448] | True |
| graph_no_contradiction | claim_only_rag | token_f1 | 58 | 0.4773 | 0.4429 | 0.0343 | [-0.0733, 0.1494] | False |
| graph_no_contradiction | claim_only_rag | citation_recall | 58 | 0.25 | 0.3017 | -0.0517 | [-0.1466, 0.0431] | False |
| graph_no_contradiction | claim_only_rag | answer_accuracy | 58 | 0.4483 | 0.3448 | 0.1034 | [-0.0172, 0.2241] | False |
| graph_top_claim | full_egrag | token_f1 | 77 | 0.2429 | 0.1924 | 0.0505 | [-0.0135, 0.1164] | False |
| graph_top_claim | full_egrag | citation_recall | 77 | 0.3247 | 0.2338 | 0.0909 | [0.0195, 0.1688] | True |
| graph_top_claim | full_egrag | answer_accuracy | 77 | 0.1558 | 0.1429 | 0.013 | [-0.039, 0.0649] | False |
| graph_top_claim | claim_only_rag | token_f1 | 74 | 0.2624 | 0.4096 | -0.1472 | [-0.2228, -0.0793] | True |
| graph_top_claim | claim_only_rag | citation_recall | 74 | 0.3243 | 0.277 | 0.0473 | [0.0, 0.1014] | False |
| graph_top_claim | claim_only_rag | answer_accuracy | 74 | 0.1622 | 0.3108 | -0.1486 | [-0.2297, -0.0676] | True |

### full_egrag relation-type composition (81 packages, 70 with >=1 edge)

```json
{
  "contradiction": 227,
  "support": 53,
  "duplicate": 3
}
```
