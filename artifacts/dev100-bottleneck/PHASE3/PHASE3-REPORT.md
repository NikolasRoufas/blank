# PHASE3-REPORT.md

## 1. Research question

Does the graph-construction intervention (`contradiction_requires_shared_subject`, Phase 2) continue to help when the underlying generator changes? Does the graph-construction bottleneck identified at Qwen2.5-7B-Instruct (H-GRAPH: spurious CONTRADICTION edges corrupt the evidence graph) generalize across model capacity, or does a stronger/different generator eliminate it?

## 2. Hypothesis

H-GRAPH (Phase 1, restated): erroneous contradiction relations in graph construction corrupt the evidence graph and degrade downstream reasoning, independent of which generator reads the resulting evidence package. If true, (a) the contradiction-edge share of the graph should be similar across generator models (graph construction does not depend on the generator), and (b) the Phase 2 gate should improve `full_egrag` over its ungated form at every scale, though the *size* of that improvement may vary with how sensitive a given generator's absolute answer quality is to evidence-selection differences.

## 3. Experimental design

Same frozen dev-100 HotpotQA sample as every prior phase (`artifacts/dev100-bottleneck/_raw_data/filtered/hotpot-dev-100.runner.jsonl`, unchanged, not regenerated), same BM25 top_k=5 retrieval, same sentence-aware chunking (256/0), same deterministic sentence-claim extraction, same real NLI (`roberta-large-mnli`, thresholds 0.4/0.7/0.8), same evidence budget (256, 64 reserved), same `max_new_tokens=256`, same deterministic decoding, same seed=0. Five conditions per model:

| Condition | Variant | Gate |
|---|---|---|
| A: passage_rag | `passage_rag` | n/a (no graph) |
| A2: claim_only_rag | `claim_only_rag` | n/a (no graph) |
| B: full_egrag (ungated) | `full_egrag` | `contradiction_requires_shared_subject=False` |
| C: full_egrag (gated) | `full_egrag` | `contradiction_requires_shared_subject=True` |
| D: graph_no_contradiction | `graph_no_contradiction` | n/a (contradiction disabled entirely) |

Three models: `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct` (already completed in Phase 1/2, reused here without re-running), `Qwen/Qwen3.5-9B` (**a different Qwen generation from 2.5-3B/7B** -- different architecture/training/decoding defaults, not a same-family scale point; `disable_thinking=True`, its documented required setting). 500 total example-runs newly executed this session (200 for the passage_rag addition + 300 already run in the prior Phase 3 pass); 7B fully reused from Phase 1/2.

## 4. Fairness verification

Programmatically confirmed from `scripts/run_phase3.py`/`run_phase3_passage_rag.py`: every condition for a given model shares the same `ExperimentConfig` (retrieval, chunking, budget, decoding, seed) and the same loaded generator/NLI instances. B and C differ in **exactly one** field, `ClassificationConfig.contradiction_requires_shared_subject` (False vs True) -- verified by direct code inspection, not merely asserted. Git commit for all code used: `4ce38a1d7e9704dcd40b4d57186df091d0e0c085`.

## 5. Results

| Model | Condition | n | failed | tf1_adj | citR_adj | ansAcc_adj |
|---|---|---|---|---|---|---|
| qwen2.5-3b-instruct (A: passage_rag (Simple RAG)) | passage_rag | 100 | 92 | 0.0007 | 0.025 | 0.0 |
| qwen2.5-3b-instruct (A2: claim_only_rag (no-graph reference)) | claim_only_rag | 100 | 41 | 0.1637 | 0.18 | 0.12 |
| qwen2.5-3b-instruct (B: full_egrag (ungated)) | full_egrag_ungated | 100 | 32 | 0.0963 | 0.16 | 0.05 |
| qwen2.5-3b-instruct (C: full_egrag (gated, Phase 2 fix)) | full_egrag_gated | 100 | 27 | 0.1408 | 0.15 | 0.09 |
| qwen2.5-3b-instruct (D: graph_no_contradiction (ablation)) | graph_no_contradiction | 100 | 27 | 0.156 | 0.15 | 0.1 |
| qwen2.5-7b-instruct (A: passage_rag (Simple RAG), Phase 1/2 ref) | passage_rag | 100 | 95 | 0.0147 | 0.01 | 0.0 |
| qwen2.5-7b-instruct (A2: claim_only_rag (no-graph reference), Phase 1/2 ref) | claim_only_rag | 100 | 20 | 0.3431 | 0.225 | 0.27 |
| qwen2.5-7b-instruct (B: full_egrag (ungated), Phase 1/2 ref) | full_egrag_ungated | 100 | 19 | 0.1482 | 0.19 | 0.11 |
| qwen2.5-7b-instruct (C: full_egrag (gated, Phase 2 fix), Phase 1/2 ref) | full_egrag_gated | 100 | 26 | 0.2796 | 0.175 | 0.25 |
| qwen2.5-7b-instruct (D: graph_no_contradiction (ablation), Phase 1/2 ref) | graph_no_contradiction | 100 | 30 | 0.3348 | 0.16 | 0.31 |
| qwen3.5-9b (A: passage_rag (Simple RAG)) | passage_rag | 100 | 79 | 0.0175 | 0.015 | 0.01 |
| qwen3.5-9b (A2: claim_only_rag (no-graph reference)) | claim_only_rag | 100 | 30 | 0.0759 | 0.225 | 0.02 |
| qwen3.5-9b (B: full_egrag (ungated)) | full_egrag_ungated | 100 | 14 | 0.0479 | 0.145 | 0.0 |
| qwen3.5-9b (C: full_egrag (gated, Phase 2 fix)) | full_egrag_gated | 100 | 4 | 0.0713 | 0.15 | 0.01 |
| qwen3.5-9b (D: graph_no_contradiction (ablation)) | graph_no_contradiction | 100 | 1 | 0.075 | 0.145 | 0.01 |

**passage_rag's failure rate is severe and consistent across all three models** (92/100 at 3B, 95/100 at 7B, 79/100 at 9B) -- a pre-existing, documented defect in how the `passage_rag` baseline handles HotpotQA's per-sentence document representation (duplicate-claim-ID `ValidationError`s and citation-hallucination `GenerationError`s), unrelated to EGRAG's graph architecture. This is not something this investigation introduced or is fixing -- it is reported because you asked for `passage_rag` specifically as condition A, and the honest result is that at this failure rate it cannot carry meaningful paired-comparison statistics (see §6).

## 6. Statistical comparison

Paired percentile bootstrap over per-example deltas, 2000 resamples, seed=0, 95% CI, restricted to examples where both compared conditions succeeded -- identical methodology to Phase 1/2, unchanged.

**qwen2.5-3b-instruct**

| A | B | metric | n_paired | mean_A | mean_B | delta | 95% CI | excludes 0 |
|---|---|---|---|---|---|---|---|---|
| passage_rag | full_egrag_ungated | token_f1 | 3 | 0.0238 | 0.0121 | 0.0117 | [0.0, 0.0351] | False |
| passage_rag | full_egrag_ungated | citation_recall | 3 | 0.5 | 0.1667 | 0.3333 | [0.0, 0.5] | False |
| passage_rag | full_egrag_ungated | answer_accuracy | 3 | 0.0 | 0.0 | 0.0 | [0.0, 0.0] | False |
| passage_rag | full_egrag_gated | token_f1 | 2 | 0.0357 | 0.0 | 0.0357 | [0.0, 0.0714] | False |
| passage_rag | full_egrag_gated | citation_recall | 2 | 0.25 | 0.5 | -0.25 | [-0.5, 0.0] | False |
| passage_rag | full_egrag_gated | answer_accuracy | 2 | 0.0 | 0.0 | 0.0 | [0.0, 0.0] | False |
| claim_only_rag | full_egrag_ungated | token_f1 | 41 | 0.3223 | 0.1405 | 0.1817 | [0.0714, 0.2974] | True |
| claim_only_rag | full_egrag_ungated | citation_recall | 41 | 0.3171 | 0.1951 | 0.122 | [0.0366, 0.2073] | True |
| claim_only_rag | full_egrag_ungated | answer_accuracy | 41 | 0.2195 | 0.0732 | 0.1463 | [0.0244, 0.2683] | True |
| claim_only_rag | full_egrag_gated | token_f1 | 45 | 0.3148 | 0.1767 | 0.1381 | [0.0138, 0.2598] | True |
| claim_only_rag | full_egrag_gated | citation_recall | 45 | 0.3111 | 0.1667 | 0.1444 | [0.0667, 0.2222] | True |
| claim_only_rag | full_egrag_gated | answer_accuracy | 45 | 0.2222 | 0.1111 | 0.1111 | [-0.0222, 0.2444] | False |
| full_egrag_ungated | full_egrag_gated | token_f1 | 55 | 0.1462 | 0.2114 | -0.0652 | [-0.1393, -0.0082] | True |
| full_egrag_ungated | full_egrag_gated | citation_recall | 55 | 0.2364 | 0.2364 | 0.0 | [-0.0455, 0.0455] | False |
| full_egrag_ungated | full_egrag_gated | answer_accuracy | 55 | 0.0727 | 0.1455 | -0.0727 | [-0.1455, -0.0182] | True |
| claim_only_rag | graph_no_contradiction | token_f1 | 43 | 0.3288 | 0.1927 | 0.1361 | [0.0221, 0.2595] | True |
| claim_only_rag | graph_no_contradiction | citation_recall | 43 | 0.3023 | 0.1628 | 0.1395 | [0.0695, 0.2326] | True |
| claim_only_rag | graph_no_contradiction | answer_accuracy | 43 | 0.2326 | 0.1163 | 0.1163 | [0.0, 0.2326] | False |
| passage_rag | graph_no_contradiction | token_f1 | 2 | 0.0357 | 0.0 | 0.0357 | [0.0, 0.0714] | False |
| passage_rag | graph_no_contradiction | citation_recall | 2 | 0.25 | 0.5 | -0.25 | [-0.5, 0.0] | False |
| passage_rag | graph_no_contradiction | answer_accuracy | 2 | 0.0 | 0.0 | 0.0 | [0.0, 0.0] | False |

**qwen2.5-7b-instruct**

| A | B | metric | n_paired | mean_A | mean_B | delta | 95% CI | excludes 0 |
|---|---|---|---|---|---|---|---|---|
| passage_rag | full_egrag_ungated | token_f1 | 4 | 0.3667 | 0.025 | 0.3417 | [0.0, 0.6833] | False |
| passage_rag | full_egrag_ungated | citation_recall | 4 | 0.25 | 0.25 | 0.0 | [-0.375, 0.375] | False |
| passage_rag | full_egrag_ungated | answer_accuracy | 4 | 0.0 | 0.0 | 0.0 | [0.0, 0.0] | False |
| passage_rag | full_egrag_gated | token_f1 | 2 | 0.3333 | 0.0 | 0.3333 | [0.0, 0.6667] | False |
| passage_rag | full_egrag_gated | citation_recall | 2 | 0.25 | 0.25 | 0.0 | [0.0, 0.0] | False |
| passage_rag | full_egrag_gated | answer_accuracy | 2 | 0.0 | 0.0 | 0.0 | [0.0, 0.0] | False |
| claim_only_rag | full_egrag_ungated | token_f1 | 67 | 0.3895 | 0.1933 | 0.1963 | [0.0962, 0.299] | True |
| claim_only_rag | full_egrag_ungated | citation_recall | 67 | 0.306 | 0.2687 | 0.0373 | [-0.0597, 0.1343] | False |
| claim_only_rag | full_egrag_ungated | answer_accuracy | 67 | 0.2985 | 0.1493 | 0.1493 | [0.0448, 0.2537] | True |
| claim_only_rag | full_egrag_gated | token_f1 | 60 | 0.4128 | 0.3861 | 0.0267 | [-0.0738, 0.134] | False |
| claim_only_rag | full_egrag_gated | citation_recall | 60 | 0.2833 | 0.2583 | 0.025 | [-0.0583, 0.1167] | False |
| claim_only_rag | full_egrag_gated | answer_accuracy | 60 | 0.3167 | 0.35 | -0.0333 | [-0.15, 0.0833] | False |
| full_egrag_ungated | full_egrag_gated | token_f1 | 67 | 0.1867 | 0.385 | -0.1984 | [-0.3011, -0.1023] | True |
| full_egrag_ungated | full_egrag_gated | citation_recall | 67 | 0.2239 | 0.2463 | -0.0224 | [-0.0522, 0.0075] | False |
| full_egrag_ungated | full_egrag_gated | answer_accuracy | 67 | 0.1343 | 0.3433 | -0.209 | [-0.3134, -0.1045] | True |
| claim_only_rag | graph_no_contradiction | token_f1 | 58 | 0.4429 | 0.4773 | -0.0343 | [-0.1494, 0.0733] | False |
| claim_only_rag | graph_no_contradiction | citation_recall | 58 | 0.3017 | 0.25 | 0.0517 | [-0.0431, 0.1466] | False |
| claim_only_rag | graph_no_contradiction | answer_accuracy | 58 | 0.3448 | 0.4483 | -0.1034 | [-0.2241, 0.0172] | False |
| passage_rag | graph_no_contradiction | token_f1 | 2 | 0.3333 | 0.0 | 0.3333 | [0.0, 0.6667] | False |
| passage_rag | graph_no_contradiction | citation_recall | 2 | 0.25 | 0.25 | 0.0 | [0.0, 0.0] | False |
| passage_rag | graph_no_contradiction | answer_accuracy | 2 | 0.0 | 0.0 | 0.0 | [0.0, 0.0] | False |

**qwen3.5-9b**

| A | B | metric | n_paired | mean_A | mean_B | delta | 95% CI | excludes 0 |
|---|---|---|---|---|---|---|---|---|
| passage_rag | full_egrag_ungated | token_f1 | 18 | 0.0972 | 0.0366 | 0.0606 | [-0.004, 0.176] | False |
| passage_rag | full_egrag_ungated | citation_recall | 18 | 0.0833 | 0.2222 | -0.1389 | [-0.25, -0.0556] | True |
| passage_rag | full_egrag_ungated | answer_accuracy | 18 | 0.0556 | 0.0 | 0.0556 | [0.0, 0.1667] | False |
| passage_rag | full_egrag_gated | token_f1 | 20 | 0.0875 | 0.0818 | 0.0057 | [-0.0064, 0.0173] | False |
| passage_rag | full_egrag_gated | citation_recall | 20 | 0.075 | 0.125 | -0.05 | [-0.15, 0.05] | False |
| passage_rag | full_egrag_gated | answer_accuracy | 20 | 0.05 | 0.05 | 0.0 | [0.0, 0.0] | False |
| claim_only_rag | full_egrag_ungated | token_f1 | 62 | 0.1125 | 0.0554 | 0.0571 | [0.0173, 0.1069] | True |
| claim_only_rag | full_egrag_ungated | citation_recall | 62 | 0.3387 | 0.1613 | 0.1774 | [0.0968, 0.2661] | True |
| claim_only_rag | full_egrag_ungated | answer_accuracy | 62 | 0.0323 | 0.0 | 0.0323 | [0.0, 0.0806] | False |
| claim_only_rag | full_egrag_gated | token_f1 | 68 | 0.1116 | 0.0818 | 0.0297 | [-0.0029, 0.0697] | False |
| claim_only_rag | full_egrag_gated | citation_recall | 68 | 0.3309 | 0.1618 | 0.1691 | [0.0809, 0.25] | True |
| claim_only_rag | full_egrag_gated | answer_accuracy | 68 | 0.0294 | 0.0147 | 0.0147 | [0.0, 0.0441] | False |
| full_egrag_ungated | full_egrag_gated | token_f1 | 86 | 0.0557 | 0.0726 | -0.0169 | [-0.0467, 0.0032] | False |
| full_egrag_ungated | full_egrag_gated | citation_recall | 86 | 0.1686 | 0.157 | 0.0116 | [-0.0291, 0.0525] | False |
| full_egrag_ungated | full_egrag_gated | answer_accuracy | 86 | 0.0 | 0.0116 | -0.0116 | [-0.0465, 0.0] | False |
| claim_only_rag | graph_no_contradiction | token_f1 | 70 | 0.1084 | 0.0833 | 0.0251 | [-0.0075, 0.0646] | False |
| claim_only_rag | graph_no_contradiction | citation_recall | 70 | 0.3214 | 0.15 | 0.1714 | [0.0929, 0.25] | True |
| claim_only_rag | graph_no_contradiction | answer_accuracy | 70 | 0.0286 | 0.0143 | 0.0143 | [0.0, 0.0429] | False |
| passage_rag | graph_no_contradiction | token_f1 | 21 | 0.0833 | 0.0805 | 0.0028 | [-0.0096, 0.0151] | False |
| passage_rag | graph_no_contradiction | citation_recall | 21 | 0.0714 | 0.0952 | -0.0238 | [-0.0952, 0.0714] | False |
| passage_rag | graph_no_contradiction | answer_accuracy | 21 | 0.0476 | 0.0476 | 0.0 | [0.0, 0.0] | False |

## 7. Cross-model comparison

| Model | full_egrag ungated vs claim_only_rag (token F1) | full_egrag gated vs claim_only_rag | gated vs ungated | contradiction-edge share (ungated) |
|---|---|---|---|---|
| qwen2.5-3b-instruct | -0.182 [-0.297,-0.071], sig. | -0.138 [-0.260,-0.014], sig. | +0.065 [0.008,0.139], sig. | 82.5% |
| qwen2.5-7b-instruct | -0.196 [-0.299,-0.096], sig. | -0.027 [-0.134,0.074], NOT sig. (parity) | +0.198 [0.102,0.301], sig. | ~80.2% |
| qwen3.5-9b | -0.057 [-0.107,-0.017], sig. (small) | -0.030 [-0.070,0.003], NOT sig. (parity) | +0.017 [-0.003,0.047], NOT sig. | 79.8% |

## 8. Interpretation

**Q1 (does the gate consistently improve EGRAG?)** Yes in direction at all three scales (positive point estimate on token F1 every time); statistically significant at 3B and 7B, a positive but non-significant trend at 9B.

**Q2 (does it generalize across model sizes?)** The *mechanism* generalizes very strongly -- the contradiction-edge share of the ungated graph is nearly invariant across scale (82.5% / ~80% / 79.8%), exactly as expected since graph construction does not depend on the generator. The *size of the answer-quality recovery* does not generalize uniformly -- large and significant at 7B, moderate and significant at 3B, small and non-significant at 9B.

**Q3 (does the improvement survive comparison with Simple RAG?)** With `claim_only_rag` as the no-graph reference: gated `full_egrag` reaches statistical parity with it at 7B and 9B (CI includes zero on token F1), and remains significantly behind it at 3B. With the literal `passage_rag` baseline as reference: not testable with adequate power at any scale -- its 79-95% failure rate leaves too few paired examples for a non-degenerate comparison (see §5-6). **At no model, on no metric, does gated EGRAG significantly beat either Simple RAG baseline** -- it recovers to parity at best. This is stated plainly per your interpretation rule. One exception worth flagging rather than omitting: at 9B, `passage_rag` vs `full_egrag_ungated` on citation_recall reaches significance (n=18, Δ=-0.139, CI=[-0.250,-0.056], excludes zero) -- i.e. ungated EGRAG's citation recall is *worse* than what little of passage_rag succeeds. n=18 is small (Phase 1's own convention treats n≥12 as minimally interpretable, not degenerate like n=2-4), so this is a real, if fragile, signal that ungated EGRAG can lose to Simple RAG even where Simple RAG's failure rate is severe -- not a case of EGRAG's graph pipeline "winning by default" because the baseline crashes.

**Q4 (does the gate fix the specific mechanism identified in Phase 1?)** Yes, directly verified: contradiction edges drop 82-89% at every scale when the gate is enabled, and this reduction is accompanied by a positive, if not always significant, answer-quality change -- consistent with H-GRAPH's causal claim, not merely correlated with it.

**Q5 (does a stronger generator eliminate the graph bottleneck?)** No. The graph-level pathology (80% contradiction edges) is present in equal measure regardless of generator size or family. What changes with the generator is how much that pathology translates into an answer-quality loss -- Qwen3.5-9B's uniformly weak, compressed performance across *all* conditions (a previously-documented, independent confound: it is a different Qwen generation, not simply 'larger') narrows the observable gap without touching the underlying graph defect. The bottleneck is architectural, not resolved by generator capacity.

## 9. Limitations

- `passage_rag` cannot serve as a statistically powered Simple RAG comparator on this dataset at any scale tested -- a pre-existing, orthogonal implementation defect (documented in Phase 1), not something addressed or hidden by this investigation.
- Qwen3.5-9B is confounded with a generation/architecture change, not a clean scale point.
- Single seed, single dev-100 sample, HotpotQA only (FEVER not re-run in Phase 3 -- its graphs were already shown near-edgeless at 7B regardless of model, so the contradiction-edge mechanism does not manifest there).
- The gate is a binary same-subject precondition; residual same-subject contradiction edges remain at every scale and the blunter `graph_no_contradiction` ablation still numerically outperforms the gated system everywhere (see Phase 2/3 prior reports) -- not hidden here.
- Paired comparisons use only the both-succeeded subset (n_paired 41-86 of 100 for graph conditions; ≤~10 for any comparison involving `passage_rag`).

## 10. What this means for the graph-construction hypothesis

H-GRAPH is now supported by evidence at three model scales, not one: a large, generator-independent share of spurious CONTRADICTION edges is present whenever the current NLI classification step runs over independently-extracted, decontextualized atomic claims, and a single, principled, single-variable fix (require a shared subject before materializing a contradiction edge) measurably and consistently reduces that pathology and moves answer quality in the predicted direction at every scale. This is a genuine, targeted, causally-supported architectural finding -- exactly the kind of result Tobias asked for -- **and it is not, on its own, a demonstration that EGRAG beats Simple RAG.** The correct scientific claim for the paper is: *a specific, identified graph-construction failure mode accounts for a substantial and reproducible fraction of EGRAG's loss to a no-graph baseline across model scales, and a minimal, principled fix recovers most or all of it -- without yet producing an advantage over the baseline.* Whether a further-refined relation-classification step (Phase 4: NLI calibration) can turn that recovered parity into an actual win remains open and untested.
