# Phase 3 -- Cross-Model Generality of the Contradiction-Gating Fix

Same frozen dev-100 pipeline as Phase 1/2 (BM25 top_k=5, sentence-aware chunking 256/0, deterministic sentence-claim extraction, real NLI roberta-large-mnli thresholds 0.4/0.7/0.8, evidence budget 256, max_new_tokens=256, seed=0, deterministic decoding). Only the generator model varies across A/B/C rows below; only `contradiction_requires_shared_subject` varies between conditions B and C.

## hotpotqa

### Model-by-model condition summary

| Model | Condition | n | failed | tf1_adj | citR_adj | ansAcc_adj | avg_edges | avg_selected |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-3b-instruct | A_claim_only_rag | 100 | 41 | 0.1637 | 0.18 | 0.12 | 0.0 | 7.492 |
| qwen2.5-3b-instruct | B_full_egrag_ungated | 100 | 32 | 0.0963 | 0.16 | 0.05 | 3.118 | 2.588 |
| qwen2.5-3b-instruct | C_full_egrag_gated | 100 | 27 | 0.1408 | 0.15 | 0.09 | 0.863 | 1.575 |
| qwen2.5-3b-instruct | D_graph_no_contradiction | 100 | 27 | 0.156 | 0.15 | 0.1 | 0.616 | 1.479 |
| qwen2.5-7b-instruct (Phase 1/2 ref) | A_claim_only_rag | 100 | 20 | 0.3431 | 0.225 | 0.27 | 0.0 | 7.237 |
| qwen2.5-7b-instruct (Phase 1/2 ref) | B_full_egrag_ungated | 100 | 19 | 0.1482 | 0.19 | 0.11 | 3.494 | 2.84 |
| qwen2.5-7b-instruct (Phase 1/2 ref) | C_full_egrag_gated | 100 | 26 | 0.2796 | 0.175 | 0.25 | 1.243 | 1.811 |
| qwen2.5-7b-instruct (Phase 1/2 ref) | D_graph_no_contradiction | 100 | 30 | 0.3348 | 0.16 | 0.31 | 0.757 | 1.557 |
| qwen3.5-9b | A_claim_only_rag | 100 | 30 | 0.0759 | 0.225 | 0.02 | 0.0 | 7.129 |
| qwen3.5-9b | B_full_egrag_ungated | 100 | 14 | 0.0479 | 0.145 | 0.0 | 2.709 | 2.512 |
| qwen3.5-9b | C_full_egrag_gated | 100 | 4 | 0.0713 | 0.15 | 0.01 | 0.875 | 1.583 |
| qwen3.5-9b | D_graph_no_contradiction | 100 | 1 | 0.075 | 0.145 | 0.01 | 0.646 | 1.475 |

### Paired bootstrap comparisons (95% CI, 2000 resamples, both-succeeded only)

**qwen2.5-3b-instruct**

| A vs B | metric | n_paired | mean_A | mean_B | delta | 95% CI | excludes 0 |
|---|---|---|---|---|---|---|---|
| B_full_egrag_ungated vs A_claim_only_rag | token_f1 | 41 | 0.1405 | 0.3223 | -0.1817 | [-0.2974, -0.0714] | True |
| B_full_egrag_ungated vs A_claim_only_rag | citation_recall | 41 | 0.1951 | 0.3171 | -0.122 | [-0.2073, -0.0366] | True |
| B_full_egrag_ungated vs A_claim_only_rag | answer_accuracy | 41 | 0.0732 | 0.2195 | -0.1463 | [-0.2683, -0.0244] | True |
| C_full_egrag_gated vs A_claim_only_rag | token_f1 | 45 | 0.1767 | 0.3148 | -0.1381 | [-0.2598, -0.0138] | True |
| C_full_egrag_gated vs A_claim_only_rag | citation_recall | 45 | 0.1667 | 0.3111 | -0.1444 | [-0.2222, -0.0667] | True |
| C_full_egrag_gated vs A_claim_only_rag | answer_accuracy | 45 | 0.1111 | 0.2222 | -0.1111 | [-0.2444, 0.0222] | False |
| C_full_egrag_gated vs B_full_egrag_ungated | token_f1 | 55 | 0.2114 | 0.1462 | 0.0652 | [0.0082, 0.1393] | True |
| C_full_egrag_gated vs B_full_egrag_ungated | citation_recall | 55 | 0.2364 | 0.2364 | 0.0 | [-0.0455, 0.0455] | False |
| C_full_egrag_gated vs B_full_egrag_ungated | answer_accuracy | 55 | 0.1455 | 0.0727 | 0.0727 | [0.0182, 0.1455] | True |
| D_graph_no_contradiction vs A_claim_only_rag | token_f1 | 43 | 0.1927 | 0.3288 | -0.1361 | [-0.2595, -0.0221] | True |
| D_graph_no_contradiction vs A_claim_only_rag | citation_recall | 43 | 0.1628 | 0.3023 | -0.1395 | [-0.2326, -0.0695] | True |
| D_graph_no_contradiction vs A_claim_only_rag | answer_accuracy | 43 | 0.1163 | 0.2326 | -0.1163 | [-0.2326, 0.0] | False |

**qwen3.5-9b**

| A vs B | metric | n_paired | mean_A | mean_B | delta | 95% CI | excludes 0 |
|---|---|---|---|---|---|---|---|
| B_full_egrag_ungated vs A_claim_only_rag | token_f1 | 62 | 0.0554 | 0.1125 | -0.0571 | [-0.1069, -0.0173] | True |
| B_full_egrag_ungated vs A_claim_only_rag | citation_recall | 62 | 0.1613 | 0.3387 | -0.1774 | [-0.2661, -0.0968] | True |
| B_full_egrag_ungated vs A_claim_only_rag | answer_accuracy | 62 | 0.0 | 0.0323 | -0.0323 | [-0.0806, 0.0] | False |
| C_full_egrag_gated vs A_claim_only_rag | token_f1 | 68 | 0.0818 | 0.1116 | -0.0297 | [-0.0697, 0.0029] | False |
| C_full_egrag_gated vs A_claim_only_rag | citation_recall | 68 | 0.1618 | 0.3309 | -0.1691 | [-0.25, -0.0809] | True |
| C_full_egrag_gated vs A_claim_only_rag | answer_accuracy | 68 | 0.0147 | 0.0294 | -0.0147 | [-0.0441, 0.0] | False |
| C_full_egrag_gated vs B_full_egrag_ungated | token_f1 | 86 | 0.0726 | 0.0557 | 0.0169 | [-0.0032, 0.0467] | False |
| C_full_egrag_gated vs B_full_egrag_ungated | citation_recall | 86 | 0.157 | 0.1686 | -0.0116 | [-0.0525, 0.0291] | False |
| C_full_egrag_gated vs B_full_egrag_ungated | answer_accuracy | 86 | 0.0116 | 0.0 | 0.0116 | [0.0, 0.0465] | False |
| D_graph_no_contradiction vs A_claim_only_rag | token_f1 | 70 | 0.0833 | 0.1084 | -0.0251 | [-0.0646, 0.0075] | False |
| D_graph_no_contradiction vs A_claim_only_rag | citation_recall | 70 | 0.15 | 0.3214 | -0.1714 | [-0.25, -0.0929] | True |
| D_graph_no_contradiction vs A_claim_only_rag | answer_accuracy | 70 | 0.0143 | 0.0286 | -0.0143 | [-0.0429, 0.0] | False |

### Relation-type composition (graph conditions B/C/D)

**qwen2.5-3b-instruct**

- B_full_egrag_ungated: 68 pkgs, 54 w/ edges, 212 total edges, 82.5% contradiction -- `{"contradiction": 175, "support": 35, "duplicate": 2}`
- C_full_egrag_gated: 73 pkgs, 28 w/ edges, 63 total edges, 30.2% contradiction -- `{"support": 42, "duplicate": 2, "contradiction": 19}`
- D_graph_no_contradiction: 73 pkgs, 22 w/ edges, 45 total edges, 0.0% contradiction -- `{"support": 43, "duplicate": 2}`

**qwen3.5-9b**

- B_full_egrag_ungated: 86 pkgs, 68 w/ edges, 233 total edges, 79.8% contradiction -- `{"contradiction": 186, "support": 45, "duplicate": 2}`
- C_full_egrag_gated: 96 pkgs, 41 w/ edges, 84 total edges, 27.4% contradiction -- `{"support": 58, "duplicate": 3, "contradiction": 23}`
- D_graph_no_contradiction: 99 pkgs, 36 w/ edges, 64 total edges, 0.0% contradiction -- `{"support": 61, "duplicate": 3}`

**qwen2.5-7b-instruct (Phase 1/2 reference)**

- B_full_egrag_ungated: 81 pkgs, 70 w/ edges -- `{"contradiction": 227, "support": 53, "duplicate": 3}`
- C_full_egrag_gated: 74 pkgs, 39 w/ edges -- `{"support": 53, "duplicate": 2, "contradiction": 37}`

## Qualitative examples (diagnostic illustration, not quantitative evidence)

Selection procedure fixed before inspecting outcomes (`scripts/phase3_qualitative_examples.py`):
lowest example_id where (1) B token_f1==0 and C token_f1>=0.3 ("ungated fails, gated recovers"),
(2) B and C both token_f1==0 ("both fail"), (3) D token_f1 exceeds C's by >=0.3
("graph_no_contradiction beats gated"). Full traces in `qualitative_examples.json`.

**qwen2.5-7b-instruct, example `5a7323ad5542994cef4bc476`** ("ungated fails, gated recovers"):
A, B, and C all select the *same single claim* ("...is the debut album of musician Chantal
Claret, released on June 19, 2012..."). A and B (ungated) both answer "Uncertain" (token F1=0).
C (gated) answers "Chantal Claret" (token F1=1.0) from the identical evidence. Traced to the
package JSON: B's graph carries an UNRESOLVED conflict between two *other*, unselected claims
(one about Chantal Claret's biography, one about an unrelated musician, Max Green -- a
contradiction edge at confidence 0.963 between claims that share no real proposition). Per
`generation/rendering.py`'s `express_uncertainty_for_unresolved_conflicts` policy, *any*
unresolved conflict in the package -- not just one touching the selected evidence -- triggers a
blanket "express uncertainty" instruction in the prompt. C has no such conflict (the gate removed
it) and the model commits to the correct answer from the same evidence. **This is a second harm
mechanism beyond selection/connectivity**: a spurious contradiction edge can degrade the answer
even when it never affects which claim gets selected, simply by triggering global uncertainty
framing. Not something Phase 1/2 identified explicitly; a candidate refinement for future work
(not pursued here) would be scoping that policy to conflicts touching selected claims only.

**qwen2.5-7b-instruct, example `5a7c34905542996dd594b8d5`** ("graph_no_contradiction beats gated"):
C and D again select the identical single claim; C answers "We cannot determine ... based on the
provided information" (token F1=0.2), D answers "Andre Agassi" (token F1=0.8, close to gold).
Here, unlike the previous example, **the package's `relations` list is empty in C** -- no
graph-level artifact (edge or conflict) explains the divergence. This instance is reported as
observed but *not* mechanistically explained by the graph structure; it may reflect ordinary
generation sensitivity to how sparse/ambiguous a single piece of evidence reads, independent of
the contradiction-gating fix. Flagged honestly rather than force-fitted to the hypothesis.

**qwen3.5-9b**: only a "both fail" example was found by the fixed selection procedure (no example
met the "ungated fails, gated recovers" or "no_contradiction_beats_gated" thresholds) -- consistent
with the much smaller, mostly non-significant deltas observed for this model (see below).

## Cross-model interpretation

Checking the six generalization criteria (Phase 3 spec §15) against all three scales:

| Criterion | 3B | 7B (ref) | 9B |
|---|---|---|---|
| 1. Ungated full_egrag < claim_only_rag (token F1) | **yes**, Δ=-0.182, CI excl. 0 | **yes**, Δ=-0.196, CI excl. 0 | **yes**, Δ=-0.057, CI excl. 0 (smaller effect) |
| 2. Contradiction edges dominate ungated graph | **yes**, 82.5% | **yes**, ~80.2% | **yes**, 79.8% |
| 3. Gate substantially reduces contradiction edges | **yes**, 175→19 (-89%) | **yes**, 227→37 (-84%) | **yes**, 186→23 (-88%) |
| 4. Gated significantly improves over ungated | **yes**, tf1 Δ=+0.065 CI excl. 0 | **yes**, tf1 Δ=+0.198 CI excl. 0 | **no**, tf1 Δ=+0.017 CI incl. 0 (positive trend, not significant) |
| 5. Gated reaches parity with claim_only_rag | **partial** -- tf1/citR still sig. worse, ansAcc at parity | **yes** -- tf1/ansAcc both at parity | **partial** -- tf1/ansAcc at parity, citR still sig. worse |
| 6. No other component changed | yes, by construction | yes | yes |

**Criteria 1-3 and 6 hold robustly and with striking numerical consistency across all three
scales** (the ~80% contradiction-edge share is nearly invariant, as expected since graph
construction is generator-independent -- only the *impact* of that same graph on answer
quality varies by which model reads it). This is strong evidence the pathology is architectural,
not an artifact of Qwen2.5-7B-Instruct specifically.

**Criteria 4-5 hold cleanly at 7B, partially at 3B, and are the weakest at 9B.** The 9B result is
not a mechanism-level counterexample -- the same 79.8%-contradiction graph structure and the same
87.6% reduction from gating are present -- but Qwen3.5-9B's absolute task performance on
HotpotQA is uniformly weak across *all four conditions* (token F1 0.05-0.11 vs 7B's 0.15-0.34),
consistent with the pre-existing, independently-documented finding (`artifacts/qwen-matrix/
ACL_REPORT.md`, `docs/reproduction.md`) that Qwen3.5-9B is a *different Qwen generation* from
2.5-3B/7B, not a same-family scale point, and produces verbose/hedging answers that diverge from
gold wording. A uniformly-compressed, low-scoring band across conditions mechanically narrows
between-condition deltas and starves the paired-bootstrap tests of power -- this is a plausible,
previously-documented confound, not a claim that the mechanism itself disappears at larger scale.

## Limitations

- Single seed (0), single dev-100 sample, single dataset (HotpotQA; FEVER not re-run in Phase 3
  -- its graphs were already shown near-edgeless at 7B regardless of model, so this mechanism does
  not manifest there and re-testing would not be informative).
- Qwen3.5-9B is confounded with a generation/architecture change, not a clean scale point -- a
  known, pre-existing limitation of this repository's Qwen matrix, not introduced here.
- Failure rates differ by condition and model (14-41 failed/100), so paired comparisons use only
  the subset where both compared conditions succeeded (n_paired = 41-86 of 100) -- smaller than
  the full sample, consistent with Phase 1/2's own methodology and caveats.
- The gate is a binary same-subject precondition, not a calibrated fix -- residual same-subject
  contradiction edges (19-37 across models) remain, and the second harm mechanism found
  qualitatively (blanket uncertainty-injection from any unresolved conflict, not just ones
  touching selected evidence) is not addressed by this gate at all.
- Three scales is not an exhaustive sweep; this is a generality check, not a proof the pattern
  holds at every possible model size.

## Recommendation

**Investigate further before changing the paper's recommended configuration.** The evidence
supports Conclusion A (the graph-construction pathology -- specifically spurious contradiction
edges -- generalizes across Qwen scales) with high confidence: criteria 1-3 hold with striking
consistency at all three scales tested. It does **not** cleanly support Conclusion B (gated EGRAG
beats Simple RAG) at any scale -- the gated system reaches parity with `claim_only_rag` at best
(7B, and partially 3B/9B), never exceeds it, and `graph_no_contradiction` (the blunter ablation)
remains numerically the strongest graph configuration at every scale tested, exactly as in Phase 2.
Given that:
- the mechanism generalizes (strong evidence for the architectural diagnosis, valuable for the
  paper's causal story regardless of which fix ships),
- the gate is a real, principled, statistically significant improvement over the status quo at
  2/3 scales and a positive (if non-significant) trend at the third,
- but the blunter ablation still wins numerically everywhere, and a second, ungated harm
  mechanism (uncertainty-injection from unresolved conflicts) was found and is not addressed by
  the gate,

the honest recommendation is: **do not yet ship either the gate or the ablation as a final
answer**. Report both findings together -- the pathology's generalization (strong) and the
partial nature of the fix (also strong, and equally informative) -- and treat NLI
calibration/relation-quality (Phase 4, not run here) as the next open question before deciding
what EGRAG's graph-construction step should look like in the paper's final configuration.
