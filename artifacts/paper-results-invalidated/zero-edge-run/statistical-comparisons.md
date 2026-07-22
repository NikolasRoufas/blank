# Statistical Comparisons

Source: `statistical-comparisons.json`. Method: **paired percentile bootstrap**
of the per-example delta (full_egrag − strongest baseline), 10,000 resamples,
fixed seed 12345, aligned by example ID. The "strongest baseline" is chosen per
dataset by mean `token_f1` among {passage_rag, reranked_passage_rag,
claim_only_rag} (here: `passage_rag`).

**No significance is claimed.** A percentile bootstrap CI describes sampling
variability of the estimate; it is **not** a hypothesis test, and at n=2 (synthetic_graph)
and n=1 (temporal_conflict) the intervals are degenerate. No effect-size statistic
is implemented, so none is reported.

## full_egrag vs. passage_rag (paired, seed 42)

| dataset | metric | mean full_egrag | mean passage_rag | mean Δ | 95% bootstrap CI |
|---|---|---|---|---|---|
| synthetic_graph | token_f1 | 0.0833 | 0.0769 | +0.0064 | [0.0000, 0.0128] |
| synthetic_graph | citation_recall | 0.5000 | 1.0000 | −0.5000 | [−0.5000, −0.5000] |
| synthetic_graph | evidence_recall | 0.5000 | 1.0000 | −0.5000 | [−0.5000, −0.5000] |
| synthetic_graph | answer_evidence_entailment | 0.0556 | 0.0500 | +0.0056 | [0.0000, 0.0111] |
| temporal_conflict | token_f1 | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| temporal_conflict | citation_recall | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] |
| temporal_conflict | evidence_recall | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] |
| temporal_conflict | answer_evidence_entailment | 0.1111 | 0.1000 | +0.0111 | [0.0111, 0.0111] |

## Factual reading (no superiority claim)

On `synthetic_graph`, full_egrag selects a single coherent connected claim, which
**lowers** citation/evidence recall (−0.50) relative to the passage baseline that
returns all retrieved sources, while marginally raising the lexical token-F1
(+0.006) and lexical entailment (+0.006). On `temporal_conflict` (n=1) the two
systems are tied on recall and token-F1, with full_egrag marginally higher on
lexical entailment. These deltas are not powered and use a fake generator; they
characterize selection behavior, not answer quality.
