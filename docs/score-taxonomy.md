# EG-RAG Score Taxonomy

EG-RAG deliberately keeps several numeric quantities **distinct**. They answer
different questions, are produced by different stages, and must never be
conflated. Collapsing them would destroy the scientific meaning of the evidence
graph.

| Score | Range | Produced by | Question it answers |
|-------|-------|-------------|---------------------|
| **Extraction confidence** | [0, 1] | Claim extractor | How confident are we that this claim was *extracted correctly* from its source span? |
| **Source reliability** | [0, 1] | Source-reliability scorer (configurable prior) | How much do we *trust the source*, independent of any specific claim? |
| **Belief** | [0, 1] | Belief scorer / propagation | How likely is the claim to be *true*, given all evidence? |
| **Relation confidence** | [0, 1] | Relation classifier | How confident are we that a *relation* (support, contradiction, …) between two claims holds? |
| **Query utility** | [0, 1] | Query-utility scorer | How *useful* is this claim for answering the current query (relevance), regardless of truth? |
| **Selection score** | [0, 1] | Subgraph selector | How strongly should this claim be *included* in the compact reasoning subgraph (a ranking/budget signal)? |

## Why they are separate

- **Extraction confidence ≠ belief.** A claim can be extracted perfectly
  (extraction confidence ≈ 1) yet be false (low belief), and vice versa. The
  extractor **never** sets belief — it only reports how cleanly it parsed the
  text. This separation is enforced in code: `AtomicClaim.belief` is left
  `None` by every extractor.
- **Source reliability ≠ belief.** A highly reliable source can still state a
  claim that other evidence contradicts. Reliability is a *prior on the source*;
  belief is a *posterior on the claim*. Reliability is a configurable input, not
  a measured truth.
- **Relation confidence ≠ belief.** Being confident that claim A *contradicts*
  claim B says nothing about whether A or B is true.
- **Query utility ≠ belief / selection score.** A claim may be highly relevant
  to the query (high utility) but false (low belief). The selection score may
  additionally weigh diversity, coherence, and budget — so it is not just a copy
  of utility.

## Where each lives in the data model

- `AtomicClaim.extraction_confidence` — extraction confidence (set by extractor).
- `AtomicClaim.source_reliability` — source reliability (set downstream; `None`
  until then).
- `AtomicClaim.belief` — belief (set downstream; `None` until then).
- `AtomicClaim.query_utility` — query utility (set downstream; `None` until then).
- `EvidenceRelation.relation_confidence` — relation confidence.
- `SelectedEvidence.selection_score` — selection score.

The extractor only ever populates `extraction_confidence`; all other scores
remain `None` on freshly extracted claims.
