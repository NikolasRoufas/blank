# EG-RAG Reasoning Subsystem

This document describes the **baseline** algorithms for initial scoring, belief
propagation, conflict resolution, and reasoning-subgraph selection. Every
equation and weight is a **configurable baseline assumption**, not universal
truth. The goal is conservative, deterministic, numerically stable behavior that
is fully explainable — not state-of-the-art accuracy.

See `docs/score-taxonomy.md` for the distinctions between the score concepts.
The reasoning code keeps all of these as **separate fields**: extraction
confidence, relation confidence, source reliability, initial belief, propagated
belief, temporal validity, provenance diversity, query utility, selection
contribution, and final selection score.

## 1. Initial claim scoring

`BaselineInitialScorer` computes six normalized signals per claim, each in
`[0, 1]`:

| Signal | Meaning |
|--------|---------|
| `retrieval` | normalized retrieval score (default 0.5 when unknown) |
| `query_relevance` | Jaccard token overlap of claim and query |
| `extraction` | the claim's extraction confidence |
| `source_reliability` | from the pluggable reliability strategy |
| `temporal_validity` | `1.0`, reduced to `superseded_validity` if superseded |
| `independent_support` | `n / (n + 1)` for `n` distinct corroborating sources |

**Initial belief** is a convex combination (weighted sum normalized by the total
weight):

```
initial_belief = ( Σ_i w_i · signal_i ) / ( Σ_i w_i )
```

This is bounded in `[0, 1]` and avoids multiplying many probabilities (which
would be poorly calibrated). Each weighted term is preserved in
`ClaimScore.contributions` for explanation. **Query utility** is computed
separately as `(query_relevance + retrieval) / 2` — it is a relevance signal,
not a truth signal, and is never reused as belief.

**Variables / defaults:** `ScoreWeights(retrieval=0.5, query_relevance=1.0,
extraction=1.0, source_reliability=1.0, temporal_validity=0.5,
independent_support=1.0)`; `superseded_validity=0.5`; `default_retrieval=0.5`.
Weights must be non-negative and sum to a positive value; NaN/inf are rejected.

## 2. Source reliability

Reliability priors are **configurable assumptions**, never inferred from
recency, repetition, ranking, or popularity. Three pluggable strategies:

- `UniformReliability(value)` — same prior everywhere.
- `ConfiguredPriorReliability(priors, default)` — per-source priors.
- `MetadataReliability(default)` — uses the source's declared
  `reliability_prior` metadata, else the default.

## 3. Belief propagation

`SignedBeliefPropagator` performs deterministic signed message passing in logit
space. For claim `v` at iteration `t`:

```
net_v   = logit(initial_belief_v)
        + support_weight       · Σ_{u ∈ support(v)}        disc(u) · rel(u→v) · 2·(b_u − 0.5)
        − contradiction_weight · Σ_{u ∈ contradiction(v)}  disc(u) · rel(u,v) · 2·(b_u − 0.5)

target_v = sigmoid( clamp(net_v, −L, +L) )
b_v(t+1) = damping · b_v(t) + (1 − damping) · target_v
```

- **Logit/sigmoid** keep values bounded; `clamp ±L` (`logit_clamp`, default 12)
  prevents overflow → no NaN/inf.
- **Signed messages:** support is positive, contradiction negative, centered at
  the neutral belief `0.5` so neutral neighbors send no message.
- **`disc(u)` (discounting):** supporters are grouped by source id (lineage) and
  by duplicate cluster; the *k*-th supporter in a group is scaled by
  `lineage_discount^(k−1) · duplicate_discount^(k−1)`. With
  `duplicate_discount=0`, a second duplicate adds nothing; with
  `lineage_discount<1`, copied sources are discounted. **`DUPLICATE_OF` therefore
  never multiplies evidence independently.**
- **`SUPERSEDES`** is *not* used here to erase evidence; it only reduces
  `temporal_validity` during scoring. Both old and new claims are preserved.
- **Damping** blends old and new beliefs, which stabilizes cycles and prevents
  oscillation.

**Convergence:** iterate until `max |b_v(t+1) − b_v(t)| < tolerance`
(`1e-4`), up to `max_iterations` (50). On non-convergence, either raise the typed
`ConvergenceError` (`on_nonconvergence="raise"`, default) or return a result with
`converged=False` (`"return"`). Per-iteration `max_delta` diagnostics are always
recorded.

**Guards:** self-edges skipped; cycles/oscillation damped; duplicated-evidence
and copied-source inflation discounted; high-degree domination bounded by the
logit clamp; NaN inputs rejected (`ClaimScore` forbids non-finite values and the
propagator re-checks); outputs clamped to `[0, 1]`.

**Ablation:** `NoPropagationBaseline` returns the initial beliefs unchanged.

**Complexity:** `O(iterations · edges)` time, `O(nodes + edges)` space.

## 4. Conflict sets

`ConflictSetResolver` finds **groups** of competing claims as connected
components over `CONTRADICTION` edges (not isolated edges). Each `ConflictSet`
preserves, per member, the propagated belief, source reliability, independent
support, relation confidence, and timestamp. Members are ranked by
**propagated belief, then independent support, then reliability — recency is
never a tiebreaker.** Outcomes:

- `EXCLUDED_IRRELEVANT` — all members below the relevance threshold.
- `REJECTED_LOW_EVIDENCE` — the strongest member is below `low_evidence_threshold`.
- `UNRESOLVED` — the top two are within `margin` (default 0.1); no winner is forced.
- `SUPERSEDED` — the belief leader also supersedes a competitor.
- `PREFERRED` — a clear belief leader, otherwise.

Contradictory evidence is always retained.

## 5. Reasoning-subgraph selection

`ReasoningSubgraph` (branching) is used by default; `ReasoningPath` is reserved
for genuinely linear sequences. The marginal objective for adding claim `c` to a
selection `S`:

```
gain(c) =  utility_weight · query_utility(c)
        +  belief_weight  · propagated_belief(c)
        +  entity_coverage_weight · (new query terms covered / |query terms|)
        +  support_coherence_weight · [c is connected by support within S]
        +  independence_weight      · [source(c) ∉ used sources]
        +  uncertainty_weight       · [c contradicts a selected claim]
        −  redundancy_penalty           · [c duplicates a selected claim]
        −  repeated_lineage_penalty     · [source(c) ∈ used sources]
        −  unresolved_conflict_penalty  · [c in unresolved conflict, no counterpart selected]
```

Three selectors:

1. **`TopClaimsSelector`** — rank by base value (`utility + belief`), fill the
   budget. Ablation baseline; may be disconnected.
2. **`GreedyConnectedSelector`** — seed with the best claim, then repeatedly add
   the connected claim of greatest positive marginal gain; finally retain the
   contradiction counterpart of any selected claim in an unresolved conflict so
   uncertainty stays explainable. Output is connected.
3. **`BeamSearchSelector`** — bounded beam search (`beam_width`, default 4) over
   connected extensions, scored by the total objective.

Every candidate (selected or rejected) yields a `SelectionEntry` with its
initial/propagated belief, utility, components, selection contribution, final
selection score, selected/rejected reason, supporting/contradicting neighbors,
duplicate cluster, source, and token count.

## 6. Token budgeting

`TokenBudget(total, reserved_output)` exposes `available = total − reserved`.
Selection never exceeds `available`. Counting uses a replaceable `TokenCounter`:
`CharacterTokenCounter` (conservative `ceil(len/chars_per_token)` fallback),
`WhitespaceTokenCounter`, or a lazy `HuggingFaceTokenCounter`. The selector never
depends on a specific generator's tokenizer.

## Configuration parameters (summary)

- Scoring: `ScoreWeights`, `superseded_validity`, `default_retrieval`.
- Propagation: `damping`, `tolerance`, `max_iterations`, `support_weight`,
  `contradiction_weight`, `duplicate_discount`, `lineage_discount`,
  `logit_clamp`, `on_nonconvergence`.
- Conflict: `margin`, `low_evidence_threshold`, `irrelevant_utility_threshold`.
- Selection: reward/penalty weights, `beam_width`, `allow_disconnected_fallback`.

## Known failure modes & scientific limitations

- The weighted-sum belief and the message scale `2·(b−0.5)` are **uncalibrated**
  heuristics; absolute belief values are not probabilities of truth.
- Lexical `query_relevance` and the lexical NLI baseline are weak; they should be
  replaced by learned models.
- Discounting by source id approximates independence; true source dependency
  graphs are richer than shared ids.
- Beam search is bounded and may miss the global optimum.
- "Required reasoning hops" is approximated by support-edge coherence.
- Damped propagation converges on the graphs we target but is not guaranteed to
  converge on every signed graph; the iteration cap + typed `ConvergenceError`
  make non-convergence explicit rather than silent.

## Why these are baselines / planned learned alternatives

These deterministic algorithms exist to make the pipeline runnable, explainable,
and testable end-to-end. Planned learned replacements: a calibrated NLI/relation
classifier, a learned reliability model, a learned (or loopy-BP/GNN) belief
estimator with calibration, and a learned selection policy. Each is pluggable
behind the existing interfaces.

## Ablation variants

- `NoPropagationBaseline` — initial beliefs only (isolate propagation's effect).
- `TopClaimsSelector` — selection without connectivity/objective shaping.
- Brute-force vs. pruned candidate generation (graph layer).

## Demonstration trace

Synthetic example (`egrag reason`): claims `c1` "Acme revenue grew in 2023"
(srcA), `c2` "…increased…" (srcB, supports c1), `c3` "…fell…" (srcC, contradicts
c1), `c4` duplicate of c1 (srcA). Query: "did Acme revenue grow in 2023".

```
initial scores:        c1=0.6943  c2=0.5743  c3=0.5543  c4=0.5943
relations:             c2 --support--> c1 ; c1 --contradiction--> c3 ; c1 --duplicate--> c4
propagation:           converged in 15 iterations (max_delta 3.9e-2 → 9.4e-5)
propagated scores:     c1=0.7325 (↑ via support)   c3=0.4617 (↓ via contradiction)
                       c2=0.5743  c4=0.5943 (duplicate c4 does NOT inflate c1)
conflict:              {c1, c3} -> PREFERRED (preferred=c1)
selected subgraph:     [c1, c2, c3, c4]  (c3 retained to explain the contradiction)
```

Reproduce with `uv run egrag reason` (or `--json`).

## Reasoning connectivity vs. evidence (BRIDGES)

Required-hop coverage is **no longer approximated through support edges alone**.
Support coherence and bridge connectivity are **separate** objective signals:

- **Support coherence** uses evidential `SUPPORTS`/`CONTRADICTS` neighbours and
  feeds belief and conflict reasoning.
- **Bridge connectivity** uses query-conditioned `BRIDGES` edges (a shared entity
  links complementary claims) purely to connect a multi-hop reasoning subgraph.

`BRIDGES` edges **do not participate in belief propagation, do not count as
evidence or corroboration, and do not resolve truth**; they only help the
selector connect required claims. See `docs/bridge-relations.md`. On the
controlled multi-hop suite, enabling bridges raises required-hop coverage from
0.0 to 1.0 with byte-identical beliefs (belief invariance).
