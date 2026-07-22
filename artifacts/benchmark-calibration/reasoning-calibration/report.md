# Propagation & selection calibration (§12) — key negative finding

**Pipeline:** deterministic (lexical classifier, fake generator), HotpotQA
`hotpot-smoke-25`, full-coverage subset (n=8), budget 256. Source:
`pilots/hotpot-smoke-25-deterministic.json`.

## Selector dominates; it currently HURTS multi-hop recall

At top-k 5 (full-coverage subset):

| variant | selection | avg selected | evidence recall | evidence precision |
|---------|-----------|--------------|-----------------|--------------------|
| passage_rag | (budget) | 4.8 | **0.875** | 0.369 |
| claim_only_rag | (budget) | 7.8 | **0.875** | 0.369 |
| graph_top_claim | top | 7.7 | **0.875** | 0.369 |
| graph_coherent_subgraph | greedy connected | 1.1 | **0.375** | 0.75 |
| graph_no_propagation | greedy connected | 1.1 | 0.375 | 0.75 |
| graph_with_propagation | greedy connected | 1.1 | 0.375 | 0.75 |
| graph_no_contradiction | greedy connected | 1.1 | 0.375 | 0.75 |
| full_egrag | greedy connected | 1.1 | 0.375 | 0.75 |

**Finding:** the **greedy connected selector** collapses to ~1 claim on HotpotQA
and **halves multi-hop evidence recall** (0.875 → 0.375) versus passage/claim/
top-claim selection, trading recall for precision. Root cause: with the
**lexical** classifier the graph has almost **no support edges** (avg edges
0.1 at k3, 0.3 at k5, 1.4 at k8), so the "connected subgraph" is a near-singleton
— it cannot span the two gold hops because the bridge edges were never created.

## Propagation and contradiction toggles are INERT here

`graph_with_propagation` ≡ `graph_no_propagation`; `graph_no_contradiction` ≡
`full_egrag` (identical metrics, node/edge counts). With the lexical classifier
there are essentially no contradiction edges and no belief spread among the few
connected claims on HotpotQA, so these mechanisms have nothing to act on. They
are **not** shown to help or harm on this deterministic data — untestable here,
not "no effect in general."

## Honest interpretation & GPU hypothesis

The connected-subgraph selector's recall loss is an artifact of **edge sparsity**
under the lexical classifier, not necessarily a property of the method. The real
NLI classifier (validated, ~61 ms/pair) produces real support/contradiction
edges and is the mechanism intended to connect multi-hop chains. **Central GPU
hypothesis:** with real NLI (and a usable generator), `graph_coherent_subgraph`/
`full_egrag` should recover multi-hop evidence that lexical-edge greedy selection
misses. Until then, on the deterministic CPU pipeline a **top-claim / budget
selector has materially higher evidence recall** — recorded honestly.

Predeclared knobs for the GPU matrix (not swept on CPU here): propagation on/off,
contradiction/support weights, damping, connected vs top selection,
bridge-connectivity weight, support-coherence weight, redundancy penalty.
