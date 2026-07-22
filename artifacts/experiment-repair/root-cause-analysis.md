# Root-Cause Analysis — zero-edge run

Machine-readable traces: `traces/{support,contradiction,multi_hop,temporal_supersession}.json`.
Reproduced on the original synthetic fixtures with the configured pipeline
(`SentenceClaimExtractor` → pruned candidate generation → `LexicalPairClassifier`
→ `GraphBuilder` with `ClassificationConfig` defaults).

## The failing stage (proximate cause)

Edges are dropped at **relation classification vs. threshold**, not at candidate
generation or serialization:

- Candidate pairs **are** generated (reason `lexical_overlap`), e.g. the two
  revenue claims in `support`.
- The `LexicalPairClassifier` scores them by Jaccard token overlap. For the
  intended SUPPORTS pair it returns **entailment = 0.2308**.
- `GraphBuilder` stores a SUPPORTS edge only when entailment ≥ `entailment_threshold`
  = **0.5**. 0.2308 < 0.5 → stored as **neutral** → **no edge**.

A control confirms the pipeline is otherwise correct: a high-overlap support pair
scores entailment = 1.0 and **does** create one SUPPORTS edge; a same-text +
negation pair scores contradiction = 0.7 and would create a CONTRADICTS edge.

## Per-mechanism diagnosis and classification

### Support (`syn-1`: "record revenue growth" vs "revenue grew sharply")
- Pair generated; entailment 0.2308 < 0.5 → no edge.
- **Classification:** fixture-design defect (paraphrase too lexically dissimilar
  for the lexical baseline) + entity-normalization defect (entities = "The",
  "Analysts" — sentence-initial tokens, so the shared-entity signal never helps).

### Contradiction (`syn-2`: "closed on schedule" vs "delayed past June")
- The two claims use **different words** for the conflict ("closed on schedule"
  vs "delayed"); there is **no shared negation parity over shared content**, so
  the classifier sees low overlap → neutral, no CONTRADICTS edge, no conflict set.
- **Classification:** fixture-design defect (contradiction not expressed as a
  shared proposition with opposite polarity that the lexical rule can detect).

### Temporal supersession (`tmp-1`: 2010 Springfield vs 2024 Rivertown)
- Year mentions appear in the **text** but the extractor populates no structured
  `valid`/`observed_at` timestamps, and the two claims share little content
  ("Springfield" vs "Rivertown") → no edge and nothing for the temporal resolver
  to order.
- **Classification:** temporal-metadata defect + fixture-design defect.

### Multi-hop (`syn-1`: requires both revenue claims)
- With no support edge, the two required claims are never connected; coherent
  selection cannot assemble a connected 2-hop subgraph (it falls back to 1 claim).
- **Classification:** downstream consequence of the support fixture-design defect.

## Overall classification

- **Primary: fixture-design defect** — every category's claims are outside the
  lexical classifier's edge-forming range (overlap < 0.5; contradictions not
  expressed as shared-proposition polarity flips; supersession lacks timestamps).
- **Secondary: entity-normalization defect** — `named_entities` are sentence-initial
  words, so shared-entity candidate pruning contributes nothing.
- **Process defect** — no mechanism-level metrics and no component-activation
  preflight, so edgeless graphs passed silently and answer/fake metrics masked it.

**Not** defects: graph builder, candidate pruning logic, thresholds themselves,
serialization, or aggregate metric reporting (all verified correct).

## Repair strategy (consistent with integrity rules)

1. **Do not lower thresholds to make edges appear.** Instead:
2. **Repair fixtures** so each mechanism is expressible to the lexical baseline
   (high-overlap support, shared-proposition polarity-flip contradictions,
   timestamped supersession) — the honest **end-to-end** test of "can the pipeline
   recover structure from text".
3. **Add an oracle mode** with a deterministic gold relation classifier so
   downstream reasoning (propagation, conflict resolution, temporal handling,
   selection) is evaluated independently of the lexical classifier's limits.
4. **Fix entity normalization** so named entities are real tokens, not
   sentence-initial words (improves the shared-entity candidate signal).
5. Add mechanism metrics, gold annotations, an activation preflight, and
   regression tests; rerun only the affected experiments.
