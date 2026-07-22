# Existing-Run Audit (zero-edge defect)

Audited run: `artifacts/paper-results/` (driver `scripts/run_paper_experiments.py`),
source fingerprint in `artifacts/paper-results/environment.json`. Generator: `fake`.

## Confirmation of the defect

Every **graph-family** variant on both datasets produced graphs with **nodes but
zero edges** (`relations: []` in the serialized graph JSON *and*
`num_graph_edges = 0` in the efficiency table — the two agree, so this is genuine
construction, not a reporting/serialization loss). Passage/claim variants have no
graph by design (nodes = 0).

## Per (run, dataset, variant) table

Counts marked **n/r** were *not recorded* by the original harness (a finding in
itself: the original run had no mechanism-level metrics). Values are seed-42 means.

| run | dataset | variant | examples | candidate pairs | classified pairs | support edges | contradiction edges | supersession edges | propagation iters | conflict sets | selected size | affected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main_synthetic_graph | synthetic_graph | passage_rag | 2 | n/a | n/a | 0 | 0 | 0 | n/a | n/r | 2.5 | no (no graph) |
| main_synthetic_graph | synthetic_graph | reranked_passage_rag | 2 | n/a | n/a | 0 | 0 | 0 | n/a | n/r | 2.5 | no (no graph) |
| main_synthetic_graph | synthetic_graph | claim_only_rag | 2 | n/a | n/a | 0 | 0 | 0 | n/a | n/r | 2.5 | no (no graph) |
| main_synthetic_graph | synthetic_graph | graph_no_propagation | 2 | n/r | n/r | **0** | **0** | **0** | n/r | n/r | 1.0 | **YES** |
| main_synthetic_graph | synthetic_graph | graph_top_claim | 2 | n/r | n/r | **0** | **0** | **0** | n/r | n/r | 2.5 | **YES** |
| main_synthetic_graph | synthetic_graph | graph_coherent_subgraph | 2 | n/r | n/r | **0** | **0** | **0** | n/r | n/r | 1.0 | **YES** |
| main_synthetic_graph | synthetic_graph | graph_no_temporal | 2 | n/r | n/r | **0** | **0** | **0** | n/r | n/r | 1.0 | **YES** |
| main_synthetic_graph | synthetic_graph | graph_no_contradiction | 2 | n/r | n/r | **0** | **0** | **0** | n/r | n/r | 1.0 | **YES** |
| main_synthetic_graph | synthetic_graph | graph_with_propagation | 2 | n/r | n/r | **0** | **0** | **0** | n/r | n/r | 1.0 | **YES** |
| main_synthetic_graph | synthetic_graph | full_egrag | 2 | n/r | n/r | **0** | **0** | **0** | n/r | n/r | 1.0 | **YES** |
| main_temporal_conflict | temporal_conflict | passage_rag | 1 | n/a | n/a | 0 | 0 | 0 | n/a | n/r | 2.0 | no (no graph) |
| main_temporal_conflict | temporal_conflict | reranked_passage_rag | 1 | n/a | n/a | 0 | 0 | 0 | n/a | n/r | 2.0 | no (no graph) |
| main_temporal_conflict | temporal_conflict | claim_only_rag | 1 | n/a | n/a | 0 | 0 | 0 | n/a | n/r | 2.0 | no (no graph) |
| main_temporal_conflict | temporal_conflict | graph_no_propagation | 1 | n/r | n/r | **0** | **0** | **0** | n/r | n/r | 1.0 | **YES** |
| main_temporal_conflict | temporal_conflict | graph_top_claim | 1 | n/r | n/r | **0** | **0** | **0** | n/r | n/r | 2.0 | **YES** |
| main_temporal_conflict | temporal_conflict | graph_coherent_subgraph | 1 | n/r | n/r | **0** | **0** | **0** | n/r | n/r | 1.0 | **YES** |
| main_temporal_conflict | temporal_conflict | graph_no_temporal | 1 | n/r | n/r | **0** | **0** | **0** | n/r | n/r | 1.0 | **YES** |
| main_temporal_conflict | temporal_conflict | graph_no_contradiction | 1 | n/r | n/r | **0** | **0** | **0** | n/r | n/r | 1.0 | **YES** |
| main_temporal_conflict | temporal_conflict | graph_with_propagation | 1 | n/r | n/r | **0** | **0** | **0** | n/r | n/r | 1.0 | **YES** |
| main_temporal_conflict | temporal_conflict | full_egrag | 1 | n/r | n/r | **0** | **0** | **0** | n/r | n/r | 1.0 | **YES** |

The budget-sweep runs (`budget_128/256/512`) reuse the same `synthetic_graph`
data and the same graph variants → **also affected** for graph-mechanism purposes.

## Determinations

- **Which runs are affected:** all graph-family variants in `main_synthetic_graph`,
  `main_temporal_conflict`, and the budget sweep. Passage/claim variants are
  unaffected (they intentionally build no graph).
- **Every graph truly contains zero edges?** Yes — confirmed in both the graph
  JSON (`relations: []`) and the recorded `num_graph_edges` (0).
- **Construction vs reporting?** **Construction.** A controlled high-overlap pair
  builds 1 edge (see `traces/` and `root-cause-analysis.md`); reporting and
  serialization are accurate.
- **Were candidate pairs generated?** **Yes** (e.g. `syn-1` generates the s1–s2
  pair via lexical overlap), so it is not a total candidate-pruning failure.
- **Were pairs classified?** Yes. The lexical classifier returned **entailment ≈
  0.23** for the intended support pair — **below** the `entailment_threshold` 0.5
  → stored as **neutral** → no edge.
- **Predictions neutral / thresholds rejected non-neutral?** Yes: predictions fell
  to neutral or below threshold. This is the proximate cause.
- **Timestamps / provenance missing?** Provenance present; the temporal fixture
  carries year mentions in text but the extractor did not populate structured
  `valid`/`observed_at` timestamps → temporal resolver had nothing to act on.
- **Variant config disabled relations?** No — `full_egrag` (all on) also had 0 edges.
- **Serialization lost edges?** No.
- **Aggregate misreported counts?** No — counts correctly report 0.

## Root cause (summary; full analysis in root-cause-analysis.md)

**Fixture-design defect (primary):** synthetic support/contradiction claims are
paraphrases with Jaccard token overlap < 0.5, below the lexical classifier's
edge-forming threshold; the supersession fixture lacks structured timestamps.
**Entity-normalization defect (secondary):** `named_entities` are sentence-initial
tokens ("The", "An"), weakening the shared-entity candidate signal.
**Missing mechanism metrics/preflight (process defect):** the run reported only
answer/fake metrics and had no check that edge-requiring examples produced edges,
so the empty graphs passed silently.
