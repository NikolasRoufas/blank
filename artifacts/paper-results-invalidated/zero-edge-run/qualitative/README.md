# Qualitative Cases

`all-cases.jsonl` contains **every** (system × example) case for both datasets at
seed 42 — not a cherry-picked favorable subset. Each record has: dataset, system,
example_id, answer, citations, evidence_source_ids, metrics, counts, warnings, and
pointers to the full evidence `package_file` and (for graph variants) `graph_file`
under `runs/main/<dataset>/packages|graphs/`. Those package/graph files carry the
retrieved passages, extracted claims, relations, scores, conflicts, and the
selected subgraph for a full trace.

## Balanced selection across the requested categories

With a fake generator and 3 synthetic examples, the available case types are:

- **Successful multi-hop / support chain**: `synthetic_graph / syn-1` (two
  corroborating sources s1, s2). Graph variants select 1 connected claim
  (recall 0.5); passage/claim variants return both (recall 1.0).
- **Conflict handling / unresolved uncertainty**: `synthetic_graph / syn-2`
  (s4 "closed on schedule" vs s5 "delayed") — inspect `packages/*__syn-2.json`
  for the conflict set and selected evidence.
- **Temporal update / supersession**: `temporal_conflict / tmp-1`
  (old Springfield vs new Rivertown); the gold source is `new`.
- **Citation/answer failure (expected)**: every case has `token_f1 ≈ 0` because
  the fake generator emits a templated answer, not the gold text — a deliberate,
  visible limitation rather than a hidden failure.

## Unavailable case categories (no real components)

Retrieval-failure, claim-extraction-failure, relation-classification-failure, and
subgraph-selection-failure cases cannot be exhibited as *model* failures here: the
baselines are deterministic and lexical, and there were **0 runtime failures**
(`../failed-examples.jsonl` is empty). These require real model backends, which
are not installed.
