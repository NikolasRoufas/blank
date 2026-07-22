# Bridge Relations (query-conditioned reasoning connectivity)

EG-RAG separates graph relations into two families.

## Evidential relations (affect evidential status)

`SUPPORTS`, `CONTRADICTS`, `DUPLICATE`, `SUPERSEDES`. These may influence belief
scoring, propagation, conflict detection, provenance discounting, and temporal
interpretation. (`DEPENDENCY` exists in the enum but is not produced by the
current builder.)

## Reasoning-connectivity relations (affect reasoning chains, NOT evidence)

`BRIDGES`. A `BRIDGES` edge means: *two claims contribute complementary
information to a query-conditioned reasoning chain through a shared entity, slot,
relation, or answer-bearing concept, without implying that either entails the
other.*

### A bridge MUST

- help reasoning-subgraph connectivity, required-hop coverage, and multi-hop
  selection;
- record the shared bridge entity (`BridgeMetadata.bridge_entity`/`bridge_terms`);
- record a query-conditioned confidence (`BridgeMetadata.bridge_confidence`,
  distinct from evidential `relation_confidence`) and `bridge_method_id`;
- remain inspectable in the evidence trace (rationale + metadata).

### A bridge MUST NOT

- increase or decrease belief;
- count as independent corroboration;
- count as entailment or contradiction;
- enter signed belief propagation;
- create or resolve conflict sets;
- imply either connected claim is true.

These prohibitions are **enforced and test-locked**: propagation consumes only
`SUPPORT`/`CONTRADICTION` (+`DUPLICATE` discount); conflict detection consumes
only `CONTRADICTION`; corroboration counts distinct-source `SUPPORT`. `BRIDGES`
is none of these, so it is ignored by all of them. The selector's connectivity
(`graph.neighbors`) does use bridges.

## Creation criteria (deterministic baseline)

A bridge is created when **all** hold:

1. both claims are individually query-relevant (share a query entity);
2. they share a meaningful **non-generic, non-stopword** entity;
3. the shared element is not a stop word or generic noun ("the", "company");
4. the claims are not duplicates (not identical text);
5. the pair is not already linked by an evidential relation;
6. the claims are complementary (each contributes distinct non-shared content),
   i.e. they connect distinct query subgoals.

Per-node bridge degree is capped to prevent bridge spam; same-source duplicate
pairs never bridge internally.

## Worked example

Query: `Who founded the company that acquired DeepMind?`
- A: `Google acquired DeepMind.`
- B: `Google was founded by Larry Page and Sergey Brin.`
- → `A BRIDGES B on entity Google` (not SUPPORTS, not CONTRADICTS). Both required
  claims are then connected and selected; required-hop coverage = 1.0.

Negative examples (no bridge): claims sharing only "the"; unrelated claims sharing
"company"; duplicate paraphrases; a shared entity unrelated to the query; pairs
linked only by lexical overlap.

## Detector interface

`egrag.graph.BridgeDetector` (protocol) with `DeterministicBridgeDetector`
(`bridge_method_id = "deterministic-bridge-v1"`). Pluggable for a future learned
detector; NLI entailment is deliberately **not** the bridge signal.
