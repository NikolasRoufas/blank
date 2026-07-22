# Bridge Milestone — Final Report

Separated graph relations into **evidential** (SUPPORTS / CONTRADICTS / DUPLICATE
/ SUPERSEDES) and **query-conditioned reasoning connectivity** (`BRIDGES`). No
benchmark experiments; controlled synthetic suite only.

## Files changed / added

- `src/egrag/domain/models.py` — `RelationType.BRIDGES`, `BridgeMetadata`,
  `EvidenceRelation.bridge`; `domain/version.py` schema → **1.6.0**.
- `src/egrag/graph/bridges.py` (new) — `BridgeDetector` protocol,
  `DeterministicBridgeDetector`, `detect_bridges`, `extract_entities`,
  `query_subgoals`, generic/stop-entity filtering.
- `src/egrag/graph/nli.py` — `StructuralContradictionGate`,
  `structural_contradiction_ok` (non-threshold contradiction gate).
- `src/egrag/graph/classification.py` (NLI adapter unchanged for bridges),
  `src/egrag/graph/__init__.py` — exports.
- `src/egrag/experiments/mechanism_eval.py` — bridge integration (bridges added
  after evidential construction; selector connectivity via `graph.neighbors`),
  `VariantFlags.bridges`, bridge metrics, belief capture, `build_run`
  `classification_config`.
- `src/egrag/experiments/mechanisms.py` — `GoldBridge`; revised multi-hop fixtures
  (bridges, not support); new `directional_support` and `hard_neutral` categories.
- `scripts/run_bridge_eval.py` (new); tests; docs; `CHANGELOG.md`.

## Schema changes

`BRIDGES` relation + `BridgeMetadata(bridge_entity, bridge_terms, query_conditioned,
bridge_confidence, bridge_method_id)` on `EvidenceRelation.bridge`. Optional/
defaulted → backward compatible; serializes to graph JSON and round-trips.

## Bridge definition / detector / confidence

See `docs/bridge-relations.md`. Deterministic baseline
(`deterministic-bridge-v1`): shared non-generic entity + both claims query-relevant
+ complementary + not duplicate + not already evidential; confidence 0.6–1.0;
per-node degree cap; NLI is **not** the bridge signal.

## Relation-family separation (enforced)

Propagation consumes only SUPPORT/CONTRADICTION (+DUPLICATE discount); conflict
consumes only CONTRADICTION; corroboration counts distinct-source SUPPORT.
`BRIDGES` is ignored by all of these and used only for selector connectivity.

## Contradiction structural gate

`StructuralContradictionGate` demotes a contradiction to neutral unless the pair
shares an entity or subject — reducing unrelated-claim false positives **without
changing NLI thresholds**.

## Corrected support-fixture policy

Near-identical paraphrases are **duplicates**; asymmetric entailment
(`directional_support`) is **support**. Multi-hop complementary claims are
**bridges**, not support.

## Tests added / passing

`tests/integration/test_bridge_milestone.py` — **17 passed**. Covers serialization,
DeepMind bridge + both-required selection, belief/conflict/corroboration
invariance, hop coverage, no-bridge ablation, duplicate/generic/article rejection,
directional support, paraphrase duplicates, structural gate (reject unrelated /
keep true / hard-negative removal), ablation isolation, determinism. Full suite
green (see gates).

## Controlled results (oracle mode, 202 fixtures)

- **bridge precision 1.0, recall 1.0, entity accuracy 1.0**; 30 accepted bridges.
- **required-hop coverage: 1.00 (full) vs 0.00 (no-bridge)** — bridges are the
  cause of multi-hop connectivity (previous real-NLI no-bridge run: 0.14).
- **belief invariance:** beliefs identical full vs no-bridge (delta 0.0).
- **conflict invariance:** bridge-induced conflict count 0.
- **support vs duplicate:** support recall 1.0 with 0 duplicate edges on support &
  directional_support; duplicate cluster accuracy 1.0; no confusion.
- **contradiction precision (hard negatives):** structural gate removes 20→0 with a
  fake always-contradict source; real roberta-large-mnli already emitted 0 spurious
  contradictions on these fixtures (0→0) — see limitation below.
- **ablations isolate:** no-bridge→0 bridges; no-propagation→0 iters;
  no-contradiction→0 contradiction/0 conflicts; no-temporal→0 supersession;
  top-claim connected-rate 0.64 vs 1.0.

## Belief / conflict invariance proof

`diagnostics/invariance.json`: `belief_invariant_full_vs_no_bridge: true`,
`bridge_induced_belief_delta: 0.0`, `bridge_induced_conflict_count: 0`.

## Remaining errors / limitations

- Hard-negatives don't trigger real-model spurious contradictions, so the gate's
  real-model precision gain is unmeasured on this set (guarantee proven via fake).
- Production `GreedyConnectedSelector`/`BeamSearchSelector` were **not** given new
  objective-term weights; bridges work through `graph.neighbors` connectivity,
  which suffices for the controlled coverage claim. Objective-term tuning deferred.
- Oracle relations are an upper bound; results are synthetic, not benchmark.
- `DEPENDS_ON` not implemented (only `BRIDGES`, as instructed).

## Documentation updates

`docs/bridge-relations.md` (new); `docs/reasoning.md` (hop coverage no longer
support-only; bridges separate from propagation/evidence); `docs/architecture.md`
(status + relation families); `CHANGELOG.md`. (Full rewrite of
experiments/evidence-graph/configuration/limitations docs is partially deferred.)

## Quality gates

Reported at session end (ruff, mypy, pytest, build, core-only import).

## Reproduction

`PYTHONPATH=src .venv/bin/python scripts/run_bridge_eval.py` (see
`reproduction-commands.md`).

## Suitability

- **Implementation claims:** YES — relation separation, bridge detector, gate,
  invariants implemented and tested.
- **Controlled multi-hop mechanism claims:** YES — bridges recover multi-hop
  connectivity (hop coverage 0→1) on controlled synthetic examples with proven
  belief/conflict invariance.
- **Real benchmark claims:** NO.

## Next milestone

Real claim extraction, a real answer generator, benchmark adapters (HotpotQA,
FEVER), and a fair baseline + ablation matrix.
