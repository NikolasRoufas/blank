# Implementation-Gap Note (bridge milestone)

Inspected the live repository before editing. Findings:

## Current relation types

`egrag.domain.models.RelationType`: `SUPPORT`, `CONTRADICTION`, `DUPLICATE`,
`DEPENDENCY`, `SUPERSESSION`, `NEUTRAL`. There is **no `BRIDGES`** type.
`DEPENDENCY` exists in the enum but has **no detector/policy** wired (not used by
construction); per the milestone we start with `BRIDGES`, not `DEPENDS_ON`.

`EvidenceRelation` carries `relation_type`, `relation_confidence` (Probability),
`direction`, `rationale`, `metadata: RelationMetadata|None`
(classifier id/version/revision/explanation/features). No query-conditioned
bridge fields.

## Current propagation edge policy

`reasoning/propagation.py` consumes only `SUPPORT` and `CONTRADICTION` for signed
message passing, plus `DUPLICATE` for source-lineage discounting. A new
`BRIDGES` type **falls through the elif chain and is ignored** → bridge edges
cannot change belief (invariant nearly free; will be test-locked).

## Current conflict edge policy

`reasoning/conflict.py` builds conflict sets only over
`relation_types={RelationType.CONTRADICTION}`. `BRIDGES` is ignored → bridges
create no conflict sets (invariant nearly free; will be test-locked).

## Current selector connectivity policy

`reasoning/selection.py` computes connectivity from `graph.neighbors(cid)`
(`any_nb`) — i.e. **all** relation types — while evidential coherence uses
`support_nb`/`contra_nb` (SUPPORT/CONTRADICTION only). Therefore **adding
`BRIDGES` edges to the graph makes the existing `GreedyConnectedSelector` /
`BeamSearchSelector` treat bridge-connected claims as connected, without giving
them evidential weight** — exactly the desired separation, with no objective-term
rewrite required. New objective weights (`bridge_connectivity_weight`, etc.) are
therefore deferred as a refinement; connectivity-via-neighbors is sufficient for
the controlled multi-hop coverage claim.

## Current multi-hop metric behavior

`experiments/mechanism_eval.py` computes `required_hop_coverage` =
`selected ⊇ required` and `selected_subgraph_connectivity` via BFS over
`run.edges`. With no bridge edges, complementary non-entailing multi-hop claims
are disconnected → real-NLI run measured required-hop coverage ≈ 0.14.

## Entity normalization

`adapters/extraction/baseline.py` strips leading capitalized function words
("The"/"An"/…) via `_NON_ENTITY_WORDS` (added earlier). Generic nouns
("company") are still emitted as entities; the **bridge detector** must ignore
generic/stop entities for bridge matching (handled in the detector, not by
changing global extraction).

## Stale documentation

`docs/architecture.md` still describes a planning-phase / empty repository
("no production code", "working directory is empty"). Requires rewrite to the
implemented system + the evidential-vs-connectivity relation distinction.
`docs/reasoning.md` describes required-hop coverage as support-edge-only.

## EvidenceGraph mutation path

`EvidenceGraph` wraps an immutable `EvidenceGraphSnapshot`; `graph.snapshot()`
exposes claims+relations. Bridge edges are added by rebuilding a graph from the
augmented snapshot (`EvidenceGraph(EvidenceGraphSnapshot(claims=…, relations=…+bridges))`).

## Files expected to change

- `src/egrag/domain/models.py` — `RelationType.BRIDGES`, `BridgeMetadata`,
  `EvidenceRelation.bridge`; `domain/version.py` schema bump → 1.6.0.
- `src/egrag/graph/bridges.py` (new) — `BridgeDetector` protocol,
  `DeterministicBridgeDetector`, query subgoals, generic-entity filter.
- `src/egrag/graph/__init__.py` — exports.
- `src/egrag/experiments/mechanism_eval.py` — bridge integration, `VariantFlags.bridges`,
  contradiction structural gate, bridge metrics, hop coverage via bridges.
- `src/egrag/experiments/mechanisms.py` — revised/added fixtures (multi-hop bridge,
  directional support, hard-negative neutral) with bridge/subgoal gold.
- `src/egrag/caching/keys.py` — bridge config in keys (if needed).
- tests + docs + `CHANGELOG.md`.

## Scope honesty

This turn implements the architecture (relation separation), the deterministic
bridge detector, query subgoals, the contradiction structural gate, revised
fixtures + bridge metrics + the `graph_no_bridge` ablation, regression tests, the
controlled rerun, and priority docs. Production-selector objective-term weights
and a full six-document rewrite are noted where deferred.
