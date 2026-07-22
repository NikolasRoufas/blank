# Validation Report — bridge milestone

Machine-readable: `validation-report.json` (`ok: true`), `diagnostics/invariance.json`.

| check | result |
|---|---|
| Non-zero bridge edges recovered | PASS (30) |
| Bridge precision / recall vs gold | PASS (1.0 / 1.0) |
| Belief invariant (full vs no-bridge; recomputed per example) | PASS (delta 0.0) |
| Bridges create zero conflict sets | PASS |
| Bridges are not corroboration (no support edges on multi-hop) | PASS |
| Required-hop coverage improves with bridges (1.0 vs 0.0) | PASS |
| `graph_no_bridge` → 0 bridge edges | PASS |
| `graph_no_contradiction` → 0 contradiction edges & 0 conflict sets | PASS |
| `graph_no_propagation` → 0 iterations | PASS |
| `graph_no_temporal` → 0 supersession edges | PASS |
| `graph_top_claim` connectivity lower than full (0.64 vs 1.0) | PASS |
| Directional support recovered as support, not duplicate | PASS (recall 1.0, 0 dup edges) |
| Paraphrases remain duplicates | PASS (cluster accuracy 1.0) |
| Aggregates recomputed from per-example match | PASS |
| Edge counts match serialized graph relations | PASS (serialization round-trip test) |
| Thresholds (NLI) unchanged; structural gate is non-threshold | PASS |
| No output manually edited; no failed example dropped | PASS |

Regression suite: `tests/integration/test_bridge_milestone.py` — **17 passed**.
Mechanism suite size: 202 (support 22, directional_support 20, contradiction 22,
temporal 22, duplicate 22, multi_hop 30, unresolved_conflict 22,
preferred_conflict 22, hard_neutral 20).
