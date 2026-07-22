# Retrieval & evidence-budget calibration (§9)

**Pipeline:** deterministic (BM25 retriever, `SentenceClaimExtractor`,
`LexicalPairClassifier`, `FakeTextGenerator`). Measures evidence selection /
gold-page coverage, **not** answer accuracy. Real-generator/real-NLI numbers are
GPU work (see `live-smoke/summary.md`).

**Data:** `hotpot-smoke-25` (HotpotQA fullwiki, avg 42.5 docs/example → real
distractors), evidence/citation metrics on the **8-example full-coverage subset**
(gold chain present) — a clearly-labelled proxy. FEVER gold-evidence is
**uninformative** for retrieval calibration (documents ARE the gold sentences; no
distractors — every variant scored P=1.0/R=0.969; see
`pilots/fever-smoke-25-deterministic.json`).

## Top-k sweep (HotpotQA, budget 256, passage_rag as the retrieval probe)

| top-k | gold-page recall (full-cov) | precision | avg graph nodes (graph variants) |
|-------|-----------------------------|-----------|----------------------------------|
| 3 | 0.625 | 0.438 | 4.6 |
| 5 | **0.875** | 0.369 | 8.0 |
| 8 | 0.875 | 0.288 | 12.5 |

**Finding:** recall jumps 3→5 (0.625→0.875) then **plateaus** at 8 while
precision keeps falling and graph size (cost) grows. **Selected top-k = 5** for
the deterministic pipeline: best recall/precision/cost trade-off.

## Evidence budget (256 vs 512)

On FEVER gold-evidence, budgets 256 and 512 were **identical** (few short gold
sentences fit either). On HotpotQA the budget interacts with the selector, not
retrieval, and the bottleneck is the **selector** (see
`reasoning-calibration/report.md`), not the budget. **Selected budget = 256** for
CPU pilots (no measured benefit from 512 at these claim counts); revisit 512 on
GPU with a real generator and longer multi-hop chains.

## Caveats

- n=8 full-coverage subset is small; treat as directional, not a benchmark claim.
- These are lexical-pipeline structural numbers; real NLI changes graph
  connectivity and therefore selection (the central GPU hypothesis).
