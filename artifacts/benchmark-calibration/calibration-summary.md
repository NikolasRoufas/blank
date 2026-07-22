# Calibration summary

## What this milestone established

1. **Environment fully recovered.** Repo relocated to `~/dev/EGRAG` (out of
   iCloud); `import egrag` 0.03 s (was ~82 s); all gates green; 35 GiB free; no
   stuck processes. Network + GPU-less CPU; full extras synced.
2. **HotpotQA unblocked.** pyarrow installed; 7,405 rows load/validate live; the
   long-standing blocker is cleared. Key validity caveat surfaced: fullwiki gold
   coverage is only 28.2% (prefer distractor split for evidence metrics).
3. **Real NLI works** (roberta-large-mnli): label mapping valid, controlled
   relations correct, ~61 ms/pair on CPU.
4. **Real generation/extraction (Qwen2.5-0.5B + current adapters) do NOT work**
   for the strict structured contract (0% valid) — a real blocker, reported, not
   faked.
5. **Deterministic structural calibration** produced discriminating data on
   HotpotQA: top-k 5 is the recall/precision/cost sweet spot; the greedy-connected
   selector halves multi-hop recall under lexical edges (edge sparsity) — the
   central hypothesis for real-NLI on GPU.

## Selected dev configuration (frozen)

top-k **5**, budget **256**, chunk 256/0, NLI roberta-large-mnli @2a8f12d2
**E0.4 / C0.7 / D0.8** + structural contradiction gate, deterministic seed 0.
Extractor + generator: deterministic for offline pilots; **real models pending a
usable structured generator** (GPU). Frozen in `frozen-configs/`; checksums in
`frozen-configs/checksums.json`.

## Rationale (not EM-driven)

Retrieval top-k chosen on gold-page recall/precision/cost, not answer EM.
Selector left at greedy-connected but flagged: switch to top-claim if real NLI
does not restore multi-hop recall. NLI thresholds retained from the prior
dev-only selection (precision floor 0.8). No setting was chosen using test data.

## Readiness

Ready for the **GPU final matrix** after three prerequisites: (1) a usable
structured generator (chat template / larger model), (2) wiring real
adapters + `DiskCacheBackend` into the runner, (3) optionally the HotpotQA
distractor split. See `final-matrix-plan.md`.
