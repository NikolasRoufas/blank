# Benchmark-calibration milestone — Final report

Repository: `/Users/nikolasroufas/dev/EGRAG` (relocated out of iCloud).
Host: Apple M3, 8 cores, 16 GiB, CPU-only. Network available. Date: 2026-07-01.
Nothing committed or pushed (not a git repo). No paper/LaTeX. Final benchmark
matrix **not** executed (plan only).

## 1. Environment recovery

The prior blockers (repo in iCloud → 0%-CPU I/O wedges; 11 stuck `UE` ruff
processes) are gone: Mac rebooted (no stuck processes), repo relocated to
`~/dev/EGRAG`. The owner's earlier `rsync` had aborted (left 4 files); I completed
it (670 files, 0 size mismatches, `pyproject.toml` SHA `b94766dd…`) and synced all
extras. `import egrag` = **0.03 s** (was ~82 s). Disk ~35 GiB free. Details:
`environment-recovery/final-verification.md`.

## 2. Datasets and fingerprints

- **FEVER** `copenlu/fever_gold_evidence` `valid`: 15,935 rows, 0 validation
  issues, fingerprint `978f5e8b26b2…`. Gold-evidence setting (no distractors).
- **HotpotQA** `hotpotqa/hotpot_qa` fullwiki validation: **7,405 rows**
  (now loadable — pyarrow 24.0.0 installed), file SHA `78933c0a…`, fingerprint
  `d126f3e43c3b…`, 0 dup ids, 0 issues. **Validity caveat:** fullwiki gold
  coverage is only **28.2%** (context is retrieved paragraphs); evidence/bridge
  metrics are proxies on the full-coverage subset. Prefer the distractor split for
  the final evidence-grounded matrix.

## 3. Pilot sizes

Fixed, seeded, stratified samples built **before** any system run (never by model
success): FEVER `fever-dev-100` (33/33/34), `fever-smoke-25` (8/8/9); HotpotQA
`hotpot-dev-100` (bridge 55/comparison 45, 25 yes/no, 28 full-coverage),
`hotpot-smoke-25` (8 full-coverage). Deterministic pilots run on the smoke-25
samples; real-model work limited to bounded smokes (see below).

## 4. Real-model results (smokes; honest)

| Component | Model / revision | Result |
|-----------|------------------|--------|
| **NLI** | roberta-large-mnli @2a8f12d2 | ✅ label mapping valid; 4/4 controlled relations correct (supports 0.98 / contradicts 0.999 / neutral / duplicate 0.994); ~61 ms/pair |
| **Generator** | Qwen2.5-0.5B-Instruct | ❌ **0/6** valid structured output (JSON + trailing prose; hallucinated, did not abstain); deterministic; ~1.6 s/gen |
| **Extractor** | Qwen2.5-0.5B-Instruct | ❌ **0/4** valid JSON (emits prose); 15–43 s/passage |

**Root cause:** the HF generation adapters feed the raw prompt to the pipeline
**without the chat template** and **no stop/JSON constraint**; a 0.5B model can't
hold the strict contract. The integrity validators correctly reject the output.
**No fake was substituted.** Real **NLI is usable** and was previously calibrated
on gold claims (dev-only thresholds, precision floor). See `live-smoke/summary.md`,
`generator/`, `claim-extraction/`, `nli-calibration/`.

## 5. Calibration (deterministic pipeline — fake generator, lexical classifier)

Measures evidence-graph behavior, not answer accuracy (no usable real generator).

- **Retrieval/budget (§9):** HotpotQA gold-page recall jumps 0.625→0.875 from
  top-k 3→5 then plateaus at 8 while precision falls; **top-k = 5**, **budget =
  256** selected. FEVER gold-evidence is non-discriminating (no distractors;
  every variant P=1.0/R=0.969, graph has 0 edges).
- **Selection/propagation (§12) — key negative finding:** on HotpotQA the
  greedy-connected selector collapses to ~1 claim and **halves multi-hop evidence
  recall (0.875 → 0.375)** vs passage/claim/top-claim selection, because the
  **lexical** classifier yields a near-edgeless graph (avg 0.1–1.4 edges).
  Propagation and contradiction toggles are **inert** here (nothing to act on).
  Central GPU hypothesis: **real NLI** restores edges → connected selection should
  recover the chain.
- **NLI/contradiction (§10):** thresholds E0.4/C0.7/D0.8 retained from prior
  dev-only selection; real model validates label mapping; structural contradiction
  gate is the conservative default.
- **Bridges (§11):** activation ≈ 0 on the lexical graph (edge-sparse) — proxy
  plan deferred to GPU with real NLI; prior bridge-milestone invariants preserved.

## 6. C1/C2/C3 comparison

**Not run with real generation** (generator unusable → would yield only malformed
output). C1/C2/C3 remain defined candidates; no winner selected (and never by EM
alone). The deterministic 8-variant comparison is fairness-controlled
(`fairness/audit.md`) but compares evidence-graph behavior only.

## 7. Selected configuration & rationale

top-k **5**, budget **256**, chunk 256/0, NLI roberta-large-mnli @2a8f12d2
**E0.4/C0.7/D0.8** + structural gate, deterministic seed 0; extractor/generator
deterministic for offline pilots (real models pending a usable structured
generator). Selector left greedy-connected but flagged to switch to top-claim if
real NLI doesn't restore recall. Chosen on dev evidence/recall/precision/cost — not
test data, not EM.

## 8. Frozen configs

`frozen-configs/fever.yaml`, `frozen-configs/hotpotqa.yaml`,
`frozen-configs/checksums.json` (fever sha `8cd04f73…`, hotpotqa sha `60dca418…`).

## 9. Benchmark pilot metrics by variant

See `pilots/hotpot-smoke-25-deterministic.json` (discriminating) and
`pilots/fever-smoke-25-deterministic.json` (non-discriminating). Headline:
greedy-connected variants R=0.375 / P=0.75; passage/claim/top-claim R=0.875 /
P≈0.37 at top-k 5 (full-coverage n=8, deterministic, proxy).

## 10. Cache performance & runtime projections

Cache backend complete but **not wired** into NLI/extraction/generation/runner →
no end-to-end caching yet. Measured: NLI load 24.4 s + 61 ms/pair; generation
~1.6 s; extraction 15–43 s/passage; deterministic pipeline <3 ms/example. **Full
matrix is infeasible on this CPU** (real CPU extraction alone ≈ 3–4 h/benchmark);
it is **GPU work**. Reduced GPU plan in `timing/report.md`.

## 11. Failed examples / negative results (preserved)

Real generator/extractor malformed outputs preserved verbatim in their JSON
artifacts; greedy-connected recall loss recorded; FEVER non-discrimination
recorded; HotpotQA 28.2% gold-coverage recorded. Nothing silently dropped.

## 12. Limitations

Deterministic-pipeline calibration ≠ real-model benchmark results. HotpotQA
fullwiki gold coverage is low (use distractor). n=8 full-coverage proxy subset is
small. Real-model end-to-end calibration is blocked on a usable generator and on
wiring real adapters + cache into the runner.

## 13. Quality gates (§17)

| Gate | Result |
|------|--------|
| `ruff format --check` | PASS (160 files) |
| `ruff check` | PASS |
| `mypy src` | PASS (100 files) |
| `pytest` (full) | **400 passed, 7 skipped, 0 failed**, 89% coverage |
| `uv build` | PASS (wheel + sdist) |
| benchmark-extra install / HotpotQA adapter / FEVER adapter tests | PASS (incl. `requires_benchmarks` gated load) |
| real NLI / generator / extractor smokes | RAN (NLI pass; generator/extractor reported unusable, honestly) |
| docs build (mkdocs) | N/A — no `mkdocs.yml` (docs are plain markdown) |

Test-suite fix: 5 import-isolation tests made robust to all-extras session
`sys.modules` pollution (subprocess / purge-and-restore); `test_hotpotqa_blocked_cleanly`
now skips when pyarrow is installed. These are root-cause fixes, not silencing
(the core isolation property still holds — verified in a clean process).

## 14. Reproduction & matrix plan

`reproduction-commands.md`; `final-matrix-plan.md` (variants, hypotheses, shared
settings, prerequisites, GPU resume command). Pilot/smoke scripts preserved under
`_scripts/`.

## 15. Readiness for the final paper experiments

**Partially ready.** Cleared: environment, HotpotQA loading, real-NLI validation,
dataset/sample/config freezing, deterministic structural calibration. **Blocking
prerequisites before the final matrix (GPU):** (1) a usable structured generator
(chat template / larger model), (2) wire real adapters + `DiskCacheBackend` into
the runner, (3) HotpotQA distractor split for evidence metrics. Until (1)–(2),
real-model benchmark numbers cannot be produced.
