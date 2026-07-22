# Resume Inventory — benchmark-calibration milestone

Captured after the repository was relocated to `~/dev/EGRAG` and the environment
verified healthy (see `environment-recovery/final-verification.md`). This is an
honest map of what is done, what the code actually supports, and what the
remaining tasks require — written **before** running model work so scope is not
overstated.

## Environment (now healthy)

| Item | Value |
|------|-------|
| Repo path | `/Users/nikolasroufas/dev/EGRAG` (non-iCloud) |
| Free disk | ~35 GiB |
| `import egrag` | 0.03 s (was ~82 s on the wedged iCloud disk) |
| Host | macOS Darwin 24, Apple M3, 8 cores, 16 GiB RAM, CPU-only |
| Network | **available** (HF + PyPI reachable) — was disabled in prior sessions |
| venv | `uv sync` with all extras + `benchmarks` + `docs`; torch 2.12.1, transformers 5.12.1, numpy 2.5.0, networkx 3.6.1, pyarrow 24.0.0, sentence-transformers, httpx, mkdocs all import |
| Gates | ruff format/check PASS; mypy strict PASS (100 files); targeted tests 21 passed/2 skipped; `uv build` OK |
| HF model cache | `~/.cache/huggingface/hub` ≈ 5.0 GiB, intact: `roberta-large-mnli`, `Qwen/Qwen2.5-0.5B-Instruct`, `cross-encoder/nli-deberta-v3-large`, `cross-encoder/nli-MiniLM2-L6-H768`, `sentence-transformers/all-MiniLM-L6-v2` |

## Local datasets (`~/.cache/huggingface/hub`)

- **FEVER** — `copenlu/fever_gold_evidence`, JSONL gold-evidence setting. `valid`
  split = 15,935 rows (REFUTES 4887 / SUPPORTS 4638 / NEI 6410). Loadable fully
  offline via stdlib. 62 examples contain ≥1 empty evidence sentence (handled).
- **HotpotQA** — `hotpotqa/hotpot_qa` fullwiki, **parquet only**
  (`validation-00000-of-00001.parquet` ≈ 28 MB, sha `78933c0a31a5f7b4…`). Was
  blocked on `pyarrow`; **pyarrow 24.0.0 is now installed** → live load is now
  possible (validated in §3 work).

## Completed work (prior sessions — preserved, not redone)

- `pyproject.toml` restored from authoritative original + 3 minimal additions
  (`benchmarks` extra, `requires_benchmarks` marker; `pyarrow` deliberately not in
  `all`). SHA `b94766dd…`.
- `benchmarks` extra wired; HotpotQA loader uses a **lazy** pyarrow import and a
  typed `MissingDependencyError(..., "benchmarks")`.
- FEVER empty-evidence handling + regression test.
- Offline calibration tests (`tests/integration/test_benchmark_calibration.py`):
  pyarrow-isolation, HotpotQA `_parse` mapping, FEVER empty-evidence,
  sample-manifest balance/stability, frozen-config checksum.
- Dataset manifests (`dataset-manifests/fever.json`, `hotpotqa.json`) and
  `dataset-validation.md`.
- Balanced FEVER dev samples: `samples/fever-dev-100.json` (33/33/34, seed
  20260629), `samples/fever-smoke-25.json` (8/8/9).
- Config-recovery + environment-recovery reports.

## Code map — what the harness actually supports (critical)

The evaluation harness is **deliberately a deterministic, offline, fake-generator
comparison runner**. This is an architectural boundary (CLAUDE.md: core-only
install must run an offline e2e pipeline with no optional model dependency), not
an oversight. Concretely:

| Component | Production/standalone adapter | Wired into the comparison runner? |
|-----------|------------------------------|-----------------------------------|
| Generator | `generation.adapters.HuggingFaceGenerator`, `OpenAICompatibleGenerator`, `FakeTextGenerator` | **Only `FakeTextGenerator`.** `runner._build_generator` raises `ConfigurationError` for any non-`"fake"` name ("out of scope for the deterministic harness"). |
| Claim extraction | `adapters.extraction.huggingface.HuggingFaceStructuredModel` + `structured.py` (real); `SentenceClaimExtractor` (deterministic) | Variants (`experiments/variants.py`) hardcode `SentenceClaimExtractor()`. |
| NLI | `graph.classification.HuggingFaceNLIClassifier` (roberta-large-mnli); `LexicalPairClassifier` (deterministic) | `_run_graph` hardcodes `LexicalPairClassifier()`. |
| Caching | `caching.DiskCacheBackend` + `build_cache_key`/`build_nli_cache_key` (complete, content+model+revision+prompt+thresholds+schema) | **Not referenced** outside `caching/` and `domain/ports.py`; not wired into NLI/extraction/generation or the runner. |

**Real-model path that DOES exist:** standalone eval scripts under `scripts/`:
`run_real_nli_eval.py` (roberta-large-mnli, dev-only threshold selection on the
controlled mechanism suite, label-mapping validation, frozen-config + sha,
ablations — writes to `artifacts/paper-results-repaired/end-to-end-real-nli/` and
`artifacts/nli-evaluation/`), `run_bridge_eval.py`, `estimate_runtime.py`,
`run_paper_experiments.py`. Prior milestones produced real-NLI results under
`artifacts/nli-evaluation/` and `artifacts/bridge-milestone/`.

### Consequence for this milestone

- "Real generator/extractor/NLI **smoke tests**" (§4–6) are feasible **directly
  against the existing adapters** (load the model, run the documented contract) —
  this is what the milestone asks ("use the existing adapter", "do not create a
  new framework").
- The **candidate comparison and variant matrix** (§8, §15) run through the
  deterministic `ExperimentRunner`, which uses the **fake generator + lexical
  classifier + sentence extractor**. Running C1/C2/C3 or the 8 variants through
  it does **not** exercise the real generator or real NLI. The real-NLI relation
  recovery is measured separately on gold claims via `run_real_nli_eval.py`.
- Driving the *full benchmark comparison* with real models end-to-end would
  require wiring real adapters + persistent caching into `variants.py`/`runner.py`
  (an interface change touching variants, runner, fairness, and tests). That is a
  scoped engineering change, not a calibration tweak, and is **out of scope** for
  "do not create a new framework / keep changes small". It is recorded as a
  limitation and a prerequisite in the final-matrix plan rather than silently
  bolted on or faked.

## C1 / C2 / C3 candidate definitions (from starting-state; not yet config files)

- **C1 (lean/CPU):** top-k 5; chunk ~256; claim limit 5/passage; NLI E0.4/C0.7/D0.8;
  contradiction gate ON; bridges ON (min_conf 0.5); evidence budget 256 tok;
  generator Qwen2.5-0.5B; deterministic; max_new_tokens 48.
- **C2 (higher recall):** top-k 10; claim limit 8; same NLI; budget 384; max_new_tokens 64.
- **C3 (precision-leaning):** top-k 5; claim limit 5; contradiction 0.8; budget 256;
  gate ON; bridges min_conf 0.6.

No `configs/C1|C2|C3.yaml` exist yet; existing `configs/` holds `baseline.yaml`,
`cpu_demo.yaml`, and ablation YAMLs (`passage_rag`, `no_graph`, `no_propagation`,
`top_claims`, `claim_only`). Variant registry (`experiments/variants.py`) defines
all 8 required variants plus `graph_with_propagation`, `graph_coherent_subgraph`.

## Missing dependencies / blockers (current)

- None hard-blocking anymore: pyarrow installed (HotpotQA), torch/transformers
  installed (real models), network up.
- **Runtime feasibility** on CPU M3 is the real constraint for staged pilots and
  the full matrix (assessed in §13). Real roberta-large-mnli + Qwen generation on
  CPU is slow; large pilots and the full matrix are expected to be GPU work.

## Exact next actions (this session, in order)

1. **§3** Validate HotpotQA live parquet load now that pyarrow is present; update
   `dataset-manifests/hotpotqa.json` with real row count + fingerprint; run the
   `requires_benchmarks` gated test (no longer skipped). Build a stratified
   HotpotQA dev sample manifest.
2. **§4–6** Bounded real-model smokes against existing adapters: real NLI
   (label-mapping validation + a few FEVER/controlled pairs + dev-threshold via
   `run_real_nli_eval.py`), real generator (Qwen2.5-0.5B on the generation
   contract: structured output, citations, abstention, determinism), real
   structured extractor (controlled passages). Record model IDs/revisions,
   decoding, latency; preserve malformed outputs.
3. **§13** Measure per-stage latency and project full-matrix runtime; produce the
   reduced/GPU execution plan.
4. **§8/§9/§10/§12** Candidate/retrieval/NLI/propagation calibration via the
   deterministic runner on FEVER dev samples (clearly labelled as deterministic-
   pipeline calibration), plus real-NLI relation calibration via the standalone
   gold-claims path.
5. **§14–18** Freeze dev-selected configs (+checksums), write the final-matrix
   plan (with the real-model wiring prerequisite stated), reports, reproduction
   commands; run final quality gates.

Constraints honored throughout: dev data only (no test tuning, no gold leakage),
no final matrix execution, no fake substituted for a failed real model (smokes
report failure honestly), no benchmark superiority claims, no paper/LaTeX, no
commit/push, all negative results preserved.
