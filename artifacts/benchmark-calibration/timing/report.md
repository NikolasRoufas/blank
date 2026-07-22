# Runtime & feasibility (§13)

All figures measured on this host (Apple M3, 8 cores, 16 GiB, CPU, fp32, torch
2.12.1 / transformers 5.12.1), offline.

## Measured per-stage costs

| Stage | Cost | Source |
|-------|------|--------|
| Cold `import egrag` | 0.03 s | final-verification |
| roberta-large-mnli load + label validation | 24.4 s (once) | nli smoke |
| NLI inference | ~61 ms / single-direction pair (batched 16); ~122 ms / undirected relation | nli smoke |
| Qwen2.5-0.5B load | ~few s (once) | gen smoke |
| Qwen2.5-0.5B generation (64 new tok) | ~1.6 s / example | gen smoke |
| Qwen2.5-0.5B extraction (256 new tok) | **15–43 s / passage** | extractor smoke |
| Deterministic pipeline (retrieve+extract+graph+select+fake gen) | 0.2–2.5 ms / example | pilots |
| HotpotQA parquet load (7405 rows) | 5.6 s (once) | §3 |
| FEVER JSONL load (15935 rows) | ~0.3 s (once) | prior |

## Cache hit-rate / disk growth

`DiskCacheBackend` exists with correct content-addressed keys but is **not wired**
into the NLI classifier / extractor / generator / runner, so cold==warm today
(0% effective hit rate end-to-end). Disk growth from the deterministic pilots is
negligible (KB of JSON). Wiring the cache is a prerequisite for any real-model
matrix (so extraction/NLI are computed once and reused across variants/seeds).

## Full-matrix projection (why it is GPU work)

Final matrix = 8 variants × {HotpotQA, FEVER} × seeds, dev sizes ~100/benchmark.
Extraction and NLI are **variant-independent** (compute once per example/pair,
reuse across variants) — but only if caching is wired.

- **Real extraction on CPU is the killer:** ~20 s/passage × ~5–8 passages/example
  × 100 examples ≈ **3–4 h per benchmark** just for extraction, even shared once
  across variants. Plus NLI (~12 claims → tens of pairs → ~5–10 s/example shared)
  and generation (~1.6 s × 8 variants × 100 ≈ 21 min/benchmark, *if* the generator
  were usable). Total CPU ≈ many hours to a day per benchmark — **infeasible**.
- The generator is moreover **unusable** as-is (0% valid structured output).

**Conclusion: the final matrix is not feasible on this Mac.** Run it on the GPU PC.

## Reduced, scientifically-defensible GPU plan

1. Fix the generation adapter (chat template + stop/JSON constraint) **and/or**
   use a larger instruct model; re-run §4–6 smokes; require valid-output rate near
   1.0 before pilots.
2. Wire `DiskCacheBackend` into extraction, NLI, and generation (keys already
   include model/revision/prompt/thresholds/schema). Precompute extraction + NLI
   **once** per example/pair; variants reuse.
3. Fixed representative subsets: FEVER balanced dev-100 (`fever-dev-100.json`),
   HotpotQA stratified dev-100 (`hotpot-dev-100.json`); evidence/bridge metrics on
   the 28-example full-coverage subset (proxy). Prefer the HotpotQA **distractor**
   split for evidence-grounded metrics (download required; fullwiki gold coverage
   is only 28.2%).
4. Deterministic single seed (0) for headline; add seeds {0,1,2} only for the
   selected configuration's variance band.
5. Only hypothesis-relevant ablations (the 8 variants); `graph_no_temporal` only
   if temporal evidence is meaningful (HotpotQA has little — keep but expect ~no
   effect).
6. Resume command on CUDA: `CUDA_VISIBLE_DEVICES=0 uv run egrag experiment run
   <frozen-config>` after the adapter fix + cache wiring.

Do **not** reduce the dataset after seeing which examples favor EG-RAG; the
subsets here are fixed by seed before any system run and selected by data
availability (gold coverage), never by model success.
