# EG-RAG Implementation Status

This summarizes what is implemented, what is not, the scientific assumptions, and
the known risks. It complements `docs/architecture.md`, `docs/reasoning.md`, and
`docs/score-taxonomy.md`.

## Implemented functionality

- **Domain** — versioned, frozen, validated Pydantic models (schema 1.5.0) with
  distinct score fields; full typed exception hierarchy.
- **Retrieval** — sentence-aware/whole-doc chunking; pure-Python BM25; dense
  retrieval with an embedding cache; weighted/RRF hybrid fusion; score + lazy
  cross-encoder reranking.
- **Extraction** — deterministic sentence baseline and a structured-generation
  extractor with strict JSON validation, versioned prompt, and injection-aware
  rendering.
- **Graph** — candidate generation (brute-force/pruned, budgets, stats),
  lexical/lazy-NLI/fake classifiers, lexical duplicates, temporal supersession,
  builder + metrics, read API, validation, JSON serialization; GraphML behind the
  `graph` extra (the only NetworkX use).
- **Reasoning** — weighted-sum initial scoring; pluggable source reliability;
  deterministic signed belief propagation (damping, convergence, discounting,
  typed `ConvergenceError`, no-propagation ablation); conflict-set resolution
  with explicit outcomes; top-claims / greedy-connected / beam selection with
  token budgeting and full explanations.
- **Generation** — model-independent `EvidencePackage`; plain/Markdown/chat/JSON
  serializers; baseline prompt-injection defense; fake / lazy HF / OpenAI-
  compatible adapters with capabilities; output parsing, attribution validation,
  and a baseline grounding verifier (+ optional lazy NLI).
- **Hardening** — hierarchical strict YAML config + ablation files; one
  composition root with fail-fast `validate_runtime`; content-addressed caching
  (memory/disk, atomic, corruption-detecting, metrics); reproducibility manifest;
  structured logging + metrics; baseline security utilities; `egrag` CLI
  (`run`, `search`, `extract`, `graph`, `reason`, `inspect-config`, `doctor`).

## Known limitations & unsupported cases

- Baselines are **not learned models**: lexical relevance, lexical NLI, and the
  weighted-sum belief are weak and uncalibrated. Planned learned replacements sit
  behind the existing protocols.
- Corpus loading supports the built-in demo corpus and a simple validated local
  directory of `.txt` files; no PDF/HTML/multimodal ingestion.
- Single-language (English) heuristics in extraction and tokenization.
- Beam search is bounded and may miss the global optimum; "required reasoning
  hops" is approximated by support coherence.
- Ablation configs set the relevant toggles; the composition root runs the full
  flow and honors enable/disable flags, but does not yet branch the graph/reason
  stages off entirely for `passage_rag`/`claim_only` (documented as future work).

## Scientific assumptions

- Source reliability is a **configurable prior**, never inferred from recency,
  repetition, ranking, or popularity.
- Newer ≠ truer: supersession needs the same update-sensitive proposition,
  temporal order, and confidence; conflicts never use recency as a tiebreaker.
- Duplicates and copied sources do not independently inflate belief.
- Belief values are not calibrated probabilities of truth.

## Performance risks

- Brute-force candidate generation and conflict/duplicate scans are O(n²); use
  pruning and budgets on larger graphs.
- The disk cache is process-local and unbounded (no eviction yet).
- Real model adapters (HF/NLI/cross-encoder) are not benchmarked here.

## Security limitations

- The prompt-injection defense is a **documented baseline, not a guarantee**.
- Path/URL/size checks and secret redaction are best-effort; redaction is
  pattern-based and may miss novel secret formats.
- Third-party model loading carries inherent risk; remote model code is **not**
  enabled by default (`security.allow_remote_model_code = false`).
- Retrieved text is always treated as untrusted data and is never executed.

## Next priorities

1. Calibrated learned relation/NLI classifier and belief estimator.
2. Real generator-adapter integration tests behind extras (still no downloads in
   CI).
3. Richer corpus ingestion (formats, streaming) with the same security guards.
4. Cache eviction/size limits and shared-cache safety.
5. Fuller ablation branching in the composition root.
