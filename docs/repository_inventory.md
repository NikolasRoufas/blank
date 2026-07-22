# Repository inventory (pre-release audit)

Working copy audited: `~/dev/EGRAG` (the non-iCloud, git-tracked, fully readable
copy). The `~/Documents/Coding/EGRAG` directory is a **stale, iCloud-evicted
copy** whose contents are not materialized on disk and must not be used. See the
"Environment note" at the end.

This document is the Phase 1 deliverable of the pre-release audit: a factual map
of what exists, what it is, and what is a candidate for cleanup. No files are
modified as part of writing it.

## What the project is

EG-RAG (Evidence Graph Retrieval-Augmented Generation) is a Python 3.12,
`src/`-layout research framework. It retrieves passages, splits them into atomic
claims, connects claims with typed relations (support, contradiction, duplicate,
dependency, supersession, and a query-conditioned bridge relation), initializes
and propagates signed belief over the resulting graph, resolves conflicts,
selects a compact evidence subgraph, serializes it, and asks a generator for a
cited answer. It is generator-agnostic through typed adapter protocols. The core
install is pure Python and runs offline; optional model integrations live behind
dependency extras.

Entry points: `egrag` CLI (`egrag.cli.main:app`) and the Python API
(`egrag.answering.answer_query`).

## A prior cleanup pass already ran here

This copy already went through a documented cleanup (those process notes are now
under `docs/archive/`: `repository-cleanup-report.md`, `readme-style-audit.md`,
`documentation-validation.md`). That pass:

- removed the root `CLAUDE.md`, folding its content into `CONTRIBUTING.md` and
  `docs/`;
- rewrote `README.md` into a factual, non-marketing document (285 lines) that
  already matches the target style for a public research release;
- created `docs/development.md`, `models.md`, `caching.md`, `benchmarks.md`,
  `reproduction.md`;
- removed environment/config-recovery scratch logs from `artifacts/`;
- git-ignored `.claude/` and generated `_cache/` directories.

Consequently this audit is a **delta**: verify the prior work and finish the
remainder, not a from-scratch rewrite. No AI-instruction files, chat logs,
diaries, or planning scratch remain at the repository root or in `docs/`.

## Directory structure and classification

| Path | Classification | Notes |
|------|----------------|-------|
| `src/egrag/` | required source | 15,152 LOC. Layers: `domain`, `application`, `adapters`, `cli`, `config`, `generation`, `graph`, `reasoning`, `experiments`, `experimental`, `serialization`, `caching`, `observability`, `security`, `fakes`. |
| `tests/` | required source (tests) | 60 test files across `unit/`, `integration/`, `e2e/`, `property/`, `sanity/`. |
| `docs/` | required documentation | 19 files incl. `adr/`. Three are cleanup process/meta docs (see below). |
| `configs/` | required source (config) | 8 YAML/env baseline + ablation configs. |
| `scripts/` | required source | 5 experiment driver scripts. |
| `.github/workflows/ci.yml` | required source | CI definition. |
| `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md` | required documentation | Current and consistent with `docs/`. |
| `pyproject.toml`, `uv.lock`, `.python-version`, `.pre-commit-config.yaml`, `.env.example`, `.gitignore` | required source (project) | Build/tooling config. |
| `artifacts/` | benchmark output + reproducibility records | 4.9 MB. Mixed: manifests, frozen configs, smoke outputs, failure records (keep) alongside generated caches and macOS junk (clean up). Detailed below. |
| `dist/` | generated artifact | Built `egrag-0.1.0` wheel + sdist. Git-ignored; regenerable via `uv build`. |
| `.venv/`, `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`, `.hypothesis/`, `.coverage` | cache | Git-ignored; regenerable. |
| `.claude/settings.local.json` | development file | Local tool config; git-ignored. |

## Cleanup candidates identified (for Phases 2–7)

Ranked by confidence. Nothing here is acted on in Phase 1.

**High confidence — macOS/sync junk (generated, non-authored):**

- `.DS_Store`, `artifacts/.DS_Store` — Finder metadata; git-ignored. Remove from
  disk.
- 32 iCloud **conflict-copy duplicates** named `* 2.json` / `* 2.log` /
  `* 2.jsonl` under `artifacts/paper-results/runs/budget/{128,256}/` and
  `artifacts/paper-results-invalidated/.../runs/budget/{128,256}/`. These are
  macOS sync artifacts. NOTE: spot-checking `aggregate.json` vs `aggregate 2.json`
  shows they **differ by a few bytes** — they are divergent conflict versions,
  not exact duplicates, so they must be **archived, not deleted**.
- `artifacts/benchmark-pipeline/cache/` — generated pipeline cache; currently
  **not** git-ignored (the ignore rule covers `_cache/`, not `cache/`).

**Medium confidence — needs an explicit decision:**

- `artifacts/paper-results-invalidated/` (202 files) — a zero-edge run marked
  with `INVALIDATED.md`. A legitimate audit/failure record, but a large
  invalidated tree is clutter for a public release. Archive candidate.
- Three parallel results trees: `artifacts/paper-results/` (201),
  `artifacts/paper-results-repaired/` (27), `artifacts/paper-results-invalidated/`
  (202). Confusing side by side; may warrant consolidation/labeling.
- `artifacts/bridge-milestone.zip` — a zip archive redundant with the
  `artifacts/bridge-milestone/` directory beside it.
- `docs/repository-cleanup-report.md`, `docs/readme-style-audit.md`,
  `docs/documentation-validation.md` — process/meta docs describing the cleanup
  itself. Useful history, but read as internal audit notes rather than project
  documentation. Archive candidates.

**Release-readiness gaps (additions, not deletions):**

- **No `LICENSE` file** although `pyproject.toml` declares MIT. README notes this
  honestly.
- `pyproject.toml` `[project.urls]` point to `https://example.invalid/egrag`
  placeholders — need real URLs once the public repo exists.

## Unknown / untouched

- The precise provenance difference between `paper-results` and
  `paper-results-repaired` is not fully established from filenames alone; both are
  retained pending a decision. No unknown file is deleted.

## Environment note (why this matters)

The originally-invoked directory `~/Documents/Coding/EGRAG` is inside macOS iCloud
Drive with "Optimize Mac Storage" active on a disk that is ~100% full. Every
source file there is currently **dataless** (evicted; `stat` shows a non-zero
size but `blocks=0`, and reads return empty or time out). iCloud returned hard
errors (`FileProviderInternalErrorDomain Code=12`) during download attempts, and
files re-evict as fast as they download. That copy therefore cannot be audited
and is treated as stale. All work is performed in `~/dev/EGRAG`, which a prior
session relocated out of iCloud specifically to avoid this failure mode.
