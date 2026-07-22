# Repository audit report (pre-release)

Final report for the pre-release audit of EG-RAG. Companion to the Phase 1
`docs/repository_inventory.md`. All work was performed in the readable, non-iCloud
working copy `~/dev/EGRAG` (see "Environment and workspace" below).

## Repository summary

EG-RAG is a Python 3.12 research framework for evidence-graph
retrieval-augmented generation: it extracts atomic claims from retrieved text,
links them with typed relations, propagates belief, resolves conflicts, selects a
compact evidence subgraph, and generates a cited answer. Core is pure Python and
offline; model-backed components are optional extras. ~15k LOC of source across a
layered `domain / adapters / application / cli` structure, 60 test files, and a
strict five-gate quality bar.

A prior session had already performed a careful, documented cleanup of this copy
(rewritten README, `CLAUDE.md` removed and folded into `CONTRIBUTING.md`/docs,
supporting docs created, scratch logs removed). This audit was therefore a
**delta**: verify that work and finish the remainder. The heavy lifting was
already done and was preserved.

## Files deleted

Only regenerable, non-authored junk was deleted:

| Path | Reason |
|------|--------|
| `.DS_Store` | macOS Finder metadata (git-ignored). |
| `artifacts/.DS_Store` | macOS Finder metadata (git-ignored). |
| `artifacts/benchmark-pipeline/cache/` | Empty generated-cache directory. |

Nothing containing project content was deleted.

## Files archived (moved to `docs/archive/`, not deleted)

Per the "archive when uncertain" policy (there is no backup and no git history):

| From | To | Reason |
|------|----|--------|
| `docs/repository-cleanup-report.md` | `docs/archive/` | Process note about the earlier cleanup, not project documentation. |
| `docs/readme-style-audit.md` | `docs/archive/` | Process note. |
| `docs/documentation-validation.md` | `docs/archive/` | Process note. |
| 32 × `artifacts/.../* 2.{json,log,jsonl}` | `docs/archive/icloud-conflict-copies/` (paths preserved) | macOS iCloud conflict duplicates. They differ from the originals by a few bytes (divergent sync copies), so they were archived rather than deleted; the authoritative run outputs remain under `artifacts/`. |

A copy of the previous README was preserved as `docs/archive/README.previous.md`
before the rewrite. `docs/archive/README.md` documents the archive contents.

## Files renamed

None (the archive moves above are the only relocations).

## Documentation

- **README.md — rewritten from scratch.** The new README is derived entirely from
  the code (CLI commands, `answer_query`/`run_pipeline` signatures, `RelationType`
  values, config schema sections, experiment variants, extras) and organized into
  the full section set: overview, motivation, architecture, installation,
  dependencies, quick start, CLI, Python API, relations, repository structure,
  configuration, running experiments, reproducing results, development workflow,
  testing, limitations, citation, license. Checked to contain no marketing
  vocabulary and no invented features; every referenced doc and artifact path was
  verified to exist.
- **Other docs — reviewed, left as-is.** The `docs/` set had already been cleaned
  and is internally consistent. No machine-specific paths, assistant references,
  duplication, or contradictions were found in the retained docs (only the
  archived process notes referenced the old `CLAUDE.md`). No merges were needed.
- **Added:** `LICENSE` (MIT, 2026 Nikolaos Roufas — `pyproject.toml` already
  declared MIT but the file was missing), `docs/repository_inventory.md` (Phase 1),
  `docs/repository_audit.md` (this file), `docs/archive/README.md`.

## Dead code removed

None. The source already passes strict `ruff` and `mypy`; scans found no
`TODO`/`FIXME`, no commented-out code, no stray debug prints, and no unused
imports. No Python source file was modified in this audit, so the code gates are
unchanged from their prior known-good state.

## Validation results

Run in `~/dev/EGRAG`:

| Gate | Result |
|------|--------|
| `uv run ruff format --check .` | 170 files already formatted — pass |
| `uv run ruff check .` | All checks passed |
| `uv run mypy src` | Success: no issues in 104 source files |
| `uv run pytest` | 441 passed, 7 skipped, 2 warnings — 87% coverage |
| `uv build` | Built `egrag-0.1.0.tar.gz` and `egrag-0.1.0-py3-none-any.whl` |

## Remaining technical debt / recommended follow-ups

Not done here, either because they need your input or fall outside a safe
non-destructive audit:

1. **`pyproject.toml` URLs** still point to `https://example.invalid/egrag`
   placeholders. Update once the public repository URL exists.
2. **Audit deliverables in `docs/`.** `docs/repository_inventory.md` and
   `docs/repository_audit.md` are themselves process documents. For consistency
   with the archived meta-docs, consider moving them to `docs/archive/` or
   removing them before publishing.
3. **`docs/archive/` ships in the sdist.** The built source distribution includes
   `docs/archive/` (process notes + conflict copies). Optionally exclude it via a
   hatchling build setting before release.
4. **Three parallel results trees** (`artifacts/paper-results`,
   `paper-results-repaired`, `paper-results-invalidated`) were retained per your
   decision. A future consolidation or a top-level `artifacts/README.md`
   explaining their relationship would reduce reader confusion.
5. **No final benchmark matrix** — a documented limitation, not a defect. Answer-
   quality numbers require a GPU run with a larger model.

## Environment and workspace (important)

The audit was invoked in `~/Documents/Coding/EGRAG`, which is inside macOS iCloud
Drive with "Optimize Mac Storage" active on a disk that is ~100% full. Every
source file there was **dataless** (evicted; non-zero size but `blocks=0`, reads
empty or time out); iCloud returned hard errors
(`FileProviderInternalErrorDomain Code=12`) and files re-evicted as fast as they
downloaded, so that copy could not be read or audited. An early copy attempt out
of iCloud produced truncated/empty files, confirming the source was unreadable;
that failed copy was removed.

The audited copy is `~/dev/EGRAG` — a fully materialized, git-tracked copy a prior
session had relocated out of iCloud specifically to avoid this failure mode, and
which contains newer content than the stale iCloud copy. **Work on `~/dev/EGRAG`
going forward.** Recommended user actions before publishing:

- Free several GB of disk (the `~/Library/Caches` and `~/.cache` trees hold the
  bulk; the latter likely includes the Hugging Face model cache) so validation
  and normal work are not fighting the eviction/low-space thrash.
- Once `~/dev/EGRAG` is confirmed canonical, delete the stale
  `~/Documents/Coding/EGRAG` copy to prevent future confusion.

## Safety notes

No commits or pushes were made (per your instruction). No algorithms were
rewritten. No source, tests, benchmark results, or failure records were deleted —
only macOS junk was removed, and everything else uncertain was archived rather
than deleted.
