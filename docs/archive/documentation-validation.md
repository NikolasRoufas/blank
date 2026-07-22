# Documentation validation

Record of the checks run against the documentation.

## Files created

- `README.md` (rewritten)
- `docs/development.md`, `docs/models.md`, `docs/caching.md`,
  `docs/benchmarks.md`, `docs/reproduction.md`
- `docs/repository-cleanup-report.md`, `docs/readme-style-audit.md`,
  `docs/documentation-validation.md`

## Files rewritten or edited

- `docs/architecture.md` — removed the stale absolute path, the `CLAUDE.md`
  references, the "proposed" tree (replaced with the actual layout), and the
  staged build plan (§11–§12).
- `docs/experiments.md` — corrected the variant table to the ten variants in the
  code (the previous table listed a non-existent `graph_no_bridge`).
- `docs/testing-strategy.md` — reworded §12 to drop the milestone framing and the
  now-dead cross-reference.
- `CONTRIBUTING.md` — removed the `CLAUDE.md` reference; points to
  `docs/development.md`.
- `CHANGELOG.md` — removed "in progress" milestone framing from the intro/heading.
- `.gitignore` — added `.claude/`, `.egrag-cache/`, `artifacts/**/_cache/`.

## Files removed

`CLAUDE.md`, `docs/documentation-inventory.md`,
`artifacts/benchmark-calibration/{environment-recovery,config-recovery}/`,
`artifacts/real-adapter-repair/_cache/` (see `repository-cleanup-report.md`).

## Links and paths

All README links to `docs/` pages resolve. All artifact paths referenced in the
README and docs exist (`samples/fever-dev-100.json`, `samples/hotpot-dev-100.json`,
`frozen-configs/`, `real-adapter-repair/_scripts/smoke.py`,
`real-adapter-repair/final-report.md`). No machine-specific absolute paths appear
in `README.md` or `docs/*.md`. No `Claude`/`ChatGPT` references remain in
user-facing files; the only `CLAUDE` mention is in `repository-cleanup-report.md`,
which documents the removal.

## Commands and CLI flags

CLI commands in the README and docs were checked against `egrag --help`,
`egrag run --help`, and `egrag experiment matrix --help`: `run`, `search`,
`extract`, `graph`, `reason`, `inspect-config`, `doctor`, `gpu-readiness`,
`experiment {run,resume,compare,summarize,inspect-example,matrix}`. The
`matrix --dry-run`/`--execute` behavior and its flags match the implementation;
`--execute` is refused.

## Examples executed

- CLI quick-start `uv run egrag run -q "…"` — runs, prints an answer and a
  citation.
- Python quick-start (`answer_query` + `build_demo_documents`) — runs, returns an
  answer and citation IDs.

## Quality gates

- `ruff format --check .` — pass (170 files)
- `ruff check .` — pass
- `mypy src` — pass (104 files)
- `pytest` — 441 passed, 7 skipped, 0 failed
- `uv build` — sdist + wheel built

## Unsupported claims removed

Superiority and "production-ready" phrasing, implied benchmark results, and
generic adjectives were removed (see `readme-style-audit.md`). The docs state
plainly that no final benchmark matrix has run and that the 0.5B model is not
suitable for evaluation.

## Remaining documentation limitations

- `docs/architecture.md` §13–§16 and several secondary pages
  (`reasoning.md`, `score-taxonomy.md`, `bridge-relations.md`, `extending.md`,
  `troubleshooting.md`, `implementation-status.md`) predate this pass; they were
  checked for stale paths and milestone framing but not fully rewritten.
- Retained `artifacts/` reports record the local dataset cache path
  (`~/.cache/huggingface/hub/...`) where inputs were read; they are kept unedited
  as reproducibility records and are not linked from user-facing docs.
