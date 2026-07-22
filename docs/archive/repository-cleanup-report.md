# Repository cleanup report

Scope: make the repository read as human-authored research software. Only
development-workflow scratch and assistant-oriented files are removed. No source,
tests, benchmark results, or failure records are deleted.

## Removed

| Path | Reason |
|------|--------|
| `CLAUDE.md` | Assistant-oriented rules file. Its technical content (layering, scientific-integrity, testing, reproducibility conventions) is moved into `CONTRIBUTING.md`, `docs/development.md`, and `docs/architecture.md` in ordinary prose. |
| `docs/documentation-inventory.md` | Working note written while drafting the docs; superseded by `README.md` and the `docs/` pages. |
| `artifacts/benchmark-calibration/environment-recovery/` | Environment-troubleshooting logs (disk/sync recovery), not experimental results. |
| `artifacts/benchmark-calibration/config-recovery/` | Config-recovery logs and a redundant `pyproject.toml` backup; the root `pyproject.toml` is the source of truth. |
| `artifacts/real-adapter-repair/_cache/` | Generated `DiskCacheBackend` files from the end-to-end smoke (regenerated on demand). Now git-ignored. |

## Rewritten

- `README.md` — new entry point (see `docs/readme-style-audit.md`).
- `CONTRIBUTING.md` — drops the `CLAUDE.md` reference; points to `docs/development.md`.
- `docs/architecture.md` — removes the stale absolute path, the `CLAUDE.md`
  reference, and the internal build-order plan; keeps the layered design.
- `docs/experiments.md` — trimmed to the evaluation method actually implemented.

## Created

- `docs/development.md`, `docs/models.md`, `docs/caching.md`,
  `docs/benchmarks.md`, `docs/reproduction.md`.
- `docs/repository-cleanup-report.md`, `docs/readme-style-audit.md`,
  `docs/documentation-validation.md`.

## Retained (technical, kept as-is or lightly edited)

- All source under `src/egrag/` and all `tests/`.
- `docs/reasoning.md`, `docs/score-taxonomy.md`, `docs/bridge-relations.md`,
  `docs/extending.md`, `docs/testing-strategy.md`, `docs/troubleshooting.md`,
  `docs/implementation-status.md`, `docs/adr/` — light edits to remove
  build-order phrasing where present.
- `CHANGELOG.md` — kept as a record.

## Retained artifacts that look generated but are reproducibility records

`artifacts/benchmark-calibration/` and `artifacts/real-adapter-repair/` hold
dataset manifests, sample manifests, frozen configs, smoke outputs, pilot data,
and failure records. Some record the local dataset cache location
(`~/.cache/huggingface/hub/...`) because that is where the inputs were read from;
these are kept unedited so the runs remain auditable. User-facing docs do not
contain machine-specific paths. The reproduction scripts under
`artifacts/**/_scripts/` are kept and referenced from `docs/reproduction.md`.

## Notes

- `.claude/settings.local.json` is local editor/tool configuration, not part of
  the project; it is now git-ignored rather than tracked.
