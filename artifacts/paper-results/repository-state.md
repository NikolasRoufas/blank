# Repository State (local evaluation)

**This project was evaluated entirely on a local machine and had NOT been
publicly released** at the time of these experiments. There is no GitHub /
Hugging Face / PyPI / Zenodo presence and no documentation website. Nothing was
pushed, uploaded, or published.

## Version control

- The working directory is **not a Git repository** (`git rev-parse` fails).
- There is therefore **no commit hash, branch, or tag**, and the
  "working-tree-clean" concept does not apply.
- In place of a commit, reproducibility is anchored by a **source-tree
  fingerprint**: a SHA-256 over the path+contents of every `*.py` file under the
  project (excluding `.venv/` and `dist/`). See `environment.json`
  (`source_fingerprint_sha256`). This fingerprint is recorded so the exact source
  state behind these results can be re-verified.

## Environment (see environment.json for the machine-readable record)

- Framework version: `egrag 0.1.0`, schema `1.5.0`.
- Python: 3.12.13 (uv-managed); OS: macOS 15.5 (arm64); CPU: Apple M3, 8 cores;
  memory: 16 GiB; GPU: none used (Apple Silicon integrated GPU, no CUDA).
- Optional dependencies present: only `pyyaml` (core). **Absent**: `numpy`,
  `matplotlib`, `transformers`, `torch`, `httpx`, `sentence-transformers`,
  `networkx`, `rank-bm25`. No network access is assumed or used.

## Pre-experiment quality gates (all passed)

Run via the project toolchain (ruff invoked as `uvx ruff@0.15.18` because the
`.venv` ruff console script cannot load on this host's filesystem; the binary is
the identical pinned version):

| gate | result |
|---|---|
| `ruff format --check .` | PASS (rc 0) |
| `ruff check .` | PASS (rc 0) |
| `mypy src` | PASS (rc 0, 94 files) |
| `pytest` | PASS (rc 0, 0 failures, 88% coverage) |
| `uv build` | PASS (sdist + wheel) |
| core import smoke | PASS |
| CLI smoke (`egrag doctor`) | PASS |
| experiment CLI smoke (`egrag experiment run`) | PASS |

## Working-tree note

Because the tree is not under version control, there is no diff to record. The
source fingerprint in `environment.json` fully identifies the evaluated source.
No source files were modified to produce these experiment results (the harness
and components were already implemented and green before this run).
