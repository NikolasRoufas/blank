# Dataset Validation — Benchmark Calibration

Offline integrity checks on the locally cached benchmark sources. No corpora are
redistributed or committed; only manifests (hashes, counts, fingerprints) live
under `dataset-manifests/`.

## FEVER (`copenlu/fever_gold_evidence`, gold-evidence setting)

- Source: cached `valid.jsonl` (development split), snapshot
  `a6b8d891d393e97a4efac791afffb2d7de5e57c6`.
- Rows: **15,935** — REFUTES 4,887 / SUPPORTS 4,638 / NOT ENOUGH INFO 6,410.
- Dataset fingerprint (sha256 over ids+claims): `978f5e8b26b2…`.
- File sha256 + size recorded in `dataset-manifests/fever.json`.

Checks (via `validate_benchmark` + manifest builder):

| Check | Result |
|-------|--------|
| Duplicate example IDs | none |
| Malformed records (unknown label, bad JSON) | none after empty-span handling |
| Missing gold labels | none |
| SUPPORTS/REFUTES without evidence pages | none |
| Empty source IDs | none |
| Gold label string leaking into the claim text the pipeline sees | **0** |
| Alternative / multi-sentence evidence preserved | yes (per-page facts kept) |
| Train/dev/test separation | calibration uses **valid** only; `test.jsonl` untouched |

**Empty-evidence handling (fix this milestone):** 62 of 15,935 examples contain
an evidence row whose sentence text is empty. The adapter now **skips the empty
span** (rather than emitting an invalid empty `Document` or aborting the load)
and records `num_evidence_skipped_empty` in example metadata. Regression test:
`tests/integration/test_benchmark_calibration.py::test_fever_skips_empty_evidence_sentences`.

**Gold isolation:** the pipeline consumes only `question` (the claim) and
`documents`. Gold label (`gold_answers`) and gold stance/pages (`gold_evidence`)
are separate fields and are never concatenated into pipeline inputs; the leakage
counter above is 0.

## HotpotQA (`hotpotqa/hotpot_qa`, fullwiki)

- Source: cached `fullwiki/validation-00000-of-00001.parquet`, **28,041,820 bytes**,
  sha256 `78933c0a31a5f7b4…` (recorded in `dataset-manifests/hotpotqa.json`).
- **Row count / per-example validation: DEFERRED.** The split is cached only as
  parquet and no parquet reader (`pyarrow`) is installed; it cannot be installed
  offline. Reading rows requires the owner action below.
- The canonical `_parse` mapping (titles, sentence identity, supporting facts,
  duplicate-title dedup, yes/no answers, malformed-row rejection) is validated
  offline with synthetic rows in
  `tests/integration/test_benchmark_calibration.py`.

**Owner action to unblock live HotpotQA:** with network access run
`uv pip install pyarrow` (or `uv sync --extra benchmarks`), then re-run the
manifest/sample builder to populate row count, label/type distribution, and the
stratified dev sample.

## Stop-condition status

Dataset integrity did **not** fail. FEVER is fully validated and ready for
development calibration. HotpotQA file identity is verified; its row-level
validation and sampling are blocked on the parquet-reader owner action and are
reported as such (not fabricated).
