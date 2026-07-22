# Dataset Validation

## FEVER — `copenlu/fever_gold_evidence` (cached JSONL) — PASS (gold-evidence setting)

- Source: local HF cache `datasets--copenlu--fever_gold_evidence/snapshots/*/{train,valid,test}.jsonl`.
- `valid` split: 15,935 records; labels SUPPORTS 4,638 / REFUTES 4,887 / NOT ENOUGH INFO 6,410.
- Record schema: `claim`, `label`, `evidence` (`[page_id, sentence_index, sentence]`),
  `id`, `verifiable`, `original_id`.
- Adapter: `FeverGoldEvidenceDataset` → canonical `DatasetExample`
  (query = claim; documents = gold evidence sentences; gold label + gold pages
  kept aside; never injected into pipeline text).
- Checks run (`validate_benchmark`): duplicate IDs (none on sampled slices),
  missing gold label (none), SUPPORTS/REFUTES without evidence pages (none),
  empty document source IDs (none).
- **Important setting note:** this is the **gold-evidence** verification setting —
  the candidate documents *are* the gold evidence sentences. It measures claim
  verification given gold evidence, **not** retrieval over full Wikipedia. Some
  `NOT ENOUGH INFO` rows still carry a non-decisive sentence; `available=False`
  marks them and the system is expected to abstain. Full-Wikipedia FEVER (retrieval
  setting) is out of scope for this pilot (no Wikipedia dump cached).

## HotpotQA — `hotpotqa/hotpot_qa` (fullwiki) — BLOCKED (parquet reader unavailable)

- Source: local HF cache, **parquet only** (`fullwiki/{train,validation,test}-*.parquet`).
- Reading parquet requires `pyarrow` / `datasets` / `pandas` — **none installed**,
  and installation needs network (offline policy). `HotpotQADataset.load()` raises
  a typed `MissingDependencyError("pyarrow", "datasets")` with an install hint.
- The adapter's parquet→`DatasetExample` mapping is implemented and unit-reachable;
  it will work once a parquet reader is present.
- **Owner action to unblock:** with network, `uv pip install pyarrow` (or add a
  `datasets`/`pyarrow` extra and `uv sync --extra …`), then re-run
  `egrag dataset validate --benchmark hotpotqa`.

## Decision

- FEVER pilot may proceed (subject to the runtime gate; CPU is slow).
- HotpotQA pilot is **not started**; it is blocked on the parquet reader and is
  reported, not faked. No HotpotQA numbers are produced this milestone.
