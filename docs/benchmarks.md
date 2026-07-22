# Benchmarks

Two datasets are wired in `egrag.experiments.benchmarks`. Both read locally cached
files and never download or redistribute a corpus. Gold answers and gold evidence
live only on the dataset example and are never passed into the pipeline.

## FEVER

`copenlu/fever_gold_evidence`, read from the cached JSONL. This is the
gold-evidence setting: the evidence sentences for each claim are supplied inline,
so there are no distractors and retrieval is not exercised. The `valid` split has
15,935 rows. A small number of examples ship an evidence row with empty sentence
text; the adapter skips the empty span, records the count in metadata, and does not
change the label. Because the documents are the gold sentences, this benchmark does
not discriminate the graph mechanisms and is not a substitute for open retrieval.

## HotpotQA

`hotpotqa/hotpot_qa`, fullwiki validation, read from parquet (needs the
`benchmarks` extra for pyarrow). The validation split has 7,405 rows. In the
fullwiki setting the per-example context is retrieved paragraphs, so the gold
supporting paragraphs are not guaranteed to be present: only about 28% of examples
have all gold pages in context. Answer metrics (exact match, token F1) use the full
set; supporting-fact, citation, and bridge-connectivity metrics are proxy
measurements reported on the full-coverage subset and labelled as such. For an
evidence-grounded evaluation the distractor split (gold paragraphs guaranteed) is
preferable and would need to be downloaded.

## Adapter behavior

Each adapter produces the harness's `DatasetExample` (`egrag.experiments.models`)
with an ID, question/claim, documents, gold answers, and gold evidence.
`dataset_fingerprint` gives a stable hash over example IDs and questions;
`validate_benchmark` reports duplicate IDs, missing gold answers, and
support/refute examples without evidence pages. Dataset fingerprints and file
hashes are recorded in `artifacts/benchmark-calibration/dataset-manifests/`.

## Metrics

`egrag.experiments.benchmark_metrics` implements exact match, token F1,
supporting-fact precision/recall/F1, the HotpotQA joint score, FEVER label
accuracy, FEVER evidence precision/recall, evidence-set recovery, and a FEVER-style
score. `egrag.experiments.metrics` adds citation precision/recall/completeness,
evidence precision/recall, invalid-citation counts, and clearly-labelled heuristic
measures.

## Splits and manifests

Development samples are fixed before any system runs and are stored as manifests
(example IDs, seed, label/type distribution, fingerprint) under
`artifacts/benchmark-calibration/samples/`: `fever-dev-100`, `fever-smoke-25`,
`hotpot-dev-100`, `hotpot-smoke-25`. Examples are selected by data availability
(for example, gold-coverage stratification), never by model success.

## Kinds of run

Keep these distinct:

- **Bounded smoke** — a handful of controlled examples to check that adapters
  produce valid, grounded, deterministic output.
- **Deterministic structural pilot** — the fake generator over a dev sample; it
  measures evidence selection and graph structure, not answer quality.
- **Development calibration** — choosing settings on development data only; the
  results are under `artifacts/benchmark-calibration/`.
- **Final benchmark run** — real models over the frozen configuration, on a GPU.
  This has not been run.
