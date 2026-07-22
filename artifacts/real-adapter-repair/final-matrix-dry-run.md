# Final-matrix dry-run (real-adapter-repair §11)

Command (dry-run only; execution disabled this milestone):
```
uv run egrag experiment matrix --benchmark <fever|hotpotqa> --dry-run \
  --sample artifacts/benchmark-calibration/samples/<manifest>.json \
  --output-dir artifacts/final-matrix/out --device auto --cache-dir .egrag-cache
```

`--execute` is intentionally refused:
```
error: matrix execution is disabled in the real-adapter-repair milestone; use --dry-run (run the matrix in the dedicated final-experiment milestone)
```

## FEVER (fever-dev-100)
```json
{
  "mode": "DRY-RUN (no inference executed)",
  "benchmark": "fever",
  "split": "validation",
  "frozen_config_path": "artifacts/benchmark-calibration/frozen-configs/fever.yaml",
  "frozen_config_found": true,
  "dataset_fingerprint": "978f5e8b26b20f13a21903db658fc9ee4d99c4752eae18b87c19e48069d2207f",
  "sample_manifest": "artifacts/benchmark-calibration/samples/fever-dev-100.json",
  "sample_count": 100,
  "seed": 0,
  "device": "auto",
  "resume": false,
  "models": {
    "extractor": "Qwen/Qwen2.5-0.5B-Instruct",
    "nli": {
      "model": "roberta-large-mnli",
      "revision": "2a8f12d27941090092df78e4ba6f0928eb5eac98"
    },
    "generator": "Qwen/Qwen2.5-0.5B-Instruct"
  },
```

## HotpotQA (hotpot-dev-100)
```json
{
  "mode": "DRY-RUN (no inference executed)",
  "benchmark": "hotpotqa",
  "split": "validation",
  "frozen_config_path": "artifacts/benchmark-calibration/frozen-configs/hotpotqa.yaml",
  "frozen_config_found": true,
  "dataset_fingerprint": "d126f3e43c3bfea9cc969179f62fcbb6032031786f5dbc15b4e3ac21bf5e1d2c",
  "sample_manifest": "artifacts/benchmark-calibration/samples/hotpot-dev-100.json",
  "sample_count": 100,
  "seed": 0,
  "device": "auto",
  "resume": false,
  "models": {
    "extractor": "Qwen/Qwen2.5-0.5B-Instruct",
    "nli": {
      "model": "roberta-large-mnli",
      "revision": "2a8f12d27941090092df78e4ba6f0928eb5eac98"
    },
    "generator": "Qwen/Qwen2.5-0.5B-Instruct"
  },
```
