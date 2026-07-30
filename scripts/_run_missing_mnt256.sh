#!/bin/bash
# Reruns only the 6 max_new_tokens=256 combinations that failed at argument
# parsing in the first pass (--tag "-mnt256" was misparsed by argparse as a
# flag, not a value, because it starts with '-'; fixed here to "_mnt256").
set -uo pipefail
cd /root/EGRAG/EGRAG
LOG=artifacts/benchmark-matrix/_full_matrix.log
echo "RESTART-MNT256 $(date -u +%FT%TZ)" >> "$LOG"

for MODEL in qwen2.5-3b-instruct qwen2.5-7b-instruct qwen3.5-9b; do
  for BENCH in fever hotpotqa; do
    echo "=== $(date -u +%FT%TZ) MODEL=$MODEL BENCH=$BENCH MNT=256 ===" | tee -a "$LOG"
    uv run python scripts/run_benchmark_matrix.py \
      --benchmark "$BENCH" --models "$MODEL" \
      --max-new-tokens 256 --tag "_mnt256" \
      >> "$LOG" 2>&1
    STATUS=$?
    echo "--- exit=$STATUS ---" | tee -a "$LOG"
    df -h / | tail -1 | tee -a "$LOG"
  done
  echo "=== cleaning HF cache for $MODEL ===" | tee -a "$LOG"
  case "$MODEL" in
    qwen2.5-3b-instruct) rm -rf /workspace/.hf_home/hub/models--Qwen--Qwen2.5-3B-Instruct ;;
    qwen2.5-7b-instruct) rm -rf /workspace/.hf_home/hub/models--Qwen--Qwen2.5-7B-Instruct ;;
    qwen3.5-9b) rm -rf /workspace/.hf_home/hub/models--Qwen--Qwen3.5-9B ;;
  esac
done
echo "DONE-MNT256 $(date -u +%FT%TZ)" >> "$LOG"
