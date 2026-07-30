#!/bin/bash
# Orchestrates the full real-benchmark Qwen matrix: 3 models x 2 benchmarks x
# 2 token budgets = 12 runs, one model cached at a time (disk-constrained
# sandbox). Not meant to be run outside this session's disk layout as-is;
# see docs/reproduction.md for the general, portable commands.
set -uo pipefail
cd /root/EGRAG/EGRAG
LOG=artifacts/benchmark-matrix/_full_matrix.log
echo "START $(date -u +%FT%TZ)" >> "$LOG"

for MODEL in qwen2.5-3b-instruct qwen2.5-7b-instruct qwen3.5-9b; do
  for BENCH in fever hotpotqa; do
    for MNT in 64 256; do
      TAG=""
      [ "$MNT" = "256" ] && TAG="_mnt256"
      echo "=== $(date -u +%FT%TZ) MODEL=$MODEL BENCH=$BENCH MNT=$MNT ===" | tee -a "$LOG"
      uv run python scripts/run_benchmark_matrix.py \
        --benchmark "$BENCH" --models "$MODEL" \
        --max-new-tokens "$MNT" --tag "$TAG" \
        >> "$LOG" 2>&1
      STATUS=$?
      echo "--- exit=$STATUS ---" | tee -a "$LOG"
      df -h / | tail -1 | tee -a "$LOG"
    done
  done
  echo "=== cleaning HF cache for $MODEL ===" | tee -a "$LOG"
  case "$MODEL" in
    qwen2.5-3b-instruct) rm -rf /workspace/.hf_home/hub/models--Qwen--Qwen2.5-3B-Instruct ;;
    qwen2.5-7b-instruct) rm -rf /workspace/.hf_home/hub/models--Qwen--Qwen2.5-7B-Instruct ;;
    qwen3.5-9b) rm -rf /workspace/.hf_home/hub/models--Qwen--Qwen3.5-9B ;;
  esac
done
echo "DONE $(date -u +%FT%TZ)" >> "$LOG"
