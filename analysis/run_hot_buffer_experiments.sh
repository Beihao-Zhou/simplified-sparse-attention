#!/usr/bin/env bash
# Focused acceptance gate for the persistent GPU LRU hot buffer.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PYENV:-/tmp/pyenv457}:${REPO}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/tmp/hfcache}"
cd "$REPO"
R() { echo "=== $* ==="; "$@"; }

echo "##### CUDA correctness + synthetic cold/warm operator #####"
R python -m unittest \
    tests.test_sparse_decode_batch.SparseDecodeBatchTest.test_hot_buffer_reuses_hits_and_evicts_only_stale_tokens
R python benchmark/sparse_decode_operator.py \
    --batch 2 --length 4096 --cap 256 --hot-size 512

echo "##### End-to-end: resident vs hot-buffer offload #####"
OUT=analysis/results/e2e_hot_buffer.csv
rm -f "$OUT"
R python benchmark/e2e_offload.py --workload passkey --n-garbage 18400 \
    --steps 64 --batches 1 2 --hot-buffer-size 4096 \
    --tag pk_6k_hot --out "$OUT"
R python benchmark/e2e_offload.py --workload passkey --n-garbage 218000 \
    --steps 64 --batches 1 --hot-buffer-size 8192 \
    --tag pk_71k_hot --out "$OUT"
R python benchmark/e2e_offload.py --workload prompt_file \
    --prompt-file analysis/prompts/gov_report_115.json --steps 128 --batches 1 \
    --hot-buffer-size 8192 --tag gov_60k_hot --out "$OUT"

echo "##### Sanitizers #####"
R bash analysis/run_sanitizer.sh
echo "ALL DONE"
