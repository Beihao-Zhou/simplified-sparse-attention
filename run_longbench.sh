#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# Autoresearch knobs (env-controlled, don't touch the python invocation):
#   ATTN_IMPL=ref|sparse          which attention to time / verify
#   MAX_SAMPLES=5             number of LongBench samples to run
export ATTN_IMPL="${ATTN_IMPL:-ref}"
export MAX_SAMPLES="${MAX_SAMPLES:-1}"


select=1
chunk=(4 4)
ckpt_path="gist-sparse-attention/GSA-FT-Qwen2-7B-Instruct-chunk${chunk[0]}-chunk${chunk[1]}"
dataset="triviaqa"
name="${NAME:-test}"

extra_arg=""
if [ "$select" -eq 1 ]; then
  extra_arg="--selective"
fi

echo "Running evaluation with the following settings:"
echo "Checkpoint Path: $ckpt_path"
echo "Name: $name"
echo "Chunk Sizes: ${chunk[@]}"
echo "Selective: $select"
echo "Dataset: $dataset"
echo "ATTN_IMPL: $ATTN_IMPL  MAX_SAMPLES: $MAX_SAMPLES"

rm -rf "${REPO_ROOT}/pred"

python benchmark/Longbench_exp/longbench_pred_qwen.py \
  --model qwen2-7b-instruct \
  --datasets $dataset \
  --ckpt_path ${ckpt_path} \
  --name "$name" \
  $extra_arg \
  --chunk_size ${chunk[@]} \
  --meta_gist 1
