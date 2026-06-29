#!/usr/bin/env bash
# Clean end-to-end REAL benchmark with the OPTIMIZED sparse attention ONLY.
#
# Runs the SSA model with: compact sparse flash-decode + block-sparse flex prefill,
# ONE generation per sample (no reference / no verification scaffolding), saves
# predictions, then scores them with the official LongBench evaluator.
#
# Usage:
#   bash run_real_benchmark.sh                 # full triviaqa (200 samples)
#   NSAMPLES=50 bash run_real_benchmark.sh     # first 50 samples
#   DATASET=triviaqa bash run_real_benchmark.sh
#
# Output: per-sample latency, mean / median / steady-state (excl. cold sample 0)
#         prefill_ms & tpot_ms, and the LongBench score (first-line F1).
#
# NOTE: the first sample pays a one-time flex torch.compile (~10-20s) + CUDA
# extension build (~30s, cached after). Set TORCHINDUCTOR_CACHE_DIR to persist
# the flex compile across launches.

set -e
cd "$(dirname "$0")"

export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- the optimized sparse attention (only) ----
export ATTN_IMPL=sparse                     # CUDA decode + flex/dense prefill (eval path)
# NB: the harness always runs ONE optimized generation per sample, and compact
# is the only decode kernel, so DECODE_KERNEL / PROD_MODE are no longer knobs.
# Selection mode (accuracy/speed trade-off):
#   head (default) = per-head top-k, then UNION within the GQA group. This is the
#                    paper's grouped unfolding (§3.3); allclose-EXACT to the eager
#                    reference (reproduces the reported accuracy).
#   group          = SUM the group heads' gist scores, one top-k per group. A
#                    DIFFERENT (cheaper) selection: F1-validated, NOT allclose. Pair
#                    with DECODE_COMPILE=1 DECODE_GMAX_BUCKET=1 for the ~18ms tpot floor.
# Uncompiled, head and group have the same ~25ms tpot; group only pulls ahead once compiled.
export DECODE_SELECT="${DECODE_SELECT:-head}"

# ---- OPTIONAL faster decode via torch.compile (gated, default off) ----
# All F1-validated (LongBench triviaqa 88-91). Pick ONE:
#   (a) flex-compatible, ~21ms decode + fast prefill (recommended faster default):
#       DECODE_COMPILE=1 DECODE_COMPILE_MODE=default DECODE_GMAX_BUCKET=1 bash run_real_benchmark.sh
#   (b) FASTEST decode ~18ms (robust, recompiles~0) — needs DENSE prefill (reduce-overhead
#       CUDA graphs conflict with flex's torch.compile in PyTorch's cudagraph_trees):
#       DECODE_COMPILE=1 DECODE_GMAX_BUCKET=1 PREFILL_KERNEL=dense bash run_real_benchmark.sh
# DECODE_GMAX_BUCKET pads Gmax+kk to a grid so the CUDA graph is reused across lengths
# (robust varying-length). Leave DECODE_COMPILE unset for the zero-warmup ~24ms group path.
export DECODE_COMPILE="${DECODE_COMPILE:-0}"
export PREFILL_KERNEL="${PREFILL_KERNEL:-flex}"   # honor an override (dense for fastest decode)

export MAX_SAMPLES="${NSAMPLES:-200}"    # full triviaqa = 200
DATASET="${DATASET:-triviaqa}"
chunk=(4 4)
ckpt="gist-sparse-attention/GSA-FT-Qwen2-7B-Instruct-chunk${chunk[0]}-chunk${chunk[1]}"
NAME="optimized"

# fresh prediction dir for this run
rm -rf "pred/qwen2-7b-instruct/${NAME}"

echo ">>> Generating predictions with OPTIMIZED sparse attention (compact decode + flex prefill)..."
python benchmark/Longbench_exp/longbench_pred_qwen.py \
  --model qwen2-7b-instruct \
  --datasets "$DATASET" \
  --ckpt_path "$ckpt" \
  --name "$NAME" \
  --selective \
  --chunk_size ${chunk[@]} \
  --meta_gist 1

echo ">>> Scoring with the official LongBench evaluator (first-line F1)..."
python benchmark/Longbench_exp/longbench_eval.py \
  --model qwen2-7b-instruct \
  --name "$NAME"

echo ">>> LongBench score (first-line F1, x100):"
cat "pred/qwen2-7b-instruct/${NAME}/result.json"; echo
echo ">>> Done. Predictions: pred/qwen2-7b-instruct/${NAME}/${DATASET}.jsonl"
