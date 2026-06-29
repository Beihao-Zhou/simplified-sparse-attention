#!/usr/bin/env bash
# Codified correctness guard for the sparse decode/prefill kernels.
#
# Runs the SAME few samples twice under greedy decoding — once on the eager
# reference (ATTN_IMPL=ref) and once on the sparse path with the paper's
# selection (ATTN_IMPL=sparse DECODE_SELECT=head, i.e. per-head top-k unioned
# within the GQA group) — and asserts the generated predictions are IDENTICAL.
#
# head selection reproduces the reference's selected set, and the kernel is
# allclose for a given selection, so greedy argmax must match token-for-token.
# A mismatch means a kernel regression or a selector bug. (group selection is a
# DIFFERENT algorithm — F1-validated, not allclose — so it is NOT checked here.)
#
# Assumes the Python env is already active (module loads + venv), like the other
# run_*.sh scripts. Usage:
#   bash check_correctness.sh                 # 5 samples, triviaqa
#   NSAMPLES=10 DATASET=hotpotqa bash check_correctness.sh
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

N="${NSAMPLES:-5}"
DATASET="${DATASET:-triviaqa}"
chunk=(4 4)
ckpt="${CKPT:-gist-sparse-attention/GSA-FT-Qwen2-7B-Instruct-chunk4-chunk4}"

run() {  # $1=name  $2=ATTN_IMPL  $3=DECODE_SELECT
  rm -rf "pred/qwen2-7b-instruct/$1"
  ATTN_IMPL="$2" DECODE_SELECT="$3" PREFILL_KERNEL="${PREFILL_KERNEL:-flex}" MAX_SAMPLES="$N" \
    python benchmark/Longbench_exp/longbench_pred_qwen.py \
      --model qwen2-7b-instruct --datasets "$DATASET" --ckpt_path "$ckpt" \
      --name "$1" --selective --chunk_size ${chunk[@]} --meta_gist 1 >/dev/null
}

echo ">>> reference (eager)..."          ; run corr_ref  ref    head
echo ">>> sparse (head = paper union)..."; run corr_head sparse head

python - "$DATASET" <<'PY'
import json, sys
ds = sys.argv[1]
def preds(name):
    p = f"pred/qwen2-7b-instruct/{name}/{ds}.jsonl"
    return [json.loads(l)["pred"] for l in open(p)]
ref, head = preds("corr_ref"), preds("corr_head")
assert len(ref) == len(head) and len(ref) > 0, f"sample count mismatch: {len(ref)} vs {len(head)}"
mism = [i for i, (a, b) in enumerate(zip(ref, head)) if a != b]
if mism:
    print(f"FAIL: {len(mism)}/{len(ref)} predictions differ (sparse head != eager). e.g. idx {mism[0]}:")
    print(f"  ref : {ref[mism[0]]!r}")
    print(f"  head: {head[mism[0]]!r}")
    sys.exit(1)
print(f"PASS: all {len(ref)} predictions identical (sparse head == eager reference).")
PY
