#!/usr/bin/env bash
# #34: is the k=16 -> 48 gain from upper levels (where beam errors compound, cheap: summaries
# only) or from the leaf level (raw KV)? oc_L1 at 8K with the leaf k fixed at 16 and larger k
# above (--k is level 0 first, the last repeats). Compare with route_budget8k_k16 / _k48.
# Eval only, one saved result per setting (preemption-safe). Run under nebius_supervise.sh.
cd "$(dirname "$0")/.."
source /data/venv/bin/activate
export HF_HOME=/data/hf PYTHONPATH="$(pwd)" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs results
for ks in "16 48" "16 32"; do
  name=levelk8k_${ks// /_}
  [ -f results/$name.jsonl ] && continue
  rm -f results/$name.tmp
  python experiments/select_gate.py --ckpts runs/oc_L1/adapters.pt --k $ks --raw_window 2048 \
    --coarse --modes decode all --out results/$name.tmp >> logs/levelk8k.log 2>&1 || exit $?
  mv results/$name.tmp results/$name.jsonl
done
echo "=== LEVELK8K DONE $(date +%T)"
