#!/usr/bin/env bash
# #34 at 32K (one level more than 8K): gate32h with leaf k 16 and larger k above, plus k 48
# everywhere. k 16 everywhere is the existing gate32h eval (decode 0.379, all 0.359).
# Eval only, one saved result per setting (preemption-safe). Run under nebius_supervise.sh.
cd "$(dirname "$0")/.."
source /data/venv/bin/activate
export HF_HOME=/data/hf PYTHONPATH="$(pwd)" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs results
A=/data/agent_data
for ks in "16 48" "48" "16 32"; do
  name=levelk32k_${ks// /_}
  [ -f results/$name.jsonl ] && continue
  rm -f results/$name.tmp
  python experiments/select_gate.py --ckpts runs/gate32h/adapters.pt --agent $A --len 32768 \
    --raw_window 8192 --window_turns 2 --k $ks --coarse --modes decode all \
    --out results/$name.tmp >> logs/levelk32k.log 2>&1 || exit $?
  mv results/$name.tmp results/$name.jsonl
done
echo "=== LEVELK32K DONE $(date +%T)"
