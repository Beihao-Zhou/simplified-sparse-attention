#!/usr/bin/env bash
# #34, two checks on oc_L1 (8K; learned beam 0.75, fit1 0.95, oracle k=16 near the teacher):
# 1. route_amortize with 4x steps and 3x lr: was the separate key's 0.79 undertrained?
# 2. the selection budget: does a larger k make up for the router's ranking? select_gate at
#    k 16 24 32 48 (decode and all). Eval only. Run under experiments/nebius_supervise.sh.
cd "$(dirname "$0")/.."
source /data/venv/bin/activate
export HF_HOME=/data/hf PYTHONPATH="$(pwd)" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs results
[ -f results/route_amortize_oc_L1_long.done ] || {
  python experiments/route_amortize.py --ckpt runs/oc_L1/adapters.pt --raw_window 2048 \
    --steps 6000 --lr 3e-3 --out results/route_amortize_oc_L1_long.jsonl \
    > logs/route_amortize_long.log 2>&1 || exit $?
  touch results/route_amortize_oc_L1_long.done; }
python experiments/select_gate.py --ckpts runs/oc_L1/adapters.pt --k 16 24 32 48 \
  --raw_window 2048 --coarse --modes decode all --out results/route_budget8k.jsonl \
  >> logs/route_budget8k.log 2>&1 || exit $?
echo "=== ROUTEBUDGET DONE $(date +%T)"
