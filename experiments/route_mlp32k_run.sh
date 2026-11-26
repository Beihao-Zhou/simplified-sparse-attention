#!/usr/bin/env bash
# #34, step A: is the router's gap at 32K (learned beam 0.44, fit1 0.85) the linear form or the
# information in x? One routing key per layer and KV group on gate32h, trained offline on the
# true mass (route_amortize.py): linear (the 8K ceiling ~0.8), an MLP on x, and an MLP on x
# plus the mean x of each node's children. Layers as route_fit's 32K fit. Eval only.
cd "$(dirname "$0")/.."
source /data/venv/bin/activate
export HF_HOME=/data/hf PYTHONPATH="$(pwd)" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs results
python experiments/route_amortize.py --ckpt runs/gate32h/adapters.pt --len 32768 \
  --raw_window 8192 --layers 9 11 13 16 17 --train_reps 12 --test_reps 2 \
  --routers linear_r64 mlp_r256 mlp_kids_r256 --steps 3000 --lr 1e-3 --wd 0.01 \
  --out results/route_mlp32k.jsonl > logs/route_mlp32k.log 2>&1 || exit $?
echo "=== ROUTEMLP32K DONE $(date +%T)"
