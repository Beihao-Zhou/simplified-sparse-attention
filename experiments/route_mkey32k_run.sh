#!/usr/bin/env bash
# #34, step A.2: mlp_kids reached heldout beam 0.615 at 32K (learned 0.47, fit1 0.85, fit4 0.91),
# train 0.66, so it underfits. Here: twice the training contexts, a wider MLP, and m keys per
# node (logsumexp, as fit4). Same frozen gate32h and layers. Eval only.
cd "$(dirname "$0")/.."
source /data/venv/bin/activate
export HF_HOME=/data/hf PYTHONPATH="$(pwd)" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs results
python experiments/route_amortize.py --ckpt runs/gate32h/adapters.pt --len 32768 \
  --raw_window 8192 --layers 9 11 13 16 17 --train_reps 24 --test_reps 2 \
  --routers mlp_kids_r256 mlp_kids_r512 mlp_kids_r256_m2 mlp_kids_r256_m4 \
  --steps 5000 --lr 1e-3 --wd 0.01 \
  --out results/route_mkey32k.jsonl > logs/route_mkey32k.log 2>&1 || exit $?
echo "=== ROUTEMKEY32K DONE $(date +%T)"
