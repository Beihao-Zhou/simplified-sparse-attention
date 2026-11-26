#!/usr/bin/env bash
# #34: before a separate routing key goes into training, can one projection of the summary's
# hidden state reach fit1 at all? Offline, model frozen, on oc_L1 (8K: learned beam 0.75,
# fit1 0.95). Near fit1 -> build the routing key; near 0.75 -> the hidden state lacks it.
# Run under experiments/nebius_supervise.sh.
cd "$(dirname "$0")/.."
source /data/venv/bin/activate
export HF_HOME=/data/hf PYTHONPATH="$(pwd)" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs results
[ -f results/route_amortize_oc_L1.done ] || {
  python experiments/route_amortize.py --ckpt runs/oc_L1/adapters.pt --raw_window 2048 \
    --out results/route_amortize_oc_L1.jsonl >> logs/route_amortize.log 2>&1 || exit $?
  touch results/route_amortize_oc_L1.done; }
echo "=== ROUTEAMORT DONE $(date +%T)"
