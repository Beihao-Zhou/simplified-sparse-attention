#!/usr/bin/env bash
# Is the router what limits gate32h at 32K? Its select_gate eval (decode KL 0.378, passkey 0 /
# 0.33, learned beam 0.44 vs fit1 0.85) rerun with one change: --oracle selects by true
# attention mass. Oracle KL far below 0.378 -> the router is the bottleneck; near it -> the
# summaries are. Eval only. Resumable; run under experiments/nebius_supervise.sh.
cd "$(dirname "$0")/.."
source /data/venv/bin/activate
export HF_HOME=/data/hf PYTHONPATH="$(pwd)" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs results
A=/data/agent_data
python experiments/select_gate.py --ckpts runs/gate32h/adapters.pt --agent $A --len 32768 \
  --raw_window 8192 --window_turns 2 --k 16 --coarse --oracle --modes decode all \
  --out results/oracle32h.jsonl >> logs/oracle32h.log 2>&1 || exit $?
echo "=== ORACLE32H DONE $(date +%T)"
