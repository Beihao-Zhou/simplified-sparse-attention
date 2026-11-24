#!/usr/bin/env bash
# Stage-1 config check at 32K: gate32r_B (selection at every position) with coarse terms
# (#34: R2 best at 8K) and down=hybrid (#41: summaries written as at inference halve the 8K
# KL). Two changes from gate32r_B by design: each was tested alone at 8K; this asks whether
# they hold together at 32K without diverging. Compare to gate32r_B: sel KL 0.139, decode KL
# 0.582, passkey 0 / 0. Resumable; run under experiments/nebius_supervise.sh.
cd "$(dirname "$0")/.."
source /data/venv/bin/activate
export HF_HOME=/data/hf PYTHONPATH="$(pwd)" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs runs results
A=/data/agent_data
if [ ! -f runs/gate32h/adapters.pt ]; then
  echo "=== gate32h $(date +%T)"
  python gist_stage1_trainer.py --data $A --heldout $A --levels 4 --lora gist --beta \
    --lambda_route 0.1 --raw_window 8192 --window_turns 2 --seq_len 32768 --batch 2 \
    --steps 300 --eval_every 50 --eval_rows 4 --eval_len 32768 --seed 0 --save_every 10 \
    --resume --stop_at ${STOP_AT:-150} --select_train --select_k 16 --select_coarse \
    --down hybrid --out runs/gate32h >> logs/gate32h.log 2>&1 || exit $?
fi
python experiments/select_gate.py --ckpts runs/gate32h/adapters.pt --agent $A --len 32768 \
  --raw_window 8192 --window_turns 2 --k 16 --coarse --out results/select_gate32h.jsonl \
  >> logs/select_gate32h.log 2>&1 || exit $?
echo "=== GATE32H DONE $(date +%T)"
