#!/usr/bin/env bash
# #29 (ADR 0003): the #28 gate again at 32K on agent trajectories (cpt_train has almost no
# rows that long): stage 1 at L = 4 without (A) and with (B) selection at every position,
# the agent setting of ADR 0001 (raw window 8K, last 2 turns), batch 2 x 300 steps (the
# tokens per step of the 8K gate's batch 8), then experiments/select_gate.py at 32K.
# Resumable; run under experiments/nebius_supervise.sh. The first runs (gate32_A/B) diverged
# (#29); these use --down raw, and A and B train side by side.
cd "$(dirname "$0")/.."
source /data/venv/bin/activate
export HF_HOME=/data/hf PYTHONPATH="$(pwd)" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs runs results
A=/data/agent_data
COMMON="--data $A --heldout $A --levels 4 --lora gist --beta --lambda_route 0.1
        --raw_window 8192 --window_turns 2 --seq_len 32768 --batch 2 --steps 300
        --eval_every 50 --eval_rows 4 --eval_len 32768 --seed 0 --save_every 10 --resume"
train() { name=$1; shift
  [ -f runs/$name/adapters.pt ] && return
  echo "=== $name $(date +%T)"
  python gist_stage1_trainer.py $COMMON --out runs/$name "$@" >> logs/$name.log 2>&1 || exit $?
}
train gate32r_A &
train gate32r_B --select_train --select_k 16 &
wait
[ -f runs/gate32r_A/adapters.pt ] && [ -f runs/gate32r_B/adapters.pt ] || exit 1
echo "=== gate32 eval $(date +%T)"
python experiments/select_gate.py --ckpts runs/gate32r_A/adapters.pt runs/gate32r_B/adapters.pt \
  --agent $A --len 32768 --raw_window 8192 --window_turns 2 --k 16 \
  --out results/select_gate32r.jsonl >> logs/select_gate32r.log 2>&1 || exit $?
echo "=== GATE32 DONE $(date +%T)"
