#!/usr/bin/env bash
# #28 gate (ADR 0002): stage 1 at L = 4, 8K, without (A) and with (B) selection at every
# position, then experiments/select_gate.py. Resumable; also keeps the agent-data
# conversion going in the background. Run under experiments/nebius_supervise.sh.
cd "$(dirname "$0")/.."
source /data/venv/bin/activate
export HF_HOME=/data/hf PYTHONPATH="$(pwd)" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs runs results
if ! grep -q "AGENT DATA DONE" logs/agent_data.log 2>/dev/null && ! pgrep -f '^bash experiments/agent_data_run.sh' >/dev/null; then
  nohup bash experiments/agent_data_run.sh >> logs/agent_data.log 2>&1 < /dev/null &
fi
D=~/spike_data
COMMON="--data $D/cpt_train --heldout $D/cpt_heldout --levels 4 --lora gist --beta --lambda_route 0.1
        --seq_len 8192 --batch 8 --steps 300 --eval_every 100 --eval_rows 8 --eval_len 8192 --seed 0 --resume"
train() { name=$1; shift
  [ -f runs/$name/adapters.pt ] && return
  echo "=== $name $(date +%T)"
  python gist_stage1_trainer.py $COMMON --out runs/$name "$@" >> logs/$name.log 2>&1 || exit $?
}
train gate_A
train gate_B --select_train --select_k 16
echo "=== gate eval $(date +%T)"
python experiments/select_gate.py --ckpts runs/gate_A/adapters.pt runs/gate_B/adapters.pt \
  --k 16 --out results/select_gate.jsonl >> logs/select_gate.log 2>&1 || exit $?
while pgrep -f '^bash experiments/agent_data_run.sh' >/dev/null; do sleep 60; done
echo "=== GATE DONE $(date +%T)"
