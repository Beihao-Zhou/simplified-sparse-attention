#!/usr/bin/env bash
# After gate32h (32K, coarse + hybrid): first #35's route_fit on it (does hybrid move the
# learned beam mass at 32K? gate32r_B: 0.26), then gate32h with --route_detach_q, one change,
# run ahead of the 8K detach verdict (route_detach_run.sh) on an otherwise idle GPU; it is
# kept only if that verdict passes. Resumable; run under experiments/nebius_supervise.sh.
cd "$(dirname "$0")/.."
source /data/venv/bin/activate
export HF_HOME=/data/hf PYTHONPATH="$(pwd)" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs runs results
A=/data/agent_data
fit() { [ -f results/route32k_fit_$1.done ] && return
  python experiments/route_fit.py --ckpt runs/$1/adapters.pt --len 32768 --raw_window 8192 \
    --layers 9 11 13 16 17 --out results/route32k_fit_$1.jsonl >> logs/route32k_fit.log 2>&1 \
    || exit $?
  touch results/route32k_fit_$1.done; }
fit gate32h
if [ ! -f runs/gate32hd/adapters.pt ]; then
  echo "=== gate32hd $(date +%T)"
  python gist_stage1_trainer.py --data $A --heldout $A --levels 4 --lora gist --beta \
    --lambda_route 0.1 --raw_window 8192 --window_turns 2 --seq_len 32768 --batch 2 \
    --steps 300 --eval_every 50 --eval_rows 4 --eval_len 32768 --seed 0 --save_every 10 \
    --resume --stop_at ${STOP_AT:-150} --select_train --select_k 16 --select_coarse \
    --down hybrid --route_detach_q --out runs/gate32hd >> logs/gate32hd.log 2>&1 || exit $?
fi
python experiments/select_gate.py --ckpts runs/gate32hd/adapters.pt --agent $A --len 32768 \
  --raw_window 8192 --window_turns 2 --k 16 --coarse --out results/select_gate32hd.jsonl \
  >> logs/select_gate32hd.log 2>&1 || exit $?
fit gate32hd
echo "=== GATE32HD DONE $(date +%T)"
