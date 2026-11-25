#!/usr/bin/env bash
# Is --select_train still worth its 2x step time under down=hybrid? gate32h without it, one
# change (ADR 0003 asks for this 32K A/B; under down=raw it was 0.148 vs 0.139, #29). Scored
# by the same select_gate and route_fit as gate32h: decode KL 0.379, passkey all 0.13 / 0.33,
# learned beam 0.44. Waits for oracle32h_run.sh. Resumable; run under nebius_supervise.sh.
cd "$(dirname "$0")/.."
source /data/venv/bin/activate
export HF_HOME=/data/hf PYTHONPATH="$(pwd)" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs runs results
until grep -q "ORACLE32H DONE" logs/oracle32h_queue.log 2>/dev/null; do sleep 120; done
A=/data/agent_data
if [ ! -f runs/gate32hA/adapters.pt ]; then
  echo "=== gate32hA $(date +%T)"
  python gist_stage1_trainer.py --data $A --heldout $A --levels 4 --lora gist --beta \
    --lambda_route 0.1 --raw_window 8192 --window_turns 2 --seq_len 32768 --batch 2 \
    --steps 300 --eval_every 50 --eval_rows 4 --eval_len 32768 --seed 0 --save_every 10 \
    --resume --stop_at ${STOP_AT:-150} --select_coarse --down hybrid \
    --out runs/gate32hA >> logs/gate32hA.log 2>&1 || exit $?
fi
python experiments/select_gate.py --ckpts runs/gate32hA/adapters.pt --agent $A --len 32768 \
  --raw_window 8192 --window_turns 2 --k 16 --coarse --out results/select_gate32hA.jsonl \
  >> logs/select_gate32hA.log 2>&1 || exit $?
[ -f results/route32k_fit_gate32hA.done ] || {
  python experiments/route_fit.py --ckpt runs/gate32hA/adapters.pt --len 32768 --raw_window 8192 \
    --layers 9 11 13 16 17 --out results/route32k_fit_gate32hA.jsonl >> logs/route32k_fit.log 2>&1 \
    || exit $?
  touch results/route32k_fit_gate32hA.done; }
echo "=== GATE32HA DONE $(date +%T)"
