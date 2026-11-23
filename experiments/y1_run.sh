#!/usr/bin/env bash
# #29 bisection: the 32K agent-data stage-1 run (gate32_A settings) that broke between steps
# 10 and 100, again with an 8K suffix-KL eval every 10 steps, to step 80. Resumable.
cd "$(dirname "$0")/.."
source /data/venv/bin/activate
export HF_HOME=/data/hf PYTHONPATH="$(pwd)" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=/data/agent_data
python gist_stage1_trainer.py --data $A --heldout $A --levels 4 --lora gist --beta \
  --lambda_route 0.1 --raw_window 8192 --window_turns 2 --eval_rows 4 --seed 0 \
  --seq_len 32768 --batch 2 --steps 300 --stop_at 80 --eval_every 10 --eval_len 8192 \
  --save_every 10 --resume --out runs/bis_Y1 >> logs/bis_Y1.log 2>&1 || exit $?
echo "=== Y1 DONE $(date +%T)"
