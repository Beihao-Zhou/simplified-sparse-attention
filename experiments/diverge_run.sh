#!/usr/bin/env bash
# #29: the 32K agent-data divergence (bis_Y1 drifts from step ~57, 8K suffix KL 0.06 -> 1.3
# by step 80). From its step-70 checkpoint, to step 90: V1 gist embeddings frozen, V2 beta
# frozen, V3 LR x 0.3. Whichever recovers names the parameter group that runs away.
cd "$(dirname "$0")/.."
source /data/venv/bin/activate
export HF_HOME=/data/hf PYTHONPATH="$(pwd)" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=/data/agent_data
C="--data $A --heldout $A --levels 4 --lora gist --beta --lambda_route 0.1 --raw_window 8192
   --window_turns 2 --eval_rows 4 --seed 0 --seq_len 32768 --batch 2 --steps 300 --stop_at 90
   --eval_every 10 --eval_len 8192 --save_every 10 --resume"
v() { name=$1; shift
  [ -f runs/$name/adapters.pt ] && return
  if [ ! -f runs/$name/ckpt.pt ]; then
    mkdir -p runs/$name && cp runs/bis_Y1/ckpt.pt runs/bis_Y1/log.jsonl runs/$name/
  fi
  python gist_stage1_trainer.py $C --out runs/$name "$@" >> logs/$name.log 2>&1
}
v bis_V1 --freeze embed &
v bis_V2 --freeze beta &
v bis_V3 --lr_mult 0.3 &
wait
echo "=== DIVERGE DONE $(date +%T)"
