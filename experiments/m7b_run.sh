#!/usr/bin/env bash
# #22: the pipeline on Qwen2.5-7B-Instruct-1M on one H200. Stage 1 at L = 4, 32K, selection
# at every position (ADR 0003), 20 steps (time and memory, and an adapter for the rest);
# 256K append prefill + decode; the full-attention teacher at 128K / 256K; OPSD at 128K for
# 3 steps. Each part writes its own log; resumable under experiments/nebius_supervise.sh.
cd "$(dirname "$0")/.."
source /data/venv/bin/activate
export HF_HOME=/data/hf PYTHONPATH="$(pwd)" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# head dim 128: the gist mask_mod inlined into flex's backward needs 312 KB of shared memory
# with the default tiles (an H200 has 227 KB)
export GIST_FLEX_KERNEL_OPTIONS=${GIST_FLEX_KERNEL_OPTIONS:-'{"BLOCK_M":64,"BLOCK_N":64,"BLOCK_M1":32,"BLOCK_N1":64,"BLOCK_M2":64,"BLOCK_N2":32,"num_stages":1}'}
M=${MODEL:-/data/models/Qwen2.5-7B-Instruct-1M}
A=/data/agent_data E=/data/agent_longeval/len1048576  # len262144 rows are all < 256K
mkdir -p logs runs results
run() { name=$1; shift
  [ -f logs/$name.done ] && return
  echo "=== $name $(date +%T)"
  "$@" >> logs/$name.log 2>&1 || exit $?
  touch logs/$name.done
}
run m7b_s1 python gist_stage1_trainer.py --model $M --data $A --heldout $A --levels 4 \
  --lora gist --beta --lambda_route 0.1 --raw_window 8192 --window_turns 2 --seq_len 32768 \
  --batch 2 --steps 20 --eval_every 1000 --eval_rows 4 --eval_len 32768 --seed 0 \
  --save_every 5 --resume --select_train --select_k 16 --out runs/m7b_s1
run m7b_decode python experiments/decode_speed.py --model $M --stage1 runs/m7b_s1/adapters.pt \
  --data $E --lens 131072 262144
run m7b_teacher python experiments/teacher_score.py --model $M --data $E --lens 131072 262144
run m7b_opsd python gist_opsd_agent.py --model $M --stage1 runs/m7b_s1/adapters.pt --data $A \
  --ctx_min 100000 --ctx_max 131072 --batch 1 --steps 3 --save_every 1 --resume \
  --out runs/m7b_opsd
echo "=== M7B DONE $(date +%T)"
