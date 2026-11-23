#!/usr/bin/env bash
# #29: V1-V3 (experiments/diverge_run.sh) did not recover, so no single parameter group runs
# away. Hypothesis: with --down hier each layer's teachers are computed on the hidden states
# the students passed down, so the target moves with the parameters and the update is a
# semi-gradient (as in TD learning, which can diverge). W1: from bis_Y1's step-70
# checkpoint, --down raw (teachers and students both read the raw stream) to step 90.
cd "$(dirname "$0")/.."
source /data/venv/bin/activate
export HF_HOME=/data/hf PYTHONPATH="$(pwd)" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=/data/agent_data
C="--data $A --heldout $A --levels 4 --lora gist --beta --lambda_route 0.1 --raw_window 8192
   --window_turns 2 --eval_rows 4 --seed 0 --seq_len 32768 --batch 2 --steps 300 --stop_at 90
   --eval_every 10 --eval_len 8192 --save_every 10 --resume"
name=bis_W1
if [ ! -f runs/$name/ckpt.pt ]; then
  mkdir -p runs/$name && cp runs/bis_Y1/ckpt.pt runs/bis_Y1/log.jsonl runs/$name/
fi
[ -f runs/$name/adapters.pt ] || python gist_stage1_trainer.py $C --out runs/$name --down raw >> logs/$name.log 2>&1
echo "=== DIVERGE2 DONE $(date +%T)"
