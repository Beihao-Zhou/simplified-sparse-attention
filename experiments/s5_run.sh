#!/usr/bin/env bash
# S5 (#14) queue: stage-1 (method D) at L = 2..5 plus routing-KL ablations at L = 4, then
# per-level recall at 32K-128K (experiments/deep_recall.py). Idempotent: rerun after a
# preemption and finished stages are skipped, unfinished ones resume from ckpt.pt.
cd "$(dirname "$0")/.."
source /data/venv/bin/activate
export PYTHONPATH="$(pwd):$PYTHONPATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/data/hf
mkdir -p logs runs results
D=~/spike_data
train() { name=$1; shift
  [ -f runs/$name/adapters.pt ] && return
  echo "=== $name $(date +%T)"
  python gist_stage1_trainer.py --data $D/cpt_train --heldout $D/cpt_heldout --lora gist --beta \
    --r 32 --seq_len 16384 --batch 8 --steps 500 --eval_every 100 --eval_rows 8 --eval_len 16384 \
    --resume --out runs/$name "$@" >> logs/$name.log 2>&1 || exit $?
}
for L in 2 3 4 5; do train s5_L$L --levels $L --lambda_route 0.1; done
train s5_L4_route0 --levels 4 --lambda_route 0
train s5_L4_route1 --levels 4 --lambda_route 1.0
echo "=== recall $(date +%T)"
python experiments/deep_recall.py --out results/s5_recall.jsonl --ckpts \
  runs/s5_L2/adapters.pt runs/s5_L3/adapters.pt runs/s5_L4/adapters.pt runs/s5_L5/adapters.pt \
  runs/s5_L4_route0/adapters.pt runs/s5_L4_route1/adapters.pt >> logs/s5_recall.log 2>&1 || exit $?
python experiments/deep_recall.py --summary results/s5_recall.jsonl > results/s5_recall.txt
echo "=== S5 DONE $(date +%T)"
