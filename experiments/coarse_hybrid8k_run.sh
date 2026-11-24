#!/usr/bin/env bash
# #41 on coarse terms: R2 (coarse, down=raw) with --down hybrid, one change. #41 measured
# hybrid on SG+SR only, and gate32h changes coarse and hybrid at once; this separates them.
# Compare to r8_R2: suffix KL 0.043, decode KL 0.203, passkey 0 / 0.40, beam mass 0.52.
# Starts once route_detach_run.sh's seed runs are done (their GPU share). Resumable; run
# under experiments/nebius_supervise.sh.
cd "$(dirname "$0")/.."
source /data/venv/bin/activate
export HF_HOME=/data/hf PYTHONPATH="$(pwd)" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs runs results
until [ -f runs/rd_R1s1/adapters.pt ] && [ -f runs/rd_Hs1/adapters.pt ]; do sleep 300; done
D=~/spike_data
if [ ! -f runs/ch8_R2H/adapters.pt ]; then
  echo "=== ch8_R2H $(date +%T)"
  python gist_stage1_trainer.py --data $D/cpt_train --heldout $D/cpt_heldout --levels 4 \
    --lora gist --beta --lambda_route 0.1 --seq_len 8192 --batch 8 --steps 300 \
    --eval_every 50 --eval_rows 8 --eval_len 8192 --seed 0 --raw_window 2048 --select_train \
    --select_k 16 --save_every 10 --resume --stop_at 150 --select_coarse --down hybrid \
    --out runs/ch8_R2H >> logs/ch8_R2H.log 2>&1 || exit $?
fi
python experiments/select_gate.py --ckpts runs/ch8_R2H/adapters.pt --k 16 --raw_window 2048 \
  --coarse --out results/coarse_hybrid8k.jsonl >> logs/coarse_hybrid8k_eval.log 2>&1 || exit $?
[ -f results/coarse_hybrid8k_fit.done ] || {
  python experiments/route_fit.py --ckpt runs/ch8_R2H/adapters.pt --raw_window 2048 \
    --out results/coarse_hybrid8k_fit.jsonl >> logs/coarse_hybrid8k_eval.log 2>&1 || exit $?
  touch results/coarse_hybrid8k_fit.done; }
echo "=== CH8 DONE $(date +%T)"
