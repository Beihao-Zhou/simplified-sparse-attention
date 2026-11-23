#!/usr/bin/env bash
# #34: stage 1 at 8K (B: selection at every position, --down raw), routing loss and visible
# rule. A raw window of 2048 forces the last quarter of the context, as the agent setting's
# 8K window does at 32K: without it only the previous chunk is forced and --route_free has
# nothing to remove. R1 SG+SR, R2 coarse, R3 coarse + --route_free, R4 R3 with lambda_route
# 1.0 and 128 routing rows; trained side by side, then select_gate (KL, passkey; the visible
# rule each was trained with) and route_fit (learned vs oracle, hits). Resumable; run under
# experiments/nebius_supervise.sh.
cd "$(dirname "$0")/.."
source /data/venv/bin/activate
export HF_HOME=/data/hf PYTHONPATH="$(pwd)" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs runs results
D=~/spike_data
COMMON="--data $D/cpt_train --heldout $D/cpt_heldout --levels 4 --lora gist --beta --lambda_route 0.1
        --seq_len 8192 --batch 8 --steps 300 --eval_every 100 --eval_rows 8 --eval_len 8192 --seed 0
        --raw_window 2048 --select_train --select_k 16 --save_every 10 --resume"
train() { name=$1; shift
  [ -f runs/$name/adapters.pt ] && return
  echo "=== $name $(date +%T)"
  python gist_stage1_trainer.py $COMMON --out runs/$name "$@" >> logs/$name.log 2>&1
}
train r8_R1 &
train r8_R2 --select_coarse &
train r8_R3 --select_coarse --route_free &
train r8_R4 --select_coarse --route_free --lambda_route 1.0 --route_rows 128 &
wait
for r in R1 R2 R3 R4; do [ -f runs/r8_$r/adapters.pt ] || exit 1; done
echo "=== route8k eval $(date +%T)"
G="--k 16 --raw_window 2048"
python experiments/select_gate.py --ckpts runs/r8_R1/adapters.pt $G \
  --out results/route8k_sgsr.jsonl >> logs/route8k_eval.log 2>&1 || exit $?
python experiments/select_gate.py --ckpts runs/r8_R2/adapters.pt runs/r8_R3/adapters.pt \
  runs/r8_R4/adapters.pt $G --coarse --out results/route8k_coarse.jsonl >> logs/route8k_eval.log 2>&1 || exit $?
for r in R1 R2 R3 R4; do
  [ -f results/route8k_fit_$r.done ] && continue
  python experiments/route_fit.py --ckpt runs/r8_$r/adapters.pt --raw_window 2048 \
    --out results/route8k_fit_$r.jsonl >> logs/route8k_eval.log 2>&1 || exit $?
  touch results/route8k_fit_$r.done
done
echo "=== ROUTE8K DONE $(date +%T)"
