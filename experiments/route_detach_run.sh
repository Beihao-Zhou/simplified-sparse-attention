#!/usr/bin/env bash
# #34 follow-up: the routing KL's true mass comes from the student's own q, so the loss can
# move the target instead of the router (R4: lowest routing loss, beam mass 0.52 -> 0.32).
# R2 (coarse) with --route_detach_q at lambda 0.1 and 1.0, each one change from R2 / R4's
# weight; pass if the learned beam mass (route_fit) clearly beats R2's 0.52 with KL no worse.
# Also #41's halving at a second seed: R1 and H at seed 1, to step 100.
# Resumable; run under experiments/nebius_supervise.sh.
cd "$(dirname "$0")/.."
source /data/venv/bin/activate
export HF_HOME=/data/hf PYTHONPATH="$(pwd)" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs runs results
D=~/spike_data
COMMON="--data $D/cpt_train --heldout $D/cpt_heldout --levels 4 --lora gist --beta --lambda_route 0.1
        --seq_len 8192 --batch 8 --steps 300 --eval_every 50 --eval_rows 8 --eval_len 8192
        --raw_window 2048 --select_train --select_k 16 --save_every 10 --resume"
train() { name=$1; shift
  [ -f runs/$name/adapters.pt ] && return
  echo "=== $name $(date +%T)"
  python gist_stage1_trainer.py $COMMON --out runs/$name "$@" >> logs/$name.log 2>&1
}
train rd_D1 --seed 0 --stop_at 150 --select_coarse --route_detach_q &
train rd_D10 --seed 0 --stop_at 150 --select_coarse --route_detach_q --lambda_route 1.0 &
train rd_R1s1 --seed 1 --stop_at 100 &
train rd_Hs1 --seed 1 --stop_at 100 --down hybrid &
wait
for r in D1 D10 R1s1 Hs1; do [ -f runs/rd_$r/adapters.pt ] || exit 1; done
echo "=== route_detach eval $(date +%T)"
G="--k 16 --raw_window 2048"
python experiments/select_gate.py --ckpts runs/rd_D1/adapters.pt runs/rd_D10/adapters.pt $G \
  --coarse --out results/route_detach_coarse.jsonl >> logs/route_detach_eval.log 2>&1 || exit $?
for r in D1 D10; do
  [ -f results/route_detach_fit_$r.done ] && continue
  python experiments/route_fit.py --ckpt runs/rd_$r/adapters.pt --raw_window 2048 \
    --out results/route_detach_fit_$r.jsonl >> logs/route_detach_eval.log 2>&1 || exit $?
  touch results/route_detach_fit_$r.done
done
echo "=== ROUTEDETACH DONE $(date +%T)"
