#!/usr/bin/env bash
# The router is the 32K gap (oracle32h: decode KL 0.379 -> 0.054, passkey 0.13 -> 0.93). The
# routing KL compares every closed node of a level, but decode compares only the children of
# the selected parents; R4 had the lowest routing loss and the worst beam. Here the KL is over
# the children of the oracle top-16 parents (--route_scope oracle_children, decode's k), on
# ch8_R2H's config (coarse + hybrid; beam 0.72, fit1 0.95, decode KL 0.137), at lambda 0.1
# and 1.0. Pass: beam clearly above 0.72 with decode KL no worse. Resumable; run under
# experiments/nebius_supervise.sh.
cd "$(dirname "$0")/.."
source /data/venv/bin/activate
export HF_HOME=/data/hf PYTHONPATH="$(pwd)" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs runs results
D=~/spike_data
COMMON="--data $D/cpt_train --heldout $D/cpt_heldout --levels 4 --lora gist --beta
        --seq_len 8192 --batch 8 --steps 300 --eval_every 50 --eval_rows 8 --eval_len 8192
        --raw_window 2048 --select_train --select_k 16 --save_every 10 --resume --seed 0
        --stop_at 150 --select_coarse --down hybrid --route_scope oracle_children
        --route_oracle_k 16"
train() { name=$1; shift
  [ -f runs/$name/adapters.pt ] && return
  echo "=== $name $(date +%T)"
  python gist_stage1_trainer.py $COMMON --out runs/$name "$@" >> logs/$name.log 2>&1
}
train oc_L01 --lambda_route 0.1 &
train oc_L1 --lambda_route 1.0 &
wait
for r in L01 L1; do [ -f runs/oc_$r/adapters.pt ] || exit 1; done
python experiments/select_gate.py --ckpts runs/oc_L01/adapters.pt runs/oc_L1/adapters.pt \
  --k 16 --raw_window 2048 --coarse --out results/route_oc8k.jsonl >> logs/route_oc8k_eval.log 2>&1 \
  || exit $?
for r in L01 L1; do
  [ -f results/route_oc8k_fit_$r.done ] && continue
  python experiments/route_fit.py --ckpt runs/oc_$r/adapters.pt --raw_window 2048 \
    --out results/route_oc8k_fit_$r.jsonl >> logs/route_oc8k_eval.log 2>&1 || exit $?
  touch results/route_oc8k_fit_$r.done
done
echo "=== ROUTEOC8K DONE $(date +%T)"
