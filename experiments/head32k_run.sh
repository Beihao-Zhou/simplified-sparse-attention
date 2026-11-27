#!/bin/bash
# #44: stage 1 with the route heads trained jointly (gist_route.py, ADR 0004). gate32hA's
# config (coarse + hybrid, 32K, no --select_train: ADR 0003) plus the heads, one change.
# GRAD=head trains only the heads from their loss (their own clip and AdamW: lambda barely
# matters); GRAD=summary also the summaries' x through them, where the loss competes with
# relMSE for the LoRA, so lambda 0.1 as lambda_route (1.0 hurt KL there, #34 R4).
# Baselines at 32K, k 16: model keys decode KL 0.379 (gate32hA / gate32h), offline heads
# on frozen gate32h 0.211 (results/route_head32k_*; passkey there has reps 3, here 6).
# A 2-step smoke run first. Resumable; run under experiments/nebius_supervise.sh, one GRAD per VM.
cd "$(dirname "$0")/.."
source /data/venv/bin/activate
export HF_HOME=/data/hf PYTHONPATH="$(pwd)" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs runs results
A=/data/agent_data
GRAD=${GRAD:-head}
[ $GRAD = summary ] && LH=${LAMBDA_HEAD:-0.1} || LH=${LAMBDA_HEAD:-1.0}
R=head32k_$GRAD
train() {  # out stop_at
  python gist_stage1_trainer.py --data $A --heldout $A --levels 4 --lora gist --beta \
    --lambda_route 0.1 --raw_window 8192 --window_turns 2 --seq_len 32768 --batch 2 \
    --steps 300 --eval_every 50 --eval_rows 4 --eval_len 32768 --seed 0 --save_every 10 \
    --resume --stop_at $2 --select_coarse --down hybrid \
    --lambda_head $LH --head_grad $GRAD --out runs/$1 >> logs/$R.log 2>&1
}
if [ ! -f runs/${R}_smoke/route_heads.pt ]; then
  echo "=== $R smoke $(date +%T)"
  train ${R}_smoke 2 || exit $?
  grep -q '"route_head"' runs/${R}_smoke/log.jsonl || { echo "no route_head loss logged"; exit 1; }
fi
if [ ! -f runs/$R/route_heads.pt ]; then
  echo "=== $R $(date +%T)"
  train $R ${STOP_AT:-150} || exit $?
fi
for ks in "16" "16 48"; do
  out=results/${R}_k${ks// /_}
  [ -f $out.jsonl ] && continue
  echo "=== $R select_gate k $ks $(date +%T)"
  python experiments/select_gate.py --ckpts runs/$R/adapters.pt --agent $A --len 32768 \
    --raw_window 8192 --window_turns 2 --k $ks --coarse --modes decode all --reps 6 \
    --route_heads runs/$R/route_heads.pt --out $out.tmp >> logs/$R.log 2>&1 || exit $?
  mv $out.tmp $out.jsonl
done
echo "=== HEAD32K_${GRAD^^} DONE $(date +%T)"
