#!/usr/bin/env bash
# #41: what down=raw costs on the inference path. The configuration of #34's R1 (8K, SG+SR,
# selection at every position, raw window 2048), with the inference path passed down
# against the fixed "base" teacher (each layer's output in the base model's own stream,
# recorded in a no-grad pass: no moving target, #29). D1 from step 0; D2 the down=raw chain
# for 75 steps, then D1's loss. Compare with runs/r8_R1 (down=raw, VM2) at the same eval.
# Resumable; run under experiments/nebius_supervise.sh.
cd "$(dirname "$0")/.."
source /data/venv/bin/activate
export HF_HOME=/data/hf PYTHONPATH="$(pwd)" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs runs results
D=~/spike_data
COMMON="--data $D/cpt_train --heldout $D/cpt_heldout --levels 4 --lora gist --beta --lambda_route 0.1
        --seq_len 8192 --batch 8 --steps 300 --eval_every 50 --eval_rows 8 --eval_len 8192 --seed 0
        --raw_window 2048 --select_train --select_k 16 --save_every 10 --resume --stop_at ${STOP_AT:-150}"
BASE="g1:base g2:base g3:base hier:base sel:base"
train() { name=$1; shift
  [ -f runs/$name/adapters.pt ] && return
  echo "=== $name $(date +%T)"
  python gist_stage1_trainer.py $COMMON --out runs/$name "$@" >> logs/$name.log 2>&1
}
train b41_D1 --down sel --pairs $BASE &
train b41_D2 --phase2_at 75 --phase2_down sel --phase2_pairs $BASE &
wait
for r in D1 D2; do [ -f runs/b41_$r/adapters.pt ] || exit 1; done
python experiments/select_gate.py --ckpts runs/b41_D1/adapters.pt runs/b41_D2/adapters.pt \
  --k 16 --raw_window 2048 --out results/base41_sgsr.jsonl >> logs/base41_eval.log 2>&1 || exit $?
echo "=== BASE41 DONE $(date +%T)"
