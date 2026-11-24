#!/usr/bin/env bash
# #41: what down=raw costs. First the 32K gate in one mode: gate32r A and B each scored with
# unfolding (sel) and without (hier); their own-path numbers mixed the two (#29). Then at 8K,
# #34 R1's configuration (SG+SR, selection at every position, raw window 2048), each run one
# change away from a control:
#   H  down=hybrid, R1's teacher chain: raw rows pass raw (the base stream), summary rows
#      their gist-mask output, so summaries are written as at inference   vs r8_R1 (VM2)
#   D0 down=raw, every student against the fixed base teacher              vs r8_R1
#   D1 down=sel, the same pairs: the inference path passed down             vs D0
# All are scored with unfolding (sel), as R1. Resumable; run under nebius_supervise.sh.
cd "$(dirname "$0")/.."
source /data/venv/bin/activate
export HF_HOME=/data/hf PYTHONPATH="$(pwd)" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs runs results
A=/data/agent_data
for r in A B; do for m in sel hier; do
  [ -f runs/gate32r_$r/eval_$m.done ] && continue
  sel=""; [ $m = sel ] && sel="--select_train --select_k 16"
  python gist_stage1_trainer.py --data $A --heldout $A --levels 4 --lora gist --beta \
    --raw_window 8192 --window_turns 2 --eval_rows 4 --eval_len 32768 --seed 0 $sel \
    --eval_only runs/gate32r_$r/adapters.pt --out runs/gate32r_$r/eval_$m \
    >> logs/gate32r_samemode.log 2>&1 || exit $?
  touch runs/gate32r_$r/eval_$m.done
done; done
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
train b41_H --down hybrid &
train b41_D0 --down raw --pairs $BASE &
train b41_D1 --down sel --pairs $BASE &
wait
for r in H D0 D1; do [ -f runs/b41_$r/adapters.pt ] || exit 1; done
python experiments/select_gate.py --ckpts runs/b41_H/adapters.pt runs/b41_D0/adapters.pt \
  runs/b41_D1/adapters.pt --k 16 --raw_window 2048 --out results/base41_sgsr.jsonl \
  >> logs/base41_eval.log 2>&1 || exit $?
echo "=== BASE41 DONE $(date +%T)"
