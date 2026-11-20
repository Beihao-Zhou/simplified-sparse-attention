#!/usr/bin/env bash
# Follow-ups to #8: (2) off-policy routing-KL-only control at the S4b budget, then
# (1) held-out KL diagnostics (reverse/forward KL, top-1 agreement) and task accuracy.
cd ~/opsd_spikes
source ~/ssa_venv/bin/activate
export PYTHONPATH="$(pwd):$PYTHONPATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ATTN_IMPL=ref
S1=runs/s3_d_joint/adapters.pt; OP=runs/s4b/adapters.pt; CT=runs/ctl_route_data/adapters.pt; D=~/spike_data/sft
echo "=== STAGE control $(date +%T)"
[ -f $CT ] || python gist_opsd_trainer.py --stage1 $S1 --data $D --ctx_len 2048 --n_new 64 --batch 4 --steps 150 \
  --check_every 25 --rkl_weight 0 --y_source data --out runs/ctl_route_data > logs/ctl_route_data.log 2>&1
echo "exit=$? control $(date +%T)"
echo "=== STAGE kl_eval $(date +%T)"
python experiments/opsd_eval.py --data $D --ckpts $S1 $OP $CT > logs/eval12_kl.log 2>&1
echo "exit=$? kl_eval $(date +%T)"
echo "=== STAGE task4k $(date +%T)"
rm -f results/task_eval_4k.jsonl
python experiments/task_eval.py --ckpts base $S1 $OP $CT --n 100 --out results/task_eval_4k.jsonl > logs/task4k.log 2>&1
echo "exit=$? task4k $(date +%T)"
echo "=== STAGE passkey8k $(date +%T)"
rm -f results/task_eval_pk8k.jsonl
python experiments/task_eval.py --ckpts base $S1 $OP $CT --tasks passkey --ctx_len 8192 --n 50 \
  --out results/task_eval_pk8k.jsonl > logs/pk8k.log 2>&1
echo "exit=$? passkey8k $(date +%T)"
echo "=== ALL DONE $(date +%T)"
