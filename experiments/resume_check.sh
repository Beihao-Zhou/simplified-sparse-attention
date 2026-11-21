#!/usr/bin/env bash
# GPU check of --resume (gist_resume.py): per trainer, run A straight through; run B is
# SIGTERMed after its first ckpt.pt (expect exit 143), resumed, and must match A up to
# GPU nondeterminism. Run from the repo root; S1_DIR = a stage-1 run directory.
export PYTHONPATH="$(pwd):$PYTHONPATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=~/spike_data
killtest() {  # name, cmd...
  n=$1; shift; rm -rf runs/rt_${n}_a runs/rt_${n}_b
  "$@" --out runs/rt_${n}_a > logs/rt_${n}_a.log 2>&1; echo "$n A exit=$?"
  "$@" --out runs/rt_${n}_b > logs/rt_${n}_b1.log 2>&1 & pid=$!
  until [ -f runs/rt_${n}_b/ckpt.pt ]; do sleep 2; done; kill -TERM $pid; wait $pid; echo "$n B1 exit=$?"
  "$@" --out runs/rt_${n}_b --resume > logs/rt_${n}_b2.log 2>&1; echo "$n B2 exit=$?"
  "$@" --out runs/rt_${n}_b --resume 2>&1 | grep already
  python - runs/rt_${n}_a runs/rt_${n}_b <<"PY"
import sys,torch,json
a,b=[torch.load(d+"/adapters.pt",weights_only=False)["state"] for d in sys.argv[1:]]
print("max|dA-dB|", max(float((a[k].float()-b[k].float()).abs().max()) for k in a))
for d in sys.argv[1:]:
    print(d,[(r["step"],round(r.get("rkl",r.get("g1|raw",0)),5)) for r in map(json.loads,open(d+"/log.jsonl")) if isinstance(r["step"],int) and "lr" in r])
PY
}
source ~/ssa_venv/bin/activate
killtest opsd env ATTN_IMPL=ref python gist_opsd_trainer.py --stage1 ${S1_DIR:-runs/s3_d_joint}/adapters.pt --data $D/sft --ctx_len 2048 --n_new 16 --batch 1 --steps 4 --save_every 2 --check_every 0
source ~/ssa_venv_d/bin/activate
export GIST_FLEX_KERNEL_OPTIONS="{\"BLOCK_M\":64,\"BLOCK_N\":64}"
killtest s1 python gist_stage1_trainer.py --data $D/cpt_train --heldout $D/cpt_heldout --lora gist --beta --lambda_route 0.1 --seq_len 2048 --batch 2 --steps 8 --save_every 2 --eval_every 100 --eval_rows 2 --eval_len 2048
