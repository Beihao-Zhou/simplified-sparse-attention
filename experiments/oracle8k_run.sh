#!/usr/bin/env bash
# #31 item 1 at 8K: the #28 gate's checkpoints under oracle selection (nodes scored by true
# attention mass), with and without coarse terms, plus the coarse control without the
# oracle (the SG+SR control is results/select_gate.jsonl). Resumable per (ckpt, mode).
cd "$(dirname "$0")/.."
source /data/venv/bin/activate
export HF_HOME=/data/hf PYTHONPATH="$(pwd)" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C="--ckpts runs/gate_A/adapters.pt runs/gate_B/adapters.pt --k 16"
python experiments/select_gate.py $C --oracle --out results/oracle8k_sgsr.jsonl >> logs/oracle8k.log 2>&1 || exit $?
python experiments/select_gate.py $C --oracle --coarse --out results/oracle8k_coarse.jsonl >> logs/oracle8k.log 2>&1 || exit $?
python experiments/select_gate.py $C --coarse --out results/select8k_coarse.jsonl >> logs/oracle8k.log 2>&1 || exit $?
echo "=== ORACLE8K DONE $(date +%T)"
