#!/usr/bin/env bash
# All 17 row groups of the nebius trajectories -> /data/agent_data (resumable; run under
# experiments/nebius_supervise.sh on a preemptible VM).
cd "$(dirname "$0")/.."
source /data/venv/bin/activate
export HF_HOME=/data/hf TOKENIZERS_PARALLELISM=false
python experiments/agent_data.py --out /data/agent_data --row_groups $(seq 0 16) \
  --cache /data/nebius_rg --workers 14 || exit $?
echo "=== AGENT DATA DONE"
