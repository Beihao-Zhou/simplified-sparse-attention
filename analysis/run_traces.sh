#!/usr/bin/env bash
# Record selection traces used by the locality benchmarks.
# transformers 4.57.1 (the version this repo pins) is shadowed in PYENV so the
# system env's transformers 5.x is not used; see analysis/prepare_prompt.py.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYENV="${PYENV:-/tmp/pyenv457}"
export PYTHONPATH="${PYENV}:${REPO}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/tmp/hfcache}"
cd "$REPO"
T=analysis/traces
R() { echo "=== $* ==="; python analysis/record_selection.py "$@" || echo "FAILED: $*"; }

R --workload passkey --n-garbage 18400  --steps 1000 --force-steps --out $T/pk_6k.npz
R --workload passkey --n-garbage 73600  --steps 1000 --force-steps --out $T/pk_24k.npz
R --workload passkey --n-garbage 218000 --steps 1000 --force-steps --out $T/pk_71k.npz
R --workload prompt_file --prompt-file analysis/prompts/gov_report_115.json  --steps 512  --out $T/gov_47k_nat.npz
R --workload prompt_file --prompt-file analysis/prompts/gov_report_115.json  --steps 1000 --force-steps --out $T/gov_47k_1000.npz
R --workload prompt_file --prompt-file analysis/prompts/narrativeqa_62.json  --steps 256  --out $T/nqa_42k_nat.npz
R --workload prompt_file --prompt-file analysis/prompts/v2_long.json --max-prompt-tokens 20700 --steps 1000 --force-steps --out $T/v2_24k.npz
R --workload prompt_file --prompt-file analysis/prompts/v2_long.json --max-prompt-tokens 61000 --steps 1000 --force-steps --out $T/v2_71k.npz
R --workload prompt_file --prompt-file analysis/prompts/v2_long.json --max-prompt-tokens 86000 --steps 1000 --force-steps --out $T/v2_100k.npz
echo "ALL DONE"
