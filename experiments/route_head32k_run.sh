#!/usr/bin/env bash
# #34, steps 1-2: decoupled route heads (gist_route.py) end to end. Train ROUTER for every
# layer of the frozen gate32h (route_amortize.py --save, 5 layers per pass to fit the GPU), then
# select_gate with --route_heads: decode / all KL and passkey at 32K, against gate32h's own
# router (decode 0.379, all 0.359, passkey all 0.13 / 0.33) and k 16 48 (0.217 / 0.180).
# Eval only, resumable per pass and per setting. Run under nebius_supervise.sh.
ROUTER=${ROUTER:-mlp_kids_r256}
cd "$(dirname "$0")/.."
source /data/venv/bin/activate
export HF_HOME=/data/hf PYTHONPATH="$(pwd)" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs results runs/route_heads
H=runs/route_heads/gate32h_$ROUTER
A=/data/agent_data
# smoke: untrained heads (zero deltas) must select exactly as the model does
if [ ! -f results/route_head32k_smoke.ok ]; then
  python - <<'PY' || exit $?
import torch
from gist_route import RouteCore
cfg = {"hidden": 896, "d": 64, "r": 8, "levels": 4, "kids": True, "m": 1}
cores = lambda: [RouteCore(**cfg).state_dict() for _ in range(2)]
torch.save({"config": cfg, "layers": {l: cores() for l in range(1, 24)}},
           "runs/route_heads/untrained.pt")
PY
  for h in none untrained; do
    extra=""; [ $h = untrained ] && extra="--route_heads runs/route_heads/untrained.pt"
    python experiments/select_gate.py --ckpts runs/gate32h/adapters.pt --agent $A --len 32768 \
      --raw_window 8192 --window_turns 2 --k 16 --coarse --modes decode --n_kl 2 --reps 1 $extra \
      --out results/route_head32k_smoke_$h.jsonl >> logs/route_head32k.log 2>&1 || exit $?
  done
  python - <<'PY' || exit $?
import json
a, b = ([json.loads(l) for l in open(f"results/route_head32k_smoke_{h}.jsonl")]
        for h in ("none", "untrained"))
key = lambda r: (r["mode"], r["kl"], r["passkey_front"], r["passkey_back"])
assert [key(r) for r in a] == [key(r) for r in b], (a, b)
print("smoke ok", [key(r) for r in a])
PY
  touch results/route_head32k_smoke.ok
fi
for layers in "1 2 3 4 5" "6 7 8 9 10" "11 12 13 14 15" "16 17 18 19 20" "21 22 23"; do
  name=${H}_${layers// /_}
  [ -f $name.pt ] && continue
  python experiments/route_amortize.py --ckpt runs/gate32h/adapters.pt --len 32768 \
    --raw_window 8192 --layers $layers --train_reps 12 --test_reps 1 --routers $ROUTER \
    --steps 5000 --lr 1e-3 --wd 0.01 --save $name.tmp \
    --out results/route_head32k_${ROUTER}_${layers// /_}.jsonl >> logs/route_head32k.log 2>&1 \
    || exit $?
  mv $name.tmp $name.pt
done
for ks in "16" "16 48"; do
  out=results/route_head32k_${ROUTER}_k${ks// /_}
  [ -f $out.jsonl ] && continue
  python experiments/select_gate.py --ckpts runs/gate32h/adapters.pt --agent $A --len 32768 \
    --raw_window 8192 --window_turns 2 --k $ks --coarse --modes decode all \
    --route_heads ${H}_*.pt --out $out.tmp >> logs/route_head32k.log 2>&1 || exit $?
  mv $out.tmp $out.jsonl
done
echo "=== ROUTEHEAD32K DONE $(date +%T)"
