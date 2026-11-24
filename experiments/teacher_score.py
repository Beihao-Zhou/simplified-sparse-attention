"""#22: the full-attention teacher of OPSD on long contexts: log-probs of the --n_new tokens
after --lens raw tokens of a held-out agent trajectory (gist_opsd_long.teacher_logprobs),
with time and peak memory.

    python experiments/teacher_score.py --model ~/models/Qwen2.5-7B-Instruct-1M \
        --data /data/agent_longeval/len262144 --lens 131072 262144
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datasets  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer, Qwen2ForCausalLM  # noqa: E402

import gist_opsd_long as ol  # noqa: E402
from gist_utils import Global_data  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--lens", type=int, nargs="+", default=[131072, 262144])
    ap.add_argument("--n_new", type=int, default=256)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    teacher = Qwen2ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                               attn_implementation="sdpa").cuda().eval()
    teacher.config.use_cache = False
    Global_data.gist_token_id = []  # raw text only: nothing to drop
    rows = datasets.load_from_disk(args.data)["input_ids"]
    for L in args.lens:
        # a len262144 bucket holds rows up to 256K, not of it: the first one long enough
        ids = next((r for r in rows if len(r) >= L + args.n_new), None)
        assert ids is not None, (L, max(map(len, rows)))
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        lp = ol.teacher_logprobs(teacher, ids[:L], ids[L:L + args.n_new], len(tok))
        torch.cuda.synchronize()
        # row i predicts stream token i
        nll = -lp.gather(1, torch.tensor(ids[L:L + args.n_new], device=lp.device)[:, None])
        print(json.dumps({"raw": L, "n_new": args.n_new, "sec": round(time.perf_counter() - t0, 1),
                          "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                          "nll": round(float(nll.mean()), 4)}), flush=True)


if __name__ == "__main__":
    main()
