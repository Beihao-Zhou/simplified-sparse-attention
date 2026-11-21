"""#18 on a GPU: append a long context onto one KV cache in pieces through flex (no dense
[q, T] mask), then decode a few steps. Reports time and peak memory per append.

    ATTN_IMPL=ref python experiments/session_check.py --total 131072 --pieces 4
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datasets  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import gist_flex_attention  # noqa: E402
from gist_qwen2 import GistQwen2ForCausalLM  # noqa: E402
from gist_session import GistSession  # noqa: E402
from gist_utils import Global_data, add_gist_tokens  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--docs", default="~/spike_data/cpt_heldout")
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--total", type=int, default=131072)
    ap.add_argument("--pieces", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()
    assert os.environ.get("ATTN_IMPL") == "ref", "0.5B (head_dim 64) decodes on the eager path"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = GistQwen2ForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=gist_flex_attention.GIST_FLEX).cuda().eval()
    gid = add_gist_tokens(tok, model, num_levels=args.levels)
    Global_data.gist_token_id, Global_data.chunk_size = gid, [4] * args.levels
    Global_data.sink_size, Global_data.num_previous_chunks = 3, 1
    Global_data.pad_token_id, Global_data.stage1 = tok.pad_token_id, None
    Global_data.top_k = [3] * max(3, args.levels)
    model.selective = True
    raw = []
    for r in datasets.load_from_disk(os.path.expanduser(args.docs)):
        raw += [t for t in r["input_ids"] if t not in set(gid)]
        if len(raw) >= args.total:
            break
    raw = raw[: args.total]
    s = GistSession(model)
    n = args.total // args.pieces
    for i in range(args.pieces):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        s.append(raw[i * n:(i + 1) * n], pinned=False, turn=i)
        torch.cuda.synchronize()
        print(json.dumps({"append": i, "raw": (i + 1) * n, "tokens": len(s.ids),
                          "sec": round(time.time() - t0, 2),
                          "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}), flush=True)
    for j in range(args.steps):
        t0 = time.time()
        logits = s.step(int(raw[j]))
        torch.cuda.synchronize()
        print(json.dumps({"step": j, "tokens": len(s.ids), "sec": round(time.time() - t0, 2),
                          "finite": bool(torch.isfinite(logits).all())}), flush=True)


if __name__ == "__main__":
    main()
