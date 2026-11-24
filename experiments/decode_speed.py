"""Decode time per raw token on the DECODE_SELECT=beam path (#25): append a held-out
long agent trajectory (experiments/agent_longeval.py) cut to --lens raw tokens, then
decode --steps tokens greedily, with the CUDA graphs on and off.

    python experiments/decode_speed.py --stage1 runs/gate_A/adapters.pt \
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


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--lens", type=int, nargs="+", default=[131072, 262144])
    ap.add_argument("--steps", type=int, default=128)
    args = ap.parse_args()
    os.environ.update(DECODE_SELECT="beam", BEAM_K="256", BEAM_MERGE="union", BEAM_COARSE="0")
    import gist_opsd_agent as A
    from gist_session import GistSession
    from gist_utils import Global_data

    tok, model, _, _ = A.load_stage1_flex(args.stage1, args.model)
    Global_data.raw_window, Global_data.window_turns = 8192, 2
    model.eval()
    ds = datasets.load_from_disk(args.data)
    for L in args.lens:
        # a len262144 bucket holds rows up to 256K: the first one long enough
        row = ds[next(i for i, n in enumerate(map(len, ds["input_ids"])) if n >= L)]
        for graph in ("1", "0"):
            os.environ["GIST_DECODE_GRAPH"] = graph
            s = GistSession(model)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t_pre = time.perf_counter()
            s.append(row["input_ids"][:L], row["pinned"][:L], row["turn"][:L])
            torch.cuda.synchronize()
            t_pre = time.perf_counter() - t_pre
            y = int(model.lm_head(s.last_hidden).argmax())
            for _ in range(8):  # warm-up (graph capture, compiles)
                y = int(s.step(y).argmax())
            n0 = len(s.ids)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.steps):
                y = int(s.step(y).argmax())
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            print(json.dumps({"raw": L, "tokens": n0, "graph": graph == "1",
                              "ms_per_token": round(1000 * dt / args.steps, 1),
                              "fed_per_token": round((len(s.ids) - n0) / args.steps, 2),
                              "prefill_s": round(t_pre, 1),
                              "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}),
                  flush=True)
            del s
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
