"""#29 step 1: where a stage-1 step with selection at every position (#28, ADR 0003) spends
its time, by component: per-row beam selection (gist_beam.select), the visible sets it
expands (gist_beam.visible), the dense [T, T] masks and their BlockMask, and the flex
attention layers as a whole. Runs gist_stage1_trainer.main with timers patched in.

    python experiments/stage1_select_profile.py --seq_len 32768 -- <trainer args>
"""

import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

T = collections.defaultdict(float)


def timed(name, fn):
    def w(*a, **k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = fn(*a, **k)
        torch.cuda.synchronize()
        T[name] += time.perf_counter() - t
        return r
    return w


def main():
    import torch.nn.attention.flex_attention as fa

    import gist_beam as gb
    import gist_flex_attention as gfa
    import gist_stage1_trainer as tr

    gb.select = timed("selection (gist_beam.select)", gb.select)
    gb.visible = timed("visible sets (gist_beam.visible)", gb.visible)
    fa.create_mask = timed("dense mask (create_mask)", fa.create_mask)
    fa.create_block_mask = timed("BlockMask (create_block_mask)", fa.create_block_mask)
    gfa._select_block_mask = timed("sel mask total (_select_block_mask)", gfa._select_block_mask)
    gfa._stage1_routing = timed("routing KL", gfa._stage1_routing)
    i = sys.argv.index("--")
    tr.main(sys.argv[i + 1:])
    steps = int(sys.argv[sys.argv.index("--steps") + 1])
    print(json.dumps({"per_step_s": {k: round(v / steps, 2) for k, v in sorted(T.items(), key=lambda kv: -kv[1])},
                      "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}), flush=True)


if __name__ == "__main__":
    main()
