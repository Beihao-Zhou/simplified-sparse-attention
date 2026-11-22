"""#29: how block-sparse the stage-1 selection mask ("sel", #28) is. For each layer's
BlockMask: partial and full 128 x 128 blocks as a share of the causal grid, and the
visible elements as a share of the elements inside them (flex computes whole blocks), so
whether flex's block granularity can carry selection at every position.

    python experiments/sel_mask_density.py -- <gist_stage1_trainer args with --select_train>
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

STATS = []


def main():
    import gist_flex_attention as gfa
    import gist_stage1_trainer as tr

    orig = gfa._select_block_mask

    def wrapped(query, key, *a, **k):
        bm = orig(query, key, *a, **k)
        if len(STATS) < 4 and query.shape[2] > 4096:  # a few training layers (not the eval at 2K)
            T = query.shape[2]
            nb = -(-T // 128)
            causal = nb * (nb + 1) // 2
            part = bm.kv_num_blocks.float().sum(-1).mean().item()  # per (batch, head)
            full = bm.full_kv_num_blocks.float().sum(-1).mean().item() if bm.full_kv_num_blocks is not None else 0.0
            dense = bm.to_dense() if hasattr(bm, "to_dense") else None
            vis = None
            if T <= 20000:  # the element mask costs T^2 per head
                from torch.nn.attention.flex_attention import create_mask
                m = create_mask(bm.mask_mod, 1, query.shape[1], T, T, device=query.device)
                vis = m.float().sum().item() / query.shape[1]
            STATS.append({"layer": len(STATS), "T": T, "causal_blocks": causal,
                          "partial_share": round(part / causal, 4), "full_share": round(full / causal, 4),
                          "visible_per_row": None if vis is None else round(vis / T, 1),
                          "visible_share_of_computed": None if vis is None else
                          round(vis / ((part + full) * 128 * 128), 4)})
            print(json.dumps(STATS[-1]), flush=True)
        return bm

    gfa._select_block_mask = wrapped
    tr.main(sys.argv[sys.argv.index("--") + 1:])


if __name__ == "__main__":
    main()
