"""Batched beam selection (kernels/select_rows.py, #29) against the reference
(gist_select_train.selected_lists): the same key set X for every row and KV group. CUDA."""

import random

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("levels", [2, 3, 4])
@pytest.mark.parametrize("merge", ["union", "sumexp"])
@pytest.mark.parametrize("coarse", [False, True])
@pytest.mark.parametrize("window", [(0, 0), (64, 2)])
def test_kernel_selects_what_the_reference_selects(levels, merge, coarse, window):
    import gist_beam as gb
    from gist_select_train import selected_lists
    from gist_utils import Global_data
    from kernels.select_rows import select_lists
    from src.data.gist_stream import gist_segments

    saved = (Global_data.num_previous_chunks, Global_data.raw_window, Global_data.window_turns)
    Global_data.num_previous_chunks, (Global_data.raw_window, Global_data.window_turns) = 1, window
    try:
        rng = random.Random(levels)
        gids = [90001 + i for i in range(levels)]
        ids, pinned, turn = [], [], []
        for _ in range(9):
            ids.append(rng.randint(1, 999)), pinned.append(True), turn.append(0)
        for t in range(1, 40):
            for _ in range(rng.randint(10, 80)):
                ids.append(rng.randint(1, 999)), pinned.append(False), turn.append(t)
        g, p, tr = gist_segments(ids, pinned, turn, gids, 4, 4, sink=3)
        dev = "cuda"
        tree = gb.build_tree(torch.tensor(g, device=dev), gids, 3, torch.tensor(p, device=dev),
                             torch.tensor(tr, device=dev))
        T = tree.level.numel()
        pos = torch.arange(T, device=dev)
        lo = gb._forced_leaf_lo(tree, pos, 1, *window)
        rows = torch.nonzero(tree.level < 0).flatten()
        H, Hkv, D = 14, 2, 64
        hpg = H // Hkv
        torch.manual_seed(levels)
        q = torch.randn(H, T, D, device=dev).bfloat16()
        k = torch.randn(Hkv, T, D, device=dev).bfloat16()
        beta = torch.randn(Hkv, T, device=dev) * (tree.level >= 0)
        cfg = gb.BeamConfig(k=(5, 3), merge=merge, coarse=coarse)
        got = select_lists(tree, lo, rows, q, k, beta, cfg, D ** -0.5, batch=500)
        for gi in range(Hkv):
            hs = slice(gi * hpg, (gi + 1) * hpg)
            want = selected_lists(tree, lo, rows, q[hs][:, rows], k[gi], beta[gi], cfg, D ** -0.5)
            for r in range(rows.numel()):
                a = sorted(x for x in got[gi, r].tolist() if x >= 0)
                b = sorted(x for x in want[r].tolist() if x >= 0)
                assert a == b, (gi, int(rows[r]), len(a), len(b))
    finally:
        Global_data.num_previous_chunks, Global_data.raw_window, Global_data.window_turns = saved
