"""Reference beam selection (gist_beam.py, #19)."""

import random

import pytest
import torch

import gist_beam as gb
from src.data.gist_stream import gist_segments

GIDS = [90001, 90002, 90003, 90004]
SINK = 3


def _seq(seed, levels=3, n_pinned=7, turns=12):
    rng = random.Random(seed)
    ids, pinned, turn = [], [], []
    for _ in range(n_pinned):
        ids.append(rng.randint(1, 999)), pinned.append(True), turn.append(0)
    for t in range(1, turns + 1):
        for _ in range(rng.randint(5, 60)):
            ids.append(rng.randint(1, 999)), pinned.append(False), turn.append(t)
    g, p, t = gist_segments(ids, pinned, turn, GIDS[:levels], 4, 4, sink=SINK)
    return torch.tensor(g), torch.tensor(p), torch.tensor(t)


def _setup(seed, levels=3, d=8, h=3):
    ids, pinned, turn = _seq(seed, levels)
    tree = gb.build_tree(ids, GIDS[:levels], SINK, pinned, turn)
    T = ids.numel()
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(T, d, generator=g, dtype=torch.float64)
    v = torch.randn(T, d, generator=g, dtype=torch.float64)
    raw_rows = torch.nonzero(tree.level < 0).flatten()
    rows = raw_rows[torch.linspace(len(raw_rows) // 3, len(raw_rows) - 1, 6).long()]
    q = torch.randn(h, rows.numel(), d, generator=g, dtype=torch.float64)
    beta = torch.randn(T, generator=g, dtype=torch.float64) * (tree.level >= 0)
    return ids, tree, q, k, v, beta, rows


@pytest.mark.parametrize("levels", [2, 3, 4])
def test_pyramid_layout_matches_positions(levels):
    ids, pinned, turn = _seq(levels, levels)
    tree = gb.build_tree(ids, GIDS[:levels], SINK, pinned, turn)
    for l in range(levels - 1):
        is_up = tree.level == l + 1
        unit_up = torch.cumsum(is_up.long(), 0) - is_up.long()     # parent by position
        par = unit_up[tree.gpos[l]]
        closed = par < len(tree.gpos[l + 1])
        assert torch.equal(par[closed], torch.arange(len(par))[closed] // 4)
    # a level-0 node covers exactly the 4 compressible raw tokens before its gist
    for j, gp in enumerate(tree.gpos[0].tolist()):
        mine = torch.nonzero(tree.compressible & (tree.leaf == j)).flatten()
        assert mine.numel() == 4 and int(mine.max()) < gp
    assert not tree.compressible[pinned.bool()].any()
    assert int(tree.always.sum()) == int(pinned.sum()) + SINK


def _coverage(tree, vis, rows):
    """How many times each compressible raw token <= p is represented: exactly (visible)
    or through a visible summary whose subtree holds it."""
    n, T = vis.shape
    cnt = (vis & tree.compressible[None]).long()
    for l in range(tree.num_levels):
        node_of = tree.leaf // 4 ** l
        for r in range(n):
            shown = tree.gpos[l][vis[r, tree.gpos[l]]]
            ids = torch.nonzero(tree.level[None, :] == l)[:, 1]
            for gp in shown.tolist():
                j = int((tree.gpos[l] == gp).nonzero())
                cnt[r] += (tree.compressible & (node_of == j)).long()
    p = rows[:, None]
    mask = tree.compressible[None] & (torch.arange(T)[None] <= p)
    return cnt, mask


@pytest.mark.parametrize("merge", ["union", "sumexp"])
@pytest.mark.parametrize("window", [(0, 0), (40, 0), (0, 2), (64, 2)])
def test_coarse_terms_partition_the_prefix(merge, window):
    ids, tree, q, k, v, beta, rows = _setup(11)
    cfg = gb.BeamConfig(k=(3, 2), merge=merge, coarse=True)
    s = gb.select(tree, q, k, beta, rows, cfg, 0.3, prev_chunks=1,
                  raw_window=window[0], window_turns=window[1])
    vis = gb.visible(tree, s, coarse=True)
    cnt, mask = _coverage(tree, vis, rows)
    assert torch.equal(cnt[mask], torch.ones_like(cnt[mask])), "each raw token exactly once"
    assert (mask & ~vis).any() and vis[:, tree.level >= 0].any(), "the test must exercise coarse terms"
    assert vis[:, tree.always].sum() == (tree.always[None] & (torch.arange(ids.numel())[None] <= rows[:, None])).sum()


def test_full_expansion_is_causal_attention_over_raw_tokens():
    ids, tree, q, k, v, beta, rows = _setup(5)
    cfg = gb.BeamConfig(k=(10**6,), coarse=True)
    s = gb.select(tree, q, k, beta, rows, cfg, 0.3)
    vis = gb.visible(tree, s, coarse=True)
    T = ids.numel()
    raw_causal = (tree.level[None] < 0) & (torch.arange(T)[None] <= rows[:, None])
    assert torch.equal(vis, raw_causal)
    out = gb.attend(q, k, v, vis, 0.3, beta)
    ref = gb.attend(q, k, v, raw_causal, 0.3, None)
    assert torch.allclose(out, ref, atol=0, rtol=0)  # noise floor 0


def test_wide_upper_beam_equals_flat_leaf_topk():
    ids, tree, q, k, v, beta, rows = _setup(7)
    cfg = gb.BeamConfig(k=(4, 10**6), merge="union")
    s = gb.select(tree, q, k, beta, rows, cfg, 0.3, prev_chunks=0)
    g0 = k[tree.gpos[0]]
    for r, p in enumerate(rows.tolist()):
        closed = (tree.gpos[0] < p)
        sc = torch.einsum("hd,gd->hg", q[:, r], g0) * 0.3 + beta[tree.gpos[0]]
        sc = sc.masked_fill(~closed[None], float("-inf"))
        flat = torch.zeros(len(closed), dtype=torch.bool)
        for h in range(q.shape[0]):
            flat[torch.topk(sc[h], min(4, int(closed.sum()))).indices] = True
        assert torch.equal(s.sel[0][r], flat & closed)


def test_sg_sr_without_coarse_terms():
    ids, tree, q, k, v, beta, rows = _setup(3)
    s = gb.select(tree, q, k, beta, rows, gb.BeamConfig(k=(3, 2)), 0.3)
    vis = gb.visible(tree, s, coarse=False)
    for l in range(tree.num_levels):
        assert torch.equal(vis[:, tree.gpos[l]], s.sel[l])


def test_rebuilt_from_recorded_ids():
    ids, tree, q, k, v, beta, rows = _setup(9, levels=4)
    for coarse in (False, True):
        s = gb.select(tree, q, k, beta, rows, gb.BeamConfig(k=(3, 2), coarse=coarse), 0.3,
                      raw_window=30, window_turns=2)
        rec = [s.ids(r) for r in range(rows.numel())]
        s2 = gb.from_ids(tree, rows, rec)
        assert torch.equal(gb.visible(tree, s, coarse), gb.visible(tree, s2, coarse))


def test_ties_go_to_the_lower_node_id():
    ids, tree, q, k, v, beta, rows = _setup(2)
    k = torch.zeros_like(k)  # every score equal
    s = gb.select(tree, q, k, None, rows[-1:], gb.BeamConfig(k=(2,)), 0.3, prev_chunks=0)
    top = torch.nonzero(s.sel[-1][0]).flatten()
    assert top.tolist() == [0, 1]


def test_all_rows_match_single_row_selection():
    ids, tree, q, k, v, beta, rows = _setup(13)
    T = ids.numel()
    g = torch.Generator().manual_seed(0)
    q_all = torch.randn(q.shape[0], T, q.shape[2], generator=g, dtype=torch.float64)
    cfg = gb.BeamConfig(k=(3, 2), coarse=True)
    full = gb.visible_all_rows(tree, q_all, k, beta, cfg, 0.3, raw_window=20, chunk=37)
    for p in rows.tolist():
        r = torch.tensor([p])
        one = gb.visible(tree, gb.select(tree, q_all[:, r], k, beta, r, cfg, 0.3,
                                         raw_window=20), True)[0]
        assert torch.equal(full[p], one)
    assert not full[tree.level >= 0].any()
