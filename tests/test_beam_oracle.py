"""Oracle selection (#31): gist_beam.oracle_mass is the true attention mass per node, and
select with cfg.oracle picks the top-k of it."""

import random

import torch


def _tree(levels=3, seed=0):
    import gist_beam as gb
    from src.data.gist_stream import gist_segments

    rng = random.Random(seed)
    gids = [90001 + i for i in range(levels)]
    ids, pinned, turn = [], [], []
    for _ in range(9):
        ids.append(rng.randint(1, 999)), pinned.append(True), turn.append(0)
    for t in range(1, 30):
        for _ in range(rng.randint(10, 60)):
            ids.append(rng.randint(1, 999)), pinned.append(False), turn.append(t)
    g, p, tr = gist_segments(ids, pinned, turn, gids, 4, 4, sink=3)
    return gb.build_tree(torch.tensor(g), gids, 3, torch.tensor(p), torch.tensor(tr))


def test_mass_is_the_softmax_over_raw_tokens_summed_per_node():
    import gist_beam as gb

    tree = _tree()
    T = tree.level.numel()
    torch.manual_seed(0)
    h, d = 3, 16
    rows = torch.nonzero(tree.level < 0).flatten()[-40:]
    q, k = torch.randn(h, rows.numel(), d), torch.randn(T, d)
    mass = gb.oracle_mass(tree, q, k, rows, d ** -0.5)
    raw = (tree.level < 0).nonzero().flatten()
    for i, r in enumerate(rows.tolist()):
        js = raw[raw <= r]
        p = (q[:, i] @ k[js].T * d ** -0.5).softmax(-1)              # [h, |js|]
        for l in range(tree.num_levels):
            want = torch.zeros(h, tree.gpos[l].numel())
            for jj, j in enumerate(js.tolist()):
                node = int(tree.leaf[j]) // 4 ** l
                if tree.compressible[j] and node < want.shape[1]:
                    want[:, node] += p[:, jj]
            assert torch.allclose(mass[l][i], want, atol=1e-6), (r, l)


def test_oracle_picks_the_top_k_of_the_mass():
    import gist_beam as gb

    tree = _tree()
    T = tree.level.numel()
    torch.manual_seed(1)
    h, d = 2, 16
    rows = torch.nonzero(tree.level < 0).flatten()[-20:]
    q, k = torch.randn(h, rows.numel(), d), torch.randn(T, d)
    cfg = gb.BeamConfig(k=(3,), merge="union", oracle=True)
    s = gb.select(tree, q, k, None, rows, cfg, d ** -0.5)
    top = tree.num_levels - 1
    mass = gb.oracle_mass(tree, q, k, rows, d ** -0.5)[top]
    lo = gb._forced_leaf_lo(tree, rows, 1, 0, 0)
    c = gb._candidates(tree, rows, top, None, 4)
    free = c & ~gb._forced(tree, rows, top, lo, 4)
    for i in range(rows.numel()):
        want = set()
        for hh in range(h):
            m = mass[i, hh].masked_fill(~free[i], -1.0)
            want |= set(torch.topk(m, min(3, int(free[i].sum()))).indices.tolist())
        got = set(torch.nonzero(s.sel[top][i] & free[i]).flatten().tolist())
        assert got == want, (int(rows[i]), got, want)
