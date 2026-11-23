"""route_fit.fit (#31): on mass that one key per node generates, the fit recovers the
ranking on held-out queries; with two directions per node, four keys beat one."""

import torch


def _mass(q, keys, scaling):
    """[n, h, G]: softmax over nodes of logsumexp_m q . k_m (keys [G, m, d])."""
    from experiments.route_fit import score

    return score(q, keys, torch.zeros(keys.shape[0]), scaling).softmax(-1)


def _ratio(keys, b, q, mass, free, scaling):
    from experiments.route_fit import flat_ratio, score

    return flat_ratio(score(q, keys, b, scaling), mass, free, 4)


def test_one_key_recovers_one_key_mass_and_four_keys_two_directions():
    from experiments.route_fit import fit

    torch.manual_seed(0)
    h, n, G, d, sc = 2, 400, 32, 16, 0.25
    q = torch.randn(h, n, d) * 2
    free = torch.ones(n, G, dtype=torch.bool)
    tr, te = slice(0, 300), slice(300, n)
    k0, b0 = torch.randn(G, d) * 0.1, torch.zeros(G)

    true1 = torch.randn(G, 1, d) * 2
    m1 = _mass(q, true1, sc)
    keys, b = fit(q[:, tr], m1[tr], free[tr], k0, b0, 1, sc, steps=400)
    assert _ratio(keys, b, q[:, te], m1[te], free[te], sc) > 0.95

    true2 = torch.randn(G, 2, d) * 2
    m2 = _mass(q, true2, sc)
    r = {}
    for m in (1, 4):
        keys, b = fit(q[:, tr], m2[tr], free[tr], k0, b0, m, sc, steps=400)
        r[m] = _ratio(keys, b, q[:, te], m2[te], free[te], sc)
    assert r[4] > r[1] + 0.05, r


def test_hit_is_the_key_leaf_selected_by_a_retrieval_row():
    """run_group's hit: under the oracle, retrieval rows whose queries align with the key
    leaf's tokens select it; anti-aligned, they do not."""
    import argparse

    import gist_beam as gb
    from experiments.route_fit import run_group
    from tests.test_beam_oracle import _tree

    tree = _tree()
    T, d, h = tree.level.numel(), 16, 2
    raw = torch.nonzero(tree.level < 0).flatten()
    rows = raw[-12:]
    n_ret = 4
    leaf = torch.tensor([3])                                    # far back: closed, not forced
    toks = torch.nonzero(tree.compressible & (tree.leaf == 3)).flatten()
    u = torch.zeros(d)
    u[0] = 1.0
    q = torch.randn(h, rows.numel(), d) * 0.1
    q[:, -n_ret:] = u
    args = argparse.Namespace(k=2, steps=1, lr=0.05)
    cfg = gb.BeamConfig(k=(2,), merge="union")
    test = torch.ones(rows.numel(), dtype=torch.bool)
    ident = (torch.ones(T, d), torch.zeros(T, d), torch.arange(T))
    for sign, want in ((10.0, True), (-10.0, False)):
        torch.manual_seed(0)
        k = torch.randn(T, d) * 0.1
        k[toks] = sign * u
        res = run_group(tree, q, k, None, rows, test, n_ret, leaf, cfg, d ** -0.5, *ident, args,
                        names=("oracle",))
        assert res["oracle"]["hit"] is want, (sign, res["oracle"])
