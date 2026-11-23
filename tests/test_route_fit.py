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
