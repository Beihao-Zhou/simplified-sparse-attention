import torch

from experiments.deep_recall import route, tree
from src.data.gist_stream import insert_gists_with_map

GIDS = [90001, 90002, 90003]  # outside the raw ids below


def _tree():
    ids, _ = insert_gists_with_map(list(range(2000)), GIDS, 4, 4, sink=3)
    return torch.tensor(ids), tree(torch.tensor(ids), GIDS, 3)


def test_parents_cover_their_children():
    ids, (lvl, gpos, unit, raw) = _tree()
    for l in range(len(GIDS) - 1):
        parent = unit[l + 1][gpos[l]]
        closed = parent < len(gpos[l + 1])
        # a child summary precedes its parent's summary, and its unit is in the parent's span
        assert (gpos[l][closed] < gpos[l + 1][parent[closed]]).all()
        assert torch.bincount(parent[closed]).eq(4).all()  # fanout 4
    assert not raw[:3].any()


def test_wide_beam_equals_flat_topk():
    ids, (lvl, gpos, unit, raw) = _tree()
    G = [len(g) for g in gpos]
    last_top = int(gpos[-1][-1])
    G = [int((g <= last_top).sum()) for g in gpos]
    parent = [unit[l + 1][gpos[l][: G[l]]] for l in range(len(GIDS) - 1)]
    z = [torch.randn(G[l], dtype=torch.float64) for l in range(len(GIDS))]
    k0 = 5
    sel, _, _ = route(z, parent, G, k0, lambda l, k, G: 10**6)
    flat = torch.topk(z[0], k0).indices
    assert set(sel[0].tolist()) == set(flat.tolist())
    narrow, _, cost = route(z, parent, G, k0, lambda l, k, G: 1)
    assert len(narrow[-1]) == 1 and cost < sum(G)
