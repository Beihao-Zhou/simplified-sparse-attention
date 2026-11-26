"""gist_route (#34): an untrained head ranks as the model does, the in-model route keys match
the offline scoring that trained them (experiments/route_amortize.py), and m keys per node
score as route_fit's logsumexp."""

import torch
from transformers.models.qwen2.modeling_qwen2 import rotate_half

import gist_beam as gb
from gist_route import RouteCore, RouteHead, children_mean, rope
from src.data.gist_stream import insert_gists_with_map

GIST = [1000, 1001, 1002]
H_KV, HPG, D, HID = 2, 3, 8, 16


def _setup(T_raw=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids, _ = insert_gists_with_map(list(range(T_raw)), GIST, 4, 4, sink=3)
    ids = torch.tensor(ids)
    tree = gb.build_tree(ids, GIST, 3)
    T = ids.numel()
    x = torch.randn(T, HID, generator=g)
    key = torch.randn(H_KV, T, D, generator=g)
    beta = torch.randn(H_KV, T, generator=g)
    ang = torch.randn(T, D // 2, generator=g)
    cos, sin = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)
    return ids, tree, x, key, beta, cos, sin


def _head(kids, m, trained, seed=1):
    torch.manual_seed(seed)
    cores = [RouteCore(HID, D, 6, 3, kids, m) for _ in range(H_KV)]
    if trained:
        for c in cores:
            for p in (c.B, c.v, c.c):
                p.data.normal_()
    return RouteHead(cores)


def test_untrained_head_keeps_the_model_keys():
    _, tree, x, key, beta, cos, sin = _setup()
    for kids in (False, True):
        k, b = _head(kids, 1, trained=False).route(x, tree, cos, sin, key, beta)
        assert torch.equal(k, key) and torch.equal(b, beta)


def test_route_keys_match_offline_scoring():
    _, tree, x, key, beta, cos, sin = _setup()
    head = _head(True, 1, trained=True)
    k, b = head.route(x, tree, cos, sin, key, beta)
    cm = children_mean(x, tree)
    for lv, gp in enumerate(tree.gpos):
        f = torch.cat([x[gp], cm[lv]], -1)
        for g, core in enumerate(head.cores):
            dk, db = core(f, torch.full((gp.numel(),), lv))
            want = key[g, gp] + (dk * cos[gp][:, None] + rotate_half(dk) * sin[gp][:, None])[:, 0]
            torch.testing.assert_close(k[g, gp], want)
            torch.testing.assert_close(b[g, gp], beta[g, gp] + db)
    raw = tree.level < 0
    assert torch.equal(k[:, raw], key[:, raw]) and torch.equal(b[:, raw], beta[:, raw])


def test_children_mean_of_a_leaf_is_its_raw_tokens():
    _, tree, x, *_ = _setup()
    comp = torch.nonzero(tree.compressible).flatten()
    torch.testing.assert_close(children_mean(x, tree)[0][0], x[comp[:4]].mean(0))


def test_m_keys_score_by_logsumexp():
    _, tree, x, key, beta, cos, sin = _setup()
    k, b = _head(False, 3, trained=True).route(x, tree, cos, sin, key, beta)
    assert k.shape == (H_KV, key.shape[1], 3, D)
    q = torch.randn(HPG, 5, D)
    rows = torch.nonzero(tree.level < 0).flatten()[-5:]
    cfg = gb.BeamConfig(k=(2,))
    s = gb.select(tree, q, k[0], b[0], rows, cfg, 0.3)
    top = len(tree.gpos) - 1
    gp = tree.gpos[top]
    ref = (torch.einsum("hnd,gmd->nhgm", q, k[0, gp]) * 0.3).logsumexp(-1) + b[0, gp]
    c = gb._closed(tree, rows, top)
    f = gb._forced(tree, rows, top, gb._forced_leaf_lo(tree, rows, 1, 0, 0), 4) & c
    assert torch.equal(s.sel[top], f | gb._pick(ref, c & ~f, 2, "union"))


def test_rope_matches_the_model_rotation():
    dk = torch.randn(4, 2, D)
    ang = torch.randn(4, D // 2)
    cos, sin = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)
    torch.testing.assert_close(rope(dk, cos, sin)[:, 1],
                               dk[:, 1] * cos + rotate_half(dk[:, 1]) * sin)
