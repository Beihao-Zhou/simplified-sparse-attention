"""routing_kl options (#19): the default is the original loss; exclude drops keys from the
true mass; scope="oracle_children" restricts candidates to children of oracle parents."""

import torch
import torch.nn.functional as F

from gist_routing import routing_kl
from src.data.gist_stream import insert_gists_with_map

GIDS = [90001, 90002, 90003]


def _old(q, q_pos, k, ids, gist_ids, scaling, beta=None):
    """routing_kl as it was before the options (commit 2d17136)."""
    logits = torch.einsum("hrd,hkd->hrk", q, k) * scaling
    raw = ~torch.isin(ids, torch.tensor(list(gist_ids)))
    losses = []
    for gid in gist_ids:
        is_g = ids == gid
        g_pos = torch.nonzero(is_g).flatten()
        G = g_pos.numel()
        if G < 2:
            continue
        unit = torch.cumsum(is_g.long(), 0)
        member = raw & (unit < G)
        cand = g_pos[None, :] < q_pos[:, None]
        rows = cand.sum(-1) >= 2
        if not rows.any():
            continue
        onehot = F.one_hot(unit[member], G).to(logits.dtype)
        x = logits[..., member]
        mx = x.amax(-1, keepdim=True)
        mass = torch.einsum("hrk,kg->hrg", (x - mx).exp(), onehot).masked_fill(~cand, 0)
        p_star = mass / mass.sum(-1, keepdim=True).clamp_min(1e-30)
        z = logits[..., g_pos]
        if beta is not None:
            z = z + beta[:, None, g_pos]
        logq = F.log_softmax(z.masked_fill(~cand, float("-inf")), dim=-1)
        kl = (p_star * (p_star.clamp_min(1e-30).log() - logq.masked_fill(~cand, 0))).sum(-1)
        losses.append(kl[:, rows].mean())
    return torch.stack(losses).mean()


def _inputs(seed=0, n=700, H=3, R=9, d=8):
    g = torch.Generator().manual_seed(seed)
    raw = torch.randint(1, 999, (n,), generator=g).tolist()
    ids = torch.tensor(insert_gists_with_map(raw, GIDS, 4, 4, sink=3)[0])
    K = ids.numel()
    q = torch.randn(H, R, d, generator=g, dtype=torch.float64)
    k = torch.randn(H, K, d, generator=g, dtype=torch.float64)
    beta = torch.randn(H, K, generator=g, dtype=torch.float64)
    q_pos = torch.linspace(K // 3, K - 1, R).long()
    return q, q_pos, k, ids, beta


def test_default_is_the_original_loss():
    q, q_pos, k, ids, beta = _inputs()
    assert torch.equal(routing_kl(q, q_pos, k, ids, GIDS, 0.3, beta), _old(q, q_pos, k, ids, GIDS, 0.3, beta))


def test_exclude_drops_keys_from_the_true_mass():
    q, q_pos, k, ids, beta = _inputs(1)
    ex = torch.zeros(ids.numel(), dtype=torch.bool)
    ex[:40] = True  # e.g. a pinned prefix (raw tokens only)
    ex &= ~torch.isin(ids, torch.tensor(GIDS))
    got = routing_kl(q, q_pos, k, ids, GIDS, 0.3, beta, exclude=ex)
    # the same as giving the excluded keys a vanishing logit (they stay out of the prediction)
    q2 = torch.cat([q, torch.ones(*q.shape[:2], 1, dtype=q.dtype)], -1)
    k2 = torch.cat([k, torch.where(ex, -1e4, 0.0).to(k.dtype)[None, :, None].expand(k.shape[0], -1, 1)], -1)
    ref = _old(q2, q_pos, k2, ids, GIDS, 0.3, beta)
    torch.testing.assert_close(got, ref, rtol=1e-9, atol=1e-12)


def test_oracle_children_with_a_wide_oracle_is_the_level_scope():
    q, q_pos, k, ids, beta = _inputs(2)
    wide = routing_kl(q, q_pos, k, ids, GIDS, 0.3, beta, scope="oracle_children", oracle_k=10**6)
    torch.testing.assert_close(wide, routing_kl(q, q_pos, k, ids, GIDS, 0.3, beta), rtol=0, atol=0)
    narrow = routing_kl(q, q_pos, k, ids, GIDS, 0.3, beta, scope="oracle_children", oracle_k=2)
    assert torch.isfinite(narrow) and not torch.equal(narrow, wide)


def test_oracle_children_has_gradients():
    q, q_pos, k, ids, beta = _inputs(3)
    k = k.clone().requires_grad_(True)
    routing_kl(q, q_pos, k, ids, GIDS, 0.3, beta, scope="oracle_children", oracle_k=3).backward()
    is_sum = torch.isin(ids, torch.tensor(GIDS))
    assert k.grad[:, is_sum].abs().sum() > 0
