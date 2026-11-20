"""Routing KL: summary scores against the true attention mass of what they summarize.

For a query at position p and one summary level, the candidates are that level's units
closed before p (their summary token is at a position < p). The label p* is the softmax
over candidates of the log-sum-exp of the query's attention logits on the unit's raw keys
(no grad); the prediction is the softmax of the logits on the units' summary keys plus
their mass bias beta. Loss = KL(p* || softmax(z)), averaged over heads, rows with at
least two candidates, and levels.
"""

from typing import Optional, Sequence

import torch
import torch.nn.functional as F


def routing_kl(
    q: torch.Tensor,
    q_pos: torch.Tensor,
    k: torch.Tensor,
    ids: torch.Tensor,
    gist_ids: Sequence[int],
    scaling: float,
    beta: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """q [H, R, d] at positions q_pos [R]; k [H, K, d] for ids [K]; beta [H, K] or None."""
    logits = torch.einsum("hrd,hkd->hrk", q, k) * scaling  # [H, R, K]
    raw = ~torch.isin(ids, torch.tensor(list(gist_ids), device=ids.device))
    losses = []
    for gid in gist_ids:
        is_g = ids == gid
        g_pos = torch.nonzero(is_g).flatten()  # [G]
        G = g_pos.numel()
        if G < 2:
            continue
        unit = torch.cumsum(is_g.long(), 0)  # raw key -> its unit
        member = raw & (unit < G)
        cand = g_pos[None, :] < q_pos[:, None]  # [R, G]
        rows = cand.sum(-1) >= 2
        if not rows.any():
            continue
        onehot = F.one_hot(unit[member], G).to(logits.dtype)  # [Kraw, G]
        with torch.no_grad():
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
    return torch.stack(losses).mean() if losses else logits.sum() * 0
