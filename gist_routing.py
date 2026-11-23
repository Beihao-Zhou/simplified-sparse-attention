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


def unit_exclude(ids: torch.Tensor, gist_ids: Sequence[int], sink: int,
                 pinned: Optional[torch.Tensor] = None) -> torch.Tensor:
    """[..., K] bool: raw keys that belong to no unit, the pinned segment and the sink (the
    first `sink` unpinned raw tokens, as src/data/gist_stream.py never chunks them). The
    sink holds 25-45% of the attention mass in middle and late layers (0.5B, 8K), all of
    which would otherwise land on unit 0 of every level although the sink is always
    visible."""
    raw = ~torch.isin(ids, torch.tensor(list(gist_ids), device=ids.device))
    pin = torch.zeros_like(raw) if pinned is None else pinned.to(device=ids.device, dtype=torch.bool)
    free = raw & ~pin
    return pin | (free & (torch.cumsum(free.long(), -1) <= max(sink, 0)))


def routing_kl(
    q: torch.Tensor,
    q_pos: torch.Tensor,
    k: torch.Tensor,
    ids: torch.Tensor,
    gist_ids: Sequence[int],
    scaling: float,
    beta: Optional[torch.Tensor] = None,
    exclude: Optional[torch.Tensor] = None,
    scope: str = "level",
    oracle_k: int = 256,
    lo_leaf: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """q [H, R, d] at positions q_pos [R]; k [H, K, d] for ids [K]; beta [H, K] or None.

    exclude [K] bool: raw keys in no unit (pinned segments, the sink), dropped from the
    true mass. scope "oracle_children": below the top level, the candidates are only the
    children of the oracle_k level-(l+1) units with the most true mass (per head and row)
    and the children of the still-open parent, as decode compares only the children of
    the selected parents (#19); "level" (default) uses every closed unit of the level.
    lo_leaf [R]: each row's lowest forced leaf (gist_beam._forced_leaf_lo). Given, forced
    units (id >= lo_leaf // 4^l at level l) leave the softmax, which then compares only the
    units the beam picks among (#33); the forced recent ones would hold most of the mass.
    """
    if scope not in ("level", "oracle_children"):
        raise ValueError(f"unknown routing scope {scope!r}")
    logits = torch.einsum("hrd,hkd->hrk", q, k) * scaling  # [H, R, K]
    raw = ~torch.isin(ids, torch.tensor(list(gist_ids), device=ids.device))
    if exclude is not None:
        raw = raw & ~exclude
    levels = []
    for gid in gist_ids:
        is_g = ids == gid
        g_pos = torch.nonzero(is_g).flatten()  # [G]
        G = g_pos.numel()
        unit = torch.cumsum(is_g.long(), 0)  # raw key -> its unit
        member = raw & (unit < G)
        cand = g_pos[None, :] < q_pos[:, None]  # [R, G]
        with torch.no_grad():
            x = logits[..., member]
            mx = x.amax(-1, keepdim=True) if x.numel() else x.sum(-1, keepdim=True)
            # sum exp per unit by scatter: a dense [raw keys, units] one-hot is O(T^2 / 4)
            # (17 GB per level at 128K)
            mass = torch.zeros(*x.shape[:2], max(G, 1), dtype=x.dtype, device=x.device)
            mass.scatter_add_(-1, unit[member].expand(*x.shape[:2], -1), (x - mx).exp())
            mass = mass[..., :G].masked_fill(~cand, 0)
        levels.append((is_g, g_pos, cand, mass))
    losses = []
    H, R = logits.shape[:2]
    for li, (is_g, g_pos, cand, mass) in enumerate(levels):
        G = g_pos.numel()
        if G < 2:
            continue
        c = cand[None].expand(H, R, G)
        if scope == "oracle_children" and li + 1 < len(levels):
            is_up, up_pos, cand_up, mass_up = levels[li + 1]
            parent = torch.cumsum(is_up.long(), 0)[g_pos]  # level-(l+1) unit of each node
            n_up = up_pos.numel()
            with torch.no_grad():
                oracle = torch.zeros(H, R, n_up + 1, dtype=torch.bool, device=ids.device)
                if n_up:
                    m = mass_up.masked_fill(~cand_up[None], -1.0)
                    top = torch.topk(m, min(oracle_k, n_up), dim=-1).indices
                    oracle[..., :n_up].scatter_(-1, top, True)
                    oracle[..., :n_up] &= cand_up[None]
                closed_up = cand_up.sum(-1)  # [R]
                open_parent = parent[None, :] >= closed_up[:, None]  # [R, G]
                under = oracle.gather(-1, parent.clamp(max=n_up)[None, None].expand(H, R, G))
            c = c & (under | open_parent[None])
        if lo_leaf is not None:
            node = torch.arange(G, device=ids.device)
            c = c & (node[None, :] < (lo_leaf // 4 ** li)[:, None])[None]
        rows = c.sum(-1) >= 2  # [H, R]
        if not rows.any():
            continue
        with torch.no_grad():
            mk = mass.masked_fill(~c, 0)
            p_star = mk / mk.sum(-1, keepdim=True).clamp_min(1e-30)
        z = logits[..., g_pos]
        if beta is not None:
            z = z + beta[:, None, g_pos]
        logq = F.log_softmax(z.masked_fill(~c, float("-inf")), dim=-1)
        kl = (p_star * (p_star.clamp_min(1e-30).log() - logq.masked_fill(~c, 0))).sum(-1)
        losses.append(kl[rows].mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0
