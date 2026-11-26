"""Decoupled routing keys (#34): a per-layer head whose keys and bias only rank summaries for
the beam selection (gist_beam.py); attention keeps the model's own k_gist and beta.

Per KV group, with f the summary's attention input x, or [x, mean x of its children]:

    h = gelu(W1 LayerNorm(f)),  route key_j = k_gist + RoPE(B_j h),  route bias = beta + v . h + c[l]
    score = logsumexp_j q . key_j * scaling + route bias            (j < m keys per node)

B (for m = 1), v and c start at 0, so an untrained head ranks as the model does. Everything a
head reads is known once a node is closed, so a routing key per node can be stored at prefill
and decode reads as many keys as today.

    attach_route_heads(model, torch.load(path))   # from experiments/route_amortize.py --save
"""

from typing import Optional

import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import rotate_half

import gist_beam as gb


class RouteCore(torch.nn.Module):
    """One KV group: features [N, F] at level-l nodes -> delta keys [N, m, d] (before RoPE)
    and delta bias [N]."""

    def __init__(self, hidden: int, d: int, r: int, levels: int, kids: bool, m: int = 1):
        super().__init__()
        self.kids, self.m = kids, m
        f = hidden * (2 if kids else 1)
        self.norm = torch.nn.LayerNorm(f)
        self.W1 = torch.nn.Linear(f, r)
        # identical keys under logsumexp get identical gradients and never separate
        self.B = torch.nn.Parameter(torch.zeros(m, d, r) if m == 1 else 0.01 * torch.randn(m, d, r))
        self.v = torch.nn.Parameter(torch.zeros(r))
        self.c = torch.nn.Parameter(torch.zeros(levels))

    def forward(self, f: torch.Tensor, level: torch.Tensor):
        h = F.gelu(self.W1(self.norm(f)))
        return torch.einsum("nr,mdr->nmd", h, self.B), h @ self.v + self.c[level]


def rope(dk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """dk [N, m, d], cos / sin [N, d]."""
    return dk * cos[:, None] + rotate_half(dk) * sin[:, None]


def children_mean(x: torch.Tensor, tree: gb.Tree, fanout: int = 4):
    """Per level: [G_l, hidden] mean x of each node's children (level l-1 summaries, raw
    tokens for leaves). x [T, hidden]."""
    G0 = tree.gpos[0].numel()
    comp = tree.compressible & (tree.leaf >= 0) & (tree.leaf < G0)
    out = []
    for lv in range(tree.num_levels):
        if lv == 0:
            src, par = x[comp], tree.leaf[comp]
        else:
            src = x[tree.gpos[lv - 1]]
            par = torch.arange(src.shape[0], device=x.device) // fanout
        G = tree.gpos[lv].numel()
        keep = par < G
        s = torch.zeros(G, x.shape[1], device=x.device, dtype=torch.float32)
        s.index_add_(0, par[keep], src[keep].float())
        n = torch.zeros(G, device=x.device).index_add_(
            0, par[keep], torch.ones(int(keep.sum()), device=x.device))
        out.append(s / n.clamp_min(1)[:, None])
    return out


class RouteHead(torch.nn.Module):
    """One layer: a RouteCore per KV group."""

    def __init__(self, cores):
        super().__init__()
        self.cores = torch.nn.ModuleList(cores)
        self.m = cores[0].m

    @torch.no_grad()
    def route(self, x: torch.Tensor, tree: gb.Tree, cos: torch.Tensor, sin: torch.Tensor,
              key: torch.Tensor, beta: Optional[torch.Tensor]):
        """One sequence. x [T, hidden]; cos / sin [T, d]; key [H_kv, T, d] (post-RoPE);
        beta [H_kv, T] or None -> route keys [H_kv, T, m, d] (or [H_kv, T, d] for m = 1) and
        route bias [H_kv, T]; raw positions keep the model's key and a zero delta."""
        H_kv, T, d = key.shape
        keys = key.float()[:, :, None].expand(H_kv, T, self.m, d).clone()
        bias = (torch.zeros(H_kv, T, device=key.device) if beta is None else beta.float()).clone()
        kids = children_mean(x, tree) if self.cores[0].kids else None
        for lv, gp in enumerate(tree.gpos):
            if gp.numel() == 0:
                continue
            f = x[gp].float()
            if kids is not None:
                f = torch.cat([f, kids[lv]], -1)
            lvl = torch.full((gp.numel(),), lv, device=key.device, dtype=torch.long)
            for g, core in enumerate(self.cores):
                dk, db = core(f, lvl)
                keys[g, gp] += rope(dk, cos[gp].float(), sin[gp].float())
                bias[g, gp] += db
        return (keys[:, :, 0] if self.m == 1 else keys), bias


def attach_route_heads(model, saved: dict) -> None:
    """saved: {"config": {hidden, d, r, levels, kids, m}, "layers": {layer: [state per KV
    group]}}. Sets attn.route_head and records each layer's RoPE tables for it."""
    cfg = saved["config"]
    dev = model.model.layers[0].self_attn.q_proj.weight.device
    for layer, states in saved["layers"].items():
        cores = []
        for st in states:
            c = RouteCore(cfg["hidden"], cfg["d"], cfg["r"], cfg["levels"], cfg["kids"], cfg["m"])
            c.load_state_dict(st)
            cores.append(c)
        attn = model.model.layers[int(layer)].self_attn
        attn.route_head = RouteHead(cores).to(dev).eval()
        attn.register_forward_pre_hook(_capture_rope, with_kwargs=True)


def _capture_rope(module, args, kwargs):
    pe = kwargs.get("position_embeddings")
    if pe is not None:
        module._route_rope = pe
    x = kwargs.get("hidden_states", args[0] if args else None)
    if x is not None:
        module._route_x = x


def selection_inputs(module, key: torch.Tensor, beta: Optional[torch.Tensor], ids: torch.Tensor,
                     features=None):
    """(keys, beta) the selection should rank with: the route head's when the layer has one,
    else the model's. key [B, H_kv, T, d] and beta [B, H_kv, T] in original order."""
    head = getattr(module, "route_head", None)
    if head is None:
        return key, beta
    from gist_utils import Global_data

    cos, sin = module._route_rope
    x = module._route_x
    ks, bs = [], []
    for b in range(key.shape[0]):
        tree = gb.build_tree(ids[b], Global_data.gist_token_id, Global_data.sink_size,
                             None if features is None or features.pinned is None
                             else features.pinned[b],
                             None if features is None or features.turn is None
                             else features.turn[b])
        k, bias = head.route(x[b], tree, cos[b], sin[b], key[b],
                             None if beta is None else beta[b])
        ks.append(k)
        bs.append(bias)
    return torch.stack(ks).to(key.dtype if head.m == 1 else torch.float32), torch.stack(bs)
