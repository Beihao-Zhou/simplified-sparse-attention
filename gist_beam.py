"""Level-wise beam selection over the gist tree: the reference for DECODE_SELECT=beam.

Decisions in docs/adr/0001-long-agent-trajectory-infra.md and 0002-selection-during-training.md
(#19). For a query at position p and one KV group:

* Tree. The i-th level-l summary is node i of level l; its children are nodes 4i..4i+3
  of level l-1 (streaming insertion guarantees this), and a level-0 node covers the 4
  compressible raw tokens before its gist. Pinned tokens (system prompt, tool
  definitions) and the sink belong to no node and are always visible.
* Candidates. All closed top-level nodes; below the top, the closed children of the
  selected nodes and of the open node on the query's own path (its ancestors that are
  not closed yet), so the frontier after the last top-level summary is selectable too.
* Forced nodes. Leaves whose raw tokens the gist mask already shows the query (its
  previous P chunks and the raw window, R2/R6), and their ancestors, are always
  expanded and do not count against k. With coarse terms this keeps the partition exact.
* Selection. Scores scaling * q.k + beta. Each level keeps the forced nodes plus k free
  candidates: "union" takes a candidate if any head of the group ranks it in its top k,
  "sumexp" ranks the group's summed exp-scores (RecSA). Ties go to the lower node id.
* Visible keys. Pinned, sink, the raw tokens of selected and open leaves; and summaries:
  with coarse terms the unselected candidates of every level (selected nodes show only
  their children, so the prefix is partitioned exactly and full expansion is plain causal
  attention over the raw tokens), without them the selected nodes' own summaries (SSA's
  SG + SR).

`select` handles any set of query rows at once (decode: the last row; #28: every row),
with dense [rows, nodes] masks per level. Selections carry only the selected node ids;
`visible` rebuilds the key set from them, so a replay uses exactly what a rollout chose.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch


@dataclass
class BeamConfig:
    k: Sequence[int] = (256,)   # free nodes kept per level (level 0 first); the last repeats
    merge: str = "union"        # "union" | "sumexp"
    coarse: bool = False        # unselected candidates stay as coarse terms (exact partition)
    fanout: int = 4
    oracle: bool = False        # score nodes by true attention mass on their raw tokens (#31)

    def k_at(self, level: int) -> int:
        return int(self.k[min(level, len(self.k) - 1)])


@dataclass
class Tree:
    level: torch.Tensor        # [T] -1 for raw tokens, l for a level-l summary
    gpos: List[torch.Tensor]   # per level: positions of its summaries, in order
    leaf: torch.Tensor         # [T] level-0 node of a compressible raw token (gists before it)
    compressible: torch.Tensor  # [T] raw, not pinned, past the sink
    always: torch.Tensor       # [T] pinned or sink: visible to every later query
    turn: Optional[torch.Tensor]
    num_levels: int


def build_tree(ids: torch.Tensor, gist_ids: Sequence[int], sink: int,
               pinned: Optional[torch.Tensor] = None, turn: Optional[torch.Tensor] = None) -> Tree:
    """ids [T] (gisted, as produced by src/data/gist_stream.py), pinned / turn [T]."""
    T, dev = ids.numel(), ids.device
    level = torch.full((T,), -1, dtype=torch.long, device=dev)
    for l, g in enumerate(gist_ids):
        level[ids == g] = l
    raw = level < 0
    pin = torch.zeros(T, dtype=torch.bool, device=dev) if pinned is None else pinned.bool().to(dev)
    unpinned_raw = raw & ~pin
    sink_tok = unpinned_raw & (torch.cumsum(unpinned_raw.long(), 0) <= sink)
    compressible = unpinned_raw & ~sink_tok
    is_g0 = level == 0
    leaf = torch.cumsum(is_g0.long(), 0) - is_g0.long()
    gpos = [torch.nonzero(level == l).flatten() for l in range(len(gist_ids))]
    return Tree(level, gpos, leaf, compressible, pin | sink_tok,
                None if turn is None else turn.long().to(dev), len(gist_ids))


def _forced_leaf_lo(tree: Tree, rows: torch.Tensor, prev_chunks: int, raw_window: int,
                    window_turns: int) -> torch.Tensor:
    """Lowest forced level-0 node per row: the P chunks before the query's own and the
    leaves of the raw window (a suffix of the prefix, so one bound suffices)."""
    closed0 = torch.searchsorted(tree.gpos[0], rows)           # level-0 nodes closed before p
    lo = closed0 - prev_chunks
    if raw_window > 0 or window_turns > 0:
        kmin = torch.zeros_like(rows)
        if raw_window > 0:
            kmin = torch.maximum(kmin, rows - raw_window + 1)
        if window_turns > 0:
            first = torch.searchsorted(tree.turn, tree.turn[rows] - window_turns + 1)
            kmin = torch.maximum(kmin, first)
        # leaf of the first compressible raw token at or after kmin
        T = tree.level.numel()
        pos = torch.arange(T, device=rows.device)
        nxt = torch.where(tree.compressible, pos, torch.full_like(pos, T))
        nxt = torch.flip(torch.cummin(torch.flip(nxt, [0]), 0).values, [0])
        k_first = nxt[kmin.clamp(max=T - 1)]
        has = k_first <= rows
        lo = torch.where(has, torch.minimum(lo, tree.leaf[k_first.clamp(max=T - 1)]), lo)
    return lo.clamp(min=0)


@dataclass
class Selection:
    rows: torch.Tensor           # [n] query positions
    sel: List[torch.Tensor]      # per level: [n, G_l] bool, selected (forced or picked)
    cand: List[torch.Tensor]     # per level: [n, G_l] bool, candidates

    def ids(self, r: int) -> List[torch.Tensor]:
        """Selected node ids per level for row r: what a rollout records."""
        return [torch.nonzero(s[r]).flatten() for s in self.sel]


def _closed(tree: Tree, rows: torch.Tensor, l: int) -> torch.Tensor:
    return tree.gpos[l][None, :] < rows[:, None]               # [n, G_l]


def _candidates(tree: Tree, rows, l: int, sel_up: Optional[torch.Tensor], fanout: int):
    closed = _closed(tree, rows, l)
    if sel_up is None:                                         # top level
        return closed
    G = closed.shape[1]
    parent = torch.arange(G, device=rows.device) // fanout
    n_up = sel_up.shape[1]
    open_parent = parent[None, :] >= _closed(tree, rows, l + 1).sum(1, keepdim=True)
    under_sel = sel_up.gather(1, parent.clamp(max=max(n_up - 1, 0))[None].expand(rows.numel(), -1)) \
        if n_up else torch.zeros_like(closed)
    under_sel = under_sel & (parent[None, :] < n_up)
    return closed & (under_sel | open_parent)


def _forced(tree: Tree, rows, l: int, lo_leaf: torch.Tensor, fanout: int):
    G = tree.gpos[l].numel()
    node = torch.arange(G, device=rows.device)
    return _closed(tree, rows, l) & (node[None, :] >= (lo_leaf // fanout ** l)[:, None])


def _pick(scores: torch.Tensor, free: torch.Tensor, k: int, merge: str) -> torch.Tensor:
    """scores [n, h, G], free [n, G] -> picked [n, G]; ties go to the lower node id."""
    n, h, G = scores.shape
    if G == 0 or k <= 0:
        return torch.zeros_like(free)
    neg = torch.finfo(scores.dtype).min
    if merge == "sumexp":
        s = scores.masked_fill(~free[:, None], neg)
        mx = s.amax(dim=(1, 2), keepdim=True)
        imp = torch.where(free, torch.exp(s - mx).sum(1), torch.full_like(free, -1.0, dtype=s.dtype))
        order = torch.sort(-imp, dim=-1, stable=True).indices[:, :k]
        picked = torch.zeros_like(free).scatter_(1, order, True)
        return picked & free
    if merge != "union":
        raise ValueError(f"unknown merge {merge!r}")
    s = scores.masked_fill(~free[:, None], neg)
    order = torch.sort(-s, dim=-1, stable=True).indices[..., :k]   # [n, h, k]
    picked = torch.zeros(n, h, G, dtype=torch.bool, device=free.device).scatter_(2, order, True)
    return picked.any(1) & free


def select(tree: Tree, q: torch.Tensor, k_keys: torch.Tensor, beta: Optional[torch.Tensor],
           rows: torch.Tensor, cfg: BeamConfig, scaling: float, prev_chunks: int = 1,
           raw_window: int = 0, window_turns: int = 0,
           scores: Optional[List[torch.Tensor]] = None) -> Selection:
    """One KV group. q [h, n, d] at positions rows [n], k_keys [T, d] (or [T, m, d]: m
    routing keys per node, gist_route.py; not with cfg.oracle), beta [T] or None.
    scores: per level [n, h, G_l], used in place of the summary scores (route_fit.py)."""
    lo = _forced_leaf_lo(tree, rows, prev_chunks, raw_window, window_turns)
    L = tree.num_levels
    sel: List[Optional[torch.Tensor]] = [None] * L
    cand: List[Optional[torch.Tensor]] = [None] * L
    mass = oracle_mass(tree, q, k_keys, rows, scaling, cfg.fanout) if cfg.oracle else None
    up = None
    for l in range(L - 1, -1, -1):
        c = _candidates(tree, rows, l, up, cfg.fanout)
        f = _forced(tree, rows, l, lo, cfg.fanout) & c
        if scores is not None:
            s = scores[l]
        elif mass is not None:                                    # log mass: union / sumexp
            s = mass[l].clamp(min=1e-30).log()                    # rank as for the logits
        else:
            gk = k_keys[tree.gpos[l]]                             # [G, d] or [G, m, d]
            if gk.dim() == 3:  # m routing keys per node (gist_route.py): logsumexp
                s = (torch.einsum("hnd,gmd->nhgm", q.float(), gk.float()) * scaling
                     ).logsumexp(-1)
            else:
                s = torch.einsum("hnd,gd->nhg", q.float(), gk.float()) * scaling
            if beta is not None:
                s = s + beta[tree.gpos[l]].float()[None, None, :]
        sel[l] = f | _pick(s, c & ~f, cfg.k_at(l), cfg.merge)
        cand[l] = c
        up = sel[l]
    return Selection(rows, sel, cand)


def oracle_mass(tree: Tree, q: torch.Tensor, k_keys: torch.Tensor, rows: torch.Tensor,
                scaling: float, fanout: int = 4) -> List[torch.Tensor]:
    """Per level: [n, h, G_l], the attention mass that full causal attention over the raw
    tokens (summaries excluded) puts on each node's compressible raw tokens: the selection
    a perfect router would make with the same q and k (#31). q [h, n, d], k_keys [T, d]."""
    T = tree.level.numel()
    pos = torch.arange(T, device=rows.device)
    raw = tree.level < 0
    logit = torch.einsum("hnd,td->nht", q.float(), k_keys.float()) * scaling
    ok = raw[None, :] & (pos[None, :] <= rows[:, None])                       # [n, T]
    p = logit.masked_fill(~ok[:, None], float("-inf")).softmax(-1)           # [n, h, T]
    comp = torch.nonzero(tree.compressible).flatten()
    leaf, pc = tree.leaf[comp], p[..., comp]
    out = []
    for l in range(tree.num_levels):
        G = tree.gpos[l].numel()
        node = leaf // fanout ** l
        keep = node < G                                     # open nodes have no summary yet
        m = torch.zeros(*p.shape[:2], G, device=p.device, dtype=p.dtype)
        out.append(m.index_add_(2, node[keep], pc[..., keep]))
    return out


def from_ids(tree: Tree, rows: torch.Tensor, ids: List[List[torch.Tensor]], fanout: int = 4):
    """Rebuild a Selection from recorded ids (ids[r][l] for row r, level l)."""
    L, n = tree.num_levels, rows.numel()
    sel, cand, up = [None] * L, [None] * L, None
    for l in range(L - 1, -1, -1):
        G = tree.gpos[l].numel()
        s = torch.zeros(n, G, dtype=torch.bool, device=rows.device)
        for r in range(n):
            s[r, ids[r][l]] = True
        cand[l] = _candidates(tree, rows, l, up, fanout)
        sel[l] = s
        up = s
    return Selection(rows, sel, cand)


def visible(tree: Tree, s: Selection, coarse: bool, fanout: int = 4) -> torch.Tensor:
    """[n, T] bool: the keys each row attends to (causal included)."""
    rows = s.rows
    n, T = rows.numel(), tree.level.numel()
    pos = torch.arange(T, device=rows.device)
    causal = pos[None, :] <= rows[:, None]
    closed0 = _closed(tree, rows, 0).sum(1, keepdim=True)
    leaf = tree.leaf.clamp(max=max(s.sel[0].shape[1] - 1, 0))
    sel_leaf = s.sel[0].gather(1, leaf[None].expand(n, -1)) if s.sel[0].shape[1] else \
        torch.zeros(n, T, dtype=torch.bool, device=rows.device)
    open_leaf = tree.leaf[None, :] >= closed0
    vis = causal & (tree.always[None, :] | (tree.compressible[None, :] & (sel_leaf | open_leaf)))
    for l in range(tree.num_levels):
        show = (s.cand[l] & ~s.sel[l]) if coarse else s.sel[l]
        g = torch.zeros(n, T, dtype=torch.bool, device=rows.device)
        g[:, tree.gpos[l]] = show
        vis |= g & causal
    return vis


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, vis: torch.Tensor,
           scaling: float, beta: Optional[torch.Tensor] = None) -> torch.Tensor:
    """q [h, n, d], k / v [T, d], vis [n, T], beta [T] -> [h, n, d]."""
    logit = torch.einsum("hnd,td->hnt", q, k) * scaling
    if beta is not None:
        logit = logit + beta[None, None, :].to(logit.dtype)
    logit = logit.masked_fill(~vis[None], float("-inf"))
    return torch.einsum("hnt,td->hnd", logit.softmax(-1).to(v.dtype), v)


# ---------------------------------------------------------------------------------------
# Decode entry point (attn_candidate.attn_dispatcher, DECODE_SELECT=beam)
# ---------------------------------------------------------------------------------------

def config_from_env() -> BeamConfig:
    """BEAM_K (comma list, level 0 first), BEAM_MERGE (union|sumexp), BEAM_COARSE (0|1)."""
    import os

    return BeamConfig(
        k=tuple(int(x) for x in os.environ.get("BEAM_K", "256").split(",")),
        merge=os.environ.get("BEAM_MERGE", "union"),
        coarse=os.environ.get("BEAM_COARSE", "0") == "1",
    )


_TREE = {"key": None, "ids": None, "tree": None}


def _decode_tree(ids: torch.Tensor):
    """The tree of the current step, shared by all layers (rebuilt when ids grow)."""
    from gist_utils import Global_data

    key = (ids.data_ptr(), ids.numel(), int(ids[-1]))
    if _TREE["key"] != key or _TREE["ids"] is None:
        seg = getattr(Global_data, "beam_segments", None)  # (pinned [T], turn [T]) or None
        pinned = turn = None
        if seg is not None:
            pinned, turn = (x[: ids.numel()].to(ids.device) for x in seg)
        _TREE["key"] = key
        _TREE["ids"] = ids  # held so its data pointer cannot be reused by another tensor
        _TREE["tree"] = build_tree(ids, Global_data.gist_token_id, Global_data.sink_size,
                                   pinned, turn)
    return _TREE["tree"]


@torch.no_grad()
def beam_decode(module, query, key, value, ids, scaling):
    """One decode step of one layer: query [1, H, 1, d], key / value [1, H_kv, T, d],
    ids [1, T]. Records the selected node ids per KV group on module._beam_ids (and in
    Global_data.beam_trace when that is a list), returns (out [1, 1, H, d], None)."""
    import os

    from gist_utils import Global_data

    if query.shape[0] != 1:
        raise NotImplementedError("DECODE_SELECT=beam is batch-1 (#24)")
    cfg = config_from_env()
    T = key.shape[2]
    beta = None
    if getattr(module, "summary_bias", None) is not None:
        from gist_lora import extend_beta_cache

        beta = extend_beta_cache(module, T, query.shape[2])[0]  # [H_kv, T]
    if (os.environ.get("ATTN_IMPL", "sparse") != "ref" and query.is_cuda
            and key.dtype == torch.bfloat16):
        # kernels A and B (gist_beam_cuda.py); this function stays the reference. A session
        # keeps the tree up to date token by token; otherwise it is rebuilt for the step.
        from gist_beam_cuda import beam_decode_fast

        inc = getattr(Global_data, "beam_incremental", None)
        src = inc if inc is not None and inc.n == T else _decode_tree(ids[0])
        return beam_decode_fast(module, query, key, value, src, scaling, cfg)
    tree = _decode_tree(ids[0])
    H, H_kv = query.shape[1], key.shape[1]
    hpg = H // H_kv
    rows = torch.tensor([T - 1], device=query.device)
    outs, rec = [], []
    for g in range(H_kv):
        qg = query[0, g * hpg:(g + 1) * hpg]                   # [hpg, 1, d]
        bg = None if beta is None else beta[g]
        s = select(tree, qg, key[0, g], bg, rows, cfg, scaling,
                   prev_chunks=Global_data.num_previous_chunks,
                   raw_window=Global_data.raw_window, window_turns=Global_data.window_turns)
        vis = visible(tree, s, cfg.coarse)
        outs.append(attend(qg, key[0, g], value[0, g], vis, scaling, bg))
        rec.append(s.ids(0))
        vis_groups = vis if g == 0 else torch.cat([vis_groups, vis])
    if Global_data.opsd_record is not None:
        from gist_opsd_long import positions

        Global_data.opsd_record[module.layer_idx] = positions(vis_groups)
    module._beam_ids = rec
    trace = getattr(Global_data, "beam_trace", None)
    if isinstance(trace, list):
        trace.append((module.layer_idx, rec))
    out = torch.cat(outs, 0)[None]                             # [1, H, 1, d]
    return out.transpose(1, 2).contiguous(), None


@torch.no_grad()
def visible_all_rows(tree: Tree, q: torch.Tensor, k_keys: torch.Tensor,
                     beta: Optional[torch.Tensor], cfg: BeamConfig, scaling: float,
                     prev_chunks: int = 1, raw_window: int = 0, window_turns: int = 0,
                     chunk: int = 1024, rows_from: int = 0) -> torch.Tensor:
    """[T, T] keys of every raw row of one KV group under beam selection, for training
    forwards that select at every position (#28). Summary rows are left empty: the
    caller keeps the gist mask for them, as for raw rows before rows_from. Selection carries
    no gradient. q [h, T, d]."""
    T = tree.level.numel()
    out = torch.zeros(T, T, dtype=torch.bool, device=q.device)
    rows_all = torch.nonzero(tree.level < 0).flatten()
    rows_all = rows_all[rows_all >= rows_from]
    for i in range(0, rows_all.numel(), chunk):
        rows = rows_all[i:i + chunk]
        s = select(tree, q[:, rows], k_keys, beta, rows, cfg, scaling, prev_chunks,
                   raw_window, window_turns)
        out[rows] = visible(tree, s, cfg.coarse)
    return out
