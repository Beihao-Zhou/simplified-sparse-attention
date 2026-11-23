"""Stage 1 with selection at every position (#29, ADR 0003): the "sel" student without a
dense [T, T] mask.

Under the decode-time beam selection (gist_beam.py) the keys a raw row p sees split into
two disjoint parts:

  S(p)  what p sees whatever is selected: pinned and sink tokens, the compressible raw
        tokens of the forced leaves (the previous P chunks, the raw window, the open
        chunk: leaf >= lo(p)) and, without coarse terms, the summaries of the forced nodes
        (id >= lo(p) // 4^l at level l). In the key order [raw | L0 | L1 | ...] these are
        runs, so flex computes them on full blocks.
  X(p)  what the picks add: the raw tokens of the picked level-0 nodes and the summaries
        of the picked nodes (without coarse terms) or of the unpicked candidates (with
        them). A few hundred keys per row and KV group, scattered over the compressed
        part: as a block mask almost every block would be partial.

S goes through flex (with return_lse); X through a gather attention over per-row key
lists; the two are merged by their log-sum-exp. Rows that do not select (summary rows,
rows before rows_from) keep the gist mask, entirely in flex, and have empty lists. The
selection carries no gradient. `select_attention` here is the PyTorch reference of the
gather part; its result equals the dense "sel" mask of gist_flex_attention.
"""

from typing import Optional

import torch

import gist_beam as gb
from gist_utils import Global_data


def _features(ids, f, b, rows_from):
    """The tree of sequence b, every row's lo (the lowest forced leaf) and its selecting rows."""
    tree = gb.build_tree(ids[b], Global_data.gist_token_id, Global_data.sink_size,
                         None if f.pinned is None else f.pinned[b],
                         None if f.turn is None else f.turn[b])
    T = ids.shape[1]
    pos = torch.arange(T, device=ids.device)
    lo = gb._forced_leaf_lo(tree, pos, Global_data.num_previous_chunks, Global_data.raw_window,
                            Global_data.window_turns)
    sel_rows = (tree.level < 0) & (pos >= rows_from)
    return tree, lo, sel_rows


def structured_mask_mod(f, trees, lo, sel_rows, perm, coarse: bool):
    """mask_mod (shared by heads) of S for the selecting rows and of the gist mask for the
    others; keys in perm order. trees / lo / sel_rows per sequence."""
    from src.data.gist_flex_mask import _in_range, make_gist_mask_mod

    hier = make_gist_mask_mod(f, kv_perm=perm)
    B, T = perm.shape
    L = trees[0].num_levels
    dev = perm.device
    k_level = torch.stack([t.level for t in trees]).gather(1, perm)
    nid = torch.full((B, T), -1, dtype=torch.long, device=dev)
    for b, t in enumerate(trees):
        for l in range(L):
            nid[b, t.gpos[l]] = torch.arange(t.gpos[l].numel(), device=dev)
    k_nid = nid.gather(1, perm)
    k_leaf = torch.stack([t.leaf for t in trees]).gather(1, perm)
    k_comp = torch.stack([t.compressible for t in trees]).gather(1, perm)
    k_always = torch.stack([t.always for t in trees]).gather(1, perm)
    lo = torch.stack(lo)                                                   # [B, T]
    lo_l = torch.stack([lo // 4 ** l for l in range(L)], 1)                # [B, L, T]
    sel_rows = torch.stack(sel_rows)

    def s_mod(b, h, q_idx, kv_idx):
        k = perm[b, kv_idx]
        lk = k_level[b, kv_idx]
        vis = (lk < 0) & (k_always[b, kv_idx] | (k_comp[b, kv_idx] & (k_leaf[b, kv_idx] >= lo[b, q_idx])))
        if not coarse:
            vis = vis | ((lk >= 0) & (k < q_idx)
                         & (k_nid[b, kv_idx] >= lo_l[b, lk.clamp(min=0), q_idx]))
        return (k <= q_idx) & vis

    s_mod = _in_range(s_mod, T)

    def mod(b, h, q_idx, kv_idx):
        on = sel_rows[b, q_idx.clamp(max=T - 1)] & (q_idx < T)
        return torch.where(on, s_mod(b, h, q_idx, kv_idx), hier(b, h, q_idx, kv_idx))

    return mod


def _compact(vals: torch.Tensor) -> torch.Tensor:
    """[n, m] with -1 holes -> [n, max count], the kept values first (order not kept)."""
    keep = vals >= 0
    n_max = int(keep.sum(1).max()) if vals.numel() else 0
    order = torch.sort(keep.to(torch.int8), dim=1, descending=True, stable=True).indices
    return vals.gather(1, order[:, :max(n_max, 1)])


def _cat_lists(lists, stack: bool = False) -> torch.Tensor:
    """Key lists of different widths, padded with -1 to the widest: concatenated along the
    rows ([n, c]) or stacked ([len, n, c])."""
    cap = max(l.shape[-1] for l in lists)
    padded = [torch.nn.functional.pad(l, (0, cap - l.shape[-1]), value=-1) for l in lists]
    return torch.stack(padded) if stack else torch.cat(padded, 0)


@torch.no_grad()
def selected_lists(tree, lo, rows, q, k, beta, cfg: gb.BeamConfig, scaling):
    """X for rows [n] of one KV group: [n, cap] key positions, -1 empty. q [h, n, d]."""
    s = gb.select(tree, q, k, beta, rows, cfg, scaling, Global_data.num_previous_chunks,
                  Global_data.raw_window, Global_data.window_turns)
    lo_r = lo[rows]
    parts = []
    comp = torch.nonzero(tree.compressible).flatten()
    for l in range(tree.num_levels):
        forced = gb._forced(tree, rows, l, lo_r, cfg.fanout) & s.cand[l]
        picked = s.sel[l] & ~forced
        show = (s.cand[l] & ~s.sel[l]) if cfg.coarse else picked
        if show.shape[1]:
            parts.append(torch.where(show, tree.gpos[l][None, :], -1))
        if l == 0 and picked.shape[1]:
            G0 = picked.shape[1]
            leaf_pos = comp[: 4 * G0].view(G0, 4)                        # a closed leaf's 4 tokens
            parts.append(torch.where(picked[..., None], leaf_pos[None], -1).flatten(1))
    if not parts:
        return torch.full((rows.numel(), 1), -1, dtype=torch.long, device=rows.device)
    return _compact(torch.cat(parts, 1))


def gather_attention(q, k, v, beta, idx, scaling):
    """Attention of each row over its listed keys. q [h, n, d], k / v [T, d], beta [T] or
    None, idx [n, c] (-1 empty). Returns (out [h, n, d], lse [h, n]; -inf for no key)."""
    valid = idx >= 0
    i = idx.clamp(min=0)
    kk, vv = k[i].float(), v[i].float()                                    # [n, c, d]
    s = torch.einsum("hnd,ncd->hnc", q.float(), kk) * scaling
    if beta is not None:
        s = s + beta.float()[i][None]
    s = s.masked_fill(~valid[None], float("-inf"))
    lse = torch.logsumexp(s, -1)
    p = torch.exp(s - torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))[..., None])
    return torch.einsum("hnc,ncd->hnd", p, vv), lse


def merge(o_s, lse_s, o_x, lse_x):
    """Two attentions over disjoint key sets as one: [.., n, d] outputs, [.., n] lse."""
    m = torch.maximum(lse_s, lse_x)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    ws, wx = torch.exp(lse_s - m), torch.exp(lse_x - m)
    return (ws[..., None] * o_s + wx[..., None] * o_x) / (ws + wx)[..., None]


_CACHE = {"f": None, "key": None, "val": None}


def select_attention(flex, q_in, k_att, v_att, gqa, score_mod, query, key, value, beta,
                     kwargs, sel, perm, scaling, kernel_options=None, chunk: int = 1024):
    """The "sel" student, [B, T, H, d] as gist_flex_attention's attend returns.

    q_in / k_att / v_att: flex's inputs (keys in perm order, with score_mod adding beta);
    query [B, H, T, d], key / value [B, H_kv, T, d] and beta [B, H_kv, T] in the original
    order, for the gather part."""
    from src.data.gist_flex_mask import make_block_mask

    f, ids = kwargs["gist_features"], kwargs["gist_input_ids"]
    B, H, T, d = query.shape
    H_kv = key.shape[1]
    hpg = H // H_kv
    cfg = sel["cfg"]
    # S and the trees depend on the positions only: built once per forward, shared by the
    # layers and their checkpoint recomputes (holding f itself, as an id can be reused)
    key_ = (sel["rows_from"], cfg.coarse)
    if _CACHE["f"] is not f or _CACHE["key"] != key_:
        feats = [_features(ids, f, b, sel["rows_from"]) for b in range(B)]
        trees, los, sel_rows = zip(*feats)
        mod = structured_mask_mod(f, trees, los, sel_rows, perm, cfg.coarse)
        bm = make_block_mask(mod, B, T, 128, str(query.device), query.is_cuda)
        _CACHE.update(f=f, key=key_, val=(trees, los, sel_rows, bm))
    trees, los, sel_rows, bm = _CACHE["val"]
    # this mask_mod is inlined into the kernel: with the default 128 x 64 forward tiles its
    # template needs 415 KB of shared memory (an H200 has 227 KB); 64 x 64 compiles
    opts = {"BLOCK_M": 64, "BLOCK_N": 64, **(kernel_options or {})}
    o_s, lse_s = flex(q_in, k_att, v_att, block_mask=bm, score_mod=score_mod, scale=scaling,
                      enable_gqa=gqa, return_lse=True, kernel_options=opts)
    o_x = torch.zeros_like(o_s, dtype=torch.float32)
    lse_x = torch.full_like(lse_s, float("-inf"), dtype=torch.float32)
    import os
    triton_path = query.is_cuda and os.environ.get("GIST_SEL_REF", "0") != "1"
    for b in range(B):
        rows_all = torch.nonzero(sel_rows[b]).flatten()
        if rows_all.numel() == 0:
            continue
        if triton_path:  # selection and gather in one kernel each, over all selecting rows
            from kernels.gather_attn import gather_attention as gather_kernel
            from kernels.select_rows import select_lists
            idx = select_lists(trees[b], los[b], rows_all, query[b].detach(), key[b].detach(),
                               None if beta is None else beta[b].detach(), cfg, scaling)
            o, lse = gather_kernel(query[b], key[b], value[b],
                                   None if beta is None else beta[b].float(), rows_all,
                                   idx.int(), scaling)
            o_x[b][:, rows_all] = o
            lse_x[b][:, rows_all] = lse
            continue
        for g in range(H_kv):
            hs = slice(g * hpg, (g + 1) * hpg)
            bg = None if beta is None else beta[b, g]
            for i in range(0, rows_all.numel(), chunk):
                rows = rows_all[i:i + chunk]
                idx = selected_lists(trees[b], los[b], rows, query[b, hs][:, rows].detach(),
                                     key[b, g].detach(), None if bg is None else bg.detach(),
                                     cfg, scaling)
                o, lse = gather_attention(query[b, hs][:, rows], key[b, g], value[b, g], bg,
                                          idx, scaling)
                o_x[b, hs, rows] = o
                lse_x[b, hs, rows] = lse
    out = merge(o_s.float(), lse_s.float(), o_x, lse_x)
    return out.to(o_s.dtype).transpose(1, 2).contiguous()
