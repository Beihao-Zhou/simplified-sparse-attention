"""CUDA fast path of DECODE_SELECT=beam (#20): kernel A (kernels/beam_select.py, Triton)
selects level by level, kernel B (kernels/beam_decode_attn.cu) attends over the listed
keys. Same rule as the reference gist_beam.py, which the tests compare against."""

import os

_EXT = None
_KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


def ext():
    """kernel B, JIT-compiled on first use."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        _EXT = load(name="gsa_beam_decode", sources=[os.path.join(_KDIR, "beam_decode_attn.cu")],
                    extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
    return _EXT


import torch  # noqa: E402

BIG = 1 << 30


def _pow2(n: int) -> int:
    return max(16, 1 << max(0, int(n - 1).bit_length()))


class StepMeta:
    """What every layer of one decode step needs, from the step's tree (gist_beam.Tree)."""

    def __init__(self, tree, prev_chunks: int, raw_window: int, window_turns: int):
        import gist_beam as gb

        T = tree.level.numel()
        dev = tree.level.device
        p = torch.tensor([T - 1], device=dev)
        self.L = tree.num_levels
        self.gpos = [g.int().contiguous() for g in tree.gpos]
        self.closed = [int(g.numel()) for g in tree.gpos]           # all summaries precede p
        self.lo = int(gb._forced_leaf_lo(tree, p, prev_chunks, raw_window, window_turns)[0])
        comp = torch.nonzero(tree.compressible).flatten()
        n0 = self.closed[0]
        self.leaf_pos = comp[: 4 * n0].view(n0, 4).int() if n0 else comp.new_zeros(0, 4).int()
        open_raw = comp[4 * n0:]
        self.always = torch.cat([torch.nonzero(tree.always).flatten(), open_raw]).int()


def select_fast(meta: StepMeta, q, K, beta, cfg, scaling):
    """Level-wise beam selection with kernel A for all KV groups. q [num_kv, HPG, D],
    K [num_kv, T, D], beta [num_kv, T]. Returns per level (cand [num_kv, C] int32 with -1
    empty, sel [num_kv, C] bool)."""
    from kernels.beam_select import beam_level

    num_kv, hpg, _ = q.shape
    dev = q.device
    out = [None] * meta.L
    up = None
    for l in range(meta.L - 1, -1, -1):
        G = meta.closed[l]
        lo_l = meta.lo // 4 ** l
        if up is None:  # top level: every closed node
            C = _pow2(G)
            ids = torch.arange(C, device=dev)
            cand = torch.where(ids < G, ids, -1).int()[None].expand(num_kv, C).contiguous()
        else:
            cand_up, sel_up = up
            n_forced_up = max(0, meta.closed[l + 1] - meta.lo // 4 ** (l + 1))
            per = cfg.k_at(l + 1) * (hpg if cfg.merge == "union" else 1)
            smax = min(cand_up.shape[1], per + n_forced_up)
            par = torch.where(sel_up, cand_up, BIG).sort(dim=1).values[:, :smax]
            par = torch.cat([par, torch.full((num_kv, 1), meta.closed[l + 1], device=dev,
                                             dtype=par.dtype)], 1)  # the open parent
            child = (par[:, :, None] * 4 + torch.arange(4, device=dev)).flatten(1)
            child = torch.where((par[:, :, None] < BIG).expand(-1, -1, 4).flatten(1) & (child < G),
                                child, BIG)
            C = _pow2(child.shape[1])
            child = torch.nn.functional.pad(child.sort(dim=1).values, (0, C - child.shape[1]), value=BIG)
            cand = torch.where(child < BIG, child, -1).int().contiguous()
        forced = ((cand >= lo_l) & (cand >= 0)).to(torch.int8)
        picked, _ = beam_level(q, K, beta, meta.gpos[l], cand, forced, cfg.k_at(l), scaling,
                               union=cfg.merge == "union")
        sel = picked | forced.bool()
        out[l] = (cand, sel)
        up = (cand, sel)
    return out


def key_list(meta: StepMeta, levels, coarse: bool):
    """[num_kv, cap] int32 key positions for kernel B (-1 = empty)."""
    cand0, sel0 = levels[0]
    num_kv = cand0.shape[0]
    parts = []
    if meta.closed[0]:
        leaf = torch.where(sel0, cand0, 0).long()
        pos = meta.leaf_pos[leaf]                                      # [num_kv, C0, 4]
        parts.append(torch.where(sel0[..., None], pos, -1).flatten(1))
    for l, (cand, sel) in enumerate(levels):
        show = (cand >= 0) & ((~sel) if coarse else sel)
        if meta.closed[l]:
            parts.append(torch.where(show, meta.gpos[l][cand.clamp(min=0).long()], -1))
    parts.append(meta.always[None].expand(num_kv, -1))
    return torch.cat(parts, 1).int().contiguous()


_META = {"key": None, "meta": None}


@torch.no_grad()
def beam_decode_fast(module, query, key, value, tree, scaling, cfg):
    """One decode step of one layer on kernels A and B; same contract as
    gist_beam.beam_decode."""
    from gist_utils import Global_data

    mkey = (id(tree), tree.level.numel())
    if _META["key"] != mkey:
        _META["key"] = mkey
        _META["meta"] = StepMeta(tree, Global_data.num_previous_chunks, Global_data.raw_window,
                                 Global_data.window_turns)
    meta = _META["meta"]
    H, num_kv, T = query.shape[1], key.shape[1], key.shape[2]
    hpg = H // num_kv
    beta = getattr(module, "_beta_cache", None)
    beta = torch.zeros(num_kv, T, device=query.device) if beta is None else beta[0].float().contiguous()
    K = key[0].contiguous()
    levels = select_fast(meta, query[0, :, 0].view(num_kv, hpg, -1), K, beta, cfg, scaling)
    idx = key_list(meta, levels, cfg.coarse)
    out = ext().beam_decode_attn(query.contiguous(), key.contiguous(), value.contiguous(), idx,
                                 beta, scaling, 8)
    module._beam_levels = levels
    return out, None
