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


_META = {"src": None, "n": None, "meta": None}


@torch.no_grad()
def beam_decode_fast(module, query, key, value, src, scaling, cfg):
    """One decode step of one layer on kernels A and B; same contract as
    gist_beam.beam_decode. src: an IncrementalTree at this step, or the step's tree."""
    from gist_utils import Global_data

    g = Global_data
    inc = isinstance(src, IncrementalTree)
    n = src.n if inc else src.level.numel()
    # once per step, shared by the layers. Holding src (not its id) matters: an id or a data
    # pointer of a freed object can come back for a new one and hand out stale buffers.
    if _META["src"] is not src or _META["n"] != n:
        _META["src"], _META["n"] = src, n
        _META["meta"] = (src.meta(g.num_previous_chunks, g.raw_window, g.window_turns) if inc else
                         StepMeta(src, g.num_previous_chunks, g.raw_window, g.window_turns))
    meta = _META["meta"]
    H, num_kv, T = query.shape[1], key.shape[1], key.shape[2]
    hpg = H // num_kv
    beta = getattr(module, "_beta_cache", None)
    beta = torch.zeros(num_kv, T, device=query.device) if beta is None else beta[0].float().contiguous()
    K = key[0].contiguous()
    levels = select_fast(meta, query[0, :, 0].view(num_kv, hpg, -1), K, beta, cfg, scaling)
    idx = key_list(meta, levels, cfg.coarse)
    # ~128 keys per split: 8 splits left a 12K-key list (an agent prompt's pinned part and
    # raw window) on 16 blocks, 1.9 ms per layer at 158K on an H200
    nsplit = max(8, min(256, -(-idx.shape[1] // 128)))
    out = ext().beam_decode_attn(query.contiguous(), key.contiguous(), value.contiguous(), idx,
                                 beta, scaling, nsplit)
    module._beam_levels = levels
    rec = Global_data.opsd_record
    if rec is not None:  # OPSD rollout: the keys this layer attended to (gist_opsd_long.py)
        rec[module.layer_idx] = idx
    return out, None


def row_kernel_ok(query, key) -> bool:
    """Kernel B can take a decode row that attends by its gist-mask row (layer 0, summary
    rows) on the DECODE_SELECT=beam path; the reference and other paths keep SDPA."""
    return (os.environ.get("DECODE_SELECT") == "beam"
            and os.environ.get("ATTN_IMPL", "sparse") != "ref" and query.is_cuda
            and key.dtype == torch.bfloat16 and query.shape[0] == 1 and query.shape[2] == 1
            and query.shape[3] in (64, 128) and query.shape[1] // key.shape[1] in (1, 2, 4, 6, 7, 8))


@torch.no_grad()
def row_decode(query, key, value, vis, beta, scaling):
    """One decode row attending exactly to the keys of vis [T] bool, with beta [num_kv, T]
    (or None) added to the logits, on kernel B: no dense mask, no repeat_kv, no sort, no
    host sync. Returns (out [1, 1, H, D], idx [num_kv, T] int32 with -1 for keys not seen,
    a view: the record of gist_opsd_long.py)."""
    num_kv, T = key.shape[1], key.shape[2]
    pos = torch.arange(T, device=key.device, dtype=torch.int32)
    idx = torch.where(vis[:T], pos, -1)[None].expand(num_kv, T)
    bias = (torch.zeros(num_kv, T, device=key.device) if beta is None
            else beta.float().contiguous())
    nsplit = max(8, min(256, -(-T // 128)))
    out = ext().beam_decode_attn(query.contiguous(), key.contiguous(), value.contiguous(),
                                 idx.contiguous(), bias, scaling, nsplit)
    return out, idx


class _Buf:
    """A GPU int32 buffer that grows by doubling; view() is the filled part."""

    def __init__(self, device, cap=1024):
        self.t = torch.empty(cap, dtype=torch.int32, device=device)
        self.n = 0

    def push(self, values):
        k = len(values)
        if self.n + k > self.t.numel():
            new = torch.empty(max(2 * self.t.numel(), self.n + k), dtype=torch.int32, device=self.t.device)
            new[: self.n] = self.t[: self.n]
            self.t = new
        self.t[self.n:self.n + k] = torch.tensor(values, dtype=torch.int32).to(self.t.device)
        self.n += k

    def view(self):
        return self.t[: self.n]


class IncrementalTree:
    """What StepMeta derives from a whole-sequence tree, kept up to date token by token
    (O(1) amortized per token), so a decode step does not rebuild the tree (O(T)). Feed
    every token the model sees, summaries included, with its pinned flag and turn."""

    def __init__(self, gist_ids, sink, device):
        self.level_of = {g: l for l, g in enumerate(gist_ids)}
        self.L, self.sink, self.device = len(gist_ids), sink, device
        self.n = 0
        self.n_unpinned_raw = 0
        self.gpos = [_Buf(device) for _ in range(self.L)]
        self.comp = _Buf(device)            # compressible raw positions, in order
        self.comp_cpu = []
        self.always = _Buf(device)          # pinned and sink positions
        self.turn_first = []                # (turn, first position) for nondecreasing turns
        self.last_turn = None
        self._pending = {"comp": [], "always": [], **{l: [] for l in range(self.L)}}

    def extend(self, tokens, pinned, turns):
        for t, p, tr in zip(tokens, pinned, turns):
            pos = self.n
            self.n += 1
            if tr != self.last_turn:
                self.turn_first.append((tr, pos))
                self.last_turn = tr
            lvl = self.level_of.get(int(t))
            if lvl is not None:
                self._pending[lvl].append(pos)
            elif p:
                self._pending["always"].append(pos)
            else:
                self.n_unpinned_raw += 1
                if self.n_unpinned_raw <= self.sink:
                    self._pending["always"].append(pos)
                else:
                    self._pending["comp"].append(pos)
                    self.comp_cpu.append(pos)
        self._flush()

    def _flush(self):
        for key, vals in self._pending.items():
            if vals:
                buf = self.comp if key == "comp" else self.always if key == "always" else self.gpos[key]
                buf.push(vals)
                vals.clear()

    def meta(self, prev_chunks: int, raw_window: int, window_turns: int) -> "StepMeta":
        """StepMeta for a decode query at the last position."""
        import bisect

        m = StepMeta.__new__(StepMeta)
        p = self.n - 1
        m.L = self.L
        m.gpos = [b.view() for b in self.gpos]
        m.closed = [b.n for b in self.gpos]
        n0 = m.closed[0]
        lo = n0 - prev_chunks
        if raw_window > 0 or window_turns > 0:
            kmin = 0
            if raw_window > 0:
                kmin = max(kmin, p - raw_window + 1)
            if window_turns > 0:
                t_q = self.last_turn
                j = bisect.bisect_left([t for t, _ in self.turn_first], t_q - window_turns + 1)
                kmin = max(kmin, self.turn_first[j][1] if j < len(self.turn_first) else self.n)
            i = bisect.bisect_left(self.comp_cpu, kmin)
            if i < len(self.comp_cpu) and self.comp_cpu[i] <= p:
                lo = min(lo, i // 4)  # leaf of the i-th compressible token
        m.lo = max(lo, 0)
        comp = self.comp.view()
        m.leaf_pos = comp[: 4 * n0].view(n0, 4)
        m.always = torch.cat([self.always.view(), comp[4 * n0:]])
        return m
