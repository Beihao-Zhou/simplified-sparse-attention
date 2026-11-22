"""Kernel A of DECODE_SELECT=beam (ADR 0001, #20): one level of the level-wise beam
selection for one decode query, all KV groups at once. Triton.

Per KV group g and candidate slot c (candidates are node ids in ascending order, -1 for an
empty slot), `beam_scores` computes s[h, c] = scaling * q[g, h] . K[g, gpos[cand]] + beta;
forced and empty slots get -inf, so only free candidates compete. `beam_pick` then keeps
k free candidates by rank-select, without sorting: higher score first, equal scores to
the lower slot (the lower node id). The k-th largest score is found bit by bit from the
top of its order-preserving 32-bit form, one count per bit, and a prefix sum settles the
ties at it. "union" picks per head (a program each) and ORs the heads; "sumexp" ranks the
group's summed exp-scores (RecSA). Scores go through a scratch buffer [num_kv, HPG_P, C].
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _scores_kernel(q_ptr, k_ptr, beta_ptr, gpos_ptr, cand_ptr, forced_ptr, out_ptr,
                   T, scaling,
                   HPG: tl.constexpr, HPG_P: tl.constexpr, D: tl.constexpr, C: tl.constexpr,
                   BLOCK_C: tl.constexpr):
    g = tl.program_id(0)
    cb = tl.program_id(1)
    hs = tl.arange(0, HPG_P)
    ds = tl.arange(0, D)
    q = tl.load(q_ptr + (g * HPG + hs[:, None]) * D + ds[None, :], mask=hs[:, None] < HPG, other=0.0)
    cs = cb * BLOCK_C + tl.arange(0, BLOCK_C)
    node = tl.load(cand_ptr + g * C + cs)
    forced = tl.load(forced_ptr + g * C + cs)
    valid = node >= 0
    pos = tl.load(gpos_ptr + tl.where(valid, node, 0), mask=valid, other=0)
    kk = tl.load(k_ptr + (g * T + pos[:, None]).to(tl.int64) * D + ds[None, :],
                 mask=valid[:, None], other=0.0)
    s = tl.dot(q.to(tl.float32), tl.trans(kk.to(tl.float32)), input_precision="ieee") * scaling
    b = tl.load(beta_ptr + g * T + pos, mask=valid, other=0.0)
    s = s + b[None, :]
    s = tl.where((valid & (forced == 0))[None, :], s, float("-inf"))
    tl.store(out_ptr + (g * HPG_P + hs[:, None]) * C + cs[None, :], s)


@triton.jit
def _ordered(x):
    """float32 -> int64 that compares like the float (-inf lowest), as an unsigned-ready key."""
    bits = x.to(tl.int32, bitcast=True)
    o = bits ^ ((bits >> 31) & 0x7FFFFFFF)          # signed-order int32
    return o.to(tl.int64) + 2147483648               # shift to [0, 2^32)


@triton.jit
def _topk(x, ok, k):
    """Top k of x among ok (all of them when fewer), higher x first and equal x to the
    lower index: the same set as ranking the keys (ordered(x) << 20) | slot. 32 counting
    passes find the k-th largest ordered value; one prefix sum settles the ties at it."""
    o = tl.where(ok, _ordered(x), 0)
    t = tl.zeros([], dtype=tl.int64)
    for b in tl.static_range(31, -1, -1):
        cand = t | (1 << b)
        cnt = tl.sum((ok & (o >= cand)).to(tl.int32), axis=0)
        t = tl.where(cnt >= k, cand, t)
    gt = ok & (o > t)
    eq = ok & (o == t)
    need = k - tl.sum(gt.to(tl.int32), axis=0)
    return gt | (eq & (tl.cumsum(eq.to(tl.int32), axis=0) <= need))


@triton.jit
def _pick_head_kernel(s_ptr, out_ptr, k, HPG_P: tl.constexpr, C: tl.constexpr):
    """"union": one program per (KV group, head), ORed into out (int32, zeroed)."""
    g = tl.program_id(0)
    h = tl.program_id(1)
    cs = tl.arange(0, C)
    s = tl.load(s_ptr + (g * HPG_P + h) * C + cs)
    picked = _topk(s, s > float("-inf"), k)
    tl.atomic_or(out_ptr + g * C + cs, picked.to(tl.int32), mask=picked)


@triton.jit
def _pick_sumexp_kernel(s_ptr, out_ptr, k, HPG: tl.constexpr, HPG_P: tl.constexpr,
                        C: tl.constexpr):
    """"sumexp": the group's summed exp-scores, one program per KV group."""
    g = tl.program_id(0)
    cs = tl.arange(0, C)
    mx = tl.full([], float("-inf"), tl.float32)
    for h in tl.static_range(HPG):
        mx = tl.maximum(mx, tl.max(tl.load(s_ptr + (g * HPG_P + h) * C + cs), axis=0))
    imp = tl.zeros([C], dtype=tl.float32)
    ok = tl.zeros([C], dtype=tl.int1)
    for h in tl.static_range(HPG):
        s = tl.load(s_ptr + (g * HPG_P + h) * C + cs)
        okh = s > float("-inf")
        imp += tl.where(okh, tl.exp(s - mx), 0.0)
        ok = ok | okh
    tl.store(out_ptr + g * C + cs, _topk(imp, ok, k).to(tl.int32))


@triton.jit(do_not_specialize=["G", "open_parent", "lo"])  # change every step: no recompiles
def _children_kernel(cand_up_ptr, sel_up_ptr, cand_ptr, forced_ptr, G, open_parent, lo,
                     C_UP: tl.constexpr, C: tl.constexpr):
    """The next level's candidates of one KV group: the children of the selected parents,
    then of the open parent, those < G, in ascending node id, compacted from slot 0; the
    caller fills cand with -1 and forced with 0. Parents are ascending in their slots and
    children < G form a prefix of each parent's 4, so prefix sums give the sorted order."""
    g = tl.program_id(0)
    s = tl.arange(0, C_UP)
    par = tl.load(cand_up_ptr + g * C_UP + s)
    keep = (tl.load(sel_up_ptr + g * C_UP + s) != 0) & (par >= 0)
    cnt = tl.where(keep, tl.minimum(tl.maximum(G - par * 4, 0), 4), 0)
    excl = tl.cumsum(cnt, axis=0) - cnt
    for j in tl.static_range(4):
        child = par * 4 + j
        ok = (j < cnt) & (excl + j < C)  # C covers them (the caller's bound); never write past
        tl.store(cand_ptr + g * C + excl + j, child, mask=ok)
        tl.store(forced_ptr + g * C + excl + j, (child >= lo).to(tl.int8), mask=ok)
    base = tl.sum(cnt, axis=0)
    for j in tl.static_range(4):
        child = open_parent * 4 + j
        if (child < G) & (base + j < C):
            tl.store(cand_ptr + g * C + base + j, child)
            tl.store(forced_ptr + g * C + base + j, (child >= lo).to(tl.int8))


def children(cand_up, sel_up, G: int, open_parent: int, lo: int, C: int):
    """(cand [num_kv, C] int32, -1 empty; forced [num_kv, C] int8) of the level below."""
    num_kv, c_up = cand_up.shape
    cand = torch.full((num_kv, C), -1, dtype=torch.int32, device=cand_up.device)
    forced = torch.zeros((num_kv, C), dtype=torch.int8, device=cand_up.device)
    _children_kernel[(num_kv,)](cand_up, sel_up, cand, forced, G, open_parent, lo,
                                C_UP=c_up, C=C)
    return cand, forced


@triton.jit
def _level_part(o, g, val, cand_ptr, sel_ptr, gpos_ptr, C, off, COARSE: tl.constexpr):
    """Slots [off, off + C): gpos[cand] where shown (selected, or unselected with coarse)."""
    m = (o >= off) & (o < off + C)
    i = tl.where(m, o - off, 0)
    c = tl.load(cand_ptr + g * C + i, mask=m, other=-1)
    s = tl.load(sel_ptr + g * C + i, mask=m, other=0) != 0
    show = m & (c >= 0) & ((~s) if COARSE else s)
    gp = tl.load(gpos_ptr + tl.where(show, c, 0), mask=show, other=-1)
    return tl.where(show, gp, val)


@triton.jit(do_not_specialize=["cap", "n_leaf", "n_always", "a_off",
                               "C0", "C1", "C2", "C3", "o0", "o1", "o2", "o3"])
def _key_list_kernel(out_ptr, cap, leaf_pos_ptr, n_leaf, always_ptr, n_always, a_off,
                     c0, s0, p0, C0, o0, c1, s1, p1, C1, o1, c2, s2, p2, C2, o2,
                     c3, s3, p3, C3, o3, COARSE: tl.constexpr, BLOCK: tl.constexpr):
    g = tl.program_id(0)
    o = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    val = tl.full([BLOCK], -1, tl.int32)
    # leaf part [0, n_leaf): the 4 raw positions of each selected level-0 node
    m = o < n_leaf
    slot = tl.where(m, o // 4, 0)
    c = tl.load(c0 + g * C0 + slot, mask=m, other=-1)
    sel = m & (tl.load(s0 + g * C0 + slot, mask=m, other=0) != 0)
    lp = tl.load(leaf_pos_ptr + tl.where(sel, c, 0) * 4 + o % 4, mask=sel, other=-1)
    val = tl.where(sel, lp, val)
    val = _level_part(o, g, val, c0, s0, p0, C0, o0, COARSE)
    val = _level_part(o, g, val, c1, s1, p1, C1, o1, COARSE)
    val = _level_part(o, g, val, c2, s2, p2, C2, o2, COARSE)
    val = _level_part(o, g, val, c3, s3, p3, C3, o3, COARSE)
    m = (o >= a_off) & (o < a_off + n_always)
    val = tl.where(m, tl.load(always_ptr + tl.where(m, o - a_off, 0), mask=m, other=-1), val)
    tl.store(out_ptr + g * cap + o, val, mask=o < cap)


def key_list(levels, gpos, closed, leaf_pos, always, coarse: bool):
    """Kernel B's key list [num_kv, cap] int32 (-1 empty) in one launch: [the raw positions
    of the selected level-0 nodes (4 per slot) | per level with closed nodes, gpos of the
    shown candidates (selected; unselected with coarse) | always]."""
    assert 1 <= len(levels) <= 4, "the key list kernel takes up to 4 levels"
    num_kv = levels[0][0].shape[0]
    args, off = [], 0
    n_leaf = 4 * levels[0][0].shape[1] if closed[0] else 0
    off = n_leaf
    for l in range(4):
        if l < len(levels) and closed[l]:
            cand, sel = levels[l]
            args += [cand, sel, gpos[l], cand.shape[1], off]
            off += cand.shape[1]
        else:
            args += [levels[0][0], levels[0][1], gpos[0], 0, off]
    cap = off + always.numel()
    out = torch.empty(num_kv, cap, dtype=torch.int32, device=always.device)
    block = 1024
    _key_list_kernel[(num_kv, triton.cdiv(cap, block))](
        out, cap, leaf_pos, n_leaf, always, always.numel(), off, *args,
        COARSE=coarse, BLOCK=block)
    return out


def beam_level(q, K, beta, gpos, cand, forced, k, scaling, union=True):
    """One level for all KV groups. q [num_kv, HPG, D], K [num_kv, T, D] (contiguous),
    beta [num_kv, T] float32, gpos [G] int32, cand [num_kv, C] int32 (ascending node ids,
    -1 empty; C a power of 2, >= 16), forced [num_kv, C] int8. Returns picked [num_kv, C]
    bool: up to k free candidates per head ("union") or per group ("sumexp")."""
    num_kv, hpg, D = q.shape
    C = cand.shape[1]
    assert C >= 16 and C & (C - 1) == 0, C
    assert C <= 1 << 20, "slot field is 20 bits"
    hpg_p = max(16, triton.next_power_of_2(hpg))
    scores = torch.empty(num_kv, hpg_p, C, dtype=torch.float32, device=q.device)
    block_c = min(C, 64)
    _scores_kernel[(num_kv, C // block_c)](
        q.contiguous(), K, beta, gpos, cand, forced, scores, K.shape[1], scaling,
        HPG=hpg, HPG_P=hpg_p, D=D, C=C, BLOCK_C=block_c)
    out = torch.zeros(num_kv, C, dtype=torch.int32, device=q.device)
    if union:  # one program per head: the old single program per group took 0.24 ms
        _pick_head_kernel[(num_kv, hpg)](scores, out, k, HPG_P=hpg_p, C=C, num_warps=8)
    else:
        _pick_sumexp_kernel[(num_kv,)](scores, out, k, HPG=hpg, HPG_P=hpg_p, C=C, num_warps=8)
    return out.bool(), scores
