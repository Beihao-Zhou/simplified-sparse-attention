"""Kernel A of DECODE_SELECT=beam (ADR 0001, #20): one level of the level-wise beam
selection for one decode query, all KV groups at once. Triton.

Per KV group g and candidate slot c (candidates are node ids in ascending order, -1 for an
empty slot), `beam_scores` computes s[h, c] = scaling * q[g, h] . K[g, gpos[cand]] + beta;
forced and empty slots get -inf, so only free candidates compete. `beam_pick` then keeps
k free candidates by rank-select, without sorting: each (score, slot) pair is packed into
an order-preserving 64-bit key (higher score first, equal scores to the lower slot, which
is the lower node id), and the k-th largest key is found bit by bit from the top, one
count per bit. "union" picks per head and ORs the heads; "sumexp" ranks the group's
summed exp-scores (RecSA). Scores go through a scratch buffer [num_kv, HPG_P, C].
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
def _kth(keys, k: tl.constexpr):
    """Largest t with count(keys >= t) >= k, keys non-negative int64 < 2^62 (bit by bit)."""
    t = tl.zeros([], dtype=tl.int64)
    for b in tl.static_range(61, -1, -1):
        cand = t | (1 << b)
        cnt = tl.sum((keys >= cand).to(tl.int32), axis=0)
        t = tl.where(cnt >= k, cand, t)
    return t


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
def _pick_kernel(s_ptr, out_ptr, k,
                 HPG: tl.constexpr, HPG_P: tl.constexpr, C: tl.constexpr, UNION: tl.constexpr):
    g = tl.program_id(0)
    cs = tl.arange(0, C)
    slot = (C - 1 - cs).to(tl.int64)                 # lower slot wins ties
    picked = tl.zeros([C], dtype=tl.int1)
    if UNION:
        for h in tl.static_range(HPG):
            s = tl.load(s_ptr + (g * HPG_P + h) * C + cs)
            ok = s > float("-inf")
            keys = tl.where(ok, (_ordered(s) << 20) | slot, 0)
            t = _kth(keys, k)
            picked = picked | (ok & (keys >= t))  # t = 0 when fewer than k: all of them
    else:
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
        keys = tl.where(ok, (_ordered(imp) << 20) | slot, 0)
        t = _kth(keys, k)
        picked = ok & (keys >= t)
    tl.store(out_ptr + g * C + cs, picked.to(tl.int8))


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
    out = torch.empty(num_kv, C, dtype=torch.int8, device=q.device)
    _pick_kernel[(num_kv,)](scores, out, k, HPG=hpg, HPG_P=hpg_p, C=C, UNION=union,
                            num_warps=8)
    return out.bool(), scores
