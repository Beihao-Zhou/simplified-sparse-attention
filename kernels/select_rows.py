"""Beam selection for every query row at once (#29): the key lists X of
gist_select_train.selected_lists, in Triton, for stage 1 with selection at every position.

Per row p and level l (top first), the free candidates (candidates minus forced, ADR 0002
/ gist_beam.select) are
  top level  the closed nodes with id < lo_top(p)
  level l    the children of the nodes picked at level l + 1, then the ids in
             [4 lo_{l+1}(p), min(lo_l(p), closed_l(p))): a picked parent is below
             lo_{l+1}, so its 4 children are free; of the forced or open parents only
             lo_{l+1} has children below lo_l
with lo_l = lo // 4^l, in ascending id (ties go to the lower id, as in the reference).
One program per (row, KV group) scores them (q . k + beta, every head of the group), picks
k per head (union) or by summed exp-scores (sumexp) with kernels.beam_select._topk, keeps
the picks for the next level and writes X: the picked summaries (with coarse terms: the
unpicked candidates) and the 4 raw tokens of every picked level-0 node.
"""

import torch
import triton
import triton.language as tl

from kernels.beam_select import _topk


@triton.jit
def _cand(j, l: tl.constexpr, TOP: tl.constexpr, pick_ptr, prow, lo, cl, KMAX: tl.constexpr):
    """Candidate ids and validity of slots j at level l."""
    lo_l = lo >> (2 * l)
    if TOP:
        return j, j < tl.minimum(cl, lo_l)
    NP: tl.constexpr = 4 * KMAX
    par = tl.load(pick_ptr + prow + j // 4, mask=j < NP, other=-1)
    cid = par * 4 + j % 4
    s0 = 4 * (lo >> (2 * (l + 1)))
    sid = s0 + (j - NP)
    ids = tl.where(j < NP, cid, sid)
    ok = tl.where(j < NP, par >= 0, (j < NP + 4) & (sid < tl.minimum(lo_l, cl)))
    return ids, ok


@triton.jit
def _level(q, g, i, r, p, lo, w, k_ptr, b_ptr, closed_ptr, gpos_ptr, comp_ptr, out_ptr,
           pick_ptr, sc_ptr, T, n, CAP, R, kl, scaling,
           l: tl.constexpr, L: tl.constexpr, TOP: tl.constexpr, C: tl.constexpr,
           CS: tl.constexpr, KMAX: tl.constexpr, HPG: tl.constexpr, HP: tl.constexpr,
           D: tl.constexpr, BC: tl.constexpr, UNION: tl.constexpr, COARSE: tl.constexpr,
           HAS_BETA: tl.constexpr):
    cl = tl.load(closed_ptr + l * T + p)
    prow_up = ((g * R + i) * L + (l + 1)) * KMAX          # picks of level l + 1
    hs = tl.arange(0, HP)
    ds = tl.arange(0, D)
    srow = (g * R + i) * HP * CS
    for c0 in tl.static_range(0, C, BC):
        j = c0 + tl.arange(0, BC)
        ids, ok = _cand(j, l, TOP, pick_ptr, prow_up, lo, cl, KMAX)
        gp = tl.load(gpos_ptr + tl.where(ok, ids, 0), mask=ok, other=0)
        K = tl.load(k_ptr + (g * T + gp[:, None]).to(tl.int64) * D + ds[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
        s = tl.dot(q, tl.trans(K), input_precision="ieee") * scaling
        if HAS_BETA:
            s += tl.load(b_ptr + g * T + gp, mask=ok, other=0.0)[None, :]
        s = tl.where(ok[None, :], s, float("-inf"))
        tl.store(sc_ptr + srow + hs[:, None] * CS + j[None, :], s)
    tl.debug_barrier()
    j = tl.arange(0, C)
    ids, ok = _cand(j, l, TOP, pick_ptr, prow_up, lo, cl, KMAX)
    if UNION:
        picked = tl.zeros([C], dtype=tl.int1)
        for h in tl.static_range(HPG):
            s = tl.load(sc_ptr + srow + h * CS + j)
            picked = picked | _topk(s, ok & (s > float("-inf")), kl)
    else:
        mx = tl.full([], float("-inf"), tl.float32)
        for h in tl.static_range(HPG):
            mx = tl.maximum(mx, tl.max(tl.load(sc_ptr + srow + h * CS + j), axis=0))
        imp = tl.zeros([C], dtype=tl.float32)
        for h in tl.static_range(HPG):
            s = tl.load(sc_ptr + srow + h * CS + j)
            imp += tl.where(ok & (s > float("-inf")), tl.exp(s - mx), 0.0)
        picked = _topk(imp, ok, kl)
    rank = tl.cumsum(picked.to(tl.int32), axis=0) - 1
    prow = ((g * R + i) * L + l) * KMAX
    tl.store(pick_ptr + prow + rank, ids, mask=picked & (rank < KMAX))
    show = (ok & ~picked) if COARSE else picked
    rs = tl.cumsum(show.to(tl.int32), axis=0) - 1
    gp = tl.load(gpos_ptr + tl.where(show, ids, 0), mask=show, other=0)
    orow = (g * n + r) * CAP
    tl.store(out_ptr + orow + w + rs, gp, mask=show & (w + rs < CAP))
    w += tl.sum(show.to(tl.int32), axis=0)
    if l == 0:
        npk = tl.sum(picked.to(tl.int32), axis=0)
        for t in tl.static_range(4):
            raw = tl.load(comp_ptr + tl.where(picked, ids, 0) * 4 + t, mask=picked, other=0)
            tl.store(out_ptr + orow + w + rank * 4 + t, raw, mask=picked & (w + rank * 4 + t < CAP))
        w += 4 * npk
    tl.debug_barrier()
    return w


@triton.jit
def _select_kernel(q_ptr, k_ptr, b_ptr, rows_ptr, lo_ptr, closed_ptr, g0, g1, g2, g3, comp_ptr,
                   out_ptr, pick_ptr, sc_ptr, T, n, CAP, R, r0, k0, k1, k2, k3, scaling,
                   L: tl.constexpr, CT: tl.constexpr, CL: tl.constexpr, CS: tl.constexpr,
                   KMAX: tl.constexpr, HPG: tl.constexpr, HP: tl.constexpr, D: tl.constexpr,
                   BC: tl.constexpr, UNION: tl.constexpr, COARSE: tl.constexpr,
                   HAS_BETA: tl.constexpr):
    i = tl.program_id(0)
    g = tl.program_id(1)
    r = r0 + i
    if r < n:
        p = tl.load(rows_ptr + r)
        lo = tl.load(lo_ptr + p)
        hs = tl.arange(0, HP)
        ds = tl.arange(0, D)
        q = tl.load(q_ptr + ((g * HPG + hs[:, None]) * T + p) * D + ds[None, :], mask=(hs < HPG)[:, None], other=0.0).to(tl.float32)
        w = tl.zeros([], dtype=tl.int32)
        if L >= 4:
            w = _level(q, g, i, r, p, lo, w, k_ptr, b_ptr, closed_ptr, g3, comp_ptr, out_ptr, pick_ptr, sc_ptr,
                       T, n, CAP, R, k3, scaling, 3, L, L == 4, CT if L == 4 else CL, CS, KMAX, HPG, HP, D, BC,
                       UNION, COARSE, HAS_BETA)
        if L >= 3:
            w = _level(q, g, i, r, p, lo, w, k_ptr, b_ptr, closed_ptr, g2, comp_ptr, out_ptr, pick_ptr, sc_ptr,
                       T, n, CAP, R, k2, scaling, 2, L, L == 3, CT if L == 3 else CL, CS, KMAX, HPG, HP, D, BC,
                       UNION, COARSE, HAS_BETA)
        if L >= 2:
            w = _level(q, g, i, r, p, lo, w, k_ptr, b_ptr, closed_ptr, g1, comp_ptr, out_ptr, pick_ptr, sc_ptr,
                       T, n, CAP, R, k1, scaling, 1, L, L == 2, CT if L == 2 else CL, CS, KMAX, HPG, HP, D, BC,
                       UNION, COARSE, HAS_BETA)
        w = _level(q, g, i, r, p, lo, w, k_ptr, b_ptr, closed_ptr, g0, comp_ptr, out_ptr, pick_ptr, sc_ptr,
                   T, n, CAP, R, k0, scaling, 0, L, L == 1, CT if L == 1 else CL, CS, KMAX, HPG, HP, D, BC,
                   UNION, COARSE, HAS_BETA)


def select_lists(tree, lo, rows, q, k, beta, cfg, scaling, batch: int = 2048):
    """X of every row in rows [n] for all KV groups: [H_kv, n, cap] int32 (-1 empty).
    q [H, T, D], k [H_kv, T, D], beta [H_kv, T] float32 or None (no gradient), lo [T] the
    lowest forced leaf per position. Same sets as gist_select_train.selected_lists."""
    assert cfg.fanout == 4, "the kernel assumes 4-ary trees"
    L = tree.num_levels
    assert 1 <= L <= 4
    H, T, D = q.shape
    Hkv = k.shape[0]
    hpg = H // Hkv
    n = rows.numel()
    dev = q.device
    union = cfg.merge == "union"
    kmax = max(cfg.k_at(l) for l in range(L)) * (hpg if union else 1)
    pos = torch.arange(T, device=dev)
    closed = torch.stack([torch.searchsorted(g, pos) for g in tree.gpos]).int().contiguous()
    gpos = [g.int().contiguous() for g in tree.gpos]
    gpos += [gpos[0]] * (4 - L)
    comp = torch.nonzero(tree.compressible).flatten().int().contiguous()
    # at least one 64-slot score block (the loop's step)
    ct = max(64, triton.next_power_of_2(max(int(tree.gpos[L - 1].numel()), 1)))
    cl = max(64, triton.next_power_of_2(4 * kmax + 4))
    cs = max(ct, cl)
    assert ct <= 1 << 14, f"{tree.gpos[L - 1].numel()} top-level nodes: too many for one program"
    cap = (ct + (L - 1) * cl if cfg.coarse else L * kmax) + 4 * kmax
    hp = max(16, triton.next_power_of_2(hpg))
    out = torch.full((Hkv, n, cap), -1, dtype=torch.int32, device=dev)
    R = min(batch, max(n, 1))
    pick = torch.empty(Hkv, R, L, kmax, dtype=torch.int32, device=dev)
    sc = torch.empty(Hkv, R, hp, cs, dtype=torch.float32, device=dev)
    ks = [cfg.k_at(l) for l in range(L)] + [0] * (4 - L)
    b = beta.float().contiguous() if beta is not None else sc.new_zeros(1)
    qc, kc = q.contiguous(), k.contiguous()
    lo32 = lo.int().contiguous()
    for r0 in range(0, n, R):
        pick.fill_(-1)
        _select_kernel[(R, Hkv)](
            qc, kc, b, rows, lo32, closed, *gpos, comp, out, pick, sc, T, n, cap, R, r0,
            *ks, scaling, L=L, CT=ct, CL=cl, CS=cs, KMAX=kmax, HPG=hpg, HP=hp, D=D, BC=64,
            UNION=union, COARSE=cfg.coarse, HAS_BETA=beta is not None)
    return out
