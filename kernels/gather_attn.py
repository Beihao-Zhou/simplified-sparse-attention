"""Gather attention with gradients (#29): each query row attends to its own list of keys.

For one sequence, query rows rows [n] (positions), the heads of KV group g attend to the
keys idx[g, i, :] (-1 empty) of row i, with beta[g, key] added to the logits. One program
per (row, KV group) streams the listed keys in blocks, all heads of the group at once,
with an online softmax; forward returns the output and the log-sum-exp. Backward
recomputes the probabilities, accumulates dq in the program and scatters dk, dv and
dbeta with atomic adds (so their last bits depend on the order the rows run in).
Gradients reach q, k, v and beta through both the output and the log-sum-exp (which
gist_select_train.merge uses).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd(q_ptr, k_ptr, v_ptr, b_ptr, rows_ptr, idx_ptr, o_ptr, lse_ptr,
         T, n, cap, scaling,
         HPG: tl.constexpr, HP: tl.constexpr, D: tl.constexpr, BC: tl.constexpr,
         HAS_BETA: tl.constexpr):
    i = tl.program_id(0)
    g = tl.program_id(1)
    hs = tl.arange(0, HP)
    ds = tl.arange(0, D)
    hm = hs < HPG
    p = tl.load(rows_ptr + i)
    q = tl.load(q_ptr + ((g * HPG + hs[:, None]) * T + p) * D + ds[None, :], mask=hm[:, None], other=0.0).to(tl.float32)
    m = tl.full([HP], float("-inf"), tl.float32)
    l = tl.zeros([HP], tl.float32)
    acc = tl.zeros([HP, D], tl.float32)
    for c0 in range(0, cap, BC):
        cs = c0 + tl.arange(0, BC)
        key = tl.load(idx_ptr + (g * n + i) * cap + cs, mask=cs < cap, other=-1)
        ok = key >= 0
        kk = tl.where(ok, key, 0).to(tl.int64)
        K = tl.load(k_ptr + (g * T + kk[:, None]) * D + ds[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
        s = tl.dot(q, tl.trans(K), input_precision="ieee") * scaling
        if HAS_BETA:
            s += tl.load(b_ptr + g * T + kk, mask=ok, other=0.0)[None, :]
        s = tl.where(ok[None, :], s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, 1))
        m_safe = tl.where(m_new > float("-inf"), m_new, 0.0)
        a = tl.exp(m - m_safe)
        pr = tl.exp(s - m_safe[:, None])
        V = tl.load(v_ptr + (g * T + kk[:, None]) * D + ds[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
        acc = acc * a[:, None] + tl.dot(pr, V, input_precision="ieee")
        l = l * a + tl.sum(pr, 1)
        m = m_new
    has = l > 0
    o = acc / tl.where(has, l, 1.0)[:, None]
    lse = tl.where(has, m + tl.log(tl.where(has, l, 1.0)), float("-inf"))
    tl.store(o_ptr + ((g * HPG + hs[:, None]) * n + i) * D + ds[None, :], o, mask=hm[:, None])
    tl.store(lse_ptr + (g * HPG + hs) * n + i, lse, mask=hm)


@triton.jit
def _bwd(q_ptr, k_ptr, v_ptr, b_ptr, rows_ptr, idx_ptr, lse_ptr, do_ptr, dlse_ptr, delta_ptr,
         dq_ptr, dk_ptr, dv_ptr, db_ptr,
         T, n, cap, scaling,
         HPG: tl.constexpr, HP: tl.constexpr, D: tl.constexpr, BC: tl.constexpr,
         HAS_BETA: tl.constexpr):
    i = tl.program_id(0)
    g = tl.program_id(1)
    hs = tl.arange(0, HP)
    ds = tl.arange(0, D)
    hm = hs < HPG
    p = tl.load(rows_ptr + i)
    q = tl.load(q_ptr + ((g * HPG + hs[:, None]) * T + p) * D + ds[None, :], mask=hm[:, None], other=0.0).to(tl.float32)
    hn = (g * HPG + hs) * n + i
    lse = tl.load(lse_ptr + hn, mask=hm, other=float("-inf"))
    has = lse > float("-inf")
    lse_s = tl.where(has, lse, 0.0)
    do = tl.load(do_ptr + hn[:, None] * D + ds[None, :], mask=hm[:, None], other=0.0)
    corr = tl.load(dlse_ptr + hn, mask=hm, other=0.0) - tl.load(delta_ptr + hn, mask=hm, other=0.0)
    dq = tl.zeros([HP, D], tl.float32)
    for c0 in range(0, cap, BC):
        cs = c0 + tl.arange(0, BC)
        key = tl.load(idx_ptr + (g * n + i) * cap + cs, mask=cs < cap, other=-1)
        ok = key >= 0
        kk = tl.where(ok, key, 0).to(tl.int64)
        K = tl.load(k_ptr + (g * T + kk[:, None]) * D + ds[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
        V = tl.load(v_ptr + (g * T + kk[:, None]) * D + ds[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
        s = tl.dot(q, tl.trans(K), input_precision="ieee") * scaling
        if HAS_BETA:
            s += tl.load(b_ptr + g * T + kk, mask=ok, other=0.0)[None, :]
        live = ok[None, :] & has[:, None] & hm[:, None]
        pr = tl.where(live, tl.exp(s - lse_s[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(V), input_precision="ieee")
        dsc = pr * (dp + corr[:, None])                        # d logit
        dq += tl.dot(dsc, K, input_precision="ieee")
        dk = tl.dot(tl.trans(dsc), q, input_precision="ieee") * scaling
        dv = tl.dot(tl.trans(pr), do, input_precision="ieee")
        off = (g * T + kk[:, None]) * D + ds[None, :]
        tl.atomic_add(dk_ptr + off, dk, mask=ok[:, None])
        tl.atomic_add(dv_ptr + off, dv, mask=ok[:, None])
        if HAS_BETA:
            tl.atomic_add(db_ptr + g * T + kk, tl.sum(dsc, 0), mask=ok)
    tl.store(dq_ptr + ((g * HPG + hs[:, None]) * T + p) * D + ds[None, :], dq * scaling, mask=hm[:, None])


class GatherAttention(torch.autograd.Function):
    """q [H, T, D], k / v [H_kv, T, D], beta [H_kv, T] float32 or None, rows [n] int64,
    idx [H_kv, n, cap] int32 (-1 empty) -> out [H, n, D] float32, lse [H, n] float32."""

    @staticmethod
    def forward(ctx, q, k, v, beta, rows, idx, scaling):
        H, T, D = q.shape
        Hkv = k.shape[0]
        hpg = H // Hkv
        n, cap = idx.shape[1], idx.shape[2]
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        idx = idx.contiguous()
        o = torch.empty(H, n, D, device=q.device, dtype=torch.float32)
        lse = torch.empty(H, n, device=q.device, dtype=torch.float32)
        b = beta.contiguous() if beta is not None else q.new_zeros(1, dtype=torch.float32)
        hp = max(16, triton.next_power_of_2(hpg))
        if n:
            _fwd[(n, Hkv)](q, k, v, b, rows, idx, o, lse, T, n, cap, scaling,
                           HPG=hpg, HP=hp, D=D, BC=64, HAS_BETA=beta is not None)
        # the contiguous beta the kernel read (SummaryBias returns a transposed view)
        ctx.save_for_backward(q, k, v, b if beta is not None else None, rows, idx, o, lse)
        ctx.scaling, ctx.has_beta = scaling, beta is not None
        return o, lse

    @staticmethod
    def backward(ctx, do, dlse):
        q, k, v, beta, rows, idx, o, lse = ctx.saved_tensors
        H, T, D = q.shape
        Hkv = k.shape[0]
        hpg = H // Hkv
        n, cap = idx.shape[1], idx.shape[2]
        do = torch.zeros_like(o) if do is None else do.float().contiguous()
        dlse = torch.zeros_like(lse) if dlse is None else dlse.float().contiguous()
        delta = (do * o).sum(-1).contiguous()
        dq = torch.zeros(H, T, D, device=q.device, dtype=torch.float32)
        dk = torch.zeros(Hkv, T, D, device=q.device, dtype=torch.float32)
        dv = torch.zeros_like(dk)
        db = torch.zeros(Hkv, T, device=q.device, dtype=torch.float32) if ctx.has_beta else dq.new_zeros(1)
        b = beta if ctx.has_beta else q.new_zeros(1, dtype=torch.float32)
        hp = max(16, triton.next_power_of_2(hpg))
        if n:
            _bwd[(n, Hkv)](q, k, v, b, rows, idx, lse, do, dlse, delta, dq, dk, dv, db,
                           T, n, cap, ctx.scaling, HPG=hpg, HP=hp, D=D, BC=64,
                           HAS_BETA=ctx.has_beta)
        return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype),
                db if ctx.has_beta else None, None, None, None)


def gather_attention(q, k, v, beta, rows, idx, scaling):
    """See GatherAttention."""
    return GatherAttention.apply(q, k, v, beta, rows, idx, scaling)
