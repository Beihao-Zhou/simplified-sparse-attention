"""The gather attention kernel (kernels/gather_attn.py, #29) against the PyTorch reference
(gist_select_train.gather_attention): outputs, log-sum-exp and every gradient. CUDA only."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("H,Hkv,D", [(14, 2, 64), (28, 4, 128), (4, 4, 64), (4, 2, 16)])
@pytest.mark.parametrize("with_beta", [True, False])
@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.bfloat16, 2e-2)])
@pytest.mark.parametrize("strided", [False, True])
def test_kernel_matches_the_reference(H, Hkv, D, with_beta, dtype, tol, strided):
    """fp32 pins the formulas (2e-6 seen); bf16 is the training dtype (~1%: rounding).
    strided: q / k / v as attention passes them, [T, heads, D] memory seen as [heads, T, D]."""
    from gist_select_train import gather_attention as ref
    from kernels.gather_attn import gather_attention

    torch.manual_seed(H + D)
    T, n, cap = 700, 90, 150
    hpg = H // Hkv
    def leaf(h):
        if strided:
            return torch.randn(T, h, D, device="cuda").to(dtype).requires_grad_()
        return torch.randn(h, T, D, device="cuda").to(dtype).requires_grad_()

    q_, k_, v_ = leaf(H), leaf(Hkv), leaf(Hkv)
    q, k, v = (t.transpose(0, 1) if strided else t for t in (q_, k_, v_))
    beta_ = None
    if with_beta:  # strided: [T, H_kv] seen as [H_kv, T], as SummaryBias returns it
        beta_ = (torch.randn(*((T, Hkv) if strided else (Hkv, T)), device="cuda") * 2).requires_grad_()
    beta = None if beta_ is None else (beta_.t() if strided else beta_)
    rows = torch.randperm(T, device="cuda")[:n].sort().values
    idx = torch.stack([torch.stack([torch.randperm(T, device="cuda")[:cap] for _ in range(n)])
                       for _ in range(Hkv)]).int()
    idx[:, :, ::5] = -1          # holes
    idx[:, 3] = -1               # a row with no key: lse -inf, output 0, no gradient
    do = torch.randn(H, n, D, device="cuda")
    dlse = torch.randn(H, n, device="cuda")
    dlse[:, 3] = 0

    o, lse = gather_attention(q, k, v, beta, rows, idx, D ** -0.5)
    assert torch.isinf(lse[:, 3]).all() and (o[:, 3] == 0).all()
    live = torch.isfinite(lse)
    (o * do).sum().backward(retain_graph=True)
    (torch.where(live, lse, 0) * dlse).sum().backward()
    got = [t.grad.clone() for t in (q_, k_, v_) + ((beta_,) if with_beta else ())]
    for t in (q_, k_, v_) + ((beta_,) if with_beta else ()):
        t.grad = None

    o_r, lse_r = [], []
    for g in range(Hkv):
        hs = slice(g * hpg, (g + 1) * hpg)
        og, lg = ref(q[hs][:, rows], k[g], v[g], None if beta is None else beta[g], idx[g].long(),
                     D ** -0.5)
        o_r.append(og), lse_r.append(lg)
    o_r, lse_r = torch.cat(o_r), torch.cat(lse_r)
    torch.testing.assert_close(o, o_r, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(lse[live], lse_r[live], rtol=1e-5, atol=1e-5)
    ((o_r * do).sum() + (torch.where(live, lse_r, 0) * dlse).sum()).backward()
    want = [t.grad for t in (q_, k_, v_) + ((beta_,) if with_beta else ())]
    for name, a, b in zip("q k v beta".split(), got, want):
        err = float((a.float() - b.float()).abs().max()); scale = float(b.float().abs().max())
        assert err <= tol * max(scale, 1.0), (name, err, scale)
