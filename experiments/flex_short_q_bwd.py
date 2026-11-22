"""torch 2.8 flex bug behind the replay's >= 256-row padding (gist_flex_attention.py, OPSD
replay): once compiled flex has seen another query length, the backward of a query of at
most 128 rows is wrong when its BlockMask was compiled (dK error ~2 against a dense
reference; the forward is right). Queries over 128 rows are right. Prints BAD per broken
case; drop the padding when the first two print ok.

    python experiments/flex_short_q_bwd.py
"""

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

H, HKV, D, T, R = 4, 2, 16, 421, 15


def dk_error(flex, q_len, seed):
    torch.manual_seed(seed)
    hpg = H // HKV
    q = torch.randn(1, H, q_len, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, HKV, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, HKV, T, D, device="cuda", dtype=torch.bfloat16)
    M = torch.rand(HKV, R, T, device="cuda") < 0.1
    M[..., 0] = True

    def mod(b, h, qi, kv):
        return M[h // hpg, qi.clamp(max=R - 1), kv.clamp(max=T - 1)] & (kv < T) & (qi < R)

    bm = create_block_mask(mod, 1, H, q_len, T, device="cuda", BLOCK_SIZE=128, _compile=True)
    out = flex(q, k, v, block_mask=bm, enable_gqa=True)[:, :, :R]
    g = torch.randn_like(out)
    dk, = torch.autograd.grad(out, k, g)
    rep = lambda x: x.repeat_interleave(hpg, 1)
    kr = k.detach().float().requires_grad_()
    s = q.float()[:, :, :R] @ rep(kr).transpose(-1, -2) / D**0.5
    ref = s.masked_fill(~rep(M[None]), float("-inf")).softmax(-1) @ rep(v.float())
    dk_ref, = torch.autograd.grad(ref, kr, g.float())
    return float((dk.float() - dk_ref).abs().max())


if __name__ == "__main__":
    for first, second in ((40, 15), (130, 128), (15, 256)):
        torch._dynamo.reset()
        flex = torch.compile(flex_attention, dynamic=False)
        dk_error(flex, first, 0)
        err = dk_error(flex, second, 1)
        print(f"{first:3d} rows then {second:3d}: dK error {err:.2e} {'BAD' if err > 0.1 else 'ok'}")
