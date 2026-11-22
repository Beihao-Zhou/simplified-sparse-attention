"""Kernels A and B of DECODE_SELECT=beam against torch (#20). CUDA only."""

import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _attn_ref(q, K, V, idx, bias, scaling):
    H, num_kv = q.shape[1], K.shape[1]
    hpg = H // num_kv
    out = torch.empty(H, q.shape[-1], device=q.device)
    for g in range(num_kv):
        keep = idx[g][idx[g] >= 0].long()
        k, v = K[0, g, keep].float(), V[0, g, keep].float()
        s = q[0, g * hpg:(g + 1) * hpg, 0].float() @ k.T * scaling + bias[g, keep][None]
        out[g * hpg:(g + 1) * hpg] = s.softmax(-1) @ v
    return out


@pytest.mark.parametrize("D,H,num_kv", [(64, 14, 2), (128, 28, 4), (64, 4, 2)])
def test_kernel_b_matches_torch(D, H, num_kv):
    sys.path.insert(0, ".")
    import gist_beam_cuda

    torch.manual_seed(0)
    T, cap = 3000, 700
    q = torch.randn(1, H, 1, D, device="cuda").bfloat16()
    K = torch.randn(1, num_kv, T, D, device="cuda").bfloat16()
    V = torch.randn(1, num_kv, T, D, device="cuda").bfloat16()
    idx = torch.stack([torch.randperm(T, device="cuda")[:cap] for _ in range(num_kv)]).int()
    idx[:, ::7] = -1  # empty slots
    bias = torch.randn(num_kv, T, device="cuda") * 2
    out = gist_beam_cuda.ext().beam_decode_attn(q, K, V, idx.contiguous(), bias, D ** -0.5, 8)
    torch.testing.assert_close(out[0, 0].float(), _attn_ref(q, K, V, idx, bias, D ** -0.5),
                               rtol=2e-2, atol=2e-2)


def _scores_ref(q, K, beta, gpos, cand, forced, scaling):
    num_kv, hpg, D = q.shape
    pos = gpos[cand.clamp(min=0).long()]
    s = torch.einsum("ghd,gcd->ghc", q.float(), torch.stack(
        [K[g, pos[g].long()].float() for g in range(num_kv)])) * scaling
    s = s + torch.stack([beta[g, pos[g].long()] for g in range(num_kv)])[:, None]
    return s.masked_fill(((cand < 0) | forced.bool())[:, None], float("-inf"))


@pytest.mark.parametrize("union", [True, False])
@pytest.mark.parametrize("C,k", [(64, 5), (1024, 37), (4096, 256)])
def test_kernel_a_level_matches_topk(union, C, k):
    from kernels.beam_select import beam_level

    torch.manual_seed(1)
    num_kv, hpg, D, T = 2, 7, 64, 20000
    q = torch.randn(num_kv, hpg, D, device="cuda").bfloat16()
    K = torch.randn(num_kv, T, D, device="cuda").bfloat16()
    beta = torch.randn(num_kv, T, device="cuda")
    gpos = torch.sort(torch.randperm(T, device="cuda")[:C + 50]).values.int()
    n_valid = C - 9
    cand = torch.full((num_kv, C), -1, dtype=torch.int32, device="cuda")
    for g in range(num_kv):
        cand[g, :n_valid] = torch.sort(torch.randperm(C + 50, device="cuda")[:n_valid]).values.int()
    forced = (torch.rand(num_kv, C, device="cuda") < 0.05).to(torch.int8)
    picked, _ = beam_level(q, K.contiguous(), beta, gpos, cand, forced, k, 0.125, union=union)
    s = _scores_ref(q, K, beta, gpos, cand, forced, 0.125)          # [g, h, C]
    for g in range(num_kv):
        rows = s[g] if union else torch.exp(s[g] - s[g].max()).sum(0, keepdim=True)
        ref = torch.zeros(C, dtype=torch.bool, device="cuda")
        amb = torch.zeros(C, dtype=torch.bool, device="cuda")
        for r in rows:
            top = torch.sort(-r, stable=True).indices[:k]
            ok = torch.isfinite(r) & (r > 0 if not union else torch.ones_like(r, dtype=torch.bool))
            ref[top[ok[top]]] = True
            kth = r[top[-1]]
            amb |= (r - kth).abs() <= 1e-4 * (1 + kth.abs())        # near-ties may flip
        assert torch.equal(picked[g][~amb], ref[~amb]), (g, (picked[g] ^ ref).sum())


def test_kernel_a_exact_ties_and_fewer_than_k():
    from kernels.beam_select import beam_level

    num_kv, hpg, D, T, C = 1, 7, 64, 500, 64
    q = torch.randn(num_kv, hpg, D, device="cuda").bfloat16()
    K = torch.zeros(num_kv, T, D, device="cuda").bfloat16()       # all scores equal
    beta = torch.zeros(num_kv, T, device="cuda")
    gpos = torch.arange(0, 2 * C, 2, device="cuda").int()
    cand = torch.arange(C, device="cuda").int()[None].clone()
    cand[0, 40:] = -1
    forced = torch.zeros(num_kv, C, dtype=torch.int8, device="cuda")
    for union in (True, False):
        picked, _ = beam_level(q, K, beta, gpos, cand, forced, 3, 0.1, union=union)
        assert torch.nonzero(picked[0]).flatten().tolist() == [0, 1, 2]
        picked, _ = beam_level(q, K, beta, gpos, cand, forced, 100, 0.1, union=union)
        assert int(picked[0].sum()) == 40 and not picked[0, 40:].any()


def _tree_inputs(levels, seed=0, n_turns=40):
    import random

    import gist_beam as gb
    from src.data.gist_stream import gist_segments

    rng = random.Random(seed)
    gids = [90001 + i for i in range(levels)]
    ids, pinned, turn = [], [], []
    for _ in range(9):
        ids.append(rng.randint(1, 999)), pinned.append(True), turn.append(0)
    for t in range(1, n_turns + 1):
        for _ in range(rng.randint(10, 120)):
            ids.append(rng.randint(1, 999)), pinned.append(False), turn.append(t)
    g, p, t = gist_segments(ids, pinned, turn, gids, 4, 4, sink=3)
    if g[-1] in gids:  # the decode query is a raw token
        g, p, t = g + [5], p + [False], t + [t[-1]]
    dev = "cuda"
    tree = gb.build_tree(torch.tensor(g, device=dev), gids, 3, torch.tensor(p, device=dev),
                         torch.tensor(t, device=dev))
    return tree


@pytest.mark.parametrize("merge,coarse", [("union", False), ("union", True), ("sumexp", False), ("sumexp", True)])
@pytest.mark.parametrize("levels", [3, 4])
def test_fast_path_matches_the_reference(merge, coarse, levels):
    import gist_beam as gb
    import gist_beam_cuda as fast

    tree = _tree_inputs(levels, seed=levels)
    T = tree.level.numel()
    torch.manual_seed(levels)
    num_kv, hpg, D = 2, 7, 64
    q = torch.randn(1, num_kv * hpg, 1, D, device="cuda").bfloat16()
    K = torch.randn(1, num_kv, T, D, device="cuda").bfloat16()
    V = torch.randn(1, num_kv, T, D, device="cuda").bfloat16()
    beta = (torch.randn(num_kv, T, device="cuda") * (tree.level >= 0)).contiguous()
    cfg = gb.BeamConfig(k=(12, 6), merge=merge, coarse=coarse)
    sc = D ** -0.5
    meta = fast.StepMeta(tree, prev_chunks=1, raw_window=40, window_turns=2)
    levels_f = fast.select_fast(meta, q[0, :, 0].view(num_kv, hpg, D), K[0].contiguous(), beta, cfg, sc)
    idx = fast.key_list(meta, levels_f, coarse)
    out = fast.ext().beam_decode_attn(q, K, V, idx, beta, sc, 8)
    rows = torch.tensor([T - 1], device="cuda")
    for g in range(num_kv):
        qg = q[0, g * hpg:(g + 1) * hpg]
        s = gb.select(tree, qg, K[0, g], beta[g], rows, cfg, sc, prev_chunks=1, raw_window=40, window_turns=2)
        for l, (cand, sel) in enumerate(levels_f):
            got = sorted(cand[g][sel[g] & (cand[g] >= 0)].tolist())
            assert got == torch.nonzero(s.sel[l][0]).flatten().tolist(), (g, l)
        vis = gb.visible(tree, s, coarse)[0]
        listed = idx[g][idx[g] >= 0].long()
        assert listed.numel() == listed.unique().numel(), "no key listed twice"
        assert torch.equal(torch.zeros(T, dtype=torch.bool, device="cuda").index_fill_(0, listed, True), vis)
        ref = gb.attend(qg.float(), K[0, g].float(), V[0, g].float(), vis[None], sc, beta[g])[:, 0]
        torch.testing.assert_close(out[0, 0, g * hpg:(g + 1) * hpg].float(), ref, rtol=2e-2, atol=2e-2)


def _tiny_session_logits(monkeypatch, impls, steps=(5, 6, 7, 8)):
    import random

    from transformers import Qwen2Config

    import gist_flex_attention
    from gist_lora import apply_lora, apply_summary_bias, summary_spans
    from gist_qwen2 import GistQwen2ForCausalLM
    from gist_session import GistSession
    from gist_utils import Global_data

    keys = ("gist_token_id", "chunk_size", "sink_size", "num_previous_chunks", "pad_token_id",
            "train_attn", "raw_window", "window_turns", "top_k", "last_pos", "stage1",
            "decode_gists", "beam_segments")
    saved = {k: getattr(Global_data, k) for k in keys}
    try:
        L = 3
        Global_data.gist_token_id, Global_data.chunk_size = [1000, 1001, 1002], [4] * L
        Global_data.sink_size, Global_data.num_previous_chunks = 3, 1
        Global_data.pad_token_id, Global_data.stage1, Global_data.top_k = 0, None, [3] * 3
        Global_data.raw_window, Global_data.window_turns = 32, 2
        monkeypatch.setenv("DECODE_SELECT", "beam")
        monkeypatch.setenv("BEAM_K", "4,2")
        cfg = Qwen2Config(vocab_size=1100, hidden_size=448, intermediate_size=256, num_hidden_layers=3,
                          num_attention_heads=7, num_key_value_heads=1, max_position_embeddings=8192)
        cfg._attn_implementation = gist_flex_attention.GIST_FLEX
        torch.manual_seed(0)
        m = GistQwen2ForCausalLM(cfg).bfloat16()          # adapters stay fp32, as in training
        apply_lora(m, "gist", r=4)
        apply_summary_bias(m, spans=summary_spans(L))
        for n, p in m.named_parameters():
            if "lora_B" in n or "summary_bias" in n:
                p.data.normal_(0, 0.3)
        m = m.cuda().eval()
        m.selective = True
        rng = random.Random(0)
        segs = [([rng.randint(1, 999) for _ in range(9)], True, 0)] + [
            ([rng.randint(1, 999) for _ in range(rng.randint(20, 90))], False, t) for t in range(1, 12)]
        logits = {}
        for impl in impls:
            monkeypatch.setenv("ATTN_IMPL", impl)
            s = GistSession(m)
            for raw, pin, turn in segs:
                s.append(raw, pin, turn)
            logits[impl] = torch.stack([s.step(t) for t in steps]).float()
        return logits
    finally:
        for k, v in saved.items():
            setattr(Global_data, k, v)


def test_model_decode_fast_equals_reference(monkeypatch):
    logits = _tiny_session_logits(monkeypatch, ("ref", "sparse"))
    assert torch.isfinite(logits["sparse"]).all()
    torch.testing.assert_close(logits["sparse"], logits["ref"], rtol=5e-2, atol=5e-2)


def test_session_decode_without_model_forward_is_bit_identical(monkeypatch):
    """GistSession._decode_fast against model.forward, over enough steps to feed summaries."""
    runs = {}
    for fast in ("0", "1"):
        monkeypatch.setenv("GIST_FAST_DECODE", fast)
        runs[fast] = _tiny_session_logits(monkeypatch, ("sparse",), steps=range(5, 30))["sparse"]
    assert torch.equal(runs["0"], runs["1"])


def test_session_fast_path_never_rebuilds_the_tree(monkeypatch):
    import gist_beam

    calls = []
    orig = gist_beam._decode_tree
    monkeypatch.setattr(gist_beam, "_decode_tree", lambda ids: calls.append(1) or orig(ids))
    _tiny_session_logits(monkeypatch, ("sparse",))
    assert not calls, f"{len(calls)} rebuilds"


@pytest.mark.parametrize("levels", [2, 4])
@pytest.mark.parametrize("window", [(0, 0), (40, 0), (0, 2), (64, 3)])
def test_incremental_tree_matches_rebuilt_meta(levels, window):
    import random

    import gist_beam as gb
    import gist_beam_cuda as fast
    from src.data.gist_stream import gist_segments

    rng = random.Random(levels * 10 + window[1])
    gids = [90001 + i for i in range(levels)]
    ids, pinned, turn = [], [], []
    for _ in range(7):
        ids.append(rng.randint(1, 999)), pinned.append(True), turn.append(0)
    for t in range(1, 30):
        for _ in range(rng.randint(3, 50)):
            ids.append(rng.randint(1, 999)), pinned.append(False), turn.append(t)
    g, p, t = gist_segments(ids, pinned, turn, gids, 4, 4, sink=3)
    inc = fast.IncrementalTree(gids, 3, "cuda")
    checked = 0
    for n in range(len(g)):
        inc.extend([g[n]], [p[n]], [t[n]])
        if g[n] in gids or n % 7:  # decode queries are raw tokens; check a sample of prefixes
            continue
        tree = gb.build_tree(torch.tensor(g[: n + 1], device="cuda"), gids, 3,
                             torch.tensor(p[: n + 1], device="cuda"), torch.tensor(t[: n + 1], device="cuda"))
        ref = fast.StepMeta(tree, 1, *window)
        got = inc.meta(1, *window)
        assert got.closed == ref.closed and got.lo == ref.lo, (n, got.lo, ref.lo)
        assert all(torch.equal(a, b) for a, b in zip(got.gpos, ref.gpos))
        assert torch.equal(got.leaf_pos, ref.leaf_pos)
        assert torch.equal(torch.sort(got.always).values, torch.sort(ref.always).values)
        checked += 1
    assert checked > 50



def _children_ref(cand_up, sel_up, G, open_parent, lo, smax, C):
    """The torch candidate generation the children kernel replaced (sort + pad)."""
    BIG = 1 << 30
    num_kv = cand_up.shape[0]
    par = torch.where(sel_up, cand_up, BIG).sort(dim=1).values[:, :smax]
    par = torch.cat([par, torch.full((num_kv, 1), open_parent, device=par.device, dtype=par.dtype)], 1)
    child = (par[:, :, None] * 4 + torch.arange(4, device=par.device)).flatten(1)
    child = torch.where((par[:, :, None] < BIG).expand(-1, -1, 4).flatten(1) & (child < G), child, BIG)
    child = torch.nn.functional.pad(child.sort(dim=1).values, (0, C - child.shape[1]), value=BIG)
    cand = torch.where(child < BIG, child, -1).int()
    return cand, ((cand >= lo) & (cand >= 0)).to(torch.int8)


@pytest.mark.parametrize("n_up,n_sel,G_extra", [(40, 5, 3), (300, 300, 0), (1000, 1, 1), (17, 0, 2)])
def test_children_kernel_matches_sort(n_up, n_sel, G_extra):
    """Bit-identical to the sort-based generation: selected parents' children, then the
    open parent's, those < G, ascending; forced from lo."""
    sys.path.insert(0, ".")
    from gist_beam_cuda import _pow2
    from kernels.beam_select import children

    torch.manual_seed(n_up)
    num_kv = 3
    C_up = _pow2(n_up)
    ids = torch.arange(C_up, device="cuda")
    cand_up = torch.where(ids < n_up, ids, -1).int()[None].expand(num_kv, C_up).contiguous()
    sel_up = torch.zeros(num_kv, C_up, dtype=torch.bool, device="cuda")
    for g in range(num_kv):
        sel_up[g, torch.randperm(n_up, device="cuda")[:n_sel]] = True
    G = 4 * n_up + G_extra            # the open parent n_up has G_extra closed children
    lo = 4 * max(n_up - 2, 0) + 1     # forced: the last leaves
    smax = min(C_up, n_sel + 2)
    C = _pow2(4 * (smax + 1))
    got = children(cand_up, sel_up, G, n_up, lo, C)
    want = _children_ref(cand_up, sel_up, G, n_up, lo, smax, C)
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])
