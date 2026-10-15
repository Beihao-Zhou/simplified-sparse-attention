"""The FlexAttention gist mask must equal the dense greedy mask AND causal.

`create_mask` evaluates the mask_mod as flex_attention does, so this runs on CPU.
"""

import itertools
import random

import pytest
import torch
from torch.nn.attention.flex_attention import create_mask

from src.data.gist import add_raw_window, make_hierarchical_gist_mask_greedy
from src.data.gist_flex_mask import build_gist_mask_features, make_gist_mask_mod

PAD = 0
GIST_IDS = [1000, 1001, 1002, 1003]


def _structured(n_raw, levels, chunk_size, group, jitter, rng):
    """Gist layout like insert_hierarchical_gist_tokens: a level-0 gist roughly every
    chunk_size raw tokens (jitter mimics word-aware shifts), a level-(l+1) gist after
    every `group` level-l gists, and the trailing gists the insertion code appends."""
    out, count = [], [0] * levels
    since = 0
    target = chunk_size + rng.randint(0, jitter)
    for i in range(n_raw):
        out.append(rng.randint(1, 999))
        since += 1
        if since >= target and i + 1 < n_raw:
            since, target = 0, chunk_size + rng.randint(0, jitter)
            out.append(GIST_IDS[0])
            count[0] += 1
            for lvl in range(1, levels):
                if count[lvl - 1] % group == 0:
                    out.append(GIST_IDS[lvl])
                    count[lvl] += 1
                else:
                    break
    for lvl in range(levels):  # trailing gists
        out.append(GIST_IDS[lvl])
    return out


def _random_layout(n, levels, rng):
    """Unstructured placement, to exercise rules outside the usual pattern."""
    return [GIST_IDS[rng.randrange(levels)] if rng.random() < 0.15 else rng.randint(1, 999)
            for _ in range(n)]


def _batch(seqs):
    """Left-pad, as the training collators do."""
    T = max(len(s) for s in seqs)
    return torch.tensor([[PAD] * (T - len(s)) + s for s in seqs])


def _check(inputs, levels, chunk_size, sink, prev, window=0):
    gist = GIST_IDS[:levels]
    B, T = inputs.shape
    dense = make_hierarchical_gist_mask_greedy(
        inputs, gist, chunk_size, attention_sink_size=sink,
        num_previous_chunks=prev, pad_token=PAD,
    )
    dense = add_raw_window(dense, inputs, gist, window, pad_token=PAD)[:, 0]
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    expected = dense & causal

    f = build_gist_mask_features(inputs, gist, chunk_size, sink, prev, pad_token=PAD, raw_window=window)
    got = create_mask(make_gist_mask_mod(f), B, None, T, T, device="cpu")[:, 0]

    diff = (got != expected).nonzero()
    assert diff.numel() == 0, (
        f"{diff.shape[0]} mismatches; first (b, q, k) = {diff[:5].tolist()}, "
        f"levels={levels} chunk={chunk_size} sink={sink} prev={prev} window={window}")


@pytest.mark.parametrize("levels,sink,prev", list(itertools.product([1, 2, 4], [0, 3, -1], [0, 2])))
def test_structured_layouts(levels, sink, prev):
    rng = random.Random(1000 * levels + 10 * (sink + 1) + prev)
    for chunk_size, group in [(4, 4), (8, 2)]:  # chunk_size only matters for the dynamic sink
        seqs = [_structured(rng.randint(20, 160), levels, chunk_size, group, jitter=2, rng=rng)
                for _ in range(3)]
        _check(_batch(seqs), levels, chunk_size, sink, prev)


@pytest.mark.parametrize("levels,sink,prev", list(itertools.product([1, 2, 3], [0, 3, -1], [0, 1])))
def test_random_layouts(levels, sink, prev):
    rng = random.Random(7 + 100 * levels + 10 * (sink + 1) + prev)
    seqs = [_random_layout(rng.randint(10, 120), levels, rng) for _ in range(4)]
    _check(_batch(seqs), levels, 4, sink, prev)


@pytest.mark.parametrize("levels,window", list(itertools.product([1, 3], [1, 32, 500])))
def test_raw_window(levels, window):
    rng = random.Random(levels * 31 + window)
    seqs = [_structured(rng.randint(40, 200), levels, 4, 4, jitter=1, rng=rng) for _ in range(3)]
    _check(_batch(seqs), levels, 4, 3, 1, window)
    _check(_batch([_random_layout(120, levels, rng) for _ in range(3)]), levels, 4, 3, 1, window)


def test_raw_window_with_pad_inside_sequence():
    """pad == eos used as a separator: pad queries and keys get no window (as dense)."""
    rng = random.Random(9)
    seqs = []
    for _ in range(3):
        a, b = (_structured(rng.randint(30, 80), 2, 4, 4, jitter=1, rng=rng) for _ in range(2))
        seqs.append(a + [PAD] + b)
    _check(_batch(seqs), 2, 4, 3, 1, window=40)


def test_raw_window_keeps_recent_raw_exact_and_gists_unchanged():
    """Every raw query sees all raw tokens in its window; summary queries are unaffected."""
    rng = random.Random(5)
    inputs = torch.tensor([_structured(300, 2, 4, 4, jitter=0, rng=rng)])
    T, W = inputs.shape[1], 64
    base = build_gist_mask_features(inputs, GIST_IDS[:2], 4, 0, 1)
    win = build_gist_mask_features(inputs, GIST_IDS[:2], 4, 0, 1, raw_window=W)
    m0 = create_mask(make_gist_mask_mod(base), 1, None, T, T, device="cpu")[0, 0]
    m1 = create_mask(make_gist_mask_mod(win), 1, None, T, T, device="cpu")[0, 0]
    is_raw = base.level[0] < 0
    pos = torch.arange(T)
    near = (pos[:, None] - pos[None, :] >= 0) & (pos[:, None] - pos[None, :] < W)
    assert m1[near & is_raw[:, None] & is_raw[None, :]].all()
    assert torch.equal(m1[~is_raw], m0[~is_raw])
    assert torch.equal(m1 & ~near, m0 & ~near), "nothing changes outside the window"


def test_example_without_gists_is_causal():
    rng = random.Random(0)
    seqs = [[rng.randint(1, 999) for _ in range(40)],
            _structured(60, 2, 4, 4, jitter=0, rng=rng)]
    _check(_batch(seqs), 2, 4, 3, 1)


@pytest.mark.parametrize("levels,prev,window", [(1, 1, 0), (2, 1, 0), (3, 0, 0), (4, 2, 0), (2, 1, 40)])
def test_gist_pool_layout_matches_dense_sdpa(levels, prev, window):
    """Reordering keys into [raw tokens | summary pool] must not change the result."""
    from torch.nn.attention.flex_attention import flex_attention

    from src.data.gist_flex_mask import gist_pool_permutation, make_gist_block_mask

    torch.manual_seed(levels)
    rng = random.Random(levels)
    chunk_size, sink = 4, 3
    seqs = [_structured(160, levels, chunk_size, 2, jitter=2, rng=rng) for _ in range(2)]
    T = min(len(s) for s in seqs)
    inputs = torch.tensor([s[:T] for s in seqs])
    gist = GIST_IDS[:levels]

    dense = make_hierarchical_gist_mask_greedy(
        inputs, gist, chunk_size, attention_sink_size=sink, num_previous_chunks=prev)
    dense = add_raw_window(dense, inputs, gist, window) & torch.ones(T, T, dtype=torch.bool).tril()

    B, H, d = inputs.shape[0], 2, 8
    q, k, v = (torch.randn(B, H, T, d, requires_grad=True) for _ in range(3))
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=dense)

    f = build_gist_mask_features(inputs, gist, chunk_size, sink, prev, raw_window=window)
    perm = gist_pool_permutation(f)
    idx = perm[:, None, :, None].expand(B, H, T, d)
    bm = make_gist_block_mask(f, T, block_size=16, device="cpu", kv_perm=perm)
    out = flex_attention(q, k.gather(2, idx), v.gather(2, idx), block_mask=bm)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    g = torch.randn_like(ref)
    for a, b in zip(torch.autograd.grad(out, (q, k, v), g), torch.autograd.grad(ref, (q, k, v), g)):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
