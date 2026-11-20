"""Stage-1 teacher masks (flex mask_mods) match their dense definitions."""

import random

import pytest
import torch
from torch.nn.attention.flex_attention import create_mask

from src.data.gist import make_hierarchical_gist_mask_greedy
from src.data.gist_flex_mask import (
    build_gist_mask_features,
    gist_pool_permutation,
    make_teacher_mask_mod,
)
from src.data.gist_stream import insert_gists_with_map

G = [1000, 1001]


def _batch(n_raw, sink, seed):
    rng = random.Random(seed)
    seqs = [
        insert_gists_with_map([rng.randrange(1, 900) for _ in range(n_raw)], G, 4, 4, sink=sink)[0]
        for _ in range(2)
    ]
    T = max(map(len, seqs))
    return torch.tensor([s + [0] * (T - len(s)) for s in seqs])


def _flex_mask(ids, kind, sink, prev, permute):
    f = build_gist_mask_features(
        ids, G, 4, attention_sink_size=sink, num_previous_chunks=prev, pad_token=0
    )
    perm = gist_pool_permutation(f) if permute else None
    T = ids.shape[1]
    m = create_mask(
        make_teacher_mask_mod(f, kind, kv_perm=perm), ids.shape[0], None, T, T, device="cpu"
    )[:, 0]
    if perm is not None:  # back to original key order
        m = torch.zeros_like(m).scatter_(2, perm[:, None, :].expand_as(m), m)
    return m


@pytest.mark.parametrize("permute", [False, True])
def test_raw_teacher_is_causal_over_raw_tokens(permute):
    ids = _batch(120, 3, 0)
    T = ids.shape[1]
    raw_k = ~torch.isin(ids, torch.tensor(G)) & (ids != 0)
    ref = torch.ones(T, T, dtype=torch.bool).tril()[None] & (
        raw_k[:, None, :] | torch.eye(T, dtype=torch.bool)[None]
    )
    ref &= (ids != 0)[:, None, :]
    assert torch.equal(_flex_mask(ids, "raw", 3, 1, permute), ref)


@pytest.mark.parametrize(
    "sink,prev,permute", [(0, 0, False), (3, 1, False), (3, 1, True), (2, 0, True)]
)
def test_g1_teacher_is_the_single_level_mask_without_meta_gists(sink, prev, permute):
    """On non-meta rows and keys, g1 equals the 1-level greedy mask of the sequence with
    meta-gists deleted; meta-gists are never keys (except to themselves)."""
    ids = _batch(160, sink, 1)
    m = _flex_mask(ids, "g1", sink, prev, permute)
    for b in range(ids.shape[0]):
        keep = (ids[b] != G[1]) & (ids[b] != 0)
        sub = ids[b][keep][None]
        ref = make_hierarchical_gist_mask_greedy(
            sub, G[:1], 4, attention_sink_size=sink, num_previous_chunks=prev, pad_token=0
        )[0, 0].bool()
        ref &= torch.ones_like(ref).tril()
        idx = torch.nonzero(keep).flatten()
        assert torch.equal(m[b][idx][:, idx], ref), b
        meta = torch.nonzero(ids[b] == G[1]).flatten()
        off_diag = m[b][:, meta] & ~torch.eye(ids.shape[1], dtype=torch.bool)[:, meta]
        assert not off_diag.any()


@pytest.mark.parametrize("L,m,permute", [(3, 2, False), (3, 2, True), (4, 2, True), (4, 3, False)])
def test_gm_teacher_is_the_m_level_mask(L, m, permute):
    """g<m> on an L-level sequence equals the m-level greedy mask of the sequence with
    levels >= m deleted; deleted summaries are keys only to themselves."""
    Gs = [1000 + i for i in range(L)]
    rng = random.Random(L * 10 + m)
    seqs = [
        insert_gists_with_map([rng.randrange(1, 900) for _ in range(700)], Gs, 4, 4, sink=3)[0]
        for _ in range(2)
    ]
    T = max(map(len, seqs))
    ids = torch.tensor([s + [0] * (T - len(s)) for s in seqs])
    f = build_gist_mask_features(
        ids, Gs, 4, attention_sink_size=3, num_previous_chunks=1, pad_token=0
    )
    perm = gist_pool_permutation(f) if permute else None
    mm = create_mask(make_teacher_mask_mod(f, f"g{m}", kv_perm=perm), 2, None, T, T, device="cpu")[
        :, 0
    ]
    if perm is not None:
        mm = torch.zeros_like(mm).scatter_(2, perm[:, None, :].expand_as(mm), mm)
    high = torch.isin(ids, torch.tensor(Gs[m:]))
    for b in range(2):
        keep = ~high[b] & (ids[b] != 0)
        ref = make_hierarchical_gist_mask_greedy(
            ids[b][keep][None], Gs[:m], 4, attention_sink_size=3, num_previous_chunks=1, pad_token=0
        )[0, 0].bool()
        ref &= torch.ones_like(ref).tril()
        idx = torch.nonzero(keep).flatten()
        assert torch.equal(mm[b][idx][:, idx], ref), b
        hi = torch.nonzero(high[b]).flatten()
        assert torch.equal(mm[b][:, hi], torch.eye(T, dtype=torch.bool)[:, hi]), b
        assert torch.equal(mm[b][hi], torch.eye(T, dtype=torch.bool)[hi]), b


def test_hier_mask_matches_greedy_at_four_levels():
    Gs = [1000, 1001, 1002, 1003]
    rng = random.Random(7)
    ids = torch.tensor(
        [insert_gists_with_map([rng.randrange(1, 900) for _ in range(700)], Gs, 4, 4, sink=3)[0]]
    )
    assert (ids == Gs[3]).sum() >= 2
    f = build_gist_mask_features(
        ids, Gs, 4, attention_sink_size=3, num_previous_chunks=1, pad_token=0
    )
    T = ids.shape[1]
    from src.data.gist_flex_mask import make_gist_mask_mod

    mm = create_mask(make_gist_mask_mod(f), 1, None, T, T, device="cpu")[0, 0]
    ref = (
        make_hierarchical_gist_mask_greedy(
            ids, Gs, 4, attention_sink_size=3, num_previous_chunks=1, pad_token=0
        )[0, 0].bool()
        & torch.ones(T, T, dtype=torch.bool).tril()
    )
    assert torch.equal(mm, ref)
