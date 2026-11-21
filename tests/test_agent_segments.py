"""Agent-trajectory segments (ADR 0001, #17): pinned tokens are never compressed and stay
visible (R7); the raw window can also be bounded by turns (R6). The flex mask_mod must
equal the dense reference (greedy mask + add_raw_window + add_pinned, AND causal)."""

import itertools
import random

import pytest
import torch
from torch.nn.attention.flex_attention import create_mask

from src.data.gist import add_pinned, add_raw_window, make_hierarchical_gist_mask_greedy
from src.data.gist_flex_mask import (
    build_gist_mask_features,
    make_gist_mask_mod,
    make_teacher_mask_mod,
)
from src.data.gist_stream import OnlineGistInserter, gist_segments, insert_gists_with_map

PAD = 0
GIST_IDS = [1000, 1001, 1002, 1003]


def _trajectory(rng, n_pinned, turns):
    """Raw tokens, pinned flags and turn ids: a pinned prefix, then turns of an assistant
    span and a tool-result span of random lengths."""
    ids, pinned, turn = [], [], []
    for _ in range(n_pinned):
        ids.append(rng.randint(1, 999)), pinned.append(True), turn.append(0)
    for t in range(1, turns + 1):
        for _ in range(rng.randint(1, 12) + rng.randint(0, 40)):
            ids.append(rng.randint(1, 999)), pinned.append(False), turn.append(t)
    return ids, pinned, turn


def test_pinned_tokens_are_skipped_and_the_stream_continues():
    rng = random.Random(0)
    rest = [rng.randint(1, 999) for _ in range(90)]
    head = [rng.randint(1, 999) for _ in range(17)]
    alone, _ = insert_gists_with_map(rest, GIST_IDS[:3], 4, 4, sink=3)
    both, _ = insert_gists_with_map(head + rest, GIST_IDS[:3], 4, 4, sink=3,
                                    pinned=[True] * 17 + [False] * 90)
    assert both[:17] == head and both[17:] == alone
    # pinned tokens in the middle do not open or fill a chunk
    ins = OnlineGistInserter(GIST_IDS[:2], 4, 4, sink=0)
    out = [ins.push(t, p) for t, p in zip(range(1, 10), [0, 0, 1, 1, 1, 0, 0, 0, 0])]
    assert [i for i, g in enumerate(out) if g] == [6]


def test_gist_segments_carries_flags_and_turns():
    rng = random.Random(1)
    ids, pinned, turn = _trajectory(rng, 9, 6)
    g, gp, gt = gist_segments(ids, pinned, turn, GIST_IDS[:2], 4, 4, sink=3)
    assert [t for t in g if t not in GIST_IDS] == ids
    raw = [i for i, t in enumerate(g) if t not in GIST_IDS]
    assert [gp[i] for i in raw] == pinned and [gt[i] for i in raw] == turn
    for i, t in enumerate(g):
        if t in GIST_IDS:
            assert not gp[i] and gt[i] == gt[i - 1]


def _inputs(rng, levels, n_pinned, turns, pad):
    ids, pinned, turn = _trajectory(rng, n_pinned, turns)
    g, gp, gt = gist_segments(ids, pinned, turn, GIST_IDS[:levels], 4, 4, sink=3)
    g, gp, gt = [PAD] * pad + g, [False] * pad + gp, [0] * pad + gt
    return torch.tensor([g]), torch.tensor([gp]), torch.tensor([gt])


@pytest.mark.parametrize(
    "levels,window,turns_back", list(itertools.product([2, 3, 4], [0, 24], [0, 1, 2])))
def test_flex_mask_equals_dense_reference(levels, window, turns_back):
    rng = random.Random(100 * levels + 10 * window + turns_back)
    for n_pinned, pad in [(0, 0), (11, 0), (23, 5)]:
        ids, pinned, turn = _inputs(rng, levels, n_pinned, 8, pad)
        gist = GIST_IDS[:levels]
        T = ids.shape[1]
        # the sink is the first 3 unpinned raw tokens: with a pinned prefix it ends at
        # absolute position pad + n_pinned + 3 (left padding counts as positions)
        dense = make_hierarchical_gist_mask_greedy(
            ids, gist, 4, attention_sink_size=pad + n_pinned + 3 if n_pinned else 3,
            num_previous_chunks=1, pad_token=PAD)
        dense = add_raw_window(dense, ids, gist, window, PAD, turn=turn, window_turns=turns_back)
        dense = add_pinned(dense, pinned)
        expected = dense[:, 0] & torch.ones(T, T, dtype=torch.bool).tril()
        f = build_gist_mask_features(ids, gist, 4, 3, 1, pad_token=PAD, raw_window=window,
                                     pinned=pinned, turn=turn, window_turns=turns_back)
        got = create_mask(make_gist_mask_mod(f), 1, None, T, T, device="cpu")[:, 0]
        diff = (got != expected).nonzero()
        assert diff.numel() == 0, f"{diff.shape[0]} mismatches, first {diff[:5].tolist()}"


def test_window_turns_bounds_the_raw_window():
    rng = random.Random(7)
    ids, pinned, turn = _inputs(rng, 2, 5, 6, 0)
    T = ids.shape[1]
    f = build_gist_mask_features(ids, GIST_IDS[:2], 4, 3, 1, pad_token=PAD, raw_window=10**6,
                                 turn=turn, window_turns=2)
    m = create_mask(make_gist_mask_mod(f), 1, None, T, T, device="cpu")[0, 0]
    raw = ~torch.isin(ids[0], torch.tensor(GIST_IDS))
    q = int(torch.nonzero(raw).max())  # the window applies to raw queries only
    # with an unbounded token window, the last query sees exactly the raw keys of its
    # turn and the previous one beyond what the gist mask already shows
    two = raw & (turn[0] >= turn[0, q] - 1)
    assert m[q][two].all()
    old = raw & (turn[0] < turn[0, q] - 1) & (torch.arange(T) >= 3)
    f0 = build_gist_mask_features(ids, GIST_IDS[:2], 4, 3, 1, pad_token=PAD)
    m0 = create_mask(make_gist_mask_mod(f0), 1, None, T, T, device="cpu")[0, 0]
    assert torch.equal(m[q][old], m0[q][old])


@pytest.mark.parametrize("kind", ["g1", "g2", "g3"])
def test_teacher_masks_follow_pinned_and_window(kind):
    rng = random.Random(3)
    ids, pinned, turn = _inputs(rng, 4, 13, 6, 0)
    T = ids.shape[1]
    f = build_gist_mask_features(ids, GIST_IDS, 4, 3, 1, pad_token=PAD, raw_window=24,
                                 pinned=pinned, turn=turn, window_turns=2)
    m = create_mask(make_teacher_mask_mod(f, kind), 1, None, T, T, device="cpu")[0, 0]
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    lvl = torch.full((T,), -1)
    for l, g in enumerate(GIST_IDS):
        lvl[ids[0] == g] = l
    m_ = int(kind[1:])
    rows = lvl < m_  # rows of deleted levels see only themselves
    assert (m[rows][:, pinned[0]] | ~causal[rows][:, pinned[0]]).all()
    raw = lvl < 0
    win = add_raw_window(torch.zeros(1, 1, T, T, dtype=torch.bool), ids, GIST_IDS, 24, PAD,
                         turn=turn, window_turns=2)[0, 0] & causal
    assert (m | ~win)[raw].all()
