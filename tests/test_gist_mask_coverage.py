"""The gist mask must show every query its whole past, each raw token exactly once.

Unlike the equivalence tests, this checks the mask against its meaning instead of
against another implementation. A level-0 gist stands for the raw tokens of its
chunk and a level-l gist for the spans of its level-(l-1) gists, so the keys a query
sees should partition its past (a cut through the gist tree):

* complete: every earlier raw token is visible or inside the span of a visible gist;
* exact with P = 0 and W = 0: no raw token is covered twice. P > 0 shows the previous
  chunks both raw and summarized, and the raw window (R6) re-exposes raw tokens that
  gists already cover, so duplicates are allowed there (within the window for W).
"""

import random

import pytest
import torch
from torch.nn.attention.flex_attention import create_mask

from src.data.gist import add_raw_window, make_hierarchical_gist_mask_greedy
from src.data.gist_flex_mask import build_gist_mask_features, make_gist_mask_mod
from src.data.gist_stream import insert_gists_by_position
from tests.test_gist_flex_mask import GIST_IDS, _structured

GIST = GIST_IDS + [1004, 1005]


def _spans(seq, gist_ids, sink):
    """Raw positions each token stands for: itself for raw tokens, its subtree for gists.
    Sink tokens are visible to every query, so no gist stands for them."""
    level = {g: lvl for lvl, g in enumerate(gist_ids)}
    pending = [[] for _ in range(len(gist_ids) + 1)]  # pending[l]: spans awaiting a level-l gist
    spans = []
    for pos, t in enumerate(seq):
        if t not in level:
            span = {pos}
        else:
            lvl = level[t]
            span = set().union(*pending[lvl])
            pending[lvl] = []
        spans.append(span)
        if t not in level:
            if pos >= sink:
                pending[0].append(span)
        elif level[t] + 1 < len(pending):
            pending[level[t] + 1].append(span)
    return spans


def _visible(seq, gist_ids, chunk, sink, prev, window, impl):
    ids = torch.tensor([seq])
    T = len(seq)
    if impl == "flex":
        f = build_gist_mask_features(ids, gist_ids, chunk, sink, prev, raw_window=window)
        return create_mask(make_gist_mask_mod(f), 1, None, T, T, device="cpu")[0, 0]
    m = make_hierarchical_gist_mask_greedy(ids, gist_ids, chunk, attention_sink_size=sink,
                                           num_previous_chunks=prev)
    m = add_raw_window(m, ids, gist_ids, window)[0, 0]
    return m & torch.ones(T, T, dtype=torch.bool).tril()


def _check_coverage(seq, gist_ids, chunk, sink, prev, window, impl):
    spans = _spans(seq, gist_ids, sink)
    raw = [t not in gist_ids for t in seq]
    vis = _visible(seq, gist_ids, chunk, sink, prev, window, impl)
    for q in range(len(seq)):
        count = {}
        for k in vis[q].nonzero().flatten().tolist():
            if k == q:
                continue
            for r in spans[k]:
                count[r] = count.get(r, 0) + 1
        past = [r for r in range(q) if raw[r]]
        missing = [r for r in past if r not in count]
        assert not missing, f"query {q}: raw tokens {missing[:8]} are neither visible nor summarized"
        if prev == 0:
            dup = [r for r, c in count.items() if c > 1 and not (window and q - r < window)]
            assert not dup, f"query {q}: raw tokens {dup[:8]} are covered more than once"


@pytest.mark.parametrize("impl", ["flex", "dense"])
@pytest.mark.parametrize("levels,groups", [(1, 4), (2, [4]), (3, [2]), (6, [4, 2])])
@pytest.mark.parametrize("sink,prev,window", [(0, 0, 0), (3, 0, 0), (2, 1, 0), (2, 0, 24), (2, 2, 24)])
def test_streaming_layout(impl, levels, groups, sink, prev, window):
    """The layout we train on: position-only insertion, no trailing gists."""
    rng = random.Random(levels * 7 + sink)
    seq = insert_gists_by_position([rng.randint(1, 999) for _ in range(300)],
                                   GIST[:levels], 4, groups, sink)
    _check_coverage(seq, GIST[:levels], 4, sink, prev, window, impl)


@pytest.mark.parametrize("impl", ["flex", "dense"])
@pytest.mark.parametrize("levels", [1, 2, 3])
@pytest.mark.parametrize("sink,prev", [(0, 0), (3, 0), (3, 1)])
def test_released_layout(impl, levels, sink, prev):
    """Word-aware-like layout with trailing gists, as in the released data, followed by
    an ungisted continuation."""
    rng = random.Random(levels * 11 + sink)
    seq = [rng.randint(1, 999) for _ in range(sink)]
    seq += _structured(150, levels, 4, 4, jitter=2, rng=rng)
    seq += [rng.randint(1, 999) for _ in range(40)]
    _check_coverage(seq, GIST_IDS[:levels], 4, sink, prev, 0, impl)
