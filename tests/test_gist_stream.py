"""Position-only gist insertion and the prefix stability decode-time insertion relies on."""

import random
from types import SimpleNamespace

import pytest
import torch

from gist_utils import Global_data
from src.data.gist_stream import (
    OnlineGistInserter,
    insert_gists_by_position,
    insert_gists_with_map,
    level_groups,
    stream_labels,
)
from tests.test_gist_flex_mask import _check

GIST = [1000, 1001, 1002, 1003, 1004, 1005]


def _raw(n, rng):
    return [rng.randint(1, 999) for _ in range(n)]


@pytest.mark.parametrize("levels,chunk,group,sink", [(1, 4, 1, 0), (2, 4, 4, 2), (4, 4, 4, 3), (6, 4, [4, 2], 2)])
def test_prefix_of_gisted_prefix(levels, chunk, group, sink):
    """gisted(prefix) is a prefix of gisted(extension), for every cut point."""
    rng = random.Random(levels)
    raw = _raw(400, rng)
    full = insert_gists_by_position(raw, GIST[:levels], chunk, group, sink)
    for cut in range(len(raw) + 1):
        part = insert_gists_by_position(raw[:cut], GIST[:levels], chunk, group, sink)
        assert full[:len(part)] == part, cut


def test_online_matches_offline_across_turns():
    """Feeding turns one after another with a persistent inserter equals one-shot."""
    rng = random.Random(0)
    turns = [_raw(rng.randint(1, 50), rng) for _ in range(12)]
    ins = OnlineGistInserter(GIST[:3], 4, 4, sink=3)
    streamed = []
    for turn in turns:
        streamed += insert_gists_with_map(turn, GIST[:3], 4, 4, inserter=ins)[0]
    assert streamed == insert_gists_by_position(sum(turns, []), GIST[:3], 4, 4, sink=3)


def test_level_groups():
    assert level_groups(4, 3) == [4, 4]
    assert level_groups([8, 2], 5) == [8, 2, 2, 2]  # the last factor repeats
    with pytest.raises(ValueError):
        level_groups([4, 0], 3)


@pytest.mark.parametrize("groups", [[3], [4, 2, 2, 4, 2]])
def test_structure(groups):
    """Sink is never chunked; a level-0 gist after every `chunk` raw tokens, a level-l
    gist right after every J_l level-(l-1) gists, nothing else (no trailing gists)."""
    chunk, sink = 4, 2
    out = insert_gists_by_position(_raw(2000, random.Random(1)), GIST, chunk, groups, sink)
    J = level_groups(groups, len(GIST))
    assert all(t not in GIST for t in out[:sink])
    raw_since, since = 0, [0] * len(GIST)  # since[l]: level-(l-1) gists since the last level-l
    prev = None
    for t in out[sink:]:
        if t not in GIST:
            raw_since += 1
        elif t == GIST[0]:
            assert raw_since == chunk
            raw_since, since[1] = 0, since[1] + 1
        else:
            lvl = GIST.index(t)
            assert prev == GIST[lvl - 1] and since[lvl] == J[lvl - 1]
            since[lvl] = 0
            if lvl + 1 < len(GIST):
                since[lvl + 1] += 1
        prev = t
    assert raw_since < chunk


def test_labels_skip_gists_but_predict_what_follows():
    rng = random.Random(2)
    out = insert_gists_by_position(_raw(40, rng), GIST[:2], 4, 2)
    labels = stream_labels(out, GIST[:2])
    for i, (t, y) in enumerate(zip(out, labels)):
        assert (y == -100) == (t in GIST), i
    after_gist = [i + 1 for i, t in enumerate(out[:-1]) if t in GIST and out[i + 1] not in GIST]
    assert after_gist and all(labels[i] == out[i] for i in after_gist)


@pytest.mark.parametrize("levels,prev,window,groups", [
    (1, 1, 0, 4), (2, 1, 0, 4), (3, 0, 0, 4), (2, 1, 24, 4),
    (5, 1, 0, [2, 2, 2, 2]), (6, 1, 32, [2]), (6, 2, 0, [4, 2, 2, 2, 2])])
def test_stream_layout_masks_agree(levels, prev, window, groups):
    """The flex mask matches the dense mask on stream-gisted sequences, up to 6 levels."""
    rng = random.Random(levels)
    n = 150 if levels <= 3 else 400
    seqs = [insert_gists_by_position(_raw(rng.randint(n // 5, n), rng), GIST[:levels], 4, groups, 2)
            for _ in range(3)]
    T = max(map(len, seqs))
    _check(torch.tensor([[0] * (T - len(s)) + s for s in seqs]), levels, 4, 3, prev, window)


class _WordTokenizer:
    """Deterministic stand-in for a tokenizer: one id per whitespace word."""

    bos_token_id, eos_token_id = 1, 2

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[3 + (sum(map(ord, w)) % 900) for w in text.split()])


@pytest.mark.parametrize("meta", [0, 1])
def test_qa_stream_format(meta):
    from src.data.gist_input_preprocessor import gist_attention_preprocessor

    saved = Global_data.sink_size
    Global_data.sink_size = 3
    try:
        tok = _WordTokenizer()
        pre = gist_attention_preprocessor(
            tok, max_len=4096, do_shuffle=False, chunk_size=[4, 4],
            gist_token_ids=GIST[:2] if meta else GIST[:1], pad_token=0,
            meta_gist=meta, stream_gists=True)
        answer = " ".join(f"a{i}" for i in range(40))
        example = {
            "question": "what is the answer to the question",
            "documents": [{"title": f"t{j}", "text": " ".join(f"w{j}_{i}" for i in range(30))} for j in range(3)],
            "generated": answer,
        }
        out = pre.gist_process_qa(example)
    finally:
        Global_data.sink_size = saved

    ids, labels = out["input_ids"], out["labels"]
    gist = set(GIST)
    ans_ids = tok(answer).input_ids + [tok.eos_token_id]
    # Raw tokens with loss are exactly the answer plus EOS, in order.
    assert [y for y in labels if y != -100] == ans_ids
    # Gists appear inside the answer region, with label -100.
    first_ans = labels.index(ans_ids[0])
    assert any(t in gist for t in ids[first_ans:])
    assert all(labels[i] == -100 for i, t in enumerate(ids) if t in gist)
    # Removing gists leaves BOS first and the answer, unbroken, at the end.
    raw_only = [t for t in ids if t not in gist]
    assert raw_only[0] == tok.bos_token_id
    assert raw_only[-len(ans_ids):] == ans_ids
