"""Appending onto a KV cache (gist_session.py, #18): appending a trajectory in pieces gives
the cache of prefilling it at once, and decoding continues from either the same way."""

import random

import pytest
import torch
from transformers import Qwen2Config

import gist_flex_attention
from gist_lora import apply_lora, apply_summary_bias, summary_spans
from gist_qwen2 import GistQwen2ForCausalLM
from gist_session import GistSession
from gist_utils import Global_data

KEYS = ("gist_token_id", "chunk_size", "sink_size", "num_previous_chunks", "pad_token_id",
        "train_attn", "raw_window", "window_turns", "top_k", "last_pos", "stage1",
        "decode_gists", "beam_segments", "lora_row_mask", "lora_row_level")


@pytest.fixture(params=[2, 4], ids=["L2", "L4"])
def model(request, monkeypatch):
    L = request.param
    monkeypatch.setenv("ATTN_IMPL", "ref")
    saved = {k: getattr(Global_data, k) for k in KEYS}
    Global_data.gist_token_id = [1000 + i for i in range(L)]
    Global_data.chunk_size = [4] * L
    Global_data.sink_size, Global_data.num_previous_chunks = 3, 1
    Global_data.pad_token_id, Global_data.stage1 = 0, None
    Global_data.raw_window, Global_data.window_turns = 24, 2
    Global_data.top_k = [3] * max(3, L)
    cfg = Qwen2Config(vocab_size=1100, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
    cfg._attn_implementation = gist_flex_attention.GIST_FLEX
    torch.manual_seed(0)
    m = GistQwen2ForCausalLM(cfg).float()
    apply_lora(m, "gist", r=4)
    apply_summary_bias(m, spans=summary_spans(L))
    for n, p in m.named_parameters():  # make the gist parameters and beta matter
        if "lora_B" in n or "summary_bias" in n:
            p.data.normal_(0, 0.3)
    m.eval()
    m.selective = True
    yield m
    for k, v in saved.items():
        setattr(Global_data, k, v)


def _segments(seed=0):
    rng = random.Random(seed)
    segs = [([rng.randint(1, 999) for _ in range(13)], True, 0)]  # system + tools
    for t in range(1, 6):
        segs.append(([rng.randint(1, 999) for _ in range(rng.randint(5, 70))], False, t))
    return segs


def _kv(past):
    if hasattr(past, "layers"):
        return [(l.keys, l.values) for l in past.layers]
    return list(zip(past.key_cache, past.value_cache))


def test_appending_in_pieces_equals_one_prefill(model):
    segs = _segments()
    one = GistSession(model)
    one.append([t for s in segs for t in s[0]], [s[1] for s in segs for _ in s[0]],
               [s[2] for s in segs for _ in s[0]])
    kv_one, beta_one = _kv(one.past), [l.self_attn._beta_cache.clone() for l in model.model.layers[1:]]
    pieces = GistSession(model)
    for raw, pinned, turn in segs:
        pieces.append(raw, pinned, turn)
    assert pieces.ids == one.ids and pieces.pinned == one.pinned and pieces.turn == one.turn
    assert any(t in Global_data.gist_token_id for t in one.ids)
    for (k1, v1), (k2, v2) in zip(kv_one, _kv(pieces.past)):
        torch.testing.assert_close(k2, k1, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(v2, v1, rtol=1e-4, atol=1e-5)
    for b1, layer in zip(beta_one, model.model.layers[1:]):
        torch.testing.assert_close(layer.self_attn._beta_cache, b1, rtol=1e-4, atol=1e-5)


def test_decode_continues_after_appends(model):
    segs = _segments(1)
    one, pieces = GistSession(model), GistSession(model)
    one.append([t for s in segs for t in s[0]], [s[1] for s in segs for _ in s[0]],
               [s[2] for s in segs for _ in s[0]])
    a = [one.step(t) for t in (5, 6, 7, 8, 9)]
    for raw, pinned, turn in segs:
        pieces.append(raw, pinned, turn)
    b = [pieces.step(t) for t in (5, 6, 7, 8, 9)]
    assert pieces.ids == one.ids
    for x, y in zip(a, b):
        torch.testing.assert_close(y, x, rtol=1e-4, atol=1e-4)
    # and appending after decoding works too (a tool result after a generated turn)
    one.append([11, 12, 13] * 9, False, 6)
    pieces.append([11, 12, 13] * 9, False, 6)
    for (k1, _), (k2, _) in zip(_kv(one.past), _kv(pieces.past)):
        torch.testing.assert_close(k2, k1, rtol=1e-4, atol=1e-5)
