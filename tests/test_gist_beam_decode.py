"""DECODE_SELECT=beam through a tiny Qwen2 (#19): every layer past the first selects at
every decode step, the recorded ids are valid nodes, and the default path is untouched."""

import pytest
import torch
from transformers import Qwen2Config

import gist_opsd
from gist_qwen2 import GistQwen2ForCausalLM
from gist_utils import Global_data
from src.data.gist_stream import insert_gists_with_map

GIST_IDS = [1000, 1001, 1002]
KEYS = ("gist_token_id", "chunk_size", "sink_size", "num_previous_chunks", "pad_token_id",
        "train_attn", "top_k", "last_pos", "beam_trace", "beam_segments")


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setenv("ATTN_IMPL", "ref")
    saved = {k: getattr(Global_data, k) for k in KEYS}
    Global_data.gist_token_id, Global_data.chunk_size = GIST_IDS, [4] * 3
    Global_data.top_k = [3] * 3
    Global_data.sink_size, Global_data.num_previous_chunks = 3, 1
    Global_data.pad_token_id, Global_data.train_attn = 0, "dense"
    cfg = Qwen2Config(vocab_size=1100, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = GistQwen2ForCausalLM(cfg).float().eval()
    model.selective = True
    raw = torch.randint(1, 1000, (300,), generator=torch.Generator().manual_seed(1)).tolist()
    ctx = torch.tensor([insert_gists_with_map(raw, GIST_IDS, 4, 4, sink=3)[0]])
    yield model, ctx
    for k, v in saved.items():
        setattr(Global_data, k, v)


def _decode(model, ctx, steps):
    attn = torch.ones_like(ctx)
    row = gist_opsd.prefill_gist_mask(ctx)
    with torch.no_grad():
        out = model(input_ids=ctx, attention_mask=attn, attention_mask_gist=row, use_cache=True)
        past, row, logits = out.past_key_values, row[:, :, -1:], [out.logits[0, -1]]
        y = out.logits[0, -1, :1000].argmax().view(1, 1)
        for _ in range(steps):
            attn = torch.cat([attn, attn.new_ones(1, 1)], -1)
            row = torch.cat([row, row.new_ones(1, 1, 1, 1)], -1)
            out = model(input_ids=y, attention_mask=attn, attention_mask_gist=row,
                        past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits.append(out.logits[0, -1])
            y = out.logits[0, -1, :1000].argmax().view(1, 1)
    return torch.stack(logits)


@pytest.mark.parametrize("coarse", ["0", "1"])
def test_beam_decode_selects_at_every_layer(setup, monkeypatch, coarse):
    model, ctx = setup
    monkeypatch.setenv("DECODE_SELECT", "beam")
    monkeypatch.setenv("BEAM_K", "2,1")
    monkeypatch.setenv("BEAM_COARSE", coarse)
    Global_data.beam_trace = []
    logits = _decode(model, ctx, 3)
    assert torch.isfinite(logits).all()
    trace = Global_data.beam_trace
    assert len(trace) == 3 * (model.config.num_hidden_layers - 1)  # layer 0 is plain attention
    n_nodes = [int((ctx[0] == g).sum()) for g in GIST_IDS]
    for layer, rec in trace:
        assert layer >= 1 and len(rec) == model.config.num_key_value_heads
        for ids in rec:
            for l, sel in enumerate(ids):
                assert sel.numel() and int(sel.max()) < n_nodes[l]


def test_default_decode_is_unchanged(setup, monkeypatch):
    model, ctx = setup
    monkeypatch.delenv("DECODE_SELECT", raising=False)
    a = _decode(model, ctx, 3)
    monkeypatch.setenv("BEAM_K", "1")  # ignored without DECODE_SELECT=beam
    b = _decode(model, ctx, 3)
    assert torch.equal(a, b)
    monkeypatch.setenv("DECODE_SELECT", "beam")
    assert not torch.equal(a, _decode(model, ctx, 3))
