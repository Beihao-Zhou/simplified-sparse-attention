"""OPSD on long agent trajectories (gist_opsd_long.py, #21): the replay recomputes exactly
what the rollout sampled from, through flex with the recorded keys."""

import random

import pytest
import torch
from transformers import Qwen2Config

import gist_flex_attention
import gist_opsd_long as ol
from gist_lora import apply_lora, apply_summary_bias, summary_spans
from gist_qwen2 import GistQwen2ForCausalLM
from gist_session import GistSession
from gist_utils import Global_data

VOCAB = 1000
KEYS = ("gist_token_id", "chunk_size", "sink_size", "num_previous_chunks", "pad_token_id",
        "train_attn", "raw_window", "window_turns", "top_k", "last_pos", "stage1",
        "decode_gists", "beam_segments", "opsd_record", "opsd_flex", "lora_row_mask",
        "lora_row_level")


@pytest.fixture
def setup(monkeypatch):
    L = 3
    monkeypatch.setenv("ATTN_IMPL", "ref")
    monkeypatch.setenv("DECODE_SELECT", "beam")
    monkeypatch.setenv("BEAM_K", "3,2")
    saved = {k: getattr(Global_data, k) for k in KEYS}
    Global_data.gist_token_id = [1000 + i for i in range(L)]
    Global_data.chunk_size = [4] * L
    Global_data.sink_size, Global_data.num_previous_chunks = 3, 1
    Global_data.pad_token_id, Global_data.stage1 = 0, None
    Global_data.raw_window, Global_data.window_turns = 24, 2
    Global_data.top_k = [3] * 3
    cfg = Qwen2Config(vocab_size=1100, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
    cfg._attn_implementation = gist_flex_attention.GIST_FLEX
    torch.manual_seed(0)
    m = GistQwen2ForCausalLM(cfg).float()
    apply_lora(m, "gist", r=4)
    apply_summary_bias(m, spans=summary_spans(L))
    for n, p in m.named_parameters():
        if "lora_B" in n or "summary_bias" in n:
            p.data.normal_(0, 0.3)
    m.eval()
    m.selective = True
    rng = random.Random(0)
    segs = [([rng.randint(1, 999) for _ in range(9)], True, 0)] + [
        ([rng.randint(1, 999) for _ in range(rng.randint(20, 60))], False, t) for t in range(1, 8)]
    yield m, segs
    for k, v in saved.items():
        setattr(Global_data, k, v)


def _rollout(model, segs, n_new=30, seed=0):
    s = GistSession(model)
    for raw, pin, turn in segs:
        s.append(raw, pin, turn)
    ctx = (list(s.ids), list(s.pinned), list(s.turn))
    first = model.lm_head(s.last_hidden)
    stream, lps, recs = ol.rollout(s, first, n_new, VOCAB, torch.Generator().manual_seed(seed))
    return ctx, stream, lps, recs


@pytest.mark.parametrize("coarse", ["0", "1"])
def test_replay_reproduces_the_rollout(setup, monkeypatch, coarse):
    model, segs = setup
    monkeypatch.setenv("BEAM_COARSE", coarse)
    ctx, stream, lps, recs = _rollout(model, segs)
    gset = set(Global_data.gist_token_id)
    assert any(t in gset for t in stream), "summaries are inserted during decoding"
    assert len(recs) == len(stream) - 1 and all(len(r) == 3 for r in recs)
    with torch.no_grad():
        rep, st = ol.replay(model, *ctx, stream, recs, VOCAB)
    ol.end_replay(st)
    assert rep.shape == lps.shape
    torch.testing.assert_close(rep, lps, rtol=0, atol=1e-4)


def test_recorded_keys_are_a_selection(setup):
    model, segs = setup
    ctx, stream, lps, recs = _rollout(model, segs, n_new=6)
    Tc = len(ctx[0])
    is_sum = torch.isin(torch.tensor(ctx[0]), torch.tensor(Global_data.gist_token_id))
    seen = torch.zeros(Tc, dtype=torch.bool)
    for g in recs[0][1]:
        seen[g[(g >= 0) & (g < Tc)].long()] = True
    assert 0 < int((seen & is_sum).sum()) < int(is_sum.sum())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex backward needs CUDA")
@pytest.mark.parametrize("coarse", ["0", "1"])
def test_gradients_reach_what_was_attended(setup, monkeypatch, coarse):
    model, segs = setup
    for n, p in model.named_parameters():  # as in training: bf16 base, fp32 adapters
        if "lora_" not in n and "summary_bias" not in n:
            p.data = p.data.bfloat16()
    model.cuda()
    monkeypatch.setenv("BEAM_COARSE", coarse)
    ctx, stream, lps, recs = _rollout(model, segs, n_new=12)
    last = model.config.num_hidden_layers - 1
    model.train()
    lp, st = ol.replay(model, *ctx, stream, recs, VOCAB, probe_layer=last)
    teacher = torch.randn_like(lp).log_softmax(-1)  # any target; only the gradient paths matter
    ol.reverse_kl(lp, teacher).mean().backward()
    ol.end_replay(st)
    key = st["probe"]                                         # [1, H_kv, T, d], last layer
    g = key.grad[0].abs().sum(-1)                             # [H_kv, T]
    Tc = len(ctx[0])
    is_sum = torch.isin(torch.tensor(ctx[0], device=g.device),
                        torch.tensor(Global_data.gist_token_id, device=g.device))
    seen = torch.zeros(g.shape[0], Tc, dtype=torch.bool, device=g.device)
    for r in recs:
        for h, pos in enumerate(r[last]):
            seen[h, pos[(pos >= 0) & (pos < Tc)].long()] = True
    if coarse == "1":  # summaries recorded at the last layer are coarse terms: gradient
        assert (g[:, :Tc][seen & is_sum] > 0).all()
    unseen = ~seen & is_sum[None]
    assert unseen.any() and (g[:, :Tc][unseen] == 0).all(), "only attended keys get gradient"
    model.zero_grad()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    _, st = ol.replay(model, *ctx, stream, recs, VOCAB, route={"scope": "level"})
    st["route"].backward()
    # the checkpoint rerun must not append a second loss per layer (it leaked its graph)
    assert len(Global_data.opsd_flex["route_losses"]) == last
    ol.end_replay(st)
    model.gradient_checkpointing_disable()
    assert float(st["route"]) > 0
    sb = [p for n, p in model.named_parameters() if "summary_bias" in n]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in sb)
    model.eval()
