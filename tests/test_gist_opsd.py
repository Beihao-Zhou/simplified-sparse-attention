"""OPSD mechanics on a tiny model: replay reproduces the rollout, gradients land where expected."""

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

import gist_opsd
from gist_qwen2 import GistQwen2ForCausalLM
from gist_utils import Global_data
from src.data.gist_stream import insert_gists_with_map

VOCAB = 1000  # base vocabulary; summary ids sit above it
GIST_IDS = [1000, 1001]


@pytest.fixture(params=[2, 3], ids=["L2", "L3"])
def setup(monkeypatch, request):
    gist_ids = GIST_IDS + [1002] * (request.param > 2)
    monkeypatch.setenv("ATTN_IMPL", "ref")
    keys = (
        "gist_token_id",
        "chunk_size",
        "sink_size",
        "num_previous_chunks",
        "pad_token_id",
        "train_attn",
        "top_k",
        "last_pos",
        "opsd_replay",
    )
    saved = {k: getattr(Global_data, k) for k in keys}
    Global_data.gist_token_id, Global_data.chunk_size = gist_ids, [4] * len(gist_ids)
    Global_data.top_k = [3] * 3
    Global_data.sink_size, Global_data.num_previous_chunks = 3, 1
    Global_data.pad_token_id, Global_data.train_attn = 0, "dense"
    cfg = Qwen2Config(
        vocab_size=1100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = GistQwen2ForCausalLM(cfg).float().eval()
    model.selective = True
    plain = Qwen2ForCausalLM(cfg).float().eval()
    plain.load_state_dict(model.state_dict())
    g = torch.Generator().manual_seed(1)
    raw = torch.randint(1, VOCAB, (240,), generator=g).tolist()
    ctx = torch.tensor([insert_gists_with_map(raw, gist_ids, 4, 4, sink=3)[0]])
    yield model, plain, ctx
    for k, v in saved.items():
        setattr(Global_data, k, v)


def test_replay_reproduces_rollout(setup):
    model, _, ctx = setup
    y, lp_roll, steps = gist_opsd.rollout(model, ctx, 8, VOCAB, torch.Generator().manual_seed(0))
    assert len(steps) == 7 and len(steps[0]) == model.config.num_hidden_layers
    lp_rep, _ = gist_opsd.replay(model, ctx, y, steps, VOCAB)
    torch.testing.assert_close(lp_rep, lp_roll, rtol=0, atol=1e-4)


def test_selection_is_sparse_and_replayed(setup):
    """Generated rows see only some summaries; replay uses exactly the recorded keys."""
    model, _, ctx = setup
    y, _, steps = gist_opsd.rollout(model, ctx, 6, VOCAB, torch.Generator().manual_seed(0))
    is_sum = torch.isin(ctx[0], torch.tensor(Global_data.gist_token_id))
    seen = steps[0][1][:, : ctx.shape[1]] & is_sum
    assert 0 < int(seen.any(0).sum()) < int(is_sum.sum())


def test_gradients(setup):
    model, plain, ctx = setup
    y, _, steps = gist_opsd.rollout(model, ctx, 6, VOCAB, torch.Generator().manual_seed(0))
    lp_t = gist_opsd.teacher_logprobs(plain, ctx, y, VOCAB)
    assert not lp_t.requires_grad

    lp_s, st = gist_opsd.replay(model, ctx, y, steps, VOCAB, probe=True)
    gist_opsd.reverse_kl(lp_s, lp_t).mean().backward()
    is_sum = torch.isin(torch.cat([ctx[0], y[:-1]]), torch.tensor(Global_data.gist_token_id))
    hpg = model.config.num_attention_heads // model.config.num_key_value_heads
    checked = 0
    for layer, key in st["probe"].items():
        vis = st["masks"][layer][0]  # [H, R, K]
        seen = vis.view(-1, hpg, *vis.shape[1:]).any(1).any(1)  # [num_kv, K]
        g = key.grad[0].abs().sum(-1)  # [num_kv, K]
        unseen_sum = ~seen & is_sum
        seen_sum = seen & is_sum
        assert g[unseen_sum].max() == 0, f"layer {layer}: unselected summary got reverse-KL grad"
        assert g[seen_sum].min() > 0, f"layer {layer}: selected summary got no grad"
        checked += int(unseen_sum.sum())
    assert checked > 0

    model.zero_grad()
    _, st = gist_opsd.replay(model, ctx, y, steps, VOCAB, route=True, probe=True)
    st["route"].backward()
    for layer, key in st["probe"].items():
        vis = st["masks"][layer][0]
        seen = vis.view(-1, hpg, *vis.shape[1:]).any(1).any(1)
        g = key.grad[0].abs().sum(-1)
        assert (
            g[~seen & is_sum].gt(0).float().mean() > 0.9
        ), f"layer {layer}: routing KL misses unselected summaries"


def test_beta_folded_into_qk_matches_explicit_bias(setup):
    """Rollout folds beta into an extra q/k coordinate inside the eager paths; replay adds
    it to the attention bias. With a nonzero beta they still agree, and beta matters."""
    from gist_lora import apply_summary_bias

    model, _, ctx = setup
    y0, lp0, _ = gist_opsd.rollout(model, ctx, 6, VOCAB, torch.Generator().manual_seed(0))
    apply_summary_bias(model)
    torch.manual_seed(5)
    for layer in model.model.layers[1:]:
        layer.self_attn.summary_bias.u.data.normal_(0, 0.3)
    y, lp_roll, steps = gist_opsd.rollout(model, ctx, 6, VOCAB, torch.Generator().manual_seed(0))
    lp_rep, _ = gist_opsd.replay(model, ctx, y, steps, VOCAB)
    torch.testing.assert_close(lp_rep, lp_roll, rtol=0, atol=1e-4)
    assert (lp_roll[0] - lp0[0]).abs().max() > 1e-3


def test_forced_and_greedy_rollout(setup):
    """Forced tokens (off-policy control) are fed as given and replay still matches;
    greedy picks the argmax of each step's distribution."""
    model, _, ctx = setup
    forced = torch.tensor([5, 9, 2, 7, 3, 1])
    y, lp_roll, steps = gist_opsd.rollout(model, ctx, 6, VOCAB, forced=forced)
    assert torch.equal(y, forced)
    lp_s, _ = gist_opsd.replay(model, ctx, y, steps, VOCAB)
    torch.testing.assert_close(lp_s, lp_roll, rtol=0, atol=1e-4)
    yg, lpg, _ = gist_opsd.rollout(model, ctx, 6, VOCAB, greedy=True)
    assert torch.equal(yg, lpg.argmax(-1))


def _raw_rows(ctx, y):
    """Row of ctx + y[:-1] that predicts each raw token of y."""
    g = torch.tensor(Global_data.gist_token_id)
    Tc = ctx.shape[1]
    nxt = torch.nonzero(~torch.isin(y[1:], g)).flatten()
    return torch.cat([torch.tensor([Tc - 1]), Tc + nxt])


def test_decode_gists_match_prefill(setup):
    """Without unfolding, decoding with summaries inserted on the fly reproduces a prefill
    of the same sequence: the rows follow the prefill mask, summaries share the last raw
    token's RoPE position, and the next raw token is predicted from the summary's row."""
    model, _, ctx = setup
    model.selective = False
    g = torch.Generator().manual_seed(5)
    forced = torch.randint(1, VOCAB, (40,), generator=g)
    y, lps, steps = gist_opsd.rollout(model, ctx, 40, VOCAB, forced=forced, insert_gists=True)
    assert torch.equal(gist_opsd.raw_tokens(y), forced)
    assert len(y) > 40 and len(steps) == len(y) - 1  # summaries were inserted and fed
    full = torch.cat([ctx[0], y[:-1]])[None]
    with torch.no_grad():
        out = model(
            input_ids=full,
            attention_mask=torch.ones_like(full),
            attention_mask_gist=gist_opsd.prefill_gist_mask(full),
        )
    ref = torch.log_softmax(out.logits[0, _raw_rows(ctx, y), :VOCAB].float(), -1)
    torch.testing.assert_close(lps, ref, rtol=0, atol=1e-4)


def test_replay_with_decode_gists(setup):
    model, plain, ctx = setup
    y, lp_roll, steps = gist_opsd.rollout(
        model, ctx, 24, VOCAB, torch.Generator().manual_seed(0), insert_gists=True
    )
    assert len(y) > 24
    lp_rep, _ = gist_opsd.replay(model, ctx, y, steps, VOCAB, route=True)
    torch.testing.assert_close(lp_rep, lp_roll, rtol=0, atol=1e-4)
    assert gist_opsd.teacher_logprobs(plain, ctx, y, VOCAB).shape == lp_roll.shape


def test_insert_gists_needs_a_completed_context(setup):
    model, _, ctx = setup
    with pytest.raises(ValueError):
        gist_opsd.rollout(
            model, ctx[:, :-1] if ctx[0, -1] >= VOCAB else ctx[:, :-2], 2, VOCAB, insert_gists=True
        )
