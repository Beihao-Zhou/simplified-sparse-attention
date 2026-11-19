"""OPSD mechanics on a tiny model: replay reproduces the rollout, gradients land where expected."""

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

import gist_opsd
from gist_qwen2 import GistQwen2ForCausalLM
from gist_utils import Global_data
from src.data.gist_stream import insert_gists_with_map

VOCAB = 1000                    # base vocabulary; summary ids sit above it
GIST_IDS = [1000, 1001]


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setenv("ATTN_IMPL", "ref")
    keys = ("gist_token_id", "chunk_size", "sink_size", "num_previous_chunks", "pad_token_id",
            "train_attn", "top_k", "last_pos", "opsd_replay")
    saved = {k: getattr(Global_data, k) for k in keys}
    Global_data.gist_token_id, Global_data.chunk_size = GIST_IDS, [4, 4]
    Global_data.sink_size, Global_data.num_previous_chunks = 3, 1
    Global_data.pad_token_id, Global_data.train_attn = 0, "dense"
    cfg = Qwen2Config(vocab_size=1100, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = GistQwen2ForCausalLM(cfg).float().eval()
    model.selective = True
    plain = Qwen2ForCausalLM(cfg).float().eval()
    plain.load_state_dict(model.state_dict())
    g = torch.Generator().manual_seed(1)
    raw = torch.randint(1, VOCAB, (240,), generator=g).tolist()
    ctx = torch.tensor([insert_gists_with_map(raw, GIST_IDS, 4, 4, sink=3)[0]])
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
    is_sum = torch.isin(ctx[0], torch.tensor(GIST_IDS))
    seen = steps[0][1][:, :ctx.shape[1]] & is_sum
    assert 0 < int(seen.any(0).sum()) < int(is_sum.sum())


def test_gradients(setup):
    model, plain, ctx = setup
    y, _, steps = gist_opsd.rollout(model, ctx, 6, VOCAB, torch.Generator().manual_seed(0))
    lp_t = gist_opsd.teacher_logprobs(plain, ctx, y, VOCAB)
    assert not lp_t.requires_grad

    lp_s, st = gist_opsd.replay(model, ctx, y, steps, VOCAB, probe=True)
    gist_opsd.reverse_kl(lp_s, lp_t).mean().backward()
    is_sum = torch.isin(torch.cat([ctx[0], y[:-1]]), torch.tensor(GIST_IDS))
    hpg = model.config.num_attention_heads // model.config.num_key_value_heads
    checked = 0
    for layer, key in st["probe"].items():
        vis = st["masks"][layer][0]                                   # [H, R, K]
        seen = vis.view(-1, hpg, *vis.shape[1:]).any(1).any(1)        # [num_kv, K]
        g = key.grad[0].abs().sum(-1)                                 # [num_kv, K]
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
        assert g[~seen & is_sum].gt(0).float().mean() > 0.9, f"layer {layer}: routing KL misses unselected summaries"


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
