"""Stage 1: gist-only LoRA leaves raw tokens untouched; per-layer MSE trains only adapters."""

import random

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

import gist_flex_attention
from gist_lora import apply_lora
from gist_qwen2 import GistQwen2ForCausalLM
from gist_utils import Global_data
from src.data.gist_stream import insert_gists_with_map

G = [1000, 1001]


@pytest.fixture
def cfg():
    keys = ("gist_token_id", "chunk_size", "sink_size", "num_previous_chunks", "pad_token_id",
            "train_attn", "raw_window", "stage1", "lora_row_mask")
    saved = {k: getattr(Global_data, k) for k in keys}
    Global_data.gist_token_id, Global_data.chunk_size = G, [4, 4]
    Global_data.sink_size, Global_data.num_previous_chunks = 3, 1
    Global_data.pad_token_id, Global_data.raw_window = 0, 0
    c = Qwen2Config(vocab_size=1100, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
    yield c
    for k, v in saved.items():
        setattr(Global_data, k, v)


def _ids(n_raw=150, seed=0):
    rng = random.Random(seed)
    return torch.tensor([insert_gists_with_map([rng.randrange(1, 900) for _ in range(n_raw)], G, 4, 4, sink=3)[0]])


def _randomize_lora(model):
    torch.manual_seed(3)
    for n, p in model.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.5)


def test_gist_lora_leaves_raw_rows_unchanged(cfg):
    """Raw queries blind to summaries: option D changes nothing, option B does."""
    cfg._attn_implementation = "sdpa"
    Global_data.train_attn = "dense"
    torch.manual_seed(0)
    base = GistQwen2ForCausalLM(cfg).float().eval()
    ids = _ids()
    T = ids.shape[1]
    raw = ~torch.isin(ids[0], torch.tensor(G))
    allowed = torch.ones(T, T, dtype=torch.bool).tril() & (raw[None, :] | torch.eye(T, dtype=torch.bool))
    kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids), attention_mask_gist=allowed[None, None])
    with torch.no_grad():
        ref = base(**kw).logits[0, raw]
    for mode, same in (("gist", True), ("all", False)):
        m = GistQwen2ForCausalLM(cfg).float().eval()
        m.load_state_dict(base.state_dict())
        apply_lora(m, mode, r=4)
        _randomize_lora(m)
        with torch.no_grad():
            out = m(**kw).logits[0, raw]
        assert torch.allclose(out, ref, atol=1e-5) == same, mode


@pytest.mark.parametrize("mode", ["gist", "all"])
def test_stage1_step_trains_only_adapters(cfg, mode):
    cfg._attn_implementation = gist_flex_attention.GIST_FLEX
    Global_data.train_attn = "flex"
    Global_data.stage1 = {"pairs": [("g1", "raw"), ("hier", "g1")], "down": "hier", "lambda_ntp": 0.0}
    torch.manual_seed(0)
    model = GistQwen2ForCausalLM(cfg).float().train()
    groups = apply_lora(model, mode, r=4)
    ids = _ids()
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone())
    assert set(Global_data.stage1["last"]) == {"g1|raw", "hier|g1"}
    assert all(v > 0 for v in Global_data.stage1["last"].values())
    out.loss.backward()
    trainable = {id(p) for g in groups.values() for p in g}
    for n, p in model.named_parameters():
        if id(p) in trainable:
            continue
        assert p.grad is None, f"frozen {n} got a gradient"
    # B starts at 0, so only B gets a gradient on the first step
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in groups["lora_kv"][1::2])
    emb_grad = groups["embed"][0].grad
    assert emb_grad[G].abs().sum() > 0 and emb_grad[:G[0]].abs().sum() == 0
    if mode == "gist":  # last layer keeps K/V adapters only
        last = model.model.layers[-1].self_attn
        assert not hasattr(last.q_proj, "lora_A") and hasattr(last.k_proj, "lora_A")


def test_no_summaries_no_loss(cfg):
    """Without summary tokens every mask is causal, so each relMSE is 0."""
    cfg._attn_implementation = gist_flex_attention.GIST_FLEX
    Global_data.train_attn = "flex"
    Global_data.stage1 = {"pairs": [("g1", "raw"), ("hier", "g1")], "down": "hier", "lambda_ntp": 0.0}
    torch.manual_seed(0)
    model = GistQwen2ForCausalLM(cfg).float().train()
    apply_lora(model, "gist", r=4)
    ids = torch.randint(1, 900, (1, 64))
    model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone())
    assert all(v < 1e-10 for v in Global_data.stage1["last"].values())
