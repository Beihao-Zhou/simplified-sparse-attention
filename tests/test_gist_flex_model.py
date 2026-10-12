"""A tiny GistQwen2 must give the same logits, loss and gradients with flex and dense."""

import copy
import random

import pytest
import torch
from transformers import Qwen2Config

import gist_flex_attention
from gist_qwen2 import GistQwen2ForCausalLM
from gist_utils import Global_data
from src.data.gist import add_raw_window, make_hierarchical_gist_mask_greedy
from tests.test_gist_flex_mask import GIST_IDS, _structured


@pytest.fixture
def gist_config():
    saved = {k: getattr(Global_data, k) for k in
             ("gist_token_id", "chunk_size", "sink_size", "num_previous_chunks",
              "pad_token_id", "train_attn", "raw_window")}
    yield
    for k, v in saved.items():
        setattr(Global_data, k, v)


@pytest.mark.parametrize("levels,prev,window", [(1, 1, 0), (2, 1, 0), (3, 0, 0), (2, 1, 48)])
def test_flex_training_matches_dense(gist_config, levels, prev, window):
    chunk_size, sink = 4, 3
    Global_data.gist_token_id = GIST_IDS[:levels]
    Global_data.chunk_size = [chunk_size, 4]
    Global_data.sink_size = sink
    Global_data.num_previous_chunks = prev
    Global_data.pad_token_id = 0
    Global_data.raw_window = window

    cfg = Qwen2Config(
        vocab_size=1100, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    dense_model = GistQwen2ForCausalLM(cfg).float().train()
    flex_model = copy.deepcopy(dense_model)
    flex_model.config._attn_implementation = gist_flex_attention.GIST_FLEX

    # Equal-length batch: padded query rows are fully masked in the dense path and
    # would leak NaNs into its gradients; padding is covered by the mask tests.
    rng = random.Random(levels)
    seqs = [_structured(200, levels, chunk_size, 4, jitter=2, rng=rng) for _ in range(2)]
    T = min(len(s) for s in seqs)
    input_ids = torch.tensor([s[:T] for s in seqs])
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    Global_data.train_attn = "dense"
    mask_gist = make_hierarchical_gist_mask_greedy(
        input_ids, Global_data.gist_token_id, chunk_size,
        attention_sink_size=sink, num_previous_chunks=prev, pad_token=0)
    mask_gist = add_raw_window(mask_gist, input_ids, Global_data.gist_token_id, window, pad_token=0)
    out_d = dense_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                        attention_mask_gist=mask_gist)
    out_d.loss.backward()

    Global_data.train_attn = "flex"
    out_f = flex_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                       attention_mask_gist=None)
    out_f.loss.backward()

    torch.testing.assert_close(out_f.logits, out_d.logits, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out_f.loss, out_d.loss, rtol=1e-5, atol=1e-5)
    for (name, pf), (_, pd) in zip(flex_model.named_parameters(), dense_model.named_parameters()):
        assert pf.grad is not None and pd.grad is not None, name
        torch.testing.assert_close(pf.grad, pd.grad, rtol=1e-3, atol=1e-5, msg=name)


def test_flex_mask_requires_flex_attention(gist_config):
    """train_attn='flex' with another attention function must fail, not train causally."""
    Global_data.gist_token_id = GIST_IDS[:1]
    Global_data.chunk_size = [4]
    Global_data.sink_size = 0
    Global_data.pad_token_id = 0
    Global_data.train_attn = "flex"
    cfg = Qwen2Config(vocab_size=1100, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1)
    cfg._attn_implementation = "sdpa"
    model = GistQwen2ForCausalLM(cfg)
    ids = torch.tensor([_structured(40, 1, 4, 4, jitter=0, rng=random.Random(0))])
    with pytest.raises(ValueError, match="attn_implementation"):
        model(input_ids=ids, attention_mask=torch.ones_like(ids), attention_mask_gist=None)


@pytest.mark.parametrize("meta", [0, 1])
def test_collators_flex_and_dense(gist_config, meta):
    """Dense: mask as before plus the window. Flex: no mask, T rounded up to 128."""
    from src.data.gist import add_raw_window, make_gist_mask
    from src.data.gist_input_preprocessor import (
        FLEX_LENGTH_MULTIPLE,
        custom_collate_chunkaug,
        meta_custom_collate_chunkaug,
    )

    levels = 2 if meta else 1
    Global_data.gist_token_id = GIST_IDS[:levels]
    Global_data.chunk_size = [4, 4]
    Global_data.sink_size = 2
    Global_data.num_previous_chunks = 1
    Global_data.pad_token_id = 0
    Global_data.raw_window = 16
    collate = meta_custom_collate_chunkaug if meta else custom_collate_chunkaug
    rng = random.Random(meta)
    batch = []
    for n in (50, 90):
        ids = _structured(n, levels, 4, 4, jitter=1, rng=rng)
        batch.append({"input_ids": ids, "labels": ids, "attention_mask": [1] * len(ids)})

    Global_data.train_attn = "dense"
    out = collate(batch)
    ids = out["input_ids"]
    if meta:
        ref = make_hierarchical_gist_mask_greedy(ids, GIST_IDS[:2], 4, 2, 1, pad_token=0)
    else:
        ref = make_gist_mask(ids, GIST_IDS[:1], 4, 2, 1, pad_token=0, meta_gist=0)
    assert torch.equal(out["attention_mask_gist"], add_raw_window(ref, ids, GIST_IDS[:levels], 16, 0))

    Global_data.train_attn = "flex"
    out_f = collate(batch)
    T = out_f["input_ids"].shape[1]
    assert out_f["attention_mask_gist"] is None and T % FLEX_LENGTH_MULTIPLE == 0
    assert torch.equal(out_f["input_ids"][:, T - ids.shape[1]:], ids)
    assert (out_f["labels"][:, : T - ids.shape[1]] == -100).all()
