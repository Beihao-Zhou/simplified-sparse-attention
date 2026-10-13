"""Chunked LM loss must equal HF's ForCausalLMLoss, in value and gradients."""

import copy
import random

import pytest
import torch
from transformers import Qwen2Config
from transformers.loss.loss_utils import ForCausalLMLoss

from gist_qwen2 import GistQwen2ForCausalLM, chunked_causal_lm_loss
from gist_utils import Global_data
from src.data.gist import make_hierarchical_gist_mask_greedy
from tests.test_gist_flex_mask import GIST_IDS, _structured


@pytest.mark.parametrize("chunk", [1, 7, 40, 64])
@pytest.mark.parametrize("num_items", [None, 37])
def test_matches_hf_loss(chunk, num_items):
    torch.manual_seed(chunk)
    B, T, d, V = 2, 40, 16, 50
    head = torch.nn.Linear(d, V, bias=False)
    h = torch.randn(B, T, d, requires_grad=True)
    labels = torch.randint(0, V, (B, T))
    labels[torch.rand(B, T) < 0.3] = -100
    n = None if num_items is None else torch.tensor(num_items)

    ref = ForCausalLMLoss(head(h), labels, V, num_items_in_batch=n)
    got = chunked_causal_lm_loss(h, head, labels, chunk, n)
    torch.testing.assert_close(got, ref)
    for a, b in zip(torch.autograd.grad(got, (h, head.weight)),
                    torch.autograd.grad(ref, (h, head.weight))):
        torch.testing.assert_close(a, b)


def test_model_training_step():
    """GistQwen2ForCausalLM: same loss and parameter gradients with and without chunking."""
    saved = {k: getattr(Global_data, k) for k in
             ("gist_token_id", "chunk_size", "sink_size", "num_previous_chunks", "pad_token_id",
              "loss_chunk_size")}
    try:
        Global_data.gist_token_id = GIST_IDS[:2]
        Global_data.chunk_size = [4, 4]
        Global_data.sink_size = 3
        Global_data.num_previous_chunks = 1
        Global_data.pad_token_id = 0
        cfg = Qwen2Config(vocab_size=1100, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
        cfg._attn_implementation = "sdpa"
        torch.manual_seed(0)
        full = GistQwen2ForCausalLM(cfg).train()
        chunked = copy.deepcopy(full)

        seqs = [_structured(120, 2, 4, 4, jitter=1, rng=random.Random(i)) for i in range(2)]
        T = min(map(len, seqs))
        ids = torch.tensor([s[:T] for s in seqs])
        labels = ids.clone()
        labels[:, :20] = -100
        mask = make_hierarchical_gist_mask_greedy(ids, GIST_IDS[:2], 4, attention_sink_size=3,
                                                  num_previous_chunks=1, pad_token=0)
        outs = []
        for model, chunk in ((full, 0), (chunked, 33)):
            Global_data.loss_chunk_size = chunk
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
                        attention_mask_gist=mask)
            out.loss.backward()
            outs.append(out)
        assert outs[1].logits is None
        torch.testing.assert_close(outs[1].loss, outs[0].loss)
        for (name, a), (_, b) in zip(chunked.named_parameters(), full.named_parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-6, msg=name)
    finally:
        for k, v in saved.items():
            setattr(Global_data, k, v)
