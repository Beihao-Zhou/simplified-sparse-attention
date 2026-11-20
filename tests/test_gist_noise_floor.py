"""Fully unfolded and blind to summaries, a gisted sequence must reproduce the original model.

OPSD's noise floor rests on this: if raw queries see every raw key and no summary key, the
student's logits at raw positions equal the original model's on the un-gisted sequence.
It holds only because RoPE positions skip summary tokens (raw tokens keep their original
positions), so this test also guards that rule.
"""

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from gist_qwen2 import GistQwen2ForCausalLM
from gist_utils import Global_data
from src.data.gist_stream import insert_gists_with_map

GIST_IDS = [1000, 1001]


def _models():
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
    gist = GistQwen2ForCausalLM(cfg).float().eval()
    plain = Qwen2ForCausalLM(cfg).float().eval()
    missing, unexpected = plain.load_state_dict(gist.state_dict(), strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return gist, plain


def _gisted(n_raw, seed, gist_ids=GIST_IDS):
    g = torch.Generator().manual_seed(seed)
    raw = torch.randint(1, 900, (n_raw,), generator=g).tolist()
    toks, src = insert_gists_with_map(raw, gist_ids, chunk_size=4, group=4)
    return torch.tensor([raw]), torch.tensor([toks]), torch.tensor(src)


def _student_logits(model, toks, src, summaries_visible, gist_ids=GIST_IDS):
    T = toks.shape[1]
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    raw_key = (src >= 0)[None, :].expand(T, T)
    allowed = causal & (raw_key | torch.eye(T, dtype=torch.bool) | summaries_visible)
    saved = (Global_data.gist_token_id, Global_data.chunk_size, Global_data.train_attn)
    Global_data.gist_token_id, Global_data.chunk_size, Global_data.train_attn = (
        gist_ids,
        [4] * len(gist_ids),
        "dense",
    )
    try:
        with torch.no_grad():
            out = model(
                input_ids=toks,
                attention_mask=torch.ones_like(toks),
                attention_mask_gist=allowed[None, None],
            )
    finally:
        Global_data.gist_token_id, Global_data.chunk_size, Global_data.train_attn = saved
    return out.logits[0, src >= 0]


def test_full_unfold_without_summaries_matches_original():
    gist, plain = _models()
    raw, toks, src = _gisted(n_raw=150, seed=1)
    assert (src < 0).sum() >= 40  # both levels present
    with torch.no_grad():
        ref = plain(input_ids=raw).logits[0]
    torch.testing.assert_close(_student_logits(gist, toks, src, False), ref, rtol=1e-4, atol=1e-4)


def test_visible_summaries_change_the_output():
    """The check above is sensitive: letting raw queries see summaries moves the logits."""
    gist, plain = _models()
    raw, toks, src = _gisted(n_raw=150, seed=2)
    with torch.no_grad():
        ref = plain(input_ids=raw).logits[0]
    diff = (_student_logits(gist, toks, src, True) - ref).abs().max()
    assert diff > 1e-2, diff


def test_full_unfold_matches_original_at_four_levels():
    gist, plain = _models()
    ids4 = GIST_IDS + [1002, 1003]
    raw, toks, src = _gisted(n_raw=600, seed=3, gist_ids=ids4)
    assert (toks == 1003).sum() >= 2
    with torch.no_grad():
        ref = plain(input_ids=raw).logits[0]
    torch.testing.assert_close(
        _student_logits(gist, toks, src, False, ids4), ref, rtol=1e-4, atol=1e-4
    )
