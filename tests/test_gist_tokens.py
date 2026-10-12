"""Summary tokens are found by name, extend to N levels, and survive resume.

Needs the Qwen2.5 tokenizer in the local HF cache; skipped otherwise.
"""

import tempfile

import pytest
import torch
from transformers import AutoTokenizer, Qwen2Config

from gist_qwen2 import GistQwen2ForCausalLM
from gist_utils import add_gist_tokens, find_gist_token_ids, gist_token_names

TOKENIZER = "Qwen/Qwen2.5-7B-Instruct-1M"


@pytest.fixture
def tok():
    try:
        return AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    except OSError:
        pytest.skip(f"{TOKENIZER} tokenizer not in the local HF cache")


def _tiny(rows, tie=False):
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=rows, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, tie_word_embeddings=tie)
    return GistQwen2ForCausalLM(cfg)


def _reload(tok):
    with tempfile.TemporaryDirectory() as d:
        tok.save_pretrained(d)
        return AutoTokenizer.from_pretrained(d)


def test_names():
    assert gist_token_names(2) == ["<gist>", "<Gist>"]
    assert gist_token_names(5) == ["<gist>", "<Gist>", "<MGist>", "<G3>", "<G4>"]


@pytest.mark.parametrize("tie", [False, True])
def test_fresh_six_levels_and_resume(tok, tie):
    n_base = len(tok)
    model = _tiny(152064, tie)
    mean = model.get_input_embeddings().weight[:n_base].mean(0).clone()
    ids = add_gist_tokens(tok, model, 6)
    assert ids == list(range(n_base, n_base + 6))
    w = model.get_input_embeddings().weight
    assert w.shape[0] == len(tok) == n_base + 6
    torch.testing.assert_close(w[ids], mean.expand(6, -1))

    tok2 = _reload(tok)
    with torch.no_grad():
        w[ids] += 1.0
    before = w.detach().clone()
    assert add_gist_tokens(tok2, model, 6) == ids
    assert torch.equal(model.get_input_embeddings().weight, before), "resume must not re-initialize"


def test_deepen_three_token_checkpoint(tok):
    """An H-SSA checkpoint (3 summary tokens) trained further with 6 levels."""
    model = _tiny(152064)
    ids3 = add_gist_tokens(tok, model, 2)  # adds levels 0-2, as the released checkpoints have
    assert len(ids3) == 2 and find_gist_token_ids(tok, 3) is not None
    with torch.no_grad():
        model.get_input_embeddings().weight[find_gist_token_ids(tok, 3)] = 7.0
    trained = model.get_input_embeddings().weight.detach().clone()
    n3 = len(tok)

    ids6 = add_gist_tokens(tok, model, 6)
    w = model.get_input_embeddings().weight
    assert ids6[:3] == find_gist_token_ids(tok, 3) and ids6[3:] == [n3, n3 + 1, n3 + 2]
    assert torch.equal(w[:n3], trained[:n3]), "existing rows unchanged"
    keep = torch.ones(n3, dtype=torch.bool)
    keep[ids6[:3]] = False
    torch.testing.assert_close(w[ids6[3:]], trained[:n3][keep].mean(0).expand(3, -1))
