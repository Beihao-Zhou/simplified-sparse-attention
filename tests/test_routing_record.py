"""The decode-time routing recorder writes what deep_routing.py expects."""

from types import SimpleNamespace

import numpy as np
import torch

from experiments.routing_record import answer_chunks, length_bucket, record_decode_call

G0_ID, G1_ID = 900, 901


class _Tok:
    """Whitespace tokenizer over integer words: '12 7' -> [12, 7]."""

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[int(w) for w in text.split()])


def _gisted():
    # chunks of 2 raw tokens; a level-1 gist after every 2 level-0 gists; 2-token tail
    raw = list(range(1, 11))
    seq = []
    for i, t in enumerate(raw[:8]):
        seq.append(t)
        if i % 2 == 1:
            seq.append(G0_ID)
            if (i // 2) % 2 == 1:
                seq.append(G1_ID)
    return seq + raw[8:]


def _cache(seq):
    ids = torch.tensor([seq])
    is0, is1 = ids == G0_ID, ids == G1_ID
    c0 = torch.where(is0, is0.cumsum(1) - 1, is0.cumsum(1))
    c1 = torch.where(is1, is1.cumsum(1) - 1, is1.cumsum(1))
    pos0 = torch.nonzero(is0[0]).flatten()[None]
    return SimpleNamespace(
        num_levels=2, Gmax=[int(is0.sum()), int(is1.sum())], is_gist_full=is0 | is1,
        chunk_ids=[c0, c1], parent_chunk_ids=[c1.gather(1, pos0), None], pos0=pos0,
        pos1=torch.nonzero(is1[0]).flatten()[None],
    )


def test_answer_chunks_and_bucket():
    seq = _gisted()
    assert answer_chunks(seq, [G0_ID, G1_ID], ["5 6"], _Tok()) == [2]
    assert answer_chunks(seq, [G0_ID, G1_ID], ["4 5"], _Tok()) == [1, 2]
    assert answer_chunks(seq, [G0_ID, G1_ID], ["9"], _Tok()) == [4]  # tail, not yet compressed
    assert [length_bucket(n) for n in (100, 9000, 20000)] == ["<8K", "8-16K", ">16K"]


def test_record_decode_call(tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTING_RECORD_LAYERS", "1")
    monkeypatch.setenv("ROUTING_RECORD_STEPS", "1")
    seq = _gisted()
    cache = _cache(seq)
    H, num_kv, d, k_len = 4, 2, 8, len(seq)
    torch.manual_seed(0)
    query = torch.randn(1, H, 1, d)
    key = torch.randn(1, num_kv, k_len, d)
    K_rep = key.repeat_interleave(H // num_kv, dim=1)                 # [1, H, k_len, d]
    cache.K_gist_bf16_T = [K_rep[:, :, cache.pos0[0]].transpose(2, 3),
                           K_rep[:, :, cache.pos1[0]].transpose(2, 3)]
    meta = {"task": "t", "sample": 0, "calls": {}, "bucket": "<8K", "answer0": [2, 4]}
    module = SimpleNamespace(layer_idx=1)

    for _ in range(2):  # second call is over the step limit and writes nothing
        record_decode_call(str(tmp_path), module, query, key, cache, [3, 3], d ** -0.5, meta)
    files = list(tmp_path.glob("*.npz"))
    assert len(files) == 1
    r = np.load(files[0])
    assert r["s0"].shape == (H, 4) and r["s1"].shape == (H, 2)
    assert r["parent0"].tolist() == [0, 0, 1, 1]
    assert r["answer0"].tolist() == [2]                   # chunk 4 is the uncompressed tail
    assert int(r["k0"]) == 1                              # min(top_k, max(1, G0 // 4))

    # oracle mass: softmax over raw keys only, summed over the 4 compressed chunks
    q = query[0, :, 0]
    logits = torch.einsum("hd,hkd->hk", q, K_rep[0]) * d ** -0.5
    raw = ~cache.is_gist_full[0]
    p = logits.masked_fill(~raw, float("-inf")).softmax(-1)
    tail = raw & (cache.chunk_ids[0][0] == 4)
    np.testing.assert_allclose(r["m0"].sum(1), 1 - p[:, tail].sum(1).numpy(), rtol=1e-5)
    np.testing.assert_allclose(r["s0"][:, 0], (q * K_rep[0, :, cache.pos0[0, 0]]).sum(-1).numpy(),
                               rtol=1e-5)
