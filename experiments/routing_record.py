"""Record two-level gist routing inputs at decode time for experiments/deep_routing.py.

Enabled by ROUTING_RECORD_DIR. The LongBench harness sets Global_data.routing_meta per
sample (task, length bucket, answer chunks); the sparse decode path then calls
record_decode_call once per (step, layer) and writes one .npz per call:

    s0 [H, N0]   q . k of every level-0 gist, per query head (unscaled)
    s1 [H, N1]   same for level-1 gists
    parent0 [N0] level-1 parent of each level-0 chunk, -1 if not yet summarized
    m0 [H, N0]   oracle mass: the query's softmax over raw keys, summed per chunk
    k0 ()        level-0 budget the selector used
    answer0 [A]  level-0 chunks holding the gold answer
    task, bucket strings

ROUTING_RECORD_LAYERS (default "2,8,16,24") and ROUTING_RECORD_STEPS (default 32)
bound what is written. Recording reads the decode cache only; outputs are unchanged.
"""

import os
from typing import List, Sequence

import numpy as np
import torch


def length_bucket(n_raw: int) -> str:
    return "<8K" if n_raw < 8192 else ("8-16K" if n_raw < 16384 else ">16K")


def _find(seq: Sequence[int], sub: Sequence[int]) -> List[int]:
    n = len(sub)
    return [i for i in range(len(seq) - n + 1) if n and list(seq[i:i + n]) == list(sub)]


def answer_chunks(gisted: Sequence[int], gist_ids: Sequence[int], answers, tokenizer) -> List[int]:
    """Level-0 chunk ids (as the decode cache numbers them) of raw tokens matching an answer.

    A raw token's level-0 chunk id is the number of level-0 gists before it.
    """
    summary = set(gist_ids)
    raw_pos = [i for i, t in enumerate(gisted) if t not in summary]
    raw = [gisted[i] for i in raw_pos]
    g0 = np.cumsum([t == gist_ids[0] for t in gisted])
    chunks = set()
    for ans in answers:
        for text in {ans.strip(), " " + ans.strip()}:
            sub = tokenizer(text, add_special_tokens=False).input_ids
            for start in _find(raw, sub):
                for j in range(start, start + len(sub)):
                    p = raw_pos[j]
                    chunks.add(int(g0[p - 1]) if p > 0 else 0)
    return sorted(chunks)


@torch.no_grad()
def record_decode_call(out_dir, module, query, key, cache, top_k, scaling, meta):
    layers = {int(x) for x in os.environ.get("ROUTING_RECORD_LAYERS", "2,8,16,24").split(",")}
    max_steps = int(os.environ.get("ROUTING_RECORD_STEPS", "32"))
    layer = module.layer_idx
    if meta is None or layer not in layers or cache.num_levels != 2:
        return
    step = meta["calls"].get(layer, 0)
    if step >= max_steps:
        return
    meta["calls"][layer] = step + 1
    G0, G1 = cache.Gmax
    if G0 == 0 or G1 == 0:
        return

    q = query[0, :, 0, :].float()                                   # [H, d]
    H, d = q.shape
    s0 = torch.einsum("hd,hdg->hg", q, cache.K_gist_bf16_T[0][0].float())
    s1 = torch.einsum("hd,hdg->hg", q, cache.K_gist_bf16_T[1][0].float())
    parent0 = cache.parent_chunk_ids[0][0].clone()
    parent0[parent0 >= G1] = -1

    K = key[0].float()                                              # [num_kv, k_len, d]
    num_kv = K.shape[0]
    logits = torch.einsum("ghd,gkd->ghk", q.view(num_kv, H // num_kv, d), K).reshape(H, -1)
    raw = ~cache.is_gist_full[0]
    p = (logits * scaling).masked_fill(~raw, float("-inf")).softmax(-1)
    chunk = cache.chunk_ids[0][0]
    sel = raw & (chunk < G0)
    m0 = torch.zeros(H, G0, device=q.device).index_add_(1, chunk[sel], p[:, sel])

    k0 = min(top_k[0], max(1, G0 // 4))
    os.makedirs(out_dir, exist_ok=True)
    name = f"{meta['task']}_{meta['sample']:03d}_L{layer:02d}_t{step:03d}.npz"
    np.savez_compressed(
        os.path.join(out_dir, name),
        s0=s0.cpu().numpy(), s1=s1.cpu().numpy(), parent0=parent0.cpu().numpy(),
        m0=m0.cpu().numpy(), k0=np.int64(k0),
        answer0=np.asarray([c for c in meta["answer0"] if c < G0], dtype=np.int64),
        task=meta["task"], bucket=meta["bucket"],
    )
