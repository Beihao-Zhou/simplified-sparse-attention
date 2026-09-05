"""Sparse-attention forward path + dispatcher.

`sparse_attention_forward` is a drop-in replacement for the reference
attention path in `gist_qwen2.py`. It dispatches between prefill and decode
internally.

`attn_dispatcher` is the single hook called from
`GistQwen2Attention.select_forward`: training routes to the differentiable
eager reference (the sparse-decode CUDA kernel is forward-only), inference to
the sparse path. `ATTN_IMPL=ref` forces eager for debugging.

Decode caches the static-per-prefix parts of the per-level metadata on the
layer (`module`) across decode steps: the prefix gist positions don't change
as new (non-gist) tokens are appended, so on a cache hit (k_len == cache.k_len
+ 1) the full-length helper tensors are extended by one constant column; on a
miss (new sample / new prefill / gist emitted) they are rebuilt from scratch.
"""

from __future__ import annotations

import os
from typing import List, Optional

import copy

import torch
import torch.nn.functional as F

# -- CUDA extension (lazy load on first decode call) -------------------------
_CUDA_EXT = None
_CUDA_EXT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


_SPARSE_DECODE_EXT = None


class _DecodeKernelStepCache:
    """Per-decode-step cache of the [k_len]-derived kernel inputs, shared across
    all layers in a step (they depend only on input_ids/masks, not the layer's
    Q/K). Keyed by id(input_ids). Eliminates ~26x redundant uint8/int32
    conversions per decode step."""
    input_ids_ref = None   # hold the tensor (identity check; prevents id() reuse)
    k_len: int = -1
    c0_i32 = None
    c1_i32 = None
    is_gist_u8 = None
    is_l0_u8 = None
    is_l1_u8 = None
    comp_u8 = None
    am_u8 = None
    causal_bias = None

    @classmethod
    def get(cls, input_ids, cache, attention_mask, causal_mask, k_len):
        # Identity (`is`) on the held tensor — robust vs id() reuse across samples.
        if cls.input_ids_ref is input_ids and cls.k_len == k_len:
            return cls
        cls.input_ids_ref = input_ids
        cls.k_len = k_len
        cls.c0_i32 = cache.chunk_ids[0].to(torch.int32).contiguous()
        cls.is_gist_u8 = cache.is_gist_full.to(torch.uint8).contiguous()
        cls.is_l0_u8 = cache.is_gist_per_level[0].to(torch.uint8).contiguous()
        if cache.num_levels >= 2:
            cls.c1_i32 = cache.chunk_ids[1].to(torch.int32).contiguous()
            cls.is_l1_u8 = cache.is_gist_per_level[1].to(torch.uint8).contiguous()
        else:
            # Single-level: no meta-gist. No-op level-1 for the kernel (is_l1=0 masks
            # the level-1 keep term; c1=0 keeps sel1[g, c1[k]] in-bounds at G1p1=1).
            _dev = cache.is_gist_full.device
            cls.c1_i32 = torch.zeros((cache.B, k_len), dtype=torch.int32, device=_dev)
            cls.is_l1_u8 = torch.zeros((cache.B, k_len), dtype=torch.uint8, device=_dev)
        # `compressed` is identical across all layers in a step (depends only on
        # last_gist/start_idx/k_len) — compute it ONCE here on rebuild instead of
        # per-layer in the hot path (saves ~27 redundant arange+compare launches/step).
        k_idx = torch.arange(k_len, device=cache.is_gist_full.device)
        after_last_key = k_idx[None, :] > cache.last_gist[:, None]
        compressed_row = (~after_last_key) & (k_idx[None, :] >= cache.start_idx[:, None])
        cls.comp_u8 = compressed_row.to(torch.uint8).contiguous()
        cls.am_u8 = (attention_mask.expand(cache.B, -1, -1, -1)[:, 0, 0, :] > -1e-4).to(torch.uint8).contiguous()
        cls.causal_bias = causal_mask.expand(cache.B, -1, -1, -1)[:, 0, 0, :].float().contiguous()
        return cls


def _get_sparse_decode_ext():
    global _SPARSE_DECODE_EXT
    if _SPARSE_DECODE_EXT is None:
        from torch.utils.cpp_extension import load
        _SPARSE_DECODE_EXT = load(
            name="gsa_sparse_decode",
            sources=[os.path.join(_CUDA_EXT_PATH, "sparse_decode_attn.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _SPARSE_DECODE_EXT


from gist_qwen2 import (
    eager_attention_forward_prefill_chunk as _ref_prefill_chunk,
    eager_attention_forward_decoding as _ref_decoding,
    repeat_kv,
    top_k_from_logits,
    top_p_from_logits,
)
from gist_utils import Global_data


# ===========================================================================
# decode cache
# ===========================================================================

class _GistDecodeCache:
    __slots__ = (
        "k_len", "B", "H", "d", "num_levels",
        "gist_token_tensor",
        "is_gist_per_level",   # list of [B, k_len] bool
        "gist_cumsum",         # list of [B, k_len] long
        "chunk_ids",           # list of [B, k_len] long
        "is_gist_full",        # [B, k_len] bool
        "last_gist",           # [B] long
        "start_idx",           # [B] long
        # Static-per-prefix per-level caches
        "Gmax",                # list of int
        "gist_pos_safe",       # list of [B, Gmax] long
        "gist_valid",          # list of [B, 1, 1, Gmax] bool
        "K_gist_bf16_T",       # list of [B, H, d, Gmax] bf16 (for fp32/bf16 scoring)
        "K_gist_kv_bf16_T",    # list of [B, num_kv, d, Gmax] bf16 (per-group scoring)
        "K_meanpool_kv_bf16_T",# list of [B, num_kv, d, Gmax] bf16 (ABLATION: mean-pooled chunk keys; DECODE_PIVOT=meanpool)
        "parent_chunk_ids",    # list of [B, Gmax_lower] long  (None for highest)
        "sel_pad",             # dict of padded/bucketed selection inputs (DECODE_GMAX_BUCKET), or None
    )


def _build_full_cache(
    input_ids, key, gist_ids, k_len, B, H, d, device, num_levels, n_rep=1,
):
    # `key` is the UN-repeated [B, num_kv, k_len, d]; K_gist is gathered from it
    # (only Gmax positions) and repeated to H — avoids materializing the full
    # repeat_kv of the whole K (hundreds of MB at large k_len).
    num_kv = key.shape[1]
    cache = _GistDecodeCache()
    cache.B, cache.H, cache.d = B, H, d
    cache.k_len = k_len
    cache.num_levels = num_levels

    cache.gist_token_tensor = torch.tensor(gist_ids, device=device)
    cache.is_gist_full = torch.isin(input_ids, cache.gist_token_tensor)

    cache.is_gist_per_level = []
    cache.gist_cumsum = []
    cache.chunk_ids = []
    cache.Gmax = []
    cache.gist_pos_safe = []
    cache.gist_valid = []
    cache.K_gist_bf16_T = []
    cache.K_gist_kv_bf16_T = []
    cache.K_meanpool_kv_bf16_T = []   # ABLATION (DECODE_PIVOT=meanpool); else filled with None
    cache.parent_chunk_ids = [None] * num_levels
    cache.sel_pad = None   # lazily built by _select_per_group when DECODE_GMAX_BUCKET=1
    _abl_meanpool = os.environ.get("DECODE_PIVOT", "gist") == "meanpool"

    for li in range(num_levels):
        is_gist_li = (input_ids == gist_ids[li])
        cache.is_gist_per_level.append(is_gist_li)
        cumsum_li = is_gist_li.cumsum(dim=1)
        cache.gist_cumsum.append(cumsum_li)
        cache.chunk_ids.append(
            torch.where(is_gist_li, cumsum_li - 1, cumsum_li).clamp_min(0)
        )

        Gmax = int(is_gist_li.sum(dim=1).max().item())
        cache.Gmax.append(Gmax)

        if Gmax == 0:
            cache.gist_pos_safe.append(None)
            cache.gist_valid.append(None)
            cache.K_gist_bf16_T.append(None)
            cache.K_gist_kv_bf16_T.append(None)
            cache.K_meanpool_kv_bf16_T.append(None)
            continue

        gist_pos = torch.zeros((B, Gmax), device=device, dtype=torch.long)
        nz = is_gist_li.nonzero(as_tuple=False)
        rank = cumsum_li[nz[:, 0], nz[:, 1]] - 1
        gist_pos[nz[:, 0], rank] = nz[:, 1]
        gist_pos_safe = gist_pos.clamp(0, k_len - 1)
        cache.gist_pos_safe.append(gist_pos_safe)
        cache.gist_valid.append((gist_pos > 0)[:, None, None, :])

        # Gather Gmax gist keys from the un-repeated K [B,num_kv,Gmax,d], then
        # repeat to [B,H,Gmax,d] (cheap: Gmax << k_len). gist_pos is per-batch
        # (same across heads), so the gather index is [B,num_kv,Gmax,d].
        gist_gather_kv = gist_pos_safe[:, None, :, None].expand(B, num_kv, Gmax, d)
        K_gist_kv = key.gather(dim=2, index=gist_gather_kv)         # [B,num_kv,Gmax,d]
        # Per-group gist-K (num_kv heads, bf16, transposed) for per-group scoring
        # reads num_kv heads not H (÷H/num_kv bytes). Only built when needed.
        if os.environ.get("DECODE_SELECT", "head") == "group":
            cache.K_gist_kv_bf16_T.append(K_gist_kv.transpose(2, 3).contiguous())
            cache.K_gist_bf16_T.append(None)
        else:
            K_gist = repeat_kv(K_gist_kv, n_rep)                    # [B,H,Gmax,d]
            K_gist_T = K_gist.transpose(2, 3).contiguous()         # bf16 [B,H,d,Gmax]
            cache.K_gist_bf16_T.append(K_gist_T)
            cache.K_gist_kv_bf16_T.append(None)

        # ABLATION pivot: mean-pooled chunk keys (mean of each chunk's tokens, by
        # chunk_ids — the same chunk grouping the gist scores). Only built when
        # DECODE_PIVOT=meanpool; otherwise None (zero cost for the default path).
        if _abl_meanpool:
            cid = cache.chunk_ids[li].clamp(0, Gmax - 1)               # [B,k_len]
            kf = key.float()
            summ = torch.zeros(B, num_kv, Gmax, d, device=device)
            summ.scatter_add_(2, cid[:, None, :, None].expand(B, num_kv, k_len, d), kf)
            cnt = torch.zeros(B, Gmax, device=device)
            cnt.scatter_add_(1, cid, torch.ones(B, k_len, device=device))
            meank = summ / cnt.clamp_min(1.0)[:, None, :, None]        # [B,num_kv,Gmax,d]
            cache.K_meanpool_kv_bf16_T.append(meank.transpose(2, 3).to(torch.bfloat16).contiguous())
        else:
            cache.K_meanpool_kv_bf16_T.append(None)

    # Cross-level parent_chunk_ids
    for li in range(num_levels - 2, -1, -1):
        if (
            cache.gist_pos_safe[li] is None
            or cache.chunk_ids[li + 1] is None
        ):
            continue
        cache.parent_chunk_ids[li] = cache.chunk_ids[li + 1].gather(
            dim=1, index=cache.gist_pos_safe[li]
        )

    cache.last_gist = torch.where(
        cache.is_gist_per_level[-1],
        torch.arange(k_len, device=device),
        torch.full((k_len,), -1, device=device, dtype=torch.long),
    ).amax(dim=1)
    cache.start_idx = (input_ids != Global_data.pad_token_id).int().argmax(dim=1)

    return cache


class _SharedExtend1D:
    """Per-step shared 1-D gist state. is_gist_full / is_gist_per_level /
    gist_cumsum / chunk_ids depend only on input_ids, so they are IDENTICAL
    across all layers. Compute the one-step extend ONCE per step (first layer)
    and let the other layers reuse the (read-only) tensors — instead of 28
    redundant torch.cat sequences. Decode is CPU-launch-bound, so removing
    ~27x of these cat launches/step directly cuts tpot."""
    input_ids_ref = None
    k_len = -1
    is_gist_full = None
    is_gist_per_level = None
    gist_cumsum = None
    chunk_ids = None


def _extend_cache_one_step(cache, B, k_len, device, input_ids=None):
    """Extend per-call full-length tensors by one new (non-gist) position.

    parent_chunk_ids / K_gist_bf16_T / gist_pos_safe / gist_valid / Gmax /
    last_gist / start_idx / gist_token_tensor are unchanged.

    The 1-D gist state is shared across layers via _SharedExtend1D (same
    input_ids => same values), so only the first layer each step pays the cats.
    """
    sh = _SharedExtend1D
    if input_ids is not None and sh.input_ids_ref is input_ids and sh.k_len == k_len:
        # Another layer already extended this step's 1-D state — reuse it.
        # Fresh list objects (so a later first-layer in-place extend can't alias),
        # sharing the read-only tensors.
        cache.is_gist_full = sh.is_gist_full
        cache.is_gist_per_level = list(sh.is_gist_per_level)
        cache.gist_cumsum = list(sh.gist_cumsum)
        cache.chunk_ids = list(sh.chunk_ids)
        cache.k_len = k_len
        return

    zero_b1_bool = torch.zeros((B, 1), dtype=torch.bool, device=device)
    cache.is_gist_full = torch.cat([cache.is_gist_full, zero_b1_bool], dim=1)

    for li in range(cache.num_levels):
        cache.is_gist_per_level[li] = torch.cat(
            [cache.is_gist_per_level[li], zero_b1_bool], dim=1,
        )
        last_cumsum = cache.gist_cumsum[li][:, -1:]
        cache.gist_cumsum[li] = torch.cat([cache.gist_cumsum[li], last_cumsum], dim=1)
        # chunk_ids extends by the same value (non-gist new pos => where(False, c-1, c) == c).
        cache.chunk_ids[li] = torch.cat([cache.chunk_ids[li], last_cumsum], dim=1)

    cache.k_len = k_len

    # Publish for the other layers in this step to reuse (copy list objects so a
    # future first-layer in-place extend can't mutate the stored references).
    if input_ids is not None:
        sh.input_ids_ref = input_ids
        sh.k_len = k_len
        sh.is_gist_full = cache.is_gist_full
        sh.is_gist_per_level = list(cache.is_gist_per_level)
        sh.gist_cumsum = list(cache.gist_cumsum)
        sh.chunk_ids = list(cache.chunk_ids)


# ===========================================================================
# decode forward
# ===========================================================================

_SELECT_PG_COMPILED = None


def _next_bucket(g):
    """Smallest power of 2 >= g+1 (>=256). Buckets Gmax so torch.compile/CUDA-graph
    sees only a handful of distinct selection shapes across context lengths."""
    b = 256
    while b < g + 1:
        b <<= 1
    return b


def _pad_last(t, target, value):
    """Pad t's last dim up to `target` with `value` (no-op if already >= target)."""
    cur = t.shape[-1]
    if cur >= target:
        return t
    return F.pad(t, (0, target - cur), value=value)


def _select_pg_impl(q_g, Kt1, Kt0, gv1, gv0, pcids0, kk1, kk0):
    """Pure-tensor 2-level per-group selection (sum-agg), compiled by torch.compile.
    Folds ~16 PyTorch ops into a (CUDA-graphable) fused region to cut launch count
    in the CPU-launch-bound decode. Returns (sel0, sel1) as **uint8** [B,num_kv,Gmax_l+1]
    (the bool->uint8 cast is folded INTO the graph so the caller needs no extra launch).
    Mirrors _select_per_group's sum path exactly (the selected SET, not bit-order)."""
    B, nkv = q_g.shape[0], q_g.shape[1]
    # Level 1 (parent)
    s1 = torch.matmul(q_g, Kt1).float().sum(dim=2)            # [B,nkv,G1]
    s1 = s1.masked_fill(~gv1, float("-inf"))
    G1 = s1.shape[2]
    top1 = s1.topk(kk1, dim=2).indices                        # [B,nkv,kk1]
    sel1 = torch.zeros((B, nkv, G1 + 1), dtype=torch.bool, device=s1.device)
    sel1.scatter_(2, top1.clamp(min=0), True)
    sel1[..., 0] = (top1 == 0).any(dim=2)
    # Level 0 (child) with parent constraint
    s0 = torch.matmul(q_g, Kt0).float().sum(dim=2)            # [B,nkv,G0]
    G0 = s0.shape[2]
    child_valid = sel1.gather(2, pcids0[:, None, :].expand(B, nkv, G0))
    s0 = s0.masked_fill(~child_valid, float("-inf"))
    s0 = s0.masked_fill(~gv0, float("-inf"))
    top0 = s0.topk(kk0, dim=2).indices
    sel0 = torch.zeros((B, nkv, G0 + 1), dtype=torch.bool, device=s0.device)
    sel0.scatter_(2, top0.clamp(min=0), True)
    sel0[..., 0] = (top0 == 0).any(dim=2)
    return sel0.to(torch.uint8), sel1.to(torch.uint8)


def _get_select_pg_compiled():
    global _SELECT_PG_COMPILED
    if _SELECT_PG_COMPILED is None:
        mode = os.environ.get("DECODE_COMPILE_MODE", "reduce-overhead")
        # dynamic=True => one compile for all context lengths (no per-length recompile;
        # better for varying-length serving). dynamic=False (default) => static shapes,
        # required by reduce-overhead's CUDA graphs (fastest at fixed/bucketed length).
        dyn = True if os.environ.get("DECODE_COMPILE_DYNAMIC", "0") == "1" else False
        _SELECT_PG_COMPILED = torch.compile(_select_pg_impl, mode=mode, dynamic=dyn)
    return _SELECT_PG_COMPILED


def _select_per_group(query, cache, num_kv, hpg, top_k, top_p, num_levels, device):
    """Per-GQA-group hierarchical selection (decode, qb=1).
    Scores num_kv groups against num_kv-head bf16 gist-K (÷H/num_kv bytes vs the
    per-head fp64 path), aggregates the hpg group heads, top-k per group, with the
    L1→L0 parent constraint. Returns sel_any[level] = [B,num_kv,Gmax+1] bool — the
    per-group selected-chunk mask the CUDA kernel consumes directly (no union)."""
    B = query.shape[0]
    d = query.shape[-1]
    agg = os.environ.get("DECODE_GROUP_AGG", "sum")
    q_g = query.reshape(B, num_kv, hpg, d)           # [B,num_kv,hpg,d] (head h -> group h//hpg)

    # Gated stretch: fuse the 2-level selection via torch.compile (fewer launches in
    # the launch-bound decode). Fallback-safe — default path below is untouched.
    if (os.environ.get("DECODE_COMPILE", "0") == "1" and agg == "sum" and top_p <= 0
            and num_levels == 2 and cache.Gmax[0] > 0 and cache.Gmax[1] > 0):
        kk1 = int(min(top_k[1], max(1, cache.Gmax[1] // 4)))
        kk0 = int(min(top_k[0], max(1, cache.Gmax[0] // 4)))
        if os.environ.get("DECODE_GMAX_BUCKET", "0") == "1":
            # Rank 1: bucket the selection shapes (Gmax + kk) so reduce-overhead
            # CUDA graphs are reused across context lengths => robust ~18ms (not just
            # fixed-length). Pad inputs to power-of-2 Gmax (dummy gists masked to -inf,
            # never selected) and round kk up to a grid (selects a superset of chunks
            # => output closer to dense, F1-safe). Computed ONCE/generation (Gmax is
            # constant during decode), cached on the per-layer cache.
            sp = cache.sel_pad
            if sp is None:
                kg = int(os.environ.get("DECODE_KK_GRAN", "8"))
                b0 = _next_bucket(cache.Gmax[0]); b1 = _next_bucket(cache.Gmax[1])
                kk0b = min(((kk0 + kg - 1) // kg) * kg, b0 - 1)
                kk1b = min(((kk1 + kg - 1) // kg) * kg, b1 - 1)
                sp = {
                    "Kt0": _pad_last(cache.K_gist_kv_bf16_T[0], b0, 0.0),
                    "Kt1": _pad_last(cache.K_gist_kv_bf16_T[1], b1, 0.0),
                    "gv0": _pad_last(cache.gist_valid[0][:, 0, :, :], b0, False),
                    "gv1": _pad_last(cache.gist_valid[1][:, 0, :, :], b1, False),
                    "pcids0": _pad_last(cache.parent_chunk_ids[0], b0, 0),
                    "kk0": kk0b, "kk1": kk1b,
                }
                cache.sel_pad = sp
            sel0, sel1 = _get_select_pg_compiled()(
                q_g, sp["Kt1"], sp["Kt0"], sp["gv1"], sp["gv0"], sp["pcids0"],
                sp["kk1"], sp["kk0"])
            return [sel0, sel1]
        gv1 = cache.gist_valid[1][:, 0, :, :]         # [B,1,Gmax1]
        gv0 = cache.gist_valid[0][:, 0, :, :]
        sel0, sel1 = _get_select_pg_compiled()(
            q_g, cache.K_gist_kv_bf16_T[1], cache.K_gist_kv_bf16_T[0],
            gv1, gv0, cache.parent_chunk_ids[0], kk1, kk0)
        return [sel0, sel1]

    sel_any = [None] * num_levels
    selected_parent = None                            # [B,num_kv,Gp+1] bool
    for level_idx in range(num_levels - 1, -1, -1):
        Gmax = cache.Gmax[level_idx]
        if Gmax == 0:
            continue
        Kt = cache.K_gist_kv_bf16_T[level_idx]        # [B,num_kv,d,Gmax] bf16
        s = torch.matmul(q_g, Kt).float()             # [B,num_kv,hpg,Gmax]
        if agg == "max":
            gscore = s.max(dim=2).values
        elif agg == "softmax":
            gscore = torch.softmax(s, dim=-1).sum(dim=2)
        else:                                         # "sum" (default)
            gscore = s.sum(dim=2)                     # [B,num_kv,Gmax]
        # --- ABLATION (gated; inert unless DECODE_PIVOT is set). Replaces the GIST
        #     score with an alternative SELECTION PIVOT to test the claim that gist
        #     tokens are good pivots. Same candidate set / top_k / parent-constraint /
        #     kernel — only the score used for ranking changes. Default "gist" =
        #     unchanged. See REBUTTAL_GIST_PIVOT.md / analysis/. ---
        _pivot = os.environ.get("DECODE_PIVOT", "gist")
        if _pivot != "gist":
            if _pivot == "random":          # lower bound: ignore content
                gscore = torch.rand(B, num_kv, Gmax, device=device)
            elif _pivot == "recent":        # position-only: prefer most-recent chunks
                gscore = torch.arange(Gmax, device=device, dtype=torch.float32)[None, None, :].expand(B, num_kv, Gmax).contiguous()
            elif _pivot == "meanpool":      # score q against mean-pooled chunk keys
                Ktm = cache.K_meanpool_kv_bf16_T[level_idx]   # [B,num_kv,d,Gmax]
                gscore = torch.matmul(q_g, Ktm).float().sum(dim=2)
            # unknown value -> fall through with the gist score
        if selected_parent is not None:
            pcids = cache.parent_chunk_ids[level_idx]              # [B,Gmax]
            pcids_exp = pcids[:, None, :].expand(B, num_kv, Gmax)
            child_valid = selected_parent.gather(dim=2, index=pcids_exp)
            gscore = gscore.masked_fill(~child_valid, float("-inf"))
        gv = cache.gist_valid[level_idx][:, 0, :, :].expand(B, num_kv, Gmax)  # [B,num_kv,Gmax]
        gscore = gscore.masked_fill(~gv, float("-inf"))
        if top_p <= 0:
            kk = min(top_k[level_idx], max(1, Gmax // 4))
            top_idx, _ = top_k_from_logits(gscore, k=kk)          # [B,num_kv,kk]
        else:
            top_idx, _ = top_p_from_logits(gscore, p=top_p)
        top_idx_safe = top_idx.clamp(min=0)
        scm = torch.zeros((B, num_kv, Gmax + 1), dtype=torch.bool, device=device)
        scm.scatter_(dim=2, index=top_idx_safe, value=1)
        scm[..., 0] = (top_idx == 0).any(dim=2)
        sel_any[level_idx] = scm
        selected_parent = scm
    return sel_any


def _decode_forward(module, query, key, value, attention_mask, scaling, dropout, **kwargs):
    # NOTE: do NOT repeat_kv here — the kernel path uses the un-repeated K/V.
    # repeat_kv materializes a [B,H,k_len,d] copy (~hundreds of MB at large
    # k_len); only the eager-SDPA fallback needs it, so it's done lazily there.
    B, H, _, d = query.shape  # q_len = 1 (decode)
    qb = 1
    _, _, k_len, _ = key.shape  # un-repeated [B, num_kv, k_len, d]
    key_states = value_states = None  # materialized lazily in the SDPA fallback

    input_ids = kwargs.get("tmp_ids", None)
    gist_ids = Global_data.gist_token_id
    causal_mask = kwargs.get("causal_mask", None)
    top_k = copy.deepcopy(Global_data.top_k)
    if Global_data.comp_factor > 0:
        comp_factor = Global_data.comp_factor
    else:
        comp_factor = (
            Global_data.chunk_size[0] if len(gist_ids) == 1
            else Global_data.chunk_size[0] * Global_data.chunk_size[1]
        )
    top_k[0] = k_len // comp_factor // module.num_key_value_groups // Global_data.chunk_size[0] + 1
    top_k[1] = top_k[0]
    top_p = Global_data.top_p

    device = query.device
    num_levels = len(gist_ids)

    # The CUDA sparse-decode path supports 1-level (SSA) and 2-level hierarchical
    # (H-SSA) gists. Single-level feeds the kernel a no-op level-1 (is_l1=0 masks
    # the (is_l1g & sel1) term; see _DecodeKernelStepCache + the sel1 dummy below).
    # >2 levels must use the eager path. Fail cleanly rather than deep in setup.
    if num_levels not in (1, 2):
        raise NotImplementedError(
            f"sparse_attention_forward decode supports 1- or 2-level gists "
            f"(got num_levels={num_levels}); use the eager path (ATTN_IMPL=ref)."
        )
    # ----- Cache lookup / refresh / extend -----
    cache: Optional[_GistDecodeCache] = getattr(module, "_gist_decode_cache", None)

    # The decode cache appends exactly one non-gist column per step. That
    # "generated tokens are non-gist" invariant is enforced at generation time by
    # the _SuppressGistTokens logits processor (installed by the benchmark harness),
    # which masks gist token ids to -inf so they can never be emitted. Hence the
    # optimistic single-step extend below is always valid (no per-step host sync).
    can_extend = (
        cache is not None
        and cache.k_len + 1 == k_len
        and cache.B == B and cache.H == H and cache.d == d
        and cache.num_levels == num_levels
    )

    if not can_extend:
        cache = _build_full_cache(
            input_ids, key, gist_ids, k_len, B, H, d, device, num_levels,
            n_rep=module.num_key_value_groups,
        )
        module._gist_decode_cache = cache
    else:
        _extend_cache_one_step(cache, B, k_len, device, input_ids)

    # ----- Per-call dynamic state -----
    # LAZY: these [B,k_len] tensors are identical across all layers in a step and
    # are ONLY consumed by the non-compact (sparse-mask / eager) branches. The
    # optimized compact-kernel path derives `compressed` once/step inside
    # _DecodeKernelStepCache.get, so computing them per-layer here is pure launch
    # overhead. Build on first use via _state().
    last_gist = cache.last_gist
    start_idx = cache.start_idx
    sink_keep = None  # ref decode forward never used sink_keep in keep_mask
    _state_cache = {}

    def _state():
        if not _state_cache:
            k_idx = torch.arange(k_len, device=device)
            alk = (k_idx[None, :] > last_gist[:, None])
            _state_cache["after_last_key"] = alk
            _state_cache["compressed"] = (~alk & (k_idx[None, :] >= start_idx[:, None]))
            _state_cache["attention_mask_bool"] = (attention_mask > -1e-4).bool()
        return _state_cache

    # ----- Top-down level loop -----
    selected_chunks_per_level: list = [None] * num_levels  # full Gmax_level+1 versions
    selected_parent_chunks: Optional[torch.Tensor] = None
    sel_any_per_level = None  # set by the per-group path

    _select_mode = os.environ.get("DECODE_SELECT", "head")
    if (_select_mode == "group" and num_levels in (1, 2)
            and module.num_key_value_groups > 1 and qb == 1 and d == 128):
        # Per-group selection: SUM the group heads' query.gist scores into
        # one per-group score, then a single top-k per group (see _select_per_group).
        # This is a DIFFERENT selection from the paper's per-head-then-union (= the
        # `head` path below); cheaper, F1-validated, NOT allclose to the eager reference.
        # _select_per_group's eager loop generalizes over num_levels (single-level
        # = top-k level-0 only, no parent constraint).
        hpg_ = module.num_key_value_groups
        num_kv_ = H // hpg_
        sel_any_per_level = _select_per_group(
            query, cache, num_kv_, hpg_, top_k, top_p, num_levels, device)

    # Per-head gist selection (DECODE_SELECT=head): bf16 tensor-core scoring over
    # the dominant K_gist read. F1-neutral vs an exact-dtype score; the per-group
    # path (DECODE_SELECT=group) is handled above in _select_per_group.
    for level_idx in range(num_levels - 1, -1, -1) if sel_any_per_level is None else range(0):
        Gmax_level = cache.Gmax[level_idx]
        if Gmax_level == 0:
            continue
        gist_valid_level = cache.gist_valid[level_idx]

        gist_scores_level = torch.matmul(query, cache.K_gist_bf16_T[level_idx]).float()

        if selected_parent_chunks is not None:
            parent_chunk_ids = cache.parent_chunk_ids[level_idx]
            parent_chunk_ids_exp = parent_chunk_ids[:, None, None, :].expand(
                B, H, qb, Gmax_level
            )
            child_valid = selected_parent_chunks.gather(dim=3, index=parent_chunk_ids_exp)
            gist_scores_level = gist_scores_level.masked_fill(~child_valid, float("-inf"))

        gist_scores_level = gist_scores_level.masked_fill(~gist_valid_level, float("-inf"))

        if top_p <= 0:
            kk_level = min(top_k[level_idx], max(1, Gmax_level // 4))
            top_in_gist_level, _ = top_k_from_logits(gist_scores_level, k=kk_level)
        else:
            top_in_gist_level, _ = top_p_from_logits(gist_scores_level, p=top_p)

        top_idx_safe = top_in_gist_level.clamp(min=0)
        selected_chunk_mask = torch.zeros(
            (B, H, qb, Gmax_level + 1), dtype=torch.bool, device=device
        )
        selected_chunk_mask.scatter_(dim=3, index=top_idx_safe, value=1)
        selected_chunk_mask[..., 0] = (top_in_gist_level == 0).any(dim=3)

        selected_chunks_per_level[level_idx] = selected_chunk_mask
        selected_parent_chunks = selected_chunk_mask

    # ----- Fused mask construction: materialize + group-union + final mask compose.
    use_nsa_gqa = getattr(Global_data, 'use_nsa_gqa', False)
    # Need level-0 selection always; level-1 only for the hierarchical variant.
    _need_l1 = (num_levels == 2)
    if sel_any_per_level is not None:
        have_sel = sel_any_per_level[0] is not None and (not _need_l1 or sel_any_per_level[1] is not None)
    else:
        have_sel = (selected_chunks_per_level[0] is not None
                    and (not _need_l1 or selected_chunks_per_level[1] is not None))

    # ===== Sparse flash-decode CUDA kernel path =====
    # Attends over ONLY the selected keys, computing keep on the fly — no
    # [H,k_len] mask materialization, no full-K SDPA.
    # On "exactness": for a GIVEN selection the kernel is allclose (not bit-exact) to
    # eager-over-that-same-selection. End-to-end allclose to the eager reference thus
    # holds only for `head` mode (union = the reference's selection); `group` mode is
    # a different selection (F1-validated, not allclose).
    if (
        use_nsa_gqa
        and have_sel
        and num_levels in (1, 2)
        and module.num_key_value_groups > 1
        and qb == 1
        and d == 128
    ):
        hpg = module.num_key_value_groups
        num_kv_heads = H // hpg
        # Single-level: feed a no-op level-1 selection (1 dummy chunk, never kept —
        # is_l1=0 in the step-cache masks the (is_l1g & sel1) term; c1=0 keeps the
        # sel1[g, c1[k]] read in-bounds at G1p1=1).
        _dummy_sel1 = torch.zeros((B, num_kv_heads, 1), dtype=torch.uint8, device=device)
        if sel_any_per_level is not None:
            # Per-group selection already produced [num_kv, Gmax+1] directly. The
            # compiled path already returns contiguous uint8 (in-graph cast), so skip
            # the redundant cast/copy launches when possible (Rank 3).
            _s0 = sel_any_per_level[0]
            sel0_any = _s0 if (_s0.dtype == torch.uint8 and _s0.is_contiguous()) else _s0.to(torch.uint8).contiguous()
            if _need_l1:
                _s1 = sel_any_per_level[1]
                sel1_any = _s1 if (_s1.dtype == torch.uint8 and _s1.is_contiguous()) else _s1.to(torch.uint8).contiguous()
            else:
                sel1_any = _dummy_sel1
        else:
            # head mode = the paper's grouped unfolding (§3.3): per-head top-k, then OR
            # (union) within the GQA group. This matches the eager reference's selected
            # set, so the end-to-end output is allclose to eager (unlike `group` above).
            sel0_any = (selected_chunks_per_level[0][:, :, 0, :]
                        .reshape(B, num_kv_heads, hpg, -1).any(dim=2).to(torch.uint8).contiguous())
            if _need_l1:
                sel1_any = (selected_chunks_per_level[1][:, :, 0, :]
                            .reshape(B, num_kv_heads, hpg, -1).any(dim=2).to(torch.uint8).contiguous())
            else:
                sel1_any = _dummy_sel1
        # [k_len]-derived inputs are identical across layers in a step → cache once.
        sc = _DecodeKernelStepCache.get(input_ids, cache,
                                        attention_mask, causal_mask, k_len)
        nsplit = int(os.environ.get("DECODE_NSPLIT", "128"))
        ext = _get_sparse_decode_ext()
        qc = query if query.is_contiguous() else query.contiguous()
        kc = key if key.is_contiguous() else key.contiguous()
        vc = value if value.is_contiguous() else value.contiguous()
        # cap = k_len → the compaction can NEVER drop a selected key (the attention
        # kernel derives its per-group split width from the on-device count, so
        # occupancy is independent of cap). Bulletproof vs the worst-case group-union
        # union / long-suffix edge cases. The compact decode = parallel
        # warp-aggregated compaction + per-group shared-mem tiling.
        cap = k_len
        out = ext.sparse_decode_attn_compact(
            qc, kc, vc, sel0_any, sel1_any, sc.c0_i32, sc.c1_i32,
            sc.is_gist_u8, sc.is_l0_u8, sc.is_l1_u8, sc.comp_u8, sc.am_u8,
            sc.causal_bias, scaling, nsplit, cap,
        )

        # One-shot debug dump of real decode tensors (env-gated) for kernel/selector
        # analysis (e.g. analysis/gist_pivot_recall.py). Inert unless DUMP_DECODE set.
        _dump = os.environ.get("DUMP_DECODE", "")
        if _dump and not getattr(module, "_dumped_decode", False) and module.layer_idx == 1:
            module._dumped_decode = True
            torch.save({
                "query": query.detach().cpu(),
                "key": key.detach().cpu(),          # un-repeated [B,num_kv,k_len,d]
                "value": value.detach().cpu(),
                "is_gist_full": cache.is_gist_full.detach().cpu(),
                "is_gist_l0": cache.is_gist_per_level[0].detach().cpu(),
                "is_gist_l1": cache.is_gist_per_level[1].detach().cpu() if num_levels == 2 else None,
                "chunk_ids_l0": cache.chunk_ids[0].detach().cpu(),
                "chunk_ids_l1": cache.chunk_ids[1].detach().cpu() if num_levels == 2 else None,
                "sel0_any": sel0_any.detach().cpu(),
                "sel1_any": sel1_any.detach().cpu(),
                "Gmax": list(cache.Gmax),
                "num_kv_groups": module.num_key_value_groups,
                "scaling": scaling,
            }, _dump)
            print(f"[DUMP_DECODE] wrote {_dump} k_len={k_len} layer={module.layer_idx}", flush=True)
        return out, None

    # Shapes the CUDA sparse-decode kernel does not cover (single-level gists,
    # non-GQA, head_dim != 128, q_len > 1). The eager PyTorch fallback was removed
    # in the CUDA-only cleanup; the differentiable reference still lives in
    # gist_qwen2.eager_attention_forward_decoding and is used for training.
    raise NotImplementedError(
        "sparse_decode kernel requires 2-level gists, GQA (num_kv_groups>1), "
        f"head_dim=128, q_len=1; got num_levels={num_levels}, "
        f"groups={module.num_key_value_groups}, d={d}, qb={qb}. "
        "Use the eager training path (ATTN_IMPL unset / module.training) for these shapes."
    )


# ===========================================================================
# v4 prefill — bit-equal rewrite of _ref_prefill_chunk with two wins:
#  (1) Path A: full-query SDPA instead of per-batch sliced + index_put;
#      removes the `q_start[b].item()` sync. Verified bit-equal for the
#      prefix portion (suffix portion is overwritten by Path B anyway).
#  (2) Per-step cache (across the 32 layers within one prefill call) of
#      input_ids-derived state: gist_pos, gist_cumsum, chunk_ids,
#      parent_chunk_ids_from_upper, last_gist, start_idx, is_gist_full.
#      Computed once, reused 32× across layers. Per-layer K_gist is
#      still gathered fresh (key_states differs per layer).
# ===========================================================================


# Lazy flex_attention handles (compiled once).
_FLEX = None
_CREATE_BLOCK_MASK = None


def _get_flex():
    global _FLEX, _CREATE_BLOCK_MASK
    if _FLEX is None:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask
        _FLEX = torch.compile(flex_attention)
        _CREATE_BLOCK_MASK = create_block_mask
    return _FLEX, _CREATE_BLOCK_MASK


class _PrefillFlexCache:
    """Per-prefill-step cache of the gist-mask key-permutation + BlockMask.
    Clusters the scattered global gist/sink columns contiguous so the (exact
    same) gist mask becomes block-sparse-efficient. Keyed by id(attention_mask)
    — shared across all layers in a step; only the K/V gather is per-layer."""
    mask_ref = None    # held attention_mask tensor (identity check; vs id() reuse)
    q_len: int = -1
    k_len: int = -1
    perm = None        # [k_len] long: [global cols, local cols], each by orig pos
    block_mask = None  # flex BlockMask on the permuted layout

    @classmethod
    def get(cls, attention_mask, q_len, k_len, device):
        if cls.mask_ref is attention_mask and cls.q_len == q_len and cls.k_len == k_len:
            return cls
        cls.mask_ref = attention_mask
        cls.q_len = q_len; cls.k_len = k_len
        amb = (attention_mask[0, 0] > -1e-4)          # [q_len, k_len] bool
        col_freq = amb.float().mean(0)                # visibility frequency per key
        is_global = col_freq > 0.3                    # gist/sink columns (shared by many q)
        idx = torch.arange(k_len, device=device)
        glob = idx[is_global]; loc = idx[~is_global]
        perm = torch.cat([glob, loc])
        cls.perm = perm
        amb_perm = amb[:, perm]                       # [q_len, k_len_permuted]
        flex, create_block_mask = _get_flex()
        def mask_mod(b, h, qi, kj):
            return amb_perm[qi, kj]
        cls.block_mask = create_block_mask(mask_mod, 1, 1, q_len, k_len, device=device)
        return cls


class _PrefillStepCache:
    """Per-prefill-step shared state. Keyed by id(input_ids); the same
    `input_ids` object is passed to all 32 decoder layers in a step, so
    we compute once and reuse 31 times."""
    input_ids_ref = None     # held tensor (identity check; vs id() reuse)
    s_k_len: int = -1
    num_levels: int = 0
    Gmax = None              # list[int]
    gist_pos = None          # list[Optional[Tensor [B, Gmax]]]
    gist_pos_safe = None     # list[Optional[Tensor [B, Gmax]]]
    gist_valid = None        # list[Optional[Tensor [B, 1, 1, Gmax]]]
    chunk_ids = None         # list[Tensor [B, k_len]]
    parent_chunk_ids = None  # list[Optional[Tensor [B, Gmax_lower]]]
    is_gist_full = None      # [B, k_len] bool
    last_gist = None         # [B] long
    start_idx = None         # [B] long
    gist_token_tensor = None

    @classmethod
    def maybe_build(cls, input_ids, gist_ids, num_levels, B, k_len, device, pad_token_id):
        if (cls.input_ids_ref is input_ids and cls.s_k_len == k_len
                and cls.num_levels == num_levels):
            return  # cache hit
        cls.input_ids_ref = input_ids
        cls.s_k_len = k_len
        cls.num_levels = num_levels

        cls.gist_token_tensor = torch.tensor(gist_ids, device=device)
        cls.is_gist_full = torch.isin(input_ids, cls.gist_token_tensor)

        cls.Gmax = []
        cls.gist_pos = []
        cls.gist_pos_safe = []
        cls.gist_valid = []
        cls.chunk_ids = []
        cls.parent_chunk_ids = [None] * num_levels

        is_gist_per_level = []
        gist_cumsum_per_level = []
        for li in range(num_levels):
            is_gist_li = (input_ids == gist_ids[li])
            is_gist_per_level.append(is_gist_li)
            cumsum_li = is_gist_li.cumsum(dim=1)
            gist_cumsum_per_level.append(cumsum_li)

            chunk_ids = torch.where(is_gist_li, cumsum_li - 1, cumsum_li).clamp_min(0)
            cls.chunk_ids.append(chunk_ids)

            Gmax = int(is_gist_li.sum(dim=1).max().item())
            cls.Gmax.append(Gmax)

            if Gmax == 0:
                cls.gist_pos.append(None)
                cls.gist_pos_safe.append(None)
                cls.gist_valid.append(None)
                continue

            gist_pos = torch.zeros((B, Gmax), device=device, dtype=torch.long)
            nz = is_gist_li.nonzero(as_tuple=False)
            rank = cumsum_li[nz[:, 0], nz[:, 1]] - 1
            gist_pos[nz[:, 0], rank] = nz[:, 1]
            cls.gist_pos.append(gist_pos)
            cls.gist_pos_safe.append(gist_pos.clamp(0, k_len - 1))
            cls.gist_valid.append((gist_pos > 0)[:, None, None, :])

        for li in range(num_levels - 2, -1, -1):
            if cls.gist_pos[li] is None or cls.Gmax[li + 1] == 0:
                continue
            cls.parent_chunk_ids[li] = cls.chunk_ids[li + 1].gather(
                dim=1, index=cls.gist_pos[li]
            )

        cls.last_gist = torch.where(
            is_gist_per_level[-1],
            torch.arange(k_len, device=device),
            torch.full((k_len,), -1, device=device, dtype=torch.long),
        ).amax(dim=1)
        cls.start_idx = (input_ids != pad_token_id).int().argmax(dim=1)


def _prefill_forward(module, query, key, value, attention_mask, scaling, dropout, chunk_q, **kwargs):
    # A prefill starts a NEW sequence → invalidate this layer's decode cache so a
    # subsequent decode can't accidentally "extend" the previous sample's cache.
    module._gist_decode_cache = None

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    B, H, q_len, d = query.shape
    _, _, k_len, _ = key_states.shape

    input_ids = kwargs.get("tmp_ids", None)
    gist_ids = Global_data.gist_token_id
    PAD_ID = Global_data.pad_token_id
    sink_size = Global_data.sink_size
    causal_mask = kwargs.get("causal_mask", None)
    top_k = copy.deepcopy(Global_data.top_k)
    if Global_data.comp_factor > 0:
        comp_factor = Global_data.comp_factor
    else:
        comp_factor = (
            Global_data.chunk_size[0] if len(gist_ids) == 1
            else Global_data.chunk_size[0] * Global_data.chunk_size[1]
        )
    top_k[0] = k_len // comp_factor // module.num_key_value_groups // Global_data.chunk_size[0] + 1
    top_k[1] = top_k[0]
    top_p = Global_data.top_p

    device = query.device
    num_levels = len(gist_ids)

    # Per-step cache (shared across all 32 layers in this prefill call).
    _PrefillStepCache.maybe_build(input_ids, gist_ids, num_levels, B, k_len, device, PAD_ID)
    pcache = _PrefillStepCache

    last_gist = pcache.last_gist
    start_idx = pcache.start_idx
    is_gist = pcache.is_gist_full

    out = torch.empty(B, q_len, H, d, device=device, dtype=query.dtype)

    k_idx = torch.arange(k_len, device=device)
    q_start = last_gist + 1
    qs_len = q_len - q_start
    qs_max = int(qs_len.max().item())

    ar = torch.arange(qs_max, device=device)
    q_pos = q_start[:, None] + ar[None, :]
    valid_q = q_pos < q_len
    q_pos_clamped = q_pos.clamp(max=q_len - 1)

    sink_keep = None
    if sink_size > 0:
        sink_keep = (k_idx[None, :] >= start_idx[:, None]) & (k_idx[None, :] < (start_idx + sink_size)[:, None])

    after_last_key = (k_idx[None, :] > last_gist[:, None])
    compressed = (~after_last_key & (k_idx[None, :] >= start_idx[:, None]))

    # One-shot prefill mask analysis (env-gated).
    _pdump = os.environ.get("DUMP_PREFILL", "")
    if _pdump and not getattr(module, "_pdumped", False) and module.layer_idx == 1:
        module._pdumped = True
        amb = (attention_mask[0, 0] > -1e-4)  # [q_len, k_len] bool (B=1)
        per_q = amb.sum(dim=1)  # keys visible per query
        torch.save({
            "q_len": q_len, "k_len": k_len,
            "per_q_keys": per_q.cpu(),
            "amb_slice": amb[::max(1, q_len // 512), ::max(1, k_len // 512)].cpu(),
            "q_start": int(q_start.min().item()),
        }, _pdump)
        print(f"[DUMP_PREFILL] {_pdump} q_len={q_len} k_len={k_len} "
              f"keys/query min={int(per_q.min())} max={int(per_q.max())} mean={per_q.float().mean():.1f}",
              flush=True)

    # ----- PATH A: attention over the gist mask for ALL queries (prefix gets
    # its final output here; suffix overwritten by Path B).
    if os.environ.get("PREFILL_KERNEL", "dense") == "flex":
        # Sparse block-sparse via key-column permutation (cluster scattered
        # global gist/sink columns contiguous) + flex_attention. Exactly the
        # same mask → allclose; ~6-7x faster, skips ~90% of blocks.
        flex, _ = _get_flex()
        pc = _PrefillFlexCache.get(attention_mask, q_len, k_len, device)
        perm = pc.perm
        kP = key_states[:, :, perm, :]
        vP = value_states[:, :, perm, :]
        out_full = flex(query, kP, vP, block_mask=pc.block_mask, scale=scaling)
        out[:] = out_full.transpose(1, 2).contiguous()
    else:
        out_full = F.scaled_dot_product_attention(
            query, key_states, value_states,
            attn_mask=attention_mask,
            dropout_p=dropout if module.training else 0.0,
            scale=scaling,
        )
        out[:] = out_full.transpose(1, 2).contiguous()

    # Per-layer K_gist gather (depends on key_states; cannot be cached cross-layer).
    K_gist_per_level = []
    for li in range(num_levels):
        Gmax = pcache.Gmax[li]
        if Gmax == 0:
            K_gist_per_level.append(None)
            continue
        gist_pos_safe = pcache.gist_pos_safe[li]
        gist_gather = gist_pos_safe[:, None, :, None].expand(B, H, Gmax, d)
        K_gist_per_level.append(key_states.gather(dim=2, index=gist_gather))

    # ----- PATH B: chunked sparse SDPA over the suffix queries.
    for s in range(0, qs_max, chunk_q):
        e = min(s + chunk_q, qs_max)
        qb = e - s

        q_pos_blk = q_pos_clamped[:, s:e]
        valid_q_blk = valid_q[:, s:e]

        q_gather = q_pos_blk[:, None, :, None].expand(B, H, qb, d)
        q_suffix_blk = query.gather(2, q_gather)
        q_suffix_blk = torch.where(
            valid_q_blk[:, None, :, None], q_suffix_blk, torch.zeros_like(q_suffix_blk),
        )

        q_gather_blk = q_pos_blk[:, None, :, None].expand(B, 1, qb, k_len)
        attention_mask_bool_blk = (attention_mask > -1e-4).bool().gather(2, q_gather_blk)
        attention_mask_suffix_blk = torch.where(
            valid_q_blk[:, None, :, None],
            attention_mask_bool_blk,
            torch.zeros_like(attention_mask_bool_blk),
        )

        cm_blk = (causal_mask.gather(2, q_gather_blk) > -1e-4)
        cm_blk = torch.where(
            valid_q_blk[:, None, :, None], cm_blk, torch.zeros_like(cm_blk),
        )

        # Top-down level loop (mirrors _ref_prefill_chunk's structure).
        selected_parent_chunks = None
        selected_gist_records = []
        selected_level0_chunks = None

        for level_idx in range(num_levels - 1, -1, -1):
            Gmax_level = pcache.Gmax[level_idx]
            if Gmax_level == 0:
                continue
            gist_pos_safe = pcache.gist_pos_safe[level_idx]
            K_gist_level = K_gist_per_level[level_idx]
            gist_valid_level = pcache.gist_valid[level_idx]

            gist_scores_level = torch.matmul(
                q_suffix_blk.double(), K_gist_level.transpose(2, 3).double(),
            )

            if selected_parent_chunks is not None:
                parent_chunk_ids = pcache.parent_chunk_ids[level_idx]
                parent_chunk_ids_exp = parent_chunk_ids[:, None, None, :].expand(
                    B, H, qb, Gmax_level
                )
                child_valid = selected_parent_chunks.gather(dim=3, index=parent_chunk_ids_exp)
                gist_scores_level = gist_scores_level.masked_fill(~child_valid, float("-inf"))

            gist_scores_level = gist_scores_level.masked_fill(~gist_valid_level, float("-inf"))
            gist_scores_level = gist_scores_level.masked_fill(~valid_q_blk[:, None, :, None], float("-inf"))

            if top_p <= 0:
                kk_level = min(top_k[level_idx], max(1, Gmax_level // 4))
                top_in_gist_level, _ = top_k_from_logits(gist_scores_level, k=kk_level)
            else:
                top_in_gist_level, _ = top_p_from_logits(gist_scores_level, p=top_p)

            top_idx_safe = top_in_gist_level.clamp(min=0)
            selected_gists = torch.zeros((B, H, qb, Gmax_level), dtype=torch.bool, device=device)
            selected_gists.scatter_(dim=3, index=top_idx_safe, value=True)
            selected_gists &= gist_valid_level
            selected_gists &= torch.isfinite(gist_scores_level)

            selected_gist_records.append((gist_pos_safe, selected_gists))

            selected_parent_chunks = torch.zeros(
                (B, H, qb, Gmax_level + 1), dtype=torch.bool, device=device
            )
            selected_parent_chunks[..., :Gmax_level] = selected_gists

            if level_idx == 0:
                selected_level0_chunks = selected_parent_chunks

        # Materialize keep_mask (mirrors _ref_prefill_chunk lines 889-908).
        if selected_level0_chunks is not None:
            chunk_ids_l0 = pcache.chunk_ids[0]
            idx = chunk_ids_l0[:, None, None, :].expand(B, H, qb, k_len)
            compressed_descendants = selected_level0_chunks.gather(dim=3, index=idx)
            compressed_descendants &= compressed[:, None, None, :]
            compressed_descendants &= ~(is_gist[:, None, None, :])
        else:
            compressed_descendants = torch.zeros((B, H, qb, k_len), dtype=torch.bool, device=device)

        all_selected_gist_pos = torch.zeros((B, H, qb, k_len), dtype=torch.bool, device=device)
        for gist_pos_safe, selected_gists in selected_gist_records:
            G = gist_pos_safe.shape[1]
            idx = gist_pos_safe[:, None, None, :].expand(B, H, qb, G)
            tmp = torch.zeros_like(all_selected_gist_pos)
            tmp.scatter_(dim=3, index=idx, src=selected_gists)
            all_selected_gist_pos |= tmp

        keep_mask = compressed_descendants | all_selected_gist_pos

        # group union
        use_nsa_gqa = getattr(Global_data, 'use_nsa_gqa', False)
        if use_nsa_gqa and module.num_key_value_groups > 1:
            heads_per_group = module.num_key_value_groups
            num_kv_heads = H // heads_per_group
            keep_mask_grouped = keep_mask.view(B, num_kv_heads, heads_per_group, qb, k_len)
            keep_mask_union = keep_mask_grouped.any(dim=2, keepdim=True)
            keep_mask_union = keep_mask_union.expand(B, num_kv_heads, heads_per_group, qb, k_len)
            keep_mask = keep_mask_union.reshape(B, H, qb, k_len)

        attention_mask_suffix_blk &= ~(is_gist[:, None, None, :])
        keep_mask = keep_mask | attention_mask_suffix_blk

        if sink_keep is not None:
            keep_mask = keep_mask | sink_keep.unsqueeze(1).unsqueeze(2)

        keep_mask = keep_mask | after_last_key.unsqueeze(1).unsqueeze(2)
        keep_mask = keep_mask & valid_q_blk.unsqueeze(1).unsqueeze(-1)

        attn_mask = cm_blk & keep_mask

        sel_suffix = F.scaled_dot_product_attention(
            q_suffix_blk, key_states, value_states,
            attn_mask=attn_mask,
            dropout_p=dropout if module.training else 0.0,
            is_causal=False,
            scale=scaling,
        )

        # Scatter the sparse suffix output back into `out`. Path A's full SDPA
        # already wrote correct values for the prefix; this overwrites only
        # the qs_max suffix positions.
        src = sel_suffix.transpose(1, 2).contiguous()
        b_flat, t_flat = valid_q_blk.nonzero(as_tuple=True)
        q_flat = q_pos_blk[b_flat, t_flat]
        out[b_flat, q_flat] = src[b_flat, t_flat]

    return out.contiguous(), None


# ---------------------------------------------------------------------------
# Candidate kernel (public)
# ---------------------------------------------------------------------------

def sparse_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout: float = 0.0,
    decode: bool = False,
    chunk_q: int = 512,
    **kwargs,
):
    """Decode and prefill both go through optimized v4 paths."""
    if decode:
        return _decode_forward(
            module, query, key, value, attention_mask,
            scaling=scaling, dropout=dropout, **kwargs,
        )
    return _prefill_forward(
        module, query, key, value, attention_mask,
        scaling=scaling, dropout=dropout, chunk_q=chunk_q, **kwargs,
    )




def _call_ref(module, q, k, v, mask, scaling, dropout, decode, chunk_q, **kwargs):
    if decode:
        return _ref_decoding(
            module, q, k, v, mask,
            scaling=scaling, dropout=dropout, **kwargs,
        )
    return _ref_prefill_chunk(
        module, q, k, v, mask,
        scaling=scaling, dropout=dropout, chunk_q=chunk_q, **kwargs,
    )


def attn_dispatcher(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout: float = 0.0,
    decode: bool = False,
    chunk_q: int = 512,
    **kwargs,
):
    # CUDA-only inference + differentiable training path.
    #   * Training (module.training) -> eager reference: the sparse-decode CUDA
    #     kernels are forward-only, so the backward must flow through
    #     gist_qwen2.eager_attention_forward_*.
    #   * Inference (eval)           -> sparse_attention_forward (flex/dense
    #     prefill + compact sparse decode). ATTN_IMPL=ref forces eager for debugging.
    force_ref = os.environ.get("ATTN_IMPL", "sparse") == "ref"
    if force_ref or module.training:
        return _call_ref(
            module, query, key, value, attention_mask,
            scaling, dropout, decode, chunk_q, **kwargs,
        )
    return sparse_attention_forward(
        module, query, key, value, attention_mask,
        scaling=scaling, dropout=dropout,
        decode=decode, chunk_q=chunk_q, **kwargs,
    )
