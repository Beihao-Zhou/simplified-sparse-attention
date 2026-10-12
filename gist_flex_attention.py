"""FlexAttention for training with the gist causal mask (`Global_data.train_attn = "flex"`).

The dense path materializes [B, 1, T, T] masks, which caps training at ~20K tokens;
here the mask is a `mask_mod` over O(T) features (src/data/gist_flex_mask.py). K/V
are reordered into [raw | summary] (`gist_pool_permutation`) so that most key blocks
can be skipped; RoPE is already applied, so positions are unaffected.

Importing this module registers GIST_FLEX. Non-selective training only.
"""

import functools
import json
import os
from typing import Optional

import torch
from torch.nn.attention.flex_attention import flex_attention
from transformers import AttentionInterface

from src.data.gist_flex_mask import (
    build_gist_mask_features,
    gist_pool_permutation,
    make_gist_block_mask,
)

GIST_FLEX = "gist_flex"

_compiled_flex = None


def _flex(device: torch.device):
    """flex_attention, compiled for CUDA tensors and eager otherwise (e.g. CPU tests)."""
    global _compiled_flex
    if device.type != "cuda" or os.environ.get("GIST_FLEX_COMPILE", "1") != "1":
        return flex_attention
    if _compiled_flex is None:
        # Not dynamic=False: that recompiles per length and, past the recompile limit,
        # falls back to eager, which materializes [B, H, T, T] scores.
        _compiled_flex = torch.compile(flex_attention)
    return _compiled_flex


@functools.lru_cache(maxsize=None)
def _kernel_options() -> Optional[dict]:
    """GIST_FLEX_KERNEL_OPTIONS (JSON) overrides the Triton tile sizes, whose defaults
    can exceed shared memory on GPUs smaller than H100 (e.g. L4)."""
    opts = os.environ.get("GIST_FLEX_KERNEL_OPTIONS")
    return json.loads(opts) if opts else None


def prepare_gist_flex(
    input_ids: torch.Tensor,
    gist_token_ids,
    chunk_size: int,
    attention_sink_size: int,
    num_previous_chunks: int,
    pad_token_id: Optional[int],
    block_size: int = 128,
    raw_window: int = 0,
) -> dict:
    """Per-batch BlockMask and key permutation for all layers. Padding is found by
    pad_token_id; attention_mask is not read."""
    if pad_token_id is None:
        raise ValueError(f"{GIST_FLEX} needs pad_token_id to mask padded keys")
    f = build_gist_mask_features(
        input_ids, gist_token_ids, chunk_size,
        attention_sink_size=attention_sink_size,
        num_previous_chunks=num_previous_chunks,
        pad_token=pad_token_id,
        raw_window=raw_window,
    )
    perm = gist_pool_permutation(f)
    block_mask = make_gist_block_mask(
        f, input_ids.shape[1], block_size=block_size, kv_perm=perm,
        compile_mask=input_ids.is_cuda,
    )
    return {"gist_block_mask": block_mask, "gist_kv_perm": perm}


def gist_flex_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,      # [B, H, T, d], RoPE applied
    key: torch.Tensor,        # [B, H_kv, T, d], RoPE applied
    value: torch.Tensor,      # [B, H_kv, T, d]
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    gist_block_mask=None,
    gist_kv_perm: Optional[torch.Tensor] = None,
    **kwargs,
):
    if gist_block_mask is None or gist_kv_perm is None:
        raise ValueError(
            f"{GIST_FLEX} attention needs gist_block_mask and gist_kv_perm; set "
            "Global_data.train_attn = 'flex' so GistQwen2Model builds them.")
    if dropout:
        raise NotImplementedError(f"{GIST_FLEX} attention does not support dropout")
    if sliding_window is not None:
        raise NotImplementedError(f"{GIST_FLEX} attention does not support sliding windows")

    B, H_kv, T, d = key.shape
    idx = gist_kv_perm[:, None, :, None].expand(B, H_kv, T, d)
    key = key.gather(2, idx)
    value = value.gather(2, idx)

    out = _flex(query.device)(
        query, key, value,
        block_mask=gist_block_mask,
        scale=scaling,
        enable_gqa=query.shape[1] != H_kv,
        kernel_options=_kernel_options(),
    )
    return out.transpose(1, 2).contiguous(), None


AttentionInterface.register(GIST_FLEX, gist_flex_attention_forward)
