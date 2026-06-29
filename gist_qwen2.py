"""Huggingface Qwen2 with gist token support."""


import math
from typing import List, Optional, Tuple, Union, Callable
import copy
import torch
from torch import nn
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2PreTrainedModel,
    Qwen2Attention,
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
)
from transformers.modeling_layers import (
    GradientCheckpointingLayer,
)
from transformers.cache_utils import (
    Cache,
    DynamicCache,
)
from transformers.utils import logging, TransformersKwargs
from transformers.utils.deprecation import deprecate_kwarg
import torch.nn.functional as F
from transformers.utils.generic import check_model_inputs
from transformers.processing_utils import Unpack
from time import time

from generation_utils import GistGenerationMixin
from gist_caching import GistActivations
from gist_utils import Global_data

logger = logging.get_logger(__name__)


PRETRAINED_VOCAB_SIZE = 151646

@torch.no_grad()
def top_p_from_logits(logits, p=0.90):
    """
    Exact nucleus sampling mask from logits.

    logits: [..., V]
    returns:
        indices: [..., V] with rejected = -1
        keep_mask: same shape, True where kept
    """

    # sort descending
    sorted_logits, sorted_idx = torch.sort(logits.float(), descending=True, dim=-1)

    # softmax on sorted logits
    probs = torch.softmax(sorted_logits, dim=-1)

    # cumulative prob
    cum = torch.cumsum(probs, dim=-1)

    # first position where cum >= p
    cutoff = cum >= p

    # find first True per row
    first = cutoff.int().argmax(dim=-1)
    never = ~cutoff.any(dim=-1)
    if never.any():
        first = torch.where(never, logits.new_full(first.shape, logits.size(-1) - 1, dtype=first.dtype), first)

    # build mask: positions <= first
    arange = torch.arange(logits.size(-1), device=logits.device)
    keep = arange <= first.unsqueeze(-1)

    # padded output indices (GPU friendly)
    out_idx = sorted_idx.masked_fill(~keep, -1)

    return out_idx, keep

@torch.no_grad()
def top_k_from_logits(logits, k=1):

    _, top_k = torch.topk(logits, k=k, dim=-1)

    logits_topk = logits.gather(dim=-1, index=top_k)
    keep = logits_topk > float("-inf")
    
    out_idx = top_k.masked_fill(~keep, -1)

    return out_idx, keep


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

def eager_attention_forward_decoding(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    B, H, _, d = query.shape  # q_len = 1 for decoding
    _, _, k_len, _ = key_states.shape

    # -------------------------------------------------------------------------
    # Hierarchical Multi-Level Gist/Sparse Attention (Decoding)
    # -------------------------------------------------------------------------
    
    input_ids = kwargs.get("tmp_ids", None)
    gist_ids = Global_data.gist_token_id
    PAD_ID = Global_data.pad_token_id
    sink_size = Global_data.sink_size
    causal_mask = kwargs.get("causal_mask", None)
    top_k = copy.deepcopy(Global_data.top_k)
    if Global_data.comp_factor > 0:
        comp_factor = Global_data.comp_factor
    else:  # The same comp_factor as the compressed context
        comp_factor = Global_data.chunk_size[0] if len(gist_ids) == 1 else Global_data.chunk_size[0] * Global_data.chunk_size[1]
    top_k[0] = k_len // comp_factor // module.num_key_value_groups // Global_data.chunk_size[0] + 1
    top_k[1] = top_k[0]
    top_p = Global_data.top_p

    device = query.device
    k_idx = torch.arange(k_len, device=device)

    # -------------------------------------------------------------------------
    # Identify gist tokens at all levels (from lowest to highest)
    # -------------------------------------------------------------------------
    num_levels = len(gist_ids)
    is_gist_per_level = []  # [B, k_len] for each level
    for level_idx in range(num_levels):
        gist_m = (input_ids == gist_ids[level_idx])
        is_gist_per_level.append(gist_m)

    # Find the last gist token at any level to determine suffix start
    last_gist = torch.where(is_gist_per_level[-1], k_idx, torch.full_like(k_idx, -1)).amax(dim=1)  # [B]

    # Sink mask on keys (optional)
    start_idx = (input_ids != PAD_ID).int().argmax(dim=1)  # [B]

    # Keys after last gist always visible
    after_last_key = (k_idx[None, :] > last_gist[:, None])  # [B, k]
    compressed = (~after_last_key & (k_idx[None, :] >= start_idx[:, None]))

    # Attention mask processing
    attention_mask_bool = (attention_mask > -1e-4).bool()  # [B, 1, 1, k_len]

    # -------------------------------------------------------------------------
    # Hierarchical Tree Search: Top-k selection from highest to lowest level
    # For decoding, q_len=1, so this is simpler than prefill
    # -------------------------------------------------------------------------
    keep_mask = torch.ones(B, H, 1, k_len, dtype=torch.bool, device=device) & compressed[:, None, None, :]
    all_selected_gist_pos = torch.zeros_like(keep_mask, dtype=torch.bool)
    pos = k_idx[None, None, None, :].expand(B, H, 1, k_len)  # [B, H, 1, k]

    # Start from highest level (last index) and work down to level 0
    for level_idx in range(num_levels - 1, -1, -1):
        is_gist_current = is_gist_per_level[level_idx]  # [B, k_len]
        Gmax_level = int(is_gist_current.sum(dim=1).max().item())
        if Gmax_level == 0:
            # No gist tokens at this level, skip
            continue

        # Build padded gist positions for current level [B, Gmax_level]
        gist_cumsum = is_gist_current.cumsum(dim=1)  # [B, k_len]
        gather_indices = torch.arange(1, Gmax_level + 1, device=device)[None, :].expand(B, Gmax_level)
        matches = (gist_cumsum[:, :, None] == gather_indices[:, None, :])  # [B, k_len, Gmax_level]
        gist_pos_level = matches.to(torch.float32).argmax(dim=1)  # [B, Gmax_level]
        gist_valid_level = gist_pos_level > 0
        gist_valid_level = gist_valid_level[:, None, None, :]  # [B, 1, 1, Gmax_level]

        gist_pos_safe = gist_pos_level.clamp(0, k_len - 1)
        gist_gather = gist_pos_safe[:, None, :, None].expand(B, H, Gmax_level, d)
        gist_pos_level = gist_pos_level[:, None, None, :].expand(B, H, 1, Gmax_level)
        is_gist_current = is_gist_current[:, None, None, :]
        gist_cumsum = gist_cumsum[:, None, None, :]

        # Compute attention scores for this level's gist tokens
        K_gist_level = key_states.gather(dim=2, index=gist_gather)  # [B, H, Gmax_level, d]
        gist_scores_level = torch.matmul(query.double(), K_gist_level.transpose(2, 3).double())  # [B, H, 1, Gmax_level]

        is_gist_current_selected = is_gist_current & keep_mask

        if level_idx < num_levels - 1:
            Gmax_ = int(is_gist_current_selected.sum(dim=-1).max().item())
            pos_masked = pos.masked_fill(~is_gist_current_selected, k_len)  # [B, k]
            pos_sorted = pos_masked.sort(dim=-1).values  # [B, k]
            gist_pos_level_ = pos_sorted[..., :Gmax_]  # [B, Gmax_]
            gist_valid_level_ = gist_pos_level_ < k_len # [B, Gmax_]
            gist_pos_mask = ((gist_pos_safe[:, None, None, :].expand(B, H, 1, Gmax_level).unsqueeze(-2) == gist_pos_level_.unsqueeze(-1)) & gist_valid_level_.unsqueeze(-1)).any(dim=-2)
            gist_valid_level = gist_valid_level.masked_fill(~gist_pos_mask, False)

        gist_scores_level = gist_scores_level.masked_fill(~gist_valid_level, float("-inf"))

        if top_p <= 0:
            kk_level = min(top_k[level_idx], max(1, Gmax_level // 4))
            top_in_gist_level, top_k_mask = top_k_from_logits(gist_scores_level, k=kk_level)  # [B, H, qs_max, kk_level]
        else:
            top_in_gist_level, top_p_mask = top_p_from_logits(gist_scores_level, p=top_p)
        top_in_gist_level_safe = top_in_gist_level.clamp(min=0)
        selected_chunk_mask = torch.zeros((B, H, 1, Gmax_level + 1), device=device, dtype=torch.bool)
        selected_chunk_mask.scatter_(dim=3, index=top_in_gist_level_safe, value=1)
        selected_chunk_mask[..., 0] = (top_in_gist_level == 0).any(dim=3).to(selected_chunk_mask.dtype)
        chunk_ids_parent_adj = torch.where(is_gist_current, gist_cumsum - 1, gist_cumsum)
        chunk_ids_parent_adj = torch.clamp(chunk_ids_parent_adj, min=0)
        chunk_ids_view = chunk_ids_parent_adj.expand(B, H, 1, k_len)
        in_selected_chunks = selected_chunk_mask.gather(dim=3, index=chunk_ids_view).to(torch.bool)

        # Update keep_mask to only include children of selected gist at this level
        keep_mask &= in_selected_chunks
        # all_selected_gist_pos &= ~in_selected_chunks
        all_selected_gist_pos = all_selected_gist_pos | (in_selected_chunks & is_gist_current)

    keep_mask = keep_mask | all_selected_gist_pos

    # -------------------------------------------------------------------------
    # Group-union: union and deduplicate selected gist positions across all levels within each GQA group
    # -------------------------------------------------------------------------
    use_nsa_gqa = getattr(Global_data, 'use_nsa_gqa', False)

    if use_nsa_gqa and hasattr(module, 'num_key_value_groups') and module.num_key_value_groups > 1:
        heads_per_group = module.num_key_value_groups
        num_kv_heads = H // heads_per_group
        keep_mask_grouped = keep_mask.view(B, num_kv_heads, heads_per_group, 1, k_len)

        # For each KV group, take the union of selected positions across all query heads
        # that share the same KV projection
        keep_mask_union = keep_mask_grouped.any(dim=2, keepdim=True)  # [B, num_kv_heads, 1, 1, k_len]

        # Broadcast back to all heads in the group
        keep_mask_union = keep_mask_union.expand(B, num_kv_heads, heads_per_group, 1, k_len)

        # Reshape back to original shape
        keep_mask = keep_mask_union.reshape(B, H, 1, k_len)

    # -------------------------------------------------------------------------
    # Finalize the keep mask with additional constraints
    # -------------------------------------------------------------------------
    gist_token_tensor = torch.tensor(Global_data.gist_token_id, device=device)
    is_gist = torch.isin(input_ids, gist_token_tensor)
    attention_mask_bool = attention_mask_bool & ~(is_gist[:, None, None, :])
    keep_mask = keep_mask | attention_mask_bool  # [B, H, 1, k_len]
    # keep_mask = keep_mask & ~(is_gist[:, None, None, :])

    # ######################################################
    # # Visualization / Logging
    # mean_select_num = (keep_mask & ~after_last_key[:, None, None, :]).sum(-1).float().mean().int().item()
    # max_select_num = (keep_mask & ~after_last_key[:, None, None, :]).sum(-1).max().item()
    # compressed_length = compressed.sum().item()
    # max_ratio = (max_select_num / compressed_length)
    # mean_ratio = (mean_select_num / compressed_length)

    # max_gain = compressed_length // 4 + 30 - max_select_num
    # mean_gain = compressed_length // 4 + 30 - mean_select_num

    # print(
    #     f"max_select_ratio: {max_ratio:.4f}, "
    #     f"mean_select_ratio: {mean_ratio:.4f}, "
    #     f"compressed_length: {compressed_length}, "
    #     f"{'max_gain:'} {max_gain:<4}, "
    #     f"{'mean_gain:'} {mean_gain:<4}"
    # )
    # ######################################################

    # Convert to float attention mask
    attn_mask_float = causal_mask.masked_fill(~keep_mask, float('-inf'))

    attn_output_select = F.scaled_dot_product_attention(
        query, key_states, value_states,
        attn_mask=attn_mask_float,
        dropout_p=dropout if module.training else 0.0,
        is_causal=False,
        scale=scaling,
    )
    attn_output_select = attn_output_select.transpose(1, 2).contiguous()

    return attn_output_select, None


def eager_attention_forward_prefill(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    # 1. Standard projection and QK^T
    # -------------------------------------------------------------------------
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    B, H, q_len, d = query.shape
    _, _, k_len, _ = key_states.shape

    device = query.device
    k_idx = torch.arange(k_len, device=device)

    input_ids = kwargs.get("tmp_ids", None)
    gist_ids = Global_data.gist_token_id
    PAD_ID = Global_data.pad_token_id
    sink_size = Global_data.sink_size
    causal_mask = kwargs.get("causal_mask", None)
    top_k = copy.deepcopy(Global_data.top_k)
    if Global_data.comp_factor > 0:
        comp_factor = Global_data.comp_factor
    else:  # The same comp_factor as the compressed context
        comp_factor = Global_data.chunk_size[0] if len(gist_ids) == 1 else Global_data.chunk_size[0] * Global_data.chunk_size[1]
    top_k[0] = k_len // comp_factor // module.num_key_value_groups // Global_data.chunk_size[0] + 1
    top_k[1] = top_k[0]
    top_p = Global_data.top_p

    # -------------------------------------------------------------------------
    # PATH A: Flash Attention (Standard)
    # -------------------------------------------------------------------------
    attn_output_o = F.scaled_dot_product_attention(
        query, key_states, value_states,
        attn_mask=attention_mask,
        dropout_p=dropout if module.training else 0.0,
        scale=scaling
    )
    out0 = attn_output_o.transpose(1, 2).contiguous()

    # -------------------------------------------------------------------------
    # PATH B: Hierarchical Multi-Level Gist/Sparse Attention
    # -------------------------------------------------------------------------

    # Identify gist tokens at all levels (from lowest to highest)
    num_levels = len(gist_ids)
    is_gist_per_level = []  # [B, k_len] for each level
    for level_idx in range(num_levels):
        gist_m = (input_ids == gist_ids[level_idx])
        assert gist_m.any(), f"No gist tokens found for level {level_idx} with gist id {gist_ids[level_idx]}"
        is_gist_per_level.append(gist_m)

    # Find the last gist token at any level to determine suffix start
    last_gist = torch.where(is_gist_per_level[-1], k_idx, torch.full_like(k_idx, -1)).amax(dim=1)  # [B]

    # suffix starts at last_gist+1
    q_start = last_gist + 1                # [B]
    qs_len  = (q_len - q_start)            # [B]
    qs_max = int(qs_len.max().item())

    # -----------------------
    # Build padded suffix query positions [B,qs_max]
    # -----------------------
    ar = torch.arange(qs_max, device=device)                   # [qs_max]
    q_pos = q_start[:, None] + ar[None, :]                     # [B,qs_max]
    valid_q = q_pos < q_len                                    # [B,qs_max]
    q_pos_clamped = q_pos.clamp(max=q_len - 1)                 # [B,qs_max]

    # Gather suffix queries and dense outputs: [B,H,qs_max,d]
    q_gather = q_pos_clamped[:, None, :, None].expand(B, H, qs_max, d)
    q_gather_ = q_pos_clamped[:, None, :, None].expand(B, 1, qs_max, k_len)
    q_suffix = query.gather(2, q_gather)
    # zero padded suffix queries
    q_suffix = torch.where(valid_q[:, None, :, None], q_suffix, torch.zeros_like(q_suffix))

    assert causal_mask is not None, "Causal mask is required for prefill."
    assert causal_mask.shape[1] == 1
    cm_suffix = (causal_mask.gather(2, q_gather_) > -1e-4)
    assert causal_mask.shape[1] == 1
    assert causal_mask.dim() == 4

    # Sink mask on keys (optional)
    start_idx = (input_ids != PAD_ID).int().argmax(dim=1)  # [B]
    sink_keep = None
    if sink_size > 0:
        sink_keep = (k_idx[None, :] >= start_idx[:, None]) & (k_idx[None, :] < (start_idx + sink_size)[:, None])  # [B,k]

    # Keys after last gist always visible in your original logic
    after_last_key = (k_idx[None, :] > last_gist[:, None])    # [B,k]
    compressed = (~after_last_key & (k_idx[None, :] >= start_idx[:, None]))

    attention_mask_bool = (attention_mask > -1e-4).bool().gather(2, q_gather_)
    attention_mask_suffix = torch.where(valid_q[:, None, :, None], attention_mask_bool, torch.zeros_like(attention_mask_bool))

    # Create mask: set all items before (and including) the last 1 up to after_last_key position to 1
    # Optimization: Use flip + argmax to find last occurrence in one pass (avoids cumsum)
    mask_before_last_key = attention_mask_suffix & ~after_last_key[:, None, None, :]  # Only consider positions before/at after_last_key

    # Flip the tensor along last dimension, find first 1 (which is last in original), then flip index back
    flipped_mask = mask_before_last_key.flip(dims=[-1])
    first_one_in_flipped = flipped_mask.to(torch.float32).argmax(dim=-1, keepdim=True)  # [B, H, qs_max, 1]
    has_any_one = flipped_mask.any(dim=-1, keepdim=True)
    last_one_pos = torch.where(has_any_one, k_len - 1 - first_one_in_flipped, torch.zeros_like(first_one_in_flipped))

    # Set all positions <= last_one_pos to 1, keep original values after
    k_positions = torch.arange(k_len, device=device)[None, None, None, :]  # [1, 1, 1, k_len]
    link_section_mask = (k_positions <= last_one_pos) | attention_mask_suffix

    # -------------------------------------------------------------------------
    # Hierarchical Tree Search: Top-k selection from highest to lowest level
    # -------------------------------------------------------------------------
    # Strategy: Start from highest gist level, select top-k, then for each selected
    # gist, only consider its children at the next lower level, and so on.

    # Track which positions to keep at each level (start with all positions valid)
    keep_mask = torch.ones(B, H, qs_max, k_len, dtype=torch.bool, device=device) & compressed[:, None, None, :] & link_section_mask
    all_selected_gist_pos = torch.zeros_like(keep_mask, dtype=torch.bool)
    pos = k_idx[None, None, None, :].expand(B, H, qs_max, k_len)  # [B, k]

    # Start from highest level (last index) and work down to level 0
    for level_idx in range(num_levels - 1, -1, -1):
        is_gist_current = is_gist_per_level[level_idx]  # [B, k_len]
        Gmax_level = int(is_gist_current.sum(dim=1).max().item())
        if Gmax_level == 0:
            # No gist tokens at this level, skip
            continue

        # Build padded gist positions for current level [B, Gmax_level]
        # Efficiently extract gist positions without sorting using cumsum-based indexing
        # Create running count of gist tokens
        gist_cumsum = is_gist_current.cumsum(dim=1)  # [B, k_len]

        # Create indices [0, 1, 2, ..., Gmax_level-1] for gathering
        gather_indices = torch.arange(1, Gmax_level + 1, device=device)[None, :].expand(B, Gmax_level)  # [B, Gmax_level]

        # For each batch and each target index, find the first position where cumsum equals it
        # Use broadcasting: compare cumsum[B, k_len, 1] with gather_indices[B, 1, Gmax_level]
        matches = (gist_cumsum[:, :, None] == gather_indices[:, None, :])  # [B, k_len, Gmax_level]

        # For each target index, find the first matching position (argmax finds first True)
        gist_pos_level = matches.to(torch.float32).argmax(dim=1)  # [B, Gmax_level]

        # Validate positions
        # gist_valid_level = gist_cumsum.gather(dim=1, index=gist_pos_level) == gather_indices  # [B, Gmax_level]
        gist_valid_level = gist_pos_level > 0
        gist_valid_level = gist_valid_level[:, None, None, :]

        gist_pos_safe = gist_pos_level.clamp(0, k_len - 1)
        gist_gather = gist_pos_safe[:, None, :, None].expand(B, H, Gmax_level, d)
        gist_pos_level = gist_pos_safe[:, None, None, :].expand(B, H, qs_max, Gmax_level)
        is_gist_current = is_gist_current[:, None, None, :]
        gist_cumsum = gist_cumsum[:, None, None, :]

        K_gist_level = key_states.gather(dim=2, index=gist_gather)  # [B, H, Gmax_level, d]
        # Compute attention scores for this level's gist tokens
        gist_scores_level = torch.matmul(q_suffix.double(), K_gist_level.transpose(2, 3).double())  # [B, H, qs_max, Gmax_level]
        link_section_mask_level = link_section_mask.gather(dim=3, index=gist_pos_level[:, 0:1])  # [B, 1, qs_max, Gmax_level]
        gist_scores_level = gist_scores_level.masked_fill(~link_section_mask_level, float("-inf"))
        is_gist_current_selected = is_gist_current & keep_mask
        
        if level_idx < num_levels - 1:
            Gmax_ = int(is_gist_current_selected.sum(dim=-1).max().item())
            pos_masked = pos.masked_fill(~is_gist_current_selected, k_len)  # [B, k]
            pos_sorted = pos_masked.sort(dim=-1).values  # [B, k]
            gist_pos_level_ = pos_sorted[..., :Gmax_]  # [B, Gmax_]
            gist_valid_level_ = gist_pos_level_ < k_len # [B, Gmax_]
            gist_pos_mask = ((gist_pos_safe[:, None, None, :].expand(B, H, qs_max, Gmax_level).unsqueeze(-2) == gist_pos_level_.unsqueeze(-1)) & gist_valid_level_.unsqueeze(-1)).any(dim=-2)
            gist_valid_level = gist_valid_level.masked_fill(~gist_pos_mask, False)
            
        gist_scores_level = gist_scores_level.masked_fill(~gist_valid_level, float("-inf"))
        gist_scores_level = gist_scores_level.masked_fill(~valid_q[:, None, :, None], float("-inf"))

        # Select top-k at this level
        if top_p <= 0:
            kk_level = min(top_k[level_idx], max(1, Gmax_level // 4))
            top_in_gist_level, top_k_mask = top_k_from_logits(gist_scores_level, k=kk_level)  # [B, H, qs_max, kk_level]
        else:
            top_in_gist_level, top_p_mask = top_p_from_logits(gist_scores_level, p=top_p)
        top_in_gist_level_safe = top_in_gist_level.clamp(min=0)
        selected_chunk_mask = torch.zeros((B, H, qs_max, Gmax_level + 1), device=device, dtype=torch.bool)
        selected_chunk_mask.scatter_(dim=3, index=top_in_gist_level_safe, value=1)
        selected_chunk_mask[..., 0] = (top_in_gist_level == 0).any(dim=3).to(selected_chunk_mask.dtype)
        chunk_ids_parent_adj = torch.where(is_gist_current, gist_cumsum - 1, gist_cumsum)
        chunk_ids_parent_adj = torch.clamp(chunk_ids_parent_adj, min=0)
        chunk_ids_view = chunk_ids_parent_adj.expand(B, H, qs_max, k_len)
        in_selected_chunks = selected_chunk_mask.gather(dim=3, index=chunk_ids_view).to(torch.bool)

        # Update keep_mask to only include children of selected gist at this level
        keep_mask &= in_selected_chunks
        # all_selected_gist_pos &= ~in_selected_chunks
        all_selected_gist_pos = all_selected_gist_pos | (in_selected_chunks & is_gist_current)

    keep_mask = keep_mask | all_selected_gist_pos

    # assert (keep_mask_ == keep_mask).all(), "Mismatch in unselected gist combination."

    # -------------------------------------------------------------------------
    # Group-union: union and deduplicate selected gist positions across all levels within each GQA group
    # -------------------------------------------------------------------------
    use_nsa_gqa = getattr(Global_data, 'use_nsa_gqa', False)

    if use_nsa_gqa and hasattr(module, 'num_key_value_groups') and module.num_key_value_groups > 1:
        heads_per_group = module.num_key_value_groups
        num_kv_heads = H // heads_per_group
        keep_mask_grouped = keep_mask.view(B, num_kv_heads, heads_per_group, qs_max, k_len)

        # For each KV group, take the union of selected positions across all query heads
        # that share the same KV projection
        keep_mask_union = keep_mask_grouped.any(dim=2, keepdim=True)  # [B, num_kv_heads, 1, qs_max, k_len]

        # Broadcast back to all heads in the group
        keep_mask_union = keep_mask_union.expand(B, num_kv_heads, heads_per_group, qs_max, k_len)

        # Reshape back to original shape
        keep_mask = keep_mask_union.reshape(B, H, qs_max, k_len)

    # -------------------------------------------------------------------------
    # Finalize the keep mask with additional constraints
    # -------------------------------------------------------------------------
    gist_token_tensor = torch.tensor(Global_data.gist_token_id, device=device)
    is_gist = torch.isin(input_ids, gist_token_tensor)
    attention_mask_suffix &= ~(is_gist[:, None, None, :])
    keep_mask = keep_mask | attention_mask_suffix  # [B, H, qs_max, k_len]
    # keep_mask = keep_mask & ~(is_gist[:, None, None, :])

    # Add sink tokens
    if sink_keep is not None:
        keep_mask = keep_mask | sink_keep.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, k_len]

    # Add tokens after last gist (always visible)
    keep_mask = keep_mask | after_last_key.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, k_len]

    # Only apply this keep-mask to valid suffix query slots
    keep_mask = keep_mask & valid_q.unsqueeze(1).unsqueeze(-1)  # [B, 1, qs_max, 1]

    ######################################################
    # Visualization / Logging
    if Global_data.link_token_id is not None:
        num_link = (input_ids==Global_data.link_token_id[0]).sum().item()
        start_p = len(Global_data.link_token_id)*num_link
    else:
        start_p = 0
    mean_select_num = (keep_mask & ~after_last_key[:, None, None, :])[:, :, start_p:].sum(-1).float().mean().int()
    max_select_num = (keep_mask & ~after_last_key[:, None, None, :])[:, :, start_p:].sum(-1).max()
    compressed_length = q_start.item()
    max_ratio = (max_select_num / q_start.float()).item()
    mean_ratio = (mean_select_num / q_start.float()).item()

    max_gain = (q_start // comp_factor - max_select_num).item()
    mean_gain = (q_start // comp_factor - mean_select_num).item()

    print(
        f"max_select_ratio: {max_ratio:.4f}, "
        f"mean_select_ratio: {mean_ratio:.4f}, "
        f"compressed_length: {compressed_length}, "
        f"{'max_gain:'} {max_gain:<4}, "
        f"{'mean_gain:'} {mean_gain:<4}"
    )
    ######################################################

    # Convert to float attention mask
    attn_mask = cm_suffix & keep_mask

    # -------------------------------------------------------------------------
    # Selective SDPA on suffix queries only
    # -------------------------------------------------------------------------
    sel_suffix = F.scaled_dot_product_attention(
        q_suffix, key_states, value_states,
        attn_mask=attn_mask,   # bool mask, no dense float mask
        dropout_p=dropout if module.training else 0.0,
        is_causal=False,
        scale=scaling,
    )  # [B, H, qs_max, d]

    # -----------------------
    # Scatter blended suffix back into dense output (prefix stays dense exactly)
    # -----------------------
    src = sel_suffix.transpose(1, 2).contiguous()             # [B,qs,H,d]

    b_flat, t_flat = valid_q.nonzero(as_tuple=True)   # both shape [N]
    q_flat = q_pos[b_flat, t_flat]                   # [N]
    src_flat = src[b_flat, t_flat]                   # [N,H,d]
    out = out0.index_put((b_flat, q_flat), src_flat, accumulate=False)

    return out.contiguous(), None


def eager_attention_forward_prefill_chunk(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    chunk_q: int = 512,
    **kwargs: Unpack[TransformersKwargs],
):
    # 1. Standard projection and QK^T
    # -------------------------------------------------------------------------
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    B, H, q_len, d = query.shape
    _, _, k_len, _ = key_states.shape

    device = query.device
    k_idx = torch.arange(k_len, device=device)

    input_ids = kwargs.get("tmp_ids", None)
    gist_ids = Global_data.gist_token_id
    PAD_ID = Global_data.pad_token_id
    sink_size = Global_data.sink_size
    causal_mask = kwargs.get("causal_mask", None)
    top_k = copy.deepcopy(Global_data.top_k)
    if Global_data.comp_factor > 0:
        comp_factor = Global_data.comp_factor
    else:  # The same comp_factor as the compressed context
        comp_factor = Global_data.chunk_size[0] if len(gist_ids) == 1 else Global_data.chunk_size[0] * Global_data.chunk_size[1]
    top_k[0] = k_len // comp_factor // module.num_key_value_groups // Global_data.chunk_size[0] + 1
    top_k[1] = top_k[0]
    top_p = Global_data.top_p

    out = torch.empty(B, q_len, H, d, device=query.device, dtype=query.dtype)

    # Identify gist tokens at all levels (from lowest to highest)
    num_levels = len(gist_ids)
    is_gist_per_level = []  # [B, k_len] for each level
    for level_idx in range(num_levels):
        gist_m = (input_ids == gist_ids[level_idx])
        assert gist_m.any(), f"No gist tokens found for level {level_idx} with gist id {gist_ids[level_idx]}"
        is_gist_per_level.append(gist_m)

    # Find the last gist token at any level to determine suffix start
    last_gist = torch.where(is_gist_per_level[-1], k_idx, torch.full_like(k_idx, -1)).amax(dim=1)  # [B]

    # suffix starts at last_gist+1
    q_start = last_gist + 1                # [B]
    qs_len  = (q_len - q_start)            # [B]
    qs_max = int(qs_len.max().item())

    # -----------------------
    # Build padded suffix query positions [B,qs_max]
    # -----------------------
    ar = torch.arange(qs_max, device=device)                   # [qs_max]
    q_pos = q_start[:, None] + ar[None, :]                     # [B,qs_max]
    valid_q = q_pos < q_len                                    # [B,qs_max]
    q_pos_clamped = q_pos.clamp(max=q_len - 1)                 # [B,qs_max]

    # Gather suffix queries and dense outputs: [B,H,qs_max,d]
    q_gather = q_pos_clamped[:, None, :, None].expand(B, H, qs_max, d)
    q_gather_ = q_pos_clamped[:, None, :, None].expand(B, 1, qs_max, k_len)
    q_suffix = query.gather(2, q_gather)
    # zero padded suffix queries
    q_suffix = torch.where(valid_q[:, None, :, None], q_suffix, torch.zeros_like(q_suffix))

    assert causal_mask is not None, "Causal mask is required for prefill."
    assert causal_mask.shape[1] == 1
    cm_suffix = (causal_mask.gather(2, q_gather_) > -1e-4)
    assert causal_mask.shape[1] == 1
    assert causal_mask.dim() == 4

    # Sink mask on keys (optional)
    start_idx = (input_ids != PAD_ID).int().argmax(dim=1)  # [B]
    sink_keep = None
    if sink_size > 0:
        sink_keep = (k_idx[None, :] >= start_idx[:, None]) & (k_idx[None, :] < (start_idx + sink_size)[:, None])  # [B,k]

    # Keys after last gist always visible in your original logic
    after_last_key = (k_idx[None, :] > last_gist[:, None])    # [B,k]
    compressed = (~after_last_key & (k_idx[None, :] >= start_idx[:, None]))

    # -------------------------------------------------------------------------
    # PATH A: Flash Attention (Standard)
    # -------------------------------------------------------------------------
    for b in range(B):
        qs = q_start[b].item()
        if qs > 0:
            out_b = F.scaled_dot_product_attention(
                query[b:b+1, :, :qs, :],
                key_states[b:b+1],
                value_states[b:b+1],
                attn_mask=attention_mask[b:b+1, :, :qs, :] if attention_mask is not None else None,
                dropout_p=dropout if module.training else 0.0,
                scale=scaling
            )
            out[b:b+1, :qs] = out_b.transpose(1, 2)

    # -------------------------------------------------------------------------
    # Hierarchical Tree Search: Top-k selection from highest to lowest level
    # -------------------------------------------------------------------------
    # Strategy: Start from highest gist level, select top-k, then for each selected
    # gist, only consider its children at the next lower level, and so on.
    gist_token_tensor = torch.tensor(Global_data.gist_token_id, device=device)
    is_gist = torch.isin(input_ids, gist_token_tensor)

    level_meta = []
    for level_idx in range(num_levels):
        is_gist_current = is_gist_per_level[level_idx]  # [B, k_len]
        counts = is_gist_current.sum(dim=1)  # [B]
        Gmax = int(counts.max().item())
        if Gmax == 0:
            level_meta.append(None)
            continue

        gist_cumsum = is_gist_current.cumsum(dim=1)  # [B, k_len]

        gist_pos = torch.full((B, Gmax), 0, device=device, dtype=torch.long)
        gist_valid = torch.arange(Gmax, device=device)[None, :] < counts[:, None]
        gist_valid = gist_valid[:, None, None, :]

        # Fill positions without O(B*k*G) broadcast
        nz = is_gist_current.nonzero(as_tuple=False)  # [N, 2] -> (b, pos)
        rank = torch.cumsum(is_gist_current, dim=1)[nz[:, 0], nz[:, 1]] - 1
        gist_pos[nz[:, 0], rank] = nz[:, 1]

        gist_gather = gist_pos[:, None, :, None].expand(B, H, Gmax, d)
        K_gist = key_states.gather(dim=2, index=gist_gather)           # [B,H,Gmax,d]

        chunk_ids = torch.where(is_gist_current, gist_cumsum - 1, gist_cumsum).clamp_min(0)

        level_meta.append({
            "is_gist": is_gist_current,         # [B,k]
            "gist_pos": gist_pos,               # [B,Gmax]
            "gist_valid": gist_valid,           # [B,Gmax]
            "K_gist": K_gist,                   # [B,H,Gmax,d]
            "chunk_ids": chunk_ids,             # [B,k]
            "Gmax": Gmax,
            "gist_cumsum": gist_cumsum,         # [B,1,1,k]
        })

    for level_idx in range(num_levels - 2, -1, -1):
        lower = level_meta[level_idx]
        upper = level_meta[level_idx + 1]
        if lower is None or upper is None:
            continue

        lower_gist_pos = lower["gist_pos"]  # [B, G_lower]
        parent_chunk_ids = upper["chunk_ids"].gather(dim=1, index=lower_gist_pos)  # [B, G_lower]

        lower["parent_chunk_ids_from_upper"] = parent_chunk_ids  # [B, G_lower], values in [0, G_upper]


    # 3) suffix selective in chunks
    for s in range(0, qs_max, chunk_q):
        e = min(s + chunk_q, qs_max)
        qb = e - s

        q_pos_blk = q_pos_clamped[:, s:e]              # [B, qb]
        valid_q_blk = valid_q[:, s:e]                  # [B, qb]

        q_gather = q_pos_blk[:, None, :, None].expand(B, H, qb, d)
        q_suffix_blk = query.gather(2, q_gather)
        q_suffix_blk = torch.where(
            valid_q_blk[:, None, :, None],
            q_suffix_blk,
            torch.zeros_like(q_suffix_blk),
        )

        q_gather_blk = q_pos_blk[:, None, :, None].expand(B, 1, qb, k_len)
        attention_mask_bool_blk = (attention_mask > -1e-4).bool().gather(2, q_gather_blk)
        attention_mask_suffix_blk = torch.where(
            valid_q_blk[:, None, :, None],
            attention_mask_bool_blk,
            torch.zeros_like(attention_mask_bool_blk),
        )

        # direct causal mask
        cm_blk = cm_suffix[:, :, s:e] # [B, 1, qb, k_len]
        cm_blk = torch.where(
            valid_q_blk[:, None, :, None],
            cm_blk,
            torch.zeros_like(cm_blk),
        )

        # Track which positions to keep at each level (start with all positions valid)
        keep_mask = torch.ones(B, H, qb, k_len, dtype=torch.bool, device=device) & compressed[:, None, None, :]
        all_selected_gist_pos = torch.zeros_like(keep_mask, dtype=torch.bool)

        selected_parent_chunks = None
        selected_gist_records = []
        selected_level0_chunks = None

        # Start from highest level (last index) and work down to level 0
        for level_idx in range(num_levels - 1, -1, -1):
            cur_level_meta = level_meta[level_idx]
            if cur_level_meta is None:
                continue
            is_gist_current = cur_level_meta['is_gist'][:, None, None, :]  # [B, 1, 1, k_len]
            Gmax_level = cur_level_meta['Gmax']
            gist_pos_safe = cur_level_meta['gist_pos'].clamp(0, k_len - 1)
            K_gist_level = cur_level_meta['K_gist']
            gist_valid_level = cur_level_meta['gist_valid']

            # Compute attention scores for this level's gist tokens
            gist_scores_level = torch.matmul(q_suffix_blk.double(), K_gist_level.transpose(2, 3).double())  # [B, H, qs_max, Gmax_level]
            
            if selected_parent_chunks is not None:
                parent_chunk_ids = cur_level_meta["parent_chunk_ids_from_upper"]   # [B,G]
                parent_chunk_ids_exp = parent_chunk_ids[:, None, None, :].expand(B, H, qb, Gmax_level)
                child_valid = selected_parent_chunks.gather(dim=3, index=parent_chunk_ids_exp)
                gist_scores_level = gist_scores_level.masked_fill(~child_valid, float("-inf"))
               
            gist_scores_level = gist_scores_level.masked_fill(~gist_valid_level, float("-inf"))
            gist_scores_level = gist_scores_level.masked_fill(~valid_q_blk[:, None, :, None], float("-inf"))

            # Select top-k at this level
            if top_p <= 0:
                kk_level = min(top_k[level_idx], max(1, Gmax_level // 4))
                top_in_gist_level, top_k_mask = top_k_from_logits(gist_scores_level, k=kk_level)  # [B, H, qb, kk_level]
            else:
                top_in_gist_level, top_p_mask = top_p_from_logits(gist_scores_level, p=top_p)

            top_idx_safe = top_in_gist_level.clamp(min=0)
            selected_gists = torch.zeros((B, H, qb, Gmax_level), dtype=torch.bool, device=device)
            selected_gists.scatter_(dim=3, index=top_idx_safe, value=True)
            selected_gists &= gist_valid_level
            selected_gists &= torch.isfinite(gist_scores_level)

            selected_gist_records.append((gist_pos_safe, selected_gists))
            
            selected_parent_chunks = torch.zeros((B, H, qb, Gmax_level + 1), dtype=torch.bool, device=device)
            selected_parent_chunks[..., :Gmax_level] = selected_gists

            if level_idx == 0:
                selected_level0_chunks = selected_parent_chunks

        # Materialize descendants only once
        if selected_level0_chunks is not None:
            chunk_ids_level0 = level_meta[0]["chunk_ids"]  # [B,k_len]
            idx = chunk_ids_level0[:, None, None, :].expand(B, H, qb, k_len)
            compressed_descendants = selected_level0_chunks.gather(dim=3, index=idx)
            compressed_descendants &= compressed[:, None, None, :]
            compressed_descendants &= ~(is_gist[:, None, None, :])
        else:
            compressed_descendants = torch.zeros((B, H, qb, k_len), dtype=torch.bool, device=device)

        # Materialize selected gist positions only once
        all_selected_gist_pos = torch.zeros((B, H, qb, k_len), dtype=torch.bool, device=device)
        for gist_pos, selected_gists in selected_gist_records:
            G = gist_pos.shape[1]
            idx = gist_pos[:, None, None, :].expand(B, H, qb, G)
            tmp = torch.zeros_like(all_selected_gist_pos)
            tmp.scatter_(dim=3, index=idx, src=selected_gists)
            all_selected_gist_pos |= tmp

        keep_mask = compressed_descendants | all_selected_gist_pos

        # -------------------------------------------------------------------------
        # Group-union: union and deduplicate selected gist positions across all levels within each GQA group
        # -------------------------------------------------------------------------
        use_nsa_gqa = getattr(Global_data, 'use_nsa_gqa', False)

        if use_nsa_gqa and hasattr(module, 'num_key_value_groups') and module.num_key_value_groups > 1:
            heads_per_group = module.num_key_value_groups
            num_kv_heads = H // heads_per_group
            keep_mask_grouped = keep_mask.view(B, num_kv_heads, heads_per_group, qb, k_len)

            # For each KV group, take the union of selected positions across all query heads
            # that share the same KV projection
            keep_mask_union = keep_mask_grouped.any(dim=2, keepdim=True)  # [B, num_kv_heads, 1, qs_max, k_len]

            # Broadcast back to all heads in the group
            keep_mask_union = keep_mask_union.expand(B, num_kv_heads, heads_per_group, qb, k_len)

            # Reshape back to original shape
            keep_mask = keep_mask_union.reshape(B, H, qb, k_len)

        # -------------------------------------------------------------------------
        # Finalize the keep mask with additional constraints
        # -------------------------------------------------------------------------
        attention_mask_suffix_blk &= ~(is_gist[:, None, None, :])
        keep_mask = keep_mask | attention_mask_suffix_blk  # [B, H, qs_max, k_len]
        # keep_mask = keep_mask & ~(is_gist[:, None, None, :])

        # Add sink tokens
        if sink_keep is not None:
            keep_mask = keep_mask | sink_keep.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, k_len]

        # Add tokens after last gist (always visible)
        keep_mask = keep_mask | after_last_key.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, k_len]

        # Only apply this keep-mask to valid suffix query slots
        keep_mask = keep_mask & valid_q_blk.unsqueeze(1).unsqueeze(-1)  # [B, 1, qs_max, 1]

        ######################################################
        # # Visualization / Logging
        # if Global_data.link_token_id is not None:
        #     num_link = (input_ids==Global_data.link_token_id[0]).sum().item()
        #     start_p = len(Global_data.link_token_id)*num_link
        # else:
        #     start_p = 0
        # mean_select_num = (keep_mask & ~after_last_key[:, None, None, :])[:, :, start_p:].sum(-1).float().mean().int()
        # max_select_num = (keep_mask & ~after_last_key[:, None, None, :])[:, :, start_p:].sum(-1).max()
        # compressed_length = q_start.item()
        # max_ratio = (max_select_num / q_start.float()).item()
        # mean_ratio = (mean_select_num / q_start.float()).item()

        # max_gain = (q_start // 4 + 30 - max_select_num).item()
        # mean_gain = (q_start // 4 + 30 - mean_select_num).item()

        # print(
        #     f"max_select_ratio: {max_ratio:.4f}, "
        #     f"mean_select_ratio: {mean_ratio:.4f}, "
        #     f"compressed_length: {compressed_length}, "
        #     f"{'max_gain:'} {max_gain:<4}, "
        #     f"{'mean_gain:'} {mean_gain:<4}"
        # )
        ######################################################

        # Convert to float attention mask
        attn_mask = cm_blk & keep_mask

        # print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
        # print("Reserved :", torch.cuda.memory_reserved() / 1024**2, "MB")

        # -------------------------------------------------------------------------
        # Selective SDPA on suffix queries only
        # -------------------------------------------------------------------------
        sel_suffix = F.scaled_dot_product_attention(
            q_suffix_blk, key_states, value_states,
            attn_mask=attn_mask,
            dropout_p=dropout if module.training else 0.0,
            is_causal=False,
            scale=scaling,
        )  # [B, H, qs_max, d]

        # -----------------------
        # Scatter blended suffix back into dense output (prefix stays dense exactly)
        # -----------------------
        src = sel_suffix.transpose(1, 2).contiguous()             # [B,qs,H,d]

        b_flat, t_flat = valid_q_blk.nonzero(as_tuple=True)   # both shape [N]
        q_flat = q_pos_blk[b_flat, t_flat]                   # [N]
        out[b_flat, q_flat] = src[b_flat, t_flat]

    return out.contiguous(), None


class GistQwen2Attention(Qwen2Attention):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def select_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        decode: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # === attention dispatch ====================================================
        # Single hook: training -> eager reference; inference -> sparse path
        # (`sparse_attention_forward`). `ATTN_IMPL=ref` forces eager. The
        # `eager_attention_forward_*` bodies must stay byte-identical.
        from attn_candidate import attn_dispatcher as _attn_dispatcher
        attn_output, attn_weights = _attn_dispatcher(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            decode=decode,
            chunk_q=512,
            **kwargs,
        )
        # ==========================================================================

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class GistQwen2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GistQwen2Attention(config=config, layer_idx=layer_idx)
        self.layer_idx = layer_idx

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        selective: int = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        
        if self.layer_idx < 1:
            # # For the first layer, use standard attention for stability
            # if selective == 1:
            #     attention_mask_gist = kwargs.get("attention_mask_gist", None)
            #     causal_mask = kwargs.get("causal_mask", None)
            #     input_ids = kwargs.get("tmp_ids", None)
            #     seq_len = input_ids.shape[1]
            #     # Find last meta-gist position for each batch element
            #     positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)  # [1, S]
            #     is_gist = torch.isin(input_ids, torch.tensor(Global_data.gist_token_id, device=input_ids.device))  # [B, S]
            #     gist_positions = torch.where(is_gist, positions, torch.tensor(-1, device=input_ids.device))
            #     last_gist_pos = gist_positions.max(dim=1)[0]  # [B]
            #     # Create mask for positions to copy (before and including last meta-gist token)
            #     # mask shape: [B, 1, S, 1] for broadcasting
            #     change_mask = (positions[:, None, :, None] > last_gist_pos[:, None, None, None])  # [B, 1, S, 1]
            #     change_mask_l0 = change_mask & is_gist[:, None, None, :]
            #     attention_mask_gist = attention_mask_gist.masked_fill(change_mask, True).masked_fill(change_mask_l0, False)

            #     attention_mask_gist_float = torch.full(
            #         attention_mask.shape,  # Use the 4D shape [8, 1, 172, 172]
            #         torch.finfo(attention_mask.dtype).min,
            #         dtype=attention_mask.dtype,
            #         device=attention_mask.device
            #     )
            #     attention_mask_gist_float = attention_mask_gist_float.masked_fill(
            #         attention_mask_gist.bool(), 0.0
            #     )
            #     attention_mask = causal_mask + attention_mask_gist_float

            selective = 0
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        if selective == 1:
            hidden_states, _ = self.self_attn.select_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        elif selective == 2:
            hidden_states, _ = self.self_attn.select_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                decode=True,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class GistQwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GistQwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_gist: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        gist_activations: Optional[torch.Tensor] = None,
        gist_offset: Optional[torch.LongTensor] = None,
        selective: int = 0,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ):

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # Decide whether to collect hidden states
        if output_hidden_states is None:
            output_hidden_states = getattr(self.config, "output_hidden_states", False)

        all_hidden_states = () if output_hidden_states else None

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if gist_activations is not None:
            if past_key_values is not None or gist_offset is not None:
                raise ValueError(
                    "You should pass in either gist_activations alone, or "
                    "past_key_values and gist_offset, not both."
                )
            past_key_values = gist_activations.past_key_values
            gist_offset = gist_activations.gist_indices

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values.get_seq_length()

        causal_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        attention_mask_gist_float = torch.full(
            attention_mask_gist.shape,  # Use the 4D shape [8, 1, 172, 172]
            torch.finfo(self.dtype).min,
            dtype=self.dtype,
            device=attention_mask_gist.device
        )
        attention_mask_gist_float = attention_mask_gist_float.masked_fill(
            attention_mask_gist.bool(), 0.0
        )

        _attention_mask = causal_mask + attention_mask_gist_float

        kwargs["causal_mask"] = causal_mask
        kwargs["attention_mask_gist"] = attention_mask_gist

        hidden_states = inputs_embeds

        if gist_offset is not None:
            # Current generation path stores one packed gist offset per batch.
            offset = gist_offset[0].item()
        else:
            offset = 0

        pos = position_ids + offset
        if input_ids.shape[1] == 1:
            rope_pos = Global_data.last_pos + 1
            Global_data.last_pos = rope_pos
        else:
            # Build gist mask (all levels)
            gist_token_tensor = torch.tensor(Global_data.gist_token_id, device=pos.device)
            gist_mask = torch.isin(input_ids, gist_token_tensor)  # [B, T] bool
            rope_shift = torch.cumsum(gist_mask.to(torch.long), dim=1)
            rope_pos = pos - rope_shift
            Global_data.last_pos = rope_pos[:, -1:]
        position_embeddings = self.rotary_emb(hidden_states, rope_pos)
        # position_embeddings = self.rotary_emb(hidden_states, position_ids + offset)

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                selective=selective,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )

def masked_kl_vocab(teacher_logits, student_logits, mask, temperature=1.0):
    """
    KL( teacher || student ) averaged over token positions where mask==1.
    teacher_logits, student_logits: [B, t_keep, V]
    mask: [B, t_keep] (bool or 0/1)
    """
    # teacher probs
    p_t = F.softmax(teacher_logits / temperature, dim=-1)
    # student log-probs
    logp_s = F.log_softmax(student_logits / temperature, dim=-1)

    # tokenwise KL: [B, t_keep]
    kl_tok = F.kl_div(logp_s, p_t, reduction="none").sum(dim=-1)

    mask_f = mask.float()
    denom = mask_f.sum().clamp_min(1.0)
    kl = (kl_tok * mask_f).sum() / denom

    # standard distillation scaling
    return kl * (temperature ** 2)


def masked_smoothl1_vocab(h1, h2, mask):
    loss = F.smooth_l1_loss(h1, h2, reduction="none").sum(dim=-1)

    mask_f = mask.float()
    denom = mask_f.sum().clamp_min(1.0)
    smooth_l1 = (loss * mask_f).sum() / denom

    # standard distillation scaling
    return smooth_l1


class GistQwen2ForCausalLM(Qwen2PreTrainedModel, GistGenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = GistQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.gist_idx = 0
        self.selective = config.selective if hasattr(config, 'selective') else False
        self.tmp_ids = None

        # Initialize weights and apply final processing
        self.post_init()

    @torch.no_grad()
    def get_gist_activations(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        attention_mask_gist: torch.FloatTensor,
        gist_token: int,
        num_gist_tokens: int,
        cache_all: bool = False,
    ) -> GistActivations:
        model_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            attention_mask_gist=attention_mask_gist,
            output_hidden_states=True,
            use_cache=True,
        )
        return GistActivations.from_model_outputs(
            model_outputs=model_outputs,
            input_ids=input_ids,
            gist_token=gist_token,
            num_gist_tokens=num_gist_tokens,
            cache_all=cache_all,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        gist_activations: Optional[GistActivations] = None,
        gist_offset: Optional[torch.LongTensor] = None,
        selective_inference: Optional[bool] = True,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        
        self.tmp_ids = input_ids if input_ids.shape[1] > 1 else torch.cat([self.tmp_ids, input_ids], dim=1)
        kwargs['tmp_ids'] = self.tmp_ids

        if len(kwargs['attention_mask_gist'].shape) == 4:

            if self.training is False and self.selective:
                if not selective_inference:
                    selective = 0
                elif input_ids.shape[1] == 1:  # selective inference in the decoding stage
                    selective = 2
                else:                        # selective inference in the prefill stage
                    selective = 1
                
                outputs: BaseModelOutputWithPast = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    gist_activations=gist_activations,
                    gist_offset=gist_offset,
                    selective=selective,
                    **kwargs,
                )

                hidden_states = outputs.last_hidden_state
                # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
                slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
                logits = self.lm_head(hidden_states[:, slice_indices, :])

                loss = None
                if labels is not None:
                    loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

            elif self.training and self.selective:
                outputs: BaseModelOutputWithPast = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    gist_activations=gist_activations,
                    gist_offset=gist_offset,
                    selective=1,
                    **kwargs,
                )

                hidden_states = outputs.last_hidden_state
                # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
                slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
                logits = self.lm_head(hidden_states[:, slice_indices, :])

                loss = None
                if labels is not None:
                    loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

            else:
                outputs: BaseModelOutputWithPast = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    gist_activations=gist_activations,
                    gist_offset=gist_offset,
                    **kwargs,
                )

                hidden_states = outputs.last_hidden_state
                # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
                slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
                logits = self.lm_head(hidden_states[:, slice_indices, :])

                loss = None
                if labels is not None:
                    loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        elif len(kwargs['attention_mask_gist'].shape) == 5:
            
            K = kwargs["attention_mask_gist"].shape[0]
            B, T = input_ids.shape

            # Expand inputs to [K*B, T]
            input_ids_ex = input_ids.repeat(K, 1)
            attention_mask_ex = attention_mask.repeat(K, 1) if attention_mask is not None else None
            position_ids_ex = position_ids.repeat(K, 1) if position_ids is not None else None
            labels_ex = labels.repeat(K, 1) if labels is not None else None

            # attention_mask_gist: from [K, ...] -> [K*B, ...] (make sure it matches model expectation)
            # If your attention_mask_gist[i] originally matches batch B, then do:
            attention_mask_gist_ex = kwargs["attention_mask_gist"].reshape(K * B, *kwargs["attention_mask_gist"].shape[2:])

            tmp_kwargs = dict(kwargs)
            tmp_kwargs["attention_mask_gist"] = attention_mask_gist_ex

            outputs = self.model(
                input_ids=input_ids_ex,
                attention_mask=attention_mask_ex,
                position_ids=position_ids_ex,
                past_key_values=None,          # usually don't use past during training here
                inputs_embeds=None,
                use_cache=False,
                cache_position=None,
                gist_activations=gist_activations,  # if these are batch-shaped, you may need to repeat them too
                gist_offset=gist_offset,
                output_hidden_states=True if Global_data.kl_loss else False,
                **tmp_kwargs,
            )

            hidden_states = outputs.last_hidden_state
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])

            loss = None
            if labels is not None:
                labels_ex_keep = labels_ex[:, slice_indices]

                # ---- CE over both halves (your original), average over K ----
                loss_ce = self.loss_function(
                    logits=logits,
                    labels=labels_ex_keep,
                    vocab_size=self.config.vocab_size,
                    **tmp_kwargs
                )
                loss = loss_ce / K

                if Global_data.kl_loss and logits.shape[0] == K * B:

                    # kl_mask_rm = (labels >= 0)

                    kl_mask_rg = (input_ids != Global_data.pad_token_id)
                    gist_token_ids = Global_data.gist_token_id
                    gist_token_tensor = torch.tensor(gist_token_ids, device=input_ids.device)
                    is_any_gist = torch.isin(input_ids, gist_token_tensor)
                    kl_mask_rg = kl_mask_rg & (~is_any_gist)

                    all_layer_h = outputs['hidden_states'][1:]
                    align_loss = 0
                    for layer_i_h in all_layer_h:
                        raw_h, gst_h = layer_i_h[:B], layer_i_h[B:]   # [B, t_keep, d]
                        layer_i_loss = masked_smoothl1_vocab(raw_h.detach(), gst_h, kl_mask_rg)
                        align_loss += layer_i_loss
                    align_loss = align_loss / len(all_layer_h)

                    Global_data.last_loss_ce.append(loss.detach().cpu().item())
                    Global_data.last_loss_align.append((Global_data.alpha * align_loss).detach().cpu().item())

                    loss += Global_data.alpha * align_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
