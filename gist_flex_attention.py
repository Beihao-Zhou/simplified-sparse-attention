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

from gist_utils import Global_data
from src.data.gist_flex_mask import (
    build_gist_mask_features,
    gist_pool_permutation,
    make_block_mask,
    make_gist_block_mask,
    make_teacher_mask_mod,
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
    out = {"gist_block_mask": block_mask, "gist_kv_perm": perm}
    if Global_data.stage1 is not None:
        # Stage-1 teacher masks share the features and key order; loss rows are raw,
        # non-pad tokens past the sink.
        B, T = input_ids.shape
        names = {n for pair in Global_data.stage1["pairs"] for n in pair} | {Global_data.stage1["down"]}
        out["gist_teacher_masks"] = {
            n: make_block_mask(make_teacher_mask_mod(f, n, kv_perm=perm), B, T, block_size,
                               str(input_ids.device), input_ids.is_cuda)
            for n in names - {"hier"}}
        pos = torch.arange(T, device=input_ids.device)[None]
        out["gist_loss_rows"] = (f.level < 0) & ~f.is_pad & (pos >= f.sink_end[:, None])
        out["gist_input_ids"] = input_ids
    return out


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
    key_orig = key
    idx = gist_kv_perm[:, None, :, None].expand(B, H_kv, T, d)
    key = key.gather(2, idx)
    value = value.gather(2, idx)

    # Per-summary mass bias beta (gist_lora.SummaryBias), added to the summary columns.
    beta = score_mod = None
    sb = getattr(module, "summary_bias", None)
    if sb is not None and getattr(module, "_bias_x", None) is not None:
        beta = sb(module._bias_x, Global_data.lora_row_level)           # [B, H_kv, T]
        beta_p = beta.gather(2, gist_kv_perm[:, None, :].expand(B, H_kv, T))
        hpg = query.shape[1] // H_kv

        def score_mod(score, b, h, q_idx, kv_idx):
            return score + beta_p[b, h // hpg, kv_idx]

    def attend(block_mask):
        out = _flex(query.device)(
            query, key, value,
            block_mask=block_mask,
            score_mod=score_mod,
            scale=scaling,
            enable_gqa=query.shape[1] != H_kv,
            kernel_options=_kernel_options(),
        )
        return out.transpose(1, 2).contiguous()

    st = Global_data.stage1
    if st is None:
        return attend(gist_block_mask), None
    out = _stage1_attention(module, attend, gist_block_mask, kwargs["gist_teacher_masks"],
                            kwargs["gist_loss_rows"], st)
    if st.get("route_rows", 0) > 0 and module.layer_idx > 0:
        _stage1_routing(module, query, key_orig, beta, kwargs["gist_input_ids"],
                        kwargs["gist_loss_rows"], scaling, st)
    return out, None


def _stage1_routing(module, query, key, beta, input_ids, rows, scaling, st):
    """Routing KL on st["route_rows"] random loss rows per sequence (layer 0 excluded)."""
    from gist_qwen2 import repeat_kv
    from gist_routing import routing_kl

    hpg = query.shape[1] // key.shape[1]
    k = repeat_kv(key, hpg)
    for b in range(query.shape[0]):
        cand = torch.nonzero(rows[b]).flatten()
        pick = cand[torch.randperm(cand.numel(), device=cand.device)[:st["route_rows"]]].sort().values
        bt = None if beta is None else beta[b].repeat_interleave(hpg, 0)
        st["losses"].setdefault("route", []).append(routing_kl(
            query[b][:, pick], pick, k[b], input_ids[b], Global_data.gist_token_id, scaling, bt))


def _rel_mse(student: torch.Tensor, teacher: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Mean over loss rows of ||s - t||^2 / ||t||^2, outputs after o_proj ([B, T, D])."""
    s, t = student[rows].float(), teacher[rows].float()
    return ((s - t).pow(2).sum(-1) / (t.pow(2).sum(-1) + 1e-6)).mean()


def _stage1_attention(module, attend, hier_mask, teacher_masks, rows, st):
    """Per-layer MSE of stage 1 (docs/opsd-handoff.md): the same q/k/v under several masks.

    st["pairs"] lists (student, teacher) mask names, e.g. ("g1", "raw"), ("hier", "g1");
    a name that appears only as a teacher is computed without grad. st["down"] is the
    output passed to the next layer. Losses are appended to st["losses"][pair].
    """
    masks = {"hier": hier_mask, **teacher_masks}
    students = {s for s, _ in st["pairs"]} | {st["down"]}
    outs = {}
    for name in sorted({n for pair in st["pairs"] for n in pair} | {st["down"]}):
        if name in students:
            outs[name] = attend(masks[name])
        else:
            with torch.no_grad():
                outs[name] = attend(masks[name])
    B, T = rows.shape
    proj = {n: module.o_proj(o.reshape(B, T, -1)) for n, o in outs.items()}
    for s_name, t_name in st["pairs"]:
        st["losses"].setdefault(f"{s_name}|{t_name}", []).append(
            _rel_mse(proj[s_name], proj[t_name].detach(), rows))
    return outs[st["down"]]


AttentionInterface.register(GIST_FLEX, gist_flex_attention_forward)
