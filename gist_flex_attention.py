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

# Compiled flex_attention backpropagates into score_mod-captured tensors from torch 2.7.
_CAPTURED_GRAD = tuple(int(x) for x in torch.__version__.split(".")[:2]) >= (2, 7)

_compiled_flex = {}


def _flex(device: torch.device):
    """flex_attention, compiled for CUDA tensors and eager otherwise (e.g. CPU tests).

    Separate compiled instances for grad and no-grad calls: once an instance holds
    no-grad graphs for other lengths (evaluation), the backward through a
    score_mod-captured tensor (beta) fails to compile in torch 2.8.
    """
    if device.type != "cuda" or os.environ.get("GIST_FLEX_COMPILE", "1") != "1":
        return flex_attention
    key = torch.is_grad_enabled()
    if not _compiled_flex:
        # Dynamo keeps one graph per (mask kind, grad mode, score_mod) on flex_attention's
        # code object; L levels need ~3 (L + 1) of them, past the default limit of 8, after
        # which it runs eager flex, whose backward into beta fails under checkpointing.
        cfg = torch._dynamo.config
        name = "recompile_limit" if hasattr(cfg, "recompile_limit") else "cache_size_limit"
        setattr(cfg, name, max(getattr(cfg, name), 64))
    if key not in _compiled_flex:
        # Not dynamic=False by default: that recompiles per length and, past the recompile
        # limit, falls back to eager, which materializes [B, H, T, T] scores.
        # GIST_FLEX_DYNAMIC=0 compiles static shapes, for a few fixed lengths.
        dynamic = None if os.environ.get("GIST_FLEX_DYNAMIC", "") != "0" else False
        _compiled_flex[key] = torch.compile(flex_attention, dynamic=dynamic)
    return _compiled_flex[key]


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
    pinned: Optional[torch.Tensor] = None,
    turn: Optional[torch.Tensor] = None,
    window_turns: int = 0,
    q_offset: int = 0,
) -> dict:
    """Per-batch BlockMask and key permutation for all layers. Padding is found by
    pad_token_id; attention_mask is not read. pinned / turn ([B, T]) carry the segment
    labels of agent trajectories (src/data/gist_flex_mask.py, R6 and R7). With q_offset > 0
    input_ids is the whole sequence and only its rows from q_offset on are queries: an
    append onto a KV cache of q_offset tokens (gist_session.py)."""
    if pad_token_id is None:
        raise ValueError(f"{GIST_FLEX} needs pad_token_id to mask padded keys")
    f = build_gist_mask_features(
        input_ids, gist_token_ids, chunk_size,
        attention_sink_size=attention_sink_size,
        num_previous_chunks=num_previous_chunks,
        pad_token=pad_token_id,
        raw_window=raw_window,
        pinned=pinned,
        turn=turn,
        window_turns=window_turns,
    )
    perm = gist_pool_permutation(f)
    block_mask = make_gist_block_mask(
        f, input_ids.shape[1], block_size=block_size, kv_perm=perm,
        compile_mask=input_ids.is_cuda, q_offset=q_offset,
    )
    out = {"gist_block_mask": block_mask, "gist_kv_perm": perm}
    if Global_data.stage1 is not None and q_offset:
        raise NotImplementedError("stage 1 trains on whole sequences, not appends")
    if Global_data.stage1 is not None:
        # Stage-1 teacher masks share the features and key order; loss rows are raw,
        # non-pad tokens past the sink.
        B, T = input_ids.shape
        st1 = Global_data.stage1
        names = {n for pair in st1["pairs"] for n in pair} | {st1["down"]}
        out["gist_teacher_masks"] = {
            n: make_block_mask(make_teacher_mask_mod(f, n, kv_perm=perm), B, T, block_size,
                               str(input_ids.device), input_ids.is_cuda)
            for n in names - {"hier", "sel"}}  # "sel" is built per layer
        pos = torch.arange(T, device=input_ids.device)[None]
        out["gist_loss_rows"] = (f.level < 0) & ~f.is_pad & (pos >= f.sink_end[:, None])
        if f.pinned is not None:  # pinned rows see their whole prefix: student = teacher
            out["gist_loss_rows"] &= ~f.pinned
        out["gist_input_ids"] = input_ids
        out["gist_features"] = f
        # keys in no unit (pinned segments, the sink) are out of the routing labels
        from gist_routing import unit_exclude
        out["gist_route_exclude"] = unit_exclude(
            input_ids, gist_token_ids, attention_sink_size, f.pinned)
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
    if (gist_block_mask is None or gist_kv_perm is None) and query.shape[2] == 1 \
            and attention_mask is not None:
        # A decode step of a flex-built model (gist_session.py): layer 0 is plain attention
        # over the gist-mask row; the other layers go through select_forward.
        from gist_qwen2 import repeat_kv
        hpg = query.shape[1] // key.shape[1]
        out = torch.nn.functional.scaled_dot_product_attention(
            query, repeat_kv(key, hpg), repeat_kv(value, hpg), attn_mask=attention_mask,
            scale=scaling)
        return out.transpose(1, 2).contiguous(), None
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

    # Per-summary mass bias beta (gist_lora.SummaryBias) on the summary columns. torch >= 2.7
    # backpropagates into a tensor captured by a score_mod; older versions cannot, so beta is
    # folded into an extra q/k coordinate instead (head dim padded to a power of 2, whose
    # backward does not fit an L4's shared memory at d = 64 -> 128).
    beta = score_mod = None
    q_in, k_in = query, key
    sb = getattr(module, "summary_bias", None)
    if sb is not None and getattr(module, "_bias_x", None) is not None:
        beta = sb(module._bias_x, Global_data.lora_row_level)           # [B, H_kv, q]
        if not module.training:
            # An append (gist_session.py) scores the cached keys too: keep beta per layer, in
            # the layout gist_lora.augment_with_beta extends at decode.
            if query.shape[2] < T:
                beta = torch.cat([module._beta_cache, beta.to(module._beta_cache.dtype)], dim=2)
            module._beta_cache = beta
        beta_p = beta.gather(2, gist_kv_perm[:, None, :].expand(B, H_kv, T))
        if _CAPTURED_GRAD:
            hpg = query.shape[1] // H_kv

            def score_mod(score, b, h, q_idx, kv_idx):
                return score + beta_p[b, h // hpg, kv_idx]
        else:
            q_in, k_in = fold_beta(query, key, beta_p, scaling)

    def attend(block_mask):
        out = _flex(query.device)(
            q_in, k_in, value,
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
    teacher_masks = kwargs["gist_teacher_masks"]
    names = {n for pair in st["pairs"] for n in pair} | {st["down"]}
    if "sel" in names:
        teacher_masks = {**teacher_masks, "sel": _select_block_mask(
            query, key_orig, beta, kwargs, st["select"], gist_kv_perm, scaling)}
    out = _stage1_attention(module, attend, gist_block_mask, teacher_masks,
                            kwargs["gist_loss_rows"], st)
    if st.get("route_rows", 0) > 0 and module.layer_idx > 0:
        _stage1_routing(module, query, key_orig, beta, kwargs["gist_input_ids"],
                        kwargs["gist_route_exclude"],
                        kwargs["gist_loss_rows"], scaling, st)
    return out, None


@torch.no_grad()
def _select_block_mask(query, key, beta, kwargs, sel, perm, scaling):
    """BlockMask "sel" (#28, ADR 0002): raw rows at or after sel["rows_from"] see what the
    decode-time beam selection (gist_beam.py) keeps, computed from this layer's q, k and
    beta without gradient; every other row keeps the gist mask. Differs per KV group, so it
    carries a head dimension. query [B, H, T, d], key / beta in original key order."""
    from torch.nn.attention.flex_attention import create_block_mask, create_mask

    import gist_beam as gb
    from src.data.gist_flex_mask import make_gist_mask_mod

    f, ids = kwargs["gist_features"], kwargs["gist_input_ids"]
    B, H, T, _ = query.shape
    H_kv = key.shape[1]
    hpg = H // H_kv
    hier = create_mask(make_gist_mask_mod(f), B, None, T, T, device=query.device)[:, 0]
    pos = torch.arange(T, device=query.device)
    masks = []
    for b in range(B):
        tree = gb.build_tree(ids[b], Global_data.gist_token_id, Global_data.sink_size,
                             None if f.pinned is None else f.pinned[b],
                             None if f.turn is None else f.turn[b])
        sel_rows = (tree.level < 0) & (pos >= sel["rows_from"])
        per = []
        for g in range(H_kv):
            vis = gb.visible_all_rows(
                tree, query[b, g * hpg:(g + 1) * hpg], key[b, g],
                None if beta is None else beta[b, g], sel["cfg"], scaling,
                prev_chunks=Global_data.num_previous_chunks, raw_window=Global_data.raw_window,
                window_turns=Global_data.window_turns, rows_from=sel["rows_from"])
            per.append(torch.where(sel_rows[:, None], vis, hier[b]))
        masks.append(torch.stack(per))
    m = torch.stack(masks).gather(3, perm[:, None, None, :].expand(B, H_kv, T, T))

    def mask_mod(b, h, q_idx, kv_idx):
        return m[b, h // hpg, q_idx, kv_idx]

    import torch._inductor.config as inductor_config
    with inductor_config.patch(realize_opcount_threshold=10_000, realize_reads_threshold=10_000,
                               realize_acc_reads_threshold=10_000):
        return create_block_mask(mask_mod, B, H, T, T, device=str(query.device),
                                 BLOCK_SIZE=128, _compile=query.is_cuda)


def _stage1_routing(module, query, key, beta, input_ids, exclude, rows, scaling, st):
    """Routing KL on st["route_rows"] random loss rows per sequence (layer 0 excluded)."""
    from gist_qwen2 import repeat_kv
    from gist_routing import routing_kl

    hpg = query.shape[1] // key.shape[1]
    k = repeat_kv(key, hpg)
    for b in range(query.shape[0]):
        cand = torch.nonzero(rows[b]).flatten()
        perm = torch.randperm(cand.numel(), device=cand.device)[: st["route_rows"]]
        pick = cand[perm].sort().values
        bt = None if beta is None else beta[b].repeat_interleave(hpg, 0)
        st["losses"].setdefault("route", []).append(routing_kl(
            query[b][:, pick], pick, k[b], input_ids[b], Global_data.gist_token_id, scaling, bt,
            exclude=None if exclude is None else exclude[b],
            scope=st.get("route_scope", "level"), oracle_k=st.get("route_oracle_k", 256)))


def fold_beta(query: torch.Tensor, key: torch.Tensor, beta: torch.Tensor, scaling: float):
    """Pad q/k to the next power-of-2 head dim (flex supports no other) with one coordinate
    1 on q and beta/scaling on k, so scaling * q'.k' = scaling * q.k + beta.
    query [B, H, T, d], key [B, H_kv, T, d], beta [B, H_kv, T] in key order."""
    d = query.shape[-1]
    E = 1 << d.bit_length()                                             # > d, power of 2
    zq = query.new_zeros(*query.shape[:3], E - d - 1)
    zk = key.new_zeros(*key.shape[:3], E - d - 1)
    q = torch.cat([query, query.new_ones(*query.shape[:3], 1), zq], dim=-1)
    k = torch.cat([key, (beta[..., None] / scaling).to(key.dtype), zk], dim=-1)
    return q, k


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
