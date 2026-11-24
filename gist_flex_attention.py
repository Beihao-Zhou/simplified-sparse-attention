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
    out = {"gist_block_mask": block_mask, "gist_kv_perm": perm, "gist_features": f,
           "gist_input_ids": input_ids}
    if Global_data.stage1 is not None and q_offset:
        raise NotImplementedError("stage 1 trains on whole sequences, not appends")
    if Global_data.stage1 is not None:
        # Stage-1 teacher masks share the features and key order; loss rows are raw,
        # non-pad tokens past the sink.
        B, T = input_ids.shape
        st1 = Global_data.stage1
        names = {n for pair in st1["pairs"] for n in pair} | down_masks(st1)
        out["gist_teacher_masks"] = {
            n: make_block_mask(make_teacher_mask_mod(f, n, kv_perm=perm), B, T, block_size,
                               str(input_ids.device), input_ids.is_cuda)
            for n in names - {"hier", "sel", "base"}}  # "sel" per layer; "base" recorded
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
        from gist_beam_cuda import row_decode, row_kernel_ok
        if row_kernel_ok(query, key):
            from gist_opsd_long import visible_row
            out, idx = row_decode(query, key, value, visible_row(attention_mask)[0], None, scaling)
            if Global_data.opsd_record is not None:
                Global_data.opsd_record[module.layer_idx] = idx
            return out, None
        from gist_qwen2 import repeat_kv
        hpg = query.shape[1] // key.shape[1]
        out = torch.nn.functional.scaled_dot_product_attention(
            query, repeat_kv(key, hpg), repeat_kv(value, hpg), attn_mask=attention_mask,
            scale=scaling)
        if Global_data.opsd_record is not None:
            from gist_opsd_long import positions, visible_row
            Global_data.opsd_record[module.layer_idx] = positions(
                visible_row(attention_mask).expand(key.shape[1], -1))
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
    key_orig, value_orig = key, value
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
                return score + beta_p[b, h // hpg, kv_idx.clamp(max=T - 1)]  # padded keys: in range
        else:
            q_in, k_in = fold_beta(query, key, beta_p, scaling)

    hpg_q = query.shape[1] // H_kv
    k_att, v_att, gqa = k_in, value, hpg_q > 1
    if gqa and hpg_q & (hpg_q - 1) and query.shape[2] < 256:
        # Few query rows (a short append, gist_session.py): compiled flex takes its decoding
        # kernel, which needs a power-of-2 group size (Qwen2.5 0.5B and 7B-1M have 7), so
        # repeat K/V per query head. ponytail: H/H_kv x the cache in transient memory (7B at
        # 1M: ~7 GB per layer); pad the append to 256 rows instead if that becomes a limit.
        from gist_qwen2 import repeat_kv
        k_att, v_att, gqa = repeat_kv(k_in, hpg_q), repeat_kv(value, hpg_q), False
        if score_mod is not None:
            def score_mod(score, b, h, q_idx, kv_idx):  # noqa: F811  (beta per KV head)
                return score + beta_p[b, h // hpg_q, kv_idx.clamp(max=T - 1)]

    def attend(block_mask):
        out = _flex(query.device)(
            q_in, k_att, v_att,
            block_mask=block_mask,
            score_mod=score_mod,
            scale=scaling,
            enable_gqa=gqa,
            kernel_options=_kernel_options(),
        )
        return out.transpose(1, 2).contiguous()

    of = Global_data.opsd_flex
    if of is not None:  # OPSD replay (gist_opsd_long.py): generated rows use recorded keys
        from gist_opsd_long import replay_block_mask, replay_routing
        if of["probe_layer"] == module.layer_idx:
            key_orig.retain_grad()
            of["probe"] = key_orig
        Tc, H = of["T_ctx"], query.shape[1]
        out = attend(gist_block_mask)[:, :Tc]  # prompt rows: the gist mask, as in the prefill
        # generated rows, padded (masked out) to whole 128-row blocks and at least two:
        # torch 2.8, once flex has compiled another shape, gets the backward of a <= 128-row
        # query wrong (forward right; unattended keys get gradient); few-row flex also needs
        # a power-of-2 GQA group. experiments/flex_short_q_bwd.py reproduces it.
        R = T - Tc
        Rp = max(-(-R // 128) * 128, 256)
        q_gen = torch.nn.functional.pad(q_in[:, :, Tc:], (0, 0, 0, Rp - R))
        gen = _flex(query.device)(
            q_gen, k_in, value,
            block_mask=replay_block_mask(module.layer_idx, H, H_kv, T, Rp, of, gist_kv_perm,
                                         query.device),
            score_mod=score_mod, scale=scaling, enable_gqa=hpg_q > 1,
            kernel_options=_kernel_options())
        out = torch.cat([out, gen[:, :, :R].transpose(1, 2)], dim=1).contiguous()
        if of["route"]:
            replay_routing(module, query, key_orig, beta, kwargs, of, scaling)
        return out, None
    st = Global_data.stage1
    if st is None:
        return attend(gist_block_mask), None
    teacher_masks = kwargs["gist_teacher_masks"]
    names = {n for pair in st["pairs"] for n in pair} | down_masks(st)
    if "sel" in names and st["select"].get("impl") == "dense":  # [T, T] mask: <= ~32K, tests
        teacher_masks = {**teacher_masks, "sel": _select_block_mask(
            query, key_orig, beta, kwargs, st["select"], gist_kv_perm, scaling)}
    elif "sel" in names:  # flex over S + gather over X, merged (gist_select_train.py, #29)
        from gist_select_train import select_attention
        teacher_masks = {**teacher_masks, "sel": lambda: select_attention(
            _flex(query.device), q_in, k_att, v_att, gqa, score_mod, query, key_orig, value_orig,
            None if beta is None else beta.float(), kwargs, st["select"], gist_kv_perm, scaling,
            _kernel_options())}
    out = _stage1_attention(module, attend, gist_block_mask, teacher_masks,
                            kwargs["gist_loss_rows"], st, kwargs["gist_features"].level >= 0)
    if st.get("route_rows", 0) > 0 and module.layer_idx > 0:
        _stage1_routing(module, query, key_orig, beta, kwargs["gist_input_ids"],
                        kwargs["gist_route_exclude"],
                        kwargs["gist_loss_rows"], scaling, st, kwargs["gist_features"])
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


def _stage1_routing(module, query, key, beta, input_ids, exclude, rows, scaling, st,
                    features=None):
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
        lo = None
        if st.get("route_free"):  # forced units out of the softmax (#33)
            from gist_select_train import _features
            # ponytail: the tree is rebuilt per layer; cache it per forward if it shows up
            lo = _features(input_ids, features, b, 0)[1][pick]
        q = query[b][:, pick]
        if st.get("route_detach_q"):  # the router alone learns: q could move the target mass
            q = q.detach()
        loss = routing_kl(
            q, pick, k[b], input_ids[b], Global_data.gist_token_id, scaling, bt,
            exclude=None if exclude is None else exclude[b],
            scope=st.get("route_scope", "level"), oracle_k=st.get("route_oracle_k", 256),
            lo_leaf=lo)
        if _first(st, (module.layer_idx, "route", b)):
            st["losses"].setdefault("route", []).append(loss)


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


def down_masks(st) -> set:
    """The masks the output passed down is made of: st["down"], or for "hybrid" (#41) raw
    for raw rows and hier (the gist mask, as at inference) for summary rows. Raw rows never
    read summary keys under raw, so they stay the base model's stream (no moving target,
    #29) while summaries are written as at inference instead of reading the whole prefix."""
    return {"raw", "hier"} if st["down"] == "hybrid" else {st["down"]}


def _stage1_attention(module, attend, hier_mask, teacher_masks, rows, st, summary_rows=None):
    """Per-layer MSE of stage 1 (docs/opsd-handoff.md): the same q/k/v under several masks.

    st["pairs"] lists (student, teacher) mask names, e.g. ("g1", "raw"), ("hier", "g1");
    a name that appears only as a teacher is computed without grad. st["down"] is the
    output passed to the next layer. Losses are appended to st["losses"][pair].

    "base" (#41) is a fixed teacher: the output passed down in an earlier no-grad forward
    with st["record"] set, normally down=raw, i.e. the base model's own stream (raw rows
    never see summaries, and the adapters act on summary rows only). A student matched to
    it can be passed down (down=sel) without the moving target of #29.
    """
    masks = {"hier": hier_mask, **teacher_masks}
    students = {s for s, _ in st["pairs"]} | down_masks(st)
    outs = {}
    def run(m):  # a BlockMask, or a function computing the output itself ("sel", #29)
        return m() if callable(m) and not hasattr(m, "kv_num_blocks") else attend(m)

    for name in sorted(({n for pair in st["pairs"] for n in pair} | down_masks(st)) - {"base"}):
        if name in students:
            outs[name] = run(masks[name])
        else:
            with torch.no_grad():
                outs[name] = run(masks[name])
    B, T = rows.shape
    proj = {n: module.o_proj(o.reshape(B, T, -1)) for n, o in outs.items()}
    if st["down"] == "hybrid":
        sm = summary_rows.reshape(B, T, *[1] * (outs["raw"].dim() - 2))
        outs["hybrid"] = torch.where(sm, outs["hier"], outs["raw"])
    if st.get("record") is not None:
        st["record"][module.layer_idx] = (proj[st["down"]] if st["down"] in proj else
                                          module.o_proj(outs[st["down"]].reshape(B, T, -1))).detach()
    if "base" in st:
        proj["base"] = st["base"][module.layer_idx]
    for s_name, t_name in st["pairs"]:
        loss = _rel_mse(proj[s_name], proj[t_name].detach(), rows)
        if _first(st, (module.layer_idx, s_name, t_name)):
            st["losses"].setdefault(f"{s_name}|{t_name}", []).append(loss)
    return outs[st["down"]]


def _first(st, key) -> bool:
    """Whether a layer's loss is seen for the first time in this forward. Gradient
    checkpointing reruns the layer during backward: the rerun must repeat the ops (saved
    tensors are matched in order) but not keep the loss again, whose graph would live
    until the next forward (as the OPSD routing loss did, #21)."""
    seen = st.setdefault("seen", set())
    if key in seen:
        return False
    seen.add(key)
    return True


AttentionInterface.register(GIST_FLEX, gist_flex_attention_forward)
