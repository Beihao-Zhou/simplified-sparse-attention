"""OPSD building blocks: rollout with recorded selections, teacher scoring, replay.

One OPSD step (docs/opsd-handoff.md, Stage 2):

1. ``rollout`` (no grad): gisted prefill, then token-by-token decode through the same
   inference path as generation (``generation_utils``: one more row of the gist mask per
   step, RoPE positions from ``Global_data.last_pos``). Every layer's visible-key mask is
   recorded at its ``scaled_dot_product_attention`` call, so the eager attention bodies
   stay untouched and layer 0 (plain attention) is recorded the same way.
2. ``teacher_logprobs`` (no grad): the original model on the raw sequence.
3. ``replay`` (grad): the context prefill exactly as in the rollout, then all generated
   rows in one parallel pass whose layers use the recorded masks (``replay_attention``,
   reached through ``attn_dispatcher`` while ``Global_data.opsd_replay`` is set).

Distributions are over the base vocabulary (the first ``vocab`` ids); summary tokens are
never sampled.
"""

import contextlib
from typing import List, Optional

import torch
import torch.nn.functional as F

from gist_routing import routing_kl
from gist_utils import Global_data
from src.data.gist import make_hierarchical_gist_mask_greedy


def prefill_gist_mask(ctx: torch.Tensor) -> torch.Tensor:
    return make_hierarchical_gist_mask_greedy(
        inputs=ctx,
        gist_token=Global_data.gist_token_id,
        chunk_size=Global_data.chunk_size[0],
        attention_sink_size=Global_data.sink_size,
        num_previous_chunks=Global_data.num_previous_chunks,
        pad_token=Global_data.pad_token_id,
    ).to(ctx.device)


def _visible(attn_mask: Optional[torch.Tensor], k_len: int, device) -> torch.Tensor:
    """[heads_or_1, k_len] bool of the keys a single decode query attends to."""
    if attn_mask is None:
        return torch.ones(1, k_len, dtype=torch.bool, device=device)
    m = attn_mask[0, :, -1, :]
    if m.dtype == torch.bool:
        return m.clone()
    return m > torch.finfo(m.dtype).min / 2


@contextlib.contextmanager
def _record_sdpa(sink: List):
    """Append the visible-key mask of every single-query SDPA call to `sink`."""
    orig = F.scaled_dot_product_attention

    def shim(query, key, value, attn_mask=None, *args, **kwargs):
        if query.shape[-2] == 1:
            sink.append(_visible(attn_mask, key.shape[-2], query.device))
        return orig(query, key, value, attn_mask, *args, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = shim
    try:
        yield
    finally:
        torch.nn.functional.scaled_dot_product_attention = orig


def _logprobs(logits: torch.Tensor, vocab: int) -> torch.Tensor:
    return F.log_softmax(logits[..., :vocab].float(), dim=-1)


@torch.no_grad()
def rollout(
    model, ctx: torch.Tensor, n_new: int, vocab: int, generator=None, greedy=False, forced=None
):
    """Sample n_new tokens at temperature 1 (argmax if greedy; the tokens of `forced` [n_new]
    if given). Returns (y [n], logprobs [n, vocab], steps) where steps[j][l] is layer l's
    visible-key mask when y[j] was fed (j < n - 1)."""
    attn = torch.ones_like(ctx)
    row = prefill_gist_mask(ctx)
    out = model(input_ids=ctx, attention_mask=attn, attention_mask_gist=row, use_cache=True)
    past, row = out.past_key_values, row[:, :, -1:]
    lps, ys, steps = [_logprobs(out.logits[0, -1], vocab)], [], []
    for t in range(n_new):
        if forced is not None:
            y = forced[t : t + 1]
        elif greedy:
            y = lps[-1].argmax().view(1)
        else:
            y = torch.multinomial(lps[-1].exp(), 1, generator=generator)
        ys.append(y)
        if t == n_new - 1:
            break
        row = torch.cat([row, row.new_ones(1, 1, 1, 1)], dim=-1)
        attn = torch.cat([attn, attn.new_ones(1, 1)], dim=-1)
        rec: List[torch.Tensor] = []
        with _record_sdpa(rec):
            out = model(
                input_ids=y.view(1, 1),
                attention_mask=attn,
                attention_mask_gist=row,
                past_key_values=past,
                use_cache=True,
            )
        assert len(rec) == model.config.num_hidden_layers, len(rec)
        steps.append(rec)
        past = out.past_key_values
        lps.append(_logprobs(out.logits[0, -1], vocab))
    return torch.cat(ys), torch.stack(lps), steps


@torch.no_grad()
def teacher_logprobs(plain, ctx: torch.Tensor, y: torch.Tensor, vocab: int) -> torch.Tensor:
    """Original model on the raw sequence: [n, vocab] distributions for y."""
    gist = torch.tensor(Global_data.gist_token_id, device=ctx.device)
    raw = ctx[0][~torch.isin(ctx[0], gist)]
    seq = torch.cat([raw, y[:-1]])[None]
    return _logprobs(plain(input_ids=seq).logits[0, raw.numel() - 1 :], vocab)


def replay(
    model,
    ctx: torch.Tensor,
    y: torch.Tensor,
    steps,
    vocab: int,
    route: bool = False,
    probe: bool = False,
):
    """Recompute the rollout's distributions with grad, following its recorded selections.

    Returns (logprobs [n, vocab], state). With route=True, state["route"] holds the mean
    routing KL over layers >= 1 and generated rows; with probe=True, state["probe"][l]
    is layer l's key tensor as used by the generated rows (retain_grad set), and
    state["masks"][l] its visibility [1, H, n-1, K].
    """
    n, Tc = y.numel(), ctx.shape[1]
    out = model(
        input_ids=ctx,
        attention_mask=torch.ones_like(ctx),
        attention_mask_gist=prefill_gist_mask(ctx),
        use_cache=True,
    )
    first = _logprobs(out.logits[0, -1:], vocab)
    state = {"route": [], "probe": {}, "masks": {}}
    if n == 1:
        return first, state
    K = Tc + n - 1
    masks = []
    for l in range(model.config.num_hidden_layers):
        h = steps[0][l].shape[0]
        m = torch.zeros(1, h, n - 1, K, dtype=torch.bool, device=ctx.device)
        for j in range(n - 1):
            m[0, :, j, : Tc + j + 1] = steps[j][l]
        masks.append(m)
    seq_ids = torch.cat([ctx[0], y[:-1]])
    pos = Global_data.last_pos.reshape(-1)[-1] + 1 + torch.arange(n - 1, device=ctx.device)
    state["masks"] = masks
    Global_data.opsd_replay = {
        "masks": masks,
        "ids": seq_ids,
        "route_on": route,
        "probe_on": probe,
        "route": state["route"],
        "probe": state["probe"],
    }
    try:
        out2 = model(
            input_ids=y[:-1][None],
            attention_mask=torch.ones(1, K, dtype=ctx.dtype, device=ctx.device),
            attention_mask_gist=masks[0],
            past_key_values=out.past_key_values,
            position_ids=pos[None],
            use_cache=True,
        )
    finally:
        Global_data.opsd_replay = None
    lps = torch.cat([first, _logprobs(out2.logits[0], vocab)])
    if route:
        state["route"] = torch.stack(state["route"]).mean()
    return lps, state


def replay_attention(module, query, key, value, scaling):
    """Generated rows of a replay: attend exactly to the recorded keys of this layer."""
    from gist_qwen2 import repeat_kv

    st = Global_data.opsd_replay
    vis = st["masks"][module.layer_idx]  # [1, H|1, R, K]
    if st["probe_on"]:
        key = key.clone()
        key.retain_grad()
        st["probe"][module.layer_idx] = key
    k = repeat_kv(key, module.num_key_value_groups)
    v = repeat_kv(value, module.num_key_value_groups)
    bias = torch.zeros(vis.shape, dtype=query.dtype, device=query.device).masked_fill(
        ~vis, float("-inf")
    )
    beta = None
    if getattr(module, "summary_bias", None) is not None:
        # beta of the prefix from the replayed prefill; generated (raw) tokens have none.
        pre = module._beta_cache  # [1, H_kv, Tc]
        beta = torch.cat([pre, pre.new_zeros(*pre.shape[:2], k.shape[2] - pre.shape[2])], dim=2)
        beta = beta.repeat_interleave(module.num_key_value_groups, dim=1)  # [1, H, K]
        bias = bias + beta[:, :, None, :].to(bias.dtype)
    out = F.scaled_dot_product_attention(query, k, v, attn_mask=bias, scale=scaling)
    if st["route_on"]:
        K = k.shape[2]
        q_pos = torch.arange(K - query.shape[2], K, device=query.device)
        st["route"].append(
            routing_kl(
                query[0],
                q_pos,
                k[0],
                st["ids"][:K],
                Global_data.gist_token_id,
                scaling,
                None if beta is None else beta[0],
            )
        )
    return out.transpose(1, 2).contiguous(), None


def reverse_kl(student_lp: torch.Tensor, teacher_lp: torch.Tensor) -> torch.Tensor:
    """Per-token KL(p_S || p_T) from log-probabilities [n, vocab]."""
    return (student_lp.exp() * (student_lp - teacher_lp)).sum(-1)


def forward_kl(student_lp: torch.Tensor, teacher_lp: torch.Tensor) -> torch.Tensor:
    """Per-token KL(p_T || p_S): penalizes teacher mass the student drops (mode collapse)."""
    return reverse_kl(teacher_lp, student_lp)
