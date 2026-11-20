"""LoRA for stage 1: on gist rows only (option D) or on every row (option B, control).

Option D keeps every raw token's computation identical to the base model: the adapter's
delta is multiplied by a row mask that is 1 on summary-token rows (``Global_data.
lora_row_mask``, set by GistQwen2Model from input_ids). In the last layer D adapts only
K and V, since its gist queries and outputs feed no loss. Everything except the adapters
and the summary-token embedding rows is frozen.
"""

import math
from typing import Dict, List

import torch
import torch.nn as nn

from gist_utils import Global_data


class LoRALinear(nn.Module):
    """base(x) + B A x. With gist_rows_only the delta applies to summary rows only; with
    num_levels > 1 each summary level has its own (A, B), selected by
    Global_data.lora_row_level (-1 raw, l for a level-l summary)."""

    def __init__(self, base: nn.Linear, r: int, gist_rows_only: bool, num_levels: int = 1):
        super().__init__()
        self.base = base
        self.gist_rows_only = gist_rows_only
        self.lora_A = nn.ParameterList(
            nn.Parameter(nn.init.kaiming_uniform_(torch.empty(r, base.in_features), a=math.sqrt(5)))
            for _ in range(num_levels)
        )
        self.lora_B = nn.ParameterList(
            nn.Parameter(torch.zeros(base.out_features, r)) for _ in range(num_levels)
        )

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        xf = x.to(torch.float32)
        if len(self.lora_A) == 1:
            delta = (xf @ self.lora_A[0].t()) @ self.lora_B[0].t()
            if self.gist_rows_only:
                delta = delta * Global_data.lora_row_mask[..., None].to(delta.dtype)
        else:
            delta = 0
            for lvl, (A, B) in enumerate(zip(self.lora_A, self.lora_B)):
                rows = (Global_data.lora_row_level == lvl)[..., None].to(xf.dtype)
                delta = delta + ((xf @ A.t()) @ B.t()) * rows
        return self.base(x) + delta.to(x.dtype)


class SummaryBias(nn.Module):
    """beta_t = b[l] + u[l] . x_t for a level-l summary token (0 for raw tokens), per KV
    head; x_t is the attention input (already RMS-normalized). b starts at log(tokens the
    level covers), u at 0."""

    def __init__(self, hidden: int, num_kv: int, spans):
        super().__init__()
        self.b = nn.Parameter(torch.tensor([[math.log(s)] * num_kv for s in spans]))
        self.u = nn.Parameter(torch.zeros(len(spans), hidden, num_kv))

    def forward(self, x: torch.Tensor, level: torch.Tensor) -> torch.Tensor:
        """x [B, T, hidden], level [B, T] -> beta [B, num_kv, T]."""
        beta = torch.zeros(*level.shape, self.b.shape[1], device=x.device)
        for lvl in range(self.b.shape[0]):
            rows = (level == lvl)[..., None]
            beta = beta + rows * (self.b[lvl] + x.float() @ self.u[lvl])
        return beta.transpose(1, 2)


def _capture_input(module, args, kwargs):
    module._bias_x = kwargs.get("hidden_states", args[0] if args else None)


def _capturing(module, fn):
    def wrapped(*args, **kwargs):
        _capture_input(module, args, kwargs)
        return fn(*args, **kwargs)

    return wrapped


def apply_summary_bias(model, spans=(4, 16)) -> list:
    """Attach a SummaryBias to every attention layer but the first (layer 0 uses plain
    attention at inference and takes no part in selection); returns its parameters.

    At inference beta runs only on the eager reference path (attn_dispatcher), not on the
    flex prefill or the CUDA decode kernel."""
    Global_data.lora_active = True
    params = []
    cfg = model.config
    for layer in model.model.layers[1:]:
        attn = layer.self_attn
        attn.summary_bias = SummaryBias(cfg.hidden_size, cfg.num_key_value_heads, spans).to(
            attn.q_proj.weight.device
        )
        attn.register_forward_pre_hook(_capture_input, with_kwargs=True)
        # Selective inference calls select_forward directly, bypassing the hook.
        attn.select_forward = _capturing(attn, attn.select_forward)
        params += list(attn.summary_bias.parameters())
    return params


def apply_lora(
    model, mode: str, r: int = 32, per_level: bool = False
) -> Dict[str, List[nn.Parameter]]:
    """mode: 'gist' (D, gist rows; last layer K/V only), 'gist_kv' (D-kv, K/V only in
    every layer) or 'all' (B). per_level: one adapter per summary level (D only).
    Freezes the model and returns trainable groups; groups['level<l>'] lists the
    level-l adapters when per_level."""
    if mode not in ("gist", "gist_kv", "all"):
        raise ValueError(f"unknown LoRA mode {mode!r}")
    for p in model.parameters():
        p.requires_grad_(False)
    Global_data.lora_active = True
    if per_level and mode == "all":
        raise ValueError("per-level adapters need gist-row LoRA")
    n_lv = len(Global_data.gist_token_id) if per_level else 1
    groups: Dict[str, List[nn.Parameter]] = {"lora_qo": [], "lora_kv": [], "embed": []}
    for lvl in range(n_lv if per_level else 0):
        groups[f"level{lvl}"] = []
    layers = model.model.layers
    for i, layer in enumerate(layers):
        attn = layer.self_attn
        last = i == len(layers) - 1
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if mode != "all" and name in ("q_proj", "o_proj") and (last or mode == "gist_kv"):
                continue
            lora = LoRALinear(getattr(attn, name), r, gist_rows_only=mode != "all", num_levels=n_lv)
            setattr(attn, name, lora.to(getattr(attn, name).weight.device))
            groups["lora_kv" if name in ("k_proj", "v_proj") else "lora_qo"] += [
                *lora.lora_A,
                *lora.lora_B,
            ]
            for lvl in range(n_lv if per_level else 0):
                groups[f"level{lvl}"] += [lora.lora_A[lvl], lora.lora_B[lvl]]

    emb = model.get_input_embeddings().weight
    rows = torch.zeros(emb.shape[0], 1, dtype=emb.dtype, device=emb.device)
    rows[Global_data.gist_token_id] = 1
    emb.requires_grad_(True)
    emb.register_hook(lambda g: g * rows)
    groups["embed"].append(emb)
    return groups


def augment_with_beta(
    module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scaling: float
):
    """Fold beta into an extra q/k coordinate for the inference attention paths.

    q' = [q, 1] and k' = [k, beta / scaling] give attention logits scaling * q.k + beta
    and selection scores q.k + beta / scaling, which rank like the routing scores
    scaling * q.k + beta. v gets a zero coordinate so every head dimension matches; drop
    the last output coordinate. beta is cached per layer: computed for the whole prefix at
    prefill, zero for each decoded (raw) token.
    """
    T = key.shape[2]
    q_len = query.shape[2]
    cache = getattr(module, "_beta_cache", None)
    if q_len > 1 or cache is None:
        x = getattr(module, "_bias_x", None)
        beta = module.summary_bias(x, Global_data.lora_row_level)  # [B, H_kv, q_len]
        if T == q_len:  # a fresh prefix
            cache = beta
        elif cache is not None and cache.shape[2] + q_len == T:  # a later prefill chunk
            cache = torch.cat([cache, beta], dim=2)
        else:
            raise RuntimeError(
                f"summary-bias cache out of sync: {T} keys, {q_len} new, cache "
                f"{None if cache is None else cache.shape[2]}"
            )
    else:
        if cache.shape[2] + 1 != T:
            raise RuntimeError(f"summary-bias cache out of sync: {T} keys, cache {cache.shape[2]}")
        cache = torch.cat([cache, cache.new_zeros(*cache.shape[:2], 1)], dim=2)
    module._beta_cache = cache
    q = torch.cat([query, query.new_ones(*query.shape[:3], 1)], dim=-1)
    k = torch.cat([key, (cache[..., None] / scaling).to(key.dtype)], dim=-1)
    v = torch.cat([value, value.new_zeros(*value.shape[:3], 1)], dim=-1)
    return q, k, v
