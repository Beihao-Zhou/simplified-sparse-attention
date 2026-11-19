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
    def __init__(self, base: nn.Linear, r: int, gist_rows_only: bool):
        super().__init__()
        self.base = base
        self.gist_rows_only = gist_rows_only
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        delta = (x.to(self.lora_A.dtype) @ self.lora_A.t()) @ self.lora_B.t()
        if self.gist_rows_only:
            delta = delta * Global_data.lora_row_mask[..., None].to(delta.dtype)
        return self.base(x) + delta.to(x.dtype)


def apply_lora(model, mode: str, r: int = 32) -> Dict[str, List[nn.Parameter]]:
    """mode: 'gist' (D, gist rows; last layer K/V only), 'gist_kv' (D-kv, K/V only in
    every layer) or 'all' (B). Freezes the model and returns trainable groups."""
    if mode not in ("gist", "gist_kv", "all"):
        raise ValueError(f"unknown LoRA mode {mode!r}")
    for p in model.parameters():
        p.requires_grad_(False)
    groups: Dict[str, List[nn.Parameter]] = {"lora_qo": [], "lora_kv": [], "embed": []}
    layers = model.model.layers
    for i, layer in enumerate(layers):
        attn = layer.self_attn
        last = i == len(layers) - 1
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if mode != "all" and name in ("q_proj", "o_proj") and (last or mode == "gist_kv"):
                continue
            lora = LoRALinear(getattr(attn, name), r, gist_rows_only=mode != "all")
            setattr(attn, name, lora.to(getattr(attn, name).weight.device))
            groups["lora_kv" if name in ("k_proj", "v_proj") else "lora_qo"] += [lora.lora_A, lora.lora_B]

    emb = model.get_input_embeddings().weight
    rows = torch.zeros(emb.shape[0], 1, dtype=emb.dtype, device=emb.device)
    rows[Global_data.gist_token_id] = 1
    emb.requires_grad_(True)
    emb.register_hook(lambda g: g * rows)
    groups["embed"].append(emb)
    return groups
