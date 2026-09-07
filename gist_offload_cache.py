"""Pinned-host KV cache for the sparse decode path.

What moves where
----------------
* The full raw attention K/V history lives in **pinned CPU memory**, one buffer
  per layer, preallocated to ``prefill + max_new_tokens`` so appending a decode
  step is a 1-column copy and never a reallocation.
* The compact Gist/selection metadata stays on the GPU: it is built once per
  prefix by ``attn_candidate._build_full_cache`` from the gist positions only
  (``Gmax`` keys per level, not the whole context).
* Newly generated K/V is appended to the host buffer before the layer attends,
  so the current token is always visible to the gather.
* Decode then calls ``sparse_decode_attn_offload``, which compacts the selected
  indices on the GPU and gathers only those rows across PCIe.

The GPU copy of the prefill K/V is dropped **per layer, during prefill**: layer
L's keys are released as soon as layer L+1 updates, so peak GPU KV is one
layer's context rather than all of them. ``get_seq_length()`` keeps working
because the released layer keeps a zero-width placeholder of the right length.

Scope of this version: bf16 only, greedy/beam-1 decoding, no cache cropping or
beam reordering, and a batch whose sequences share one padded length. Anything
outside that should keep using the resident path.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache


class GistOffloadCache(DynamicCache):
    """DynamicCache that keeps raw K/V in pinned host memory during decode."""

    def __init__(self, config=None, max_new_tokens: int = 1024,
                 resident_layers=(0,), hot_buffer_size: int | None = None, **kw):
        super().__init__(config=config, **kw)
        self.max_new_tokens = int(max_new_tokens)
        # GistQwen2DecoderLayer.forward forces `selective = 0` for layer_idx < 1
        # ("use standard attention for stability"), so layer 0 runs dense SDPA and
        # needs its K/V on the GPU. Those layers keep plain DynamicCache behaviour.
        self.resident_layers = set(resident_layers)
        if hot_buffer_size is None:
            hot_buffer_size = 0
        if hot_buffer_size < 0:
            raise ValueError("hot_buffer_size must be non-negative")
        self.hot_buffer_size = int(hot_buffer_size)
        self.host_k: dict[int, torch.Tensor] = {}
        self.host_v: dict[int, torch.Tensor] = {}
        self.filled: dict[int, int] = {}
        self._released: set[int] = set()
        self.pinned_bytes = 0
        self.hot: dict[int, tuple[torch.Tensor, ...]] = {}

    # -- bookkeeping --------------------------------------------------------
    def _alloc(self, layer_idx, keys):
        B, num_kv, n, D = keys.shape
        buf = n + self.max_new_tokens
        opts = dict(dtype=keys.dtype, device="cpu", pin_memory=True)
        self.host_k[layer_idx] = torch.empty(B, num_kv, buf, D, **opts)
        self.host_v[layer_idx] = torch.empty(B, num_kv, buf, D, **opts)
        self.pinned_bytes += 2 * self.host_k[layer_idx].numel() * keys.element_size()
        if self.hot_buffer_size:
            self._alloc_hot(layer_idx, keys.device)

    def _alloc_hot(self, layer_idx, device):
        host = self.host_k[layer_idx]
        B, num_kv, token_capacity, D = host.shape
        hot = self.hot_buffer_size
        bf16 = dict(dtype=host.dtype, device=device)
        i32 = dict(dtype=torch.int32, device=device)
        hot_k = torch.empty(B, num_kv, hot, D, **bf16)
        hot_v = torch.empty_like(hot_k)
        token_to_slot = torch.full((B, num_kv, token_capacity), -1, **i32)
        slot_token_ids = torch.full((B, num_kv, hot), -1, **i32)
        lru_order = (torch.arange(hot, **i32).view(1, 1, hot)
                     .expand(B, num_kv, hot).contiguous())
        lru_scratch = torch.empty_like(lru_order)
        slot_marks = torch.empty(B, num_kv, hot, dtype=torch.uint8, device=device)
        stats = torch.zeros(2, dtype=torch.int64, device=device)  # selected, misses
        self.hot[layer_idx] = (
            hot_k, hot_v, token_to_slot, slot_token_ids,
            lru_order, lru_scratch, slot_marks, stats,
        )

    def _release_gpu(self, layer_idx):
        """Free this layer's GPU K/V, keeping only a zero-width length marker."""
        if layer_idx in self._released or layer_idx >= len(self.layers):
            return
        layer = self.layers[layer_idx]
        if getattr(layer, "keys", None) is None:
            return
        B, num_kv, n, _ = layer.keys.shape
        marker = torch.empty(B, num_kv, n, 0, dtype=layer.keys.dtype, device=layer.keys.device)
        layer.keys, layer.values = marker, marker
        self._released.add(layer_idx)

    def view(self, layer_idx):
        n = self.filled[layer_idx]
        return self.host_k[layer_idx][:, :, :n], self.host_v[layer_idx][:, :, :n]

    def hot_state(self, layer_idx, device):
        """Return persistent GPU hot-buffer tensors for one sparse layer."""
        if self.hot_buffer_size == 0:
            return None
        if layer_idx not in self.hot:
            self._alloc_hot(layer_idx, device)
        return self.hot[layer_idx]

    def hot_counts(self):
        if not self.hot:
            return 0, 0
        totals = torch.stack([state[-1] for state in self.hot.values()]).sum(0).cpu()
        return int(totals[0]), int(totals[1])

    # -- the hot path -------------------------------------------------------
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx in self.resident_layers:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)
        if layer_idx not in self.host_k:
            # Prefill: run the normal cache so this layer's attention still sees
            # GPU K/V, mirror the result into pinned host memory, then release
            # every earlier layer's GPU copy.
            keys, values = super().update(key_states, value_states, layer_idx, cache_kwargs)
            self._alloc(layer_idx, keys)
            n = keys.shape[2]
            self.host_k[layer_idx][:, :, :n].copy_(keys)
            self.host_v[layer_idx][:, :, :n].copy_(values)
            self.filled[layer_idx] = n
            for prev in list(self.host_k):
                if prev != layer_idx and prev not in self.resident_layers:
                    self._release_gpu(prev)
            return keys, values

        # Decode: append one column to the host buffer; nothing grows on the GPU.
        self._release_gpu(layer_idx)
        n = self.filled[layer_idx]
        add = key_states.shape[2]
        cap = self.host_k[layer_idx].shape[2]
        if n + add > cap:
            raise RuntimeError(
                f"GistOffloadCache layer {layer_idx} overflowed: {n}+{add} > {cap}. "
                f"Construct it with max_new_tokens >= the generation length.")
        self.host_k[layer_idx][:, :, n:n + add].copy_(key_states, non_blocking=False)
        self.host_v[layer_idx][:, :, n:n + add].copy_(value_states, non_blocking=False)
        self.filled[layer_idx] = n + add
        # Keep the zero-width marker's length in step so get_seq_length() is right.
        layer = self.layers[layer_idx]
        layer.keys = layer.values = torch.empty(
            key_states.shape[0], key_states.shape[1], n + add, 0,
            dtype=key_states.dtype, device=key_states.device)
        return self.view(layer_idx)
