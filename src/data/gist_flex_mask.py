"""Gist causal mask as a FlexAttention `mask_mod`.

Every rule of `make_hierarchical_gist_mask_greedy` depends only on per-token level,
chunk and section ids, so the mask needs O(T) features instead of [B, 1, T, T].
It implements greedy mask AND causal, which is what the model applies.

Not supported: `link_tokens` (multi-document RAG) and `add_raw` (unused KL path).
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass
class GistMaskFeatures:
    """Per-token features from which the gist causal mask is computed."""

    is_gist: torch.Tensor   # [L, B, T] bool: token is a level-l summary token
    chunk: torch.Tensor     # [L, B, T] long: level-l chunk id (a gist belongs to its own chunk)
    section: torch.Tensor   # [L, B, T] long: number of gists above level l up to and including t
    any_ge: torch.Tensor    # [L+1, B, T] bool: token is a summary token of level >= l
    sink_end: torch.Tensor  # [B] long: keys with index < sink_end are visible to every query
    has_gist: torch.Tensor  # [B] bool: example contains at least one summary token
    is_pad: torch.Tensor    # [B, T] bool
    num_previous_chunks: int
    raw_window: int = 0


def _chunk_ids(is_level: torch.Tensor) -> torch.Tensor:
    """Chunk id per token; a gist closes its chunk, so it belongs to it."""
    c = is_level.long().cumsum(-1)
    return torch.where(is_level, c - 1, c)


def build_gist_mask_features(
    inputs: torch.Tensor,
    gist_token: Sequence[int],
    chunk_size: int,
    attention_sink_size: int = 0,
    num_previous_chunks: int = 0,
    pad_token: Optional[int] = None,
    raw_window: int = 0,
) -> GistMaskFeatures:
    """Arguments mirror `make_hierarchical_gist_mask_greedy`, plus `raw_window` (R6)."""
    B, T = inputs.shape
    device = inputs.device
    L = len(gist_token)

    is_gist = torch.stack([inputs == g for g in gist_token])          # [L, B, T]
    chunk = torch.stack([_chunk_ids(is_gist[lvl]) for lvl in range(L)])   # [L, B, T]

    # any_ge[l] = summary token of level >= l; any_ge[L] is all False.
    any_ge = torch.zeros(L + 1, B, T, dtype=torch.bool, device=device)
    for lvl in range(L - 1, -1, -1):
        any_ge[lvl] = any_ge[lvl + 1] | is_gist[lvl]

    # Sections at level l are delimited by any summary token above level l.
    section = torch.stack([any_ge[lvl + 1].long().cumsum(-1) for lvl in range(L)])

    positions = torch.arange(T, device=device)
    if attention_sink_size < 0:
        # Dynamic sink: everything before (first level-0 gist - chunk_size).
        first = torch.where(is_gist[0], positions[None, :], torch.full_like(positions[None, :], T))
        sink_end = first.min(dim=1).values - chunk_size
    elif attention_sink_size > 0:
        sink_end = torch.full((B,), attention_sink_size, dtype=torch.long, device=device)
    else:
        sink_end = torch.zeros(B, dtype=torch.long, device=device)

    is_pad = (inputs == pad_token) if pad_token is not None else torch.zeros_like(inputs, dtype=torch.bool)

    return GistMaskFeatures(
        is_gist=is_gist,
        chunk=chunk,
        section=section,
        any_ge=any_ge,
        sink_end=sink_end,
        has_gist=any_ge[0].any(-1),
        is_pad=is_pad,
        num_previous_chunks=num_previous_chunks,
        raw_window=raw_window,
    )


def make_gist_mask_mod(f: GistMaskFeatures, kv_perm: Optional[torch.Tensor] = None):
    """Return a FlexAttention mask_mod equivalent to the greedy gist mask AND causal.

    If kv_perm [B, T] is given, keys are assumed to be stored in that order (see
    `gist_pool_permutation`): key slot j holds original position kv_perm[b, j], and
    the rules below are evaluated on original positions.

    Rules (q = query index, k = key index):
      R1  summary tokens: the top level is visible to every query; level l < top only
          within the query's section of level l (no higher gist in between).
      R2  local window: raw and level-0 keys in the same level-0 chunk or the previous P
          chunks, for queries that are not summary tokens of level >= 1. (Higher gists
          are left to R1; see make_hierarchical_gist_mask_greedy.)
      R3  a level-l gist sees the level-(l-1) gists of its own level-l chunk.
      R4  if P > 0, queries below level l+1 also see level-(l-1) gists in the previous
          P level-l chunks (this window may cross section boundaries, as in the
          original mask).
      R5  attention sink: keys before sink_end.
      R6  raw window (W > 0): raw queries also see raw keys among the last W
          positions. Once gists are inserted everywhere, R2 alone would leave a
          generated token only 4-8 exact tokens (L=4, P=1); the window keeps recent
          text exact. Summary queries are excluded, so a gist still summarizes
          exactly its chunk.
    Examples without summary tokens fall back to plain causal attention; padded keys
    are never visible.
    """
    L = f.is_gist.shape[0]
    P = f.num_previous_chunks
    is_gist, chunk, section, any_ge = f.is_gist, f.chunk, f.section, f.any_ge
    sink_end, has_gist, is_pad = f.sink_end, f.has_gist, f.is_pad
    W = f.raw_window

    def mask_mod(b, h, q_idx, kv_idx):
        if kv_perm is not None:
            kv_idx = kv_perm[b, kv_idx]
        # R1
        vis = is_gist[L - 1][b, kv_idx]
        for lvl in range(L - 1):
            vis = vis | (is_gist[lvl][b, kv_idx] & (section[lvl][b, q_idx] == section[lvl][b, kv_idx]))

        # R2
        d0 = chunk[0][b, q_idx] - chunk[0][b, kv_idx]
        vis = vis | (((d0 == 0) | ((d0 > 0) & (d0 <= P))) & ~any_ge[1][b, q_idx] & ~any_ge[1][b, kv_idx])

        # R3 / R4
        for lvl in range(1, L):
            dl = chunk[lvl][b, q_idx] - chunk[lvl][b, kv_idx]
            prev_gist_k = is_gist[lvl - 1][b, kv_idx]
            vis = vis | ((dl == 0) & prev_gist_k & is_gist[lvl][b, q_idx])
            if P > 0:
                vis = vis | ((dl > 0) & (dl <= P) & prev_gist_k & ~any_ge[lvl + 1][b, q_idx])

        # R5
        vis = vis | (kv_idx < sink_end[b])

        # R6
        if W > 0:
            raw_q = ~any_ge[0][b, q_idx] & ~is_pad[b, q_idx]
            vis = vis | ((q_idx - kv_idx < W) & raw_q & ~any_ge[0][b, kv_idx])

        vis = vis | ~has_gist[b]
        return (kv_idx <= q_idx) & vis & ~is_pad[b, kv_idx]

    return mask_mod


def gist_pool_permutation(f: GistMaskFeatures) -> torch.Tensor:
    """Key order [raw tokens | summary tokens], each in original order.

    In the original order a summary token lands in every key block, so no block can
    be skipped; pooled, the raw keys form a band around the diagonal. Returns perm
    [B, T] (slot j holds position perm[b, j]) for gathering K/V and for
    `make_gist_mask_mod(f, kv_perm=perm)`.
    """
    is_summary = f.any_ge[0]                                         # [B, T]
    T = is_summary.shape[1]
    pos = torch.arange(T, device=is_summary.device)
    return torch.argsort(is_summary.long() * T + pos[None, :], dim=1)


def make_gist_block_mask(
    f: GistMaskFeatures,
    seq_len: int,
    block_size: int = 128,
    device: Optional[str] = None,
    kv_perm: Optional[torch.Tensor] = None,
    compile_mask: bool = False,
):
    """BlockMask shared by all heads. Use compile_mask=True on GPU: otherwise the
    mask is evaluated on the full T x T grid."""
    from torch.nn.attention.flex_attention import create_block_mask

    B = f.is_gist.shape[1]
    return create_block_mask(
        make_gist_mask_mod(f, kv_perm=kv_perm), B, None, seq_len, seq_len,
        device=device or str(f.is_gist.device), BLOCK_SIZE=block_size,
        _compile=compile_mask,
    )
