"""Gist causal mask as a FlexAttention `mask_mod`.

Every rule of `make_hierarchical_gist_mask_greedy` can be decided from a few
per-token positions, so the mask needs O(T) features instead of [B, 1, T, T]. It
implements greedy mask AND causal, which is what the model applies.

Each rule is one comparison per (q, k), independent of the number of gist levels: a
gist stays visible from its position until the first gist of a higher level closes it
(`close`), and is read by its parent (`par`). Comparing chunk and section ids level by
level instead would cost O(levels) per element.

Not supported: `link_tokens` (multi-document RAG) and `add_raw` (unused KL path).
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass
class GistMaskFeatures:
    """Per-token positions from which the gist causal mask is computed ([B, T] int32).
    close / par / par_p describe a gist as a key; start describes a token as a query."""

    level: torch.Tensor     # -1 for raw tokens, l for a level-l summary token
    close: torch.Tensor     # where the gist's lifetime ends: the next higher-level gist, else T
    par: torch.Tensor       # the next gist one level up (its parent), else T
    par_p: torch.Tensor     # the (P+1)-th gist one level up, else T (R4, P > 0)
    start: torch.Tensor     # first position of the query's level-0 chunk, P chunks back
    sink_end: torch.Tensor  # [B]: keys before it are visible to every query
    has_gist: torch.Tensor  # [B] bool: example contains at least one summary token
    is_pad: torch.Tensor    # [B, T] bool
    num_levels: int
    num_previous_chunks: int
    raw_window: int = 0


def _occurrence_table(is_x: torch.Tensor) -> torch.Tensor:
    """occ[b, m] = position of the m-th (0-based) True in is_x[b], else T; width T + 1."""
    B, T = is_x.shape
    cnt = is_x.long().cumsum(-1)
    occ = torch.full((B, T + 1), T, dtype=torch.long, device=is_x.device)
    pos = torch.arange(T, device=is_x.device).expand(B, T)
    occ.scatter_(1, torch.where(is_x, cnt - 1, T), pos)
    occ[:, T] = T
    return occ


def _nth_after(is_x: torch.Tensor, n: int = 1) -> torch.Tensor:
    """Position of the n-th True strictly after each position, else T."""
    T = is_x.shape[1]
    index = (is_x.long().cumsum(-1) + n - 1).clamp_max(T)
    return _occurrence_table(is_x).gather(1, index)


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
    L, P = len(gist_token), num_previous_chunks

    level = torch.full((B, T), -1, dtype=torch.long, device=device)
    for lvl, g in enumerate(gist_token):
        level[inputs == g] = lvl

    close = torch.full_like(level, T)
    par = torch.full_like(level, T)
    par_p = torch.full_like(level, T)
    for lvl in range(L - 1):  # the top level is never closed and has no parent
        here = level == lvl
        close = torch.where(here, _nth_after(level > lvl), close)
        par = torch.where(here, _nth_after(level == lvl + 1), par)
        if P > 0:
            par_p = torch.where(here, _nth_after(level == lvl + 1, P + 1), par_p)

    # start[q]: just after the n-th level-0 gist, n = (level-0 gists before q) - P.
    is_g0 = level == 0
    n = is_g0.long().cumsum(-1) - is_g0.long() - P
    g0_pos = _occurrence_table(is_g0)
    start = torch.where(n > 0, g0_pos.gather(1, (n - 1).clamp(0, T)) + 1, 0)

    positions = torch.arange(T, device=device)
    if attention_sink_size < 0:
        # Dynamic sink: everything before (first level-0 gist - chunk_size).
        first = torch.where(is_g0, positions[None, :], torch.full_like(positions[None, :], T))
        sink_end = first.min(dim=1).values - chunk_size
    else:
        sink_end = torch.full((B,), max(attention_sink_size, 0), dtype=torch.long, device=device)

    is_pad = (inputs == pad_token) if pad_token is not None else torch.zeros_like(inputs, dtype=torch.bool)
    return GistMaskFeatures(
        level=level.int(), close=close.int(), par=par.int(), par_p=par_p.int(), start=start.int(),
        sink_end=sink_end.int(), has_gist=(level >= 0).any(-1), is_pad=is_pad,
        num_levels=L, num_previous_chunks=P, raw_window=raw_window,
    )


def make_gist_mask_mod(f: GistMaskFeatures, kv_perm: Optional[torch.Tensor] = None):
    """Return a FlexAttention mask_mod equivalent to the greedy gist mask AND causal.

    If kv_perm [B, T] is given, keys are assumed to be stored in that order (see
    `gist_pool_permutation`): key slot j holds original position kv_perm[b, j]. Key
    features are gathered into that order once here, so the mask_mod reads them
    directly by slot.

    Rules (q = query position, k = key position, l(.) = level, -1 for raw tokens):
      R1  summary visible: l(k) >= 0 and q < close[k]. A gist is visible until a gist
          of a higher level closes it (the top level never closes).
      R2  local chunk: l(q) <= 0, l(k) <= 0 and k >= start[q] (the query's level-0
          chunk, or P chunks back).
      R3  parent reads its children: l(q) = l(k) + 1 and q = par[k].
      R4  if P > 0: l(k) >= 0 below the top, par[k] < q <= par_p[k] and l(q) <= l(k)+1
          (children of the previous P parents).
      R5  attention sink: k < sink_end.
      R6  raw window (W > 0): raw q and raw k with q - k < W. Once gists are inserted
          everywhere, R2 alone would leave a generated token only 4-8 exact tokens
          (L=4, P=1); the window keeps recent text exact. Summary queries are
          excluded, so a gist still summarizes exactly its chunk.
    Examples without summary tokens fall back to plain causal attention; padded keys
    are never visible.
    """
    P, W, top = f.num_previous_chunks, f.raw_window, f.num_levels - 1
    level, start, sink_end, has_gist, is_pad = f.level, f.start, f.sink_end, f.has_gist, f.is_pad
    if kv_perm is None:
        k_pos = None
        k_level, k_close, k_par, k_par_p, k_pad = level, f.close, f.par, f.par_p, is_pad
    else:
        k_pos = kv_perm.to(torch.int32)
        k_level, k_close, k_par, k_par_p, k_pad = (
            t.gather(1, kv_perm) for t in (level, f.close, f.par, f.par_p, is_pad))

    def mask_mod(b, h, q_idx, kv_idx):
        k = kv_idx if k_pos is None else k_pos[b, kv_idx]
        lq, lk = level[b, q_idx], k_level[b, kv_idx]
        gist_k = lk >= 0
        vis = gist_k & (q_idx < k_close[b, kv_idx])                                  # R1
        vis = vis | ((lq <= 0) & (lk <= 0) & (k >= start[b, q_idx]))                  # R2
        vis = vis | (gist_k & (lq == lk + 1) & (q_idx == k_par[b, kv_idx]))           # R3
        if P > 0:                                                                     # R4
            par_k = k_par[b, kv_idx]
            vis = vis | (gist_k & (lk < top) & (par_k < q_idx) & (q_idx <= k_par_p[b, kv_idx])
                         & (lq <= lk + 1))
        vis = vis | (k < sink_end[b])                                                 # R5
        if W > 0:                                                                     # R6
            vis = vis | ((lq < 0) & ~is_pad[b, q_idx] & (lk < 0) & (q_idx - k < W))
        vis = vis | ~has_gist[b]
        return (k <= q_idx) & vis & ~k_pad[b, kv_idx]

    return mask_mod


def gist_pool_permutation(f: GistMaskFeatures) -> torch.Tensor:
    """Key order [raw tokens | summary tokens], each in original order.

    In the original order a summary token lands in every key block, so no block can
    be skipped; pooled, the raw keys form a band around the diagonal. Returns perm
    [B, T] (slot j holds position perm[b, j]) for gathering K/V and for
    `make_gist_mask_mod(f, kv_perm=perm)`.
    """
    is_summary = f.level >= 0                                        # [B, T]
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
    import torch._inductor.config as inductor_config
    from torch.nn.attention.flex_attention import create_block_mask

    B = f.level.shape[0]
    # The mask_mod exceeds inductor's realize thresholds, so it would write the [T, T]
    # mask to memory (16 GB at 128K) instead of fusing it into the block reduction.
    with inductor_config.patch(realize_opcount_threshold=10_000, realize_reads_threshold=10_000,
                               realize_acc_reads_threshold=10_000):
        return create_block_mask(
            make_gist_mask_mod(f, kv_perm=kv_perm), B, None, seq_len, seq_len,
            device=device or str(f.level.device), BLOCK_SIZE=block_size,
            _compile=compile_mask,
        )
