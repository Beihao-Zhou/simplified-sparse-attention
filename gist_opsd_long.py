"""OPSD on long agent trajectories (#21; ADR 0001 and 0002).

`rollout` samples on a GistSession: the prompt is appended through flex, decoding runs on
the DECODE_SELECT=beam path (kernels A/B on CUDA, the reference otherwise) and inserts
summaries as it goes. At every fed token each layer records the key positions it
attended to ([num_kv, n] int32, -1 empty): the key list of kernel B, the reference's
visible set, or the gist-mask row (layer 0, summary rows).

`replay` recomputes the generated rows with gradient in one flex forward over prompt and
generation: prompt rows keep the gist mask (as in the prefill), generated rows attend to
exactly the recorded keys (a per-layer, per-KV-group mask over a few hundred rows), so the
reverse KL reaches the summaries of the prompt and, with coarse terms on, the unselected
siblings. The routing KL is computed on the generated raw rows, excluding pinned tokens
and the sink.
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from gist_utils import Global_data

BIG = 1 << 30


def visible_row(attention_mask: torch.Tensor) -> torch.Tensor:
    """[1, T] bool keys of the last row of a 4D additive (or bool) attention mask."""
    m = attention_mask[0, :, -1, :]
    if m.dtype == torch.bool:
        return m[:1]
    return (m > torch.finfo(m.dtype).min / 2)[:1]


def positions(vis: torch.Tensor) -> torch.Tensor:
    """[G, T] bool -> [G, n] int32 positions, -1 padded."""
    T = vis.shape[1]
    n = max(int(vis.sum(1).max()), 1)
    idx = torch.where(vis, torch.arange(T, device=vis.device), BIG).sort(dim=1).values[:, :n]
    return torch.where(idx < BIG, idx, -1).int()


def _logprobs(logits: torch.Tensor, vocab: int) -> torch.Tensor:
    return F.log_softmax(logits[..., :vocab].float(), dim=-1)


@torch.no_grad()
def rollout(session, first_logits: torch.Tensor, n_new: int, vocab: int, generator=None,
            greedy: bool = False):
    """Sample n_new raw tokens after the session's content (first_logits: the logits of
    its last row). Returns (stream, logprobs [n_new, vocab], records): stream = the fed
    tokens (summaries included) followed by the last raw token, which is not fed;
    records[i] = {layer: [num_kv, n]} for fed token i."""
    lp = _logprobs(first_logits, vocab)
    lps, records = [lp], []
    start = len(session.ids)
    y = None
    for t in range(n_new):
        if greedy:
            y = int(lp.argmax())
        else:
            probs = lp.exp()
            if generator is not None and generator.device != probs.device:
                probs = probs.to(generator.device)
            y = int(torch.multinomial(probs, 1, generator=generator))
        if t == n_new - 1:
            break
        logits = session.step(y, record=True)
        records += session.last_records
        lp = _logprobs(logits, vocab)
        lps.append(lp)
    return session.ids[start:] + [y], torch.stack(lps), records


def _replay_masks(records: List[Dict[int, torch.Tensor]], layer: int, num_kv: int, T: int,
                  device) -> torch.Tensor:
    """[num_kv, R, T] bool: recorded keys of the generated rows at one layer."""
    M = torch.zeros(num_kv, len(records), T + 1, dtype=torch.bool, device=device)
    for r, rec in enumerate(records):
        idx = rec[layer].long()
        M[:, r].scatter_(1, torch.where(idx >= 0, idx, T), True)
    return M[..., :T]


def replay_block_mask(module, query, num_kv, kwargs, of, perm):
    """BlockMask of a replay layer: rows before of["T_ctx"] keep the gist mask, the others
    attend to their recorded keys (a head dimension, as KV groups differ)."""
    from torch.nn.attention.flex_attention import create_block_mask

    from src.data.gist_flex_mask import make_gist_mask_mod

    B, H, T, _ = query.shape
    hpg = H // num_kv
    Tc = of["T_ctx"]
    M = _replay_masks(of["records"], module.layer_idx, num_kv, T, query.device)
    Mp = M.gather(2, perm[0][None, None, :].expand(num_kv, M.shape[1], T))
    base = make_gist_mask_mod(kwargs["gist_features"], kv_perm=perm)

    R = Mp.shape[1]

    def mod(b, h, q_idx, kv_idx):
        # flex also evaluates padded rows / columns (lengths rounded up to the block size):
        # keep every index in range, an out-of-range read poisoned the backward
        gen = Mp[h // hpg, (q_idx - Tc).clamp(0, R - 1), kv_idx.clamp(max=T - 1)] & (kv_idx < T)
        return torch.where(q_idx < Tc, base(b, h, q_idx, kv_idx), gen)

    import torch._inductor.config as inductor_config
    with inductor_config.patch(realize_opcount_threshold=10_000, realize_reads_threshold=10_000,
                               realize_acc_reads_threshold=10_000):
        return create_block_mask(mod, B, H, T, T, device=str(query.device), BLOCK_SIZE=128,
                                 _compile=query.is_cuda)


def replay_routing(module, query, key, beta, kwargs, of, scaling):
    """Routing KL of the generated raw rows of one layer (layer 0 excluded)."""
    from gist_qwen2 import repeat_kv
    from gist_routing import routing_kl, unit_exclude

    if module.layer_idx == 0:
        return
    ids = kwargs["gist_input_ids"][0]
    T, Tc = ids.numel(), of["T_ctx"]
    gids = torch.tensor(Global_data.gist_token_id, device=ids.device)
    rows = torch.arange(Tc, T, device=ids.device)
    rows = rows[~torch.isin(ids[rows], gids)]
    if rows.numel() == 0:
        return
    hpg = query.shape[1] // key.shape[1]
    f = kwargs["gist_features"]
    ex = unit_exclude(ids, Global_data.gist_token_id, Global_data.sink_size,
                      None if f.pinned is None else f.pinned[0])
    of["route_losses"].append(routing_kl(
        query[0][:, rows], rows, repeat_kv(key, hpg)[0], ids, Global_data.gist_token_id, scaling,
        None if beta is None else beta[0].repeat_interleave(hpg, 0), exclude=ex,
        scope=of["route"].get("scope", "level"), oracle_k=of["route"].get("oracle_k", 256)))


def replay(model, ctx_ids: List[int], ctx_pinned: List[bool], ctx_turn: List[int],
           stream: List[int], records, vocab: int, route: Optional[dict] = None,
           probe_layer: Optional[int] = None):
    """Recompute the rollout with gradient. Returns (logprobs [n_raw, vocab] of the
    stream's raw tokens, state) with state["route"] the mean routing KL (if route) and
    state["probe"] the probed layer's keys (retain_grad) if probe_layer is set.

    The replay settings stay in Global_data until end_replay(state): gradient
    checkpointing recomputes the forward during backward, which must take the same path.
    Call end_replay after backward."""
    dev = next(model.parameters()).device
    Tc = len(ctx_ids)
    fed = stream[:-1]
    ids = torch.tensor([ctx_ids + fed], device=dev)
    pin = torch.tensor([ctx_pinned + [False] * len(fed)], device=dev)
    turn = torch.tensor([ctx_turn + [ctx_turn[-1] if ctx_turn else 0] * len(fed)], device=dev)
    of = {"T_ctx": Tc, "records": records, "route": route, "route_losses": [],
          "probe_layer": probe_layer, "probe": None}
    saved = (Global_data.train_attn, Global_data.opsd_flex, Global_data.stage1)
    Global_data.train_attn, Global_data.opsd_flex, Global_data.stage1 = "flex", of, None
    try:
        h = model.model(input_ids=ids, attention_mask=torch.ones_like(ids), gist_pinned=pin,
                        gist_turn=turn, use_cache=False).last_hidden_state[0]
    except BaseException:
        Global_data.train_attn, Global_data.opsd_flex, Global_data.stage1 = saved
        raise
    gset = set(Global_data.gist_token_id)
    rows = [Tc - 1 + j for j, t in enumerate(stream) if t not in gset]
    lps = _logprobs(model.lm_head(h[rows]), vocab)
    state = {"probe": of["probe"], "_saved": saved}
    if route:
        losses = of["route_losses"]
        state["route"] = torch.stack(losses).mean() if losses else lps.sum() * 0
    return lps, state


def end_replay(state):
    """Restore what replay changed in Global_data (after backward)."""
    Global_data.train_attn, Global_data.opsd_flex, Global_data.stage1 = state.pop("_saved")


def reverse_kl(student_lp: torch.Tensor, teacher_lp: torch.Tensor) -> torch.Tensor:
    """Per-token KL(p_S || p_T) from log-probabilities [n, vocab]."""
    return (student_lp.exp() * (student_lp - teacher_lp)).sum(-1)


@torch.no_grad()
def teacher_logprobs(plain, ctx_ids, stream, vocab: int) -> torch.Tensor:
    """The base model on the raw text: [n_raw, vocab] for the stream's raw tokens."""
    gset = set(Global_data.gist_token_id)
    raw_ctx = [t for t in ctx_ids if t not in gset]
    raw_y = [t for t in stream if t not in gset]
    dev = next(plain.parameters()).device
    seq = torch.tensor([raw_ctx + raw_y[:-1]], device=dev)
    # logits of the generated rows only: over a 128K prompt they would take ~77 GB
    h = plain.model(input_ids=seq).last_hidden_state[0, len(raw_ctx) - 1:]
    return _logprobs(plain.lm_head(h), vocab)
