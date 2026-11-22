"""One conversation on one KV cache: append raw segments, decode, append again (#18).

ADR 0001: an agent turn appends a tool result (1K-50K tokens) and then generates, so the
cache lives across turns. `append` gists the new raw tokens by the streaming rule,
continuing the inserter of earlier appends (prefix-stable, so appending in pieces gives
the same sequence as gisting everything at once), and prefills them with the gist mask
(no unfolding) through flex against the whole cache: O(T) mask features, never a dense
[q, T] mask. Segment labels (pinned, turn) are carried for the mask (R6, R7) and for
DECODE_SELECT=beam. `step` decodes one raw token on the existing decode paths and feeds
the summaries the inserter emits after it.

The model must be built with attn_implementation=gist_flex_attention.GIST_FLEX; decode
steps go through select_forward / attn_dispatcher and do not use it.
"""

from typing import List, Optional, Sequence, Union

import torch

from gist_utils import Global_data
from src.data.gist_flex_mask import build_gist_mask_features, make_gist_mask_mod
from src.data.gist_stream import OnlineGistInserter


class GistSession:
    def __init__(self, model):
        g = Global_data
        self.model = model
        self.gist_ids = list(g.gist_token_id)
        self.ins = OnlineGistInserter(self.gist_ids, g.chunk_size[0], g.chunk_size[1:] or 4,
                                      g.sink_size)
        self.ids: List[int] = []
        self.pinned: List[bool] = []
        self.turn: List[int] = []
        self.past = None
        self.device = next(model.parameters()).device
        self._buf = torch.zeros(3, 0, dtype=torch.long, device=self.device)  # ids, pinned, turn
        self._n_raw = []  # raw tokens up to and including position i
        self.tree = None
        if self.device.type == "cuda":  # kept token by token for the beam fast path (#20)
            from gist_beam_cuda import IncrementalTree

            self.tree = IncrementalTree(self.gist_ids, g.sink_size, self.device)

    # -- bookkeeping ---------------------------------------------------------------------

    def _push(self, raw: Sequence[int], pinned: Sequence[bool], turn: Sequence[int],
              extend_tree: bool = True) -> int:
        """Gist raw tokens onto the stream; returns how many tokens (raw + summaries) were added."""
        n0 = len(self.ids)
        gset = set(self.gist_ids)
        for t, p, tr in zip(raw, pinned, turn):
            self.ids.append(int(t))
            self.pinned.append(bool(p))
            self.turn.append(int(tr))
            for g in self.ins.push(int(t), bool(p)):
                self.ids.append(g)
                self.pinned.append(False)
                self.turn.append(int(tr))  # a summary takes the turn of the raw token before it
        n = (self._n_raw[-1] if self._n_raw else 0)
        for t in self.ids[n0:]:
            n += t not in gset
            self._n_raw.append(n)
        new = torch.tensor([self.ids[n0:], self.pinned[n0:], self.turn[n0:]], dtype=torch.long,
                           device=self.device)
        self._buf = torch.cat([self._buf, new], 1)  # ponytail: amortized copy, grows the buffer each push
        if self.tree is not None and extend_tree:
            self.tree.extend(self.ids[n0:], self.pinned[n0:], self.turn[n0:])
        return len(self.ids) - n0

    def _last_pos(self, upto: int) -> torch.Tensor:
        """RoPE position of token upto-1: raw tokens keep their raw index, a summary shares
        the position of the raw token before it. Computed from this session's own ids, not
        the global Global_data.last_pos, so sessions can interleave."""
        return torch.tensor([[(self._n_raw[upto - 1] if upto else 0) - 1]], device=self.device)

    def _tensors(self):
        b = self._buf
        return b[0:1], b[1:2].bool(), b[2:3]

    # -- prefill -------------------------------------------------------------------------

    @torch.no_grad()
    def append(self, raw: Sequence[int], pinned: Union[bool, Sequence[bool]] = False,
               turn: Union[int, Sequence[int]] = 0) -> torch.Tensor:
        """Append raw tokens (pinned / turn per token or for all); returns the last hidden
        state of the new tokens [q, d] (summary rows included)."""
        n = len(raw)
        pinned = [pinned] * n if isinstance(pinned, bool) else list(pinned)
        turn = [turn] * n if isinstance(turn, int) else list(turn)
        past_len = len(self.ids)
        q = self._push(raw, pinned, turn)
        ids, pin, tr = self._tensors()
        new = ids[:, past_len:]
        pos = None
        if past_len:
            pos = self._last_pos(past_len) + 1 + torch.arange(q, device=ids.device)[None]
        saved = Global_data.train_attn
        Global_data.train_attn = "flex"
        try:
            out = self.model.model(
                input_ids=new, attention_mask=torch.ones_like(ids), past_key_values=self.past,
                position_ids=pos, use_cache=True, gist_full_ids=ids, gist_pinned=pin, gist_turn=tr)
        finally:
            Global_data.train_attn = saved
        self.past = out.past_key_values
        self.model.tmp_ids = ids  # what decode steps extend
        return out.last_hidden_state[0]

    # -- decode --------------------------------------------------------------------------

    def _row(self, ids, pin, tr) -> torch.Tensor:
        """The gist-mask row of the last token, [1, 1, 1, T] (the prefill mask's row)."""
        g = Global_data
        f = build_gist_mask_features(ids, self.gist_ids, g.chunk_size[0], g.sink_size,
                                     g.num_previous_chunks, pad_token=g.pad_token_id,
                                     raw_window=g.raw_window, pinned=pin, turn=tr,
                                     window_turns=g.window_turns)
        T = ids.shape[1]
        z = torch.zeros((), dtype=torch.long, device=ids.device)
        return make_gist_mask_mod(f)(z, z, z + T - 1, torch.arange(T, device=ids.device)).view(1, 1, 1, T)

    @torch.no_grad()
    def step(self, token: int, turn: Optional[int] = None) -> torch.Tensor:
        """Decode one raw token and the summaries the streaming rule puts after it; returns
        the logits of the raw token's row [V]."""
        turn = self.turn[-1] if turn is None and self.turn else (turn or 0)
        start = len(self.ids)
        self._push([token], [False], [turn], extend_tree=False)  # extended as each token is fed
        logits = None
        saved = Global_data.decode_gists
        Global_data.decode_gists = True  # summaries share the position of the raw token before
        try:
            for i in range(start, len(self.ids)):
                ids, pin, tr = (x[:, : i + 1] for x in self._tensors())
                Global_data.beam_segments = (pin[0], tr[0])
                Global_data.last_pos = self._last_pos(i)
                Global_data.decoding_summary = self.ids[i] in self.gist_ids
                if self.tree is not None:
                    self.tree.extend([self.ids[i]], [self.pinned[i]], [self.turn[i]])
                Global_data.beam_incremental = self.tree
                out = self.model(
                    input_ids=ids[:, -1:], attention_mask=torch.ones_like(ids),
                    attention_mask_gist=self._row(ids, pin, tr), past_key_values=self.past,
                    use_cache=True)
                self.past = out.past_key_values
                if i == start:
                    logits = out.logits[0, -1]
        finally:
            Global_data.decode_gists = saved
            Global_data.decoding_summary = None
            Global_data.beam_incremental = None
        return logits
