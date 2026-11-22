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
        self.last_hidden = out.last_hidden_state[0, -1]
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

    def _fast_ok(self) -> bool:
        """The DECODE_SELECT=beam CUDA path can skip model.forward (see _decode_fast)."""
        import os

        m = self.model
        return (self.tree is not None and os.environ.get("DECODE_SELECT") == "beam"
                and os.environ.get("ATTN_IMPL", "sparse") != "ref"
                and os.environ.get("GIST_FAST_DECODE", "1") == "1"
                and type(m).__name__ == "GistQwen2ForCausalLM" and getattr(m, "selective", False)
                and m.model.embed_tokens.weight.dtype == torch.bfloat16)

    def _decode_graph(self, ids, vis) -> torch.Tensor:
        """A raw decode row with each layer's dense part replayed from a CUDA graph
        (_DecodeGraphs); the cache update and attention run between them. Same kernels as
        _decode_fast, bit-identical (tests/test_beam_kernels.py)."""
        from gist_beam import beam_decode
        from gist_beam_cuda import row_decode

        g = Global_data
        G = getattr(self.model, "_decode_graphs", None)
        if G is None:
            G = self.model._decode_graphs = _DecodeGraphs(self.model)
        rope_pos = g.last_pos + 1
        g.last_pos = rope_pos
        G.tok.copy_(ids[:, -1:])
        G.pos.copy_(rope_pos)
        G.graphs[0].replay()
        rec = g.opsd_record
        layers = self.model.model.layers
        for li in range(G.L):
            a = layers[li].self_attn
            k, v = self.past.update(G.k[li], G.v[li], li, {"sin": G.sin, "cos": G.cos})
            if li == 0:
                o, idx = row_decode(G.q[li], k, v, vis, None, a.scaling)
                if rec is not None:
                    rec[0] = idx
            else:
                o, _ = beam_decode(a, G.q[li], k, v, ids, a.scaling)
            G.o[li].copy_(o)
            G.graphs[li + 1].replay()
        return G.logits.clone()  # the next replay overwrites it

    def _decode_fast(self, i: int, ids, vis) -> torch.Tensor:
        """Fed token i through the layers as model.forward would on the beam path, without
        its per-step O(T) work (the [T] causal and additive masks, the tmp_ids concatenation,
        isin over the sequence): layer 0 and summary rows attend by their gist-mask row vis
        (kernel B), raw rows of the other layers by beam selection. Bit-identical to
        model.forward (tests/test_beam_kernels.py). Returns the logits [V]."""
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

        from gist_beam import beam_decode
        from gist_beam_cuda import row_decode
        from gist_lora import extend_beta_cache

        import os

        g = Global_data
        m = self.model.model
        tok = self.ids[i]
        summ = tok in self.gist_ids
        if not summ and os.environ.get("GIST_DECODE_GRAPH", "1") == "1":
            return self._decode_graph(ids, vis)
        x = m.embed_tokens(ids[:, -1:])
        if g.lora_active:
            g.lora_row_mask = torch.tensor([[summ]], device=self.device)
            g.lora_row_level = torch.tensor([[self.gist_ids.index(tok) if summ else -1]],
                                            device=self.device)
        rope_pos = g.last_pos + (0 if summ else 1)
        g.last_pos = rope_pos
        cos, sin = m.rotary_emb(x, rope_pos)
        rec = g.opsd_record
        for layer in m.layers[: m.config.num_hidden_layers]:
            a = layer.self_attn
            res = x
            h = layer.input_layernorm(x)
            shape = (1, 1, -1, a.head_dim)
            q = a.q_proj(h).view(shape).transpose(1, 2)
            k = a.k_proj(h).view(shape).transpose(1, 2)
            v = a.v_proj(h).view(shape).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k, v = self.past.update(k, v, layer.layer_idx, {"sin": sin, "cos": cos})
            if layer.layer_idx == 0 or summ:
                beta = None
                if layer.layer_idx > 0:
                    a._gist_decode_cache = None
                    if getattr(a, "summary_bias", None) is not None:
                        a._bias_x = h
                        beta = extend_beta_cache(a, k.shape[2], 1)[0]
                o, idx = row_decode(q, k, v, vis, beta, a.scaling)
                if rec is not None:
                    rec[layer.layer_idx] = idx
            else:
                o, _ = beam_decode(a, q, k, v, ids, a.scaling)
            x = res + a.o_proj(o.reshape(1, 1, -1).contiguous())
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.model.lm_head(m.norm(x))[0, -1]

    @torch.no_grad()
    def step(self, token: int, turn: Optional[int] = None, record: bool = False) -> torch.Tensor:
        """Decode one raw token and the summaries the streaming rule puts after it; returns
        the logits of the last fed row [V], the prediction of the next raw token (a summary
        row predicts the raw token after it, as the training data has it). record: self.last_records gets, per fed
        token, the key positions each layer attended to (gist_opsd_long.py)."""
        turn = self.turn[-1] if turn is None and self.turn else (turn or 0)
        start = len(self.ids)
        self._push([token], [False], [turn], extend_tree=False)  # extended as each token is fed
        logits = None
        self.last_records = []
        saved = Global_data.decode_gists
        fast = self._fast_ok()
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
                Global_data.opsd_record = {} if record else None
                if fast:
                    logits = self._decode_fast(i, ids, self._row(ids, pin, tr)[0, 0, 0])
                    self.model.tmp_ids = ids  # as model.forward leaves it
                else:
                    out = self.model(
                        input_ids=ids[:, -1:], attention_mask=torch.ones_like(ids),
                        attention_mask_gist=self._row(ids, pin, tr), past_key_values=self.past,
                        use_cache=True)
                    self.past = out.past_key_values
                    logits = out.logits[0, -1]
                if record:
                    self.last_records.append(Global_data.opsd_record)
        finally:
            Global_data.decode_gists = saved
            Global_data.decoding_summary = None
            Global_data.beam_incremental = None
            Global_data.opsd_record = None
        return logits


class _DecodeGraphs:
    """CUDA graphs of the dense part of a raw decode row (#25), kept on the model: graph 0
    is the embedding, RoPE and layer 0's pre-attention part (norm, q/k/v, RoPE); graph
    i (1..L) is layer i-1's post-attention part (o_proj, residual, norm, MLP, residual)
    and layer i's pre-attention part, the last one the final norm and lm_head. Inputs and
    outputs are static tensors; the attention between them runs eagerly, so the shapes
    that grow with the sequence stay out of the graphs. Weights updated in place (the
    optimizer) are seen by the graphs; a raw row has no gist LoRA delta and no beta."""

    def __init__(self, model):
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

        g = Global_data
        m = model.model
        self.L = m.config.num_hidden_layers
        layers = m.layers[: self.L]
        dev = m.embed_tokens.weight.device
        a0 = layers[0].self_attn
        H, Hkv, D = m.config.num_attention_heads, m.config.num_key_value_heads, a0.head_dim
        self.tok = torch.zeros(1, 1, dtype=torch.long, device=dev)
        self.pos = torch.zeros(1, 1, dtype=torch.long, device=dev)
        dt = m.embed_tokens.weight.dtype
        self.o = [torch.zeros(1, 1, H, D, dtype=dt, device=dev) for _ in range(self.L)]
        self.q, self.k, self.v, res = [None] * self.L, [None] * self.L, [None] * self.L, [None] * self.L

        def pre(li, x):
            a = layers[li].self_attn
            h = layers[li].input_layernorm(x)
            shape = (1, 1, -1, a.head_dim)
            q = a.q_proj(h).view(shape).transpose(1, 2)
            k = a.k_proj(h).view(shape).transpose(1, 2)
            v = a.v_proj(h).view(shape).transpose(1, 2)
            self.q[li], self.k[li] = apply_rotary_pos_emb(q, k, self.cos, self.sin)
            self.v[li], res[li] = v, x

        def first():
            x = m.embed_tokens(self.tok)
            self.cos, self.sin = m.rotary_emb(x, self.pos)
            pre(0, x)

        def piece(li):  # post of layer li - 1, then pre of layer li (or the head)
            def run():
                lay = layers[li - 1]
                x = res[li - 1] + lay.self_attn.o_proj(self.o[li - 1].reshape(1, 1, -1).contiguous())
                x = x + lay.mlp(lay.post_attention_layernorm(x))
                if li < self.L:
                    pre(li, x)
                else:
                    self.logits = model.lm_head(m.norm(x))[0, -1]
            return run

        fns = [first] + [piece(li) for li in range(1, self.L + 1)]
        saved = g.decoding_summary
        g.decoding_summary = False  # the raw-row branch of the gist LoRA (gist_lora.py)
        try:
            with torch.no_grad():
                s = torch.cuda.Stream(dev)
                s.wait_stream(torch.cuda.current_stream(dev))
                with torch.cuda.stream(s):  # warm-up (cuBLAS handles, allocator) off the graph
                    for fn in fns:
                        fn()
                torch.cuda.current_stream(dev).wait_stream(s)
                self.pool = torch.cuda.graph_pool_handle()
                self.graphs = []
                for fn in fns:
                    gr = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gr, pool=self.pool):
                        fn()
                    self.graphs.append(gr)
        finally:
            g.decoding_summary = saved
