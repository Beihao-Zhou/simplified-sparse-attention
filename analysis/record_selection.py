"""Record the exact per-decode-step selected-KV set of the real Gist model.

The recorder wraps the sparse-decode CUDA extension and reconstructs, for every
(step, layer, batch, KV-group), the *same* logical key set that
``build_compact_idx_kernel`` compacts:

    keep = (sel0 & compressed & !is_gist)     # selected compressed raw tokens
         | (is_l0_gist & sel0)                # selected level-0 gist tokens
         | (is_l1_gist & sel1)                # selected level-1 (meta) gist tokens
         | (attn_mask & !is_gist)             # always-attended raw suffix + sinks

Nothing is approximated: the very tensors handed to the kernel are used, so
level-0/level-1 selection, gist vs raw tokens, the compressed range, the
always-attended suffix, the attention mask, every layer, every KV group and the
batch dimension are all accounted for by construction.

Traces are written as one compressed .npz per run and consumed by
analysis/locality_report.py.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


# --------------------------------------------------------------------------
# recorder
# --------------------------------------------------------------------------

class SelectionRecorder:
    """Drop-in replacement for the loaded CUDA extension module.

    Layer identity is read from the caller's frame (``_decode_forward`` holds
    the attention ``module``) rather than inferred from a call counter, so a
    layer that ever takes a different code path cannot silently shift the
    layer axis of the trace.
    """

    def __init__(self, ext):
        self._ext = ext
        self.calls = 0
        self.step_of_layer = {}
        self.rec_step, self.rec_layer, self.rec_b, self.rec_g = [], [], [], []
        self.rec_len, self.rec_recent, self.rec_klen = [], [], []
        self.keys = []

    def _layer_idx(self):
        f = sys._getframe(2)
        for _ in range(6):
            mod = f.f_locals.get("module")
            if mod is not None and hasattr(mod, "layer_idx"):
                return int(mod.layer_idx)
            if f.f_back is None:
                break
            f = f.f_back
        raise RuntimeError("could not locate the attention module in the call stack")

    def sparse_decode_attn_compact(self, q, K, V, sel0, sel1, c0, c1,
                                   is_gist, is_l0g, is_l1g, comp, am, bias,
                                   scaling, nsplit, cap):
        layer = self._layer_idx()
        step = self.step_of_layer.get(layer, 0)
        self.step_of_layer[layer] = step + 1
        self.calls += 1

        B, num_kv = sel0.shape[0], sel0.shape[1]
        k_len = is_gist.shape[-1]
        c0l, c1l = c0.long(), c1.long()
        ig, il0, il1 = is_gist.bool(), is_l0g.bool(), is_l1g.bool()
        cb, ab = comp.bool(), am.bool()
        for b in range(B):
            s0 = sel0[b].bool()[:, c0l[b]]          # [num_kv, k_len]
            s1 = sel1[b].bool()[:, c1l[b]]
            keep = ((s0 & (cb[b] & ~ig[b])) | (il0[b] & s0)
                    | (il1[b] & s1) | (ab[b] & ~ig[b]))
            recent = keep & (ab[b] & ~ig[b])
            counts = keep.sum(1).to(torch.int32)
            rcounts = recent.sum(1).to(torch.int32)
            idx = torch.nonzero(keep, as_tuple=False)[:, 1].to(torch.int32)
            counts_c = counts.cpu().numpy()
            rcounts_c = rcounts.cpu().numpy()
            idx_c = idx.cpu().numpy()
            off = 0
            for g in range(num_kv):
                n = int(counts_c[g])
                self.keys.append(idx_c[off:off + n].copy())
                off += n
                self.rec_step.append(step)
                self.rec_layer.append(layer)
                self.rec_b.append(b)
                self.rec_g.append(g)
                self.rec_len.append(n)
                self.rec_recent.append(int(rcounts_c[g]))
                self.rec_klen.append(k_len)
        return self._ext.sparse_decode_attn_compact(
            q, K, V, sel0, sel1, c0, c1, is_gist, is_l0g, is_l1g, comp, am,
            bias, scaling, nsplit, cap)

    def __getattr__(self, name):        # everything else passes straight through
        return getattr(self._ext, name)

    def save(self, path, meta):
        np.savez_compressed(
            path,
            keys=np.concatenate(self.keys) if self.keys else np.zeros(0, np.int32),
            lens=np.asarray(self.rec_len, np.int32),
            step=np.asarray(self.rec_step, np.int32),
            layer=np.asarray(self.rec_layer, np.int16),
            b=np.asarray(self.rec_b, np.int16),
            g=np.asarray(self.rec_g, np.int16),
            recent=np.asarray(self.rec_recent, np.int32),
            k_len=np.asarray(self.rec_klen, np.int32),
            meta=np.array(json.dumps(meta)),
        )


class _ForceLength:
    """Suppress EOS so a trace reaches a target length. Marks the run synthetic."""

    def __init__(self, eos_ids, gist_ids):
        self.ids = [int(i) for i in list(eos_ids) + list(gist_ids)]

    def __call__(self, input_ids, scores):
        scores[:, self.ids] = float("-inf")
        return scores


# --------------------------------------------------------------------------
# workloads
# --------------------------------------------------------------------------

def passkey_prompt(n_garbage, seed=0):
    """Same construction as benchmark/Passkey_exp/passkey_pred_qwen.py."""
    rng = random.Random(seed)
    n_prefix = int(n_garbage * 0.5)
    task = ("There is an important info hidden inside a lot of irrelevant text. "
            "Find it and memorize them. I will quiz you about the important "
            "information there.")
    garbage = ("The grass is green. The sky is blue. The sun is yellow. Here we go. "
               "There and back again.")
    inf = " ".join([garbage] * 20000)
    pass_key = rng.randint(1, 50000)
    info = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    prompt = "\n".join([task, inf[:n_prefix], info, inf[:n_garbage - n_prefix]])
    return prompt + "What is the pass key? The pass key is", task


def load_prompt_file(path):
    """Prompt prepared by analysis/prepare_prompt.py (see the note there)."""
    with open(path) as f:
        d = json.load(f)
    return d["prompt"], d["tail"], d


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gist-sparse-attention/GSA-FT-Qwen2-7B-Instruct-chunk8-chunk4")
    ap.add_argument("--workload", default="passkey",
                    choices=["passkey", "prompt_file"])
    ap.add_argument("--prompt-file", default="",
                    help="JSON written by analysis/prepare_prompt.py")
    ap.add_argument("--n-garbage", type=int, default=40000)
    ap.add_argument("--max-prompt-tokens", type=int, default=0,
                    help="truncate the raw prompt to this many tokens before gist insertion")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--force-steps", action="store_true",
                    help="suppress EOS to reach --steps; marks the trace synthetic")
    ap.add_argument("--top-k", type=int, nargs="+", default=[3])
    ap.add_argument("--chunk-size", type=int, nargs="+", default=[8, 4])
    ap.add_argument("--sink-size", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    random.seed(0)
    torch.manual_seed(0)

    from transformers import AutoTokenizer, LogitsProcessorList

    import attn_candidate
    from gist_qwen2 import GistQwen2ForCausalLM
    from gist_utils import Global_data
    from src.data import gist

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = GistQwen2ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).eval().cuda()
    model.selective = True   # routes decode through select_forward -> sparse CUDA kernel
    if tok.pad_token is None:
        tok.pad_token, tok.pad_token_id = tok.eos_token, tok.eos_token_id
    tok.padding_side = "left"

    Global_data.gist_token_id = tok.additional_special_tokens_ids[:2]
    Global_data.gist_token = tok.additional_special_tokens[:2]
    Global_data.top_k = (args.top_k * 3)[:3]
    Global_data.top_p = 0
    Global_data.sink_size = args.sink_size
    Global_data.pad_token_id = tok.pad_token_id
    Global_data.use_nsa_gqa = True
    Global_data.use_grouped_sdpa_optimization = False
    Global_data.chunk_size = args.chunk_size
    Global_data.num_previous_chunks = 1
    Global_data.sep_num = 0
    Global_data.add_raw = False
    Global_data.kl_loss = False

    # ---- build the prompt, then splice in hierarchical gist tokens ----
    if args.workload == "passkey":
        prompt, head = passkey_prompt(args.n_garbage)
        ids = tok([prompt]).input_ids[0]
        head_ids = tok([head]).input_ids[0]
        prefix, doc, suffix = ids[:len(head_ids)], ids[len(head_ids):-10], ids[-10:]
    else:
        prompt, after, pinfo = load_prompt_file(args.prompt_file)
        ids = tok(prompt).input_ids
        tail_ids = tok(after, add_special_tokens=False).input_ids
        doc, suffix = ids[:-len(tail_ids)], tail_ids
        if args.max_prompt_tokens and len(doc) > args.max_prompt_tokens:
            doc = doc[:args.max_prompt_tokens]
        prefix = []

    doc_with_gist = gist.insert_hierarchical_gist_tokens(
        doc, Global_data.gist_token_id, Global_data.chunk_size[0],
        Global_data.chunk_size[1], tok, Global_data.sink_size)
    seq = list(prefix) + list(doc_with_gist) + list(suffix)
    input_ids = torch.tensor([seq])
    attn_mask = torch.ones_like(input_ids)
    mask_gist = gist.make_hierarchical_gist_mask_greedy(
        inputs=input_ids, gist_token=Global_data.gist_token_id,
        chunk_size=Global_data.chunk_size[0], attention_sink_size=-1,
        num_previous_chunks=Global_data.num_previous_chunks,
        pad_token=tok.pad_token_id, add_raw=Global_data.add_raw)
    ctx_len = input_ids.shape[-1]
    print(f"[trace] context tokens (with gists) = {ctx_len}", flush=True)

    real_ext = attn_candidate._get_sparse_decode_ext()
    rec = SelectionRecorder(real_ext)
    attn_candidate._SPARSE_DECODE_EXT = rec

    gen = dict(max_new_tokens=args.steps, do_sample=False, num_beams=1,
               pad_token_id=tok.eos_token_id)
    if args.force_steps:
        gen["logits_processor"] = LogitsProcessorList(
            [_ForceLength([tok.eos_token_id], Global_data.gist_token_id)])
        gen["min_new_tokens"] = args.steps
    else:
        gen["logits_processor"] = LogitsProcessorList(
            [_ForceLength([], Global_data.gist_token_id)])

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(input_ids=input_ids.cuda(),
                             attention_mask=attn_mask.cuda(),
                             attention_mask_gist=mask_gist.cuda(), **gen)
    dt = time.time() - t0
    attn_candidate._SPARSE_DECODE_EXT = real_ext

    n_new = int(out.shape[-1] - ctx_len)
    steps_done = max(rec.step_of_layer.values()) if rec.step_of_layer else 0
    print(f"[trace] generated {n_new} tokens in {dt:.1f}s "
          f"({dt / max(1, n_new) * 1000:.0f} ms/token), "
          f"{rec.calls} kernel calls, {steps_done} decode steps, "
          f"{len(set(rec.rec_layer))} layers", flush=True)
    text = tok.decode(out[0, ctx_len:], skip_special_tokens=True)
    print(f"[trace] output: {text[:200]!r}", flush=True)

    meta = dict(
        model=args.model, workload=args.workload,
        prompt_file=args.prompt_file, n_garbage=args.n_garbage,
        context_tokens=int(ctx_len), decode_steps=int(steps_done),
        generated_tokens=n_new, requested_steps=args.steps,
        forced=bool(args.force_steps),
        synthetic=bool(args.force_steps) or args.workload == "passkey",
        num_layers=int(model.config.num_hidden_layers),
        top_k=Global_data.top_k, chunk_size=args.chunk_size,
        sink_size=args.sink_size, generate_seconds=dt,
        output_preview=text[:400],
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rec.save(args.out, meta)
    print(f"[trace] wrote {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
