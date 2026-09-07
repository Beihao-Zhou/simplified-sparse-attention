"""End-to-end resident vs pinned-host-offload decode on the real Gist model.

Runs the same prompt through both paths with greedy decoding and reports
generated-token agreement, per-phase latency, GPU peak memory and pinned host
memory. The offload path is selected by handing `generate` a GistOffloadCache,
which is the only difference between the two runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis.record_selection import load_prompt_file, passkey_prompt  # noqa: E402


class StepTimer:
    """One call per generated token -- generate() invokes this per step."""

    def __init__(self):
        self.t = []
        self.last_scores = None
        self.scores = []
        self.decode_base = None
        self.prefill_peak = 0

    def __call__(self, input_ids, scores):
        torch.cuda.synchronize()
        if not self.t:
            # First call == end of prefill. Peak memory over the whole run is set
            # by prefill (which is identical in both modes at long context), so
            # restart the counter here to see what *decode* actually holds.
            self.prefill_peak = torch.cuda.max_memory_allocated()
            self.decode_base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        self.t.append(time.perf_counter())
        self.last_scores = scores.detach().float().cpu()
        self.scores.append(self.last_scores)
        return scores


class SuppressGist:
    def __init__(self, ids):
        self.ids = [int(i) for i in ids]

    def __call__(self, input_ids, scores):
        scores[:, self.ids] = float("-inf")
        return scores


def build_inputs(args, tok, Global_data, gist):
    if args.workload == "passkey":
        prompt, head = passkey_prompt(args.n_garbage)
        ids = tok([prompt]).input_ids[0]
        head_ids = tok([head]).input_ids[0]
        prefix, doc, suffix = ids[:len(head_ids)], ids[len(head_ids):-10], ids[-10:]
    else:
        prompt, tail, _ = load_prompt_file(args.prompt_file)
        ids = tok(prompt).input_ids
        tail_ids = tok(tail, add_special_tokens=False).input_ids
        prefix, doc, suffix = [], ids[:-len(tail_ids)], tail_ids
        if args.max_prompt_tokens and len(doc) > args.max_prompt_tokens:
            doc = doc[:args.max_prompt_tokens]
    doc_with_gist = gist.insert_hierarchical_gist_tokens(
        doc, Global_data.gist_token_id, Global_data.chunk_size[0],
        Global_data.chunk_size[1], tok, Global_data.sink_size)
    seq = list(prefix) + list(doc_with_gist) + list(suffix)
    input_ids = torch.tensor([seq])
    mask_gist = gist.make_hierarchical_gist_mask_greedy(
        inputs=input_ids, gist_token=Global_data.gist_token_id,
        chunk_size=Global_data.chunk_size[0], attention_sink_size=-1,
        num_previous_chunks=Global_data.num_previous_chunks,
        pad_token=tok.pad_token_id, add_raw=Global_data.add_raw)
    return input_ids, torch.ones_like(input_ids), mask_gist


def run(model, tok, ids, am, mg, steps, offload, batch, gist_ids,
        hot_buffer_size=0):
    from transformers import LogitsProcessorList

    from gist_offload_cache import GistOffloadCache

    if batch > 1:
        ids, am, mg = ids.repeat(batch, 1), am.repeat(batch, 1), mg.repeat(batch, 1, 1, 1)
    for m in model.modules():                      # drop the per-prefix decode cache
        if hasattr(m, "_gist_decode_cache"):
            m._gist_decode_cache = None
    timer = StepTimer()
    # NB: return_dict_in_generate/output_scores hit a stale import in this repo's
    # generation_utils._sample, so the last-step logits are captured by the timer
    # logits processor instead.
    kw = dict(max_new_tokens=steps, min_new_tokens=steps, do_sample=False, num_beams=1,
              pad_token_id=tok.eos_token_id,
              logits_processor=LogitsProcessorList([SuppressGist(gist_ids), timer]))
    cache = None
    if offload:
        cache = GistOffloadCache(config=model.config, max_new_tokens=steps + 8,
                                 hot_buffer_size=hot_buffer_size)
        kw["past_key_values"] = cache
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(input_ids=ids.cuda(), attention_mask=am.cuda(),
                             attention_mask_gist=mg.cuda(), **kw)
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    decode_peak = torch.cuda.max_memory_allocated()
    peak = max(timer.prefill_peak, decode_peak) - base
    ts = timer.t
    prefill_ms = (ts[0] - t0) * 1000 if ts else float("nan")
    first_ms = (ts[1] - ts[0]) * 1000 if len(ts) > 1 else float("nan")
    steady = [(ts[i] - ts[i - 1]) * 1000 for i in range(2, len(ts))]
    steady_ms = sum(steady) / len(steady) if steady else float("nan")
    # Always present, so both modes write the same CSV columns.
    selected = misses = 0
    if offload and hot_buffer_size:
        selected, misses = cache.hot_counts()
        print(
            f"[e2e] hot buffer selected={selected} misses={misses} "
            f"hit={1.0 - misses / max(1, selected):.3%}",
            flush=True,
        )
    return dict(
        hot_size=hot_buffer_size if offload else 0,
        hot_selected=selected, hot_misses=misses,
        hot_hit_rate=(1.0 - misses / selected) if selected else float("nan"),
        total_s=total, prefill_ms=prefill_ms, first_decode_ms=first_ms,
        steady_ms_per_token=steady_ms,
        tokens_per_s=(len(ts) - 1) * ids.shape[0] / max(1e-9, total - (ts[0] - t0)),
        peak_gpu_MiB=peak / 2**20,
        decode_peak_gpu_MiB=decode_peak / 2**20,
        decode_resident_MiB=(timer.decode_base or 0) / 2**20,
        pinned_MiB=getattr(cache, "pinned_bytes", 0) / 2**20,
        tokens=out[:, ids.shape[1]:].cpu(),
        last_logits=timer.last_scores,
        per_step_logits=timer.scores,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gist-sparse-attention/GSA-FT-Qwen2-7B-Instruct-chunk8-chunk4")
    ap.add_argument("--workload", default="passkey", choices=["passkey", "prompt_file"])
    ap.add_argument("--prompt-file", default="")
    ap.add_argument("--n-garbage", type=int, default=18400)
    ap.add_argument("--max-prompt-tokens", type=int, default=0)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--batches", type=int, nargs="+", default=[1])
    ap.add_argument("--hot-buffer-size", type=int, default=0)
    ap.add_argument("--control", action="store_true",
                    help="run the RESIDENT path twice and compare it against itself; "
                         "the compaction orders idx_list by a racing atomicAdd, so this "
                         "is the run-to-run noise floor any offload difference sits on")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="analysis/results/e2e_offload.csv")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from gist_qwen2 import GistQwen2ForCausalLM
    from gist_utils import Global_data
    from src.data import gist

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = GistQwen2ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).eval().cuda()
    model.selective = True
    if tok.pad_token is None:
        tok.pad_token, tok.pad_token_id = tok.eos_token, tok.eos_token_id
    Global_data.gist_token_id = tok.additional_special_tokens_ids[:2]
    Global_data.gist_token = tok.additional_special_tokens[:2]
    Global_data.top_k = [3, 3, 3]
    Global_data.top_p = 0
    Global_data.sink_size = 3
    Global_data.pad_token_id = tok.pad_token_id
    Global_data.use_nsa_gqa = True
    Global_data.use_grouped_sdpa_optimization = False
    Global_data.chunk_size = [8, 4]
    Global_data.num_previous_chunks = 1
    Global_data.sep_num = 0
    Global_data.add_raw = False
    Global_data.kl_loss = False

    ids, am, mg = build_inputs(args, tok, Global_data, gist)
    ctx = ids.shape[1]
    print(f"[e2e] context tokens (with gists) = {ctx}", flush=True)

    # Warm both paths first: the CUDA extension JIT-loads, the selector compiles and
    # cuDNN autotunes on the first call, and whichever mode ran first would
    # otherwise absorb all of that and look several times slower.
    for mode in ("resident", "offload"):
        run(model, tok, ids, am, mg, 4, mode == "offload", min(args.batches),
            Global_data.gist_token_id,
            hot_buffer_size=args.hot_buffer_size if mode == "offload" else 0)
        torch.cuda.empty_cache()
    print("[e2e] warmup done", flush=True)

    rows = []
    for B in args.batches:
        res = {}
        modes = ("resident", "resident2") if args.control else ("resident", "offload")
        for mode in modes:
            r = run(model, tok, ids, am, mg, args.steps, mode == "offload", B,
                    Global_data.gist_token_id,
                    hot_buffer_size=args.hot_buffer_size if mode == "offload" else 0)
            res[mode] = r
            print(f"[e2e] B={B} {mode:8s} prefill {r['prefill_ms']:8.1f} ms | "
                  f"first {r['first_decode_ms']:7.2f} ms | steady "
                  f"{r['steady_ms_per_token']:6.2f} ms/tok | "
                  f"{r['tokens_per_s']:6.1f} tok/s | peak GPU "
                  f"{r['peak_gpu_MiB']:8.1f} MiB | decode-phase peak "
                  f"{r['decode_peak_gpu_MiB']:8.1f} MiB | resident-at-decode "
                  f"{r['decode_resident_MiB']:8.1f} MiB | pinned {r['pinned_MiB']:8.1f} MiB",
                  flush=True)
            torch.cuda.empty_cache()
        a, b = res[modes[0]], res[modes[1]]
        match = int((a["tokens"] == b["tokens"]).all())
        agree = float((a["tokens"] == b["tokens"]).float().mean())
        first_div = int((a["tokens"] != b["tokens"]).float().argmax()) if not match else -1
        # SuppressGist writes -inf into the gist columns, so -inf - -inf = nan;
        # compare only the entries that are finite in both runs.
        def dmax(la, lb):
            fin = torch.isfinite(la) & torch.isfinite(lb)
            return (la[fin] - lb[fin]).abs().max().item() if fin.any() else float("nan")

        lerr = dmax(a["last_logits"], b["last_logits"])
        # Greedy decoding is a hard argmax: once one near-tied step flips, the two
        # runs generate different text and every later logit comparison is between
        # different prefixes. What actually tests correctness is the logit gap
        # while the prefixes still agree, against the top-1/top-2 margin at the
        # step that flipped.
        n_agree = first_div if first_div >= 0 else len(a["per_step_logits"])
        pre = [dmax(a["per_step_logits"][i], b["per_step_logits"][i])
               for i in range(min(n_agree + 1, len(a["per_step_logits"]),
                                  len(b["per_step_logits"])))]
        pre_err = max(pre) if pre else float("nan")
        margin = float("nan")
        if first_div >= 0 and first_div < len(a["per_step_logits"]):
            v = a["per_step_logits"][first_div][0]
            v = v[torch.isfinite(v)]
            top2 = v.topk(2).values
            margin = (top2[0] - top2[1]).item()
        print(f"[e2e] B={B} token match: {bool(match)} (agreement {agree:.3f}, "
              f"first divergence idx {first_div}), max|Δlogits| while prefixes "
              f"agree = {pre_err:.4e}, top1-top2 margin at divergence = {margin:.4e}, "
              f"max|Δlast logits| = {lerr:.4e}",
              flush=True)
        for mode in modes:
            r = dict(res[mode])
            for k in ("tokens", "last_logits", "per_step_logits"):
                r.pop(k)
            rows.append(dict(tag=args.tag, context=ctx, steps=args.steps, batch=B,
                             mode=mode,
                             token_match=match, token_agreement=agree,
                             first_divergence=first_div,
                             max_abs_logit_diff=lerr,
                             max_logit_diff_pre_divergence=pre_err,
                             argmax_margin_at_divergence=margin, **r))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    new = not Path(args.out).exists()
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"[e2e] appended {len(rows)} rows to {args.out}")
    print(json.dumps({k: v for k, v in rows[-1].items()}, indent=2, default=str))


if __name__ == "__main__":
    main()
