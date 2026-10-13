"""GPU validation of the FlexAttention training path (fits one 24 GB L4).

What the CPU tests cannot check: the compiled Triton kernel in bf16, recompiles
across lengths, BlockMask build time at 128K+, speed and memory against dense
FlashAttention, and an end-to-end run with real weights.

    python tests/gpu_validate_flex.py                      # all sections
    python tests/gpu_validate_flex.py --sections attn mask # a subset
    python tests/gpu_validate_flex.py --device cpu --small # dry run of the script itself

Sections:
  numerics   flex (compiled, bf16) vs dense SDPA, both against an fp32 reference;
             2-level and 6-level trees, with and without the raw window
  recompile  call the attention at several lengths; count compiled graphs
  mask       BlockMask build time and memory at 32K-256K
  attn       one attention layer (7B shapes), fwd+bwd time and peak memory,
             gist flex vs causal SDPA (FlashAttention)
  e2e        Qwen2.5-0.5B (Qwen2 architecture, same tokenizer as 7B): dense vs flex
             loss and gradients at 4K, then flex-only AdamW steps at 16K/32K with the
             chunked LM loss

Results are printed and written to --out as JSON.
"""

import argparse
import copy
import gc
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gist_flex_attention  # noqa: E402
from gist_flex_attention import gist_flex_attention_forward, prepare_gist_flex  # noqa: E402
from gist_utils import Global_data  # noqa: E402
from src.data.gist import add_raw_window, make_hierarchical_gist_mask_greedy  # noqa: E402
from src.data.gist_stream import insert_gists_by_position, stream_labels  # noqa: E402

GIST = [200000 + i for i in range(6)]  # synthetic ids, outside any real vocabulary
CHUNK, SINK, PREV = 4, 2, 1
TREES = {"2-level": (2, [4]), "6-level": (6, [4, 2])}  # (levels, J per level)

RESULTS = {}


def log(section, **kv):
    RESULTS.setdefault(section, []).append(kv)
    print(f"[{section}] " + "  ".join(f"{k}={v}" for k, v in kv.items()), flush=True)


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, dev, reps=1):
    sync(dev)
    t = time.perf_counter()
    for _ in range(reps):
        out = fn()
    sync(dev)
    return out, (time.perf_counter() - t) / reps * 1000


def peak_reset(dev):
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_gb(dev):
    return round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev.type == "cuda" else None


def gisted_ids(n_tokens, levels, groups, rng, vocab=150000, gist=GIST):
    """Random raw tokens with gists inserted by the streaming rule, cut to n_tokens."""
    n_raw = n_tokens  # gists add ~1/3; over-generate then cut
    seq = insert_gists_by_position([rng.randint(10, vocab) for _ in range(n_raw)],
                                   gist[:levels], CHUNK, groups, SINK)
    return seq[:n_tokens]


def flex_attn(q, k, v, ids, levels, window):
    """The production path: features + permutation + BlockMask, then the registered op."""
    meta = prepare_gist_flex(ids, GIST[:levels], CHUNK, SINK, PREV, pad_token_id=0,
                             raw_window=window)
    out, _ = gist_flex_attention_forward(None, q, k, v, None, scaling=1 / math.sqrt(q.shape[-1]),
                                         **meta)
    return out.transpose(1, 2)  # back to [B, H, T, d]


def dense_mask(ids, levels, window):
    T = ids.shape[1]
    m = make_hierarchical_gist_mask_greedy(ids, GIST[:levels], CHUNK, attention_sink_size=SINK,
                                           num_previous_chunks=PREV, pad_token=0)
    m = add_raw_window(m, ids, GIST[:levels], window, pad_token=0)
    return m & torch.ones(T, T, dtype=torch.bool, device=ids.device).tril()


def sdpa_gqa(q, k, v, **kw):
    rep = q.shape[1] // k.shape[1]
    return F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), **kw)


# --------------------------------------------------------------------------- numerics
def numerics(dev, small):
    T, Hq, Hkv, d = (512, 4, 2, 32) if small else (4096, 28, 4, 128)
    for name, (levels, groups) in TREES.items():
        for window in (0, 256 if small else 1024):
            rng = random.Random(levels + window)
            ids = torch.tensor([gisted_ids(T, levels, groups, rng)], device=dev)
            mask = dense_mask(ids, levels, window)
            torch.manual_seed(0)
            q32, k32, v32 = (torch.randn(1, H, T, d, device=dev) for H in (Hq, Hkv, Hkv))
            g32 = torch.randn(1, Hq, T, d, device=dev)

            def run(fn, dtype):
                q, k, v = (x.to(dtype).requires_grad_() for x in (q32, k32, v32))
                out = fn(q, k, v)
                grads = torch.autograd.grad(out, (q, k, v), g32.to(dtype))
                return [out.float()] + [g.float() for g in grads]

            ref = run(lambda q, k, v: sdpa_gqa(q, k, v, attn_mask=mask), torch.float32)
            dense = run(lambda q, k, v: sdpa_gqa(q, k, v, attn_mask=mask), torch.bfloat16)
            flex = run(lambda q, k, v: flex_attn(q, k, v, ids, levels, window), torch.bfloat16)
            err = lambda a: max((x - r).abs().max().item() for x, r in zip(a, ref))  # noqa: E731
            e_d, e_f = err(dense), err(flex)
            log("numerics", tree=name, window=window, T=T, dense_bf16_err=f"{e_d:.2e}",
                flex_bf16_err=f"{e_f:.2e}", ok=e_f <= 2 * e_d + 1e-2)


# --------------------------------------------------------------------------- recompile
def recompile(dev, small):
    from torch._dynamo.utils import counters

    Hq, Hkv, d = (4, 2, 32) if small else (28, 4, 128)
    lengths = [512, 640, 1024, 1536] if small else [2048, 2176, 3072, 4096, 6144, 8192, 12288, 16384]
    levels, groups = TREES["2-level"]
    rng = random.Random(0)
    for T in lengths:
        ids = torch.tensor([gisted_ids(T, levels, groups, rng)], device=dev)
        q, k, v = (torch.randn(1, H, T, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
                   for H in (Hq, Hkv, Hkv))
        before = counters["stats"]["unique_graphs"]
        _, ms = timed(lambda: flex_attn(q, k, v, ids, levels, 0).sum().backward(), dev)
        _, ms2 = timed(lambda: flex_attn(q, k, v, ids, levels, 0).sum().backward(), dev)
        log("recompile", T=T, first_call_ms=round(ms), second_call_ms=round(ms2),
            new_graphs=counters["stats"]["unique_graphs"] - before)
    print("[recompile] expect new graphs only at the first one or two lengths; a jump in "
          "first_call_ms at every length means per-length recompiles", flush=True)


# --------------------------------------------------------------------------- mask
def mask_build(dev, small, max_len):
    lengths = [4096, 8192] if small else [n for n in (32768, 131072, 262144) if n <= max_len]
    for name, (levels, groups) in TREES.items():
        for T in lengths:
            rng = random.Random(T)
            ids = torch.tensor([gisted_ids(T, levels, groups, rng)], device=dev)
            peak_reset(dev)
            build = lambda: prepare_gist_flex(ids, GIST[:levels], CHUNK, SINK, PREV, 0,  # noqa: E731
                                              raw_window=1024)
            meta, first = timed(build, dev)
            _, again = timed(build, dev)
            bm = meta["gist_block_mask"]
            log("mask", tree=name, T=T, first_ms=round(first), steady_ms=round(again),
                computed_blocks=f"{1 - bm.sparsity() / 100:.3f}", peak_gb=peak_gb(dev))
            del meta, bm


# --------------------------------------------------------------------------- attn
def attn_bench(dev, small, max_len):
    Hq, Hkv, d = (4, 2, 32) if small else (28, 4, 128)  # Qwen2.5-7B
    lengths = [2048, 4096] if small else [n for n in (32768, 65536, 131072) if n <= max_len]
    reps = 1 if small else 3
    for T in lengths:
        mk = lambda H: torch.randn(1, H, T, d, device=dev, dtype=torch.bfloat16,  # noqa: E731
                                   requires_grad=True)
        q, k, v = mk(Hq), mk(Hkv), mk(Hkv)

        def causal():
            if dev.type == "cuda":
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            else:
                out = sdpa_gqa(q, k, v, is_causal=True)
            out.sum().backward()

        peak_reset(dev)
        causal()
        _, ms = timed(causal, dev, reps)
        log("attn", T=T, kind="causal SDPA (dense)", fwd_bwd_ms=round(ms, 1), peak_gb=peak_gb(dev))

        for name, (levels, groups) in TREES.items():
            for window in (0, 1024):
                ids = torch.tensor([gisted_ids(T, levels, groups, random.Random(T))], device=dev)
                meta = prepare_gist_flex(ids, GIST[:levels], CHUNK, SINK, PREV, 0, raw_window=window)

                def gist():
                    out, _ = gist_flex_attention_forward(None, q, k, v, None,
                                                         scaling=1 / math.sqrt(d), **meta)
                    out.sum().backward()

                peak_reset(dev)
                gist()
                _, ms = timed(gist, dev, reps)
                density = 1 - meta["gist_block_mask"].sparsity() / 100
                log("attn", T=T, kind=f"gist flex {name} W={window}", fwd_bwd_ms=round(ms, 1),
                    peak_gb=peak_gb(dev), computed_blocks=f"{density:.3f}")
                meta = None  # free before the next configuration
        q = k = v = None


# --------------------------------------------------------------------------- e2e
def e2e(dev, small, model_name, max_len):
    from transformers import AutoConfig, AutoTokenizer, Qwen2Config

    from gist_qwen2 import GistQwen2ForCausalLM
    from gist_utils import add_gist_tokens

    if small:
        tok = None
        cfg = Qwen2Config(vocab_size=1100, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2)

        def load(attn):
            torch.manual_seed(0)
            c = copy.deepcopy(cfg)
            c._attn_implementation = attn
            return GistQwen2ForCausalLM(c).to(dev, torch.float32)
        gist_ids_all, vocab, pad = [1000 + i for i in range(6)], 999, 0
    else:
        assert AutoConfig.from_pretrained(model_name).model_type == "qwen2", model_name
        tok = AutoTokenizer.from_pretrained(model_name)

        def load(attn):
            m = GistQwen2ForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16,
                                                     attn_implementation=attn).to(dev)
            return m
        vocab, pad = 151000, tok.pad_token_id

    for name, (levels, groups) in TREES.items():
        models = {}
        for attn in ("sdpa", gist_flex_attention.GIST_FLEX):
            m = load(attn)
            if tok is not None:
                t = AutoTokenizer.from_pretrained(model_name)
                gist_ids_all = add_gist_tokens(t, m, 6)
            models[attn] = m
        gist_ids = gist_ids_all[:levels]
        Global_data.gist_token_id = gist_ids
        Global_data.chunk_size = [CHUNK] + groups
        Global_data.sink_size = SINK
        Global_data.num_previous_chunks = PREV
        Global_data.pad_token_id = pad
        Global_data.raw_window = 256 if small else 1024

        # Equivalence at a length the dense path can still hold.
        T = 512 if small else 4096
        rng = random.Random(levels)
        seq = gisted_ids(T, levels, groups, rng, vocab=vocab, gist=gist_ids_all)
        ids = torch.tensor([seq], device=dev)
        labels = torch.tensor([stream_labels(seq, gist_ids)], device=dev)
        am = torch.ones_like(ids)
        losses, grads = {}, {}
        for attn, m in models.items():
            m.train()
            Global_data.train_attn = "dense" if attn == "sdpa" else "flex"
            kw = {}
            if attn == "sdpa":
                kw["attention_mask_gist"] = add_raw_window(
                    make_hierarchical_gist_mask_greedy(ids, gist_ids, CHUNK, attention_sink_size=SINK,
                                                       num_previous_chunks=PREV, pad_token=pad),
                    ids, gist_ids, Global_data.raw_window, pad)
            else:
                kw["attention_mask_gist"] = None
            out = m(input_ids=ids, attention_mask=am, labels=labels, **kw)
            out.loss.backward()
            losses[attn] = out.loss.item()
            grads[attn] = torch.cat([p.grad.float().flatten() for p in m.parameters() if p.grad is not None])
            m.zero_grad(set_to_none=True)
        cos = F.cosine_similarity(grads["sdpa"], grads[gist_flex_attention.GIST_FLEX], dim=0).item()
        l_d, l_f = losses["sdpa"], losses[gist_flex_attention.GIST_FLEX]
        log("e2e", tree=name, T=T, dense_loss=round(l_d, 4), flex_loss=round(l_f, 4),
            grad_cosine=round(cos, 5), ok=abs(l_d - l_f) < 0.02 * abs(l_d) + 1e-3 and cos > 0.99)
        del models["sdpa"], grads
        peak_reset(dev)

        # Flex only: a few optimizer steps at long length, gradient checkpointing on.
        m = models[gist_flex_attention.GIST_FLEX]
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        m.config.use_cache = False
        opt = torch.optim.AdamW(m.parameters(), lr=1e-5)
        Global_data.train_attn = "flex"
        Global_data.loss_chunk_size = 256 if small else 4096  # full fp32 logits OOM at 16K
        for T in ([1024] if small else [n for n in (16384, 32768) if n <= max_len]):
            seq = gisted_ids(T, levels, groups, rng, vocab=vocab, gist=gist_ids_all)
            ids = torch.tensor([seq], device=dev)
            labels = torch.tensor([stream_labels(seq, gist_ids)], device=dev)
            peak_reset(dev)
            try:
                times, losses_t = [], []
                for _ in range(3):
                    def step():
                        out = m(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
                                attention_mask_gist=None)
                        out.loss.backward()
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        return out.loss.item()
                    loss, ms = timed(step, dev)
                    times.append(ms)
                    losses_t.append(round(loss, 3))
                log("e2e", tree=name, T=T, step_ms=[round(t) for t in times], losses=losses_t,
                    peak_gb=peak_gb(dev), ok=all(math.isfinite(x) for x in losses_t))
            except torch.cuda.OutOfMemoryError:
                log("e2e", tree=name, T=T, result="OOM")
                break
        m = opt = None
        models.clear()
        Global_data.loss_chunk_size = 0
        peak_reset(dev)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", nargs="+", default=["numerics", "recompile", "mask", "attn", "e2e"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--small", action="store_true", help="tiny sizes, for a CPU dry run")
    ap.add_argument("--max-len", type=int, default=262144, help="skip lengths above this")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default="gpu_validate_flex.json")
    args = ap.parse_args()

    dev = torch.device(args.device)
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(dev)
        print(f"{p.name}, {p.total_memory / 2**30:.0f} GB, sm_{p.major}{p.minor}, torch {torch.__version__}")
    RESULTS["env"] = [{"device": str(dev), "torch": torch.__version__,
                       "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None}]

    sections = {
        "numerics": lambda: numerics(dev, args.small),
        "recompile": lambda: recompile(dev, args.small),
        "mask": lambda: mask_build(dev, args.small, args.max_len),
        "attn": lambda: attn_bench(dev, args.small, args.max_len),
        "e2e": lambda: e2e(dev, args.small, args.model, args.max_len),
    }
    for s in args.sections:
        print(f"\n=== {s}", flush=True)
        try:
            sections[s]()
        except Exception as e:  # keep going; one failing section should not hide the rest
            log(s, error=f"{type(e).__name__}: {str(e)[:300]}")
    with open(args.out, "w") as f:
        json.dump(RESULTS, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
