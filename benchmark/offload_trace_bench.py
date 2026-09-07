"""Trace-driven benchmark of the sparse-decode offload operator.

Rather than a fixed synthetic prefix, this replays *exact* selected index sets
recorded from the real model (analysis/record_selection.py), so both the number
of gathered keys and their scatter pattern in host memory match production. The
replay works by giving every logical position its own level-0 chunk and setting
``sel0`` on exactly the recorded positions, which makes the kernel's keep rule
reduce to ``keep = sel0``.

Per-phase times are read from the CUDA kernel timeline via torch.profiler --
each kernel's own duration -- not inferred by subtracting two noisy totals. The
"no-IO oracle" row is the sum of the same measured kernels with the gather
excluded; it bounds what any transfer optimisation can reach.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
import torch

D_HEAD = 128
PHASES = {
    "build_idx": "build_compact_idx_kernel",
    "gather": "gather_compact_kv_from_host_kernel",
}


def load_extension():
    from torch.utils.cpp_extension import load
    return load(
        name="gsa_sparse_decode_operator_bench",
        sources=[str(Path(__file__).parents[1] / "kernels" / "sparse_decode_attn.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


def pick_records(trace, batch, pct):
    """B recorded (index set, k_len) pairs whose sizes sit at percentile `pct`."""
    d = np.load(trace, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    lens = d["lens"].astype(np.int64)
    off = np.concatenate([[0], np.cumsum(lens)])
    target = np.percentile(lens, pct)
    order = np.argsort(np.abs(lens - target))
    keys, layer_col, step_col = d["keys"], d["layer"], d["step"]
    picks, seen = [], set()
    for i in order:
        sig = (int(layer_col[i]), int(step_col[i]))
        if sig in seen:
            continue
        seen.add(sig)
        picks.append((keys[off[i]:off[i + 1]].astype(np.int64), int(d["k_len"][i])))
        if len(picks) == batch:
            break
    while len(picks) < batch:                    # tiny traces: replay with repeats
        picks.append(picks[len(picks) % max(1, len(picks))])
    return picks, meta, lens


def build_inputs(picks, kv_heads, heads_per_group, device="cuda"):
    """Metadata whose keep rule reduces to exactly the recorded index sets."""
    B = len(picks)
    k_len = max(p[1] for p in picks)
    heads = kv_heads * heads_per_group
    q = torch.randn(B, heads, 1, D_HEAD, device=device, dtype=torch.bfloat16)
    host_k = torch.randn(B, kv_heads, k_len, D_HEAD, dtype=torch.bfloat16).pin_memory()
    host_v = torch.randn_like(host_k).pin_memory()
    chunk = torch.arange(k_len, device=device, dtype=torch.int32).expand(B, -1).contiguous()
    chunk_l1 = torch.zeros_like(chunk)
    sel0 = torch.zeros(B, kv_heads, k_len, device=device, dtype=torch.uint8)
    for b, (idx, _) in enumerate(picks):
        t = torch.from_numpy(idx).to(device)
        sel0[b, :, t] = 1
    sel1 = torch.zeros(B, kv_heads, 1, device=device, dtype=torch.uint8)
    zeros = torch.zeros(B, k_len, device=device, dtype=torch.uint8)
    meta = (sel0.contiguous(), sel1, chunk, chunk_l1, zeros, zeros, zeros,
            torch.ones_like(zeros), zeros, torch.zeros(B, k_len, device=device))
    counts = [len(p[0]) for p in picks]
    return q, host_k, host_v, meta, k_len, counts


def time_iters(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
           for _ in range(iters)]
    for a, b in evs:
        a.record()
        fn()
        b.record()
    torch.cuda.synchronize()
    ms = [a.elapsed_time(b) for a, b in evs]
    return dict(mean_ms=statistics.fmean(ms), median_ms=statistics.median(ms),
                min_ms=min(ms), max_ms=max(ms),
                std_ms=statistics.pstdev(ms) if len(ms) > 1 else 0.0)


def peak_temp_bytes(fn):
    fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base


def kernel_breakdown(fn, iters=20):
    """Per-kernel CUDA time, measured on the kernel timeline."""
    from torch.profiler import ProfilerActivity, profile
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    out = {}
    for e in prof.key_averages():
        if e.device_type.name != "CUDA" and e.self_device_time_total == 0:
            continue
        us = e.self_device_time_total
        if us <= 0:
            continue
        out[e.key] = out.get(e.key, 0.0) + us / iters / 1000.0
    return out


def classify(bd):
    phases = {k: 0.0 for k in PHASES}
    attn = 0.0
    for name, ms in bd.items():
        hit = None
        for tag, needle in PHASES.items():
            if needle in name:
                hit = tag
        if hit:
            phases[hit] += ms
        elif "kernel" in name or "decode" in name or "combine" in name:
            attn += ms
    phases["attn_combine"] = attn
    phases["total_kernels"] = sum(bd.values())
    return phases


def try_cuda_graph(fn):
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        g.replay()
        torch.cuda.synchronize()
        return "captured"
    except Exception as e:  # noqa: BLE001 - the failure mode is the result
        return f"failed: {type(e).__name__}: {str(e)[:140]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--percentiles", type=float, nargs="+", default=[50, 90, 99])
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--heads-per-group", type=int, default=7)
    ap.add_argument("--splits", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--cuda-graph", action="store_true")
    ap.add_argument("--out", default="analysis/results/operator_bench.csv")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    ext = load_extension()
    scaling = D_HEAD**-0.5
    rows = []
    for pct in args.percentiles:
        for B in args.batches:
            picks, tmeta, all_lens = pick_records(args.trace, B, pct)
            q, host_k, host_v, meta, k_len, counts = build_inputs(
                picks, args.kv_heads, args.heads_per_group)
            cap = k_len
            gpu_k, gpu_v = host_k.cuda(), host_v.cuda()

            def resident():
                return ext.sparse_decode_attn_compact(
                    q, gpu_k, gpu_v, *meta, scaling, args.splits, cap)

            ref = resident()
            r_res = time_iters(resident, args.warmup, args.iterations)
            bd_res = classify(kernel_breakdown(resident))
            full_kv = gpu_k.nbytes + gpu_v.nbytes
            peak_res = peak_temp_bytes(resident)
            graph_res = try_cuda_graph(resident) if args.cuda_graph else ""
            gpu_k = gpu_v = None
            torch.cuda.empty_cache()

            variants = {"offload_sync": 0, "offload_async": 1}
            outs = {}
            per_variant = {}
            for name, flag in variants.items():
                def fn(flag=flag):
                    return ext.sparse_decode_attn_offload(
                        q, host_k, host_v, *meta, scaling, args.splits, cap, flag)
                outs[name] = fn()
                per_variant[name] = (time_iters(fn, args.warmup, args.iterations),
                                     classify(kernel_breakdown(fn)),
                                     peak_temp_bytes(fn),
                                     try_cuda_graph(fn) if args.cuda_graph else "")

            sel_total = int(sum(counts))
            # The gather runs once per (batch, kv-head): sel0 carries a selection
            # per KV group, so every group fetches its own rows. Forgetting the
            # num_kv factor understates the achieved bandwidth by 4x here.
            xfer_bytes = sel_total * args.kv_heads * 2 * D_HEAD * 2   # K and V, bf16
            base = dict(trace=Path(args.trace).stem, context=tmeta["context_tokens"],
                        percentile=pct, batch=B, k_len=k_len,
                        selected_total=sel_total, selected_per_seq=sel_total / B,
                        xfer_MiB=xfer_bytes / 2**20,
                        kv_heads=args.kv_heads,
                        full_kv_MiB=full_kv / 2**20)
            rows.append(dict(base, variant="resident", **r_res, **bd_res,
                             peak_temp_MiB=peak_res / 2**20, max_abs_err=0.0,
                             gather_GiBs=float("nan"), cuda_graph=graph_res))
            for name, (t, bd, peak, gr) in per_variant.items():
                err = (outs[name].float() - ref.float()).abs().max().item()
                gbs = (xfer_bytes / 2**30) / (bd["gather"] / 1000) if bd["gather"] else float("nan")
                rows.append(dict(base, variant=name, **t, **bd,
                                 peak_temp_MiB=peak / 2**20, max_abs_err=err,
                                 gather_GiBs=gbs, cuda_graph=gr))
            # no-IO oracle: the same measured kernels with the gather removed
            bd_sync = per_variant["offload_sync"][1]
            rows.append(dict(base, variant="offload_no_io_oracle",
                             mean_ms=bd_sync["total_kernels"] - bd_sync["gather"],
                             median_ms=float("nan"), min_ms=float("nan"),
                             max_ms=float("nan"), std_ms=float("nan"),
                             build_idx=bd_sync["build_idx"], gather=0.0,
                             attn_combine=bd_sync["attn_combine"],
                             total_kernels=bd_sync["total_kernels"] - bd_sync["gather"],
                             peak_temp_MiB=float("nan"), max_abs_err=float("nan"),
                             gather_GiBs=float("nan"), cuda_graph=""))
            print(f"[bench] p{pct:g} B={B} k_len={k_len} sel/seq={sel_total / B:.0f} "
                  f"resident {r_res['mean_ms']:.3f} ms | "
                  f"sync {per_variant['offload_sync'][0]['mean_ms']:.3f} ms "
                  f"(gather {per_variant['offload_sync'][1]['gather']:.3f} ms, "
                  f"{rows[-2]['gather_GiBs']:.1f} GiB/s) | "
                  f"async {per_variant['offload_async'][0]['mean_ms']:.3f} ms", flush=True)
            outs = ref = None
            torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[bench] wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
