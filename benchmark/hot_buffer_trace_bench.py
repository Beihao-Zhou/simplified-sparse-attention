"""Trace-driven benchmark of the persistent GPU LRU hot buffer.

benchmark/sparse_decode_operator.py replays one identical selection forever, so
it reports a 100% warm hit rate by construction and says nothing about locality.
This replays *consecutive* recorded selections instead -- the real per-(layer,
KV-group) index sets from analysis/record_selection.py, step after step -- so the
hit rate, the miss count and the bytes actually moved are the model's own.

Batched runs use independent series (a different layer per batch element, and the
recorded per-group sets within it) rather than one request duplicated B times,
which would make every group hit or miss in lockstep.

Per-phase times come from the CUDA kernel timeline (torch.profiler, each kernel's
own duration), not from subtracting noisy totals.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from offload_trace_bench import D_HEAD, kernel_breakdown, load_extension

HOT_PHASES = {
    "build_idx": "build_compact_idx_kernel",
    "resolve": "resolve_hot_slots_lru_kernel",
    "gather_hot": "gather_hot_misses_from_host_kernel",
    "gather_full": "gather_compact_kv_from_host_kernel",
}


def load_series(trace, layers, groups, steps, start=0):
    """Consecutive recorded index sets for each (layer, group) series.

    Returns ``series[(layer, group)] = [idx_step0, idx_step1, ...]`` plus the
    per-step k_len.  Reads every column once: NpzFile decompresses the whole
    array on each __getitem__, so indexing it inside the loop is what turned an
    earlier version of this analysis into a 30-minute job.
    """
    d = np.load(trace, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    lens = d["lens"].astype(np.int64)
    layer_col = d["layer"].astype(np.int64)
    group_col = d["g"].astype(np.int64)
    step_col = d["step"].astype(np.int64)
    k_len_col = d["k_len"].astype(np.int64)
    keys = d["keys"]
    off = np.concatenate([[0], np.cumsum(lens)])

    want_steps = set(range(start, start + steps))
    series = {(layer, g): {} for layer in layers for g in groups}
    # Capacity is set from each series' k_max over the WHOLE trace, which is the
    # definition analysis/cache_sim.py uses for B = 1x/2x/4x k_max. Using the
    # replay window's max instead would
    # quietly shrink the buffer and understate the hit rate.
    series_kmax = {key: 0 for key in series}
    k_len_by_step = {}
    for i in range(len(lens)):
        key = (int(layer_col[i]), int(group_col[i]))
        if key not in series:
            continue
        if lens[i] > series_kmax[key]:
            series_kmax[key] = int(lens[i])
        s = int(step_col[i])
        if s not in want_steps:
            continue
        series[key][s] = keys[off[i]:off[i + 1]].astype(np.int64)
        k_len_by_step[s] = max(k_len_by_step.get(s, 0), int(k_len_col[i]))

    ordered = sorted(want_steps & set(k_len_by_step))
    for key, by_step in series.items():
        missing = [s for s in ordered if s not in by_step]
        if missing:
            raise SystemExit(f"trace {trace} lacks steps {missing[:5]} for {key}")
    return series, ordered, [k_len_by_step[s] for s in ordered], meta, series_kmax


def build_replay(series, layers, groups, ordered, k_len, device="cuda"):
    """One sel0 tensor per replayed step; batch element b uses layers[b]."""
    B, num_kv = len(layers), len(groups)
    sels = []
    for si, _ in enumerate(ordered):
        sel0 = torch.zeros(B, num_kv, k_len, dtype=torch.uint8, device=device)
        for b, layer in enumerate(layers):
            for gi, g in enumerate(groups):
                idx = series[(layer, g)][ordered[si]]
                sel0[b, gi, torch.from_numpy(idx).to(device)] = 1
        sels.append(sel0.contiguous())
    counts = np.array([[[len(series[(layer, g)][s]) for g in groups]
                        for layer in layers] for s in ordered])   # [steps, B, num_kv]
    return sels, counts


def static_metadata(B, num_kv, k_len, heads, device="cuda"):
    chunk = torch.arange(k_len, device=device, dtype=torch.int32).expand(B, -1).contiguous()
    zeros = torch.zeros(B, k_len, device=device, dtype=torch.uint8)
    q = torch.randn(B, heads, 1, D_HEAD, device=device, dtype=torch.bfloat16)
    sel1 = torch.zeros(B, num_kv, 1, device=device, dtype=torch.uint8)
    tail = (sel1, chunk, torch.zeros_like(chunk), zeros, zeros, zeros,
            torch.ones_like(zeros), zeros, torch.zeros(B, k_len, device=device))
    return q, tail


def hot_state(B, num_kv, k_len, hot_size, device="cuda"):
    i32 = dict(dtype=torch.int32, device=device)
    hot_k = torch.empty(B, num_kv, hot_size, D_HEAD, dtype=torch.bfloat16, device=device)
    hot_v = torch.empty_like(hot_k)
    state = (
        hot_k, hot_v,
        torch.full((B, num_kv, k_len), -1, **i32),
        torch.full((B, num_kv, hot_size), -1, **i32),
        torch.arange(hot_size, **i32).view(1, 1, -1).expand(B, num_kv, -1).contiguous(),
        torch.empty((B, num_kv, hot_size), **i32),
        torch.empty((B, num_kv, hot_size), dtype=torch.uint8, device=device),
        torch.zeros(2, dtype=torch.int64, device=device),
    )
    persistent = sum(t.numel() * t.element_size() for t in state)
    return state, persistent


def replay_timed(step_fn, n_steps, repeats):
    """Per-step latency over the whole replay; every step is a different set."""
    ms = []
    for _ in range(repeats):
        evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
               for _ in range(n_steps)]
        for i, (a, b) in enumerate(evs):
            a.record()
            step_fn(i)
            b.record()
        torch.cuda.synchronize()
        ms += [a.elapsed_time(b) for a, b in evs]
    ms.sort()
    pick = lambda p: ms[min(len(ms) - 1, int(p / 100 * len(ms)))]   # noqa: E731
    return dict(mean_ms=statistics.fmean(ms), p50_ms=pick(50),
                p95_ms=pick(95), p99_ms=pick(99), max_ms=ms[-1])


def classify(bd):
    out = {k: 0.0 for k in HOT_PHASES}
    attn = 0.0
    for name, ms in bd.items():
        hit = next((t for t, needle in HOT_PHASES.items() if needle in name), None)
        if hit:
            out[hit] += ms
        elif "kernel" in name or "combine" in name:
            attn += ms
    out["attn_combine"] = attn
    out["total_kernels"] = sum(bd.values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--multipliers", type=float, nargs="+", default=[1, 2, 4],
                    help="hot capacity as a multiple of each series k_max over the whole trace")
    ap.add_argument("--replay-steps", type=int, default=128)
    ap.add_argument("--start-step", type=int, default=0)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--heads-per-group", type=int, default=7)
    ap.add_argument("--splits", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="analysis/results/hot_buffer_trace.csv")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    ext = load_extension()
    scaling = D_HEAD**-0.5
    groups = list(range(args.kv_heads))
    rows = []

    for B in args.batches:
        # Independent series: a different recorded layer per batch element, and
        # that layer's own per-KV-group sets. Never one request duplicated.
        all_layers = list(range(1, 28))
        layers = [all_layers[int(round(i * (len(all_layers) - 1) / max(1, B - 1)))]
                  for i in range(B)] if B > 1 else [all_layers[len(all_layers) // 2]]
        series, ordered, k_lens, meta, series_kmax = load_series(
            args.trace, layers, groups, args.replay_steps, args.start_step)
        k_len = max(k_lens)
        heads = args.kv_heads * args.heads_per_group
        sels, counts = build_replay(series, layers, groups, ordered, k_len)
        n_steps = len(sels)
        k_max = max(series_kmax.values())      # capacity basis: whole-trace
        window_max = int(counts.max())         # largest set actually replayed
        sel_rows_total = int(counts.sum())        # summed over steps, batch, groups

        q, tail = static_metadata(B, args.kv_heads, k_len, heads)
        host_k = torch.randn(B, args.kv_heads, k_len, D_HEAD,
                             dtype=torch.bfloat16).pin_memory()
        host_v = torch.randn_like(host_k).pin_memory()
        cap = k_len

        def resident_step(i, gk=None, gv=None):
            return ext.sparse_decode_attn_compact(
                q, gk, gv, sels[i], *tail, scaling, args.splits, cap)

        gpu_k, gpu_v = host_k.cuda(), host_v.cuda()
        ref = [resident_step(i, gpu_k, gpu_v) for i in range(n_steps)]
        r_res = replay_timed(lambda i: resident_step(i, gpu_k, gpu_v), n_steps, args.repeats)
        bd_res = classify(kernel_breakdown(
            lambda: [resident_step(i, gpu_k, gpu_v) for i in range(n_steps)], iters=1))
        bd_res = {k: v / n_steps for k, v in bd_res.items()}
        full_kv_MiB = (gpu_k.nbytes + gpu_v.nbytes) / 2**20
        gpu_k = gpu_v = None
        torch.cuda.empty_cache()

        base = dict(trace=Path(args.trace).stem, context=meta["context_tokens"],
                    batch=B, layers="|".join(map(str, layers)), k_len=k_len,
                    steps=n_steps, k_max=k_max, window_max=window_max,
                    sel_rows_per_step=sel_rows_total / n_steps,
                    full_kv_MiB=full_kv_MiB)
        rows.append(dict(base, variant="resident", hot_size=0, hit_rate=float("nan"),
                         hit_rate_cold=float("nan"), hit_rate_warm=float("nan"), miss_rows_per_step=float("nan"),
                         xfer_MiB_per_step=0.0, persistent_MiB=full_kv_MiB,
                         peak_temp_MiB=float("nan"), max_abs_err=0.0, **r_res, **bd_res))
        print(f"[{Path(args.trace).stem} B={B}] steps={n_steps} k_len={k_len} "
              f"k_max={k_max} (window max {window_max}) "
              f"resident p50={r_res['p50_ms']:.3f} ms")

        def full_step(i):
            return ext.sparse_decode_attn_offload(
                q, host_k, host_v, sels[i], *tail, scaling, args.splits, cap, 0)

        outs = [full_step(i) for i in range(n_steps)]
        err = max((o.float() - r.float()).abs().max().item() for o, r in zip(outs, ref))
        r_full = replay_timed(full_step, n_steps, args.repeats)
        bd_full = classify(kernel_breakdown(
            lambda: [full_step(i) for i in range(n_steps)], iters=1))
        bd_full = {k: v / n_steps for k, v in bd_full.items()}
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base_alloc = torch.cuda.memory_allocated()
        full_step(0)
        torch.cuda.synchronize()
        peak_full = (torch.cuda.max_memory_allocated() - base_alloc) / 2**20
        # Every group gathers its own rows, so the row count is already summed
        # over groups; K and V, bf16.
        full_xfer = sel_rows_total / n_steps * 2 * D_HEAD * 2 / 2**20
        rows.append(dict(base, variant="offload_full_gather", hot_size=0,
                         hit_rate=0.0, hit_rate_cold=0.0, hit_rate_warm=0.0,
                         miss_rows_per_step=sel_rows_total / n_steps,
                         xfer_MiB_per_step=full_xfer, persistent_MiB=0.0,
                         peak_temp_MiB=peak_full, max_abs_err=err, **r_full, **bd_full))
        print(f"    full-gather p50={r_full['p50_ms']:.3f} ms "
              f"gather={bd_full['gather_full']:.4f} ms xfer={full_xfer:.2f} MiB/step")

        for mult in args.multipliers:
            hot_size = int(round(mult * k_max))
            if hot_size < k_max:
                continue
            state, persistent = hot_state(B, args.kv_heads, k_len, hot_size)

            def hot_step(i):
                return ext.sparse_decode_attn_offload_cached(
                    q, host_k, host_v, *state, sels[i], *tail, scaling, args.splits, cap)

            # First pass from an empty buffer: this is exactly what
            # analysis/cache_sim.py simulates, so the two are comparable.
            outs, half = [], n_steps // 2
            sel_h = miss_h = 0
            for i in range(n_steps):
                outs.append(hot_step(i))
                if i + 1 == half:
                    sel_h, miss_h = state[-1].cpu().tolist()
            sel_c, miss_c = state[-1].cpu().tolist()
            err = max((o.float() - r.float()).abs().max().item() for o, r in zip(outs, ref))
            hit_cold = 1.0 - miss_c / max(1, sel_c)
            hit_warm = 1.0 - (miss_c - miss_h) / max(1, sel_c - sel_h)

            r_hot = replay_timed(hot_step, n_steps, args.repeats)
            bd_hot = classify(kernel_breakdown(
                lambda: [hot_step(i) for i in range(n_steps)], iters=1))
            bd_hot = {k: v / n_steps for k, v in bd_hot.items()}
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base_alloc = torch.cuda.memory_allocated()
            hot_step(0)
            torch.cuda.synchronize()
            peak_hot = (torch.cuda.max_memory_allocated() - base_alloc) / 2**20

            # Steady-state misses, measured over the timed replay itself.
            state[-1].zero_()
            for i in range(n_steps):
                hot_step(i)
            sel_s, miss_s = state[-1].cpu().tolist()
            xfer = miss_s / n_steps * 2 * D_HEAD * 2 / 2**20
            rows.append(dict(base, variant=f"offload_hot_{mult:g}x", hot_size=hot_size,
                             hit_rate=1.0 - miss_s / max(1, sel_s),
                             hit_rate_cold=hit_cold, hit_rate_warm=hit_warm,
                             miss_rows_per_step=miss_s / n_steps,
                             xfer_MiB_per_step=xfer,
                             persistent_MiB=persistent / 2**20,
                             peak_temp_MiB=peak_hot, max_abs_err=err, **r_hot, **bd_hot))
            print(f"    hot {mult:g}x (hot_size={hot_size}) p50={r_hot['p50_ms']:.3f} ms "
                  f"hit={1.0 - miss_s / max(1, sel_s):.1%} cold={hit_cold:.1%} warm={hit_warm:.1%} xfer={xfer:.2f} MiB/step "
                  f"resolve={bd_hot['resolve']:.4f} ms "
                  f"({100 * bd_hot['resolve'] / max(bd_hot['total_kernels'], 1e-9):.1f}% kernels) "
                  f"speedup_vs_full={r_full['p50_ms'] / r_hot['p50_ms']:.2f}x "
                  f"persistent={persistent / 2**20:.0f} MiB")
            state = None
            torch.cuda.empty_cache()
        host_k = host_v = sels = None
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
