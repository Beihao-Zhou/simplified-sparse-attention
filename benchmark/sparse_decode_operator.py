"""Standalone GPU gate for batched resident and pinned-host sparse decode."""

import argparse
from functools import partial
from pathlib import Path

import torch


def _load_extension():
    from torch.utils.cpp_extension import load

    return load(
        name="gsa_sparse_decode_operator_bench",
        sources=[str(Path(__file__).parents[1] / "kernels" / "sparse_decode_attn.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


def _time_ms(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _peak_temp_bytes(fn):
    """Peak GPU memory the call allocates above the already-resident baseline."""
    fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--cap", type=int, default=256)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--heads-per-group", type=int, default=7)
    parser.add_argument("--splits", type=int, default=128)
    parser.add_argument("--hot-size", type=int, default=0,
                        help="enable persistent LRU hot buffer (must be >= --cap)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if not 0 < args.cap <= args.length:
        parser.error("--cap must be in [1, --length]")
    if args.hot_size and args.hot_size < args.cap:
        parser.error("--hot-size must be zero or >= --cap")

    torch.manual_seed(0)
    device, dim = "cuda", 128
    batch, kv_heads = args.batch, args.kv_heads
    heads = kv_heads * args.heads_per_group
    q = torch.randn(batch, heads, 1, dim, device=device, dtype=torch.bfloat16)
    host_k = torch.randn(batch, kv_heads, args.length, dim, dtype=torch.bfloat16).pin_memory()
    host_v = torch.randn_like(host_k).pin_memory()

    chunk = torch.arange(args.length, device=device, dtype=torch.int32).expand(batch, -1).contiguous()
    chunk_l1 = torch.zeros_like(chunk)
    selected = torch.zeros(batch, kv_heads, args.length, device=device, dtype=torch.uint8)
    selected[:, :, : args.cap] = 1
    selected_l1 = torch.zeros(batch, kv_heads, 1, device=device, dtype=torch.uint8)
    zeros = torch.zeros(batch, args.length, device=device, dtype=torch.uint8)
    compressed = torch.ones_like(zeros)
    bias = torch.zeros(batch, args.length, device=device)
    metadata = (
        selected,
        selected_l1,
        chunk,
        chunk_l1,
        zeros,
        zeros,
        zeros,
        compressed,
        zeros,
        bias,
    )
    extension = _load_extension()
    scaling = dim**-0.5

    gpu_k = host_k.to(device)
    gpu_v = host_v.to(device)
    resident = partial(
        extension.sparse_decode_attn_compact,
        q,
        gpu_k,
        gpu_v,
        *metadata,
        scaling,
        args.splits,
        args.length,
    )

    resident_out = resident()
    resident_ms = _time_ms(resident, args.warmup, args.iterations)
    full_kv_bytes = gpu_k.nbytes + gpu_v.nbytes
    del resident, gpu_k, gpu_v
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    def offload_variant(async_gather):
        return partial(
            extension.sparse_decode_attn_offload,
            q,
            host_k,
            host_v,
            *metadata,
            scaling,
            args.splits,
            args.cap,
            async_gather,
        )

    # async_gather=0 keeps the gather on the compute stream (the baseline offload
    # path); async_gather=1 runs it on a dedicated stream ordered by CUDA events.
    variants = [("sync-stream", offload_variant(0)), ("separate-stream", offload_variant(1))]

    results = []
    for name, fn in variants:
        out = fn()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, resident_out, rtol=2e-2, atol=2e-2)
        results.append((name, _time_ms(fn, args.warmup, args.iterations), _peak_temp_bytes(fn), out))

    hot_result = None
    if args.hot_size:
        i32 = dict(dtype=torch.int32, device=device)
        hot_k = torch.empty(batch, kv_heads, args.hot_size, dim,
                            dtype=torch.bfloat16, device=device)
        hot_v = torch.empty_like(hot_k)
        token_to_slot = torch.full((batch, kv_heads, args.length), -1, **i32)
        slot_token_ids = torch.full((batch, kv_heads, args.hot_size), -1, **i32)
        lru_order = (torch.arange(args.hot_size, **i32).view(1, 1, -1)
                     .expand(batch, kv_heads, -1).contiguous())
        hot_state = (
            hot_k, hot_v, token_to_slot, slot_token_ids, lru_order,
            torch.empty_like(lru_order),
            torch.empty(batch, kv_heads, args.hot_size, dtype=torch.uint8, device=device),
            torch.zeros(2, dtype=torch.int64, device=device),
        )
        hot_fn = partial(
            extension.sparse_decode_attn_offload_cached,
            q, host_k, host_v, *hot_state, *metadata,
            scaling, args.splits, args.length,
        )
        cold_ms = _time_ms(hot_fn, 0, 1)
        hot_state[-1].zero_()
        hot_ms = _time_ms(hot_fn, args.warmup, args.iterations)
        hot_peak = _peak_temp_bytes(hot_fn)
        selected_count, miss_count = hot_state[-1].cpu().tolist()
        hot_out = hot_fn()
        torch.cuda.synchronize()
        torch.testing.assert_close(hot_out, resident_out, rtol=2e-2, atol=2e-2)
        hot_result = (cold_ms, hot_ms, hot_peak, selected_count, miss_count)

    # NOT bit-exact, by design: build_compact_idx_kernel orders idx_list by a racing
    # atomicAdd, so the online-softmax accumulation order varies run to run — two
    # *resident* calls differ in the last ulps too. Report the spread instead.
    sync_out, async_out = results[0][3], results[1][3]
    stream_diff = (async_out.float() - sync_out.float()).abs().max().item()
    resident_diff = (async_out.float() - resident_out.float()).abs().max().item()

    print(f"shape: B={batch} Hq={heads} Hkv={kv_heads} L={args.length} cap={args.cap}")
    print(f"resident KV:              {resident_ms:.3f} ms, full KV={full_kv_bytes / 2**20:.1f} MiB")
    for name, ms, peak, _ in results:
        print(
            f"offload ({name:15s}): {ms:.3f} ms, peak temporary={peak / 2**20:.1f} MiB"
            f", {ms / resident_ms:.2f}x resident"
        )
    if hot_result is not None:
        cold_ms, hot_ms, hot_peak, selected_count, miss_count = hot_result
        hit_rate = 1.0 - miss_count / max(1, selected_count)
        print(
            f"offload (hot LRU):       cold={cold_ms:.3f} ms, warm={hot_ms:.3f} ms, "
            f"hit={hit_rate:.3%}, peak temporary={hot_peak / 2**20:.1f} MiB"
        )
    sync_ms, async_ms = results[0][1], results[1][1]
    print(f"separate-stream / sync-stream offload latency: {async_ms / sync_ms:.3f}x")
    print(f"max|separate-stream - sync-stream| = {stream_diff:.3e}")
    print(f"max|separate-stream - resident|      = {resident_diff:.3e}")
    print(f"output checksum: {async_out.float().sum().item():.6f}")


if __name__ == "__main__":
    main()
