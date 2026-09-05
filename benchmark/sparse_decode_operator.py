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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--cap", type=int, default=256)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--heads-per-group", type=int, default=7)
    parser.add_argument("--splits", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if not 0 < args.cap <= args.length:
        parser.error("--cap must be in [1, --length]")

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

    offload = partial(
        extension.sparse_decode_attn_offload,
        q,
        host_k,
        host_v,
        *metadata,
        scaling,
        args.splits,
        args.cap,
    )

    offload_out = offload()
    torch.cuda.synchronize()
    torch.testing.assert_close(offload_out, resident_out, rtol=2e-2, atol=2e-2)
    offload_ms = _time_ms(offload, args.warmup, args.iterations)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    measured_out = offload()
    torch.cuda.synchronize()
    offload_peak_bytes = torch.cuda.max_memory_allocated() - baseline

    print(f"shape: B={batch} Hq={heads} Hkv={kv_heads} L={args.length} cap={args.cap}")
    print(f"resident: {resident_ms:.3f} ms, full KV={full_kv_bytes / 2**20:.1f} MiB")
    print(f"offload:  {offload_ms:.3f} ms, peak temporary={offload_peak_bytes / 2**20:.1f} MiB")
    print(f"offload/resident latency: {offload_ms / resident_ms:.2f}x")
    print(f"output checksum: {measured_out.float().sum().item():.6f}")


if __name__ == "__main__":
    main()
