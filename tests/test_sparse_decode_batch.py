import unittest
from pathlib import Path

import torch


def _load_extension():
    from torch.utils.cpp_extension import load

    return load(
        name="gsa_sparse_decode_batch_test",
        sources=[str(Path(__file__).parents[1] / "kernels" / "sparse_decode_attn.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SparseDecodeBatchTest(unittest.TestCase):
    def test_matches_reference_with_different_sequence_lengths(self):
        torch.manual_seed(0)
        device = "cuda"
        batch, heads, kv_heads, length, dim = 3, 28, 4, 17, 128
        heads_per_group = heads // kv_heads
        groups = 4
        scale = dim**-0.5

        q = torch.randn(batch, heads, 1, dim, device=device, dtype=torch.bfloat16)
        k = torch.randn(batch, kv_heads, length, dim, device=device, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        chunk_l0 = (torch.arange(length, device=device) // 5).clamp_max(groups - 1)
        chunk_l0 = chunk_l0.expand(batch, -1).to(torch.int32).contiguous()
        chunk_l1 = torch.zeros_like(chunk_l0)
        sel0 = torch.zeros(batch, kv_heads, groups, device=device, dtype=torch.uint8)
        sel0[0, :, [0, 2]] = 1
        sel0[1, 0, [0, 1]] = 1
        sel0[1, 1, [1, 2]] = 1
        sel0[2, :, [0, 3]] = 1
        sel1 = torch.zeros(batch, kv_heads, 1, device=device, dtype=torch.uint8)
        starts = torch.tensor([0, 2, 4], device=device)
        prefix_ends = torch.tensor([12, 9, 7], device=device)
        seq_ends = torch.tensor([17, 13, 9], device=device)
        positions = torch.arange(length, device=device)[None, :]
        valid = (positions >= starts[:, None]) & (positions < seq_ends[:, None])
        compressed = (valid & (positions <= prefix_ends[:, None])).to(torch.uint8).contiguous()
        always_attended = (valid & (positions > prefix_ends[:, None])).to(torch.uint8).contiguous()
        zeros = torch.zeros(batch, length, device=device, dtype=torch.uint8)
        bias = torch.randn(batch, length, device=device, dtype=torch.float32) * 0.01

        extension = _load_extension()
        actual = extension.sparse_decode_attn_compact(
            q, k, v, sel0, sel1, chunk_l0, chunk_l1,
            zeros, zeros, zeros, compressed, always_attended, bias,
            scale, 4, length,
        )
        host_k = k.cpu().pin_memory()
        host_v = v.cpu().pin_memory()
        offloaded = extension.sparse_decode_attn_offload(
            q, host_k, host_v,
            sel0, sel1, chunk_l0, chunk_l1,
            zeros, zeros, zeros, compressed, always_attended, bias,
            scale, 4, 12,
        )

        expected = torch.empty_like(actual)
        for b in range(batch):
            for h in range(heads):
                g = h // heads_per_group
                keep = ((compressed[b].bool() & sel0[b, g, chunk_l0[b].long()].bool())
                        | always_attended[b].bool())
                scores = (q[b, h, 0].float() * k[b, g, keep].float()).sum(-1)
                scores = scores * scale + bias[b, keep]
                expected[b, 0, h] = (scores.softmax(0) @ v[b, g, keep].float()).to(torch.bfloat16)

        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(offloaded, expected, rtol=2e-2, atol=2e-2)

    def test_offload_gpu_allocation_scales_with_cap_not_context(self):
        device = "cuda"
        batch, heads, kv_heads, length, cap, dim = 2, 28, 4, 4096, 64, 128
        extension = _load_extension()
        q = torch.randn(batch, heads, 1, dim, device=device, dtype=torch.bfloat16)
        host_k = torch.randn(batch, kv_heads, length, dim, dtype=torch.bfloat16).pin_memory()
        host_v = torch.randn_like(host_k).pin_memory()
        chunk = torch.arange(length, device=device, dtype=torch.int32).expand(batch, -1).contiguous()
        sel = torch.zeros(batch, kv_heads, length, device=device, dtype=torch.uint8)
        sel[:, :, :cap] = 1
        zeros = torch.zeros(batch, length, device=device, dtype=torch.uint8)
        ones = torch.ones_like(zeros)
        bias = torch.zeros(batch, length, device=device)

        def run():
            return extension.sparse_decode_attn_offload(
                q, host_k, host_v, sel, sel[:, :, :1].contiguous(), chunk,
                torch.zeros_like(chunk), zeros, zeros, zeros, ones, zeros, bias,
                dim**-0.5, 4, cap,
            )

        run()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        output = run()
        torch.cuda.synchronize()
        allocated = torch.cuda.max_memory_allocated() - baseline
        full_gpu_kv_bytes = 2 * batch * kv_heads * length * dim * 2
        self.assertLess(allocated, full_gpu_kv_bytes // 2)
        self.assertEqual(output.shape, (batch, 1, heads, dim))


if __name__ == "__main__":
    unittest.main()
