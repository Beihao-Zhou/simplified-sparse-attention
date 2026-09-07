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


def _make_case(batch, kv_heads, heads_per_group, length, seed):
    """Random decode problem whose keep-set is dense enough to stress the gather."""
    torch.manual_seed(seed)
    device, dim = "cuda", 128
    heads = kv_heads * heads_per_group
    q = torch.randn(batch, heads, 1, dim, device=device, dtype=torch.bfloat16)
    gpu_k = torch.randn(batch, kv_heads, length, dim, device=device, dtype=torch.bfloat16)
    gpu_v = torch.randn_like(gpu_k)
    chunk_l0 = (torch.arange(length, device=device) // 8).to(torch.int32)
    chunk_l0 = chunk_l0.expand(batch, -1).contiguous()
    chunk_l1 = torch.zeros_like(chunk_l0)
    num_chunks = int(chunk_l0.max()) + 1
    sel0 = (torch.rand(batch, kv_heads, num_chunks, device=device) < 0.35).to(torch.uint8)
    sel0[:, :, 0] = 1  # guarantee a non-empty keep set per batch/group
    sel0 = sel0.contiguous()
    sel1 = torch.zeros(batch, kv_heads, 1, device=device, dtype=torch.uint8)
    zeros = torch.zeros(batch, length, device=device, dtype=torch.uint8)
    metadata = (
        sel0, sel1, chunk_l0, chunk_l1, zeros, zeros, zeros,
        torch.ones_like(zeros),                       # compressed
        zeros,                                        # always-attended
        (torch.randn(batch, length, device=device) * 0.01).contiguous(),
    )
    return q, gpu_k, gpu_v, gpu_k.cpu().pin_memory(), gpu_v.cpu().pin_memory(), metadata


def _make_hot_state(batch, kv_heads, length, hot_size):
    i32 = dict(dtype=torch.int32, device="cuda")
    hot_k = torch.empty(batch, kv_heads, hot_size, 128, device="cuda", dtype=torch.bfloat16)
    hot_v = torch.empty_like(hot_k)
    token_to_slot = torch.full((batch, kv_heads, length), -1, **i32)
    slot_token_ids = torch.full((batch, kv_heads, hot_size), -1, **i32)
    lru_order = (torch.arange(hot_size, **i32).view(1, 1, -1)
                 .expand(batch, kv_heads, -1).contiguous())
    return (
        hot_k, hot_v, token_to_slot, slot_token_ids, lru_order,
        torch.empty_like(lru_order),
        torch.empty(batch, kv_heads, hot_size, device="cuda", dtype=torch.uint8),
        torch.zeros(2, device="cuda", dtype=torch.int64),
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SparseDecodeBatchTest(unittest.TestCase):
    def test_hot_buffer_reuses_hits_and_evicts_only_stale_tokens(self):
        extension = _load_extension()
        batch, kv_heads, hpg, length, hot_size = 2, 4, 7, 32, 8
        q, gpu_k, gpu_v, host_k, host_v, _ = _make_case(
            batch, kv_heads, hpg, length, seed=19
        )
        chunk = torch.arange(length, device="cuda", dtype=torch.int32)
        chunk = chunk.expand(batch, -1).contiguous()
        zeros = torch.zeros(batch, length, device="cuda", dtype=torch.uint8)
        ones = torch.ones_like(zeros)
        bias = torch.randn(batch, length, device="cuda") * 0.01
        sel1 = torch.zeros(batch, kv_heads, 1, device="cuda", dtype=torch.uint8)
        state = _make_hot_state(batch, kv_heads, length, hot_size)

        def run(selected_tokens):
            sel = torch.zeros(batch, kv_heads, length, device="cuda", dtype=torch.uint8)
            sel[:, :, selected_tokens] = 1
            metadata = (sel, sel1, chunk, torch.zeros_like(chunk),
                        zeros, zeros, zeros, ones, zeros, bias)
            resident = extension.sparse_decode_attn_compact(
                q, gpu_k, gpu_v, *metadata, 128**-0.5, 4, length
            )
            cached = extension.sparse_decode_attn_offload_cached(
                q, host_k, host_v, *state, *metadata, 128**-0.5, 4, length
            )
            torch.testing.assert_close(cached, resident, rtol=2e-2, atol=2e-2)

        run(list(range(8)))
        first_slots = state[2].clone()
        misses_before_hit = int(state[-1][1].item())
        run(list(range(8)))
        torch.testing.assert_close(state[2], first_slots)
        self.assertEqual(int(state[-1][1].item()), misses_before_hit)

        run(list(range(4, 12)))
        for bg_tokens in state[3].reshape(-1, hot_size):
            self.assertEqual(set(bg_tokens.cpu().tolist()), set(range(4, 12)))
        self.assertTrue(torch.all(state[2][..., :4] == -1).item())
        self.assertTrue(torch.all(state[2][..., 4:12] >= 0).item())

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

    def test_full_keep_rule_matches_the_analysis_reconstruction(self):
        """The keep rule analysis/record_selection.py replays must be the kernel's.

        All four terms are exercised with non-trivial gist masks; the PyTorch
        reference attends over exactly the set that reconstruction produces, so a
        drift between the two rules shows up as a numerical mismatch here rather
        than as a silently wrong locality table.
        """
        torch.manual_seed(3)
        device, dim = "cuda", 128
        batch, kv_heads, hpg, length = 2, 4, 7, 96
        heads = kv_heads * hpg
        scale = dim**-0.5
        q = torch.randn(batch, heads, 1, dim, device=device, dtype=torch.bfloat16)
        k = torch.randn(batch, kv_heads, length, dim, device=device, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        pos = torch.arange(length, device=device)[None, :].expand(batch, -1)

        # every 8th position is a level-0 gist, every 32nd a level-1 (meta) gist
        is_l0g = ((pos % 8 == 7) & (pos % 32 != 31)).to(torch.uint8).contiguous()
        is_l1g = (pos % 32 == 31).to(torch.uint8).contiguous()
        is_gist = (is_l0g | is_l1g).contiguous()
        c0 = (pos // 8).to(torch.int32).contiguous()
        c1 = (pos // 32).to(torch.int32).contiguous()
        last_gist = int(length - 20)
        compressed = ((pos <= last_gist) & (pos >= 2)).to(torch.uint8).contiguous()
        am = ((pos > last_gist) | (pos < 3)).to(torch.uint8).contiguous()   # suffix + sinks
        G0, G1 = int(c0.max()) + 1, int(c1.max()) + 1
        sel0 = (torch.rand(batch, kv_heads, G0, device=device) < 0.4).to(torch.uint8).contiguous()
        sel1 = (torch.rand(batch, kv_heads, G1, device=device) < 0.5).to(torch.uint8).contiguous()
        bias = (torch.randn(batch, length, device=device) * 0.01).contiguous()

        extension = _load_extension()
        actual = extension.sparse_decode_attn_compact(
            q, k, v, sel0, sel1, c0, c1, is_gist, is_l0g, is_l1g, compressed, am,
            bias, scale, 8, length)

        expected = torch.empty_like(actual)
        counts = []
        for b in range(batch):
            ig, il0, il1 = is_gist[b].bool(), is_l0g[b].bool(), is_l1g[b].bool()
            cb, ab = compressed[b].bool(), am[b].bool()
            for g in range(kv_heads):
                # verbatim the rule in analysis/record_selection.SelectionRecorder
                s0 = sel0[b, g].bool()[c0[b].long()]
                s1 = sel1[b, g].bool()[c1[b].long()]
                keep = ((s0 & cb & ~ig) | (il0 & s0) | (il1 & s1) | (ab & ~ig))
                counts.append(int(keep.sum()))
                for h in range(g * hpg, (g + 1) * hpg):
                    sc = (q[b, h, 0].float() * k[b, g, keep].float()).sum(-1) * scale
                    sc = sc + bias[b, keep]
                    expected[b, 0, h] = (sc.softmax(0) @ v[b, g, keep].float()).to(torch.bfloat16)

        self.assertTrue(all(0 < c < length for c in counts), counts)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_offload_accepts_a_view_of_a_longer_pinned_buffer(self):
        """The end-to-end cache hands the operator host K/V that is a *slice* of a
        ring buffer sized prefill+max_new_tokens, so the gather takes explicit
        host strides instead of assuming a contiguous [B,num_kv,k_len,D]."""
        q, gpu_k, gpu_v, host_k, host_v, meta = _make_case(2, 4, 7, 200, seed=11)
        extension = _load_extension()
        scale = 128**-0.5
        length = gpu_k.shape[2]
        reference = extension.sparse_decode_attn_compact(
            q, gpu_k, gpu_v, *meta, scale, 8, length)

        pad = 137
        big_k = torch.empty(gpu_k.shape[0], gpu_k.shape[1], length + pad, 128,
                            dtype=torch.bfloat16).pin_memory()
        big_v = torch.empty_like(big_k).pin_memory()
        big_k[:, :, :length].copy_(gpu_k)
        big_v[:, :, :length].copy_(gpu_v)
        big_k[:, :, length:].fill_(float("nan"))     # reading past k_len must not happen
        big_v[:, :, length:].fill_(float("nan"))
        view_k, view_v = big_k[:, :, :length], big_v[:, :, :length]
        self.assertFalse(view_k.is_contiguous())
        out = extension.sparse_decode_attn_offload(
            q, view_k, view_v, *meta, scale, 8, length, 0)
        torch.testing.assert_close(out, reference, rtol=2e-2, atol=2e-2)
        out_async = extension.sparse_decode_attn_offload(
            q, view_k, view_v, *meta, scale, 8, length, 1)
        torch.testing.assert_close(out_async, reference, rtol=2e-2, atol=2e-2)

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

    def test_separate_stream_gather_matches_sync_across_batch_sizes(self):
        extension = _load_extension()
        length, kv_heads, heads_per_group = 512, 4, 7
        scale = 128**-0.5
        for batch in (1, 2, 3, 5):
            with self.subTest(batch=batch):
                q, gpu_k, gpu_v, host_k, host_v, metadata = _make_case(
                    batch, kv_heads, heads_per_group, length, seed=batch
                )
                resident = extension.sparse_decode_attn_compact(
                    q, gpu_k, gpu_v, *metadata, scale, 32, length
                )
                sync = extension.sparse_decode_attn_offload(
                    q, host_k, host_v, *metadata, scale, 32, length, 0
                )
                separate = extension.sparse_decode_attn_offload(
                    q, host_k, host_v, *metadata, scale, 32, length, 1
                )
                # cap == length, so the compaction cannot drop a selected key and all
                # three paths attend over exactly the same keys.
                torch.testing.assert_close(sync, resident, rtol=2e-2, atol=2e-2)
                torch.testing.assert_close(separate, resident, rtol=2e-2, atol=2e-2)
                torch.testing.assert_close(separate, sync, rtol=2e-2, atol=2e-2)

    def test_repeated_separate_stream_calls_survive_allocator_reuse(self):
        """Repeat under allocator churn: the staging/index buffers are allocated on
        the compute stream but written on the gather stream, so a missing event edge
        or record_stream shows up as a corrupted output on some later iteration."""
        extension = _load_extension()
        batch, kv_heads, heads_per_group, length = 3, 4, 7, 512
        scale = 128**-0.5
        q, gpu_k, gpu_v, host_k, host_v, metadata = _make_case(
            batch, kv_heads, heads_per_group, length, seed=7
        )
        reference = extension.sparse_decode_attn_compact(
            q, gpu_k, gpu_v, *metadata, scale, 32, length
        )
        stage_numel = batch * kv_heads * length * 128
        for i in range(40):
            # Free-and-reallocate a block the size of a staging buffer between calls
            # so the caching allocator hands the same memory back to the next gather.
            churn = torch.empty(stage_numel, device="cuda", dtype=torch.bfloat16)
            churn.fill_(float(i))
            del churn
            out = extension.sparse_decode_attn_offload(
                q, host_k, host_v, *metadata, scale, 32, length,
                1 if i % 2 else 0,     # interleave separate-stream and sync gathers
            )
            torch.testing.assert_close(
                out, reference, rtol=2e-2, atol=2e-2, msg=f"iteration {i}"
            )
            del out
        torch.cuda.synchronize()

    def test_async_gather_argument_is_validated(self):
        extension = _load_extension()
        q, _, _, host_k, host_v, metadata = _make_case(1, 4, 7, 128, seed=1)
        scale = 128**-0.5
        # -1 (the default) means "consult GSA_ASYNC_GATHER"; only -1/0/1 are legal.
        extension.sparse_decode_attn_offload(
            q, host_k, host_v, *metadata, scale, 32, 128, -1
        )
        with self.assertRaisesRegex(RuntimeError, "async_gather"):
            extension.sparse_decode_attn_offload(
                q, host_k, host_v, *metadata, scale, 32, 128, 2
            )


if __name__ == "__main__":
    unittest.main()
