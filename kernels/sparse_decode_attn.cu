// Fused sparse flash-decoding attention kernel.
//
// Computes, for a batch of single-token decode queries (q_len=1), attention over ONLY the
// gist-selected keys, with the keep set computed ON THE FLY (no [H,k_len] mask
// materialization, no full-K SDPA). FlashDecoding split-K for occupancy.
//
// Semantics (allclose to gist_qwen2.eager_attention_forward_decoding):
//   For kv group g and key k:
//     sel0 = sel_l0_any[g, chunk_l0[k]]   (OR of selected_chunks_l0 over heads in g)
//     sel1 = sel_l1_any[g, chunk_l1[k]]
//     keep = (sel0 && compressed[k] && !is_gist[k])
//          | (is_l0g[k] && sel0)
//          | (is_l1g[k] && sel1)
//          | (am_bool[k] && !is_gist[k])
//   score_h(k) = scaling * dot(q[h], K[g,k]) + causal_bias[k]   for kept k
//   out[h] = softmax over kept k (score) @ V[g,k]
//   (heads h in group g share the same keep set — group union.)
//
// fp32 accumulation throughout → matches SDPA's fp32-accum efficient_attention
// to within ~1 ulp (allclose passes).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

#ifndef D_HEAD
#define D_HEAD 128
#endif

namespace {

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

// Parallel compaction: MANY blocks per batch/group
// (grid (B * num_kv, blocks_per_group))
// instead of 1 — the v1 build_compact_idx launches only num_kv(=4) blocks for an
// O(k_len) scan (~0.4% GPU util => it dominates). Warp-aggregated global atomicAdd
// minimizes contention. counts[] must be zeroed before launch. Order is arbitrary
// (fine: attention is order-independent). count may exceed cap; readers clamp.
__global__ void build_compact_idx_kernel(
    const uint8_t* __restrict__ sel0_any, const uint8_t* __restrict__ sel1_any,
    const int* __restrict__ chunk_l0, const int* __restrict__ chunk_l1,
    const uint8_t* __restrict__ is_gist, const uint8_t* __restrict__ is_l0g,
    const uint8_t* __restrict__ is_l1g, const uint8_t* __restrict__ compressed,
    const uint8_t* __restrict__ am_bool,
    int* __restrict__ idx_list, int* __restrict__ counts,
    int B, int num_kv, int k_len, int cap, int G0p1, int G1p1, int blocks_per_group)
{
    const int bg = blockIdx.x;
    const int b = bg / num_kv;
    const int g = bg - b * num_kv;
    const int blk = blockIdx.y;
    const int S = (k_len + blocks_per_group - 1) / blocks_per_group;
    const int k0 = blk * S;
    int k1 = k0 + S; if (k1 > k_len) k1 = k_len;
    const uint8_t* s0g = sel0_any + (size_t)bg * G0p1;
    const uint8_t* s1g = sel1_any + (size_t)bg * G1p1;
    const uint8_t* is_gist_b = is_gist + (size_t)b * k_len;
    const uint8_t* is_l0g_b = is_l0g + (size_t)b * k_len;
    const uint8_t* is_l1g_b = is_l1g + (size_t)b * k_len;
    const uint8_t* compressed_b = compressed + (size_t)b * k_len;
    const uint8_t* am_bool_b = am_bool + (size_t)b * k_len;
    const int* chunk_l0_i32_b = chunk_l0 + (size_t)b * k_len;
    const int* chunk_l1_b = chunk_l1 + (size_t)b * k_len;
    int* ig = idx_list + (size_t)bg * cap;
    const int lane = threadIdx.x & 31;
    // Uniform loop bound (base_k same for all threads) so the whole warp reaches
    // __ballot_sync together; per-thread validity is masked via `valid`.
    for (int base_k = k0; base_k < k1; base_k += blockDim.x) {
        int k = base_k + threadIdx.x;
        bool valid = (k < k1);
        bool keep = false;
        if (valid) {
            bool isgf = is_gist_b[k] != 0;
            bool s0 = s0g[chunk_l0_i32_b[k]] != 0;
            bool s1 = s1g[chunk_l1_b[k]] != 0;
            keep = ((s0 && compressed_b[k] && !isgf) | (is_l0g_b[k] && s0)
                  | (is_l1g_b[k] && s1) | (am_bool_b[k] && !isgf));
        }
        unsigned m = __ballot_sync(0xffffffff, keep);
        int n = __popc(m);
        if (n > 0) {
            int leader = __ffs(m) - 1;
            int base = 0;
            if (lane == leader) base = atomicAdd(&counts[bg], n);
            base = __shfl_sync(0xffffffff, base, leader);
            if (keep) {
                int rank = __popc(m & ((1u << lane) - 1));
                int pos = base + rank;
                if (pos < cap) ig[pos] = k;
            }
        }
    }
}

// ponytail: overwrite the compact staging buffer every step; add vectorized
// copies and an LRU hot buffer only after the prototype measures worthwhile I/O.
__global__ void gather_compact_kv_from_host_kernel(
    const __nv_bfloat16* __restrict__ host_K,  // pinned CPU [B, num_kv, k_len, D]
    const __nv_bfloat16* __restrict__ host_V,
    const int* __restrict__ idx_list,          // GPU [B, num_kv, cap]
    const int* __restrict__ counts,            // GPU [B, num_kv]
    __nv_bfloat16* __restrict__ stage_K,       // GPU [B, num_kv, cap, D]
    __nv_bfloat16* __restrict__ stage_V,
    int k_len, int cap)
{
    const int bg = blockIdx.y;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = e / D_HEAD;
    const int d = e - i * D_HEAD;
    int count = counts[bg];
    if (count > cap) count = cap;
    if (i >= count) return;

    const int logical_k = idx_list[(size_t)bg * cap + i];
    const size_t src = ((size_t)bg * k_len + logical_k) * D_HEAD + d;
    const size_t dst = ((size_t)bg * cap + i) * D_HEAD + d;
    stage_K[dst] = host_K[src];
    stage_V[dst] = host_V[src];
}

// Combine partials across splits. One block per batch/head, D_HEAD threads.
__global__ void sparse_decode_combine_kernel(
    const float* __restrict__ partial_m,   // [B, H, nsplit]
    const float* __restrict__ partial_l,   // [B, H, nsplit]
    const float* __restrict__ partial_acc, // [B, H, nsplit, D]
    __nv_bfloat16* __restrict__ out,       // [B, 1, H, D]
    int B, int H, int nsplit)
{
    const int bh = blockIdx.x;
    const int d = threadIdx.x;  // 0..D_HEAD-1
    if (bh >= B * H) return;

    // global max over splits
    float m_glob = -INFINITY;
    for (int s = 0; s < nsplit; ++s) {
        float ms = partial_m[(size_t)bh * nsplit + s];
        if (ms > m_glob) m_glob = ms;
    }
    if (m_glob == -INFINITY) {  // no kept keys at all (shouldn't happen)
        out[(size_t)bh * D_HEAD + d] = __float2bfloat16(0.f);
        return;
    }
    float l_tot = 0.f;
    float acc = 0.f;
    for (int s = 0; s < nsplit; ++s) {
        float ms = partial_m[(size_t)bh * nsplit + s];
        float ls = partial_l[(size_t)bh * nsplit + s];
        if (ls == 0.f) continue;
        float w = __expf(ms - m_glob);
        l_tot += ls * w;
        acc += w * partial_acc[((size_t)bh * nsplit + s) * D_HEAD + d];
    }
    float o = (l_tot > 0.f) ? (acc / l_tot) : 0.f;
    out[(size_t)bh * D_HEAD + d] = __float2bfloat16(o);
}

// Compact-attention: per-GROUP shared-memory tiled flash-decode. Loads each
// kept key's K/V ONCE per group into shared memory and reuses it across the HPG
// heads (the v1 kernel re-loads K/V per head => HPG x redundant load
// instructions / L1 pressure). One block = (group g, split s); HPG warps.
template <int HPG, int DPL /* D_HEAD/32 */, int BLOCK_N, bool STAGED>
__global__ void sparse_decode_partial_compact_kernel(
    const __nv_bfloat16* __restrict__ q,    // [B, H, 1, D]
    const __nv_bfloat16* __restrict__ K,    // [B, num_kv, k_len, D]
    const __nv_bfloat16* __restrict__ V,    // [B, num_kv, k_len, D]
    const int* __restrict__ idx_list,       // [B, num_kv, cap]
    const int* __restrict__ counts,         // [B, num_kv]
    const float* __restrict__ causal_bias,  // [B, k_len]
    float* __restrict__ partial_m,          // [B, H, nsplit]
    float* __restrict__ partial_l,          // [B, H, nsplit]
    float* __restrict__ partial_acc,        // [B, H, nsplit, D]
    int B, int H, int num_kv, int k_len, int kv_stride, int cap, int nsplit, float scaling)
{
    const int bg = blockIdx.x;
    const int b = bg / num_kv;
    const int g = bg - b * num_kv;
    const int s = blockIdx.y;
    const int hg = threadIdx.y;             // warp = head within group
    const int lane = threadIdx.x;           // 0..31
    const int tid = hg * 32 + lane;
    const int nthreads = HPG * 32;
    const int h = g * HPG + hg;

    int count = counts[bg];
    if (count > cap) count = cap;   // parallel compaction may overshoot cap
    int sc_g = (count + nsplit - 1) / nsplit;
    if (sc_g < 1) sc_g = 1;
    int i_start = s * sc_g;
    int i_end = i_start + sc_g;
    if (i_end > count) i_end = count;

    __shared__ __nv_bfloat16 sK[BLOCK_N][D_HEAD];
    __shared__ __nv_bfloat16 sV[BLOCK_N][D_HEAD];
    __shared__ float sCB[BLOCK_N];

    const __nv_bfloat16* Kg = K + (size_t)bg * kv_stride * D_HEAD;
    const __nv_bfloat16* Vg = V + (size_t)bg * kv_stride * D_HEAD;
    const int* ig = idx_list + (size_t)bg * cap;
    const float* causal_bias_b = causal_bias + (size_t)b * k_len;

    float qf[DPL];
#pragma unroll
    for (int i = 0; i < DPL; ++i)
        qf[i] = (h < H) ? __bfloat162float(q[((size_t)b * H + h) * D_HEAD + lane + 32 * i]) : 0.f;

    float m = -INFINITY, l = 0.f, acc[DPL];
#pragma unroll
    for (int i = 0; i < DPL; ++i) acc[i] = 0.f;

    for (int t0 = i_start; t0 < i_end; t0 += BLOCK_N) {
        int tn = i_end - t0; if (tn > BLOCK_N) tn = BLOCK_N;
        // cooperative coalesced load of tn keys' K/V (+causal bias) into shared
        for (int e = tid; e < tn * D_HEAD; e += nthreads) {
            int j = e / D_HEAD, dd = e - j * D_HEAD;
            int k = ig[t0 + j];
            int kv_k = STAGED ? (t0 + j) : k;
            sK[j][dd] = Kg[(size_t)kv_k * D_HEAD + dd];
            sV[j][dd] = Vg[(size_t)kv_k * D_HEAD + dd];
        }
        for (int j = tid; j < tn; j += nthreads) sCB[j] = causal_bias_b[ig[t0 + j]];
        __syncthreads();

        if (h < H) {
            for (int j = 0; j < tn; ++j) {
                float partial = 0.f;
#pragma unroll
                for (int i = 0; i < DPL; ++i)
                    partial += qf[i] * __bfloat162float(sK[j][lane + 32 * i]);
                float dot = warp_reduce_sum(partial);
                float scr = scaling * dot + sCB[j];
                float m_new = fmaxf(m, scr);
                float corr = __expf(m - m_new);
                float p = __expf(scr - m_new);
                l = l * corr + p;
#pragma unroll
                for (int i = 0; i < DPL; ++i)
                    acc[i] = acc[i] * corr + p * __bfloat162float(sV[j][lane + 32 * i]);
                m = m_new;
            }
        }
        __syncthreads();  // before reusing shared for next tile
    }

    if (h < H) {
        const size_t base = ((size_t)b * H + h) * nsplit + s;
        if (lane == 0) { partial_m[base] = m; partial_l[base] = l; }
        float* acc_out = partial_acc + base * D_HEAD;
#pragma unroll
        for (int i = 0; i < DPL; ++i) acc_out[lane + 32 * i] = acc[i];
    }
}

}  // namespace

// Compaction + split-K sparse flash-decode: parallel warp-aggregated compaction
// (build_compact_idx) + per-group shared-mem tiled partial (K/V loaded once per
// group, reused across the HPG heads) + online-softmax combine.
torch::Tensor sparse_decode_attn_compact_impl(
    torch::Tensor q, torch::Tensor K, torch::Tensor V,
    torch::Tensor sel0_any, torch::Tensor sel1_any,
    torch::Tensor chunk_l0, torch::Tensor chunk_l1,
    torch::Tensor is_gist, torch::Tensor is_l0g, torch::Tensor is_l1g,
    torch::Tensor compressed, torch::Tensor am_bool, torch::Tensor causal_bias,
    double scaling, int64_t nsplit, int64_t cap, bool host_offload)
{
    int B = q.size(0), H = q.size(1), num_kv = K.size(1), k_len = K.size(2), D = q.size(3);
    TORCH_CHECK(D == D_HEAD, "kernel compiled for D_HEAD=", D_HEAD, " got ", D);
    // Explicit invariants — this kernel is a batched single-token (q_len=1), bf16
    // GQA flash-decode.
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    if (host_offload) {
        TORCH_CHECK(!K.is_cuda() && !V.is_cuda() && K.is_pinned() && V.is_pinned(),
                    "offloaded K/V must be pinned CPU tensors");
    } else {
        TORCH_CHECK(K.is_cuda() && V.is_cuda(), "resident K/V must be CUDA tensors");
    }
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && K.scalar_type() == at::kBFloat16
                && V.scalar_type() == at::kBFloat16, "q/K/V must be bf16");
    TORCH_CHECK(nsplit > 0 && cap > 0, "nsplit and cap must be positive");
    TORCH_CHECK(q.dim() == 4 && q.size(2) == 1,
                "decode kernel requires q shape [B, H, q_len=1, D]; got ", q.sizes());
    TORCH_CHECK(K.dim() == 4 && V.dim() == 4 && K.sizes() == V.sizes(),
                "K/V must have matching [B, num_kv, k_len, D] shapes");
    TORCH_CHECK(K.size(0) == B, "q/K batch mismatch: ", B, " vs ", K.size(0));
    TORCH_CHECK(K.size(3) == D, "q/K head-dim mismatch: ", D, " vs ", K.size(3));
    TORCH_CHECK(num_kv > 0 && H % num_kv == 0, "H must be divisible by num_kv; got H=", H, " num_kv=", num_kv);
    TORCH_CHECK(q.is_contiguous() && K.is_contiguous() && V.is_contiguous(), "q/K/V must be contiguous");
    int hpg = H / num_kv;
    TORCH_CHECK(sel0_any.dim() == 3 && sel1_any.dim() == 3
                && sel0_any.size(0) == B && sel0_any.size(1) == num_kv
                && sel1_any.size(0) == B && sel1_any.size(1) == num_kv,
                "selection tensors must be [B, num_kv, num_chunks]");
    TORCH_CHECK(sel0_any.is_cuda() && sel1_any.is_cuda() && chunk_l0.is_cuda()
                && chunk_l1.is_cuda() && is_gist.is_cuda() && is_l0g.is_cuda()
                && is_l1g.is_cuda() && compressed.is_cuda() && am_bool.is_cuda()
                && causal_bias.is_cuda(), "selection and metadata tensors must be CUDA tensors");
    TORCH_CHECK(sel0_any.is_contiguous() && sel1_any.is_contiguous()
                && chunk_l0.is_contiguous() && chunk_l1.is_contiguous()
                && is_gist.is_contiguous() && is_l0g.is_contiguous()
                && is_l1g.is_contiguous() && compressed.is_contiguous()
                && am_bool.is_contiguous() && causal_bias.is_contiguous(),
                "selection and metadata tensors must be contiguous");
    auto is_bk = [B, k_len](const torch::Tensor& t) {
        return t.dim() == 2 && t.size(0) == B && t.size(1) == k_len;
    };
    TORCH_CHECK(is_bk(chunk_l0) && is_bk(chunk_l1) && is_bk(is_gist)
                && is_bk(is_l0g) && is_bk(is_l1g) && is_bk(compressed)
                && is_bk(am_bool) && is_bk(causal_bias),
                "decode metadata must be [B, k_len]");
    int G0p1 = sel0_any.size(2), G1p1 = sel1_any.size(2);
    if (cap > k_len) cap = k_len;
    auto iopt = torch::TensorOptions().dtype(torch::kInt32).device(q.device());
    auto idx_list = torch::empty({B, num_kv, (long)cap}, iopt);
    auto counts = torch::zeros({B, num_kv}, iopt);   // zeroed for the parallel atomicAdd
    auto stream = at::cuda::getCurrentCUDAStream();

    // parallel compaction: ~enough blocks/group to saturate the GPU on the O(k_len) scan
    int blocks_per_group = (k_len + 256 - 1) / 256;
    if (blocks_per_group > 512) blocks_per_group = 512;
    if (blocks_per_group < 1) blocks_per_group = 1;
    build_compact_idx_kernel<<<dim3(B * num_kv, blocks_per_group), 256, 0, stream>>>(
        sel0_any.data_ptr<uint8_t>(), sel1_any.data_ptr<uint8_t>(),
        chunk_l0.data_ptr<int>(), chunk_l1.data_ptr<int>(),
        is_gist.data_ptr<uint8_t>(), is_l0g.data_ptr<uint8_t>(), is_l1g.data_ptr<uint8_t>(),
        compressed.data_ptr<uint8_t>(), am_bool.data_ptr<uint8_t>(),
        idx_list.data_ptr<int>(), counts.data_ptr<int>(),
        B, num_kv, k_len, (int)cap, G0p1, G1p1, blocks_per_group);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    auto partial_m = torch::empty({B, H, (long)nsplit}, fopt);
    auto partial_l = torch::empty({B, H, (long)nsplit}, fopt);
    auto partial_acc = torch::empty({B, H, (long)nsplit, D}, fopt);
    auto out = torch::empty({B, 1, H, D}, q.options());

    dim3 grid(B * num_kv, nsplit);
    dim3 block(32, hpg);
    const __nv_bfloat16* qp = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>());
    torch::Tensor stage_K, stage_V;
    const __nv_bfloat16* Kp;
    const __nv_bfloat16* Vp;
    int kv_stride = k_len;
    if (host_offload) {
        stage_K = torch::empty({B, num_kv, (long)cap, D}, q.options());
        stage_V = torch::empty_like(stage_K);
        auto host_Kp = reinterpret_cast<const __nv_bfloat16*>(K.data_ptr<at::BFloat16>());
        auto host_Vp = reinterpret_cast<const __nv_bfloat16*>(V.data_ptr<at::BFloat16>());
        auto stage_Kp = reinterpret_cast<__nv_bfloat16*>(stage_K.data_ptr<at::BFloat16>());
        auto stage_Vp = reinterpret_cast<__nv_bfloat16*>(stage_V.data_ptr<at::BFloat16>());
        int copy_blocks = ((int)cap * D + 255) / 256;
        gather_compact_kv_from_host_kernel<<<dim3(copy_blocks, B * num_kv), 256, 0, stream>>>(
            host_Kp, host_Vp, idx_list.data_ptr<int>(), counts.data_ptr<int>(),
            stage_Kp, stage_Vp, k_len, (int)cap);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        Kp = stage_Kp;
        Vp = stage_Vp;
        kv_stride = cap;
    } else {
        Kp = reinterpret_cast<const __nv_bfloat16*>(K.data_ptr<at::BFloat16>());
        Vp = reinterpret_cast<const __nv_bfloat16*>(V.data_ptr<at::BFloat16>());
    }

#define LAUNCH_COMPACT(HPGV, STAGEDV) \
    sparse_decode_partial_compact_kernel<HPGV, D_HEAD/32, 64, STAGEDV><<<grid, block, 0, stream>>>( \
        qp, Kp, Vp, idx_list.data_ptr<int>(), counts.data_ptr<int>(), causal_bias.data_ptr<float>(), \
        partial_m.data_ptr<float>(), partial_l.data_ptr<float>(), partial_acc.data_ptr<float>(), \
        B, H, num_kv, k_len, kv_stride, (int)cap, (int)nsplit, (float)scaling)
#define DISPATCH_HPG(STAGEDV) \
    switch (hpg) { \
        case 1: LAUNCH_COMPACT(1, STAGEDV); break;  case 2: LAUNCH_COMPACT(2, STAGEDV); break; \
        case 4: LAUNCH_COMPACT(4, STAGEDV); break;  case 7: LAUNCH_COMPACT(7, STAGEDV); break; \
        case 8: LAUNCH_COMPACT(8, STAGEDV); break;  case 16: LAUNCH_COMPACT(16, STAGEDV); break; \
        default: TORCH_CHECK(false, "unsupported heads_per_group=", hpg); \
    }
    if (host_offload) { DISPATCH_HPG(true); } else { DISPATCH_HPG(false); }
#undef DISPATCH_HPG
#undef LAUNCH_COMPACT
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    sparse_decode_combine_kernel<<<dim3(B * H), dim3(D), 0, stream>>>(
        partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        B, H, (int)nsplit);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor sparse_decode_attn_compact(
    torch::Tensor q, torch::Tensor K, torch::Tensor V,
    torch::Tensor sel0_any, torch::Tensor sel1_any,
    torch::Tensor chunk_l0, torch::Tensor chunk_l1,
    torch::Tensor is_gist, torch::Tensor is_l0g, torch::Tensor is_l1g,
    torch::Tensor compressed, torch::Tensor am_bool, torch::Tensor causal_bias,
    double scaling, int64_t nsplit, int64_t cap)
{
    return sparse_decode_attn_compact_impl(
        q, K, V, sel0_any, sel1_any, chunk_l0, chunk_l1,
        is_gist, is_l0g, is_l1g, compressed, am_bool, causal_bias,
        scaling, nsplit, cap, false);
}

torch::Tensor sparse_decode_attn_offload(
    torch::Tensor q, torch::Tensor host_K, torch::Tensor host_V,
    torch::Tensor sel0_any, torch::Tensor sel1_any,
    torch::Tensor chunk_l0, torch::Tensor chunk_l1,
    torch::Tensor is_gist, torch::Tensor is_l0g, torch::Tensor is_l1g,
    torch::Tensor compressed, torch::Tensor am_bool, torch::Tensor causal_bias,
    double scaling, int64_t nsplit, int64_t cap)
{
    return sparse_decode_attn_compact_impl(
        q, host_K, host_V, sel0_any, sel1_any, chunk_l0, chunk_l1,
        is_gist, is_l0g, is_l1g, compressed, am_bool, causal_bias,
        scaling, nsplit, cap, true);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_decode_attn_compact", &sparse_decode_attn_compact, "compact per-group shared-mem tiled sparse flash-decode (CUDA)");
    m.def("sparse_decode_attn_offload", &sparse_decode_attn_offload, "pinned-host KV to compact GPU staging sparse flash-decode (CUDA)");
}
