// Fused sparse flash-decoding attention kernel.
//
// Computes, for a single decode query (q_len=1), attention over ONLY the
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

// Parallel compaction: MANY blocks per group (grid (num_kv, blocks_per_group))
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
    int num_kv, int k_len, int cap, int G0p1, int G1p1, int blocks_per_group)
{
    const int g = blockIdx.x;
    const int blk = blockIdx.y;
    const int S = (k_len + blocks_per_group - 1) / blocks_per_group;
    const int k0 = blk * S;
    int k1 = k0 + S; if (k1 > k_len) k1 = k_len;
    const uint8_t* s0g = sel0_any + (size_t)g * G0p1;
    const uint8_t* s1g = sel1_any + (size_t)g * G1p1;
    int* ig = idx_list + (size_t)g * cap;
    const int lane = threadIdx.x & 31;
    // Uniform loop bound (base_k same for all threads) so the whole warp reaches
    // __ballot_sync together; per-thread validity is masked via `valid`.
    for (int base_k = k0; base_k < k1; base_k += blockDim.x) {
        int k = base_k + threadIdx.x;
        bool valid = (k < k1);
        bool keep = false;
        if (valid) {
            bool isgf = is_gist[k] != 0;
            bool s0 = s0g[chunk_l0[k]] != 0;
            bool s1 = s1g[chunk_l1[k]] != 0;
            keep = ((s0 && compressed[k] && !isgf) | (is_l0g[k] && s0)
                  | (is_l1g[k] && s1) | (am_bool[k] && !isgf));
        }
        unsigned m = __ballot_sync(0xffffffff, keep);
        int n = __popc(m);
        if (n > 0) {
            int leader = __ffs(m) - 1;
            int base = 0;
            if (lane == leader) base = atomicAdd(&counts[g], n);
            base = __shfl_sync(0xffffffff, base, leader);
            if (keep) {
                int rank = __popc(m & ((1u << lane) - 1));
                int pos = base + rank;
                if (pos < cap) ig[pos] = k;
            }
        }
    }
}

// Combine partials across splits. One block per head, D_HEAD threads.
__global__ void sparse_decode_combine_kernel(
    const float* __restrict__ partial_m,   // [H, nsplit]
    const float* __restrict__ partial_l,   // [H, nsplit]
    const float* __restrict__ partial_acc, // [H, nsplit, D]
    __nv_bfloat16* __restrict__ out,        // [H, D]  (caller views as [1,1,H,D])
    int H, int nsplit)
{
    const int h = blockIdx.x;
    const int d = threadIdx.x;  // 0..D_HEAD-1
    if (h >= H) return;

    // global max over splits
    float m_glob = -INFINITY;
    for (int s = 0; s < nsplit; ++s) {
        float ms = partial_m[(size_t)h * nsplit + s];
        if (ms > m_glob) m_glob = ms;
    }
    if (m_glob == -INFINITY) {  // no kept keys at all (shouldn't happen)
        out[(size_t)h * D_HEAD + d] = __float2bfloat16(0.f);
        return;
    }
    float l_tot = 0.f;
    float acc = 0.f;
    for (int s = 0; s < nsplit; ++s) {
        float ms = partial_m[(size_t)h * nsplit + s];
        float ls = partial_l[(size_t)h * nsplit + s];
        if (ls == 0.f) continue;
        float w = __expf(ms - m_glob);
        l_tot += ls * w;
        acc += w * partial_acc[((size_t)h * nsplit + s) * D_HEAD + d];
    }
    float o = (l_tot > 0.f) ? (acc / l_tot) : 0.f;
    out[(size_t)h * D_HEAD + d] = __float2bfloat16(o);
}

// Compact-attention: per-GROUP shared-memory tiled flash-decode. Loads each
// kept key's K/V ONCE per group into shared memory and reuses it across the HPG
// heads (the v1 kernel re-loads K/V per head => HPG x redundant load
// instructions / L1 pressure). One block = (group g, split s); HPG warps.
template <int HPG, int DPL /* D_HEAD/32 */, int BLOCK_N>
__global__ void sparse_decode_partial_compact_kernel(
    const __nv_bfloat16* __restrict__ q,    // [H, D]
    const __nv_bfloat16* __restrict__ K,    // [num_kv, k_len, D]
    const __nv_bfloat16* __restrict__ V,    // [num_kv, k_len, D]
    const int* __restrict__ idx_list,       // [num_kv, cap]
    const int* __restrict__ counts,         // [num_kv]
    const float* __restrict__ causal_bias,  // [k_len]
    float* __restrict__ partial_m,          // [H, nsplit]
    float* __restrict__ partial_l,          // [H, nsplit]
    float* __restrict__ partial_acc,        // [H, nsplit, D]
    int H, int num_kv, int k_len, int cap, int nsplit, float scaling)
{
    const int g = blockIdx.x;
    const int s = blockIdx.y;
    const int hg = threadIdx.y;             // warp = head within group
    const int lane = threadIdx.x;           // 0..31
    const int tid = hg * 32 + lane;
    const int nthreads = HPG * 32;
    const int h = g * HPG + hg;

    int count = counts[g];
    if (count > cap) count = cap;   // parallel compaction may overshoot cap
    int sc_g = (count + nsplit - 1) / nsplit;
    if (sc_g < 1) sc_g = 1;
    int i_start = s * sc_g;
    int i_end = i_start + sc_g;
    if (i_end > count) i_end = count;

    __shared__ __nv_bfloat16 sK[BLOCK_N][D_HEAD];
    __shared__ __nv_bfloat16 sV[BLOCK_N][D_HEAD];
    __shared__ float sCB[BLOCK_N];

    const __nv_bfloat16* Kg = K + (size_t)g * k_len * D_HEAD;
    const __nv_bfloat16* Vg = V + (size_t)g * k_len * D_HEAD;
    const int* ig = idx_list + (size_t)g * cap;

    float qf[DPL];
#pragma unroll
    for (int i = 0; i < DPL; ++i)
        qf[i] = (h < H) ? __bfloat162float(q[h * D_HEAD + lane + 32 * i]) : 0.f;

    float m = -INFINITY, l = 0.f, acc[DPL];
#pragma unroll
    for (int i = 0; i < DPL; ++i) acc[i] = 0.f;

    for (int t0 = i_start; t0 < i_end; t0 += BLOCK_N) {
        int tn = i_end - t0; if (tn > BLOCK_N) tn = BLOCK_N;
        // cooperative coalesced load of tn keys' K/V (+causal bias) into shared
        for (int e = tid; e < tn * D_HEAD; e += nthreads) {
            int j = e / D_HEAD, dd = e - j * D_HEAD;
            int k = ig[t0 + j];
            sK[j][dd] = Kg[(size_t)k * D_HEAD + dd];
            sV[j][dd] = Vg[(size_t)k * D_HEAD + dd];
        }
        for (int j = tid; j < tn; j += nthreads) sCB[j] = causal_bias[ig[t0 + j]];
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
        const size_t base = (size_t)h * nsplit + s;
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
torch::Tensor sparse_decode_attn_compact(
    torch::Tensor q, torch::Tensor K, torch::Tensor V,
    torch::Tensor sel0_any, torch::Tensor sel1_any,
    torch::Tensor chunk_l0, torch::Tensor chunk_l1,
    torch::Tensor is_gist, torch::Tensor is_l0g, torch::Tensor is_l1g,
    torch::Tensor compressed, torch::Tensor am_bool, torch::Tensor causal_bias,
    double scaling, int64_t nsplit, int64_t cap)
{
    int H = q.size(1), num_kv = K.size(1), k_len = K.size(2), D = q.size(3);
    TORCH_CHECK(D == D_HEAD, "kernel compiled for D_HEAD=", D_HEAD, " got ", D);
    // Explicit invariants — this kernel is a single-token (q_len=1), batch-1, bf16
    // GQA flash-decode. Fail loudly rather than silently computing the wrong thing
    // (e.g. B>1 would only attend the first sample and return [1,1,H,D]).
    TORCH_CHECK(q.is_cuda() && K.is_cuda() && V.is_cuda(), "q/K/V must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && K.scalar_type() == at::kBFloat16
                && V.scalar_type() == at::kBFloat16, "q/K/V must be bf16");
    TORCH_CHECK(q.dim() == 4 && q.size(0) == 1 && q.size(2) == 1,
                "decode kernel requires q shape [B=1, H, q_len=1, D]; got ", q.sizes());
    TORCH_CHECK(K.size(0) == 1 && V.size(0) == 1, "decode kernel requires batch size 1; got K batch ", K.size(0));
    TORCH_CHECK(num_kv > 0 && H % num_kv == 0, "H must be divisible by num_kv; got H=", H, " num_kv=", num_kv);
    TORCH_CHECK(q.is_contiguous() && K.is_contiguous() && V.is_contiguous(), "q/K/V must be contiguous");
    int hpg = H / num_kv;
    int G0p1 = sel0_any.size(1), G1p1 = sel1_any.size(1);
    if (cap > k_len) cap = k_len;
    auto iopt = torch::TensorOptions().dtype(torch::kInt32).device(q.device());
    auto idx_list = torch::empty({num_kv, (long)cap}, iopt);
    auto counts = torch::zeros({num_kv}, iopt);   // zeroed for the parallel atomicAdd
    auto stream = at::cuda::getCurrentCUDAStream();

    // parallel compaction: ~enough blocks/group to saturate the GPU on the O(k_len) scan
    int blocks_per_group = (k_len + 256 - 1) / 256;
    if (blocks_per_group > 512) blocks_per_group = 512;
    if (blocks_per_group < 1) blocks_per_group = 1;
    build_compact_idx_kernel<<<dim3(num_kv, blocks_per_group), 256, 0, stream>>>(
        sel0_any.data_ptr<uint8_t>(), sel1_any.data_ptr<uint8_t>(),
        chunk_l0.data_ptr<int>(), chunk_l1.data_ptr<int>(),
        is_gist.data_ptr<uint8_t>(), is_l0g.data_ptr<uint8_t>(), is_l1g.data_ptr<uint8_t>(),
        compressed.data_ptr<uint8_t>(), am_bool.data_ptr<uint8_t>(),
        idx_list.data_ptr<int>(), counts.data_ptr<int>(),
        num_kv, k_len, (int)cap, G0p1, G1p1, blocks_per_group);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    auto partial_m = torch::empty({H, (long)nsplit}, fopt);
    auto partial_l = torch::empty({H, (long)nsplit}, fopt);
    auto partial_acc = torch::empty({H, (long)nsplit, D}, fopt);
    auto out = torch::empty({1, 1, H, D}, q.options());

    dim3 grid(num_kv, nsplit);
    dim3 block(32, hpg);
    const __nv_bfloat16* qp = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>());
    const __nv_bfloat16* Kp = reinterpret_cast<const __nv_bfloat16*>(K.data_ptr<at::BFloat16>());
    const __nv_bfloat16* Vp = reinterpret_cast<const __nv_bfloat16*>(V.data_ptr<at::BFloat16>());

#define LAUNCH_COMPACT(HPGV) \
    sparse_decode_partial_compact_kernel<HPGV, D_HEAD/32, 64><<<grid, block, 0, stream>>>( \
        qp, Kp, Vp, idx_list.data_ptr<int>(), counts.data_ptr<int>(), causal_bias.data_ptr<float>(), \
        partial_m.data_ptr<float>(), partial_l.data_ptr<float>(), partial_acc.data_ptr<float>(), \
        H, num_kv, k_len, (int)cap, (int)nsplit, (float)scaling)
    switch (hpg) {
        case 1: LAUNCH_COMPACT(1); break;  case 2: LAUNCH_COMPACT(2); break;
        case 4: LAUNCH_COMPACT(4); break;  case 7: LAUNCH_COMPACT(7); break;
        case 8: LAUNCH_COMPACT(8); break;  case 16: LAUNCH_COMPACT(16); break;
        default: TORCH_CHECK(false, "unsupported heads_per_group=", hpg);
    }
#undef LAUNCH_COMPACT
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    sparse_decode_combine_kernel<<<dim3(H), dim3(D), 0, stream>>>(
        partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        H, (int)nsplit);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_decode_attn_compact", &sparse_decode_attn_compact, "compact per-group shared-mem tiled sparse flash-decode (CUDA)");
}
