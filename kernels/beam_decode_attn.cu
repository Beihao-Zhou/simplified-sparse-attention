// Kernel B of DECODE_SELECT=beam (ADR 0001, #20): split-K flash-decode over a per-KV-group
// list of key positions, for one decode query.
//
// For kv group g, the keys are idx[g, 0 .. cap) (entries < 0 are empty slots; the list
// is built from the selection of kernel A: raw tokens of selected leaves, summaries of
// unselected candidates or of selected nodes, pinned, sink and open-chunk tokens).
//   score_h(k) = scaling * dot(q[h], K[g, k]) + bias[g, k]     (bias = beta on summaries)
//   out[h]     = softmax_k(score_h) @ V[g, k]                  (heads of g share the list)
// One block per (group, split) loads each listed key once and reuses it across the HPG
// heads of the group; a combine kernel merges the splits by log-sum-exp. fp32
// accumulation. Head dim D and HPG are template parameters (0.5B: D = 64, HPG = 7;
// Qwen2.5-7B-1M: D = 128, HPG = 7). The 2-level kernel (sparse_decode_attn.cu) is
// separate and unchanged.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <math.h>

namespace {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

template <int D, int HPG, int BLOCK_N>
__global__ void beam_partial_kernel(
    const __nv_bfloat16* __restrict__ q,      // [H, D]
    const __nv_bfloat16* __restrict__ K,      // [num_kv, T, D]
    const __nv_bfloat16* __restrict__ V,      // [num_kv, T, D]
    const int* __restrict__ idx,              // [num_kv, cap], < 0 = empty
    const float* __restrict__ bias,           // [num_kv, T]
    float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc,
    int H, int T, int cap, int nsplit, float scaling)
{
    constexpr int DPL = D / 32;
    const int g = blockIdx.x, s = blockIdx.y;
    const int hg = threadIdx.y, lane = threadIdx.x;
    const int tid = hg * 32 + lane, nthreads = HPG * 32;
    const int h = g * HPG + hg;
    const int per = (cap + nsplit - 1) / nsplit;
    const int i0 = s * per;
    int i1 = i0 + per; if (i1 > cap) i1 = cap;

    __shared__ __nv_bfloat16 sK[BLOCK_N][D];
    __shared__ __nv_bfloat16 sV[BLOCK_N][D];
    __shared__ float sB[BLOCK_N];
    __shared__ int sOk[BLOCK_N];

    const __nv_bfloat16* Kg = K + (size_t)g * T * D;
    const __nv_bfloat16* Vg = V + (size_t)g * T * D;
    const int* ig = idx + (size_t)g * cap;
    const float* bg = bias + (size_t)g * T;

    float qf[DPL];
#pragma unroll
    for (int i = 0; i < DPL; ++i) qf[i] = __bfloat162float(q[h * D + lane + 32 * i]);
    float m = -INFINITY, l = 0.f, acc[DPL];
#pragma unroll
    for (int i = 0; i < DPL; ++i) acc[i] = 0.f;

    for (int t0 = i0; t0 < i1; t0 += BLOCK_N) {
        int tn = i1 - t0; if (tn > BLOCK_N) tn = BLOCK_N;
        for (int e = tid; e < tn * D; e += nthreads) {
            int j = e / D, dd = e - j * D;
            int k = ig[t0 + j];
            __nv_bfloat16 zero = __float2bfloat16(0.f);
            sK[j][dd] = k >= 0 ? Kg[(size_t)k * D + dd] : zero;
            sV[j][dd] = k >= 0 ? Vg[(size_t)k * D + dd] : zero;
        }
        for (int j = tid; j < tn; j += nthreads) {
            int k = ig[t0 + j];
            sOk[j] = k >= 0;
            sB[j] = k >= 0 ? bg[k] : 0.f;
        }
        __syncthreads();
        for (int j = 0; j < tn; ++j) {
            if (!sOk[j]) continue;
            float part = 0.f;
#pragma unroll
            for (int i = 0; i < DPL; ++i) part += qf[i] * __bfloat162float(sK[j][lane + 32 * i]);
            float scr = scaling * warp_sum(part) + sB[j];
            float m_new = fmaxf(m, scr);
            float corr = __expf(m - m_new), p = __expf(scr - m_new);
            l = l * corr + p;
#pragma unroll
            for (int i = 0; i < DPL; ++i)
                acc[i] = acc[i] * corr + p * __bfloat162float(sV[j][lane + 32 * i]);
            m = m_new;
        }
        __syncthreads();
    }
    const size_t base = (size_t)h * nsplit + s;
    if (lane == 0) { pm[base] = m; pl[base] = l; }
#pragma unroll
    for (int i = 0; i < DPL; ++i) pacc[base * D + lane + 32 * i] = acc[i];
}

template <int D>
__global__ void beam_combine_kernel(const float* __restrict__ pm, const float* __restrict__ pl,
                                    const float* __restrict__ pacc, __nv_bfloat16* __restrict__ out,
                                    int nsplit)
{
    const int h = blockIdx.x, d = threadIdx.x;
    float mg = -INFINITY;
    for (int s = 0; s < nsplit; ++s) mg = fmaxf(mg, pm[(size_t)h * nsplit + s]);
    float lt = 0.f, a = 0.f;
    if (mg != -INFINITY) {
        for (int s = 0; s < nsplit; ++s) {
            float ls = pl[(size_t)h * nsplit + s];
            if (ls == 0.f) continue;
            float w = __expf(pm[(size_t)h * nsplit + s] - mg);
            lt += ls * w;
            a += w * pacc[((size_t)h * nsplit + s) * D + d];
        }
    }
    out[(size_t)h * D + d] = __float2bfloat16(lt > 0.f ? a / lt : 0.f);
}

template <int D, int HPG>
void launch(const torch::Tensor& q, const torch::Tensor& K, const torch::Tensor& V,
            const torch::Tensor& idx, const torch::Tensor& bias, torch::Tensor& pm,
            torch::Tensor& pl, torch::Tensor& pacc, torch::Tensor& out, int nsplit, float scaling)
{
    int H = q.size(1), num_kv = K.size(1), T = K.size(2), cap = idx.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto bf = [](const torch::Tensor& t) {
        return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>());
    };
    beam_partial_kernel<D, HPG, 64><<<dim3(num_kv, nsplit), dim3(32, HPG), 0, stream>>>(
        bf(q), bf(K), bf(V), idx.data_ptr<int>(), bias.data_ptr<float>(), pm.data_ptr<float>(),
        pl.data_ptr<float>(), pacc.data_ptr<float>(), H, T, cap, nsplit, scaling);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    beam_combine_kernel<D><<<dim3(H), dim3(D), 0, stream>>>(
        pm.data_ptr<float>(), pl.data_ptr<float>(), pacc.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), nsplit);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// q [1, H, 1, D], K / V [1, num_kv, T, D] (bf16, contiguous), idx [num_kv, cap] int32,
// bias [num_kv, T] float32. Returns [1, 1, H, D] bf16.
torch::Tensor beam_decode_attn(torch::Tensor q, torch::Tensor K, torch::Tensor V,
                               torch::Tensor idx, torch::Tensor bias, double scaling,
                               int64_t nsplit)
{
    TORCH_CHECK(q.is_cuda() && K.is_cuda() && V.is_cuda() && idx.is_cuda() && bias.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && K.scalar_type() == at::kBFloat16 &&
                V.scalar_type() == at::kBFloat16, "q/K/V must be bf16");
    TORCH_CHECK(idx.scalar_type() == at::kInt && bias.scalar_type() == at::kFloat,
                "idx must be int32, bias float32");
    TORCH_CHECK(q.dim() == 4 && q.size(0) == 1 && q.size(2) == 1, "q must be [1, H, 1, D]");
    TORCH_CHECK(K.size(0) == 1 && V.size(0) == 1, "batch 1 only (#24)");
    TORCH_CHECK(q.is_contiguous() && K.is_contiguous() && V.is_contiguous() &&
                idx.is_contiguous() && bias.is_contiguous(), "inputs must be contiguous");
    int H = q.size(1), num_kv = K.size(1), D = q.size(3);
    TORCH_CHECK(H % num_kv == 0, "H must be a multiple of num_kv");
    TORCH_CHECK(idx.size(0) == num_kv && bias.size(0) == num_kv && bias.size(1) == K.size(2),
                "idx / bias shapes must match K");
    int hpg = H / num_kv;
    auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    auto pm = torch::empty({H, nsplit}, fopt), pl = torch::empty({H, nsplit}, fopt);
    auto pacc = torch::empty({H, nsplit, D}, fopt);
    auto out = torch::empty({1, 1, H, D}, q.options());
    float sc = (float)scaling;
    int ns = (int)nsplit;
#define CASE(DV, HV) if (D == DV && hpg == HV) { launch<DV, HV>(q, K, V, idx, bias, pm, pl, pacc, out, ns, sc); return out; }
    CASE(64, 1) CASE(64, 2) CASE(64, 4) CASE(64, 6) CASE(64, 7) CASE(64, 8)
    CASE(128, 1) CASE(128, 2) CASE(128, 4) CASE(128, 6) CASE(128, 7) CASE(128, 8)
#undef CASE
    TORCH_CHECK(false, "unsupported head dim ", D, " / heads per group ", hpg);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("beam_decode_attn", &beam_decode_attn, "kernel B: split-K flash-decode over listed keys");
}
