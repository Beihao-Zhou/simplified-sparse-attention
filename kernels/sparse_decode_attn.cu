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
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <assert.h>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

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

// Gather the selected rows out of pinned host memory into the compact staging
// buffer, overwritten every step. `VEC` bf16 per thread: a 2-byte-per-thread
// gather measured ~5.5 GiB/s against ~17-23 GiB/s for the same scattered
// 256-byte row pattern at 4-16 bytes per thread, so the width is the throughput
// knob, not PCIe.
//
// Grid-stride over `count` rather than one thread per `cap` slot: `cap` is
// k_len on the model path (so that compaction can never drop a selected key),
// which would otherwise launch ~30k blocks per group to do ~1k blocks of work.
template <int VEC>
__global__ void gather_compact_kv_from_host_kernel(
    const __nv_bfloat16* __restrict__ host_K,  // pinned CPU, [B, num_kv, k_len, D] logical
    const __nv_bfloat16* __restrict__ host_V,
    const int* __restrict__ idx_list,          // GPU [B, num_kv, cap]
    const int* __restrict__ counts,            // GPU [B, num_kv]
    __nv_bfloat16* __restrict__ stage_K,       // GPU [B, num_kv, cap, D]
    __nv_bfloat16* __restrict__ stage_V,
    long host_bg_stride, long host_k_stride, int cap)
{
    const int bg = blockIdx.y;
    int count = counts[bg];
    if (count > cap) count = cap;
    constexpr int PER_ROW = D_HEAD / VEC;
    const long total = (long)count * PER_ROW;
    const long stride = (long)gridDim.x * blockDim.x;
    for (long e = (long)blockIdx.x * blockDim.x + threadIdx.x; e < total; e += stride) {
        const int i = (int)(e / PER_ROW);
        const int d = (int)(e - (long)i * PER_ROW) * VEC;
        const int logical_k = idx_list[(size_t)bg * cap + i];
        const size_t src = (size_t)bg * host_bg_stride + (size_t)logical_k * host_k_stride + d;
        const size_t dst = ((size_t)bg * cap + i) * D_HEAD + d;
        if (VEC == 8) {
            *reinterpret_cast<float4*>(stage_K + dst) =
                *reinterpret_cast<const float4*>(host_K + src);
            *reinterpret_cast<float4*>(stage_V + dst) =
                *reinterpret_cast<const float4*>(host_V + src);
        } else if (VEC == 2) {
            *reinterpret_cast<float*>(stage_K + dst) =
                *reinterpret_cast<const float*>(host_K + src);
            *reinterpret_cast<float*>(stage_V + dst) =
                *reinterpret_cast<const float*>(host_V + src);
        } else {
            stage_K[dst] = host_K[src];
            stage_V[dst] = host_V[src];
        }
    }
}

// Threads per (batch, KV-group) in the LRU resolver. 1024 measured fastest of
// {128, 256, 512, 1024} at hot_size 8192 (12-14 us vs 27-28 at 256) and was never
// worse at 512; override at build time to re-measure. See the report.
#ifndef HOT_RESOLVE_BLOCK_SIZE
#define HOT_RESOLVE_BLOCK_SIZE 1024
#endif
constexpr int HOT_RESOLVE_BLOCK = HOT_RESOLVE_BLOCK_SIZE;

// Order-preserving block-wide exclusive scan of a 0/1 predicate. Returns this
// thread's exclusive prefix and writes the block sum to `total`. Every thread in
// the block must call it: it synchronises internally.
template <int BLOCK>
__device__ inline int block_scan_exclusive(int flag, int* s_warp, int& total)
{
    constexpr int NWARP = BLOCK / 32;
    // 64-bit shift then narrow: at BLOCK=1024, NWARP is 32 and (1u << 32) would
    // be undefined -- in practice mask 0, which silently breaks the shuffle.
    constexpr unsigned NWARP_MASK = (unsigned)((1ull << NWARP) - 1ull);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __syncthreads();                     // s_warp is reused on every tile
    int x = flag;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, off);
        if (lane >= off) x += y;
    }
    if (lane == 31) s_warp[warp] = x;
    __syncthreads();
    if (threadIdx.x < NWARP) {
        int w = s_warp[threadIdx.x];
#pragma unroll
        for (int off = 1; off < NWARP; off <<= 1) {
            const int y = __shfl_up_sync(NWARP_MASK, w, off);
            if ((int)threadIdx.x >= off) w += y;
        }
        s_warp[threadIdx.x] = w;
    }
    __syncthreads();
    total = s_warp[NWARP - 1];
    return (warp ? s_warp[warp - 1] : 0) + x - flag;
}

// Resolve logical selected positions against a persistent per-(batch, KV-group)
// hot buffer, and update exact step-granularity LRU.
//
// One *block* owns one group. The cost of this kernel is three O(hot_size)
// passes -- clear the marks, compact the evictable slots, rebuild the order --
// so they run cooperatively with coalesced accesses. A one-thread-per-group
// version measured 1.15 ms at hot_size=8192 because every pass was a serial
// chain of dependent global loads.
// Precondition: the selected set of a group holds no duplicate tokens
// (build_compact_idx_kernel appends each kept position once). A duplicate would
// leave two slots claiming one token; the closing assert catches it in debug.
template <int BLOCK>
__global__ void resolve_hot_slots_lru_kernel(
    const int* __restrict__ idx_list, const int* __restrict__ counts,
    int* __restrict__ slot_list,
    int* __restrict__ token_to_slot, int* __restrict__ slot_token_ids,
    int* __restrict__ lru_order, int* __restrict__ lru_scratch,
    uint8_t* __restrict__ slot_marks,
    int* __restrict__ miss_tokens, int* __restrict__ miss_slots,
    int* __restrict__ miss_counts,
    int64_t* __restrict__ stats,
    int cap, int token_capacity, int hot_size)
{
    constexpr int NWARP = BLOCK / 32;
    __shared__ int s_warp[NWARP];
    __shared__ int s_evictable;
    __shared__ int s_misses;

    const int bg = blockIdx.x;
    int count = counts[bg];
    if (count > cap) count = cap;
    // A selected set must fit, so that no token this step needs can be evicted to
    // make room for another token of the same step. GSA_HOT_BUFFER_SIZE is the
    // explicit deployment capacity knob.
    assert(count <= hot_size);

    const int* selected = idx_list + (size_t)bg * cap;
    int* selected_slots = slot_list + (size_t)bg * cap;
    int* t2s = token_to_slot + (size_t)bg * token_capacity;
    int* s2t = slot_token_ids + (size_t)bg * hot_size;
    int* order = lru_order + (size_t)bg * hot_size;
    int* scratch = lru_scratch + (size_t)bg * hot_size;
    uint8_t* marks = slot_marks + (size_t)bg * hot_size;
    int* misses = miss_tokens + (size_t)bg * hot_size;
    int* destinations = miss_slots + (size_t)bg * hot_size;

    for (int i = threadIdx.x; i < hot_size; i += BLOCK) marks[i] = 0;
    __syncthreads();

    // Mark every slot still needed this step so it cannot be chosen as a victim.
    for (int i = threadIdx.x; i < count; i += BLOCK) {
        const int token = selected[i];
        assert(token >= 0 && token < token_capacity);
        const int slot = t2s[token];
        assert(slot < hot_size);
        selected_slots[i] = slot;
        if (slot >= 0) marks[slot] = 1;
    }
    __syncthreads();

    // Compact the evictable slots out of `order`, oldest first: entry m is the
    // victim for the m-th miss. The order-preserving scan keeps LRU exact.
    if (threadIdx.x == 0) s_evictable = 0;
    __syncthreads();
    for (int tile = 0; tile < hot_size; tile += BLOCK) {
        const int i = tile + threadIdx.x;
        const int slot = (i < hot_size) ? order[i] : -1;
        assert(i >= hot_size || (slot >= 0 && slot < hot_size));
        const int keep = (i < hot_size && !marks[slot]) ? 1 : 0;
        int tile_total = 0;
        const int rank = block_scan_exclusive<BLOCK>(keep, s_warp, tile_total);
        if (keep) scratch[s_evictable + rank] = slot;
        __syncthreads();
        if (threadIdx.x == 0) s_evictable += tile_total;
        __syncthreads();
    }
    const int evictable_count = s_evictable;

    // Rank the misses in selection order; the m-th takes the m-th oldest slot.
    if (threadIdx.x == 0) s_misses = 0;
    __syncthreads();
    for (int tile = 0; tile < count; tile += BLOCK) {
        const int i = tile + threadIdx.x;
        const int is_miss = (i < count && selected_slots[i] < 0) ? 1 : 0;
        int tile_total = 0;
        const int rank = block_scan_exclusive<BLOCK>(is_miss, s_warp, tile_total);
        if (is_miss) {
            const int m = s_misses + rank;
            assert(m < evictable_count);
            const int token = selected[i];
            const int victim = scratch[m];
            const int old_token = s2t[victim];
            // A victim slot is unmarked, so its old token is not selected this
            // step: no other thread reads or writes these two t2s entries here.
            if (old_token >= 0) t2s[old_token] = -1;
            t2s[token] = victim;
            s2t[victim] = token;
            selected_slots[i] = victim;
            misses[m] = token;
            destinations[m] = victim;
        }
        __syncthreads();
        if (threadIdx.x == 0) s_misses += tile_total;
        __syncthreads();
    }
    const int miss_count = s_misses;

    // Stale entries stay oldest; every slot used this step becomes MRU.
    const int stale = evictable_count - miss_count;
    assert(stale + count == hot_size);
    for (int i = threadIdx.x; i < stale; i += BLOCK) order[i] = scratch[miss_count + i];
    for (int i = threadIdx.x; i < count; i += BLOCK) order[stale + i] = selected_slots[i];
    for (int m = threadIdx.x; m < miss_count; m += BLOCK) assert(t2s[misses[m]] == destinations[m]);

    if (threadIdx.x == 0) {
        miss_counts[bg] = miss_count;
        atomicAdd(reinterpret_cast<unsigned long long*>(stats),
                  static_cast<unsigned long long>(count));
        atomicAdd(reinterpret_cast<unsigned long long*>(stats + 1),
                  static_cast<unsigned long long>(miss_count));
    }
}

template <int VEC>
__global__ void gather_hot_misses_from_host_kernel(
    const __nv_bfloat16* __restrict__ host_K,
    const __nv_bfloat16* __restrict__ host_V,
    const int* __restrict__ miss_tokens,
    const int* __restrict__ miss_slots,
    const int* __restrict__ miss_counts,
    __nv_bfloat16* __restrict__ hot_K,
    __nv_bfloat16* __restrict__ hot_V,
    long host_bg_stride, long host_k_stride, int hot_size)
{
    const int bg = blockIdx.y;
    const int count = miss_counts[bg];
    constexpr int PER_ROW = D_HEAD / VEC;
    const long total = (long)count * PER_ROW;
    const long stride = (long)gridDim.x * blockDim.x;
    for (long e = (long)blockIdx.x * blockDim.x + threadIdx.x; e < total; e += stride) {
        const int i = (int)(e / PER_ROW);
        const int d = (int)(e - (long)i * PER_ROW) * VEC;
        const int logical_k = miss_tokens[(size_t)bg * hot_size + i];
        const int hot_slot = miss_slots[(size_t)bg * hot_size + i];
        const size_t src = (size_t)bg * host_bg_stride + (size_t)logical_k * host_k_stride + d;
        const size_t dst = ((size_t)bg * hot_size + hot_slot) * D_HEAD + d;
        if (VEC == 8) {
            *reinterpret_cast<float4*>(hot_K + dst) =
                *reinterpret_cast<const float4*>(host_K + src);
            *reinterpret_cast<float4*>(hot_V + dst) =
                *reinterpret_cast<const float4*>(host_V + src);
        } else if (VEC == 2) {
            *reinterpret_cast<float*>(hot_K + dst) =
                *reinterpret_cast<const float*>(host_K + src);
            *reinterpret_cast<float*>(hot_V + dst) =
                *reinterpret_cast<const float*>(host_V + src);
        } else {
            hot_K[dst] = host_K[src];
            hot_V[dst] = host_V[src];
        }
    }
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
template <int HPG, int DPL /* D_HEAD/32 */, int BLOCK_N, bool STAGED, bool CACHED>
__global__ void sparse_decode_partial_compact_kernel(
    const __nv_bfloat16* __restrict__ q,    // [B, H, 1, D]
    const __nv_bfloat16* __restrict__ K,    // [B, num_kv, k_len, D]
    const __nv_bfloat16* __restrict__ V,    // [B, num_kv, k_len, D]
    const int* __restrict__ idx_list,       // [B, num_kv, cap]
    const int* __restrict__ kv_idx_list,    // cached path: selected -> hot slot
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
    const int* kg = kv_idx_list + (size_t)bg * cap;
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
            int kv_k = CACHED ? kg[t0 + j] : (STAGED ? (t0 + j) : k);
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

// ---- host-KV gather stream plumbing ----------------------------------------
// One dedicated gather stream per device, created once and reused. A stable
// stream (rather than round-robining getStreamFromPool per call) keeps the
// event chain between successive decode steps trivially ordered and keeps the
// gather off the 32 pool streams other code may be using.
at::cuda::CUDAStream gather_stream_for(c10::DeviceIndex dev) {
    static std::mutex mu;
    static std::unordered_map<int, at::cuda::CUDAStream> streams;
    std::lock_guard<std::mutex> lk(mu);
    auto it = streams.find((int)dev);
    if (it == streams.end())
        it = streams.emplace((int)dev, at::cuda::getStreamFromPool(/*isHighPriority=*/false, dev)).first;
    return it->second;
}

// Default for the `async_gather=-1` (unspecified) operator argument. Read once;
// pass 0/1 explicitly to switch paths within a single process (the benchmark
// compares both, so it cannot rely on the environment).
bool async_gather_env_default() {
    static const bool v = [] {
        const char* s = std::getenv("GSA_ASYNC_GATHER");
        return s != nullptr && std::strcmp(s, "1") == 0;
    }();
    return v;
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
    double scaling, int64_t nsplit, int64_t cap, bool host_offload, bool async_gather,
    torch::Tensor hot_K, torch::Tensor hot_V, torch::Tensor token_to_slot,
    torch::Tensor slot_token_ids, torch::Tensor lru_order,
    torch::Tensor lru_scratch, torch::Tensor slot_marks, torch::Tensor hot_stats)
{
    int B = q.size(0), H = q.size(1), num_kv = K.size(1), k_len = K.size(2), D = q.size(3);
    TORCH_CHECK(D == D_HEAD, "kernel compiled for D_HEAD=", D_HEAD, " got ", D);
    // Explicit invariants — this kernel is a batched single-token (q_len=1), bf16
    // GQA flash-decode.
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    const bool cached = hot_K.defined();
    if (host_offload) {
        TORCH_CHECK(!K.is_cuda() && !V.is_cuda() && K.is_pinned() && V.is_pinned(),
                    "offloaded K/V must be pinned CPU tensors");
    } else {
        TORCH_CHECK(K.is_cuda() && V.is_cuda(), "resident K/V must be CUDA tensors");
    }
    TORCH_CHECK(!cached || host_offload, "hot buffer is only valid for host-offloaded K/V");
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
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    // A view of a longer buffer is allowed -- pinned host ring on the offload
    // path, preallocated GPU ring on the resident one -- so long as the layout is
    // still regular: the kernels are given the KV-plane stride they need. Forcing
    // .contiguous() instead would copy the entire context every decode step,
    // which is the very cost a preallocated ring exists to avoid.
    auto regular = [&](const torch::Tensor& t) {
        return t.stride(3) == 1 && t.stride(2) == D
            && t.stride(1) % D == 0
            && t.stride(0) == (int64_t)num_kv * t.stride(1);
    };
    TORCH_CHECK(regular(K) && regular(V) && K.strides() == V.strides(),
                host_offload ? "offloaded" : "resident",
                " K/V must be [B, num_kv, k_len, D] views with unit last-dim "
                "stride, D-strided keys and a uniform batch stride; got K.strides=",
                K.strides(), " V.strides=", V.strides());
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
    auto stream = at::cuda::getCurrentCUDAStream();   // compute stream

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
    torch::Tensor stage_K, stage_V, slot_list;
    const __nv_bfloat16* Kp;
    const __nv_bfloat16* Vp;
    int kv_stride = k_len;
    if (cached) {
        TORCH_CHECK(hot_V.defined() && token_to_slot.defined() && slot_token_ids.defined()
                    && lru_order.defined() && lru_scratch.defined() && slot_marks.defined()
                    && hot_stats.defined(),
                    "all hot-buffer state tensors must be provided");
        TORCH_CHECK(hot_K.is_cuda() && hot_V.is_cuda() && hot_K.is_contiguous()
                    && hot_V.is_contiguous() && hot_K.scalar_type() == at::kBFloat16
                    && hot_V.scalar_type() == at::kBFloat16 && hot_K.sizes() == hot_V.sizes()
                    && hot_K.dim() == 4 && hot_K.size(0) == B && hot_K.size(1) == num_kv
                    && hot_K.size(3) == D,
                    "hot K/V must be contiguous CUDA bf16 [B, num_kv, hot_size, D]");
        const int hot_size = hot_K.size(2);
        TORCH_CHECK(hot_size > 0, "hot_size must be positive");
        auto valid_i32 = [&](const torch::Tensor& t) {
            return t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kInt;
        };
        TORCH_CHECK(valid_i32(token_to_slot) && token_to_slot.dim() == 3
                    && token_to_slot.size(0) == B && token_to_slot.size(1) == num_kv
                    && token_to_slot.size(2) >= k_len,
                    "token_to_slot must be contiguous CUDA int32 [B, num_kv, >= k_len]");
        auto valid_hot_i32 = [&](const torch::Tensor& t) {
            return valid_i32(t) && t.dim() == 3 && t.size(0) == B
                && t.size(1) == num_kv && t.size(2) == hot_size;
        };
        TORCH_CHECK(valid_hot_i32(slot_token_ids) && valid_hot_i32(lru_order)
                    && valid_hot_i32(lru_scratch),
                    "slot ids and LRU tensors must be contiguous CUDA int32 "
                    "[B, num_kv, hot_size]");
        TORCH_CHECK(slot_marks.is_cuda() && slot_marks.is_contiguous()
                    && slot_marks.scalar_type() == at::kByte && slot_marks.dim() == 3
                    && slot_marks.size(0) == B && slot_marks.size(1) == num_kv
                    && slot_marks.size(2) == hot_size,
                    "slot_marks must be contiguous CUDA uint8 [B, num_kv, hot_size]");
        TORCH_CHECK(hot_stats.is_cuda() && hot_stats.is_contiguous()
                    && hot_stats.scalar_type() == at::kLong && hot_stats.numel() == 2,
                    "hot_stats must be contiguous CUDA int64 [2]");

        slot_list = torch::empty_like(idx_list);
        auto miss_tokens = torch::empty({B, num_kv, hot_size}, iopt);
        auto miss_slots = torch::empty_like(miss_tokens);
        auto miss_counts = torch::empty({B, num_kv}, iopt);
        resolve_hot_slots_lru_kernel<HOT_RESOLVE_BLOCK><<<B * num_kv, HOT_RESOLVE_BLOCK, 0, stream>>>(
            idx_list.data_ptr<int>(), counts.data_ptr<int>(), slot_list.data_ptr<int>(),
            token_to_slot.data_ptr<int>(), slot_token_ids.data_ptr<int>(),
            lru_order.data_ptr<int>(), lru_scratch.data_ptr<int>(),
            slot_marks.data_ptr<uint8_t>(), miss_tokens.data_ptr<int>(),
            miss_slots.data_ptr<int>(), miss_counts.data_ptr<int>(),
            hot_stats.data_ptr<int64_t>(),
            (int)cap, (int)token_to_slot.size(2), hot_size);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        auto host_Kp = reinterpret_cast<const __nv_bfloat16*>(K.data_ptr<at::BFloat16>());
        auto host_Vp = reinterpret_cast<const __nv_bfloat16*>(V.data_ptr<at::BFloat16>());
        auto hot_Kp = reinterpret_cast<__nv_bfloat16*>(hot_K.data_ptr<at::BFloat16>());
        auto hot_Vp = reinterpret_cast<__nv_bfloat16*>(hot_V.data_ptr<at::BFloat16>());
        const long host_bg_stride = (long)K.stride(1);
        const long host_k_stride = (long)K.stride(2);
        auto aligned = [&](int v) {
            const size_t bytes = (size_t)v * sizeof(__nv_bfloat16);
            return host_bg_stride % v == 0 && host_k_stride % v == 0
                && (reinterpret_cast<uintptr_t>(host_Kp) % bytes) == 0
                && (reinterpret_cast<uintptr_t>(host_Vp) % bytes) == 0
                && (reinterpret_cast<uintptr_t>(hot_Kp) % bytes) == 0
                && (reinterpret_cast<uintptr_t>(hot_Vp) % bytes) == 0;
        };
        const int vec = aligned(8) ? 8 : (aligned(2) ? 2 : 1);
        long want_blocks = ((long)hot_size * (D / vec) + 255) / 256;
        if (want_blocks > 4096) want_blocks = 4096;
        if (want_blocks < 1) want_blocks = 1;
        dim3 gather_grid((int)want_blocks, B * num_kv);
#define LAUNCH_HOT_GATHER(VEC) \
        gather_hot_misses_from_host_kernel<VEC><<<gather_grid, 256, 0, stream>>>( \
            host_Kp, host_Vp, miss_tokens.data_ptr<int>(), miss_slots.data_ptr<int>(), \
            miss_counts.data_ptr<int>(), hot_Kp, hot_Vp, host_bg_stride, host_k_stride, hot_size)
        if (vec == 8) { LAUNCH_HOT_GATHER(8); }
        else if (vec == 2) { LAUNCH_HOT_GATHER(2); }
        else { LAUNCH_HOT_GATHER(1); }
#undef LAUNCH_HOT_GATHER
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        Kp = hot_Kp;
        Vp = hot_Vp;
        kv_stride = hot_size;
    } else if (host_offload) {
        stage_K = torch::empty({B, num_kv, (long)cap, D}, q.options());
        stage_V = torch::empty_like(stage_K);
        auto host_Kp = reinterpret_cast<const __nv_bfloat16*>(K.data_ptr<at::BFloat16>());
        auto host_Vp = reinterpret_cast<const __nv_bfloat16*>(V.data_ptr<at::BFloat16>());
        auto stage_Kp = reinterpret_cast<__nv_bfloat16*>(stage_K.data_ptr<at::BFloat16>());
        auto stage_Vp = reinterpret_cast<__nv_bfloat16*>(stage_V.data_ptr<at::BFloat16>());
        const long host_bg_stride = (long)K.stride(1);
        const long host_k_stride = (long)K.stride(2);
        // Widest vector whose every access stays 16/4-byte aligned. Both strides
        // are element counts, so a stride not divisible by VEC would misalign
        // rows even when the base pointer is fine.
        auto aligned = [&](int v) {
            const size_t bytes = (size_t)v * sizeof(__nv_bfloat16);
            return host_bg_stride % v == 0 && host_k_stride % v == 0
                && (reinterpret_cast<uintptr_t>(host_Kp) % bytes) == 0
                && (reinterpret_cast<uintptr_t>(host_Vp) % bytes) == 0
                && (reinterpret_cast<uintptr_t>(stage_Kp) % bytes) == 0
                && (reinterpret_cast<uintptr_t>(stage_Vp) % bytes) == 0;
        };
        const int vec = aligned(8) ? 8 : (aligned(2) ? 2 : 1);
        // Enough blocks to fill the device for a full-cap gather, but never more
        // than the work needs; the kernel grid-strides over the on-device count.
        long want_blocks = ((long)cap * (D / vec) + 255) / 256;
        if (want_blocks > 4096) want_blocks = 4096;
        if (want_blocks < 1) want_blocks = 1;
        int copy_blocks = (int)want_blocks;
        dim3 gather_grid(copy_blocks, B * num_kv);
        auto launch_gather = [&](cudaStream_t st) {
            if (vec == 8)
                gather_compact_kv_from_host_kernel<8><<<gather_grid, 256, 0, st>>>(
                    host_Kp, host_Vp, idx_list.data_ptr<int>(), counts.data_ptr<int>(),
                    stage_Kp, stage_Vp, host_bg_stride, host_k_stride, (int)cap);
            else if (vec == 2)
                gather_compact_kv_from_host_kernel<2><<<gather_grid, 256, 0, st>>>(
                    host_Kp, host_Vp, idx_list.data_ptr<int>(), counts.data_ptr<int>(),
                    stage_Kp, stage_Vp, host_bg_stride, host_k_stride, (int)cap);
            else
                gather_compact_kv_from_host_kernel<1><<<gather_grid, 256, 0, st>>>(
                    host_Kp, host_Vp, idx_list.data_ptr<int>(), counts.data_ptr<int>(),
                    stage_Kp, stage_Vp, host_bg_stride, host_k_stride, (int)cap);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        };
        if (async_gather) {
            // Separate-stream gather. Dependency chain:
            //   compute: build_compact_idx --[idx_ready]-. . .-[gather_done]--> attention
            //   copy:              .-[wait idx_ready]--> gather --[gather_done]
            // The attention kernel consumes the WHOLE staging buffer, so this does
            // NOT overlap useful work yet; it exists to establish and validate the
            // stream/event plumbing that tiled double-buffering will need.
            auto copy_stream = gather_stream_for(q.device().index());
            at::cuda::CUDAEvent idx_ready;
            idx_ready.record(stream);
            idx_ready.block(copy_stream);   // gather must see compacted idx/counts
            launch_gather(copy_stream);
            // These four blocks come from the caching allocator on `stream` but are
            // read/written on `copy_stream`. The gather_done->stream edge below
            // already orders every later `stream` op after the gather, which covers
            // same-stream block reuse; record_stream states the cross-stream use to
            // the allocator directly, so the invariant does not rest on that
            // reasoning and survives loosening the edge for real overlap later.
            idx_list.record_stream(copy_stream);
            counts.record_stream(copy_stream);
            stage_K.record_stream(copy_stream);
            stage_V.record_stream(copy_stream);
            at::cuda::CUDAEvent gather_done;
            gather_done.record(copy_stream);
            gather_done.block(stream);      // attention must see the staging KV
        } else {
            launch_gather(stream);
        }
        Kp = stage_Kp;
        Vp = stage_Vp;
        kv_stride = cap;
    } else {
        Kp = reinterpret_cast<const __nv_bfloat16*>(K.data_ptr<at::BFloat16>());
        Vp = reinterpret_cast<const __nv_bfloat16*>(V.data_ptr<at::BFloat16>());
        // Rows of a resident ring buffer are k_len long only when the view is
        // contiguous; otherwise the plane stride is the ring's full width.
        kv_stride = (int)(K.stride(1) / D);
    }

#define LAUNCH_COMPACT(HPGV, STAGEDV, CACHEDV) \
    sparse_decode_partial_compact_kernel<HPGV, D_HEAD/32, 64, STAGEDV, CACHEDV><<<grid, block, 0, stream>>>( \
        qp, Kp, Vp, idx_list.data_ptr<int>(), \
        (cached ? slot_list.data_ptr<int>() : idx_list.data_ptr<int>()), \
        counts.data_ptr<int>(), causal_bias.data_ptr<float>(), \
        partial_m.data_ptr<float>(), partial_l.data_ptr<float>(), partial_acc.data_ptr<float>(), \
        B, H, num_kv, k_len, kv_stride, (int)cap, (int)nsplit, (float)scaling)
#define DISPATCH_HPG(STAGEDV, CACHEDV) \
    switch (hpg) { \
        case 1: LAUNCH_COMPACT(1, STAGEDV, CACHEDV); break;  case 2: LAUNCH_COMPACT(2, STAGEDV, CACHEDV); break; \
        case 4: LAUNCH_COMPACT(4, STAGEDV, CACHEDV); break;  case 7: LAUNCH_COMPACT(7, STAGEDV, CACHEDV); break; \
        case 8: LAUNCH_COMPACT(8, STAGEDV, CACHEDV); break;  case 16: LAUNCH_COMPACT(16, STAGEDV, CACHEDV); break; \
        default: TORCH_CHECK(false, "unsupported heads_per_group=", hpg); \
    }
    if (cached) { DISPATCH_HPG(false, true); }
    else if (host_offload) { DISPATCH_HPG(true, false); }
    else { DISPATCH_HPG(false, false); }
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
        scaling, nsplit, cap, /*host_offload=*/false, /*async_gather=*/false,
        {}, {}, {}, {}, {}, {}, {}, {});
}

torch::Tensor sparse_decode_attn_offload(
    torch::Tensor q, torch::Tensor host_K, torch::Tensor host_V,
    torch::Tensor sel0_any, torch::Tensor sel1_any,
    torch::Tensor chunk_l0, torch::Tensor chunk_l1,
    torch::Tensor is_gist, torch::Tensor is_l0g, torch::Tensor is_l1g,
    torch::Tensor compressed, torch::Tensor am_bool, torch::Tensor causal_bias,
    double scaling, int64_t nsplit, int64_t cap, int64_t async_gather)
{
    // async_gather: 0 = gather on the compute stream (baseline), 1 = gather on a
    // dedicated stream ordered by CUDA events, -1 = take GSA_ASYNC_GATHER (default 0).
    TORCH_CHECK(async_gather >= -1 && async_gather <= 1,
                "async_gather must be -1 (env default), 0 (sync) or 1 (separate stream); got ",
                async_gather);
    bool use_async = (async_gather < 0) ? async_gather_env_default() : (async_gather == 1);
    return sparse_decode_attn_compact_impl(
        q, host_K, host_V, sel0_any, sel1_any, chunk_l0, chunk_l1,
        is_gist, is_l0g, is_l1g, compressed, am_bool, causal_bias,
        scaling, nsplit, cap, /*host_offload=*/true, use_async,
        {}, {}, {}, {}, {}, {}, {}, {});
}

torch::Tensor sparse_decode_attn_offload_cached(
    torch::Tensor q, torch::Tensor host_K, torch::Tensor host_V,
    torch::Tensor hot_K, torch::Tensor hot_V, torch::Tensor token_to_slot,
    torch::Tensor slot_token_ids, torch::Tensor lru_order,
    torch::Tensor lru_scratch, torch::Tensor slot_marks, torch::Tensor hot_stats,
    torch::Tensor sel0_any, torch::Tensor sel1_any,
    torch::Tensor chunk_l0, torch::Tensor chunk_l1,
    torch::Tensor is_gist, torch::Tensor is_l0g, torch::Tensor is_l1g,
    torch::Tensor compressed, torch::Tensor am_bool, torch::Tensor causal_bias,
    double scaling, int64_t nsplit, int64_t cap)
{
    return sparse_decode_attn_compact_impl(
        q, host_K, host_V, sel0_any, sel1_any, chunk_l0, chunk_l1,
        is_gist, is_l0g, is_l1g, compressed, am_bool, causal_bias,
        scaling, nsplit, cap, /*host_offload=*/true, /*async_gather=*/false,
        hot_K, hot_V, token_to_slot, slot_token_ids,
        lru_order, lru_scratch, slot_marks, hot_stats);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_decode_attn_compact", &sparse_decode_attn_compact, "compact per-group shared-mem tiled sparse flash-decode (CUDA)");
    m.def("sparse_decode_attn_offload", &sparse_decode_attn_offload,
          "pinned-host KV to compact GPU staging sparse flash-decode (CUDA)",
          py::arg("q"), py::arg("host_K"), py::arg("host_V"),
          py::arg("sel0_any"), py::arg("sel1_any"),
          py::arg("chunk_l0"), py::arg("chunk_l1"),
          py::arg("is_gist"), py::arg("is_l0g"), py::arg("is_l1g"),
          py::arg("compressed"), py::arg("am_bool"), py::arg("causal_bias"),
          py::arg("scaling"), py::arg("nsplit"), py::arg("cap"),
          py::arg("async_gather") = -1);
    m.def("sparse_decode_attn_offload_cached", &sparse_decode_attn_offload_cached,
          "pinned-host KV with persistent GPU LRU hot buffer (CUDA)",
          py::arg("q"), py::arg("host_K"), py::arg("host_V"),
          py::arg("hot_K"), py::arg("hot_V"), py::arg("token_to_slot"),
          py::arg("slot_token_ids"), py::arg("lru_order"),
          py::arg("lru_scratch"), py::arg("slot_marks"), py::arg("hot_stats"),
          py::arg("sel0_any"), py::arg("sel1_any"),
          py::arg("chunk_l0"), py::arg("chunk_l1"),
          py::arg("is_gist"), py::arg("is_l0g"), py::arg("is_l1g"),
          py::arg("compressed"), py::arg("am_bool"), py::arg("causal_bias"),
          py::arg("scaling"), py::arg("nsplit"), py::arg("cap"));
}
