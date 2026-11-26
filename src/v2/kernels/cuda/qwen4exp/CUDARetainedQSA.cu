/**
 * @file CUDARetainedQSA.cu
 * @brief Spill-free cooperative CUDA QSA Pre-Attention and Post-Attention megakernels.
 *
 * Pre-Attention fuses HC prefix + replay/row_params publication + Wq (with direct
 * Q/fa_gate split), Wk, Wv, qsa_key (+ optional ranking-guarded qsa_query) +
 * Q/K RMSNorm + 64-dim partial RoPE + FP16 KV ring append + metadata advance into
 * one cooperative kernel running at 4 CTAs/SM (Q8_0 fast path) or 2 CTAs/SM
 * (all-18-codebook general path) with zero register spills.
 *
 * Post-Attention fuses foldAttentionPrefix (Parts=33 or 8) + commitPartial +
 * sigmoid fa_gate + blockwise Q8 staging + Wo projection into one cooperative
 * kernel running directly after selectedAttentionFP16Tiles.
 */
#include "kernels/cuda/qwen4exp/CUDARetainedQSA.h"

#include "backends/cuda/CUDAHardwareMetadata.h"
#include "kernels/common/NativeCooperativeKernel.cuh"
#include "kernels/common/NativeCounter64.h"
#include "kernels/cuda/moe/CUDARetainedHCPrefix.cuh"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace llaminar2::cuda::qwen4exp
{
namespace
{
    constexpr unsigned kHidden = 2560u;
    constexpr unsigned kLowRank = 320u;
    constexpr unsigned kThreads = 256u;
    constexpr unsigned kKPartChunk = 8u;
    constexpr unsigned kMaxOutElems = kMaxCUDARetainedQSAQHeads * 256u; // 6144
    constexpr unsigned kMaxOutBlocks = kMaxCUDARetainedQSAQHeads * 8u;  // 192

    constexpr std::size_t kScratchCodesOffset = 0u;
    constexpr std::size_t kScratchScalesOffset = (kMaxOutElems + 15u) & ~std::size_t(15u);
    constexpr std::size_t kScratchHcDownDoneOffset =
        ((kScratchScalesOffset + kMaxOutBlocks * sizeof(float)) + 15u) & ~std::size_t(15u);
    constexpr std::size_t kScratchTotalBytes =
        ((kScratchHcDownDoneOffset + 32u * sizeof(std::uint32_t)) + 15u) & ~std::size_t(15u);

    /** @brief Phase-exclusive SMEM for QSA Pre-Attention (<= 12,288 B -> 4 CTAs/SM). */
    union CUDARetainedQSAPreShared
    {
        moe::RetainedHCPrefixShared<kHidden> hc;
        struct
        {
            alignas(16) int8_t act_codes[kHidden];          // 2560 B
            alignas(16) float act_scales[kHidden / 32u];    // 320 B
            alignas(16) float partials[kKPartChunk][32];    // 1024 B (256 floats -> also floating-row reduce)
        } proj;                                             // 3904 B
        struct
        {
            float warp_sums[32];                             // 128 B
            float s_cos[128];                                // 512 B
            float s_sin[128];                                // 512 B
            float vals[256];                                 // 1024 B
        } post;                                              // 2176 B
    };
    static_assert(sizeof(CUDARetainedQSAPreShared) <= 12288u,
        "CUDARetainedQSAPreShared must fit 4 CTAs/SM inside default 48 KB SMEM");

    /** @brief Phase-exclusive SMEM for QSA Post-Attention (<= 12,288 B -> 4 CTAs/SM). */
    union CUDARetainedQSAPostShared
    {
        struct
        {
            float scales[33];
            float denominator;
            float cached_maximum[33];
            float cached_denominator[33];
            std::uint32_t cached_keys[33];
            std::uint32_t cached_values[33];
        } fold;
        struct
        {
            alignas(16) int8_t act_codes[kMaxOutElems];     // 6144 B
            alignas(16) float act_scales[kMaxOutBlocks];    // 768 B
            alignas(16) float partials[kKPartChunk][16];    // 512 B
        } wo;                                               // 7424 B
    };
    static_assert(sizeof(CUDARetainedQSAPostShared) <= 12288u,
        "CUDARetainedQSAPostShared must fit 4 CTAs/SM inside default 48 KB SMEM");

    /** @brief Exact CUDA blockwise Q8 quantization matching quantize_activations_blockwise_kernel. */
    __device__ __forceinline__ void qsaQuantizeBlockwiseWarp32(
        float val, int8_t *dst_codes, float *dst_scale, unsigned lane)
    {
        float abs_val = fabsf(val);
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            abs_val = fmaxf(abs_val, __shfl_xor_sync(0xffffffffu, abs_val, mask));
        const float scale = (abs_val > 0.0f) ? (abs_val / 127.0f) : 1.0f;
        const float inv_scale = 1.0f / scale;
        if (lane == 0u)
            *dst_scale = scale;
        const float qval = val * inv_scale;
        const int32_t q = static_cast<int32_t>(rintf(fminf(127.0f, fmaxf(-127.0f, qval))));
        dst_codes[lane] = static_cast<int8_t>(q);
    }

    /** @brief Coalesced 256B/warp 8-column Q8_0 sub-tile with 2-block software pipelining. */
    __device__ __forceinline__ float qsaAccumulateQ8SubTile8(
        const DeviceNativeQuantisedMatrixDesc &w,
        const uint16_t *w_scales,
        const int8_t *act_codes,
        const float *act_scales,
        unsigned col_sub,
        unsigned blk_begin,
        unsigned blk_end,
        unsigned lane)
    {
        const unsigned n = static_cast<unsigned>(w.n);
        const unsigned col_in_sub = lane >> 2;
        const unsigned q_part = lane & 3u;
        const unsigned col = col_sub + col_in_sub;
        const bool valid_col = col < n;
        if (blk_begin >= blk_end)
            return 0.0f;

        unsigned elem = blk_begin * n + col;
        int2 w_cur = valid_col
            ? reinterpret_cast<const int2 *>(w.payload + static_cast<std::size_t>(elem) * 32u)[q_part]
            : make_int2(0, 0);
        uint16_t sc_cur = (q_part == 0u && valid_col) ? w_scales[elem] : 0u;

        float sum = 0.0f;
#pragma unroll 1
        for (unsigned b = blk_begin; b < blk_end; ++b)
        {
            int2 w_next = make_int2(0, 0);
            uint16_t sc_next = 0u;
            if (b + 1u < blk_end)
            {
                const unsigned next_elem = elem + n;
                if (valid_col)
                {
                    w_next = reinterpret_cast<const int2 *>(
                        w.payload + static_cast<std::size_t>(next_elem) * 32u)[q_part];
                    if (q_part == 0u)
                        sc_next = w_scales[next_elem];
                }
            }
            const int2 a2 = reinterpret_cast<const int2 *>(act_codes + b * 32u)[q_part];
            int d = __dp4a(w_cur.x, a2.x, 0);
            d = __dp4a(w_cur.y, a2.y, d);
            d += __shfl_xor_sync(0xffffffffu, d, 1);
            d += __shfl_xor_sync(0xffffffffu, d, 2);
            if (q_part == 0u && valid_col)
            {
                const float ws = __half2float(__ushort_as_half(sc_cur));
                sum = __fadd_rn(sum,
                    device_native_quantised_contract::singleScaleBlock(d, ws, act_scales[b]));
            }
            w_cur = w_next;
            sc_cur = sc_next;
            elem += n;
        }
        return sum;
    }

    /** @brief Complete 32-column Q8_0 projection tile, optionally deinterleaving Wq directly into Q + fa_gate. */
    template <bool GeneralCodebooks>
    __device__ __forceinline__ void qsaCompleteTile32(
        const CUDARetainedQSAProjection &proj,
        unsigned col_base,
        const int8_t *act_codes,
        const float *act_scales,
        float (*partials)[32],
        bool split_q_gate,
        float *dst_q,
        float *dst_gate)
    {
        const unsigned tid = threadIdx.x;
        const unsigned warp = tid >> 5;
        const unsigned lane = tid & 31u;
        const auto &w = proj.weights;
        const unsigned n = static_cast<unsigned>(w.n);
        const unsigned blocks = static_cast<unsigned>(w.blocks_per_row);
        const unsigned kb = static_cast<unsigned>(proj.k_partitions);
        const unsigned bps = static_cast<unsigned>(proj.blocks_per_partition);

        float total = 0.0f;
#pragma unroll 1
        for (unsigned p_base = 0u; p_base < kb; p_base += kKPartChunk)
        {
            const unsigned chunk = min(kKPartChunk, kb - p_base);
            if constexpr (!GeneralCodebooks)
            {
                const unsigned col_in_sub = lane >> 2;
                const unsigned q_part = lane & 3u;
                const auto *w_scales = static_cast<const uint16_t *>(w.scales);
                const unsigned total_warp_tasks = 4u * chunk;
                for (unsigned task = warp; task < total_warp_tasks; task += 8u)
                {
                    const unsigned dp = task >> 2;
                    const unsigned p = p_base + dp;
                    const unsigned st = task & 3u;
                    const unsigned blk_begin = p * bps;
                    const unsigned blk_end = min(blocks, blk_begin + bps);
                    const float sum = qsaAccumulateQ8SubTile8(
                        w, w_scales, act_codes, act_scales, col_base + st * 8u, blk_begin, blk_end, lane);
                    if (q_part == 0u)
                        partials[dp][st * 8u + col_in_sub] = sum;
                }
            }
            else
            {
                const unsigned col = col_base + lane;
                const bool valid_col = col < n;
                if (warp < chunk)
                {
                    const unsigned p = p_base + warp;
                    const unsigned blk_begin = p * bps;
                    const unsigned blk_end = min(blocks, blk_begin + bps);
                    float sum = 0.0f;
                    if (valid_col)
                    {
#pragma unroll 1
                        for (unsigned b = blk_begin; b < blk_end; ++b)
                        {
                            const unsigned linear = b * n + col;
                            sum = __fadd_rn(sum, moe::retainedNativeQuantisedBlockDynamic(
                                w, linear, act_codes + b * 32u, act_scales[b]));
                        }
                    }
                    partials[warp][lane] = sum;
                }
            }
            __syncthreads();

            if (warp == 0u && (col_base + lane) < n)
            {
                for (unsigned dp = 0u; dp < chunk; ++dp)
                    total = __fadd_rn(total, partials[dp][lane]);
            }
            __syncthreads();
        }

        if (warp == 0u)
        {
            const unsigned col = col_base + lane;
            if (col < n)
            {
                if (split_q_gate)
                {
                    const unsigned head = col >> 9;
                    const unsigned within = col & 511u;
                    if (within < 256u)
                        dst_q[head * 256u + within] = total;
                    else
                        dst_gate[head * 256u + (within - 256u)] = total;
                }
                else
                {
                    proj.output[col] = total;
                }
            }
        }
    }

    /** @brief Complete 16-column Wo projection tile (Q8_0 fast path or all-18-codebook general path). */
    template <bool GeneralCodebooks>
    __device__ __forceinline__ void qsaCompleteTile16(
        const CUDARetainedQSAProjection &proj,
        unsigned col_base,
        const int8_t *act_codes,
        const float *act_scales,
        float (*partials)[16])
    {
        const unsigned tid = threadIdx.x;
        const unsigned warp = tid >> 5;
        const unsigned lane = tid & 31u;
        const auto &w = proj.weights;
        const unsigned n = static_cast<unsigned>(w.n);
        const unsigned blocks = static_cast<unsigned>(w.blocks_per_row);
        const unsigned kb = static_cast<unsigned>(proj.k_partitions);
        const unsigned bps = static_cast<unsigned>(proj.blocks_per_partition);

        float total = 0.0f;
#pragma unroll 1
        for (unsigned p_base = 0u; p_base < kb; p_base += kKPartChunk)
        {
            const unsigned chunk = min(kKPartChunk, kb - p_base);
            if constexpr (!GeneralCodebooks)
            {
                const unsigned col_in_sub = lane >> 2;
                const unsigned q_part = lane & 3u;
                const auto *w_scales = static_cast<const uint16_t *>(w.scales);
                const unsigned total_warp_tasks = 2u * chunk;
                for (unsigned task = warp; task < total_warp_tasks; task += 8u)
                {
                    const unsigned dp = task >> 1;
                    const unsigned p = p_base + dp;
                    const unsigned st = task & 1u;
                    const unsigned blk_begin = p * bps;
                    const unsigned blk_end = min(blocks, blk_begin + bps);
                    const float sum = qsaAccumulateQ8SubTile8(
                        w, w_scales, act_codes, act_scales, col_base + st * 8u, blk_begin, blk_end, lane);
                    if (q_part == 0u)
                        partials[dp][st * 8u + col_in_sub] = sum;
                }
            }
            else
            {
                const unsigned col = col_base + lane;
                const bool valid_col = (lane < 16u) && (col < n);
                if (warp < chunk)
                {
                    const unsigned p = p_base + warp;
                    const unsigned blk_begin = p * bps;
                    const unsigned blk_end = min(blocks, blk_begin + bps);
                    float sum = 0.0f;
                    if (valid_col)
                    {
#pragma unroll 1
                        for (unsigned b = blk_begin; b < blk_end; ++b)
                        {
                            const unsigned linear = b * n + col;
                            sum = __fadd_rn(sum, moe::retainedNativeQuantisedBlockDynamic(
                                w, linear, act_codes + b * 32u, act_scales[b]));
                        }
                    }
                    if (lane < 16u)
                        partials[warp][lane] = sum;
                }
            }
            __syncthreads();

            if (tid < 16u && (col_base + tid) < n)
            {
                for (unsigned dp = 0u; dp < chunk; ++dp)
                    total = __fadd_rn(total, partials[dp][tid]);
            }
            __syncthreads();
        }

        if (tid < 16u && (col_base + tid) < n)
            proj.output[col_base + tid] = total;
    }

    /** @brief Exact stride-256 floating projection row (BF16/FP16/FP32) matching floating_tiny_block_projection_kernel. */
    __device__ __forceinline__ void qsaCompleteFloatingRow(
        const float *mixed,
        const void *weights,
        CUDARetainedQSAIndexerFormat format,
        unsigned row,
        float *output,
        float *partials_flat)
    {
        const unsigned tid = threadIdx.x;
        float sum = 0.0f;
        const std::size_t row_base = static_cast<std::size_t>(row) * kHidden;
        if (format == CUDARetainedQSAIndexerFormat::BF16)
        {
            const auto *w = static_cast<const std::uint16_t *>(weights) + row_base;
            for (unsigned j = tid; j < kHidden; j += kThreads)
            {
                const float wv = __uint_as_float(static_cast<std::uint32_t>(w[j]) << 16);
                sum += mixed[j] * wv;
            }
        }
        else if (format == CUDARetainedQSAIndexerFormat::FP16)
        {
            const auto *w = static_cast<const std::uint16_t *>(weights) + row_base;
            for (unsigned j = tid; j < kHidden; j += kThreads)
            {
                const float wv = __half2float(__ushort_as_half(w[j]));
                sum += mixed[j] * wv;
            }
        }
        else
        {
            const auto *w = static_cast<const float *>(weights) + row_base;
            for (unsigned j = tid; j < kHidden; j += kThreads)
                sum += mixed[j] * w[j];
        }
        partials_flat[tid] = sum;
        __syncthreads();
        if (tid < 32u)
        {
            const float low = (partials_flat[tid] + partials_flat[tid + 128u]) +
                              (partials_flat[tid + 64u] + partials_flat[tid + 192u]);
            const float high = (partials_flat[tid + 32u] + partials_flat[tid + 160u]) +
                               (partials_flat[tid + 96u] + partials_flat[tid + 224u]);
            float result = low + high;
#pragma unroll
            for (int stride = 16; stride > 0; stride >>= 1)
                result += __shfl_down_sync(0xffffffffu, result, stride);
            if (tid == 0u)
                output[row] = result;
        }
        __syncthreads();
    }

    /** @brief Exact 256-dim per-head RMSNorm matching rmsnorm_fp32_kernel. */
    __device__ __forceinline__ float qsaHeadRMSNorm256(
        float val, const float *gamma, float eps, float *warp_sums)
    {
        const unsigned tid = threadIdx.x;
        const unsigned warp_id = tid >> 5;
        const unsigned lane_id = tid & 31u;
        float sum_sq = val * val;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, offset);
        if (lane_id == 0u)
            warp_sums[warp_id] = sum_sq;
        __syncthreads();
        if (tid < 32u)
        {
            float v = (tid < 8u) ? warp_sums[tid] : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                v += __shfl_xor_sync(0xffffffffu, v, offset);
            if (tid == 0u)
                warp_sums[0] = v;
        }
        __syncthreads();
        const float inv_rms = rsqrtf(warp_sums[0] / 256.0f + eps);
        return val * inv_rms * gamma[tid];
    }

    /** @brief Resident cooperative CUDA QSA Pre-Attention megakernel. */
    template <bool GeneralCodebooks>
    __global__ __launch_bounds__(kThreads, GeneralCodebooks ? 2 : 3)
    void cudaRetainedQSAPreAttentionKernel(const __grid_constant__ CUDARetainedQSAPreInvocation inv)
    {
        __shared__ CUDARetainedQSAPreShared shared;
        const auto grid = cooperative_groups::this_grid();
        const unsigned tid = threadIdx.x;
        const unsigned bid = blockIdx.x;
        const unsigned num_ctas = gridDim.x;

        // Publish pre-append replay & row_parameters[0] before any KV advance.
        // Zero race: publishRetainedHCPrefix concludes with a full grid.sync().
        if (bid == 0u && tid == 0u)
        {
            const int prefix = *inv.d_cached_tokens;
            const int cursor = *inv.d_ring_head;
            const int cap = inv.kv_token_capacity;
            const int pos = inv.position_ids_device[0];
            if (cursor < 0 || cursor >= cap || prefix < 0 || prefix >= cap || pos < prefix)
                asm volatile("trap;");
            const int origin = (cursor >= prefix) ? (cursor - prefix) : (cursor - prefix + cap);
            *inv.replay_out = {prefix, 1, pos - prefix};
            inv.row_params_out[0] = {prefix + 1, cap, pos, 0, origin, cap};
        }

        auto *hc_down_done = reinterpret_cast<std::uint32_t *>(
            static_cast<unsigned char *>(inv.scratch) + kScratchHcDownDoneOffset);
        if (bid == 0u && tid < 32u)
            hc_down_done[tid] = 0u;
        moe::publishRetainedHCPrefix<false, false, kHidden, true, GeneralCodebooks>(
            inv.hc, shared.hc, grid, nullptr, hc_down_done);

        // =====================================================================
        // Phase 1: Shared blockwise Q8 staging + Wq(split)/Wk/Wv + qsa_key/query
        // =====================================================================
        const float *mixed = inv.hc.mixed;
        {
            const unsigned warp = tid >> 5;
            const unsigned lane = tid & 31u;
#pragma unroll 1
            for (unsigned b = warp; b < kHidden / 32u; b += 8u)
            {
                const float val = mixed[b * 32u + lane];
                qsaQuantizeBlockwiseWarp32(
                    val, shared.proj.act_codes + b * 32u, shared.proj.act_scales + b, lane);
            }
        }
        __syncthreads();

        const int prefix_tokens = inv.replay_out->prefix_tokens;
        const bool run_qsa_query =
            (inv.qsa_query.format != CUDARetainedQSAIndexerFormat::Absent) &&
            (!inv.qsa_query.ranking_only_guard ||
             prefix_tokens >= llaminar2::qwen4exp::QwenSparseFirstRankingLogicalRow);

        const unsigned wq_tiles = (static_cast<unsigned>(inv.wq.weights.n) + 31u) >> 5;
        const unsigned wk_tiles = (static_cast<unsigned>(inv.wk.weights.n) + 31u) >> 5;
        const unsigned wv_tiles = (static_cast<unsigned>(inv.wv.weights.n) + 31u) >> 5;
        const unsigned key_q_tiles = (inv.qsa_key.format == CUDARetainedQSAIndexerFormat::NativeQuantised)
            ? ((static_cast<unsigned>(inv.qsa_key.n) + 31u) >> 5) : 0u;
        const unsigned qry_q_tiles = (run_qsa_query &&
            inv.qsa_query.format == CUDARetainedQSAIndexerFormat::NativeQuantised)
            ? ((static_cast<unsigned>(inv.qsa_query.n) + 31u) >> 5) : 0u;
        const unsigned total_q_tiles = wq_tiles + wk_tiles + wv_tiles + key_q_tiles + qry_q_tiles;

#pragma unroll 1
        for (unsigned tile = bid; tile < total_q_tiles; tile += num_ctas)
        {
            const CUDARetainedQSAProjection *proj = nullptr;
            unsigned col_base = 0u;
            bool split_q = false;
            if (tile < wq_tiles)
            {
                proj = &inv.wq;
                col_base = tile << 5;
                split_q = true;
            }
            else if (tile < wq_tiles + wk_tiles)
            {
                proj = &inv.wk;
                col_base = (tile - wq_tiles) << 5;
            }
            else if (tile < wq_tiles + wk_tiles + wv_tiles)
            {
                proj = &inv.wv;
                col_base = (tile - wq_tiles - wk_tiles) << 5;
            }
            else if (tile < wq_tiles + wk_tiles + wv_tiles + key_q_tiles)
            {
                proj = &inv.qsa_key.quantised;
                col_base = (tile - wq_tiles - wk_tiles - wv_tiles) << 5;
            }
            else
            {
                proj = &inv.qsa_query.quantised;
                col_base = (tile - wq_tiles - wk_tiles - wv_tiles - key_q_tiles) << 5;
            }
            qsaCompleteTile32<GeneralCodebooks>(
                *proj, col_base, shared.proj.act_codes, shared.proj.act_scales,
                shared.proj.partials, split_q, inv.Q, inv.fa_gate);
        }

        // Schedule floating BF16/FP16/FP32 indexer rows onto tail CTAs first so they
        // run in the tail shadow of the quantized tiles.
        const unsigned key_f_rows = (inv.qsa_key.format == CUDARetainedQSAIndexerFormat::FP32 ||
                                     inv.qsa_key.format == CUDARetainedQSAIndexerFormat::FP16 ||
                                     inv.qsa_key.format == CUDARetainedQSAIndexerFormat::BF16)
            ? static_cast<unsigned>(inv.qsa_key.n) : 0u;
        const unsigned qry_f_rows = (run_qsa_query &&
                                     (inv.qsa_query.format == CUDARetainedQSAIndexerFormat::FP32 ||
                                      inv.qsa_query.format == CUDARetainedQSAIndexerFormat::FP16 ||
                                      inv.qsa_query.format == CUDARetainedQSAIndexerFormat::BF16))
            ? static_cast<unsigned>(inv.qsa_query.n) : 0u;
        const unsigned total_f_rows = key_f_rows + qry_f_rows;
        if (total_f_rows > 0u)
        {
            const unsigned shift = total_q_tiles % num_ctas;
            const unsigned shifted_bid = (bid + num_ctas - shift) % num_ctas;
            float *partials_flat = &shared.proj.partials[0][0];
#pragma unroll 1
            for (unsigned r = shifted_bid; r < total_f_rows; r += num_ctas)
            {
                const bool is_key = r < key_f_rows;
                const void *fw = is_key ? inv.qsa_key.floating_weights : inv.qsa_query.floating_weights;
                const auto fmt = is_key ? inv.qsa_key.format : inv.qsa_query.format;
                const unsigned row_idx = is_key ? r : (r - key_f_rows);
                float *fout = is_key ? inv.qsa_key.floating_output : inv.qsa_query.floating_output;
                qsaCompleteFloatingRow(mixed, fw, fmt, row_idx, fout, partials_flat);
            }
        }

        grid.sync();

        // =====================================================================
        // Phase 2: Per-head Q/K RMSNorm + 64-dim partial RoPE + FP16 KV append
        //          + canonical device ring_head/cached_tokens advance
        // =====================================================================
        const int pos = inv.position_ids_device[0];
        const int cap = inv.kv_token_capacity;
        const int old_head = (inv.row_params_out[0].ring_row_origin + prefix_tokens) % cap;
        const unsigned n_q = static_cast<unsigned>(inv.local_q_heads);
        const unsigned n_kv = static_cast<unsigned>(inv.local_kv_heads);
        const unsigned total_heads = n_q + n_kv;
        const unsigned rotary_dim = static_cast<unsigned>(inv.rotary_dim);
        const unsigned half_rotary = rotary_dim >> 1;

        if (bid == 0u && tid == 0u)
        {
            *inv.d_ring_head = (old_head + 1) % cap;
            const int next_count = prefix_tokens + 1;
            *inv.d_cached_tokens = (next_count > cap) ? cap : next_count;
        }

        // Precompute sin/cos once per participating CTA (all heads share the same pos).
        if (bid < total_heads)
        {
            if (tid < half_rotary)
            {
                const float exponent = (2.0f * static_cast<float>(tid)) / static_cast<float>(rotary_dim);
                const float inv_freq = expf(inv.rope_neg_log_theta * exponent);
                __sincosf(static_cast<float>(pos) * inv_freq,
                          &shared.post.s_sin[tid], &shared.post.s_cos[tid]);
            }
            __syncthreads();
        }

#pragma unroll 1
        for (unsigned h = bid; h < total_heads; h += num_ctas)
        {
            if (h < n_q)
            {
                float *q_head = inv.Q + static_cast<std::size_t>(h) * 256u;
                float val = q_head[tid];
                if (inv.q_norm_gamma)
                    val = qsaHeadRMSNorm256(val, inv.q_norm_gamma, inv.qk_norm_eps, shared.post.warp_sums);
                shared.post.vals[tid] = val;
                __syncthreads();
                float out = val;
                if (tid < half_rotary)
                    out = shared.post.vals[tid] * shared.post.s_cos[tid] -
                          shared.post.vals[tid + half_rotary] * shared.post.s_sin[tid];
                else if (tid < rotary_dim)
                {
                    const unsigned p = tid - half_rotary;
                    out = shared.post.vals[p] * shared.post.s_sin[p] +
                          shared.post.vals[tid] * shared.post.s_cos[p];
                }
                q_head[tid] = out;
                __syncthreads();
            }
            else
            {
                const unsigned kv_h = h - n_q;
                float *k_head = inv.wk.output + static_cast<std::size_t>(kv_h) * 256u;
                const float *v_head = inv.wv.output + static_cast<std::size_t>(kv_h) * 256u;
                float kval = k_head[tid];
                const float vval = v_head[tid];
                if (inv.k_norm_gamma)
                    kval = qsaHeadRMSNorm256(kval, inv.k_norm_gamma, inv.qk_norm_eps, shared.post.warp_sums);
                if (!inv.skip_k_rope)
                {
                    shared.post.vals[tid] = kval;
                    __syncthreads();
                    if (tid < half_rotary)
                        kval = shared.post.vals[tid] * shared.post.s_cos[tid] -
                               shared.post.vals[tid + half_rotary] * shared.post.s_sin[tid];
                    else if (tid < rotary_dim)
                    {
                        const unsigned p = tid - half_rotary;
                        kval = shared.post.vals[p] * shared.post.s_sin[p] +
                               shared.post.vals[tid] * shared.post.s_cos[p];
                    }
                    __syncthreads();
                }
                k_head[tid] = kval;
                const std::size_t byte_off =
                    static_cast<std::size_t>(old_head) * inv.kv_token_stride_bytes +
                    static_cast<std::size_t>(kv_h) * inv.kv_head_stride_bytes;
                reinterpret_cast<__half *>(static_cast<unsigned char *>(inv.kv_keys_data) + byte_off)[tid] =
                    __float2half_rn(kval);
                reinterpret_cast<__half *>(static_cast<unsigned char *>(inv.kv_values_data) + byte_off)[tid] =
                    __float2half_rn(vval);
            }
        }
    }

    /** @brief Fatal device validation matching QwenSparseGPU.cuh::require. */
    __device__ __forceinline__ void qsaRequire(bool cond)
    {
        if (!cond)
            asm volatile("trap;");
    }

    /** @brief Authenticate causal QSA replay without host readback. */
    __device__ __forceinline__ llaminar2::qwen4exp::QwenSparseReplay qsaLoadReplay(
        const llaminar2::qwen4exp::QwenSparseReplay *state, int rows, int tokens)
    {
        const auto r = *state;
        qsaRequire(r.prefix_tokens >= 0 && r.live_rows >= 0 && r.live_rows <= rows &&
                   static_cast<long long>(r.prefix_tokens) + r.live_rows <= tokens &&
                   r.position_origin >= 0 &&
                   static_cast<long long>(r.position_origin) + r.prefix_tokens + r.live_rows <= 0x7fffffff);
        return r;
    }

    /** @brief Exact CUDA foldAttentionPrefix matching QwenSparseGPU.cuh::foldAttentionPrefix. */
    template <int StorageParts>
    __device__ __forceinline__ float2 qsaFoldAttentionPrefixCUDA(
        const llaminar2::qwen4exp::QwenSparseAttention &p,
        const llaminar2::qwen4exp::QwenSparseAttentionPartial *parts,
        int fold_parts,
        CUDARetainedQSAPostShared &shared,
        int head,
        unsigned tid)
    {
        if constexpr (StorageParts == 33)
        {
            if (tid < static_cast<unsigned>(fold_parts))
            {
                shared.fold.cached_maximum[tid] = parts[tid].maximum;
                shared.fold.cached_denominator[tid] = parts[tid].denominator;
                if (p.observations)
                {
                    shared.fold.cached_keys[tid] = parts[tid].key_elements;
                    shared.fold.cached_values[tid] = parts[tid].value_elements;
                }
            }
            __syncthreads();
            if (tid < 32u)
            {
                float maximum = -INFINITY;
#pragma unroll 1
                for (int part = 0; part < fold_parts; ++part)
                    maximum = fmaxf(maximum, shared.fold.cached_maximum[part]);
                float localDenominator = 0.0f, localScale = 0.0f;
                if (tid < static_cast<unsigned>(fold_parts))
                {
                    localDenominator = shared.fold.cached_denominator[tid];
                    localScale = (localDenominator == 0.0f) ? 0.0f : expf(shared.fold.cached_maximum[tid] - maximum);
                    shared.fold.scales[tid] = localScale;
                }
                float sum = 0.0f;
                const int subgroup_parts = (fold_parts < 32) ? fold_parts : 32;
#pragma unroll 1
                for (int part = 0; part < subgroup_parts; ++part)
                {
                    const float pd = __shfl_sync(0xffffffffu, localDenominator, part, 32);
                    const float ps = __shfl_sync(0xffffffffu, localScale, part, 32);
                    sum = fmaf(pd, ps, sum);
                }
                if (fold_parts == 33)
                {
                    const float pd = shared.fold.cached_denominator[32];
                    const float ps = (pd == 0.0f) ? 0.0f : expf(shared.fold.cached_maximum[32] - maximum);
                    sum = fmaf(pd, ps, sum);
                    if (tid == 0u)
                        shared.fold.scales[32] = ps;
                }
                if (tid == 0u)
                {
                    qsaRequire(isfinite(sum) && sum > 0.0f);
                    shared.fold.denominator = sum;
                }
            }
        }
        if (tid == 0u)
        {
            if constexpr (StorageParts == 8)
            {
                float maximum = -INFINITY;
#pragma unroll 1
                for (int part = 0; part < fold_parts; ++part)
                    maximum = fmaxf(maximum, parts[part].maximum);
                float sum = 0.0f;
#pragma unroll 1
                for (int part = 0; part < fold_parts; ++part)
                {
                    const float pd = parts[part].denominator;
                    const float pm = parts[part].maximum;
                    shared.fold.scales[part] = (pd == 0.0f) ? 0.0f : expf(pm - maximum);
                    sum = fmaf(pd, shared.fold.scales[part], sum);
                }
                qsaRequire(isfinite(sum) && sum > 0.0f);
                shared.fold.denominator = sum;
            }
            if (p.observations)
            {
                llaminar2::qwen4exp::QwenSparseAttentionObservation obs{};
#pragma unroll 1
                for (int part = 0; part < fold_parts; ++part)
                {
                    obs.key_elements += (StorageParts == 33) ? shared.fold.cached_keys[part] : parts[part].key_elements;
                    obs.value_elements += (StorageParts == 33) ? shared.fold.cached_values[part] : parts[part].value_elements;
                }
                obs.selected_tokens = p.selected_count[0];
                p.observations[head] = obs;
            }
        }
        __syncthreads();
        float result = 0.0f;
#pragma unroll 1
        for (int part = 0; part < fold_parts; ++part)
        {
            const float pd = (StorageParts == 33) ? shared.fold.cached_denominator[part] : parts[part].denominator;
            const float weighted = (pd == 0.0f) ? 0.0f : parts[part].weighted[tid];
            result = fmaf(weighted, shared.fold.scales[part], result);
        }
        if (fold_parts < StorageParts)
            result = fmaf(0.0f, 0.0f, result);
        if (fold_parts + 1 < StorageParts)
            result = fmaf(0.0f, 0.0f, result);
        return make_float2(result, shared.fold.denominator);
    }

    /** @brief Resident cooperative CUDA QSA Post-Attention megakernel (fold + commit + gate + Wo). */
    template <int Parts, bool GeneralCodebooks>
    __global__ __launch_bounds__(kThreads, GeneralCodebooks ? 2 : 4)
    void cudaRetainedQSAPostAttentionKernel(const __grid_constant__ CUDARetainedQSAPostInvocation inv)
    {
        static_assert(Parts == 8 || Parts == 33, "Closed CUDA FP16 QSA partition program");
        __shared__ CUDARetainedQSAPostShared shared;
        const auto grid = cooperative_groups::this_grid();
        const unsigned tid = threadIdx.x;
        const unsigned bid = blockIdx.x;
        const unsigned num_ctas = gridDim.x;
        const unsigned n_q = static_cast<unsigned>(inv.attention.query_heads);

        auto *scratch_bytes = static_cast<unsigned char *>(inv.scratch);
        auto *wo_codes_global = reinterpret_cast<int8_t *>(scratch_bytes + kScratchCodesOffset);
        auto *wo_scales_global = reinterpret_cast<float *>(scratch_bytes + kScratchScalesOffset);

        // =====================================================================
        // Phase 0: Per-head foldAttentionPrefix + sigmoid gate + Q8 staging,
        //          overlapped with commitPartial on CTA n_q % num_ctas.
        // =====================================================================
        const auto r = qsaLoadReplay(
            inv.attention.replay, inv.attention.row_capacity, inv.attention.token_capacity);
        const int live_rows = r.live_rows;

#pragma unroll 1
        for (unsigned head = bid; head < n_q; head += num_ctas)
        {
            if (live_rows <= 0)
            {
                if (tid == 0u && inv.attention.observations)
                    inv.attention.observations[head] = {};
                continue;
            }
            const auto *parts = inv.attention.gpu_partials + static_cast<std::size_t>(head) * Parts;
            int fold_parts = 8;
            if constexpr (Parts == 33)
            {
                const int selected = inv.attention.selected_count[0];
                qsaRequire(selected > 0 && selected <= llaminar2::qwen4exp::QwenSparseSelected);
                if (selected <= 8 * 64)
                    fold_parts = 8;
                else if (selected <= 12 * 64)
                    fold_parts = 12;
                else if (selected <= 16 * 64)
                    fold_parts = 16;
                else if (selected <= 32 * 64)
                    fold_parts = 32;
                else
                    fold_parts = 33;
            }
            const float2 folded = qsaFoldAttentionPrefixCUDA<Parts>(
                inv.attention, parts, fold_parts, shared, static_cast<int>(head), tid);

            float result = folded.x / folded.y;
            const unsigned elem = head * 256u + tid;
            const float sig = 1.0f / (1.0f + expf(-inv.fa_gate[elem]));
            result = sig * result;
            inv.attention.output[elem] = result;

            const unsigned warp = tid >> 5;
            const unsigned lane = tid & 31u;
            qsaQuantizeBlockwiseWarp32(
                result, wo_codes_global + (head * 256u + warp * 32u),
                wo_scales_global + (head * 8u + warp), lane);
            __syncthreads();
        }

        if (inv.has_commit && bid == (n_q % num_ctas))
        {
            const auto &cp = inv.commit;
            const auto cr = qsaLoadReplay(
                cp.replay, cp.acceptance.candidates().rowsPerRequest(),
                cp.transaction_prior ? cp.token_capacity : 0x7fffffff);
            qsaRequire(!cp.transaction_prior ||
                       cp.transaction_prior->row_capacity == cp.acceptance.candidates().rowsPerRequest());
            const int accepted = cp.acceptance.resolveNative(0);
            qsaRequire(accepted >= 0 && accepted <= cr.live_rows);
            const bool active = !(cr.live_rows == 0 || (accepted == 0 && !cp.transaction_prior));
            const int total = cr.prefix_tokens + accepted;
            const int begin = (total / 4) * 4;
            const int rem = total % 4;
            float values[3];
            if (active && tid < 128u)
            {
                for (int member = 0; member < rem; ++member)
                {
                    const int token = begin + member;
                    values[member] = token < cr.prefix_tokens
                        ? cp.prior_partial[static_cast<std::size_t>(token - (cr.prefix_tokens / 4) * 4) * 128u + tid]
                        : cp.raw_keys[static_cast<std::size_t>(token - cr.prefix_tokens) * 128u + tid];
                }
            }
            __syncthreads();
            if (active && tid < 128u)
            {
                for (int member = 0; member < rem; ++member)
                    cp.partial_keys[static_cast<std::size_t>(member) * 128u + tid] = values[member];
                if (tid == 0u && cp.work_counters.bound())
                    qsaRequire(::llaminar2::addNativeCounter64(
                        cp.work_counters.metric(llaminar2::qwen4exp::QwenSparseWorkCounter::AcceptedPayload),
                        static_cast<std::uint64_t>(rem) * 128u * sizeof(float)));
            }
        }

        grid.sync();

        // =====================================================================
        // Phase 1: Load Q8 attn_output into SMEM + coalesced 16-col Wo projection
        // =====================================================================
        const unsigned wo_k = n_q * 256u;
        const unsigned wo_blocks = wo_k >> 5;
        const auto *src_v4 = reinterpret_cast<const int4 *>(wo_codes_global);
        auto *dst_v4 = reinterpret_cast<int4 *>(shared.wo.act_codes);
        for (unsigned v = tid; v < (wo_k >> 4); v += kThreads)
            dst_v4[v] = src_v4[v];
        if (tid < wo_blocks)
            shared.wo.act_scales[tid] = wo_scales_global[tid];
        __syncthreads();

        const unsigned wo_tiles = (static_cast<unsigned>(inv.wo.weights.n) + 15u) >> 4;
#pragma unroll 1
        for (unsigned tile = bid; tile < wo_tiles; tile += num_ctas)
        {
            qsaCompleteTile16<GeneralCodebooks>(
                inv.wo, tile << 4, shared.wo.act_codes, shared.wo.act_scales, shared.wo.partials);
        }
    }

    bool supportedCUDANativeCodebook(std::uint8_t cb) noexcept
    {
        switch (cb)
        {
        case 0u: case 4u: case 5u: case 6u: case 7u: case 8u: case 9u: case 10u:
        case 11u: case 12u: case 13u: case 14u: case 15u: case 16u: case 17u: case 19u:
        case kNativeQuantisedExpandedInt8MinCodebook:
        case kNativeQuantisedIQ4XSSourceScaleCodebook:
            return true;
        default:
            return false;
        }
    }

    bool validProj(const CUDARetainedQSAProjection &p, int expected_n, int expected_k, bool require_output) noexcept
    {
        return (!require_output || p.output != nullptr) &&
               p.weights.valid() && p.weights.n == expected_n && p.weights.k == expected_k &&
               p.weights.blocks_per_row == static_cast<unsigned>(expected_k / 32) &&
               p.weights.source_identity_present &&
               supportedCUDANativeCodebook(p.weights.codebook_id) &&
               p.k_partitions > 0 && p.k_partitions <= static_cast<int>(p.weights.blocks_per_row) &&
               p.blocks_per_partition > 0 &&
               (p.ordered_kpar || p.k_partitions == 1);
    }

    bool isQ8Proj(const CUDARetainedQSAProjection &p) noexcept
    {
        return p.weights.codebook_id == 19u && !p.weights.mins && !p.weights.emins;
    }

    bool validIndexerProj(const CUDARetainedQSAIndexerProj &p, int expected_n, bool optional) noexcept
    {
        if (p.format == CUDARetainedQSAIndexerFormat::Absent)
            return optional;
        if (p.n != expected_n)
            return false;
        if (p.format == CUDARetainedQSAIndexerFormat::NativeQuantised)
            return validProj(p.quantised, expected_n, static_cast<int>(kHidden), true);
        return p.floating_weights != nullptr && p.floating_output != nullptr;
    }

    bool isQ8OrFloatingIndexer(const CUDARetainedQSAIndexerProj &p) noexcept
    {
        return p.format != CUDARetainedQSAIndexerFormat::NativeQuantised || isQ8Proj(p.quantised);
    }
} // namespace

std::size_t retainedCUDAQSAScratchBytes() noexcept
{
    return kScratchTotalBytes;
}

CUDARetainedQSAPreAdmission prepareCUDARetainedQSAPreAttention(
    DeviceId device,
    const CUDARetainedQSAPreInvocation &invocation,
    std::uint64_t workspace_generation)
{
    if (!device.is_cuda())
        throw std::invalid_argument("CUDARetainedQSAPreAttention requires a CUDA device");
    const auto &hc = invocation.hc;
    constexpr int hc_packets = (4 * static_cast<int>(kHidden)) / 32;
    if (hc.norm.hidden != kHidden || !hc.norm.gain || !hc.norm.output ||
        !hc.mixed || !hc.raw_down || !hc.norm_codes || !hc.norm_scales ||
        !hc.down.valid() || !hc.up.weights.valid() ||
        !supportedCUDANativeCodebook(hc.down.codebook_id) ||
        !supportedCUDANativeCodebook(hc.up.weights.codebook_id) ||
        hc.down.n != static_cast<int>(kLowRank) || hc.down.k != static_cast<int>(4u * kHidden) ||
        hc.up.weights.n != static_cast<int>(4u * kHidden) || hc.up.weights.k != static_cast<int>(kLowRank) ||
        hc.down_partitions.count <= 0 || hc.down_partitions.count > hc_packets ||
        hc.down_partitions.blocks_per_partition != (hc_packets + hc.down_partitions.count - 1) / hc.down_partitions.count ||
        (hc.down_fold == moe::HCDownFold::Direct ? hc.down_partitions.count != 1 :
            hc.down_fold != moe::HCDownFold::OrderedPartitions || !hc.down_partials))
    {
        throw std::invalid_argument("CUDARetainedQSAPreAttention received invalid HC prefix operands");
    }
    if (invocation.local_q_heads < 1 ||
        static_cast<unsigned>(invocation.local_q_heads) > kMaxCUDARetainedQSAQHeads ||
        invocation.local_kv_heads < 1 ||
        static_cast<unsigned>(invocation.local_kv_heads) > kMaxCUDARetainedQSAKVHeads ||
        invocation.rotary_dim <= 0 || invocation.rotary_dim > 256 || (invocation.rotary_dim & 1) != 0 ||
        !invocation.Q || !invocation.fa_gate ||
        !invocation.position_ids_device || !invocation.d_ring_head || !invocation.d_cached_tokens ||
        invocation.kv_token_capacity <= 0 ||
        !invocation.kv_keys_data || !invocation.kv_values_data ||
        invocation.kv_token_stride_bytes == 0 || invocation.kv_head_stride_bytes == 0 ||
        !invocation.replay_out || !invocation.row_params_out ||
        !invocation.scratch || invocation.scratch_bytes < retainedCUDAQSAScratchBytes() ||
        (reinterpret_cast<std::uintptr_t>(invocation.scratch) & 15u) != 0u)
    {
        throw std::invalid_argument("CUDARetainedQSAPreAttention received invalid QSA geometry or pointers");
    }
    if (!validProj(invocation.wq, invocation.local_q_heads * 512, static_cast<int>(kHidden), false) ||
        !validProj(invocation.wk, invocation.local_kv_heads * 256, static_cast<int>(kHidden), true) ||
        !validProj(invocation.wv, invocation.local_kv_heads * 256, static_cast<int>(kHidden), true) ||
        !validIndexerProj(invocation.qsa_key, 128, false) ||
        !validIndexerProj(invocation.qsa_query, 512, true))
    {
        throw std::invalid_argument("CUDARetainedQSAPreAttention received invalid projection descriptors");
    }

    const bool all_q8 =
        hc.down.codebook_id == 19u && !hc.down.mins && !hc.down.emins &&
        hc.up.weights.codebook_id == 19u && !hc.up.weights.mins && !hc.up.weights.emins &&
        isQ8Proj(invocation.wq) && isQ8Proj(invocation.wk) && isQ8Proj(invocation.wv) &&
        isQ8OrFloatingIndexer(invocation.qsa_key) && isQ8OrFloatingIndexer(invocation.qsa_query);

    const void *fn = all_q8
        ? reinterpret_cast<const void *>(&cudaRetainedQSAPreAttentionKernel<false>)
        : reinterpret_cast<const void *>(&cudaRetainedQSAPreAttentionKernel<true>);
    const auto &hw = cudaHardwareMetadata().at(device.ordinal);
    const unsigned sms = static_cast<unsigned>(std::max(1, hw.sm_count));
    const unsigned reserve = (sms > 16u) ? 8u : 0u;
    const auto co_residency = NativeCoResidencyReservation::reserveProcessors(reserve);

    auto native = NativeCooperativeKernelBoundary<NativeCooperativeBackend::CUDA>::prepare(
        device, fn, kThreads, 0u, workspace_generation,
        NativeResidentLaunchPolicy::Cooperative, co_residency);
    if (native.resources().private_bytes != 0u)
        throw std::invalid_argument("CUDARetainedQSAPreAttention must be zero-spill (localSizeBytes == 0)");
    const unsigned grid = native.maximumGrid();
    if (grid < 4u)
        throw std::invalid_argument("CUDARetainedQSAPreAttention requires at least 4 resident CTAs");
    return CUDARetainedQSAPreAdmission{invocation, native, grid};
}

void launchCUDARetainedQSAPreAttention(
    const CUDARetainedQSAPreAdmission &admission,
    void *stream,
    std::uint64_t workspace_generation)
{
    auto inv = admission.invocation;
    void *args[] = {&inv};
    NativeCooperativeKernelBoundary<NativeCooperativeBackend::CUDA>::launch(
        admission.native, args, admission.grid, static_cast<cudaStream_t>(stream), workspace_generation);
}

CUDARetainedQSAPostAdmission prepareCUDARetainedQSAPostAttention(
    DeviceId device,
    const CUDARetainedQSAPostInvocation &invocation,
    std::uint64_t workspace_generation)
{
    if (!device.is_cuda())
        throw std::invalid_argument("CUDARetainedQSAPostAttention requires a CUDA device");
    const auto &a = invocation.attention;
    if (a.row_capacity != 1 || a.query_heads < 1 ||
        static_cast<unsigned>(a.query_heads) > kMaxCUDARetainedQSAQHeads ||
        (a.arithmetic != llaminar2::qwen4exp::QwenSparseAttentionArithmetic::TileFP16WithFP32Totals &&
         a.arithmetic != llaminar2::qwen4exp::QwenSparseAttentionArithmetic::TileFP16WithFP32Totals33) ||
        !a.replay || !a.selected_count || !a.gpu_partials || !a.output ||
        !invocation.fa_gate ||
        !validProj(invocation.wo, static_cast<int>(kHidden), a.query_heads * 256, true) ||
        !invocation.scratch || invocation.scratch_bytes < retainedCUDAQSAScratchBytes() ||
        (reinterpret_cast<std::uintptr_t>(invocation.scratch) & 15u) != 0u)
    {
        throw std::invalid_argument("CUDARetainedQSAPostAttention received invalid geometry or bindings");
    }

    const bool parts33 =
        a.arithmetic == llaminar2::qwen4exp::QwenSparseAttentionArithmetic::TileFP16WithFP32Totals33;
    const bool q8_wo = isQ8Proj(invocation.wo);
    const void *fn = parts33
        ? (q8_wo ? reinterpret_cast<const void *>(&cudaRetainedQSAPostAttentionKernel<33, false>)
                 : reinterpret_cast<const void *>(&cudaRetainedQSAPostAttentionKernel<33, true>))
        : (q8_wo ? reinterpret_cast<const void *>(&cudaRetainedQSAPostAttentionKernel<8, false>)
                 : reinterpret_cast<const void *>(&cudaRetainedQSAPostAttentionKernel<8, true>));

    const auto &hw = cudaHardwareMetadata().at(device.ordinal);
    const unsigned sms = static_cast<unsigned>(std::max(1, hw.sm_count));
    const unsigned reserve = (sms > 16u) ? 8u : 0u;
    const auto co_residency = NativeCoResidencyReservation::reserveProcessors(reserve);

    auto native = NativeCooperativeKernelBoundary<NativeCooperativeBackend::CUDA>::prepare(
        device, fn, kThreads, 0u, workspace_generation,
        NativeResidentLaunchPolicy::Cooperative, co_residency);
    if (native.resources().private_bytes != 0u)
        throw std::invalid_argument("CUDARetainedQSAPostAttention must be zero-spill (localSizeBytes == 0)");
    const unsigned grid = std::max(1u, native.maximumGrid());
    return CUDARetainedQSAPostAdmission{invocation, native, grid};
}

void launchCUDARetainedQSAPostAttention(
    const CUDARetainedQSAPostAdmission &admission,
    void *stream,
    std::uint64_t workspace_generation)
{
    auto inv = admission.invocation;
    void *args[] = {&inv};
    NativeCooperativeKernelBoundary<NativeCooperativeBackend::CUDA>::launch(
        admission.native, args, admission.grid, static_cast<cudaStream_t>(stream), workspace_generation);
}
} // namespace llaminar2::cuda::qwen4exp
