/**
 * @file CUDARetainedGDN.cu
 * @brief Spill-free cooperative CUDA Full GDN Attention megakernel implementation.
 *
 * Replaces all 25 participant-local CUDA GDN attention kernels (HC prefix,
 * QKV/Z/Alpha/Beta projections, ShortConv1d, GDN recurrence, GatedRMSNorm,
 * blockwise Q8 quantizers, and ssm_out projection) with one cooperative kernel
 * executing at 3-4 CTAs/SM with 8-SM co-residency reservation.
 *
 * QKV, Z, and ssm_out use 8-column warp sub-tiles (4 lanes/column, 256-byte
 * coalesced GDDR6X transactions, 2-block software pipelining) so every even
 * canonical K-partition count (kb=10, 14, 16, ...) achieves 100% warp balance
 * with zero register spills and exact bit-for-bit serial M=1 identity.
 */
#include "kernels/cuda/qwen4exp/CUDARetainedGDN.h"

#include "backends/cuda/CUDAHardwareMetadata.h"
#include "kernels/common/NativeCooperativeKernel.cuh"
#include "kernels/cuda/moe/CUDARetainedHCPrefix.cuh"

#include <cooperative_groups.h>
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
    constexpr unsigned kMaxKPartitions = 32u;
    constexpr unsigned kMaxOutElems = kMaxCUDARetainedGDNVHeads * 128u;      // 6144
    constexpr unsigned kMaxOutBlocks = kMaxCUDARetainedGDNVHeads * 4u;       // 192

    constexpr std::size_t kScratchCodesOffset = 0u;
    constexpr std::size_t kScratchScalesOffset = (kMaxOutElems + 15u) & ~std::size_t(15u);
    constexpr std::size_t kScratchHeadDoneOffset =
        ((kScratchScalesOffset + kMaxOutBlocks * sizeof(float)) + 15u) & ~std::size_t(15u);
    constexpr std::size_t kScratchHcDownDoneOffset =
        ((kScratchHeadDoneOffset + kMaxCUDARetainedGDNVHeads * sizeof(std::uint32_t)) + 15u) & ~std::size_t(15u);
    constexpr std::size_t kScratchTotalBytes =
        ((kScratchHcDownDoneOffset + 32u * sizeof(std::uint32_t)) + 15u) & ~std::size_t(15u);

    /** @brief Phase-exclusive CTA shared memory (12,096 B: fits 4 CTAs/SM in 48 KB SMEM). */
    union CUDARetainedGDNShared
    {
        moe::RetainedHCPrefixShared<kHidden> hc;
        struct
        {
            alignas(16) int8_t act_codes[kHidden];           // 2560 B
            alignas(16) float act_scales[kHidden / 32u];     // 320 B
            alignas(16) __half mixed_fp16[kHidden];          // 5120 B
            alignas(16) float partials[kMaxKPartitions][32]; // 4096 B
        } p2;                                                // 12096 B
        struct
        {
            alignas(16) float q_local[128];                  // 512 B
            alignas(16) float k_local[128];                  // 512 B
            alignas(16) float reduce_kv[256];                // 1024 B
            alignas(16) float reduce_out[256];               // 1024 B
            alignas(16) float state_lo[8][256];              // 8192 B (bank-conflict-free [r][tid])
            float warp_sums[8];                              // 32 B
            float decay;
            float beta_h;
            unsigned finished;
        } p3;                                                // ~11312 B (<= 12096 B!)
        struct
        {
            alignas(16) int8_t act_codes[kMaxOutElems];      // 6144 B
            alignas(16) float act_scales[kMaxOutBlocks];     // 768 B
            alignas(16) float partials[kMaxKPartitions][16]; // 2048 B
        } p4;                                                // 8960 B
    };

    static_assert(sizeof(CUDARetainedGDNShared) <= 12288u,
        "CUDARetainedGDNShared must fit 4 CTAs/SM inside default 48 KB SMEM");

    /** @brief Exact CUDA blockwise Q8 quantization matching quantize_activations_blockwise_kernel. */
    __device__ __forceinline__ void quantizeBlockwiseWarp32(
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

    /**
     * @brief Compute one 8-column x 1-partition Q8_0 sub-tile with 256B coalesced warp loads
     *        and 2-block software pipelining (4 lanes/col x int2 = 256B = 2 full 128B sectors).
     */
    __device__ __forceinline__ float accumulateQ8SubTile8(
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
        const unsigned col_in_sub = lane >> 2; // 0..7
        const unsigned q_part = lane & 3u;     // 0..3 (8 bytes = int2 per lane)
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

    /**
     * @brief Zero-imbalance 32-column Q8_0 projection (4 sub-tiles x kb partitions across 8 warps).
     *
     * For QKV (kb=10): 40 warp-tasks / 8 warps = 5 tasks/warp (100% balanced).
     * For Z   (kb=14): 56 warp-tasks / 8 warps = 7 tasks/warp (100% balanced).
     * Optionally fuses causal 4-tap ShortConv1d + SiLU directly from registers.
     */
    template <bool InlineConv>
    __device__ __forceinline__ void completeQ8Tile32(
        const CUDARetainedGDNProjection &proj,
        unsigned col_base,
        const int8_t *act_codes,
        const float *act_scales,
        float (*partials)[32],
        const float *conv_weight = nullptr,
        const float *conv_bias = nullptr,
        float *conv_state = nullptr,
        float *gdn_recurrence_in = nullptr)
    {
        const unsigned tid = threadIdx.x;
        const unsigned warp = tid >> 5;
        const unsigned lane = tid & 31u;
        const unsigned col_in_sub = lane >> 2;
        const unsigned q_part = lane & 3u;

        const auto &w = proj.weights;
        const unsigned n = static_cast<unsigned>(w.n);
        const unsigned blocks = static_cast<unsigned>(w.blocks_per_row);
        const unsigned kb = static_cast<unsigned>(proj.k_partitions);
        const unsigned bps = static_cast<unsigned>(proj.blocks_per_partition);
        const auto *w_scales = static_cast<const uint16_t *>(w.scales);

        const unsigned total_warp_tasks = 4u * kb;
        for (unsigned task = warp; task < total_warp_tasks; task += 8u)
        {
            const unsigned p = task >> 2;
            const unsigned st = task & 3u;
            const unsigned blk_begin = p * bps;
            const unsigned blk_end = min(blocks, blk_begin + bps);
            const float sum = accumulateQ8SubTile8(
                w, w_scales, act_codes, act_scales, col_base + st * 8u, blk_begin, blk_end, lane);
            if (q_part == 0u)
                partials[p][st * 8u + col_in_sub] = sum;
        }
        __syncthreads();

        if (warp == 0u)
        {
            const unsigned col = col_base + lane;
            if (col < n)
            {
                float total = 0.0f;
                for (unsigned p = 0u; p < kb; ++p)
                    total = __fadd_rn(total, partials[p][lane]);
                proj.output[col] = total;

                if constexpr (InlineConv)
                {
                    const std::size_t s_base = static_cast<std::size_t>(col) * 3u;
                    const std::size_t w_base = static_cast<std::size_t>(col) * 4u;
                    const float c0 = conv_state[s_base + 0u];
                    const float c1 = conv_state[s_base + 1u];
                    const float c2 = conv_state[s_base + 2u];
                    float sum = 0.0f;
                    sum += c0 * conv_weight[w_base + 0u];
                    sum += c1 * conv_weight[w_base + 1u];
                    sum += c2 * conv_weight[w_base + 2u];
                    sum += total * conv_weight[w_base + 3u];
                    if (conv_bias)
                        sum += conv_bias[col];
                    gdn_recurrence_in[col] = sum / (1.0f + expf(-sum));
                    conv_state[s_base + 0u] = c1;
                    conv_state[s_base + 1u] = c2;
                    conv_state[s_base + 2u] = total;
                }
            }
        }
        __syncthreads();
    }

    /**
     * @brief Zero-imbalance 16-column Q8_0 projection for ssm_out (2 sub-tiles x kb partitions).
     */
    __device__ __forceinline__ void completeQ8Tile16(
        const CUDARetainedGDNProjection &proj,
        unsigned col_base,
        const int8_t *act_codes,
        const float *act_scales,
        float (*partials)[16])
    {
        const unsigned tid = threadIdx.x;
        const unsigned warp = tid >> 5;
        const unsigned lane = tid & 31u;
        const unsigned col_in_sub = lane >> 2;
        const unsigned q_part = lane & 3u;

        const auto &w = proj.weights;
        const unsigned n = static_cast<unsigned>(w.n);
        const unsigned blocks = static_cast<unsigned>(w.blocks_per_row);
        const unsigned kb = static_cast<unsigned>(proj.k_partitions);
        const unsigned bps = static_cast<unsigned>(proj.blocks_per_partition);
        const auto *w_scales = static_cast<const uint16_t *>(w.scales);

        const unsigned total_warp_tasks = 2u * kb;
        for (unsigned task = warp; task < total_warp_tasks; task += 8u)
        {
            const unsigned p = task >> 1;
            const unsigned st = task & 1u;
            const unsigned blk_begin = p * bps;
            const unsigned blk_end = min(blocks, blk_begin + bps);
            const float sum = accumulateQ8SubTile8(
                w, w_scales, act_codes, act_scales, col_base + st * 8u, blk_begin, blk_end, lane);
            if (q_part == 0u)
                partials[p][st * 8u + col_in_sub] = sum;
        }
        __syncthreads();

        if (tid < 16u && (col_base + tid) < n)
        {
            float total = 0.0f;
            for (unsigned p = 0u; p < kb; ++p)
                total = __fadd_rn(total, partials[p][tid]);
            proj.output[col_base + tid] = total;
        }
        __syncthreads();
    }

    /**
     * @brief Exact FP16RNE alpha/beta dot product matching short_half_projection_group_kernel.
     */
    __device__ __forceinline__ void completeHalfControlRow(
        const __half *mixed_fp16,
        const std::uint16_t *weight_row,
        float *output,
        float *partials_flat)
    {
        const unsigned tid = threadIdx.x;
        const auto *b = reinterpret_cast<const __half *>(weight_row);
        float sum = 0.0f;
        for (unsigned j = tid; j < kHidden; j += kThreads)
            sum += __half2float(mixed_fp16[j]) * __half2float(b[j]);
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
                *output = result;
        }
        __syncthreads();
    }

    /** @brief Load 16 state rows split across SMEM [0..7] and registers [8..15] for <=64 reg occupancy. */
    template <GDNRecurrenceStateLayout StateLayout>
    __device__ __forceinline__ void loadGDNStateSplit(
        const float *head, float (&state_lo)[8][256], float (&state_hi)[8],
        int row_begin, int vi, unsigned tid)
    {
        if constexpr (StateLayout == GDNRecurrenceStateLayout::ValueMajorKeyContiguous)
        {
            const float *src = head + vi * 128 + row_begin;
            if ((reinterpret_cast<std::uintptr_t>(src) & 15u) == 0u)
            {
#pragma unroll
                for (int r = 0; r < 8; r += 4)
                {
                    const float4 v = *reinterpret_cast<const float4 *>(src + r);
                    state_lo[r + 0][tid] = v.x;
                    state_lo[r + 1][tid] = v.y;
                    state_lo[r + 2][tid] = v.z;
                    state_lo[r + 3][tid] = v.w;
                }
#pragma unroll
                for (int r = 0; r < 8; r += 4)
                {
                    const float4 v = *reinterpret_cast<const float4 *>(src + 8 + r);
                    state_hi[r + 0] = v.x;
                    state_hi[r + 1] = v.y;
                    state_hi[r + 2] = v.z;
                    state_hi[r + 3] = v.w;
                }
            }
            else
            {
#pragma unroll
                for (int r = 0; r < 8; ++r)
                    state_lo[r][tid] = src[r];
#pragma unroll
                for (int r = 0; r < 8; ++r)
                    state_hi[r] = src[8 + r];
            }
        }
        else
        {
#pragma unroll
            for (int r = 0; r < 8; ++r)
                state_lo[r][tid] = head[(row_begin + r) * 128 + vi];
#pragma unroll
            for (int r = 0; r < 8; ++r)
                state_hi[r] = head[(row_begin + 8 + r) * 128 + vi];
        }
    }

    /** @brief Store 16 state rows split across SMEM [0..7] and registers [8..15]. */
    template <GDNRecurrenceStateLayout StateLayout>
    __device__ __forceinline__ void storeGDNStateSplit(
        float *head, const float (&state_lo)[8][256], const float (&state_hi)[8],
        int row_begin, int vi, unsigned tid)
    {
        if constexpr (StateLayout == GDNRecurrenceStateLayout::ValueMajorKeyContiguous)
        {
            float *dst = head + vi * 128 + row_begin;
            if ((reinterpret_cast<std::uintptr_t>(dst) & 15u) == 0u)
            {
#pragma unroll
                for (int r = 0; r < 8; r += 4)
                {
                    *reinterpret_cast<float4 *>(dst + r) = make_float4(
                        state_lo[r + 0][tid], state_lo[r + 1][tid],
                        state_lo[r + 2][tid], state_lo[r + 3][tid]);
                }
#pragma unroll
                for (int r = 0; r < 8; r += 4)
                {
                    *reinterpret_cast<float4 *>(dst + 8 + r) = make_float4(
                        state_hi[r + 0], state_hi[r + 1], state_hi[r + 2], state_hi[r + 3]);
                }
            }
            else
            {
#pragma unroll
                for (int r = 0; r < 8; ++r)
                    dst[r] = state_lo[r][tid];
#pragma unroll
                for (int r = 0; r < 8; ++r)
                    dst[8 + r] = state_hi[r];
            }
        }
        else
        {
#pragma unroll
            for (int r = 0; r < 8; ++r)
                head[(row_begin + r) * 128 + vi] = state_lo[r][tid];
#pragma unroll
            for (int r = 0; r < 8; ++r)
                head[(row_begin + 8 + r) * 128 + vi] = state_hi[r];
        }
    }

    /** @brief General all-18-codebook 32-column projection tile (supports every NativeQuantised codebook). */
    template <bool InlineConv>
    __device__ __forceinline__ void completeGeneralTile32(
        const CUDARetainedGDNProjection &proj,
        unsigned col_base,
        const int8_t *act_codes,
        const float *act_scales,
        float (*partials)[32],
        const float *conv_weight = nullptr,
        const float *conv_bias = nullptr,
        float *conv_state = nullptr,
        float *gdn_recurrence_in = nullptr)
    {
        const unsigned tid = threadIdx.x;
        const unsigned warp = tid >> 5;
        const unsigned lane = tid & 31u;
        const auto &w = proj.weights;
        const unsigned n = static_cast<unsigned>(w.n);
        const unsigned blocks = static_cast<unsigned>(w.blocks_per_row);
        const unsigned kb = static_cast<unsigned>(proj.k_partitions);
        const unsigned bps = static_cast<unsigned>(proj.blocks_per_partition);
        const unsigned col = col_base + lane;

        for (unsigned p = warp; p < kb; p += 8u)
        {
            float sum = 0.0f;
            if (col < n)
            {
                const unsigned blk_begin = p * bps;
                const unsigned blk_end = min(blocks, blk_begin + bps);
#pragma unroll 1
                for (unsigned b = blk_begin; b < blk_end; ++b)
                {
                    const unsigned linear = b * n + col;
                    sum = __fadd_rn(sum, moe::retainedNativeQuantisedBlockDynamic(
                        w, linear, act_codes + b * 32u, act_scales[b]));
                }
            }
            partials[p][lane] = sum;
        }
        __syncthreads();

        if (warp == 0u && col < n)
        {
            float total = 0.0f;
            for (unsigned p = 0u; p < kb; ++p)
                total = __fadd_rn(total, partials[p][lane]);
            proj.output[col] = total;

            if constexpr (InlineConv)
            {
                const std::size_t s_base = static_cast<std::size_t>(col) * 3u;
                const std::size_t w_base = static_cast<std::size_t>(col) * 4u;
                const float c0 = conv_state[s_base + 0u];
                const float c1 = conv_state[s_base + 1u];
                const float c2 = conv_state[s_base + 2u];
                float sum = 0.0f;
                sum += c0 * conv_weight[w_base + 0u];
                sum += c1 * conv_weight[w_base + 1u];
                sum += c2 * conv_weight[w_base + 2u];
                sum += total * conv_weight[w_base + 3u];
                if (conv_bias)
                    sum += conv_bias[col];
                gdn_recurrence_in[col] = sum / (1.0f + expf(-sum));
                conv_state[s_base + 0u] = c1;
                conv_state[s_base + 1u] = c2;
                conv_state[s_base + 2u] = total;
            }
        }
        __syncthreads();
    }

    /** @brief General all-18-codebook 16-column ssm_out projection tile. */
    __device__ __forceinline__ void completeGeneralTile16(
        const CUDARetainedGDNProjection &proj,
        unsigned col_base,
        const int8_t *act_codes,
        const float *act_scales,
        float (*partials)[16])
    {
        const unsigned tid = threadIdx.x;
        const unsigned half_warp = tid >> 4;
        const unsigned col_in_tile = tid & 15u;
        const auto &w = proj.weights;
        const unsigned n = static_cast<unsigned>(w.n);
        const unsigned blocks = static_cast<unsigned>(w.blocks_per_row);
        const unsigned kb = static_cast<unsigned>(proj.k_partitions);
        const unsigned bps = static_cast<unsigned>(proj.blocks_per_partition);
        const unsigned col = col_base + col_in_tile;

        for (unsigned p = half_warp; p < kb; p += 16u)
        {
            float sum = 0.0f;
            if (col < n)
            {
                const unsigned blk_begin = p * bps;
                const unsigned blk_end = min(blocks, blk_begin + bps);
#pragma unroll 1
                for (unsigned b = blk_begin; b < blk_end; ++b)
                {
                    const unsigned linear = b * n + col;
                    sum = __fadd_rn(sum, moe::retainedNativeQuantisedBlockDynamic(
                        w, linear, act_codes + b * 32u, act_scales[b]));
                }
            }
            partials[p][col_in_tile] = sum;
        }
        __syncthreads();

        if (tid < 16u && (col_base + tid) < n)
        {
            float total = 0.0f;
            for (unsigned p = 0u; p < kb; ++p)
                total = __fadd_rn(total, partials[p][tid]);
            proj.output[col_base + tid] = total;
        }
        __syncthreads();
    }

    /**
     * @brief Complete cooperative CUDA Full GDN Attention megakernel.
     *
     * `GeneralCodebooks == false`: 4-CTA/SM (`<=64` regs, 0 spills) specialized Q8_0 path.
     * `GeneralCodebooks == true`: 2-CTA/SM (0 spills) complete all-18-codebook general path.
     */
    template <GDNRecurrenceStateLayout StateLayout, bool GeneralCodebooks = false>
    __global__ __launch_bounds__(kThreads, GeneralCodebooks ? 2 : 4)
    void cudaRetainedGDNKernel(const __grid_constant__ CUDARetainedGDNInvocation args)
    {
        __shared__ CUDARetainedGDNShared shared;
        auto grid = cooperative_groups::this_grid();
        const unsigned tid = threadIdx.x;
        const unsigned warp = tid >> 5;
        const unsigned lane = tid & 31u;

        // Zero HC Down atomic completion counters and GDN head_done counters before grid.sync() #1.
        auto *hc_down_done = reinterpret_cast<std::uint32_t *>(
            static_cast<unsigned char *>(args.scratch) + kScratchHcDownDoneOffset);
        if (blockIdx.x == 0u)
        {
            if (tid < 32u)
                hc_down_done[tid] = 0u;
            if (tid < static_cast<unsigned>(args.n_v_heads))
            {
                auto *head_done = reinterpret_cast<std::uint32_t *>(
                    static_cast<unsigned char *>(args.scratch) + kScratchHeadDoneOffset);
                head_done[tid] = 0u;
            }
        }

        // -----------------------------------------------------------------
        // Phase 1: Complete POST/HC prefix (Write or PublishedTrunk, Norm,
        //          single-sync HC Down + overlapped 4-CTA Injection, HC Up, Read).
        // -----------------------------------------------------------------
        moe::publishRetainedHCPrefix<false, false, kHidden, true, GeneralCodebooks>(
            args.hc, shared.hc, grid, nullptr, hc_down_done);

        // -----------------------------------------------------------------
        // Phase 2: Selectively stage Q8 (QKV/Z CTAs) and FP16RNE (Alpha/Beta CTAs),
        //          fitting all 256 TP2 QKV+Z tiles + 48 Alpha/Beta rows in 1 single wave!
        // -----------------------------------------------------------------
        {
            const float *mixed = args.hc.mixed;
            const unsigned qkv_tiles = (static_cast<unsigned>(args.qkv.weights.n) + 31u) >> 5;
            const unsigned z_tiles = (static_cast<unsigned>(args.z.weights.n) + 31u) >> 5;
            const unsigned total_tiles = qkv_tiles + z_tiles;

            const unsigned half_tasks = 2u * static_cast<unsigned>(args.half.heads);
            const unsigned tail_shift = total_tiles % gridDim.x;
            const unsigned shifted_bid = (blockIdx.x + gridDim.x - tail_shift) % gridDim.x;
            const unsigned tail_ctas = (tail_shift > 0u) ? (gridDim.x - tail_shift) : gridDim.x;
            const unsigned h_begin = (shifted_bid < tail_ctas) ? (shifted_bid * half_tasks / tail_ctas) : 0u;
            const unsigned h_end = (shifted_bid < tail_ctas) ? ((shifted_bid + 1u) * half_tasks / tail_ctas) : 0u;

            if (blockIdx.x < total_tiles)
            {
                constexpr unsigned blocks = kHidden / 32u;
                for (unsigned b = warp; b < blocks; b += 8u)
                {
                    const unsigned idx = b * 32u + lane;
                    quantizeBlockwiseWarp32(mixed[idx], shared.p2.act_codes + b * 32u,
                        shared.p2.act_scales + b, lane);
                }
                __syncthreads();

                for (unsigned tile = blockIdx.x; tile < total_tiles; tile += gridDim.x)
                {
                    if (tile < qkv_tiles)
                    {
                        if constexpr (GeneralCodebooks)
                        {
                            completeGeneralTile32<true>(
                                args.qkv, tile << 5, shared.p2.act_codes, shared.p2.act_scales,
                                shared.p2.partials, args.conv_weight, args.conv_bias,
                                args.conv_state, args.gdn_recurrence_in);
                        }
                        else
                        {
                            completeQ8Tile32<true>(
                                args.qkv, tile << 5, shared.p2.act_codes, shared.p2.act_scales,
                                shared.p2.partials, args.conv_weight, args.conv_bias,
                                args.conv_state, args.gdn_recurrence_in);
                        }
                    }
                    else
                    {
                        if constexpr (GeneralCodebooks)
                        {
                            completeGeneralTile32<false>(
                                args.z, (tile - qkv_tiles) << 5, shared.p2.act_codes, shared.p2.act_scales,
                                shared.p2.partials);
                        }
                        else
                        {
                            completeQ8Tile32<false>(
                                args.z, (tile - qkv_tiles) << 5, shared.p2.act_codes, shared.p2.act_scales,
                                shared.p2.partials);
                        }
                    }
                }
            }

            // Tail-shadow FP16RNE Alpha & Beta rows on CTAs with spare capacity (e.g. 40 CTAs on TP2).
            if (h_begin < h_end)
            {
                for (unsigned idx = tid; idx < kHidden; idx += kThreads)
                    shared.p2.mixed_fp16[idx] = __float2half_rn(mixed[idx]);
                __syncthreads();
                float *partials_flat = &shared.p2.partials[0][0];
                for (unsigned h = h_begin; h < h_end; ++h)
                {
                    const bool is_beta = h >= static_cast<unsigned>(args.half.heads);
                    const unsigned row = is_beta ? (h - static_cast<unsigned>(args.half.heads)) : h;
                    const std::uint16_t *w_row = (is_beta ? args.half.beta_weights : args.half.alpha_weights) +
                        static_cast<std::size_t>(row) * kHidden;
                    float *out_ptr = (is_beta ? args.half.beta_output : args.half.alpha_output) + row;
                    completeHalfControlRow(shared.p2.mixed_fp16, w_row, out_ptr, partials_flat);
                }
            }
        }
        grid.sync();

        // -----------------------------------------------------------------
        // Phase 3: Exact CUDA row-split GDN recurrence + last-chunk inline
        //          GatedRMSNorm + blockwise Q8 staging.
        //
        // For ValueMajorKeyContiguous ([128 V][128 K]), each warp owns 2
        // contiguous K-rows (16 V rows/CTA -> 8 chunks/head -> 192 CTAs on
        // TP2) and streams S[vi, 0..127] via 100% coalesced 512B float4
        // warp transactions (0 wasted sectors, 0 inner __syncthreads())
        // while preserving the exact 8-split x 16-element ascending FMA
        // order and 0..7 fold of serial decode.
        // -----------------------------------------------------------------
        {
            constexpr float kQueryScale = 0x1.6a09e6p-4f; // Exact 1/sqrt(128) FP32 constant.
            const int n_v = args.n_v_heads;
            const int n_k = args.n_k_heads;
            constexpr unsigned kChunksPerHead =
                (StateLayout == GDNRecurrenceStateLayout::ValueMajorKeyContiguous) ? 8u : 4u;
            const unsigned total_chunks = static_cast<unsigned>(n_v) * kChunksPerHead;
            const std::size_t key_span = static_cast<std::size_t>(n_k) * 128u;
            const int split_id = static_cast<int>(tid >> 5);
            const int col_in_block = static_cast<int>(tid & 31u);
            const int row_begin = split_id * 16;

            for (unsigned chunk = blockIdx.x; chunk < total_chunks; chunk += gridDim.x)
            {
                const int h = static_cast<int>(chunk / kChunksPerHead);
                const int c = static_cast<int>(chunk % kChunksPerHead);

                int source_head = (h + args.global_v_offset) % n_k;
                if (source_head < 0)
                    source_head += n_k;

                if (tid < 128u)
                {
                    const float *q_head =
                        args.gdn_recurrence_in + static_cast<std::size_t>(source_head) * 128u;
                    const float *k_head =
                        args.gdn_recurrence_in + key_span + static_cast<std::size_t>(source_head) * 128u;
                    shared.p3.q_local[tid] = q_head[tid];
                    shared.p3.k_local[tid] = k_head[tid];
                }
                else if (tid == 128u)
                {
                    const float x = args.half.alpha_output[h] + args.dt_bias[h];
                    const float sp = (x > 20.0f) ? x : log1pf(expf(x));
                    shared.p3.decay = expf(args.A_log[h] * sp);
                    shared.p3.beta_h = 1.0f / (1.0f + expf(-args.half.beta_output[h]));
                }
                __syncthreads();

                if (args.use_qk_l2norm)
                {
                    if (warp == 0u)
                    {
                        float q_sum = 0.0f;
                        for (int i = static_cast<int>(lane); i < 128; i += 32)
                            q_sum += shared.p3.q_local[i] * shared.p3.q_local[i];
#pragma unroll
                        for (int off = 16; off > 0; off >>= 1)
                            q_sum += __shfl_xor_sync(0xffffffffu, q_sum, off);
                        if (lane == 0u)
                            shared.p3.warp_sums[0] = q_sum;
                    }
                    else if (warp == 1u)
                    {
                        float k_sum = 0.0f;
                        for (int i = static_cast<int>(lane); i < 128; i += 32)
                            k_sum += shared.p3.k_local[i] * shared.p3.k_local[i];
#pragma unroll
                        for (int off = 16; off > 0; off >>= 1)
                            k_sum += __shfl_xor_sync(0xffffffffu, k_sum, off);
                        if (lane == 0u)
                            shared.p3.warp_sums[1] = k_sum;
                    }
                    __syncthreads();
                    if (tid < 128u)
                    {
                        const float q_inv = kQueryScale / sqrtf(shared.p3.warp_sums[0] + 1e-6f);
                        const float k_inv = 1.0f / sqrtf(shared.p3.warp_sums[1] + 1e-6f);
                        shared.p3.q_local[tid] *= q_inv;
                        shared.p3.k_local[tid] *= k_inv;
                    }
                    __syncthreads();
                }
                else
                {
                    if (tid < 128u)
                        shared.p3.q_local[tid] *= kQueryScale;
                    __syncthreads();
                }

                const float decay = shared.p3.decay;
                const float beta_h = shared.p3.beta_h;
                float *S_head = args.recurrence_state + static_cast<std::size_t>(h) * 128u * 128u;

                if constexpr (StateLayout == GDNRecurrenceStateLayout::ValueMajorKeyContiguous)
                {
                    const int vi0 = c * 16 + static_cast<int>(warp) * 2;
                    const int vi1 = vi0 + 1;
                    const int sp = static_cast<int>(lane >> 2);
                    const unsigned sub = lane & 3u;
                    const int rbase = sp * 16;
                    const bool aligned16 = (reinterpret_cast<std::uintptr_t>(S_head) & 15u) == 0u;
                    const float4 kv4 = *reinterpret_cast<const float4 *>(shared.p3.k_local + lane * 4u);

                    const auto load_row = [&](int vi) -> float4 {
                        const float *src = S_head + static_cast<std::size_t>(vi) * 128u + lane * 4u;
                        return aligned16 ? *reinterpret_cast<const float4 *>(src)
                                         : make_float4(src[0], src[1], src[2], src[3]);
                    };
                    const auto store_row = [&](int vi, float4 v) {
                        float *dst = S_head + static_cast<std::size_t>(vi) * 128u + lane * 4u;
                        if (aligned16)
                            *reinterpret_cast<float4 *>(dst) = v;
                        else
                        {
                            dst[0] = v.x;
                            dst[1] = v.y;
                            dst[2] = v.z;
                            dst[3] = v.w;
                        }
                    };

                    // Issue both 512-byte coalesced warp loads up front to hide GDDR6X latency.
                    float4 sv0 = load_row(vi0);
                    float4 sv1 = load_row(vi1);

                    const auto finish_vi = [&](int vi, float4 sv) {
                        sv.x *= decay;
                        sv.y *= decay;
                        sv.z *= decay;
                        sv.w *= decay;

                        float pkv = 0.0f;
                        if (sub == 0u)
                        {
                            pkv += sv.x * shared.p3.k_local[rbase + 0];
                            pkv += sv.y * shared.p3.k_local[rbase + 1];
                            pkv += sv.z * shared.p3.k_local[rbase + 2];
                            pkv += sv.w * shared.p3.k_local[rbase + 3];
                        }
#pragma unroll
                        for (unsigned off = 1u; off < 4u; ++off)
                        {
                            const float tx = __shfl_down_sync(0xffffffffu, sv.x, off, 4);
                            const float ty = __shfl_down_sync(0xffffffffu, sv.y, off, 4);
                            const float tz = __shfl_down_sync(0xffffffffu, sv.z, off, 4);
                            const float tw = __shfl_down_sync(0xffffffffu, sv.w, off, 4);
                            if (sub == 0u)
                            {
                                const int kb = rbase + static_cast<int>(off) * 4;
                                pkv += tx * shared.p3.k_local[kb + 0];
                                pkv += ty * shared.p3.k_local[kb + 1];
                                pkv += tz * shared.p3.k_local[kb + 2];
                                pkv += tw * shared.p3.k_local[kb + 3];
                            }
                        }

                        float kv = 0.0f;
#pragma unroll
                        for (int s = 0; s < 8; ++s)
                        {
                            const float ps = __shfl_sync(0xffffffffu, pkv, s * 4);
                            if (lane == 0u)
                                kv += ps;
                        }
                        float delta = 0.0f;
                        if (lane == 0u)
                        {
                            const float v_val =
                                args.gdn_recurrence_in[2u * key_span + static_cast<std::size_t>(h) * 128u + vi];
                            delta = (v_val - kv) * beta_h;
                        }
                        delta = __shfl_sync(0xffffffffu, delta, 0);

                        sv.x = sv.x + kv4.x * delta;
                        sv.y = sv.y + kv4.y * delta;
                        sv.z = sv.z + kv4.z * delta;
                        sv.w = sv.w + kv4.w * delta;
                        store_row(vi, sv);

                        float pout = 0.0f;
                        if (sub == 0u)
                        {
                            pout += sv.x * shared.p3.q_local[rbase + 0];
                            pout += sv.y * shared.p3.q_local[rbase + 1];
                            pout += sv.z * shared.p3.q_local[rbase + 2];
                            pout += sv.w * shared.p3.q_local[rbase + 3];
                        }
#pragma unroll
                        for (unsigned off = 1u; off < 4u; ++off)
                        {
                            const float tx = __shfl_down_sync(0xffffffffu, sv.x, off, 4);
                            const float ty = __shfl_down_sync(0xffffffffu, sv.y, off, 4);
                            const float tz = __shfl_down_sync(0xffffffffu, sv.z, off, 4);
                            const float tw = __shfl_down_sync(0xffffffffu, sv.w, off, 4);
                            if (sub == 0u)
                            {
                                const int qb = rbase + static_cast<int>(off) * 4;
                                pout += tx * shared.p3.q_local[qb + 0];
                                pout += ty * shared.p3.q_local[qb + 1];
                                pout += tz * shared.p3.q_local[qb + 2];
                                pout += tw * shared.p3.q_local[qb + 3];
                            }
                        }

                        float out_vi = 0.0f;
#pragma unroll
                        for (int s = 0; s < 8; ++s)
                        {
                            const float ps = __shfl_sync(0xffffffffu, pout, s * 4);
                            if (lane == 0u)
                                out_vi += ps;
                        }
                        if (lane == 0u)
                            args.attn_output[h * 128 + vi] = out_vi;
                    };

                    finish_vi(vi0, sv0);
                    finish_vi(vi1, sv1);
                }
                else
                {
                    const int vi = c * 32 + col_in_block;
                    float state_hi[8];
                    loadGDNStateSplit<StateLayout>(S_head, shared.p3.state_lo, state_hi, row_begin, vi, tid);

                    float partial_kv = 0.0f;
#pragma unroll
                    for (int r = 0; r < 8; ++r)
                    {
                        const float s = shared.p3.state_lo[r][tid] * decay;
                        shared.p3.state_lo[r][tid] = s;
                        partial_kv += s * shared.p3.k_local[row_begin + r];
                    }
#pragma unroll
                    for (int r = 0; r < 8; ++r)
                    {
                        state_hi[r] *= decay;
                        partial_kv += state_hi[r] * shared.p3.k_local[row_begin + 8 + r];
                    }
                    shared.p3.reduce_kv[tid] = partial_kv;
                    __syncthreads();

                    float kv = 0.0f;
#pragma unroll
                    for (int sp_idx = 0; sp_idx < 8; ++sp_idx)
                        kv += shared.p3.reduce_kv[col_in_block + sp_idx * 32];
                    const float v_val =
                        args.gdn_recurrence_in[2u * key_span + static_cast<std::size_t>(h) * 128u + vi];
                    const float delta = (v_val - kv) * beta_h;

                    float partial_out = 0.0f;
#pragma unroll
                    for (int r = 0; r < 8; ++r)
                    {
                        const int sr = row_begin + r;
                        const float s_new = shared.p3.state_lo[r][tid] + shared.p3.k_local[sr] * delta;
                        shared.p3.state_lo[r][tid] = s_new;
                        partial_out += s_new * shared.p3.q_local[sr];
                    }
#pragma unroll
                    for (int r = 0; r < 8; ++r)
                    {
                        const int sr = row_begin + 8 + r;
                        const float s_new = state_hi[r] + shared.p3.k_local[sr] * delta;
                        state_hi[r] = s_new;
                        partial_out += s_new * shared.p3.q_local[sr];
                    }
                    shared.p3.reduce_out[tid] = partial_out;
                    storeGDNStateSplit<StateLayout>(S_head, shared.p3.state_lo, state_hi, row_begin, vi, tid);
                    __syncthreads();

                    if (split_id == 0)
                    {
                        float out_vi = 0.0f;
#pragma unroll
                        for (int sp_idx = 0; sp_idx < 8; ++sp_idx)
                            out_vi += shared.p3.reduce_out[col_in_block + sp_idx * 32];
                        args.attn_output[h * 128 + vi] = out_vi;
                    }
                }

                // Publish attn_output to L2 before signaling head chunk completion.
                __threadfence();
                __syncthreads();
                if (tid == 0u)
                {
                    auto *head_done = reinterpret_cast<std::uint32_t *>(
                        static_cast<unsigned char *>(args.scratch) + kScratchHeadDoneOffset);
                    shared.p3.finished = atomicAdd(head_done + h, 1u) + 1u;
                }
                __syncthreads();

                if (shared.p3.finished == kChunksPerHead)
                {
                    // Exact 128-thread (4-warp) cuda_gated_rmsnorm_kernel + blockwise Q8 pack.
                    const unsigned head_off = static_cast<unsigned>(h) * 128u;
                    float val = 0.0f;
                    if (tid < 128u)
                    {
                        val = args.attn_output[head_off + tid];
                        float sum_sq = val * val;
#pragma unroll
                        for (int off = 16; off > 0; off >>= 1)
                            sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, off);
                        if (lane == 0u)
                            shared.p3.warp_sums[warp] = sum_sq;
                    }
                    __syncthreads();
                    if (tid == 0u)
                    {
                        float total = 0.0f;
                        for (int w = 0; w < 4; ++w)
                            total += shared.p3.warp_sums[w];
                        shared.p3.warp_sums[0] = total;
                    }
                    __syncthreads();

                    if (tid < 128u)
                    {
                        const float inv_rms = rsqrtf(shared.p3.warp_sums[0] / 128.0f + args.ssm_norm_eps);
                        const float normalized = val * inv_rms;
                        const float gamma_eff = args.ssm_norm_subtract_one
                            ? (1.0f + args.ssm_norm_gamma[tid]) : args.ssm_norm_gamma[tid];
                        const float gate_val = args.z.output[head_off + tid];
                        float gate_act;
                        if (args.ssm_norm_activation == GatedNormActivation::SiLU)
                            gate_act = gate_val / (1.0f + expf(-gate_val));
                        else if (args.ssm_norm_activation == GatedNormActivation::Sigmoid)
                            gate_act = 1.0f / (1.0f + expf(-gate_val));
                        else
                            gate_act = gate_val;
                        const float out_val = normalized * gamma_eff * gate_act;
                        args.attn_output[head_off + tid] = out_val;

                        auto *scratch_base = static_cast<unsigned char *>(args.scratch);
                        auto *out_q8_codes = reinterpret_cast<int8_t *>(scratch_base + kScratchCodesOffset);
                        auto *out_q8_scales = reinterpret_cast<float *>(scratch_base + kScratchScalesOffset);
                        const unsigned blk = static_cast<unsigned>(h) * 4u + warp;
                        quantizeBlockwiseWarp32(out_val, out_q8_codes + blk * 32u,
                            out_q8_scales + blk, lane);
                    }
                    __syncthreads();
                }
            }
        }
        grid.sync();

        // -----------------------------------------------------------------
        // Phase 4: Pull Q8 gated-norm output into SMEM, then zero-imbalance
        //          16-column ssm_out tiles (160 tiles in 1 single wave!).
        // -----------------------------------------------------------------
        {
            auto *scratch_base = static_cast<unsigned char *>(args.scratch);
            const auto *out_q8_codes = reinterpret_cast<const int8_t *>(scratch_base + kScratchCodesOffset);
            const auto *out_q8_scales = reinterpret_cast<const float *>(scratch_base + kScratchScalesOffset);
            const unsigned value_dim = static_cast<unsigned>(args.n_v_heads) * 128u;
            const unsigned v_blocks = value_dim >> 5;
            const auto *src_v4 = reinterpret_cast<const int4 *>(out_q8_codes);
            auto *dst_v4 = reinterpret_cast<int4 *>(shared.p4.act_codes);
            const unsigned code_v4s = value_dim >> 4;
            for (unsigned i = tid; i < code_v4s; i += kThreads)
                dst_v4[i] = src_v4[i];
            for (unsigned b = tid; b < v_blocks; b += kThreads)
                shared.p4.act_scales[b] = out_q8_scales[b];
            __syncthreads();

            const unsigned ssm_tiles = (static_cast<unsigned>(args.ssm_out.weights.n) + 15u) >> 4;
            for (unsigned tile = blockIdx.x; tile < ssm_tiles; tile += gridDim.x)
            {
                if constexpr (GeneralCodebooks)
                {
                    completeGeneralTile16(args.ssm_out, tile << 4,
                        shared.p4.act_codes, shared.p4.act_scales, shared.p4.partials);
                }
                else
                {
                    completeQ8Tile16(args.ssm_out, tile << 4,
                        shared.p4.act_codes, shared.p4.act_scales, shared.p4.partials);
                }
            }
        }
    }

    /** @brief Return whether codebook_id is one of the 18 compiled NativeQuantised decoders. */
    bool isSupportedNativeCodebook(std::uint8_t cb) noexcept
    {
        switch (cb)
        {
        case 0: case 4: case 5: case 6: case 7: case 8: case 9: case 10:
        case 11: case 12: case 13: case 14: case 15: case 16: case 17: case 19:
        case kNativeQuantisedExpandedInt8MinCodebook:
        case kNativeQuantisedIQ4XSSourceScaleCodebook:
            return true;
        default:
            return false;
        }
    }

    /** @brief Return whether a prepared matrix descriptor is canonical non-superblock Q8_0. */
    bool isCanonicalQ8Matrix(const DeviceNativeQuantisedMatrixDesc &w) noexcept
    {
        return w.codebook_id == 19 && w.source_identity_present &&
               w.source_codebook_id == 19 && !w.source_is_superblock &&
               !w.mins && !w.emins;
    }

    /** @brief Validate one NativeQuantised projection descriptor and its exact M1 K-partition schedule. */
    bool validNativeProjection(const CUDARetainedGDNProjection &p, int expected_n, int expected_k) noexcept
    {
        if (!p.output || expected_n <= 0 || expected_k <= 0 || (expected_k % 32) != 0)
            return false;
        const auto &w = p.weights;
        const int blocks = expected_k / 32;
        return w.valid() && w.n == expected_n && w.k == expected_k &&
               w.blocks_per_row == static_cast<unsigned>(blocks) &&
               w.source_identity_present && isSupportedNativeCodebook(w.codebook_id) &&
               p.k_partitions >= 1 && p.k_partitions <= static_cast<int>(kMaxKPartitions) &&
               p.k_partitions <= blocks &&
               p.blocks_per_partition == (blocks + p.k_partitions - 1) / p.k_partitions &&
               (p.ordered_kpar || p.k_partitions == 1);
    }
} // namespace

std::size_t retainedCUDAGDNScratchBytes() noexcept
{
    return kScratchTotalBytes;
}

CUDARetainedGDNAdmission prepareCUDARetainedGDN(
    DeviceId device,
    const CUDARetainedGDNInvocation &invocation,
    std::uint64_t workspace_generation)
{
    if (!device.is_cuda())
        throw std::invalid_argument("CUDARetainedGDN requires a CUDA device");
    const auto &hardware = cudaHardwareMetadata().at(device.ordinal);
    const auto &hc = invocation.hc;
    const auto valid_hc_native = [](const DeviceNativeQuantisedMatrixDesc &d, int n, int k) {
        return d.valid() && d.n == n && d.k == k &&
               d.blocks_per_row == static_cast<unsigned>(k / 32) &&
               d.source_identity_present && isSupportedNativeCodebook(d.codebook_id);
    };

    if (hc.norm.extent.rows != 1 || hc.norm.extent.liveRows != nullptr ||
        hc.up.rows.capacity() != 1 || hc.up.rows.countOwner() != nullptr ||
        hc.down_publication.rows.capacity() != 1 || hc.down_publication.rows.countOwner() != nullptr ||
        hc.norm.hidden != static_cast<int>(kHidden) || hc.down_publication.columns != static_cast<int>(kLowRank) ||
        !valid_hc_native(hc.down, static_cast<int>(kLowRank), 4 * static_cast<int>(kHidden)) ||
        !valid_hc_native(hc.up.weights, 4 * static_cast<int>(kHidden), static_cast<int>(kLowRank)) ||
        !std::isfinite(hc.norm.epsilon) || hc.norm.epsilon <= 0.0f ||
        !hc.norm.input || !hc.norm.gain || !hc.norm.output ||
        !hc.norm_codes || !hc.norm_scales || !hc.raw_down || !hc.mixed ||
        !hc.up.primary || !hc.up.residual || !hc.up.primaryScales || !hc.up.residualScales ||
        hc.up.quantizer != Q8ResidualQuantizerPolicy::CUDAFloatScaleNativeReciprocal ||
        (static_cast<bool>(hc.injection_weights) != static_cast<bool>(hc.injection_output)))
    {
        throw std::invalid_argument("CUDARetainedGDN has invalid HC prefix operands");
    }

    if (hc.has_previous_write)
    {
        if (hc.write.extent.rows != 1 || hc.write.extent.liveRows != nullptr ||
            hc.write.hidden != static_cast<int>(kHidden) ||
            !hc.write.residual || !hc.write.block || !hc.write.gateLogits ||
            !hc.write.output || hc.norm.input != hc.write.output)
            throw std::invalid_argument("CUDARetainedGDN has invalid HC previous-write bindings");
    }

    constexpr int hc_packets = (4 * static_cast<int>(kHidden)) / 32;
    if (hc.down_partitions.count <= 0 || hc.down_partitions.count > hc_packets ||
        hc.down_partitions.blocks_per_partition != (hc_packets + hc.down_partitions.count - 1) / hc.down_partitions.count ||
        (hc.down_fold == moe::HCDownFold::Direct ? hc.down_partitions.count != 1 :
            hc.down_fold != moe::HCDownFold::OrderedPartitions || !hc.down_partials))
    {
        throw std::invalid_argument("CUDARetainedGDN has invalid HC Down partition schedule");
    }

    const int n_k = invocation.n_k_heads;
    const int n_v = invocation.n_v_heads;
    const int expected_qkv = 2 * n_k * 128 + n_v * 128;
    const int expected_z = n_v * 128;
    if (n_k <= 0 || n_v < 1 || static_cast<unsigned>(n_v) > kMaxCUDARetainedGDNVHeads ||
        invocation.half.heads != n_v ||
        !validNativeProjection(invocation.qkv, expected_qkv, static_cast<int>(kHidden)) ||
        !validNativeProjection(invocation.z, expected_z, static_cast<int>(kHidden)) ||
        !validNativeProjection(invocation.ssm_out, static_cast<int>(kHidden), expected_z) ||
        !invocation.half.alpha_weights || !invocation.half.beta_weights ||
        !invocation.half.alpha_output || !invocation.half.beta_output ||
        invocation.half.alpha_output == invocation.half.beta_output ||
        !invocation.conv_weight || !invocation.conv_state || !invocation.gdn_recurrence_in ||
        !invocation.A_log || !invocation.dt_bias || !invocation.recurrence_state || !invocation.attn_output ||
        !invocation.ssm_norm_gamma || !std::isfinite(invocation.ssm_norm_eps) || invocation.ssm_norm_eps <= 0.0f ||
        !validGatedNormActivation(invocation.ssm_norm_activation) ||
        !validGDNRecurrenceStateLayout(invocation.state_layout) ||
        !invocation.scratch || invocation.scratch_bytes < kScratchTotalBytes ||
        (reinterpret_cast<std::uintptr_t>(invocation.scratch) & 15u) != 0u)
    {
        throw std::invalid_argument("CUDARetainedGDN has invalid GDN operands or scratch");
    }

    const bool all_q8 = isCanonicalQ8Matrix(hc.down) && isCanonicalQ8Matrix(hc.up.weights) &&
                        isCanonicalQ8Matrix(invocation.qkv.weights) &&
                        isCanonicalQ8Matrix(invocation.z.weights) &&
                        isCanonicalQ8Matrix(invocation.ssm_out.weights);
    const bool value_major = (invocation.state_layout == GDNRecurrenceStateLayout::ValueMajorKeyContiguous);
    const void *entry = all_q8
        ? (value_major
               ? reinterpret_cast<const void *>(&cudaRetainedGDNKernel<GDNRecurrenceStateLayout::ValueMajorKeyContiguous, false>)
               : reinterpret_cast<const void *>(&cudaRetainedGDNKernel<GDNRecurrenceStateLayout::KeyMajorValueContiguous, false>))
        : (value_major
               ? reinterpret_cast<const void *>(&cudaRetainedGDNKernel<GDNRecurrenceStateLayout::ValueMajorKeyContiguous, true>)
               : reinterpret_cast<const void *>(&cudaRetainedGDNKernel<GDNRecurrenceStateLayout::KeyMajorValueContiguous, true>));

    // Reserve up to 8 SMs for co-resident graph services and collectives so cooperative grid.sync()
    // never starves or deadlocks under multi-rank/multi-stream overlap.
    const auto co_residency = NativeCoResidencyReservation::reserveProcessors(
        hardware.sm_count > 16 ? 8u : (hardware.sm_count > 4 ? 4u : (hardware.sm_count > 2 ? 2u : 0u)));
    const auto admitted = NativeCooperativeKernelBoundary<NativeCooperativeBackend::CUDA>::prepare(
        device, entry, kThreads, 0u, workspace_generation,
        NativeResidentLaunchPolicy::Cooperative, co_residency);

    if (admitted.resources().private_bytes != 0u)
        throw std::invalid_argument("CUDARetainedGDN requires zero register spills");

    const unsigned grid = admitted.maximumGrid();
    if (grid < 4u)
        throw std::invalid_argument("CUDARetainedGDN requires at least 4 resident CTAs");

    return {invocation, admitted, grid};
}

void launchCUDARetainedGDN(
    const CUDARetainedGDNAdmission &admission,
    void *stream,
    std::uint64_t workspace_generation)
{
    const auto native = static_cast<cudaStream_t>(stream);
    if (!native || native == cudaStreamLegacy || native == cudaStreamPerThread ||
        workspace_generation != admission.native.generation() || admission.grid < 4u)
    {
        throw std::invalid_argument("CUDARetainedGDN received invalid stream, grid, or stale workspace generation");
    }
    auto args = admission.invocation;
    void *params[]{&args};
    NativeCooperativeKernelBoundary<NativeCooperativeBackend::CUDA>::launch(
        admission.native, params, admission.grid, native, workspace_generation);
}
} // namespace llaminar2::cuda::qwen4exp
