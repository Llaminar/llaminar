/**
 * @file ROCmFlashAttentionKernels.hip
 * @brief HIP device kernels for Flash Attention 2 and Flash Decoding on MI50
 *
 * This file contains the HIP device kernels and extern "C" wrapper functions
 * called from the C++ implementation file.
 *
 * Target Architecture: AMD MI50 (gfx906 / Vega 20)
 *
 * Algorithms implemented:
 * - Flash Attention 2: Tiled attention with online softmax
 *   - Uses v_dot2_f32_f16 for native FP16 dot products with FP32 accumulation
 *   - LDS double-buffering for overlapped load/compute
 *   - Explicit wavefront reductions via DPP
 *   - NO WMMA/Matrix Cores - pure VALU implementation
 * - Flash Decoding: Split-K parallelism for single-token decode
 *   - Ordinary and grouped rows share the same ordered parallel reduction,
 *     including deterministic mode; there is no serial-only policy branch.
 *
 * MI50 Optimization Strategy:
 *   1. Maximize LDS utilization (64KB per CU)
 *   2. Use FP16 packed math where possible for 2x ALU throughput
 *   3. Wavefront-wide reductions using __shfl_xor (DPP under the hood)
 *   4. Balance VGPR usage for occupancy (256 VGPRs per wavefront limit)
 *   5. Explicit prefetching via buffer load instructions
 *
 * @author David Sanftenberg
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>
#include "../../attention/AttentionDeviceParams.h"
#include "ROCmFlashAttentionLaunchPolicy.h"
#include "../../../tensors/BlockStructures.h"
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <limits>

#include "../../../backends/rocm/HipDeviceGuard.h"
#include "../../../utils/DebugEnv.h"

namespace
{
    // =========================================================================
    // Constants - MI50 Specific
    // =========================================================================

    // Wavefront size on GCN/CDNA architectures
    constexpr int WAVEFRONT_SIZE = 64;

    // LDS (Local Data Share) size per CU on MI50
    constexpr int LDS_SIZE_BYTES = 65536; // 64KB

    // Maximum VGPRs per wavefront (for occupancy tuning)
    constexpr int MAX_VGPRS_PER_WAVEFRONT = 256;

    /// Maximum compact verifier rows represented by DEVICE_PARAMS.
    constexpr int MAX_DYNAMIC_ATTENTION_PARAM_ROWS =
        llaminar2::attention::kMaxGroupedVerifierAttentionRows;

    // =========================================================================
    // KV Cache Type Tags for Template Dispatch
    // =========================================================================
    //
    // The decode kernel is templated on KV_TYPE so it can read K/V
    // directly in their native storage format.
    //   FP32: K/V are float*    (4 bytes/element) — baseline
    //   FP16: K/V are __half*   (2 bytes/element) — 2× BW savings + fdot2
    //   BF16: K/V are hip_bfloat16* (2 bytes/element) — native FP32 conversion
    //   Q8_1: K/V are Q8_1Block* (36 bytes/32 elements) — ~3.5× BW savings
    enum class KVTypeTag { FP32, FP16, BF16, Q8_1 };

    /**
     * @brief Derive graph-captured attention params from live KV cache count.
     *
     * This tiny helper runs in graph order immediately before attention, after
     * KV append has advanced the device-owned cache count. The resulting
     * `AttentionDeviceParams` block remains device-owned through MTP replay.
     */
    __global__ void hip_derive_attention_params_from_cached_tokens_kernel(
        llaminar2::attention::AttentionDeviceParams *__restrict__ out,
        const int *__restrict__ post_append_cached_tokens,
        int seq_len,
        int query_rows,
        int kv_stride,
        const int *__restrict__ active_query_rows_device,
        const int *__restrict__ ring_head_device,
        int ring_capacity)
    {
        const int row = static_cast<int>(threadIdx.x);
        if (!out || !post_append_cached_tokens || row >= query_rows)
            return;

        const int raw_kv_len = *post_append_cached_tokens;
        const int kv_len = raw_kv_len > 1 ? raw_kv_len : 1;
        const int launch_seq_len = seq_len > 1 ? seq_len : 1;
        const int raw_logical_seq_len = active_query_rows_device
                                            ? *active_query_rows_device
                                            : launch_seq_len;
        const int logical_seq_len =
            raw_logical_seq_len > 1
                ? (raw_logical_seq_len < launch_seq_len
                       ? raw_logical_seq_len
                       : launch_seq_len)
                : 1;
        int ring_row_origin = 0;
        if (ring_head_device && ring_capacity > 0)
        {
            ring_row_origin = (*ring_head_device - kv_len) % ring_capacity;
            if (ring_row_origin < 0)
                ring_row_origin += ring_capacity;
        }

        /*
         * Ordinary prefill and M=1 decode share one parameter record across the
         * launch.  The attention kernel owns per-query causal masking, so this
         * record exposes the full resident KV span and identifies only the first
         * query's absolute position.  Multiple records instead denote grouped
         * verifier rows and intentionally shorten the visible prefix per row.
         */
        if (query_rows == 1)
        {
            out[0].kv_len = kv_len;
            out[0].kv_stride = kv_stride > kv_len ? kv_stride : kv_len;
            const int base_position = kv_len - logical_seq_len;
            out[0].position_offset = base_position > 0 ? base_position : 0;
            out[0].mask_stride = kv_len;
            out[0].ring_row_origin = ring_row_origin;
            out[0].ring_row_capacity = ring_head_device ? ring_capacity : 0;
            return;
        }

        const int grouped_logical_seq_len =
            logical_seq_len < query_rows ? logical_seq_len : query_rows;
        if (row >= grouped_logical_seq_len)
        {
            out[row].kv_len = 1;
            out[row].kv_stride = kv_stride > kv_len ? kv_stride : kv_len;
            out[row].position_offset = 0;
            out[row].mask_stride = 1;
            out[row].ring_row_origin = ring_row_origin;
            out[row].ring_row_capacity = ring_head_device ? ring_capacity : 0;
            return;
        }
        const int base_position_raw = kv_len - grouped_logical_seq_len;
        const int base_position = base_position_raw > 0 ? base_position_raw : 0;
        const int row_kv_len_raw =
            kv_len - (grouped_logical_seq_len - 1 - row);
        const int row_kv_len = row_kv_len_raw > 1 ? row_kv_len_raw : 1;
        out[row].kv_len = row_kv_len;
        out[row].kv_stride = kv_stride > kv_len ? kv_stride : kv_len;
        out[row].position_offset = base_position + row;
        out[row].mask_stride = kv_len;
        out[row].ring_row_origin = ring_row_origin;
        out[row].ring_row_capacity = ring_head_device ? ring_capacity : 0;
    }

    /**
     * @brief Materialize explicit attention geometry directly in device memory.
     *
     * This is the stream-owned counterpart to the device-count path above for
     * eager tests and callers that already possess fixed launch geometry.  A
     * tiny HIP kernel writes the canonical parameter records, so capture never
     * depends on a pinned host allocation or an H2D parameter transfer.
     *
     * Earlier verifier rows expose successively shorter KV prefixes, matching
     * independent serial decode calls byte for byte.  Absolute positions still
     * advance from @p position_offset and all rows share the full KV mask
     * stride supplied by @p kv_len.
     */
    __global__ void hip_write_attention_params_from_geometry_kernel(
        llaminar2::attention::AttentionDeviceParams *__restrict__ out,
        int kv_len,
        int kv_stride,
        int position_offset,
        int query_rows)
    {
        const int row = static_cast<int>(threadIdx.x);
        if (!out || row >= query_rows)
            return;

        const int row_kv_len_raw = kv_len - (query_rows - 1 - row);
        const int row_kv_len = row_kv_len_raw > 1 ? row_kv_len_raw : 1;
        out[row].kv_len = row_kv_len;
        out[row].kv_stride = kv_stride > kv_len ? kv_stride : kv_len;
        out[row].position_offset = position_offset + row;
        out[row].mask_stride = kv_len;
        out[row].ring_row_origin = 0;
        out[row].ring_row_capacity = 0;
    }

    /**
     * @brief Derive request-major attention metadata from canonical device counts.
     *
     * The resident request cache exposes one post-append token count per
     * request.  A request may contribute several compact verifier rows, so an
     * earlier row must see a shorter prefix than the final row.  Flattening the
     * request/query coordinates here keeps that visibility rule entirely on
     * the execution stream and makes the operation legal inside HIP graph
     * capture and replay.
     *
     * @param out Device workspace containing one parameter record per flat row.
     * @param post_append_cached_tokens Contiguous cache-owned request counts.
     * @param request_count Number of independent request cache banks.
     * @param query_rows Number of consecutive decode rows owned by each request.
     */
    __global__ void hip_derive_attention_params_from_request_counts_kernel(
        llaminar2::attention::AttentionDeviceParams *__restrict__ out,
        const int *__restrict__ post_append_cached_tokens,
        int request_count,
        int query_rows,
        int kv_stride)
    {
        const int flat_row = static_cast<int>(threadIdx.x);
        const int total_rows = request_count * query_rows;
        if (!out || !post_append_cached_tokens || flat_row >= total_rows)
            return;

        const int request = flat_row / query_rows;
        const int query_row = flat_row - request * query_rows;
        const int raw_post_append_kv_len = post_append_cached_tokens[request];
        const int post_append_kv_len =
            raw_post_append_kv_len > 1 ? raw_post_append_kv_len : 1;
        const int base_position_raw = post_append_kv_len - query_rows;
        const int base_position = base_position_raw > 0 ? base_position_raw : 0;
        const int row_kv_len_raw =
            post_append_kv_len - (query_rows - 1 - query_row);
        const int row_kv_len = row_kv_len_raw > 1 ? row_kv_len_raw : 1;

        out[flat_row].kv_len = row_kv_len;
        out[flat_row].kv_stride =
            kv_stride > post_append_kv_len ? kv_stride : post_append_kv_len;
        out[flat_row].position_offset = base_position + query_row;
        out[flat_row].mask_stride = post_append_kv_len;
        out[flat_row].ring_row_origin = 0;
        out[flat_row].ring_row_capacity = 0;
    }

    // Hardware exponential helper is defined beside the decode implementation
    // and forward-declared here for the context reducer above prefill utilities.
    __device__ __forceinline__ float fast_expf(float x);

    /**
     * @brief Merge two scaled online-softmax terms with one fixed instruction order.
     *
     * gfx906 LLVM can otherwise choose a fused multiply-add in one kernel
     * specialization and two rounded operations in another. Calling this same
     * explicit primitive from direct FA2 and the context reducer makes physical
     * scheduling irrelevant to both output and denominator bytes.
     */
    __device__ __forceinline__ float merge_fa2_scaled_terms(
        float previous,
        float previous_scale,
        float partition,
        float partition_scale)
    {
        const float scaled_previous = previous * previous_scale;
        return fmaf(partition, partition_scale, scaled_previous);
    }

    /**
     * @brief Merge canonical ROCm FA2 context summaries in ascending K/V order.
     *
     * Phase one publishes an unnormalized `(m, l, O)` tuple for every logical
     * 256-key interval. Each lane below owns disjoint output dimensions and
     * traverses every live interval in the same ascending order. Dimension
     * striping changes occupancy only: there are no atomics, cross-workgroup
     * reductions, allocations, transfers, or stream synchronizations.
     *
     * The graph grid is fixed by the captured query geometry. Live partition
     * count comes directly from the device-owned attention parameter record,
     * so stale summary bytes above a shorter replay can never participate.
     *
     * @tparam DIMENSION_WAVEFRONTS Number of independent 64-dimension stripes
     *         assigned to one query-row/head reduction.
     */
    template <int DIMENSION_WAVEFRONTS>
    __global__ __launch_bounds__(256, 2)
    void flash_attention_2_context_reduce_mi50_kernel(
        const float *__restrict__ O_partial,
        const float *__restrict__ m_partial,
        const float *__restrict__ l_partial,
        float *__restrict__ O,
        int batch_size,
        int seq_len,
        int n_heads,
        int head_dim,
        int max_context_partitions,
        int device_direct_partition_limit,
        const llaminar2::attention::AttentionDeviceParams *__restrict__
            device_params)
    {
        constexpr int wavefronts_per_block =
            256 / WAVEFRONT_SIZE;
        static_assert(
            DIMENSION_WAVEFRONTS == 1 ||
                DIMENSION_WAVEFRONTS == 2 ||
                DIMENSION_WAVEFRONTS == 4,
            "MI50 FA2 reducer supports one, two, or four dimension wavefronts");
        static_assert(
            wavefronts_per_block % DIMENSION_WAVEFRONTS == 0,
            "Reducer dimension stripes must divide the physical workgroup");
        constexpr int rows_per_block =
            wavefronts_per_block / DIMENSION_WAVEFRONTS;

        const int live_kv_len = device_params->kv_len;
        const int live_partitions = min(
            max_context_partitions,
            max(
                1,
                (live_kv_len +
                 llaminar2::rocm::fa2_policy::
                     kROCmFA2CanonicalContextPartitionKeys -
                 1) /
                    llaminar2::rocm::fa2_policy::
                        kROCmFA2CanonicalContextPartitionKeys));
        if (live_partitions <= device_direct_partition_limit)
            return;

        const int wavefront = threadIdx.x / WAVEFRONT_SIZE;
        const int lane = threadIdx.x % WAVEFRONT_SIZE;
        const int row_in_block = wavefront / DIMENSION_WAVEFRONTS;
        const int dimension_wavefront =
            wavefront % DIMENSION_WAVEFRONTS;
        const int logical_row_blocks =
            (seq_len + rows_per_block - 1) / rows_per_block;
        const int blocks_per_batch = n_heads * logical_row_blocks;
        const int logical_blocks = batch_size * blocks_per_batch;

        /*
         * Grid width is an economy choice embedded at capture time. Each
         * physical workgroup strides disjoint logical row/head tasks, making
         * the grid persistent without introducing an atomic work queue. The
         * arithmetic performed for one output element is therefore identical
         * for every physical block count.
         */
        for (int logical_block = static_cast<int>(blockIdx.x);
             logical_block < logical_blocks;
             logical_block += static_cast<int>(gridDim.x))
        {
            const int batch = logical_block / blocks_per_batch;
            const int batch_block = logical_block % blocks_per_batch;
            const int head = batch_block / logical_row_blocks;
            const int row_block = batch_block % logical_row_blocks;
            const int query_row =
                row_block * rows_per_block + row_in_block;
            if (query_row >= seq_len)
                continue;

            const size_t scalar_base =
                ((static_cast<size_t>(batch) * seq_len + query_row) *
                     n_heads +
                 head) *
                max_context_partitions;
            float *O_row =
                O +
                ((static_cast<size_t>(batch) * seq_len + query_row) *
                     n_heads +
                 head) *
                    static_cast<size_t>(head_dim);

            for (int dimension =
                     dimension_wavefront * WAVEFRONT_SIZE + lane;
                 dimension < head_dim;
                 dimension += DIMENSION_WAVEFRONTS * WAVEFRONT_SIZE)
            {
                float merged_m = -FLT_MAX;
                float merged_l = 0.0f;
                float merged_output = 0.0f;

                for (int partition = 0;
                     partition < live_partitions;
                     ++partition)
                {
                    const size_t summary_index =
                        scalar_base + static_cast<size_t>(partition);
                    const float partition_l = l_partial[summary_index];
                    if (!(partition_l > 0.0f))
                        continue;

                    const float partition_m = m_partial[summary_index];
                    const float partition_output =
                        O_partial[
                            summary_index * static_cast<size_t>(head_dim) +
                            dimension];
                    if (!(merged_l > 0.0f))
                    {
                        merged_m = partition_m;
                        merged_l = partition_l;
                        merged_output = partition_output;
                        continue;
                    }

                    const float next_m = fmaxf(merged_m, partition_m);
                    const float previous_scale =
                        fast_expf(merged_m - next_m);
                    const float partition_scale =
                        fast_expf(partition_m - next_m);
                    merged_output = merge_fa2_scaled_terms(
                        merged_output,
                        previous_scale,
                        partition_output,
                        partition_scale);
                    merged_l = merge_fa2_scaled_terms(
                        merged_l,
                        previous_scale,
                        partition_l,
                        partition_scale);
                    merged_m = next_m;
                }

                /*
                 * Preserve the direct schedule's exact normalization
                 * expression. A source-level division can select a different
                 * gfx906 rounding sequence from `reciprocal then multiply`,
                 * even though both are numerically close.
                 */
                const float inverse_l =
                    merged_l > 0.0f ? 1.0f / merged_l : 0.0f;
                O_row[dimension] =
                    merged_l > 0.0f
                        ? merged_output * inverse_l
                        : 0.0f;
            }
        }
    }

    // =========================================================================
    // Flash Attention 2 Configuration for MI50
    // =========================================================================
    //
    // Key design decisions for MI50 (no Matrix Cores):
    //   1. Tile sizes chosen to balance LDS usage and occupancy
    //   2. Use vectorized FP16 for Q@K^T computation
    //   3. FP32 accumulation for numerical stability
    //   4. Double-buffered K/V loading to hide memory latency
    //
    // LDS Layout:
    //   Q_tile:    [TILE_Q, head_dim] in FP16
    //   K_buffer:  [2, TILE_KV, head_dim] in FP16 (double-buffered)
    //   V_buffer:  [2, TILE_KV, head_dim] in FP16 (double-buffered)
    //   Scores:    [TILE_Q, TILE_KV] in FP32
    //
    // =========================================================================

    // Tile sizes for prefill kernel
    // Chosen to fit comfortably in 64KB LDS with good occupancy
    constexpr int FA2_TILE_Q = 64;   // Q tile rows
    constexpr int FA2_TILE_KV = 64;  // KV tile rows
    constexpr int FA2_NUM_STAGES = 2; // Double buffering

    // LDS padding to avoid bank conflicts (32 banks, 4-byte granularity)
    // The critical access pattern: threads in a wavefront access DIFFERENT ROWS
    // at the SAME column (e.g., different K rows in the Q@K^T dot product).
    // Bank index = (row × stride_dwords + col) mod 32.
    // To avoid conflicts, stride_dwords must be coprime with 32 (i.e., odd).
    //
    // For FP16: stride_dwords = (head_dim + pad) / 2
    //   head_dim=128: (128+2)/2 = 65 → 65 mod 32 = 1 → gcd(1,32) = 1 ✓
    //   head_dim=64:  (64+2)/2  = 33 → 33 mod 32 = 1 → gcd(1,32) = 1 ✓
    // For FP32: stride_dwords = tile_kv + pad
    //   tile_kv=32: 32+1 = 33 → coprime with 32 ✓
    //   tile_kv=64: 64+1 = 65 → coprime with 32 ✓
    //
    // Previous values (8/4) gave stride_dwords ≡ 4 (mod 32) → gcd(4,32)=4
    // → 8-way bank conflicts on K/V reads in the dot product inner loop!
    constexpr int FA2_LDS_PAD_FP16 = 2;
    constexpr int FA2_LDS_PAD_FP32 = 1;

    // Block configuration
    // We use 256 threads = 4 wavefronts per block for good occupancy
    constexpr int FA2_THREADS_PER_BLOCK = 256;
    constexpr int FA2_WAVEFRONTS_PER_BLOCK = FA2_THREADS_PER_BLOCK / WAVEFRONT_SIZE; // 4

    // =========================================================================
    // Flash Decoding Configuration for MI50
    // =========================================================================

    // Max splits supported by reduction kernel shared-memory staging
    constexpr int FD_MAX_SPLITS_SUPPORTED = 32;

    constexpr int FD_TARGET_ELEMS_PER_SPLIT = 16384;

    // =========================================================================
    // Device Property Caching
    // =========================================================================

    struct MI50DeviceConfig
    {
        int gcn_arch = 0;      // e.g., 906 for gfx906
        int max_lds = 0;       // Max LDS per block
        int max_vgprs = 0;     // Max VGPRs per wavefront
        int num_cus = 60;      // Number of compute units
        bool initialized = false;
    };

    // Per-device cached config
    static MI50DeviceConfig g_device_config[8]; // Support up to 8 GPUs

    /**
     * @brief Get cached device configuration (lazy init)
     */
    inline MI50DeviceConfig &getDeviceConfig(int device)
    {
        if (device < 0 || device >= 8)
            device = 0;
        MI50DeviceConfig &cfg = g_device_config[device];

        if (!cfg.initialized)
        {
            hipDeviceProp_t prop;
            (void)hipGetDeviceProperties(&prop, device);

            // Parse gcnArchName string (e.g., "gfx906") to extract arch number
            // gcnArchName format: "gfx" + arch_number
            cfg.gcn_arch = 0;
            if (strncmp(prop.gcnArchName, "gfx", 3) == 0) {
                cfg.gcn_arch = atoi(prop.gcnArchName + 3);
            }
            cfg.max_lds = static_cast<int>(prop.sharedMemPerBlock);
            cfg.max_vgprs = MAX_VGPRS_PER_WAVEFRONT; // Could query via rocm-smi
            cfg.num_cus = prop.multiProcessorCount;
            cfg.initialized = true;
        }
        return cfg;
    }

    // =========================================================================
    // HIP-specific Intrinsics and Helpers
    // =========================================================================

    /**
     * @brief Wavefront-wide reduction using DPP (Data Parallel Primitives)
     *
     * On AMD GPUs, __shfl_xor maps to DPP instructions for fast cross-lane ops.
     * Wavefront size is 64, so we need 6 steps for full reduction.
     */
    __device__ __forceinline__ float wavefrontReduceSum(float val)
    {
        // Butterfly reduction pattern for wavefront size 64
        for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2)
        {
            val += __shfl_xor(val, offset);
        }
        return val;
    }

    __device__ __forceinline__ float wavefrontReduceMax(float val)
    {
        for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2)
        {
            val = fmaxf(val, __shfl_xor(val, offset));
        }
        return val;
    }

    // exp2f(x * LOG2E) maps to hardware v_exp_f32 on GCN.
    // Forward-declared here so both prefill and decode kernels can use it.
    __device__ __forceinline__ float fast_expf(float x);

    /**
     * @brief Warp-level reduction of softmax state (m, l, O) for Flash Decoding
     *
     * Reduces across all 64 lanes in a wavefront.
     */
    __device__ void wavefrontReduceSoftmaxState(
        float &m, float &l, float *O, int head_dim)
    {
        for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2)
        {
            float other_m = __shfl_xor(m, offset);
            float other_l = __shfl_xor(l, offset);

            float m_new = fmaxf(m, other_m);
            float scale_self = expf(m - m_new);
            float scale_other = expf(other_m - m_new);

            l = scale_self * l + scale_other * other_l;

            for (int d = 0; d < head_dim; d++)
            {
                float other_O = __shfl_xor(O[d], offset);
                O[d] = scale_self * O[d] + scale_other * other_O;
            }

            m = m_new;
        }
    }

    /**
     * @brief Convert FP32 to FP16 using HIP intrinsics
     */
    __device__ __forceinline__ __half float_to_half(float x)
    {
        return __float2half(x);
    }

    /**
     * @brief Convert FP16 to FP32 using HIP intrinsics
     */
    __device__ __forceinline__ float half_to_float(__half x)
    {
        return __half2float(x);
    }

    // =========================================================================
    // Flash Attention 2 - Prefill Kernel for MI50
    // =========================================================================
    //
    // This implementation uses vectorized FP16 operations since MI50 lacks
    // Matrix Cores. Key optimizations:
    //   1. Vectorized FP16 dot products (2 elements at a time)
    //   2. LDS double-buffering for K/V tiles
    //   3. Online softmax to avoid materializing full attention matrix
    //   4. FP32 accumulators for numerical stability
    //
    // Grid: (n_heads, num_q_tiles, batch_size)
    // Block: (256,) threads = 4 wavefronts
    //
    // =========================================================================

    /**
     * @brief Compute LDS layout sizes for given head_dim
     */
    struct FA2LDSLayout
    {
        int q_tile_size;      // Q tile in FP16 with padding
        int kv_tile_size;     // K or V tile in FP16 with padding
        int scores_size;      // Scores tile in FP32 with padding
        int total_size;       // Total LDS bytes
        int q_stride;         // Q leading dimension
        int kv_stride;        // K/V leading dimension
        int scores_stride;    // Scores leading dimension

        __host__ __device__ void compute(int head_dim, int tile_q, int tile_kv)
        {
            // Add padding to avoid bank conflicts
            q_stride = head_dim + FA2_LDS_PAD_FP16;
            kv_stride = head_dim + FA2_LDS_PAD_FP16;
            scores_stride = tile_kv + FA2_LDS_PAD_FP32;

            q_tile_size = tile_q * q_stride * sizeof(__half);
            kv_tile_size = tile_kv * kv_stride * sizeof(__half);
            scores_size = tile_q * scores_stride * sizeof(float);

            // Total: Q + 2*(K+V) for double buffer + scores
            total_size = q_tile_size + FA2_NUM_STAGES * 2 * kv_tile_size + scores_size;
        }
    };

    /**
     * @brief Load one logical KV element and convert it to the FP16 LDS format.
     *
     * The prefill kernel keeps K and V in their native cache representation
     * while they reside in HBM.  Conversion happens only as the live tile is
     * staged into LDS, preserving the bandwidth advantage of FP16 and Q8_1
     * caches without allocating a graph-unstable full-cache conversion buffer.
     *
     * @tparam KV_TYPE Physical cache representation selected at kernel launch.
     * @param source Beginning of the request-major native cache allocation.
     * @param logical_element Dense FP32-equivalent element index.  Q8_1 rows
     *        are block aligned because every supported head dimension is a
     *        multiple of the 32-element Q8_1 block width.
     * @return The requested value rounded to the FP16 arithmetic format used
     *         by the MI50 dot-product pipeline.
     */
    template <KVTypeTag KV_TYPE>
    __device__ __forceinline__ __half load_fa2_kv_element_as_half(
        const void *__restrict__ source,
        size_t logical_element)
    {
        if constexpr (KV_TYPE == KVTypeTag::FP32)
        {
            return float_to_half(static_cast<const float *>(source)[logical_element]);
        }
        else if constexpr (KV_TYPE == KVTypeTag::FP16)
        {
            return static_cast<const __half *>(source)[logical_element];
        }
        else if constexpr (KV_TYPE == KVTypeTag::BF16)
        {
            return float_to_half(
                static_cast<float>(
                    static_cast<const hip_bfloat16 *>(source)[logical_element]));
        }
        else
        {
            constexpr size_t ELEMENTS_PER_BLOCK = llaminar2::Q8_1Block::BLOCK_SIZE;
            const auto *blocks = static_cast<const llaminar2::Q8_1Block *>(source);
            const auto &block = blocks[logical_element / ELEMENTS_PER_BLOCK];
            const int element_in_block =
                static_cast<int>(logical_element % ELEMENTS_PER_BLOCK);
            const float scale = __half2float(
                *reinterpret_cast<const __half *>(&block.d));
            return float_to_half(scale * static_cast<float>(block.qs[element_in_block]));
        }
    }

    /**
     * @brief Load two adjacent KV values in the packed format consumed by fdot2.
     *
     * FP32 and FP16 caches retain vectorized global-memory reads.  Q8_1 uses
     * two scalar block lookups because its scale metadata is shared by 32
     * values; the resulting pair is still packed before entering LDS.
     */
    template <KVTypeTag KV_TYPE>
    __device__ __forceinline__ __half2 load_fa2_kv_pair_as_half2(
        const void *__restrict__ source,
        size_t logical_element)
    {
        if constexpr (KV_TYPE == KVTypeTag::FP32)
        {
            const float2 values = *reinterpret_cast<const float2 *>(
                static_cast<const float *>(source) + logical_element);
            return __float22half2_rn(values);
        }
        else if constexpr (KV_TYPE == KVTypeTag::FP16)
        {
            return *reinterpret_cast<const __half2 *>(
                static_cast<const __half *>(source) + logical_element);
        }
        else if constexpr (KV_TYPE == KVTypeTag::BF16)
        {
            return __halves2half2(
                load_fa2_kv_element_as_half<KV_TYPE>(source, logical_element),
                load_fa2_kv_element_as_half<KV_TYPE>(source, logical_element + 1));
        }
        else
        {
            return __halves2half2(
                load_fa2_kv_element_as_half<KV_TYPE>(source, logical_element),
                load_fa2_kv_element_as_half<KV_TYPE>(source, logical_element + 1));
        }
    }

    /**
     * @brief Flash Attention 2 prefill kernel for MI50
     *
     * Uses vectorized FP16 VALU operations for Q@K^T computation.
     * No WMMA/Matrix Cores - pure ALU implementation optimized for Vega.
     */
    template <KVTypeTag KV_TYPE,
              int THREADS_PER_ROW,
              int STATIC_HEAD_DIM = 0,
              int STATIC_TILE_Q = 0,
              int STATIC_TILE_KV = 0,
              bool WRITE_CONTEXT_PARTIAL = false>
    __global__ __launch_bounds__(256, 2) void flash_attention_2_mi50_kernel(
        const float *__restrict__ Q,
        const void *__restrict__ K,
        const void *__restrict__ V,
        float *__restrict__ O,
        int batch_size,
        int seq_len,
        int kv_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float softmax_scale,
        bool causal,
        int window_size,
        int position_offset,
        const llaminar2::attention::AttentionDeviceParams *__restrict__ device_params,
        const float *__restrict__ mask,
        int tile_q,
        int tile_kv,
        int head_start,
        int gqa_n_rep,
        float *__restrict__ O_partial = nullptr,
        float *__restrict__ m_partial = nullptr,
        float *__restrict__ l_partial = nullptr,
        int context_partition_size = 0,
        int max_context_partitions = 0,
        int context_partition_slots = 0,
        int device_direct_partition_limit = 0)
    {
        const int kernel_head_dim =
            STATIC_HEAD_DIM > 0 ? STATIC_HEAD_DIM : head_dim;
        const int kernel_tile_q =
            STATIC_TILE_Q > 0 ? STATIC_TILE_Q : tile_q;
        const int kernel_tile_kv =
            STATIC_TILE_KV > 0 ? STATIC_TILE_KV : tile_kv;

        int kv_stride = kv_len;
        int kv_len_runtime = kv_len;
        int position_offset_runtime = position_offset;
        int mask_stride = kv_len;
        int ring_row_origin = 0;
        int ring_row_capacity = 0;

        if (device_params)
        {
            kv_len_runtime = device_params->kv_len;
            kv_stride = device_params->kv_stride;
            position_offset_runtime = device_params->position_offset;
            mask_stride = device_params->mask_stride;
            ring_row_origin = device_params->ring_row_origin;
            ring_row_capacity = device_params->ring_row_capacity;
        }

        const int logical_query_tiles =
            (seq_len + kernel_tile_q - 1) / kernel_tile_q;
        const int phase_blocks_per_slot =
            batch_size * logical_query_tiles * n_heads;
        int graph_live_context_partitions = 1;
        int active_context_partition_slots = 1;
        if constexpr (WRITE_CONTEXT_PARTIAL)
        {
            graph_live_context_partitions = min(
                max_context_partitions,
                max(
                    1,
                    (kv_len_runtime + context_partition_size - 1) /
                        context_partition_size));
            if (graph_live_context_partitions <=
                device_direct_partition_limit)
            {
                return;
            }
            active_context_partition_slots = min(
                context_partition_slots,
                graph_live_context_partitions);
        }
        else if (device_direct_partition_limit > 0)
        {
            const int live_context_partitions = max(
                1,
                (kv_len_runtime +
                 llaminar2::rocm::fa2_policy::
                     kROCmFA2CanonicalContextPartitionKeys -
                 1) /
                    llaminar2::rocm::fa2_policy::
                        kROCmFA2CanonicalContextPartitionKeys);
            if (live_context_partitions > device_direct_partition_limit)
                return;
        }
        const int logical_phase_blocks = WRITE_CONTEXT_PARTIAL
                                             ? phase_blocks_per_slot *
                                                   active_context_partition_slots
                                             : 1;
        const int first_logical_block = WRITE_CONTEXT_PARTIAL
                                            ? static_cast<int>(blockIdx.x)
                                            : 0;
        const int logical_block_stride = WRITE_CONTEXT_PARTIAL
                                             ? static_cast<int>(gridDim.x)
                                             : 1;
        const int tid = threadIdx.x;
        const int threads_per_block = blockDim.x;
        const int wavefront_id = tid / WAVEFRONT_SIZE;
        const int lane_id = tid % WAVEFRONT_SIZE;

        /*
         * Context graphs expose a CU-sized physical grid. Every workgroup
         * deterministically strides the larger logical space of partition
         * slots, batches, query tiles, and heads. All threads in a workgroup
         * execute the same task sequence, so the barriers in the tiled
         * attention body remain valid and no atomic work queue is required.
         */
        for (int logical_block = first_logical_block;
             logical_block < logical_phase_blocks;
             logical_block += logical_block_stride)
        {
            int batch_idx = static_cast<int>(blockIdx.z);
            int q_tile_idx = static_cast<int>(blockIdx.y);
            int head_idx = static_cast<int>(blockIdx.x);
            int context_partition_slot = 0;
            if constexpr (WRITE_CONTEXT_PARTIAL)
            {
                context_partition_slot =
                    logical_block / phase_blocks_per_slot;
                int participant_block =
                    logical_block % phase_blocks_per_slot;
                head_idx = participant_block % n_heads;
                participant_block /= n_heads;
                q_tile_idx = participant_block % logical_query_tiles;
                batch_idx = participant_block / logical_query_tiles;
            }

        // GQA mapping: multiple Q heads can share one KV head
        // When gqa_n_rep > 0, KV heads are REPLICATED (each device has all KV heads),
        // so use the global head position (head_start + head_idx). When gqa_n_rep == 0,
        // KV heads are SHARDED (each device has only local_n_kv_heads), so use local indexing.
        const int effective_gqa_ratio = (gqa_n_rep > 0) ? gqa_n_rep : (n_heads / n_kv_heads);
        const int kv_head_idx = (gqa_n_rep > 0)
            ? (head_start + head_idx) / effective_gqa_ratio
            : head_idx / effective_gqa_ratio;

        // Q row range for this block
        const int q_block_start = q_tile_idx * kernel_tile_q;
        if (q_block_start >= seq_len)
        {
            if constexpr (WRITE_CONTEXT_PARTIAL)
                continue;
            else
                return;
        }
        const int q_tile_rows = min(kernel_tile_q, seq_len - q_block_start);

        /*
         * Context execution launches a bounded persistent partition grid. Each
         * physical slot strides over logical partitions, while summary storage
         * is indexed by the logical partition number. The reducer can therefore
         * traverse one fixed ascending arithmetic order regardless of the
         * physical slot count selected by the capture-time policy.
         */
        int first_context_partition = 0;
        int context_partition_stride = 1;
        int live_context_partitions = 1;
        if constexpr (WRITE_CONTEXT_PARTIAL)
        {
            live_context_partitions = graph_live_context_partitions;
            first_context_partition = context_partition_slot;
            context_partition_stride = context_partition_slots;
            if (first_context_partition >= live_context_partitions ||
                context_partition_stride <= 0)
            {
                continue;
            }
        }

        // Compute LDS layout
        FA2LDSLayout lds_layout;
        lds_layout.compute(kernel_head_dim, kernel_tile_q, kernel_tile_kv);

        // LDS allocation
        extern __shared__ char smem[];

        __half *Q_tile = reinterpret_cast<__half *>(smem);
        __half *KV_buffers = Q_tile + kernel_tile_q * lds_layout.q_stride;
        float *scores = reinterpret_cast<float *>(KV_buffers + FA2_NUM_STAGES * 2 * kernel_tile_kv * lds_layout.kv_stride);

        // Helper lambdas for double-buffered K/V access
        auto get_K_tile = [&](int stage) -> __half *
        {
            return KV_buffers + stage * 2 * kernel_tile_kv * lds_layout.kv_stride;
        };
        auto get_V_tile = [&](int stage) -> __half *
        {
            return KV_buffers + (stage * 2 + 1) * kernel_tile_kv * lds_layout.kv_stride;
        };

        // =====================================================================
        // Thread-local accumulators for output
        // =====================================================================
        // Each thread is responsible for a subset of Q rows
        // With 256 threads and tile_q=64, each thread handles 64/256 = 0.25 rows
        // So we use a strided pattern where thread t handles rows t, t+256, ...

        // Use 4 threads per Q row so all 256 threads participate when tile_q=64.
        // Each thread owns a disjoint subset of output dims for that row.
        const int local_q_row = tid / THREADS_PER_ROW;
        const int row_lane = tid % THREADS_PER_ROW;
        const bool thread_has_row = (local_q_row < q_tile_rows);

        /*
         * Size static Qwen instantiations from their exact head dimension so
         * LLVM never reserves a 256-wide accumulator for HD64/HD128. Generic
         * uncommon dimensions retain the maximum bound and runtime predicate.
         * Four lanes cooperate through HD128 and sixteen lanes above it; each
         * thread therefore owns only the output values that can be live for its
         * compiled geometry, keeping the array in VGPRs instead of scratch.
         */
        constexpr int COMPILED_HEAD_DIM =
            STATIC_HEAD_DIM > 0 ? STATIC_HEAD_DIM : 256;
        constexpr int OUTPUTS_PER_THREAD =
            (COMPILED_HEAD_DIM + THREADS_PER_ROW - 1) /
            THREADS_PER_ROW;
        float O_acc[OUTPUTS_PER_THREAD];
        float m_i = -FLT_MAX;
        float l_i = 0.0f;
        float merged_m = -FLT_MAX;
        float merged_l = 0.0f;

        #pragma unroll
        for (int d_local = 0; d_local < OUTPUTS_PER_THREAD; ++d_local)
        {
            O_acc[d_local] = 0.0f;
        }

        // Batch/head pointers
        const float *Q_batch = Q + batch_idx * seq_len * n_heads * kernel_head_dim;
        const size_t request_element_offset =
            static_cast<size_t>(batch_idx) * static_cast<size_t>(kv_stride) *
            static_cast<size_t>(n_kv_heads) * static_cast<size_t>(kernel_head_dim);

        // =====================================================================
        // Load Q tile into LDS (done once per block)
        // =====================================================================
        if ((kernel_head_dim & 1) == 0)
        {
            const int head_dim_vec2 = kernel_head_dim / 2;
            for (int i = tid; i < q_tile_rows * head_dim_vec2; i += threads_per_block)
            {
                int local_row = i / head_dim_vec2;
                int d2 = i % head_dim_vec2;
                int d = d2 * 2;
                int global_row = q_block_start + local_row;

                const float2 qv = *reinterpret_cast<const float2 *>(
                    Q_batch + global_row * n_heads * kernel_head_dim + head_idx * kernel_head_dim + d);
                reinterpret_cast<__half2 *>(Q_tile + local_row * lds_layout.q_stride + d)[0] = __float22half2_rn(qv);
            }
        }
        else
        {
            for (int i = tid; i < q_tile_rows * kernel_head_dim; i += threads_per_block)
            {
                int local_row = i / kernel_head_dim;
                int d = i % kernel_head_dim;
                int global_row = q_block_start + local_row;
                float val = Q_batch[global_row * n_heads * kernel_head_dim + head_idx * kernel_head_dim + d];
                Q_tile[local_row * lds_layout.q_stride + d] = float_to_half(val);
            }
        }
        __syncthreads();

        for (int context_partition = first_context_partition;
             context_partition < live_context_partitions;
             context_partition += context_partition_stride)
        {
            /*
             * A persistent physical slot may own several logical partitions.
             * Reset only its private online-softmax tuple between them; Q stays
             * resident in LDS and the graph launches no additional work.
             */
            if (thread_has_row)
            {
                #pragma unroll
                for (int d_local = 0;
                     d_local < OUTPUTS_PER_THREAD;
                     ++d_local)
                {
                    O_acc[d_local] = 0.0f;
                }
                m_i = -FLT_MAX;
                l_i = 0.0f;
            }

            const int kv_partition_begin = WRITE_CONTEXT_PARTIAL
                                               ? context_partition *
                                                     context_partition_size
                                               : 0;
            const int kv_partition_end = WRITE_CONTEXT_PARTIAL
                                             ? min(
                                                   kv_partition_begin +
                                                       context_partition_size,
                                                   kv_len_runtime)
                                             : kv_len_runtime;
            const int partition_kv_len =
                max(0, kv_partition_end - kv_partition_begin);
            const int num_kv_tiles =
                (partition_kv_len + kernel_tile_kv - 1) /
                kernel_tile_kv;

            // =================================================================
            // Prologue: Load the first K/V tile in this arithmetic partition.
            // =================================================================
            {
            const int kv_start = kv_partition_begin;
            const int kv_end = min(
                kv_start + kernel_tile_kv,
                kv_partition_end);
            const int actual_len = kv_end - kv_start;

            __half *K_dst = get_K_tile(0);
            __half *V_dst = get_V_tile(0);

            if ((kernel_head_dim & 1) == 0)
            {
                const int head_dim_vec2 = kernel_head_dim / 2;
                for (int i = tid; i < actual_len * head_dim_vec2; i += threads_per_block)
                {
                    int local_row = i / head_dim_vec2;
                    int d2 = i % head_dim_vec2;
                    int d = d2 * 2;
                    const int logical_row = kv_start + local_row;
                    const int global_row = ring_row_capacity > 0
                                               ? (ring_row_origin + logical_row) %
                                                     ring_row_capacity
                                               : logical_row;
                    const size_t element =
                        request_element_offset +
                        (static_cast<size_t>(global_row) * n_kv_heads + kv_head_idx) *
                            static_cast<size_t>(kernel_head_dim) +
                        static_cast<size_t>(d);

                    reinterpret_cast<__half2 *>(
                        K_dst + local_row * lds_layout.kv_stride + d)[0] =
                        load_fa2_kv_pair_as_half2<KV_TYPE>(K, element);
                    reinterpret_cast<__half2 *>(
                        V_dst + local_row * lds_layout.kv_stride + d)[0] =
                        load_fa2_kv_pair_as_half2<KV_TYPE>(V, element);
                }
            }
            else
            {
                for (int i = tid; i < actual_len * kernel_head_dim; i += threads_per_block)
                {
                    int local_row = i / kernel_head_dim;
                    int d = i % kernel_head_dim;
                    const int logical_row = kv_start + local_row;
                    const int global_row = ring_row_capacity > 0
                                               ? (ring_row_origin + logical_row) %
                                                     ring_row_capacity
                                               : logical_row;
                    const size_t element =
                        request_element_offset +
                        (static_cast<size_t>(global_row) * n_kv_heads + kv_head_idx) *
                            static_cast<size_t>(kernel_head_dim) +
                        static_cast<size_t>(d);

                    K_dst[local_row * lds_layout.kv_stride + d] =
                        load_fa2_kv_element_as_half<KV_TYPE>(K, element);
                    V_dst[local_row * lds_layout.kv_stride + d] =
                        load_fa2_kv_element_as_half<KV_TYPE>(V, element);
                }
            }
        }
        __syncthreads();

        // =====================================================================
        // Main loop over KV tiles
        // =====================================================================
        for (int kv_tile_iter = 0; kv_tile_iter < num_kv_tiles; kv_tile_iter++)
        {
            const int current_stage = kv_tile_iter % FA2_NUM_STAGES;
            const int next_stage = (kv_tile_iter + 1) % FA2_NUM_STAGES;

            const int kv_start =
                kv_partition_begin + kv_tile_iter * kernel_tile_kv;
            const int kv_end = min(
                kv_start + kernel_tile_kv,
                kv_partition_end);
            const int actual_kv_len = kv_end - kv_start;

            // Every later tile is also masked for the complete query block.
            if (causal && kv_start > (q_block_start + kernel_tile_q - 1) + position_offset_runtime)
            {
                break;
            }

            __half *K_tile = get_K_tile(current_stage);
            __half *V_tile = get_V_tile(current_stage);

            // =================================================================
            // PHASE 1: Start loading next KV tile (if any)
            // =================================================================
            if (kv_tile_iter + 1 < num_kv_tiles)
            {
                const int next_kv_start =
                    kv_partition_begin +
                    (kv_tile_iter + 1) * kernel_tile_kv;
                const int next_kv_end = min(
                    next_kv_start + kernel_tile_kv,
                    kv_partition_end);
                const int next_len = next_kv_end - next_kv_start;

                __half *K_next = get_K_tile(next_stage);
                __half *V_next = get_V_tile(next_stage);

                // Cooperative load by all threads
                if ((kernel_head_dim & 1) == 0)
                {
                    const int head_dim_vec2 = kernel_head_dim / 2;
                    for (int i = tid; i < next_len * head_dim_vec2; i += threads_per_block)
                    {
                        int local_row = i / head_dim_vec2;
                        int d2 = i % head_dim_vec2;
                        int d = d2 * 2;
                        const int logical_row = next_kv_start + local_row;
                        const int global_row = ring_row_capacity > 0
                                                   ? (ring_row_origin + logical_row) %
                                                         ring_row_capacity
                                                   : logical_row;
                        const size_t element =
                            request_element_offset +
                            (static_cast<size_t>(global_row) * n_kv_heads + kv_head_idx) *
                                static_cast<size_t>(kernel_head_dim) +
                            static_cast<size_t>(d);

                        reinterpret_cast<__half2 *>(
                            K_next + local_row * lds_layout.kv_stride + d)[0] =
                            load_fa2_kv_pair_as_half2<KV_TYPE>(K, element);
                        reinterpret_cast<__half2 *>(
                            V_next + local_row * lds_layout.kv_stride + d)[0] =
                            load_fa2_kv_pair_as_half2<KV_TYPE>(V, element);
                    }
                }
                else
                {
                    for (int i = tid; i < next_len * kernel_head_dim; i += threads_per_block)
                    {
                        int local_row = i / kernel_head_dim;
                        int d = i % kernel_head_dim;
                        const int logical_row = next_kv_start + local_row;
                        const int global_row = ring_row_capacity > 0
                                                   ? (ring_row_origin + logical_row) %
                                                         ring_row_capacity
                                                   : logical_row;
                        const size_t element =
                            request_element_offset +
                            (static_cast<size_t>(global_row) * n_kv_heads + kv_head_idx) *
                                static_cast<size_t>(kernel_head_dim) +
                            static_cast<size_t>(d);

                        K_next[local_row * lds_layout.kv_stride + d] =
                            load_fa2_kv_element_as_half<KV_TYPE>(K, element);
                        V_next[local_row * lds_layout.kv_stride + d] =
                            load_fa2_kv_element_as_half<KV_TYPE>(V, element);
                    }
                }
            }

            // =================================================================
            // PHASE 2: Compute Q @ K^T (v_dot2_f32_f16 with FP32 accumulator)
            // =================================================================
            // Each thread computes a subset of score matrix elements

            // Score matrix is [q_tile_rows, actual_kv_len]
            // We distribute computation: thread t computes elements at stride
            const int total_score_elements = q_tile_rows * actual_kv_len;

            for (int s_idx = tid; s_idx < total_score_elements; s_idx += threads_per_block)
            {
                int q_row = s_idx / actual_kv_len;
                int kv_col = s_idx % actual_kv_len;

                // Compute dot product Q[q_row] @ K[kv_col]
                float score = 0.0f;

                const __half *Q_ptr = Q_tile + q_row * lds_layout.q_stride;
                const __half *K_ptr = K_tile + kv_col * lds_layout.kv_stride;

                // Vectorized dot product using v_dot2_f32_f16 (MI50 gfx906+)
                // Each instruction: d.f32 = a.f16x2 · b.f16x2 + c.f32
                // Replaces 4× fp16→fp32 convert + 2× FP32 FMA per half2 pair
                // with a single hardware dot-product instruction.
                if ((kernel_head_dim & 1) == 0)
                {
                    const int head_dim_vec2 = kernel_head_dim / 2;
                    const __half2 *Q_ptr2 = reinterpret_cast<const __half2 *>(Q_ptr);
                    const __half2 *K_ptr2 = reinterpret_cast<const __half2 *>(K_ptr);

                    // Use 4 independent accumulators to break dependency chain.
                    // v_dot2_f32_f16 has ~4 cycle latency; with 4 accumulators
                    // the pipeline stays full (ILP hiding).
                    float score0 = 0.0f, score1 = 0.0f, score2 = 0.0f, score3 = 0.0f;
                    int d2 = 0;
                    const int head_dim_vec2_aligned = head_dim_vec2 & ~3; // round down to multiple of 4
                    for (; d2 < head_dim_vec2_aligned; d2 += 4)
                    {
                        score0 = __builtin_amdgcn_fdot2(Q_ptr2[d2 + 0], K_ptr2[d2 + 0], score0, false);
                        score1 = __builtin_amdgcn_fdot2(Q_ptr2[d2 + 1], K_ptr2[d2 + 1], score1, false);
                        score2 = __builtin_amdgcn_fdot2(Q_ptr2[d2 + 2], K_ptr2[d2 + 2], score2, false);
                        score3 = __builtin_amdgcn_fdot2(Q_ptr2[d2 + 3], K_ptr2[d2 + 3], score3, false);
                    }
                    // Handle remaining pairs (head_dim not multiple of 8)
                    for (; d2 < head_dim_vec2; d2++)
                    {
                        score0 = __builtin_amdgcn_fdot2(Q_ptr2[d2], K_ptr2[d2], score0, false);
                    }
                    score = score0 + score1 + score2 + score3;
                }
                else
                {
                    for (int d = 0; d < kernel_head_dim; d++)
                    {
                        score += half_to_float(Q_ptr[d]) * half_to_float(K_ptr[d]);
                    }
                }

                scores[q_row * lds_layout.scores_stride + kv_col] = score * softmax_scale;
            }
            __syncthreads();

            // =================================================================
            // PHASE 3: Apply softmax and accumulate O = softmax(QK^T) @ V
            // =================================================================
            // Each thread processes its assigned Q rows
            //
            // Optimization: check at the tile level whether ALL scores in this
            // tile are unmasked (fully valid). For causal attention, ~90% of
            // tiles are fully valid since only the diagonal tiles need masking.
            // This eliminates 3-5 conditional branches per score element.

            if (thread_has_row)
            {
                int global_q_row = q_block_start + local_q_row;
                float *my_scores = scores + local_q_row * lds_layout.scores_stride;

                // Determine if this tile needs per-element masking
                // Tile is fully unmasked when:
                //   - No external mask
                //   - No sliding window (or window covers entire tile)
                //   - For causal: last KV position <= first Q position in block
                const bool tile_needs_masking =
                    mask != nullptr ||
                    (window_size > 0) ||
                    (causal && (kv_start + actual_kv_len - 1) > (global_q_row + position_offset_runtime));

                // Find max for this row (with masking)
                float m_ij = -FLT_MAX;

                if (!tile_needs_masking)
                {
                    // FAST PATH: no masking needed — scores already pre-scaled
                    for (int j = 0; j < actual_kv_len; j++)
                    {
                        m_ij = fmaxf(m_ij, my_scores[j]);
                    }
                }
                else
                {
                    // SLOW PATH: per-element masking
                    for (int j = 0; j < actual_kv_len; j++)
                    {
                        int kv_pos = kv_start + j;
                        bool masked = false;

                        // Causal masking
                        if (causal && kv_pos > global_q_row + position_offset_runtime)
                            masked = true;

                        // Sliding window
                        if (window_size > 0)
                        {
                            int q_pos = global_q_row + position_offset_runtime;
                            if (kv_pos < q_pos - window_size || kv_pos > q_pos + window_size)
                                masked = true;
                        }

                        if (mask)
                        {
                            float mask_val = mask[(batch_idx * seq_len + global_q_row) * mask_stride + kv_pos];
                            if (mask_val <= -1.0e20f)
                            {
                                masked = true;
                            }
                            else
                            {
                                my_scores[j] += mask_val;
                            }
                        }

                        if (masked)
                        {
                            my_scores[j] = -FLT_MAX;
                        }
                        else
                        {
                            m_ij = fmaxf(m_ij, my_scores[j]);
                        }
                    }
                }

                // Online softmax update
                float m_i_new = fmaxf(m_i, m_ij);
                float scale_old = fast_expf(m_i - m_i_new);

                // Rescale previous accumulators
                #pragma unroll
                for (int d_local = 0; d_local < OUTPUTS_PER_THREAD; ++d_local)
                {
                    const int d = row_lane + d_local * THREADS_PER_ROW;
                    if (d < kernel_head_dim)
                        O_acc[d_local] *= scale_old;
                }
                l_i *= scale_old;

                // Accumulate P @ V for this tile
                float l_ij = 0.0f;
                for (int j = 0; j < actual_kv_len; j++)
                {
                    float s = my_scores[j];
                    if (s > -FLT_MAX / 2)
                    {
                        float p = fast_expf(s - m_i_new);
                        l_ij += p;

                        const __half *V_row = V_tile + j * lds_layout.kv_stride;
                        #pragma unroll
                        for (int d_local = 0; d_local < OUTPUTS_PER_THREAD; ++d_local)
                        {
                            const int d = row_lane + d_local * THREADS_PER_ROW;
                            if (d < kernel_head_dim)
                                O_acc[d_local] += p * half_to_float(V_row[d]);
                        }
                    }
                }

                l_i += l_ij;
                m_i = m_i_new;
            }

            constexpr int canonical_partition_keys =
                llaminar2::rocm::fa2_policy::
                    kROCmFA2CanonicalContextPartitionKeys;
            const int maximum_visible_key =
                q_block_start + kernel_tile_q - 1 +
                position_offset_runtime;
            const bool next_tile_is_causally_empty =
                causal && kv_end > maximum_visible_key;

            if constexpr (!WRITE_CONTEXT_PARTIAL)
            {
                const bool closes_canonical_partition =
                    kv_end % canonical_partition_keys == 0;
                const bool closes_scan =
                    kv_tile_iter + 1 == num_kv_tiles ||
                    next_tile_is_causally_empty;
                if ((closes_canonical_partition || closes_scan) &&
                    thread_has_row)
                {
                    const int global_q_row =
                        q_block_start + local_q_row;
                    float *O_row =
                        O +
                        ((static_cast<size_t>(batch_idx) * seq_len +
                          global_q_row) *
                             n_heads +
                         head_idx) *
                            static_cast<size_t>(kernel_head_dim);

                    if (l_i > 0.0f)
                    {
                        if (!(merged_l > 0.0f))
                        {
                            merged_m = m_i;
                            merged_l = l_i;
                            #pragma unroll
                            for (int d_local = 0;
                                 d_local < OUTPUTS_PER_THREAD;
                                 ++d_local)
                            {
                                const int d =
                                    row_lane +
                                    d_local * THREADS_PER_ROW;
                                if (d < kernel_head_dim)
                                    O_row[d] = O_acc[d_local];
                            }
                        }
                        else
                        {
                            const float next_m =
                                fmaxf(merged_m, m_i);
                            const float previous_scale =
                                fast_expf(merged_m - next_m);
                            const float partition_scale =
                                fast_expf(m_i - next_m);
                            #pragma unroll
                            for (int d_local = 0;
                                 d_local < OUTPUTS_PER_THREAD;
                                 ++d_local)
                            {
                                const int d =
                                    row_lane +
                                    d_local * THREADS_PER_ROW;
                                if (d < kernel_head_dim)
                                {
                                    O_row[d] = merge_fa2_scaled_terms(
                                        O_row[d],
                                        previous_scale,
                                        O_acc[d_local],
                                        partition_scale);
                                }
                            }
                            merged_l = merge_fa2_scaled_terms(
                                merged_l,
                                previous_scale,
                                l_i,
                                partition_scale);
                            merged_m = next_m;
                        }
                    }

                    if (!closes_scan)
                    {
                        #pragma unroll
                        for (int d_local = 0;
                             d_local < OUTPUTS_PER_THREAD;
                             ++d_local)
                        {
                            O_acc[d_local] = 0.0f;
                        }
                        m_i = -FLT_MAX;
                        l_i = 0.0f;
                    }
                }
            }

            __syncthreads();
            if (next_tile_is_causally_empty)
                break;
        }

        // =====================================================================
        // Publish one canonical summary or the final normalized output.
        // =====================================================================
        if (thread_has_row)
        {
            const int global_q_row = q_block_start + local_q_row;
            float *O_row =
                O +
                ((static_cast<size_t>(batch_idx) * seq_len +
                  global_q_row) *
                     n_heads +
                 head_idx) *
                    static_cast<size_t>(kernel_head_dim);

            if constexpr (WRITE_CONTEXT_PARTIAL)
            {
                const size_t summary_index =
                    ((static_cast<size_t>(batch_idx) * seq_len +
                      global_q_row) *
                         n_heads +
                     head_idx) *
                        max_context_partitions +
                    context_partition;
                float *O_summary =
                    O_partial +
                    summary_index * static_cast<size_t>(kernel_head_dim);
                #pragma unroll
                for (int d_local = 0;
                     d_local < OUTPUTS_PER_THREAD;
                     ++d_local)
                {
                    const int d =
                        row_lane + d_local * THREADS_PER_ROW;
                    if (d < kernel_head_dim)
                        O_summary[d] = O_acc[d_local];
                }
                if (row_lane == 0)
                {
                    m_partial[summary_index] = m_i;
                    l_partial[summary_index] = l_i;
                }
            }
            else
            {
                const float inverse_l =
                    merged_l > 0.0f ? 1.0f / merged_l : 0.0f;
                #pragma unroll
                for (int d_local = 0;
                     d_local < OUTPUTS_PER_THREAD;
                     ++d_local)
                {
                    const int d =
                        row_lane + d_local * THREADS_PER_ROW;
                    if (d < kernel_head_dim)
                    {
                        O_row[d] = merged_l > 0.0f
                                       ? O_row[d] * inverse_l
                                       : 0.0f;
                    }
                }
            }
        }

        if constexpr (WRITE_CONTEXT_PARTIAL)
        {
            // Every thread must finish consuming LDS before this persistent
            // slot advances and reuses it for the next logical partition.
            __syncthreads();
        }
        }
        }
    }

    // =========================================================================
    // Flash Decoding - Split-K Kernel for MI50 (v3: wavefront-cooperative)
    // =========================================================================
    //
    // For single-token decode (seq_len=1), we parallelize over the KV cache
    // using split-K. Each block handles a portion of the KV cache and produces
    // partial (m, l, O) that are merged in a reduction kernel.
    //
    // Grid: (n_heads, num_splits, batch_size)
    // Block: (256,) threads = 4 wavefronts
    //
    // v3 architecture (2026-02-09):
    //   LANES OWN OUTPUT DIMS, NOT KV POSITIONS.
    //
    //   Each wavefront cooperates on one KV position at a time:
    //     1. All 64 lanes load contiguous K elements (coalesced!)
    //     2. Partial dot product → wavefrontReduceSum → full score
    //     3. Online softmax update (score is uniform across lanes)
    //     4. All 64 lanes load contiguous V elements (coalesced!)
    //     5. Each lane updates only its 2 output dims of O
    //
    //   Benefits vs v2:
    //     - Coalesced K/V memory access (adjacent lanes → adjacent addresses)
    //     - 2-4 VGPRs for O per lane instead of 128 → 4x occupancy potential
    //     - Zero cross-lane O reduction (no shuffles for O at all)
    //     - Only 6 shuffles per KV position (dot product reduce)
    //     - v2 had 128 × wavefrontReduceSum = 768 shuffles TOTAL for O
    //
    //   v3+ optimization: exp2f hardware intrinsic
    //     - All expf() replaced with exp2f(x * LOG2E) for single-cycle v_exp_f32
    //     - GCN ISA: v_exp_f32 is a transcendental ALU op (~4 cycles vs ~50+ for expf)
    //
    // =========================================================================

    // exp2f(x * LOG2E) maps to hardware v_exp_f32 on GCN (single-cycle transcendental)
    __device__ __forceinline__ float fast_expf(float x)
    {
        return exp2f(x * 1.4426950408889634f); // log2(e) = 1/ln(2)
    }

    /**
     * @brief Physical cache ownership represented by one decode grid row.
     *
     * The arithmetic body is shared across all policies.  Only parameter-row
     * selection and the K/V bank base differ, which is essential for proving
     * grouped execution byte-identical to the ordinary scalar decode kernel.
     */
    enum class FlashDecodeRowMode
    {
        OrdinaryBatch,      ///< One physical K/V bank per ordinary batch row.
        SharedKVVerifier,   ///< Every verifier row reads one shared cache bank.
        IndependentRequest ///< Flat rows map to fixed-stride request cache banks.
    };

    /**
     * @brief Scalar-equivalent launch policy for one grouped decode row.
     */
    struct FlashDecodeRowConfig
    {
        int splits;
        int threads_per_block;
    };

    /**
     * @brief Clamp an integer without relying on host-only standard-library code.
     */
    __device__ __forceinline__ int deviceClampInt(int value, int lo, int hi)
    {
        return value < lo ? lo : (value > hi ? hi : value);
    }

    /**
     * @brief Round upward to a power of two, bounded by a row's split cap.
     */
    __device__ __forceinline__ int deviceRoundUpPow2Clamped(
        int value,
        int max_allowed)
    {
        int result = 1;
        while (result < value && result < max_allowed)
            result <<= 1;
        return deviceClampInt(result, 1, max_allowed);
    }

    /**
     * @brief Reproduce the production scalar ROCm decode policy on device.
     *
     * Grouped rows can own unequal live KV lengths even though graph capture
     * requires one fixed grid and block shape. The phase kernel launches the
     * maximum physical geometry, then only the
     * split and wavefront prefix selected from the live device-owned KV length
     * participates. Consequently ordinary M=1 and grouped verifier rows retain
     * the same KV partition, wavefront merge order, and reduction order while
     * one graph executable spans every short- and long-sequence regime.
     */
    __device__ __forceinline__ FlashDecodeRowConfig selectDecodeRowConfig(
        int kv_len,
        int head_dim,
        int n_heads,
        int num_cus,
        int max_launch_splits,
        bool forced_splits_present,
        int forced_splits,
        bool forced_tpb_present,
        int forced_tpb)
    {
        /*
         * These regimes are offline-tuned launch policy, not online
         * autotuning. The 256-thread physical workgroup is immutable in the
         * graph. Only the active split prefix changes from device-owned
         * sequence metadata:
         *
         *   - eight splits amortize phase/reduction overhead at short context;
         *   - sixteen splits fill gfx906 for ordinary Qwen3.6 head counts;
         *   - a LocalTP shard with at most eight query heads needs 32 splits
         *     from 1K onward to expose the same block-level parallelism;
         *   - ordinary 16-head execution crosses to 32 splits at 8K.
         *
         * The byte-equivalence sweep compares grouped rows against independent
         * M=1 calls across every boundary, so changing prefix size cannot
         * silently change grouped verifier arithmetic.
         */
        int requested_cap = 8;
        if (kv_len >= 8192)
            requested_cap = 32;
        else if (n_heads <= 8 && kv_len >= 1024)
            requested_cap = 32;
        else if (kv_len >= 256)
            requested_cap = 16;

        const int max_allowed = deviceClampInt(
            deviceClampInt(
                requested_cap < max_launch_splits
                    ? requested_cap
                    : max_launch_splits,
                1,
                kv_len > 0 ? kv_len : 1),
            1,
            FD_MAX_SPLITS_SUPPORTED);
        const int safe_head_dim = head_dim > 0 ? head_dim : 1;
        const int64_t total_elements =
            static_cast<int64_t>(kv_len) * static_cast<int64_t>(safe_head_dim);
        int base_splits = static_cast<int>(
            (total_elements + FD_TARGET_ELEMS_PER_SPLIT - 1) /
            FD_TARGET_ELEMS_PER_SPLIT);
        if (base_splits < 1)
            base_splits = 1;

        if (kv_len >= safe_head_dim * 8)
            base_splits = (base_splits * 3 + 1) / 2;
        else if (kv_len <= safe_head_dim * 2)
            base_splits = (base_splits + 1) / 2 > 1
                              ? (base_splits + 1) / 2
                              : 1;

        if (base_splits > kv_len)
            base_splits = kv_len > 0 ? kv_len : 1;
        int row_splits =
            deviceRoundUpPow2Clamped(base_splits, max_allowed);

        if (n_heads > 0 && num_cus > 0)
        {
            /*
             * The 36-VGPR FP16 specialization sustains seven resident
             * workgroups per CU. Filling that envelope is especially
             * important for LocalTP shards whose reduced head count would
             * otherwise leave most CUs idle at long context.
             */
            constexpr int target_resident_blocks_per_cu = 7;
            const int target_blocks =
                num_cus * target_resident_blocks_per_cu;
            if (n_heads * row_splits < target_blocks)
            {
                const int occupancy_splits =
                    (target_blocks + n_heads - 1) / n_heads;
                const int rounded_occupancy_splits =
                    deviceRoundUpPow2Clamped(occupancy_splits, max_allowed);
                if (rounded_occupancy_splits > row_splits)
                    row_splits = rounded_occupancy_splits;
            }
        }
        row_splits = deviceClampInt(row_splits, 1, max_allowed);
        if (forced_splits_present)
        {
            /*
             * Offline profiling must be able to compare every split count
             * admitted by the immutable capture envelope, including counts
             * above the production heuristic for this live KV length.
             */
            row_splits = deviceClampInt(
                forced_splits,
                1,
                deviceClampInt(
                    max_launch_splits,
                    1,
                    FD_MAX_SPLITS_SUPPORTED));
        }

        /*
         * Offline A/B measurements showed that deactivating two wavefronts
         * roughly doubled short-context latency and remained slower at every
         * tested Qwen3.6 sequence length. Keep the logical workgroup identical
         * to the captured 256-thread workgroup in production.
         */
        int row_threads = 256;
        if (forced_tpb_present)
            row_threads = deviceClampInt(forced_tpb, 64, 256);

        return FlashDecodeRowConfig{row_splits, row_threads};
    }

    /**
     * @brief Flash Decoding kernel for single-query decode on MI50
     *        v3: wavefront-cooperative KV processing + exp2f intrinsic
     */
    template <KVTypeTag KV_TYPE = KVTypeTag::FP32>
    __global__ __launch_bounds__(256, 4)
    void flash_decoding_mi50_kernel(
        const float *__restrict__ Q,
        const void *__restrict__ K_cache,
        const void *__restrict__ V_cache,
        float *__restrict__ O_partial,
        float *__restrict__ m_partial,
        float *__restrict__ l_partial,
        int kv_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int num_splits,
        float softmax_scale,
        const llaminar2::attention::AttentionDeviceParams *__restrict__ device_params,
        FlashDecodeRowMode row_mode,
        int query_rows_per_request,
        int num_cus,
        bool forced_splits_present,
        int forced_splits,
        bool forced_tpb_present,
        int forced_tpb,
        int head_start = 0,
        int gqa_n_rep = 0)
    {
        /*
         * Ownership is runtime metadata, not a template axis. Keeping one
         * compiled kernel per physical KV format guarantees that ordinary M=1
         * and grouped verifier rows execute the same arithmetic instruction
         * stream. The mode affects addresses only.
         */
        const bool row_local_params =
            row_mode != FlashDecodeRowMode::OrdinaryBatch;
        const bool shared_kv =
            row_mode == FlashDecodeRowMode::SharedKVVerifier;

        int kv_len_runtime = kv_len;
        int kv_batch_stride = kv_len;
        int ring_row_origin = 0;
        int ring_row_capacity = 0;
        if (device_params)
        {
            const int param_row = row_local_params ? blockIdx.z : 0;
            kv_len_runtime = device_params[param_row].kv_len;
            kv_batch_stride = device_params[param_row].kv_stride;
            ring_row_origin = device_params[param_row].ring_row_origin;
            ring_row_capacity = device_params[param_row].ring_row_capacity;
        }

        const int head_idx = blockIdx.x;
        const int split_idx = blockIdx.y;
        const int batch_idx = blockIdx.z;

        FlashDecodeRowConfig row_config{
            num_splits,
            static_cast<int>(blockDim.x)};
        row_config = selectDecodeRowConfig(
            kv_len_runtime,
            head_dim,
            n_heads,
            num_cus,
            num_splits,
            forced_splits_present,
            forced_splits,
            forced_tpb_present,
            forced_tpb);
        const int row_num_splits = row_config.splits;
        const int logical_threads = row_config.threads_per_block;

        /*
         * Every thread in the physical 256-thread block still reaches both
         * barriers.  Only the scalar-selected prefix performs arithmetic.  In
         * particular, a row whose ordinary decode uses 128 threads executes
         * with exactly two active wavefronts and merges those two wavefronts in
         * the same order even when another row in this grid needs four.
         */
        const int tid = threadIdx.x;
        const bool active_thread = tid < logical_threads;
        const int wavefront_id = tid / WAVEFRONT_SIZE;
        const int lane_id = tid % WAVEFRONT_SIZE;
        const int num_wavefronts =
            logical_threads / WAVEFRONT_SIZE > 0
                ? logical_threads / WAVEFRONT_SIZE
                : 1;

        const int effective_gqa_ratio = (gqa_n_rep > 0) ? gqa_n_rep : (n_heads / n_kv_heads);
        const int kv_head_idx = (gqa_n_rep > 0)
            ? (head_start + head_idx) / effective_gqa_ratio
            : head_idx / effective_gqa_ratio;

        /* Surplus split planes are outside this row's scalar launch shape. */
        if (split_idx >= row_num_splits)
            return;

        // Split range: identical integer partitioning to scalar M=1 decode.
        const int split_size =
            (kv_len_runtime + row_num_splits - 1) / row_num_splits;
        const int kv_start = split_idx * split_size;
        const int kv_end = min(kv_start + split_size, kv_len_runtime);

        const int partial_idx = (batch_idx * n_heads + head_idx) * num_splits + split_idx;

        // Handle empty splits
        if (kv_start >= kv_len_runtime)
        {
            if (tid == 0)
            {
                m_partial[partial_idx] = -FLT_MAX;
                l_partial[partial_idx] = 0.0f;
            }
            for (int d = tid * 2;
                 active_thread && d + 1 < head_dim;
                 d += logical_threads * 2)
            {
                const int base = (batch_idx * n_heads + head_idx) * head_dim;
                const int idx0 = (base + d + 0) * num_splits + split_idx;
                const int idx1 = (base + d + 1) * num_splits + split_idx;
                O_partial[idx0] = 0.0f;
                O_partial[idx1] = 0.0f;
            }
            for (int d = (head_dim & ~1) + tid;
                 active_thread && d < head_dim;
                 d += logical_threads)
            {
                const int base = (batch_idx * n_heads + head_idx) * head_dim;
                const int idx = (base + d) * num_splits + split_idx;
                O_partial[idx] = 0.0f;
            }
            return;
        }

        // Load Q into LDS (all threads cooperate)
        // FP16 KV path: store Q as half in LDS (convert once, use many times)
        // FP32/BF16/Q8_1 paths store Q as float in LDS so BF16 and Q8_1
        // dequantization retains the scalar FP32 decode arithmetic contract.
        extern __shared__ char smem[];
        [[maybe_unused]] float *Q_shared = nullptr;
        [[maybe_unused]] __half *Q_shared_h = nullptr;

        const float *Q_ptr = Q + (batch_idx * n_heads + head_idx) * head_dim;

        if constexpr (KV_TYPE == KVTypeTag::FP16)
        {
            // Store Q as FP16 in LDS — eliminates per-KV-position FP32→FP16 conversion
            // Halves LDS bandwidth for Q reads (4B → 2B per element per lane)
            Q_shared_h = reinterpret_cast<__half *>(smem);
            for (int d = tid;
                 active_thread && d < head_dim;
                 d += logical_threads)
            {
                Q_shared_h[d] = __float2half(Q_ptr[d]);
            }
        }
        else
        {
            Q_shared = reinterpret_cast<float *>(smem);
            for (int d = tid * 2;
                 active_thread && d + 1 < head_dim;
                 d += logical_threads * 2)
            {
                const float2 qv = reinterpret_cast<const float2 *>(Q_ptr + d)[0];
                reinterpret_cast<float2 *>(Q_shared + d)[0] = qv;
            }
            for (int d = (head_dim & ~1) + tid;
                 active_thread && d < head_dim;
                 d += logical_threads)
            {
                Q_shared[d] = Q_ptr[d];
            }
        }
        __syncthreads();

        // =====================================================================
        // Thread-local O accumulators — only this lane's output dimensions
        //
        // head_dim / WAVEFRONT_SIZE dims per lane (strided by WAVEFRONT_SIZE).
        // head_dim=128 => 2 dims/lane, head_dim=256 => 4 dims/lane.
        // Lane i owns dims: i, i+64, i+128, i+192 (up to head_dim).
        //
        // This reduces VGPR usage to ceil(head_dim/64) floats per lane.
        // =====================================================================
        float O_lane[4] = {0, 0, 0, 0}; // Supports head_dim up to 256 (256/64=4)
        float m_local = -FLT_MAX;
        float l_local = 0.0f;

        // Byte-level base pointers for type-dispatched K/V access (KV_TYPE)
        // Batch + head offsets hoisted out of the main KV loop.
        const int kv_head_offset_elems = kv_head_idx * head_dim;
        const int kv_row_stride_elems = n_kv_heads * head_dim;

        // Q8_1 block layout (compiled out for non-Q8_1 via if constexpr)
        [[maybe_unused]] constexpr int Q8_BS = 32;
        [[maybe_unused]] const int q8_bph = head_dim / Q8_BS;    // blocks per head
        [[maybe_unused]] const int q8_bpr = n_kv_heads * q8_bph; // blocks per KV row

        int cache_batch_idx = batch_idx;
        if (shared_kv)
        {
            cache_batch_idx = 0;
        }
        else if (row_mode == FlashDecodeRowMode::IndependentRequest)
        {
            cache_batch_idx =
                batch_idx / (query_rows_per_request > 0
                                 ? query_rows_per_request
                                 : 1);
        }

        size_t row_byte_stride;
        size_t base_byte_off;
        if constexpr (KV_TYPE == KVTypeTag::Q8_1) {
            row_byte_stride = static_cast<size_t>(q8_bpr) * sizeof(llaminar2::Q8_1Block);
            base_byte_off = static_cast<size_t>(cache_batch_idx) * kv_batch_stride * row_byte_stride
                          + static_cast<size_t>(kv_head_idx) * q8_bph * sizeof(llaminar2::Q8_1Block);
        } else if constexpr (KV_TYPE == KVTypeTag::FP16) {
            row_byte_stride = static_cast<size_t>(kv_row_stride_elems) * sizeof(__half);
            base_byte_off = static_cast<size_t>(cache_batch_idx) * kv_batch_stride * row_byte_stride
                          + static_cast<size_t>(kv_head_offset_elems) * sizeof(__half);
        } else if constexpr (KV_TYPE == KVTypeTag::BF16) {
            row_byte_stride = static_cast<size_t>(kv_row_stride_elems) * sizeof(hip_bfloat16);
            base_byte_off = static_cast<size_t>(cache_batch_idx) * kv_batch_stride * row_byte_stride
                          + static_cast<size_t>(kv_head_offset_elems) * sizeof(hip_bfloat16);
        } else {
            row_byte_stride = static_cast<size_t>(kv_row_stride_elems) * sizeof(float);
            base_byte_off = static_cast<size_t>(cache_batch_idx) * kv_batch_stride * row_byte_stride
                          + static_cast<size_t>(kv_head_offset_elems) * sizeof(float);
        }
        const char *K_base = reinterpret_cast<const char*>(K_cache) + base_byte_off;
        const char *V_base = reinterpret_cast<const char*>(V_cache) + base_byte_off;

        // =====================================================================
        // Main loop: wavefront-cooperative KV processing
        //
        // Each wavefront processes a strided subset of KV positions.
        // Within each position, all 64 lanes cooperate on loads and dot product.
        // K/V loads are fully coalesced (adjacent lanes → adjacent addresses).
        // =====================================================================
        for (int kv_pos = kv_start + wavefront_id;
             active_thread && kv_pos < kv_end;
             kv_pos += num_wavefronts)
        {
            const int physical_kv_pos = ring_row_capacity > 0
                                            ? (ring_row_origin + kv_pos) %
                                                  ring_row_capacity
                                            : kv_pos;
            const char *K_pos =
                K_base + physical_kv_pos * row_byte_stride;

            // Type-dispatched K load + dot product
            // FP16: half bandwidth, fdot2 for vectorized path (Q→FP16 on the fly)
            // Q8_1: inline dequant, ~3.5× BW savings
            // FP32: original vectorized path (float2)
            float partial_dot = 0.0f;
            if constexpr (KV_TYPE == KVTypeTag::FP16)
            {
                const __half *Kh = reinterpret_cast<const __half*>(K_pos);
                if ((head_dim & 1) == 0)
                {
                    const int head_dim_vec2 = head_dim / 2;
                    const __half2 *Q_ptr2_h = reinterpret_cast<const __half2 *>(Q_shared_h);
                    const __half2 *K_ptr2 = reinterpret_cast<const __half2 *>(Kh);
                    for (int d2 = lane_id; d2 < head_dim_vec2; d2 += WAVEFRONT_SIZE)
                    {
                        const __half2 qh2 = Q_ptr2_h[d2]; // Pre-converted FP16 from LDS
                        const __half2 kh2 = K_ptr2[d2];
                        partial_dot = __builtin_amdgcn_fdot2(qh2, kh2, partial_dot, false);
                    }
                }
                else
                {
                    for (int d = lane_id; d < head_dim; d += WAVEFRONT_SIZE)
                    {
                        partial_dot += __half2float(Q_shared_h[d]) * __half2float(Kh[d]);
                    }
                }
            }
            else if constexpr (KV_TYPE == KVTypeTag::Q8_1)
            {
                const llaminar2::Q8_1Block *Kq = reinterpret_cast<const llaminar2::Q8_1Block*>(K_pos);
                for (int d = lane_id; d < head_dim; d += WAVEFRONT_SIZE)
                {
                    const int bi = d / Q8_BS;
                    const int bo = d % Q8_BS;
                    const float ks = __half2float(*reinterpret_cast<const __half*>(&Kq[bi].d));
                    partial_dot += Q_shared[d] * (static_cast<float>(Kq[bi].qs[bo]) * ks);
                }
            }
            else if constexpr (KV_TYPE == KVTypeTag::BF16)
            {
                const auto *K_bf16 =
                    reinterpret_cast<const hip_bfloat16 *>(K_pos);
                for (int d = lane_id; d < head_dim; d += WAVEFRONT_SIZE)
                {
                    partial_dot +=
                        Q_shared[d] * static_cast<float>(K_bf16[d]);
                }
            }
            else
            {
                const float *K_ptr = reinterpret_cast<const float*>(K_pos);
                if ((head_dim & 1) == 0)
                {
                    const int head_dim_vec2 = head_dim / 2;
                    const float2 *Q_ptr2 = reinterpret_cast<const float2 *>(Q_shared);
                    const float2 *K_ptr2 = reinterpret_cast<const float2 *>(K_ptr);
                    for (int d2 = lane_id; d2 < head_dim_vec2; d2 += WAVEFRONT_SIZE)
                    {
                        const float2 qv = Q_ptr2[d2];
                        const float2 kv = K_ptr2[d2];
                        partial_dot += qv.x * kv.x + qv.y * kv.y;
                    }
                }
                else
                {
                    for (int d = lane_id; d < head_dim; d += WAVEFRONT_SIZE)
                    {
                        partial_dot += Q_shared[d] * K_ptr[d];
                    }
                }
            }

            // Reduce across wavefront → full dot product (6 shuffles)
            // After this, ALL lanes have the same score value
            float score = wavefrontReduceSum(partial_dot) * softmax_scale;

            // Online softmax — score is uniform across all lanes,
            // so m_local and l_local stay wavefront-uniform too
            // Uses fast_expf (exp2f hardware intrinsic) for ~10x faster exp
            float m_new = fmaxf(m_local, score);
            float scale_old = fast_expf(m_local - m_new);
            float p = fast_expf(score - m_new);

            l_local = l_local * scale_old + p;

            // Type-dispatched V load + O accumulation
            // Each lane updates only its own output dims (head_dim/64 per lane)
            const char *V_pos =
                V_base + physical_kv_pos * row_byte_stride;
            int o_idx = 0;
            if constexpr (KV_TYPE == KVTypeTag::FP16)
            {
                const __half *Vh = reinterpret_cast<const __half*>(V_pos);
                for (int d = lane_id; d < head_dim; d += WAVEFRONT_SIZE, o_idx++)
                {
                    O_lane[o_idx] = O_lane[o_idx] * scale_old + p * __half2float(Vh[d]);
                }
            }
            else if constexpr (KV_TYPE == KVTypeTag::Q8_1)
            {
                const llaminar2::Q8_1Block *Vq = reinterpret_cast<const llaminar2::Q8_1Block*>(V_pos);
                for (int d = lane_id; d < head_dim; d += WAVEFRONT_SIZE, o_idx++)
                {
                    const int bi = d / Q8_BS;
                    const int bo = d % Q8_BS;
                    const float vs = __half2float(*reinterpret_cast<const __half*>(&Vq[bi].d));
                    O_lane[o_idx] = O_lane[o_idx] * scale_old + p * (static_cast<float>(Vq[bi].qs[bo]) * vs);
                }
            }
            else if constexpr (KV_TYPE == KVTypeTag::BF16)
            {
                const auto *V_bf16 =
                    reinterpret_cast<const hip_bfloat16 *>(V_pos);
                for (int d = lane_id;
                     d < head_dim;
                     d += WAVEFRONT_SIZE, ++o_idx)
                {
                    O_lane[o_idx] =
                        O_lane[o_idx] * scale_old +
                        p * static_cast<float>(V_bf16[d]);
                }
            }
            else
            {
                const float *V_ptr = reinterpret_cast<const float*>(V_pos);
                for (int d = lane_id; d < head_dim; d += WAVEFRONT_SIZE, o_idx++)
                {
                    O_lane[o_idx] = O_lane[o_idx] * scale_old + p * V_ptr[d];
                }
            }

            m_local = m_new;
        }

        // =====================================================================
        // Inter-wavefront reduction using LDS
        //
        // Each wavefront has:
        //   - (m_local, l_local) — uniform across its lanes
        //   - O distributed across lanes (each lane owns head_dim/64 dims)
        //
        // We merge 4 wavefronts' results, then write the final partial output.
        // =====================================================================
        __shared__ float block_m[4];
        __shared__ float block_l[4];
        __shared__ float block_O[4 * 256]; // 4 wavefronts × max head_dim

        // Write this wavefront's results to LDS
        if (active_thread && lane_id == 0)
        {
            block_m[wavefront_id] = m_local;
            block_l[wavefront_id] = l_local;
        }
        if (active_thread && head_dim == 128)
        {
            const int base = wavefront_id * 128;
            block_O[base + lane_id] = O_lane[0];
            block_O[base + lane_id + WAVEFRONT_SIZE] = O_lane[1];
        }
        else if (active_thread)
        {
            int o_idx = 0;
            for (int d = lane_id; d < head_dim; d += WAVEFRONT_SIZE, o_idx++)
            {
                block_O[wavefront_id * head_dim + d] = O_lane[o_idx];
            }
        }
        __syncthreads();

        // Thread 0 computes per-wavefront rescaling factors
        __shared__ float wf_scales[4];
        if (tid == 0)
        {
            float final_m = block_m[0];
            float final_l = block_l[0];
            wf_scales[0] = 1.0f;

            for (int w = 1; w < num_wavefronts; w++)
            {
                float other_m = block_m[w];
                float other_l = block_l[w];

                float m_new = fmaxf(final_m, other_m);
                float scale_self = fast_expf(final_m - m_new);
                float scale_other = fast_expf(other_m - m_new);

                for (int prev = 0; prev < w; prev++)
                    wf_scales[prev] *= scale_self;
                wf_scales[w] = scale_other;

                final_l = scale_self * final_l + scale_other * other_l;
                final_m = m_new;
            }

            m_partial[partial_idx] = final_m;
            l_partial[partial_idx] = final_l;
        }
        __syncthreads(); // All threads hit this — wf_scales now visible

        // Parallel output write — wavefront 0 lanes each handle their output dims
        if (active_thread && wavefront_id == 0)
        {
            const int base = (batch_idx * n_heads + head_idx) * head_dim;

            if (head_dim == 128)
            {
                float sum0 = 0.0f;
                float sum1 = 0.0f;

                if (num_wavefronts == 4)
                {
                    sum0 = wf_scales[0] * block_O[lane_id] +
                           wf_scales[1] * block_O[128 + lane_id] +
                           wf_scales[2] * block_O[256 + lane_id] +
                           wf_scales[3] * block_O[384 + lane_id];

                    const int lane1 = lane_id + WAVEFRONT_SIZE;
                    sum1 = wf_scales[0] * block_O[lane1] +
                           wf_scales[1] * block_O[128 + lane1] +
                           wf_scales[2] * block_O[256 + lane1] +
                           wf_scales[3] * block_O[384 + lane1];
                }
                else
                {
                    for (int w = 0; w < num_wavefronts; w++)
                    {
                        const int wbase = w * head_dim;
                        sum0 += wf_scales[w] * block_O[wbase + lane_id];
                        sum1 += wf_scales[w] * block_O[wbase + lane_id + WAVEFRONT_SIZE];
                    }
                }

                const int idx0 = (base + lane_id) * num_splits + split_idx;
                const int idx1 = (base + lane_id + WAVEFRONT_SIZE) * num_splits + split_idx;
                O_partial[idx0] = sum0;
                O_partial[idx1] = sum1;
            }
            else
            {
                for (int d = lane_id; d < head_dim; d += WAVEFRONT_SIZE)
                {
                    float sum = 0.0f;
                    for (int w = 0; w < num_wavefronts; w++)
                    {
                        sum += wf_scales[w] * block_O[w * head_dim + d];
                    }
                    const int idx = (base + d) * num_splits + split_idx;
                    O_partial[idx] = sum;
                }
            }
        }
    }

    /**
     * @brief Flash Decoding reduction kernel
     *
     * Combines partial outputs from all splits using stable softmax merge.
     */
    __global__ __launch_bounds__(256, 4)
    void flash_decoding_reduce_mi50_kernel(
        const float *__restrict__ O_partial,
        const float *__restrict__ m_partial,
        const float *__restrict__ l_partial,
        float *__restrict__ O,
        int n_heads,
        int head_dim,
        int num_splits,
        int kv_len,
        const llaminar2::attention::AttentionDeviceParams *__restrict__ device_params,
        FlashDecodeRowMode row_mode,
        int num_cus,
        bool forced_splits_present,
        int forced_splits,
        bool forced_tpb_present,
        int forced_tpb)
    {
        const int head_idx = blockIdx.x;
        const int batch_idx = blockIdx.y;
        const int output_tile = blockIdx.z;
        const int output_tiles = gridDim.z;

        const int tid = threadIdx.x;
        const int param_row =
            row_mode != FlashDecodeRowMode::OrdinaryBatch ? batch_idx : 0;
        const int row_kv_len =
            device_params ? device_params[param_row].kv_len : kv_len;
        const int row_num_splits =
            selectDecodeRowConfig(
                row_kv_len,
                head_dim,
                n_heads,
                num_cus,
                num_splits,
                forced_splits_present,
                forced_splits,
                forced_tpb_present,
                forced_tpb)
                .splits;
        const int base_idx = (batch_idx * n_heads + head_idx) * num_splits;

        // Find global max and compute scales
        __shared__ float global_l;
        __shared__ float split_scales[32]; // Support up to 32 splits

        if (tid == 0)
        {
            float m_max = -FLT_MAX;
            for (int s = 0; s < row_num_splits; s++)
            {
                m_max = fmaxf(m_max, m_partial[base_idx + s]);
            }
            float l_sum = 0.0f;
            for (int s = 0; s < row_num_splits; s++)
            {
                float scale = expf(m_partial[base_idx + s] - m_max);
                split_scales[s] = scale;
                l_sum += scale * l_partial[base_idx + s];
            }
            global_l = l_sum;
        }
        __syncthreads();

        float inv_l = (global_l > 0.0f) ? (1.0f / global_l) : 0.0f;
        float *O_out = O + (batch_idx * n_heads + head_idx) * head_dim;
        const int head_base = (batch_idx * n_heads + head_idx) * head_dim;
        const int output_begin =
            (head_dim * output_tile) / output_tiles;
        const int output_end =
            (head_dim * (output_tile + 1)) / output_tiles;
        const int vectorized_elements =
            (output_end - output_begin) & ~1;

        /*
         * One wave owns each output tile. The scalar softmax merge is
         * intentionally repeated by each tile, while every output component
         * accumulates split partials in the original ascending order.
         */
        for (int offset = tid * 2;
             offset + 1 < vectorized_elements;
             offset += blockDim.x * 2)
        {
            const int d = output_begin + offset;
            float O_sum0 = 0.0f;
            float O_sum1 = 0.0f;
            const int split_base0 = (head_base + d + 0) * num_splits;
            const int split_base1 = (head_base + d + 1) * num_splits;
            for (int s = 0; s < row_num_splits; s++)
            {
                const float scale = split_scales[s];
                O_sum0 += scale * O_partial[split_base0 + s];
                O_sum1 += scale * O_partial[split_base1 + s];
            }
            reinterpret_cast<float2 *>(O_out + d)[0] = make_float2(O_sum0 * inv_l, O_sum1 * inv_l);
        }

        for (int d = output_begin + vectorized_elements + tid;
             d < output_end;
             d += blockDim.x)
        {
            float O_sum = 0.0f;
            const int split_base = (head_base + d) * num_splits;
            for (int s = 0; s < row_num_splits; s++)
            {
                O_sum += split_scales[s] * O_partial[split_base + s];
            }
            O_out[d] = O_sum * inv_l;
        }
    }

} // anonymous namespace

// =============================================================================
// Device-resident prefill launch template
// =============================================================================

/**
 * @brief Launch the native-format ROCm prefill kernel with stable graph shape.
 *
 * Tile selection depends only on the immutable head dimension and MI50 LDS
 * capacity.  In particular, neither the live KV length nor the current cache
 * fill level changes the grid or shared-memory allocation.  A captured graph
 * can therefore replay after later cache appends: the kernel reads the new
 * logical length and fixed physical request stride from @p device_params.
 *
 * The cache remains compressed in HBM for FP16 and Q8_1.  Only each live K/V
 * tile is converted to the half-precision LDS representation used by gfx906
 * `v_dot2_f32_f16`, avoiding full-cache conversion buffers and host-sized GEMM
 * launches.
 *
 * @tparam KV_TYPE Physical representation of the resident K/V cache.
 * @return Zero after a valid launch, or -1 when geometry is unsupported or
 *         HIP reports a launch error.
 */
template <KVTypeTag KV_TYPE, bool WRITE_CONTEXT_PARTIAL = false>
static int launch_device_resident_prefill(
    const float *Q,
    const void *K,
    const void *V,
    float *O,
    int batch_size,
    int seq_len,
    int kv_len,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    bool causal,
    int window_size,
    int position_offset,
    const llaminar2::attention::AttentionDeviceParams *device_params,
    const float *mask,
    hipStream_t hip_stream,
    int head_start,
    int gqa_n_rep,
    float *O_partial = nullptr,
    float *m_partial = nullptr,
    float *l_partial = nullptr,
    int context_partition_size = 0,
    int max_context_partitions = 0,
    int context_partition_slots = 0,
    int context_phase_block_slots = 0,
    int device_direct_partition_limit = 0,
    int reducer_dimension_wavefronts = 0,
    int reducer_block_slots = 0)
{
    if (!Q || !K || !V || !O || !hip_stream ||
        batch_size <= 0 || seq_len <= 0 ||
        kv_len <= 0 || n_heads <= 0 || n_kv_heads <= 0 ||
        head_dim <= 0 || head_dim > 256)
    {
        printf("[launch_device_resident_prefill] Invalid pointers or geometry\n");
        return -1;
    }
    const bool valid_head_mapping =
        gqa_n_rep > 0
            ? head_start >= 0 &&
                  static_cast<long long>(head_start) + n_heads <=
                      static_cast<long long>(n_kv_heads) * gqa_n_rep
            : n_heads % n_kv_heads == 0;
    if (!valid_head_mapping)
    {
        printf("[launch_device_resident_prefill] Invalid local or replicated GQA head mapping\n");
        return -1;
    }
    if constexpr (KV_TYPE == KVTypeTag::Q8_1)
    {
        if (head_dim % static_cast<int>(llaminar2::Q8_1Block::BLOCK_SIZE) != 0)
        {
            printf("[launch_device_resident_prefill] Q8_1 head_dim must be a multiple of 32\n");
            return -1;
        }
    }

    if constexpr (WRITE_CONTEXT_PARTIAL)
    {
        const int expected_partitions =
            (kv_len +
             llaminar2::rocm::fa2_policy::
                 kROCmFA2CanonicalContextPartitionKeys -
             1) /
            llaminar2::rocm::fa2_policy::
                kROCmFA2CanonicalContextPartitionKeys;
        if (!O_partial || !m_partial || !l_partial || !device_params ||
            context_partition_size !=
                llaminar2::rocm::fa2_policy::
                    kROCmFA2CanonicalContextPartitionKeys ||
            max_context_partitions != expected_partitions ||
            context_partition_slots <= 0 ||
            context_partition_slots > max_context_partitions ||
            context_phase_block_slots <= 0 ||
            device_direct_partition_limit <= 0 ||
            device_direct_partition_limit > max_context_partitions ||
            (reducer_dimension_wavefronts != 1 &&
             reducer_dimension_wavefronts != 2 &&
             reducer_dimension_wavefronts != 4))
        {
            printf("[launch_device_resident_prefill] Invalid context transaction geometry\n");
            return -1;
        }
        constexpr int wavefronts_per_block =
            llaminar2::rocm::fa2_policy::kROCmFA2ThreadsPerBlock /
            llaminar2::rocm::fa2_policy::kROCmFA2WavefrontSize;
        const int rows_per_block =
            wavefronts_per_block / reducer_dimension_wavefronts;
        const long long logical_reducer_blocks =
            static_cast<long long>(batch_size) * n_heads *
            ((static_cast<long long>(seq_len) + rows_per_block - 1) /
             rows_per_block);
        if (logical_reducer_blocks <= 0 ||
            logical_reducer_blocks > std::numeric_limits<int>::max() ||
            reducer_block_slots <= 0 ||
            reducer_block_slots > logical_reducer_blocks)
        {
            printf("[launch_device_resident_prefill] Invalid persistent reducer grid\n");
            return -1;
        }
    }

    const auto tile_plan =
        llaminar2::rocm::fa2_policy::selectROCmFA2Tile(
            head_dim,
            LDS_SIZE_BYTES);
    if (!tile_plan.valid)
    {
        printf("[launch_device_resident_prefill] No LDS tile fits head_dim=%d\n", head_dim);
        return -1;
    }
    const int tile_q = tile_plan.query_rows;
    const int tile_kv = tile_plan.kv_rows;
    const int threads_per_row = tile_plan.threads_per_row;
    FA2LDSLayout lds_layout{};
    lds_layout.compute(head_dim, tile_q, tile_kv);
    if (static_cast<size_t>(lds_layout.total_size) != tile_plan.lds_bytes)
    {
        printf("[launch_device_resident_prefill] Policy/kernel LDS layout mismatch\n");
        return -1;
    }

    const float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    const int num_q_tiles = (seq_len + tile_q - 1) / tile_q;
    if constexpr (WRITE_CONTEXT_PARTIAL)
    {
        /*
         * Capture the exact direct schedule before the partition transaction.
         * Its kernel reads the same device-owned live length and returns before
         * touching output when that length exceeds the immutable direct span.
         * The following phase and reducer apply the opposite predicate. This
         * three-node transaction gives HIP conditional execution without host
         * dispatch, graph recapture, atomics, or a merged high-register kernel.
         */
        const int direct_status =
            launch_device_resident_prefill<KV_TYPE, false>(
                Q, K, V, O,
                batch_size, seq_len, kv_len,
                n_heads, n_kv_heads, head_dim,
                causal, window_size, position_offset,
                device_params, mask,
                hip_stream,
                head_start, gqa_n_rep,
                nullptr, nullptr, nullptr,
                0,
                max_context_partitions,
                0,
                0,
                device_direct_partition_limit,
                0,
                0);
        if (direct_status != 0)
        {
            printf("[launch_device_resident_prefill] Guarded direct launch failed\n");
            return -1;
        }

        const long long logical_phase_blocks =
            static_cast<long long>(batch_size) * n_heads * num_q_tiles *
            context_partition_slots;
        if (logical_phase_blocks <= 0 ||
            logical_phase_blocks > std::numeric_limits<int>::max() ||
            context_phase_block_slots > logical_phase_blocks)
        {
            printf("[launch_device_resident_prefill] Invalid persistent context phase grid\n");
            return -1;
        }
    }
    const dim3 grid(
        WRITE_CONTEXT_PARTIAL
            ? context_phase_block_slots
            : n_heads,
        WRITE_CONTEXT_PARTIAL ? 1 : num_q_tiles,
        WRITE_CONTEXT_PARTIAL ? 1 : batch_size);
    const dim3 block(
        llaminar2::rocm::fa2_policy::kROCmFA2ThreadsPerBlock);

    // Discard an error left by an earlier asynchronous operation before the
    // launch, then report only the status produced by this kernel.
    (void)hipGetLastError();
    if (head_dim == 64 && tile_q == 64 && tile_kv == 64)
    {
        hipLaunchKernelGGL(
            (flash_attention_2_mi50_kernel<
                KV_TYPE, 4, 64, 64, 64, WRITE_CONTEXT_PARTIAL>),
            grid, block, lds_layout.total_size, hip_stream,
            Q, K, V, O,
            batch_size, seq_len, kv_len,
            n_heads, n_kv_heads, head_dim,
            softmax_scale, causal, window_size, position_offset,
            device_params, mask,
            tile_q, tile_kv,
            head_start, gqa_n_rep,
            O_partial, m_partial, l_partial,
            context_partition_size,
            max_context_partitions,
            context_partition_slots,
            device_direct_partition_limit);
    }
    else if (head_dim == 128 && tile_q == 64 && tile_kv == 32)
    {
        hipLaunchKernelGGL(
            (flash_attention_2_mi50_kernel<
                KV_TYPE, 4, 128, 64, 32, WRITE_CONTEXT_PARTIAL>),
            grid, block, lds_layout.total_size, hip_stream,
            Q, K, V, O,
            batch_size, seq_len, kv_len,
            n_heads, n_kv_heads, head_dim,
            softmax_scale, causal, window_size, position_offset,
            device_params, mask,
            tile_q, tile_kv,
            head_start, gqa_n_rep,
            O_partial, m_partial, l_partial,
            context_partition_size,
            max_context_partitions,
            context_partition_slots,
            device_direct_partition_limit);
    }
    else if (threads_per_row == 4)
    {
        hipLaunchKernelGGL(
            (flash_attention_2_mi50_kernel<
                KV_TYPE, 4, 0, 0, 0, WRITE_CONTEXT_PARTIAL>),
            grid, block, lds_layout.total_size, hip_stream,
            Q, K, V, O,
            batch_size, seq_len, kv_len,
            n_heads, n_kv_heads, head_dim,
            softmax_scale, causal, window_size, position_offset,
            device_params, mask,
            tile_q, tile_kv,
            head_start, gqa_n_rep,
            O_partial, m_partial, l_partial,
            context_partition_size,
            max_context_partitions,
            context_partition_slots,
            device_direct_partition_limit);
    }
    else if (head_dim == 256 && tile_q == 16 && tile_kv == 16)
    {
        hipLaunchKernelGGL(
            (flash_attention_2_mi50_kernel<
                KV_TYPE, 16, 256, 16, 16, WRITE_CONTEXT_PARTIAL>),
            grid, block, lds_layout.total_size, hip_stream,
            Q, K, V, O,
            batch_size, seq_len, kv_len,
            n_heads, n_kv_heads, head_dim,
            softmax_scale, causal, window_size, position_offset,
            device_params, mask,
            tile_q, tile_kv,
            head_start, gqa_n_rep,
            O_partial, m_partial, l_partial,
            context_partition_size,
            max_context_partitions,
            context_partition_slots,
            device_direct_partition_limit);
    }
    else
    {
        hipLaunchKernelGGL(
            (flash_attention_2_mi50_kernel<
                KV_TYPE, 16, 0, 0, 0, WRITE_CONTEXT_PARTIAL>),
            grid, block, lds_layout.total_size, hip_stream,
            Q, K, V, O,
            batch_size, seq_len, kv_len,
            n_heads, n_kv_heads, head_dim,
            softmax_scale, causal, window_size, position_offset,
            device_params, mask,
            tile_q, tile_kv,
            head_start, gqa_n_rep,
            O_partial, m_partial, l_partial,
            context_partition_size,
            max_context_partitions,
            context_partition_slots,
            device_direct_partition_limit);
    }

    hipError_t error = hipGetLastError();
    if (error != hipSuccess)
    {
        printf("[launch_device_resident_prefill] Kernel launch failed: %s\n",
               hipGetErrorString(error));
        return -1;
    }

    if constexpr (WRITE_CONTEXT_PARTIAL)
    {
        const dim3 reducer_grid(reducer_block_slots);
        if (reducer_dimension_wavefronts == 1)
        {
            hipLaunchKernelGGL(
                (flash_attention_2_context_reduce_mi50_kernel<1>),
                reducer_grid, block, 0, hip_stream,
                O_partial, m_partial, l_partial, O,
                batch_size, seq_len, n_heads, head_dim,
                max_context_partitions,
                device_direct_partition_limit,
                device_params);
        }
        else if (reducer_dimension_wavefronts == 2)
        {
            hipLaunchKernelGGL(
                (flash_attention_2_context_reduce_mi50_kernel<2>),
                reducer_grid, block, 0, hip_stream,
                O_partial, m_partial, l_partial, O,
                batch_size, seq_len, n_heads, head_dim,
                max_context_partitions,
                device_direct_partition_limit,
                device_params);
        }
        else
        {
            hipLaunchKernelGGL(
                (flash_attention_2_context_reduce_mi50_kernel<4>),
                reducer_grid, block, 0, hip_stream,
                O_partial, m_partial, l_partial, O,
                batch_size, seq_len, n_heads, head_dim,
                max_context_partitions,
                device_direct_partition_limit,
                device_params);
        }
        error = hipGetLastError();
        if (error != hipSuccess)
        {
            printf("[launch_device_resident_prefill] Context reducer launch failed: %s\n",
                   hipGetErrorString(error));
            return -1;
        }
    }
    return 0;
}

// =============================================================================
// Flash Decoding launch template (outside extern "C" — templates need C++ linkage)
// KV_TYPE selects native K/V format:
//   FP16: half bandwidth, fdot2 dot product on gfx906
//   Q8_1: inline dequant, ~3.5× BW savings
//   FP32: original path (backward-compatible)
// =============================================================================
template <KVTypeTag KV_TYPE>
static int launch_decode(
    const float *Q, const void *K_cache, const void *V_cache, float *O,
    float *O_partial, float *m_partial, float *l_partial,
    int batch_size, int kv_len,
    int n_heads, int n_kv_heads, int head_dim,
    int num_splits,
    const llaminar2::attention::AttentionDeviceParams *device_params,
    hipStream_t hip_stream,
    int head_start = 0, int gqa_n_rep = 0)
{
    if (!Q || !K_cache || !V_cache || !O || !O_partial || !m_partial ||
        !l_partial || !hip_stream || batch_size <= 0 || kv_len <= 0 ||
        n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 ||
        head_dim > 256 || num_splits <= 0 ||
        num_splits > FD_MAX_SPLITS_SUPPORTED)
    {
        printf("[launch_decode] Invalid stable decode geometry: "
               "batch=%d kv_len=%d heads=%d kv_heads=%d head_dim=%d "
               "split_envelope=%d stream=%p\n",
               batch_size,
               kv_len,
               n_heads,
               n_kv_heads,
               head_dim,
               num_splits,
               static_cast<void *>(hip_stream));
        return -1;
    }

    const float softmax_scale =
        1.0f / sqrtf(static_cast<float>(head_dim));

    int current_device = 0;
    (void)hipGetDevice(&current_device);
    const int num_cus = getDeviceConfig(current_device).num_cus;
    const auto &rocm_env = llaminar2::debugEnv().rocm;
    const bool forced_splits_present =
        rocm_env.fa_decode_num_splits_present &&
        rocm_env.fa_decode_num_splits.has_value();
    const int forced_splits =
        forced_splits_present ? *rocm_env.fa_decode_num_splits : 0;
    const bool forced_tpb_present =
        rocm_env.fa_decode_num_splits_present &&
        rocm_env.fa_decode_tpb.has_value();
    const int forced_tpb =
        forced_tpb_present ? *rocm_env.fa_decode_tpb : 0;

    // Split-K kernel (type-dispatched via template)
    // FP16 path stores Q as half in shared memory (halves LDS usage + bandwidth)
    size_t smem_size;
    if constexpr (KV_TYPE == KVTypeTag::FP16) {
        smem_size = head_dim * sizeof(__half);
    } else {
        smem_size = head_dim * sizeof(float);
    }
    const dim3 grid_split(n_heads, num_splits, batch_size);
    constexpr int physical_threads = 256;

    hipLaunchKernelGGL(
        (flash_decoding_mi50_kernel<KV_TYPE>),
        grid_split, dim3(physical_threads), smem_size, hip_stream,
        Q, K_cache, V_cache,
        O_partial, m_partial, l_partial,
        kv_len, n_heads, n_kv_heads, head_dim,
        num_splits, softmax_scale, device_params,
        FlashDecodeRowMode::OrdinaryBatch,
        /*query_rows_per_request=*/1,
        num_cus,
        forced_splits_present,
        forced_splits,
        forced_tpb_present,
        forced_tpb,
        head_start, gqa_n_rep);

    // Reduction kernel (unchanged — operates on FP32 partials only)
    const int output_tiles = head_dim >= 256 ? 2 : 1;
    dim3 grid_reduce(n_heads, batch_size, output_tiles);
    dim3 block_reduce(64);

    hipLaunchKernelGGL(
        flash_decoding_reduce_mi50_kernel,
        grid_reduce, block_reduce, 0, hip_stream,
        O_partial, m_partial, l_partial, O,
        n_heads, head_dim, num_splits,
        kv_len,
        device_params,
        FlashDecodeRowMode::OrdinaryBatch,
        num_cus,
        forced_splits_present,
        forced_splits,
        forced_tpb_present,
        forced_tpb);

    (void)hipGetLastError();  // Clear stale errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        printf("[launch_decode] Kernel launch failed: %s\n", hipGetErrorString(err));
        return -1;
    }

    return 0;
}

/**
 * @brief Launch one economical decode phase and reduction for grouped rows.
 *
 * Unlike an ordinary batch, grouped verifier rows may share one K/V bank and
 * grouped request rows may map several query rows to each independent bank.
 * Device-resident metadata supplies every row's logical KV length.  The phase
 * grid uses the longest row's split cap and a 256-thread physical block, while
 * the kernel activates only the split and wavefront prefix selected by the
 * ordinary scalar ROCm policy for that row.  This preserves byte identity
 * without host count reads or one launch pair per row.
 *
 * Explicit fixed split/TPB overrides remain available for offline profiling.
 * Production never rotates candidates online; the device-owned row policy is
 * deterministic for a given geometry and live KV length.
 */
template <KVTypeTag KV_TYPE, FlashDecodeRowMode ROW_MODE>
static int launch_grouped_decode(
    const float *Q,
    const void *K_cache,
    const void *V_cache,
    float *O,
    float *O_partial,
    float *m_partial,
    float *l_partial,
    int row_count,
    int query_rows_per_request,
    int max_kv_len,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_num_splits,
    const llaminar2::attention::AttentionDeviceParams *device_params,
    hipStream_t hip_stream,
    int head_start,
    int gqa_n_rep)
{
    static_assert(
        ROW_MODE != FlashDecodeRowMode::OrdinaryBatch,
        "Grouped launcher requires a row-local cache ownership policy");

    if (!Q || !K_cache || !V_cache || !O || !O_partial || !m_partial ||
        !l_partial || !device_params || !hip_stream || row_count < 2 ||
        row_count >
            llaminar2::attention::kMaxGroupedVerifierAttentionRows ||
        query_rows_per_request <= 0 || max_kv_len <= 0 ||
        n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || head_dim > 256 ||
        max_num_splits <= 0 || max_num_splits > FD_MAX_SPLITS_SUPPORTED)
    {
        printf("[launch_grouped_decode] Invalid grouped decode contract: "
               "rows=%d query_rows=%d max_kv_len=%d heads=%d kv_heads=%d "
               "head_dim=%d max_splits=%d stream=%p\n",
               row_count,
               query_rows_per_request,
               max_kv_len,
               n_heads,
               n_kv_heads,
               head_dim,
               max_num_splits,
               static_cast<void *>(hip_stream));
        return -1;
    }

    const auto &rocm_env = llaminar2::debugEnv().rocm;
    const bool forced_splits_present =
        rocm_env.fa_decode_num_splits_present &&
        rocm_env.fa_decode_num_splits.has_value();
    const int forced_splits =
        forced_splits_present ? *rocm_env.fa_decode_num_splits : 0;
    const bool forced_tpb_present =
        rocm_env.fa_decode_num_splits_present &&
        rocm_env.fa_decode_tpb.has_value();
    const int forced_tpb =
        forced_tpb_present ? *rocm_env.fa_decode_tpb : 0;

    int current_device = 0;
    (void)hipGetDevice(&current_device);
    const int num_cus = getDeviceConfig(current_device).num_cus;
    const float softmax_scale =
        1.0f / sqrtf(static_cast<float>(head_dim));
    const size_t smem_size =
        KV_TYPE == KVTypeTag::FP16
            ? static_cast<size_t>(head_dim) * sizeof(__half)
            : static_cast<size_t>(head_dim) * sizeof(float);

    /* Clear a pre-existing asynchronous launch error before attributing one. */
    (void)hipGetLastError();

    const dim3 split_grid(n_heads, max_num_splits, row_count);
    constexpr int physical_threads = 256;
    hipLaunchKernelGGL(
        (flash_decoding_mi50_kernel<KV_TYPE>),
        split_grid,
        dim3(physical_threads),
        smem_size,
        hip_stream,
        Q,
        K_cache,
        V_cache,
        O_partial,
        m_partial,
        l_partial,
        max_kv_len,
        n_heads,
        n_kv_heads,
        head_dim,
        max_num_splits,
        softmax_scale,
        device_params,
        ROW_MODE,
        query_rows_per_request,
        num_cus,
        forced_splits_present,
        forced_splits,
        forced_tpb_present,
        forced_tpb,
        head_start,
        gqa_n_rep);
    hipError_t error = hipGetLastError();
    if (error != hipSuccess)
    {
        printf("[launch_grouped_decode] Decode phase launch failed: %s\n",
               hipGetErrorString(error));
        return -1;
    }

    const int output_tiles = head_dim >= 256 ? 2 : 1;
    const dim3 reduce_grid(n_heads, row_count, output_tiles);
    hipLaunchKernelGGL(
        flash_decoding_reduce_mi50_kernel,
        reduce_grid,
        dim3(64),
        0,
        hip_stream,
        O_partial,
        m_partial,
        l_partial,
        O,
        n_heads,
        head_dim,
        max_num_splits,
        max_kv_len,
        device_params,
        ROW_MODE,
        num_cus,
        forced_splits_present,
        forced_splits,
        forced_tpb_present,
        forced_tpb);
    error = hipGetLastError();
    if (error != hipSuccess)
    {
        printf("[launch_grouped_decode] Reduction launch failed: %s\n",
               hipGetErrorString(error));
        return -1;
    }

    return 0;
}

// =============================================================================
// Extern "C" Wrapper Functions
// =============================================================================

extern "C"
{
    /**
     * @brief Launch Flash Attention 2 prefill kernel for MI50 (FP32 KV)
     *
     * @return 0 on success, -1 on invalid input, -2 on unsupported arch
     */
    int hipFlashAttn_prefill_fa2(
        const float *Q, const float *K, const float *V, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        return launch_device_resident_prefill<KVTypeTag::FP32>(
            Q, K, V, O,
            batch_size, seq_len, kv_len,
            n_heads, n_kv_heads, head_dim,
            causal, window_size, position_offset,
            device_params, mask,
            static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch Flash Attention 2 prefill kernel with FP16 KV cache
     *
     * Reads K/V directly as FP16 from device memory, eliminating the
     * FP32 conversion kernel. For head_dim >= 64 (DIMS_PER_LANE=2), uses
     * v_dot2_f32_f16 for 2× ALU throughput on the Q@K^T dot product.
     * Halves HBM bandwidth for K/V reads vs FP32.
     *
     * @return 0 on success, -1 on error
     */
    int hipFlashAttn_prefill_fa2_fp16(
        const float *Q, const void *K, const void *V, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        return launch_device_resident_prefill<KVTypeTag::FP16>(
            Q, K, V, O,
            batch_size, seq_len, kv_len,
            n_heads, n_kv_heads, head_dim,
            causal, window_size, position_offset,
            device_params, mask,
            static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch device-resident prefill over a native BF16 KV cache.
     *
     * BF16 values remain two-byte resident data through HBM and are converted
     * only while the current tile enters LDS.  This avoids the packed FP32
     * shadow whose request stride could not follow a growing captured cache.
     */
    int hipFlashAttn_prefill_fa2_bf16(
        const float *Q, const void *K, const void *V, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        return launch_device_resident_prefill<KVTypeTag::BF16>(
            Q, K, V, O,
            batch_size, seq_len, kv_len,
            n_heads, n_kv_heads, head_dim,
            causal, window_size, position_offset,
            device_params, mask,
            static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch Flash Attention 2 prefill kernel with Q8_1 KV cache
     *
     * Reads K/V directly as Q8_1Block from device memory, performing inline
     * dequantization (scale × int8 → float) per element. Eliminates the
     * FP32 conversion kernel and reduces HBM bandwidth ~3.5× vs FP32.
     *
     * @return 0 on success, -1 on error
     */
    int hipFlashAttn_prefill_fa2_q8_1(
        const float *Q, const void *K, const void *V, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        return launch_device_resident_prefill<KVTypeTag::Q8_1>(
            Q, K, V, O,
            batch_size, seq_len, kv_len,
            n_heads, n_kv_heads, head_dim,
            causal, window_size, position_offset,
            device_params, mask,
            static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch captured context-parallel FA2 over an FP32 K/V cache.
     *
     * The guarded direct kernel, guarded phase, and guarded reducer are
     * submitted to the same explicit stream and therefore captured as one
     * ordered, device-owned transaction.
     */
    int hipFlashAttn_prefill_fa2_context_parallel(
        const float *Q, const void *K, const void *V, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int batch_size, int seq_len, int kv_capacity,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask,
        int context_partition_size,
        int max_context_partitions,
        int context_partition_slots,
        int context_phase_block_slots,
        int device_direct_partition_limit,
        int reducer_dimension_wavefronts,
        int reducer_block_slots,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        return launch_device_resident_prefill<
            KVTypeTag::FP32,
            true>(
            Q, K, V, O,
            batch_size, seq_len, kv_capacity,
            n_heads, n_kv_heads, head_dim,
            causal, window_size, position_offset,
            device_params, mask,
            static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep,
            O_partial, m_partial, l_partial,
            context_partition_size,
            max_context_partitions,
            context_partition_slots,
            context_phase_block_slots,
            device_direct_partition_limit,
            reducer_dimension_wavefronts,
            reducer_block_slots);
    }

    /** @brief Launch captured context-parallel FA2 over native FP16 K/V. */
    int hipFlashAttn_prefill_fa2_fp16_context_parallel(
        const float *Q, const void *K, const void *V, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int batch_size, int seq_len, int kv_capacity,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask,
        int context_partition_size,
        int max_context_partitions,
        int context_partition_slots,
        int context_phase_block_slots,
        int device_direct_partition_limit,
        int reducer_dimension_wavefronts,
        int reducer_block_slots,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        return launch_device_resident_prefill<
            KVTypeTag::FP16,
            true>(
            Q, K, V, O,
            batch_size, seq_len, kv_capacity,
            n_heads, n_kv_heads, head_dim,
            causal, window_size, position_offset,
            device_params, mask,
            static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep,
            O_partial, m_partial, l_partial,
            context_partition_size,
            max_context_partitions,
            context_partition_slots,
            context_phase_block_slots,
            device_direct_partition_limit,
            reducer_dimension_wavefronts,
            reducer_block_slots);
    }

    /** @brief Launch captured context-parallel FA2 over native BF16 K/V. */
    int hipFlashAttn_prefill_fa2_bf16_context_parallel(
        const float *Q, const void *K, const void *V, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int batch_size, int seq_len, int kv_capacity,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask,
        int context_partition_size,
        int max_context_partitions,
        int context_partition_slots,
        int context_phase_block_slots,
        int device_direct_partition_limit,
        int reducer_dimension_wavefronts,
        int reducer_block_slots,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        return launch_device_resident_prefill<
            KVTypeTag::BF16,
            true>(
            Q, K, V, O,
            batch_size, seq_len, kv_capacity,
            n_heads, n_kv_heads, head_dim,
            causal, window_size, position_offset,
            device_params, mask,
            static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep,
            O_partial, m_partial, l_partial,
            context_partition_size,
            max_context_partitions,
            context_partition_slots,
            context_phase_block_slots,
            device_direct_partition_limit,
            reducer_dimension_wavefronts,
            reducer_block_slots);
    }

    /** @brief Launch captured context-parallel FA2 over native Q8_1 K/V. */
    int hipFlashAttn_prefill_fa2_q8_1_context_parallel(
        const float *Q, const void *K, const void *V, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int batch_size, int seq_len, int kv_capacity,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask,
        int context_partition_size,
        int max_context_partitions,
        int context_partition_slots,
        int context_phase_block_slots,
        int device_direct_partition_limit,
        int reducer_dimension_wavefronts,
        int reducer_block_slots,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        return launch_device_resident_prefill<
            KVTypeTag::Q8_1,
            true>(
            Q, K, V, O,
            batch_size, seq_len, kv_capacity,
            n_heads, n_kv_heads, head_dim,
            causal, window_size, position_offset,
            device_params, mask,
            static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep,
            O_partial, m_partial, l_partial,
            context_partition_size,
            max_context_partitions,
            context_partition_slots,
            context_phase_block_slots,
            device_direct_partition_limit,
            reducer_dimension_wavefronts,
            reducer_block_slots);
    }

    /**
     * @brief Launch Flash Decoding kernel for MI50 (FP32 KV cache)
     */
    int hipFlashAttn_decode_fp32(
        const float *Q, const float *K_cache, const float *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int batch_size, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        return launch_decode<KVTypeTag::FP32>(
            Q, K_cache, V_cache, O,
            O_partial, m_partial, l_partial,
            batch_size, kv_len, n_heads, n_kv_heads, head_dim,
            num_splits, device_params, static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch Flash Decoding kernel for MI50 (FP16 KV cache — native half reads)
     *
     * K_cache/V_cache point to __half data in the same logical layout as FP32.
     * Halves KV read bandwidth; uses fdot2 (v_dot2_f32_f16) for K·Q dot product.
     */
    int hipFlashAttn_decode_fp16(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int batch_size, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        return launch_decode<KVTypeTag::FP16>(
            Q, K_cache, V_cache, O,
            O_partial, m_partial, l_partial,
            batch_size, kv_len, n_heads, n_kv_heads, head_dim,
            num_splits, device_params, static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch ordinary split-K decode with native BF16 K/V reads.
     */
    int hipFlashAttn_decode_bf16(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int batch_size, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        return launch_decode<KVTypeTag::BF16>(
            Q, K_cache, V_cache, O,
            O_partial, m_partial, l_partial,
            batch_size, kv_len, n_heads, n_kv_heads, head_dim,
            num_splits, device_params, static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch Flash Decoding kernel for MI50 (Q8_1 KV cache — inline dequant)
     *
     * K_cache/V_cache point to Q8_1Block data. Each block covers 32 elements.
     * ~3.5× bandwidth savings vs FP32. head_dim must be multiple of 32.
     */
    int hipFlashAttn_decode_q8_1(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int batch_size, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        if (head_dim % 32 != 0)
        {
            printf("[hipFlashAttn_decode_q8_1] Error: head_dim must be multiple of 32 for Q8_1 (got %d)\n", head_dim);
            return -1;
        }
        return launch_decode<KVTypeTag::Q8_1>(
            Q, K_cache, V_cache, O,
            O_partial, m_partial, l_partial,
            batch_size, kv_len, n_heads, n_kv_heads, head_dim,
            num_splits, device_params, static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch shared-cache verifier rows through one grouped FP32 decode.
     */
    int hipFlashAttn_decode_fp32_grouped_verifier_rows(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int verifier_rows, int max_kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int max_num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        return launch_grouped_decode<
            KVTypeTag::FP32,
            FlashDecodeRowMode::SharedKVVerifier>(
            Q, K_cache, V_cache, O,
            O_partial, m_partial, l_partial,
            verifier_rows,
            /*query_rows_per_request=*/1,
            max_kv_len,
            n_heads, n_kv_heads, head_dim,
            max_num_splits, device_params,
            static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch shared-cache verifier rows through one grouped FP16 decode.
     */
    int hipFlashAttn_decode_fp16_grouped_verifier_rows(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int verifier_rows, int max_kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int max_num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        return launch_grouped_decode<
            KVTypeTag::FP16,
            FlashDecodeRowMode::SharedKVVerifier>(
            Q, K_cache, V_cache, O,
            O_partial, m_partial, l_partial,
            verifier_rows,
            /*query_rows_per_request=*/1,
            max_kv_len,
            n_heads, n_kv_heads, head_dim,
            max_num_splits, device_params,
            static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch shared-cache verifier rows with native BF16 K/V reads.
     */
    int hipFlashAttn_decode_bf16_grouped_verifier_rows(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int verifier_rows, int max_kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int max_num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        return launch_grouped_decode<
            KVTypeTag::BF16,
            FlashDecodeRowMode::SharedKVVerifier>(
            Q, K_cache, V_cache, O,
            O_partial, m_partial, l_partial,
            verifier_rows,
            /*query_rows_per_request=*/1,
            max_kv_len,
            n_heads, n_kv_heads, head_dim,
            max_num_splits, device_params,
            static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch shared-cache verifier rows through one grouped Q8_1 decode.
     */
    int hipFlashAttn_decode_q8_1_grouped_verifier_rows(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int verifier_rows, int max_kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int max_num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        if (head_dim % 32 != 0)
            return -1;
        return launch_grouped_decode<
            KVTypeTag::Q8_1,
            FlashDecodeRowMode::SharedKVVerifier>(
            Q, K_cache, V_cache, O,
            O_partial, m_partial, l_partial,
            verifier_rows,
            /*query_rows_per_request=*/1,
            max_kv_len,
            n_heads, n_kv_heads, head_dim,
            max_num_splits, device_params,
            static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch independent FP32 request banks through one grouped decode.
     */
    int hipFlashAttn_decode_fp32_grouped_request_rows(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int request_count, int query_rows, int max_kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int max_num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        return launch_grouped_decode<
            KVTypeTag::FP32,
            FlashDecodeRowMode::IndependentRequest>(
            Q, K_cache, V_cache, O,
            O_partial, m_partial, l_partial,
            request_count * query_rows,
            query_rows,
            max_kv_len,
            n_heads, n_kv_heads, head_dim,
            max_num_splits, device_params,
            static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch independent FP16 request banks through one grouped decode.
     */
    int hipFlashAttn_decode_fp16_grouped_request_rows(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int request_count, int query_rows, int max_kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int max_num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        return launch_grouped_decode<
            KVTypeTag::FP16,
            FlashDecodeRowMode::IndependentRequest>(
            Q, K_cache, V_cache, O,
            O_partial, m_partial, l_partial,
            request_count * query_rows,
            query_rows,
            max_kv_len,
            n_heads, n_kv_heads, head_dim,
            max_num_splits, device_params,
            static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch independent request banks with native BF16 K/V reads.
     */
    int hipFlashAttn_decode_bf16_grouped_request_rows(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int request_count, int query_rows, int max_kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int max_num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        return launch_grouped_decode<
            KVTypeTag::BF16,
            FlashDecodeRowMode::IndependentRequest>(
            Q, K_cache, V_cache, O,
            O_partial, m_partial, l_partial,
            request_count * query_rows,
            query_rows,
            max_kv_len,
            n_heads, n_kv_heads, head_dim,
            max_num_splits, device_params,
            static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch independent Q8_1 request banks through one grouped decode.
     */
    int hipFlashAttn_decode_q8_1_grouped_request_rows(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int request_count, int query_rows, int max_kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int max_num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep)
    {
        if (head_dim % 32 != 0)
            return -1;
        return launch_grouped_decode<
            KVTypeTag::Q8_1,
            FlashDecodeRowMode::IndependentRequest>(
            Q, K_cache, V_cache, O,
            O_partial, m_partial, l_partial,
            request_count * query_rows,
            query_rows,
            max_kv_len,
            n_heads, n_kv_heads, head_dim,
            max_num_splits, device_params,
            static_cast<hipStream_t>(stream),
            head_start, gqa_n_rep);
    }

    /**
     * @brief Derive AttentionDeviceParams from device-owned KV cache count.
     */
    int hipFlashAttn_prepare_device_params_from_count(
        void *device_params,
        const int *post_append_cached_tokens,
        int seq_len,
        int query_rows,
        int kv_stride,
        const int *active_query_rows_device,
        const int *ring_head_device,
        int ring_capacity,
        void *stream)
    {
        if (!device_params || !post_append_cached_tokens || seq_len <= 0 ||
            query_rows <= 0 || kv_stride <= 0 ||
            query_rows > MAX_DYNAMIC_ATTENTION_PARAM_ROWS || !stream ||
            ((ring_head_device == nullptr) != (ring_capacity == 0)) ||
            (ring_capacity > 0 && ring_capacity != kv_stride))
        {
            return -1;
        }

        hipLaunchKernelGGL(hip_derive_attention_params_from_cached_tokens_kernel,
                           dim3(1), dim3(query_rows), 0, static_cast<hipStream_t>(stream),
                           static_cast<llaminar2::attention::AttentionDeviceParams *>(device_params),
                           post_append_cached_tokens,
                           seq_len,
                           query_rows,
                           kv_stride,
                           active_query_rows_device,
                           ring_head_device,
                           ring_capacity);
        const hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            printf("[hipFlashAttn_prepare_device_params_from_count] Kernel launch failed: %s\n",
                   hipGetErrorString(err));
            return -1;
        }
        return 0;
    }

    /**
     * @brief Write explicit decode geometry to the device-owned param block.
     */
    int hipFlashAttn_prepare_device_params_from_geometry(
        void *device_params,
        int kv_len,
        int kv_stride,
        int position_offset,
        int query_rows,
        void *stream)
    {
        if (!device_params || kv_len <= 0 || kv_stride < kv_len ||
            position_offset < 0 ||
            query_rows <= 0 ||
            query_rows > MAX_DYNAMIC_ATTENTION_PARAM_ROWS || !stream)
        {
            return -1;
        }

        hipLaunchKernelGGL(
            hip_write_attention_params_from_geometry_kernel,
            dim3(1),
            dim3(query_rows),
            0,
            static_cast<hipStream_t>(stream),
            static_cast<llaminar2::attention::AttentionDeviceParams *>(device_params),
            kv_len,
            kv_stride,
            position_offset,
            query_rows);
        const hipError_t error = hipGetLastError();
        if (error != hipSuccess)
        {
            printf("[hipFlashAttn_prepare_device_params_from_geometry] "
                   "Kernel launch failed: %s\n",
                   hipGetErrorString(error));
            return -1;
        }
        return 0;
    }

    /**
     * @brief Derive grouped request-row params from contiguous device counts.
     */
    int hipFlashAttn_prepare_device_params_from_request_counts(
        void *device_params,
        const int *post_append_cached_tokens,
        int request_count,
        int query_rows,
        int kv_stride,
        void *stream)
    {
        const int total_rows = request_count * query_rows;
        if (!device_params || !post_append_cached_tokens || request_count < 2 ||
            query_rows <= 0 ||
            total_rows >
                llaminar2::attention::kMaxGroupedVerifierAttentionRows ||
            kv_stride <= 0 || !stream)
        {
            return -1;
        }

        hipLaunchKernelGGL(
            hip_derive_attention_params_from_request_counts_kernel,
            dim3(1),
            dim3(total_rows),
            0,
            static_cast<hipStream_t>(stream),
            static_cast<llaminar2::attention::AttentionDeviceParams *>(device_params),
            post_append_cached_tokens,
            request_count,
            query_rows,
            kv_stride);
        const hipError_t error = hipGetLastError();
        if (error != hipSuccess)
        {
            printf("[hipFlashAttn_prepare_device_params_from_request_counts] "
                   "Kernel launch failed: %s\n",
                   hipGetErrorString(error));
            return -1;
        }
        return 0;
    }

    /**
     * @brief Set active HIP device
     */
    int hipFlashAttn_setDevice(int device_idx)
    {
        return llaminar2::HipDeviceGuard::setDevice(device_idx);
    }

    // hipFlashAttn_synchronize() removed - caller manages coherence via events
}
