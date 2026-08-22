/**
 * @file CUDAFlashAttentionKernels.cu
 * @brief CUDA device kernels for Flash Attention 2 (Ampere) and Flash Decoding
 *
 * This file contains the CUDA device kernels and extern "C" wrapper functions
 * called from the C++ implementation file.
 *
 * Algorithms implemented:
 * - Flash Attention 2 with Pipelined Prefetching: Optimized for Ampere (SM >= 8.0)
 *   - Uses dedicated producer warps to overlap global-to-shared K/V loads
 *   - Double-buffered shared memory for K/V tiles
 *   - Producer/consumer warp specialization
 *   - WMMA (Tensor Core) acceleration for Q @ K^T matmul
 *   - Adaptive tile sizing for head_dim=64, head_dim=128, and head_dim=256
 * - Flash Decoding: Split-K parallelism for single-token decode
 *
 *
 * @author David Sanftenberg
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h> // WMMA (Warp Matrix Multiply-Accumulate) for Tensor Cores
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <atomic>

#include "../../../backends/cuda/CUDAGraphCapture.h"
#include "../../attention/AttentionDeviceParams.h"
#include "CUDAFlashAttentionLaunchPolicy.h"
#include "utils/DebugEnv.h"

// WMMA namespace for Tensor Core operations
using namespace nvcuda;

namespace
{
    // =========================================================================
    // Constants
    // =========================================================================

    constexpr int WARP_SIZE = 32;

    // WMMA fragment dimensions (M x N x K)
    // We use 16x16x16 for FP16->FP32 accumulation which is well-supported
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    // =========================================================================
    // Flash Attention 2 - Pipelined Prefill (Ampere, SM >= 8.0)
    // =========================================================================
    //
    // This implements Flash Attention 2 (Dao et al., 2023) with software
    // pipelining. Dedicated producer warps populate the next shared K/V stage
    // while consumer warps execute QK, softmax, and P@V on the current stage.
    //
    // Key optimizations:
    //   1. Producer/consumer overlap for global-to-shared K/V transfers
    //   2. Double-buffered K/V tiles to overlap load and compute
    //   3. Producer/consumer warp specialization
    //   4. WMMA (16x16x16) for Tensor Core Q @ K^T computation
    //   5. Adaptive tile sizing for different head_dim values
    //
    // Pipeline structure (2 stages, software-managed):
    //   Stage 0: Load K[i], V[i] while computing on K[i-1], V[i-1]
    //   Stage 1: Load K[i+1], V[i+1] while computing on K[i], V[i]
    //
    // Warp roles (configurable via template):
    //   Warps 0-1: Producers - load the next K/V tile into shared memory
    //   Warps 2+:  Consumers - WMMA compute + online softmax
    //
    // =========================================================================

    // Pipeline configuration
    constexpr int FA2_NUM_STAGES = 2;       // Double buffering
    constexpr int FA2_PRODUCER_WARPS = 2;   // Warps dedicated to loading (fixed)
    /// Maximum compact verifier rows represented by DEVICE_PARAMS.
    constexpr int MAX_DYNAMIC_ATTENTION_PARAM_ROWS =
        llaminar2::attention::kMaxGroupedVerifierAttentionRows;
    // Shared-memory padding constants.
    //
    // WMMA constraint: ldm must be a multiple of 8 (for half) or 8 (for float).
    // This eliminates pad values like 2, 4, 12, etc.
    //
    // QKV padding (half precision, ldm = head_dim + pad, must be multiple of 8):
    //   pad=0:  stride=128, 64 words, 64 mod 32=0  → 32-way conflict
    //   pad=8:  stride=136, 68 words, 68 mod 32=4  → 2-way conflict (best achievable)
    //   pad=16: stride=144, 72 words, 72 mod 32=8  → 4-way conflict (worse)
    //
    // Scores padding (float precision, ldm = tile_kv + pad, must be multiple of 8):
    //   tile_kv=16, pad=0:  ld=16, 4 words/row, 4 mod 32=4  → 2-way conflict
    //   tile_kv=16, pad=8:  ld=24, 6 words/row, 6 mod 32=6  → 0 conflicts
    //   tile_kv=16, pad=16: ld=32, 8 words/row, 8 mod 32=8  → 4-way conflict
    //   tile_kv=32, pad=8:  ld=40, 10 words/row, 10 mod 32=10 → 0 conflicts
    //   tile_kv=64, pad=8:  ld=72, 18 words/row, 18 mod 32=18 → 0 conflicts
    constexpr int FA2_SCORES_LD_PAD = 8;
    constexpr int FA2_QKV_PAD = 8;

    // =========================================================================
    // Cached Device Properties for Fast Kernel Launch
    // =========================================================================

    struct FA2DeviceConfig
    {
        int sm_major = 0;
        int sm_minor = 0;
        int sm_count = 0;
        int max_smem_optin = 0; // Max dynamic shared memory with opt-in
        bool initialized = false;
    };

    // Per-device cached config (thread-safe via atomics on first init)
    static FA2DeviceConfig g_fa2_device_config[8]; // Support up to 8 GPUs

    /**
     * @brief Get cached device configuration (lazy init, thread-safe)
     */
    inline FA2DeviceConfig &getFA2DeviceConfig(int device)
    {
        if (device < 0 || device >= 8)
            device = 0;
        FA2DeviceConfig &cfg = g_fa2_device_config[device];

        if (!cfg.initialized)
        {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, device);
            cfg.sm_major = prop.major;
            cfg.sm_minor = prop.minor;
            cfg.sm_count = prop.multiProcessorCount;
            cudaDeviceGetAttribute(&cfg.max_smem_optin,
                                   cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
            cfg.initialized = true;
        }
        return cfg;
    }

#if CUDART_VERSION >= 12030
    /**
     * @brief Publish only the non-default side of an adaptive prefill decision.
     *
     * `CUDAActiveCaptureConditional` creates every handle with a zero launch
     * default and `cudaGraphCondAssignDefault`. CUDA therefore restores the
     * short-prefix decision before each graph replay. Calling
     * `cudaGraphSetConditional(..., 0)` again from the producer needlessly
     * enters the device graph runtime on the most latency-sensitive path. The
     * producer only has to override that replay-local default when the live K/V
     * span selects the context-parallel body.
     *
     * This is not a sticky-state optimization: the graph runtime resets the
     * handle for every launch, so a long-prefix replay followed by a short-prefix
     * replay cannot inherit the earlier non-zero value.
     *
     * @param condition Opaque native conditional handle, or zero when the
     *        enclosing graph has no adaptive branch.
     * @param kv_len Device-owned live K/V length for this replay.
     * @param direct_kv_limit Inclusive direct-query K/V limit in tokens.
     */
    __device__ __forceinline__ void publish_context_condition_if_non_default(
        unsigned long long condition,
        int kv_len,
        int direct_kv_limit)
    {
        if (condition != 0 && kv_len > direct_kv_limit)
        {
            cudaGraphSetConditional(
                static_cast<cudaGraphConditionalHandle>(condition),
                /*value=*/1u);
        }
    }
#endif

    /**
     * @brief Derive graph-captured attention params from live KV cache count.
     *
     * The stage records this kernel immediately before attention. That keeps
     * `AttentionDeviceParams` tied to the cache's device-owned sequence state
     * after KV append, which is the ordering needed for vLLM-style resident MTP
     * publication without a host scalar handoff.
     */
    __global__ void cuda_derive_attention_params_from_cached_tokens_kernel(
        llaminar2::attention::AttentionDeviceParams *__restrict__ out,
        const int *__restrict__ post_append_cached_tokens,
        int seq_len,
        int query_rows,
        int kv_stride,
        const int *__restrict__ active_query_rows_device,
        const int *__restrict__ ring_head_device,
        int ring_capacity,
        unsigned long long prefill_branch_condition,
        int direct_kv_limit)
    {
        const int row = static_cast<int>(threadIdx.x);
        if (!out || !post_append_cached_tokens || row >= query_rows)
            return;

        const int kv_len = max(1, *post_append_cached_tokens);
        const int launch_seq_len = max(1, seq_len);
        const int logical_seq_len = active_query_rows_device
                                        ? max(1, min(launch_seq_len,
                                                     *active_query_rows_device))
                                        : launch_seq_len;
        int ring_row_origin = 0;
        if (ring_head_device && ring_capacity > 0)
        {
            ring_row_origin = (*ring_head_device - kv_len) % ring_capacity;
            if (ring_row_origin < 0)
                ring_row_origin += ring_capacity;
        }

        /*
         * A single parameter record describes an entire ordinary prefill or an
         * M=1 decode.  FA2 applies the causal boundary per query row, so this
         * shared record must expose the complete post-append KV span and only
         * carry the absolute starting position.  The row-shortening rule below
         * belongs exclusively to grouped verifier records, where each record
         * represents one independently replayable serial-decode row.
         */
        if (query_rows == 1)
        {
            out[0].kv_len = kv_len;
            out[0].kv_stride = max(kv_len, kv_stride);
            out[0].position_offset = max(0, kv_len - logical_seq_len);
            out[0].mask_stride = kv_len;
            out[0].ring_row_origin = ring_row_origin;
            out[0].ring_row_capacity = ring_head_device ? ring_capacity : 0;
#if CUDART_VERSION >= 12030
            if (row == 0)
                publish_context_condition_if_non_default(
                    prefill_branch_condition,
                    kv_len,
                    direct_kv_limit);
#endif
            return;
        }

        const int grouped_logical_seq_len = min(query_rows, logical_seq_len);
        if (row >= grouped_logical_seq_len)
        {
            out[row].kv_len = 1;
            out[row].kv_stride = max(kv_len, kv_stride);
            out[row].position_offset = 0;
            out[row].mask_stride = 1;
            out[row].ring_row_origin = ring_row_origin;
            out[row].ring_row_capacity = ring_head_device ? ring_capacity : 0;
            return;
        }
        const int base_position = max(0, kv_len - grouped_logical_seq_len);
        const int row_kv_len =
            max(1, kv_len - (grouped_logical_seq_len - 1 - row));
        out[row].kv_len = row_kv_len;
        out[row].kv_stride = max(kv_len, kv_stride);
        out[row].position_offset = base_position + row;
        out[row].mask_stride = kv_len;
        out[row].ring_row_origin = ring_row_origin;
        out[row].ring_row_capacity = ring_head_device ? ring_capacity : 0;
#if CUDART_VERSION >= 12030
        if (row == 0)
            publish_context_condition_if_non_default(
                prefill_branch_condition,
                kv_len,
                direct_kv_limit);
#endif
    }

    /**
     * @brief Materialize explicit attention geometry directly in device memory.
     *
     * Eager kernel tests and non-cache callers already know their logical KV
     * geometry as launch arguments.  Writing those arguments with a device
     * kernel keeps the parameter block under stream ownership: there is no
     * host mirror, no pinned allocation, and no host-to-device memcpy node.
     * The launch is valid both before capture and as part of a captured graph.
     *
     * For a compact verifier group, row zero observes the shortest serial
     * prefix and the final row observes @p kv_len.  The absolute query position
     * advances with the row while every row retains the full mask stride.
     */
    __global__ void cuda_write_attention_params_from_geometry_kernel(
        llaminar2::attention::AttentionDeviceParams *__restrict__ out,
        int kv_len,
        int kv_stride,
        int position_offset,
        int query_rows,
        unsigned long long prefill_branch_condition,
        int direct_kv_limit)
    {
        const int row = static_cast<int>(threadIdx.x);
        if (!out || row >= query_rows)
            return;

        const int row_kv_len = max(1, kv_len - (query_rows - 1 - row));
        out[row].kv_len = row_kv_len;
        out[row].kv_stride = max(kv_len, kv_stride);
        out[row].position_offset = position_offset + row;
        out[row].mask_stride = kv_len;
        out[row].ring_row_origin = 0;
        out[row].ring_row_capacity = 0;
#if CUDART_VERSION >= 12030
        if (row == 0)
            publish_context_condition_if_non_default(
                prefill_branch_condition,
                kv_len,
                direct_kv_limit);
#endif
    }

    /**
     * @brief Derive request-major row metadata from independent device KV counts.
     *
     * Every request owns one canonical post-append count.  A request may carry
     * several compact verifier rows, in which case the earlier rows expose
     * successively shorter prefixes exactly as serial decode would.  The kernel
     * performs no host read and is intentionally small enough to remain inside
     * the captured attention graph.
     */
    __global__ void cuda_derive_attention_params_from_request_counts_kernel(
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
        const int post_append_kv_len = max(1, post_append_cached_tokens[request]);
        const int base_position = max(0, post_append_kv_len - query_rows);
        const int row_kv_len =
            max(1, post_append_kv_len - (query_rows - 1 - query_row));

        out[flat_row].kv_len = row_kv_len;
        out[flat_row].kv_stride = max(post_append_kv_len, kv_stride);
        out[flat_row].position_offset = base_position + query_row;
        out[flat_row].mask_stride = post_append_kv_len;
        out[flat_row].ring_row_origin = 0;
        out[flat_row].ring_row_capacity = 0;
    }

    /**
     * @brief Compute shared memory size for a given FA2 configuration
     */
    inline size_t computeFA2SmemSize(int tile_q, int tile_kv, int head_dim,
                                     int qkv_pad, int scores_pad)
    {
        const int query_warp_groups = tile_q / WMMA_M;
        if (qkv_pad !=
                llaminar2::cuda::fa2_policy::kFA2QKVSharedMemoryPad ||
            scores_pad !=
                llaminar2::cuda::fa2_policy::kFA2ScoreSharedMemoryPad ||
            FA2_NUM_STAGES !=
                llaminar2::cuda::fa2_policy::kFA2KVSharedMemoryStages)
        {
            return 0;
        }
        return llaminar2::cuda::fa2_policy::fa2DynamicSharedMemoryBytes(
            head_dim,
            query_warp_groups,
            tile_kv);
    }

    /**
     * @brief Compute FA2 kernel configuration based on head_dim
     *
     * Apply the shared capture-time physical K/V tile policy while preserving
     * the mandatory bank-conflict padding and exact query geometry. A geometry
     * that does not fit is invalid; callers fail capture rather than silently
     * changing padding, arithmetic, or work ownership.
     *
     * QKV padding is critical: stride(head_dim=128)=256B → 32-way bank conflict.
     * With qkv_pad=8: stride=272B → 4-way conflict (8x better).
     *
     * Override: set LLAMINAR_FA2_TILE_KV=N to force a specific tile_kv value.
     *
     * Configurations:
     *   - head_dim <= 64:  tile_q=96, 6 consumers, 256 threads
     *   - head_dim <= 128: tile_q=64, 4 consumers, 192 threads
     *   - head_dim <= 256: tile_q=16 or 32, 1 or 2 Q groups, with P@V
     *                      striping selected independently
     */
    struct FA2KernelConfig
    {
        int tile_q;
        int tile_kv;
        int num_q_warp_groups;
        int pv_warps_per_q_group;
        int block_size;
        size_t smem_size;
        int qkv_pad;
        int scores_pad;

        /**
         * @brief Return the number of consumer warps in one thread block.
         *
         * A Q-warp group owns one 16-row WMMA score tile.  Additional P@V
         * warps in that group reuse those scores while owning disjoint output
         * dimension stripes.  The split lowers each thread's live accumulator
         * footprint without duplicating QK work or introducing multi-writer
         * output reductions.
         */
        [[nodiscard]] int consumerWarps() const
        {
            return num_q_warp_groups * pv_warps_per_q_group;
        }
    };

    inline FA2KernelConfig computeFA2Config(
        int head_dim,
        int max_smem,
        int selected_q_warp_groups)
    {
        FA2KernelConfig cfg{};

        // Query grouping was selected once from immutable launch geometry.
        // Every grouping retains the same per-row QK, online-softmax, and P@V
        // arithmetic; only the number of independent rows owned by a block
        // changes.
        const int target_q_warp_groups = selected_q_warp_groups;
        const int target_tile_q = target_q_warp_groups * WMMA_M;

        // Check for env-var tile_kv override (for parameter sweeps)
        const int forced_tile_kv = llaminar2::debugEnv().attention.cuda_fa2_tile_kv;
        if (forced_tile_kv > 0 &&
            forced_tile_kv != 16 &&
            forced_tile_kv != 32 &&
            forced_tile_kv != 64)
        {
            return cfg;
        }

        const llaminar2::cuda::fa2_policy::FA2KVTileGeometry tile_geometry{
            .head_dim = head_dim,
            .query_warp_groups = target_q_warp_groups,
            .max_dynamic_smem = static_cast<std::size_t>(max_smem),
        };
        const int selected_tile_kv =
            llaminar2::cuda::fa2_policy::selectFA2KVTile(
                tile_geometry,
                forced_tile_kv);

        if (selected_tile_kv > 0)
        {
            cfg.tile_q = target_tile_q;
            cfg.tile_kv = selected_tile_kv;
            cfg.num_q_warp_groups = target_q_warp_groups;
            cfg.qkv_pad = FA2_QKV_PAD;
            cfg.scores_pad = FA2_SCORES_LD_PAD;
            cfg.smem_size = computeFA2SmemSize(
                target_tile_q,
                selected_tile_kv,
                head_dim,
                FA2_QKV_PAD,
                FA2_SCORES_LD_PAD);
        }

        if (selected_tile_kv == 0)
            return cfg;

        cfg.pv_warps_per_q_group =
            head_dim > 128
                ? llaminar2::debugEnv().attention.cuda_fa2_hd256_pv_warps
                : 1;
        if (!llaminar2::cuda::fa2_policy::isCompiledFA2PVWarpGeometry(
                head_dim,
                cfg.pv_warps_per_q_group))
        {
            return {};
        }
        cfg.block_size = (FA2_PRODUCER_WARPS + cfg.consumerWarps()) * WARP_SIZE;
        return cfg;
    }

    /**
     * @brief Flash Attention 2 kernel with pipelined prefetching (Ampere, SM >= 8.0)
     *
     * Uses producer/consumer warp specialization to overlap K/V loads with
     * computation, double-buffered shared memory, and WMMA Tensor Core QK.
     *
     * Template parameters:
     *   MAX_Q_WARP_GROUPS: Maximum independent 16-row WMMA score groups
     *                      represented by this canonical arithmetic body (6
     *                      for head_dim=64, 4 for head_dim=128, and 2 for
     *                      head_dim=256). Capture may launch fewer consumer
     *                      warps and a smaller runtime tile_q; missing groups
     *                      simply do not exist in that CTA.
     *   PV_WARPS_PER_Q_GROUP: Number of P@V warps sharing each score group.
     *                         Values greater than one divide HEAD_DIM into
     *                         disjoint stripes, reducing accumulator registers
     *                         while retaining one QK producer and one output
     *                         writer per element.
     *   HEAD_DIM: Compile-time head dimension (64, 128, or 256). Critical for enabling
     *             full unrolling of the P@V accumulation loop and keeping O_acc in
     *             registers. Without this, dims_per_lane is runtime → O_acc spills
     *             to local memory → 10-20x performance loss.
     *   TILE_KV: Compile-time KV tile size (16, 32, or 64). Enables full unrolling of
     *            the P@V outer (j) loop over KV positions, allowing the compiler to
     *            interleave V loads with FMAs. Also removes the warp-divergent
     *            masking branch from the P@V hot loop.
     *
     * Runtime `tile_q` may represent any positive group count through
     * MAX_Q_WARP_GROUPS. The launch block contains producer warps plus exactly
     * the active consumer warps, so narrow geometry does not carry idle
     * threads while every row executes one canonical compiled instruction body.
     *   KV_FP16: When true, K/V pointers are const half* (FP16) and producer
     *            warps copy them directly into shared memory.
     *            When false, K/V are const float* — converted to FP16 per element.
     *            FP16 path eliminates the FP16→FP32→FP16 round-trip when KV cache
     *            is already FP16, roughly halving K/V global memory bandwidth.
     */
    template <int MAX_Q_WARP_GROUPS,
              int PV_WARPS_PER_Q_GROUP,
              int HEAD_DIM,
              int TILE_KV,
              bool KV_FP16 = false,
              bool WRITE_CONTEXT_PARTIAL = false>
    __global__ void flash_attention_2_pipelined_kernel(
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
        int qkv_pad,
        int scores_pad,
        int head_start = 0,
        int gqa_n_rep = 0,
        float *__restrict__ O_partial = nullptr,
        float *__restrict__ m_partial = nullptr,
        float *__restrict__ l_partial = nullptr,
        int context_partition_size = 0,
        int max_context_partitions = 0,
        int device_direct_partition_limit = 0,
        int explicit_context_partition = -1)
    {
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

        if constexpr (!WRITE_CONTEXT_PARTIAL)
        {
            /*
             * An adaptive transaction launches the ordinary lean query kernel
             * as its first captured node. The positive partition limit marks
             * that use and gates the whole grid from device-owned live K/V
             * length. Ordinary query execution passes zero and has no mode
             * decision. This keeps context-publication code out of the direct
             * specialization's register allocation and arithmetic path.
             */
            if (device_direct_partition_limit > 0)
            {
                const int live_partitions = min(
                    max_context_partitions,
                    max(
                        1,
                        (kv_len_runtime + context_partition_size - 1) /
                            context_partition_size));
                if (live_partitions > device_direct_partition_limit)
                    return;
            }
        }

        // Block/thread indexing
        const int batch_idx = blockIdx.z;
        const int flattened_head_partition = blockIdx.x;
        const int head_idx = WRITE_CONTEXT_PARTIAL
                                 ? flattened_head_partition % n_heads
                                 : flattened_head_partition;
        const int q_tile_idx = blockIdx.y;
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        // GQA mapping: when gqa_n_rep > 0 KV heads are REPLICATED (use global head position),
        // when gqa_n_rep == 0 KV heads are SHARDED (use local indexing only).
        const int effective_gqa = (gqa_n_rep > 0) ? gqa_n_rep : ((n_heads == n_kv_heads) ? 1 : (n_heads / n_kv_heads));
        const int kv_head_idx = (gqa_n_rep > 0)
                                    ? (head_start + head_idx) / effective_gqa
                                    : head_idx / effective_gqa;

        // Q row range
        const int q_block_start = q_tile_idx * tile_q;
        if (q_block_start >= seq_len)
            return;

        /*
         * Context execution uses a bounded persistent grid. A physical slot
         * walks a strided subset of only the live canonical partitions; graph
         * capacity therefore never turns into a grid of empty CTAs. The
         * ordered-node diagnostic supplies one explicit partition and executes
         * at most that one item. Every decision is uniform within the CTA.
         */
        int first_context_partition = 0;
        int context_partition_stride = 1;
        int live_context_partitions = 1;
        bool publish_single_partition_output = false;
        if constexpr (WRITE_CONTEXT_PARTIAL)
        {
            live_context_partitions = min(
                max_context_partitions,
                max(
                    1,
                    (kv_len_runtime + context_partition_size - 1) /
                        context_partition_size));
            /*
             * A one-partition context phase already owns the complete online-
             * softmax tuple and can normalize it directly. That common case
             * needs no extra query node. Wider direct windows are owned by the
             * separately specialized query node, so this entire grid retires
             * uniformly and output ownership remains mutually exclusive.
             */
            publish_single_partition_output =
                device_direct_partition_limit == 1 &&
                live_context_partitions == 1;
            if (device_direct_partition_limit > 1 &&
                live_context_partitions <= device_direct_partition_limit)
            {
                return;
            }

            if (explicit_context_partition >= 0)
            {
                first_context_partition = explicit_context_partition;
                context_partition_stride = max_context_partitions;
            }
            else
            {
                first_context_partition =
                    flattened_head_partition / n_heads;
                context_partition_stride =
                    static_cast<int>(gridDim.x) / n_heads;
            }
            if (first_context_partition >= live_context_partitions ||
                context_partition_stride <= 0)
            {
                return;
            }
        }

        // Warp role: producer (warps 0-1) or consumer (warps 2+)
        // Use template parameter for consumer warp count
        const bool is_producer = (warp_id < FA2_PRODUCER_WARPS);
        constexpr int MAX_CONSUMER_WARPS =
            MAX_Q_WARP_GROUPS * PV_WARPS_PER_Q_GROUP;
        static_assert(PV_WARPS_PER_Q_GROUP == 1 ||
                          PV_WARPS_PER_Q_GROUP == 2 ||
                          PV_WARPS_PER_Q_GROUP == 4,
                      "FA2 P@V warp striping supports 1, 2, or 4 warps per Q group");
        static_assert(HEAD_DIM % (2 * PV_WARPS_PER_Q_GROUP) == 0,
                      "Each P@V lane must own an integral output-dimension stripe");
        static_assert(
            llaminar2::cuda::fa2_policy::kFA2CanonicalContextPartitionKeys %
                    TILE_KV ==
                0,
            "Every physical K/V tile must divide the canonical context partition");

        const int consumer_warp_id = warp_id - FA2_PRODUCER_WARPS;
        const bool is_active_consumer =
            !is_producer && consumer_warp_id >= 0 &&
            consumer_warp_id < MAX_CONSUMER_WARPS;
        const int q_warp_group = is_active_consumer
                                     ? consumer_warp_id / PV_WARPS_PER_Q_GROUP
                                     : -1;
        const int pv_warp_in_group = is_active_consumer
                                         ? consumer_warp_id % PV_WARPS_PER_Q_GROUP
                                         : -1;
        const bool owns_score_tile =
            is_active_consumer && pv_warp_in_group == 0;

        // Shared memory layout with double buffering:
        // Buffer 0: K_tile[tile_kv, head_dim], V_tile[tile_kv, head_dim]
        // Buffer 1: K_tile[tile_kv, head_dim], V_tile[tile_kv, head_dim]
        // Q_tile: [tile_q, head_dim] (loaded once)
        // scores: [tile_q, tile_kv]
        extern __shared__ char smem[];

        const int smem_stride = head_dim + qkv_pad;
        const int kv_tile_size = tile_kv * smem_stride;
        const int scores_ld = tile_kv + scores_pad; // padded LD to reduce shared-memory bank conflicts

        half *Q_tile_fp16 = reinterpret_cast<half *>(smem);
        half *KV_buffers = Q_tile_fp16 + tile_q * smem_stride;
        float *scores = reinterpret_cast<float *>(KV_buffers + FA2_NUM_STAGES * 2 * kv_tile_size);

        // Helper to get K/V tile for a stage
        auto get_K_tile = [&](int stage) -> half *
        {
            return KV_buffers + stage * 2 * kv_tile_size;
        };
        auto get_V_tile = [&](int stage) -> half *
        {
            return KV_buffers + stage * 2 * kv_tile_size + kv_tile_size;
        };

        // Consumer warps: warp-cooperative per-row accumulators.  Every warp
        // contributes two lanes per Q row.  Multiple warps in one Q group own
        // adjacent, non-overlapping dimension stripes while sharing the exact
        // score tile emitted by the group's first warp.  Consequently each
        // output element retains the historical serial j accumulation order,
        // but HD256 no longer requires 128 live FP32 accumulators per thread.
        const int q_tile_rows = min(tile_q, seq_len - q_block_start);
        const int row_in_warp = lane_id & (WMMA_M - 1); // lane_id % 16
        const int lane_pair_index = lane_id >> 4;       // 0 for lanes 0-15, 1 for 16-31
        const int dim_partition =
            pv_warp_in_group * 2 + lane_pair_index;
        const int my_consumer_q_row = is_active_consumer
                                          ? q_block_start + q_warp_group * WMMA_M + row_in_warp
                                          : -1;
        const bool owns_row = (my_consumer_q_row >= 0 &&
                               my_consumer_q_row < seq_len &&
                               (my_consumer_q_row - q_block_start) < q_tile_rows);

        // O_acc sized at compile-time HEAD_DIM to keep in registers.
        // active_dims_per_lane uses runtime head_dim for correct bounds
        // when head_dim < HEAD_DIM (e.g., head_dim=32 with HEAD_DIM=64).
        constexpr int dims_per_lane =
            HEAD_DIM / (2 * PV_WARPS_PER_Q_GROUP);
        const int active_dims_per_lane =
            head_dim / (2 * PV_WARPS_PER_Q_GROUP);
        const int dim_start = dim_partition * active_dims_per_lane;
        // Pointers for this batch/head
        const float *Q_batch = Q + batch_idx * seq_len * n_heads * head_dim;

        // K/V pointers depend on KV_FP16 template parameter.
        // When KV_FP16=true, K/V are already FP16 in global memory (KV cache);
        // when false, they are FP32 workspace buffers.

        // ALL warps: Load Q tile (done once, always FP32→FP16)
        for (int i = threadIdx.x; i < q_tile_rows * head_dim; i += blockDim.x)
        {
            int local_row = i / head_dim;
            int d = i % head_dim;
            int global_row = q_block_start + local_row;
            float val = Q_batch[global_row * n_heads * head_dim + head_idx * head_dim + d];
            Q_tile_fp16[local_row * smem_stride + d] = __float2half(val);
        }

        for (int context_partition = first_context_partition;
             context_partition < live_context_partitions;
             context_partition += context_partition_stride)
        {
            float O_acc[dims_per_lane];
            float m_i = -FLT_MAX;
            float l_i = 0.0f;

            // Direct execution merges each canonical partition into the output
            // row as unnormalized private scratch. Each thread owns disjoint
            // dimensions, while every P@V stripe reconstructs the same scalar
            // merge state. Context execution resets this state for each exact
            // canonical partition owned by the persistent physical slot.
            float merged_m = -FLT_MAX;
            float merged_l = 0.0f;

            if (owns_row)
            {
#pragma unroll
                for (int d = 0; d < dims_per_lane; d++)
                    O_acc[d] = 0.0f;
            }

        /*
         * Context publication owns one immutable canonical K/V interval. The
         * direct specialization scans the complete live span and merges those
         * same canonical intervals before normalizing the output.
         */
        const int kv_partition_begin = WRITE_CONTEXT_PARTIAL
                                           ? context_partition * context_partition_size
                                           : 0;
        const int kv_partition_end = WRITE_CONTEXT_PARTIAL
                                         ? min(kv_partition_begin + context_partition_size,
                                               kv_len_runtime)
                                         : kv_len_runtime;
        const int partition_kv_len = max(0, kv_partition_end - kv_partition_begin);
        const int num_kv_tiles =
            (partition_kv_len + tile_kv - 1) / tile_kv;

        // =====================================================================
        // Pipeline prologue: Start loading first tile(s)
        // =====================================================================
        if (is_producer && num_kv_tiles > 0)
        {
            const int kv_start_0 = kv_partition_begin;
            const int kv_end_0 = min(kv_start_0 + tile_kv, kv_partition_end);
            const int actual_len_0 = kv_end_0 - kv_start_0;

            half *K_dst_0 = get_K_tile(0);
            half *V_dst_0 = get_V_tile(0);

            const int producer_local_id = warp_id;
            const int elems_per_producer = (actual_len_0 * head_dim + 1) / 2;
            const int my_start = producer_local_id * elems_per_producer;
            const int my_end = min(my_start + elems_per_producer, actual_len_0 * head_dim);

            if constexpr (KV_FP16)
            {
                // FP16 path: K/V already half in global — direct copy
                const half *K_batch_fp16 = static_cast<const half *>(K) + batch_idx * kv_stride * n_kv_heads * head_dim;
                const half *V_batch_fp16 = static_cast<const half *>(V) + batch_idx * kv_stride * n_kv_heads * head_dim;
                for (int i = my_start + lane_id; i < my_end; i += WARP_SIZE)
                {
                    int local_row = i / head_dim;
                    int d = i % head_dim;
                    const int logical_row = kv_start_0 + local_row;
                    const int physical_row = ring_row_capacity > 0
                                                 ? (ring_row_origin + logical_row) %
                                                       ring_row_capacity
                                                 : logical_row;
                    int kv_offset = physical_row * n_kv_heads * head_dim + kv_head_idx * head_dim + d;
                    K_dst_0[local_row * smem_stride + d] = K_batch_fp16[kv_offset];
                    V_dst_0[local_row * smem_stride + d] = V_batch_fp16[kv_offset];
                }
            }
            else
            {
                // FP32 path: convert float → half per element
                const float *K_batch_fp32 = static_cast<const float *>(K) + batch_idx * kv_stride * n_kv_heads * head_dim;
                const float *V_batch_fp32 = static_cast<const float *>(V) + batch_idx * kv_stride * n_kv_heads * head_dim;
                for (int i = my_start + lane_id; i < my_end; i += WARP_SIZE)
                {
                    int local_row = i / head_dim;
                    int d = i % head_dim;
                    const int logical_row = kv_start_0 + local_row;
                    const int physical_row = ring_row_capacity > 0
                                                 ? (ring_row_origin + logical_row) %
                                                       ring_row_capacity
                                                 : logical_row;
                    int kv_offset = physical_row * n_kv_heads * head_dim + kv_head_idx * head_dim + d;
                    K_dst_0[local_row * smem_stride + d] = __float2half(K_batch_fp32[kv_offset]);
                    V_dst_0[local_row * smem_stride + d] = __float2half(V_batch_fp32[kv_offset]);
                }
            }
        }
        __syncthreads();

        // WMMA fragments (consumer warps only, but declared for all)
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> q_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> k_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag;

        // =====================================================================
        // Main pipeline loop
        // =====================================================================
        for (int kv_tile_iter = 0; kv_tile_iter < num_kv_tiles; kv_tile_iter++)
        {
            const int current_stage = kv_tile_iter % FA2_NUM_STAGES;
            const int next_stage = (kv_tile_iter + 1) % FA2_NUM_STAGES;

            const int kv_start = kv_partition_begin + kv_tile_iter * tile_kv;
            const int kv_end_tile = min(kv_start + tile_kv, kv_partition_end);
            const int actual_tile_kv_len = kv_end_tile - kv_start;

            // Early exit for causal
            if (causal && kv_start > (q_block_start + tile_q - 1) + position_offset_runtime)
            {
                break;
            }

            // Get current buffers
            half *K_tile_fp16 = get_K_tile(current_stage);
            half *V_tile_fp16 = get_V_tile(current_stage);

            // ----------------------------------------------------------------
            // PRODUCER WARPS: Start loading next tile into next_stage buffer
            // ----------------------------------------------------------------
            if (is_producer && kv_tile_iter + 1 < num_kv_tiles)
            {
                const int next_kv_start =
                    kv_partition_begin + (kv_tile_iter + 1) * tile_kv;
                const int next_kv_end =
                    min(next_kv_start + tile_kv, kv_partition_end);
                const int next_actual_len = next_kv_end - next_kv_start;

                half *K_dst = get_K_tile(next_stage);
                half *V_dst = get_V_tile(next_stage);

                const int producer_local_id = warp_id;
                const int elems_per_producer = (next_actual_len * head_dim + 1) / 2;
                const int my_start = producer_local_id * elems_per_producer;
                const int my_end = min(my_start + elems_per_producer, next_actual_len * head_dim);

                if constexpr (KV_FP16)
                {
                    // FP16 path: direct half copy from KV cache
                    const half *K_batch_fp16 = static_cast<const half *>(K) + batch_idx * kv_stride * n_kv_heads * head_dim;
                    const half *V_batch_fp16 = static_cast<const half *>(V) + batch_idx * kv_stride * n_kv_heads * head_dim;
                    for (int i = my_start + lane_id; i < my_end; i += WARP_SIZE)
                    {
                        int local_row = i / head_dim;
                        int d = i % head_dim;
                        const int logical_row = next_kv_start + local_row;
                        const int physical_row = ring_row_capacity > 0
                                                     ? (ring_row_origin + logical_row) %
                                                           ring_row_capacity
                                                     : logical_row;
                        int kv_offset = physical_row * n_kv_heads * head_dim + kv_head_idx * head_dim + d;
                        K_dst[local_row * smem_stride + d] = K_batch_fp16[kv_offset];
                        V_dst[local_row * smem_stride + d] = V_batch_fp16[kv_offset];
                    }
                }
                else
                {
                    // FP32 path: float→half conversion per element
                    const float *K_batch_fp32 = static_cast<const float *>(K) + batch_idx * kv_stride * n_kv_heads * head_dim;
                    const float *V_batch_fp32 = static_cast<const float *>(V) + batch_idx * kv_stride * n_kv_heads * head_dim;
                    for (int i = my_start + lane_id; i < my_end; i += WARP_SIZE)
                    {
                        int local_row = i / head_dim;
                        int d = i % head_dim;
                        const int logical_row = next_kv_start + local_row;
                        const int physical_row = ring_row_capacity > 0
                                                     ? (ring_row_origin + logical_row) %
                                                           ring_row_capacity
                                                     : logical_row;
                        int kv_offset = physical_row * n_kv_heads * head_dim + kv_head_idx * head_dim + d;
                        K_dst[local_row * smem_stride + d] = __float2half(K_batch_fp32[kv_offset]);
                        V_dst[local_row * smem_stride + d] = __float2half(V_batch_fp32[kv_offset]);
                    }
                }
            }

            // ----------------------------------------------------------------
            // CONSUMER WARPS: Compute Q @ K^T using WMMA
            // ----------------------------------------------------------------
            if (owns_score_tile)
            {
                const int warp_q_start = q_warp_group * WMMA_M;

                if (warp_q_start < q_tile_rows)
                {
                    for (int kv_col = 0; kv_col < actual_tile_kv_len; kv_col += WMMA_N)
                    {
                        wmma::fill_fragment(s_frag, 0.0f);

                        for (int k = 0; k < head_dim; k += WMMA_K)
                        {
                            const half *Q_ptr = Q_tile_fp16 + warp_q_start * smem_stride + k;
                            wmma::load_matrix_sync(q_frag, Q_ptr, smem_stride);

                            const half *K_ptr = K_tile_fp16 + kv_col * smem_stride + k;
                            wmma::load_matrix_sync(k_frag, K_ptr, smem_stride);

                            wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
                        }

                        float *scores_ptr = scores + warp_q_start * scores_ld + kv_col;
                        wmma::store_matrix_sync(scores_ptr, s_frag, scores_ld, wmma::mem_row_major);
                    }
                }
            }

            // Sync all warps before softmax
            __syncthreads();

            // ----------------------------------------------------------------
            // CONSUMER WARPS: Apply softmax and accumulate P @ V.
            //
            // Exactly one warp in each Q group owns masking/scaling writes.
            // Every P@V stripe must consume those exact published FP32 bytes;
            // independently reconstructing the value changes the canonical
            // partition merge by one or more ULPs under CUDA fast math. A full
            // CTA barrier is retained because the named consumer-only barrier
            // is slower on GA102 and the producer warps must join the later
            // shared-stage reuse edge in either case.
            // ----------------------------------------------------------------
            if (owns_score_tile)
            {
                if (owns_row && lane_pair_index == 0)
                {
                    const int local_q_row =
                        q_warp_group * WMMA_M + row_in_warp;
                    float *my_scores = scores + local_q_row * scores_ld;

                    for (int j = 0; j < actual_tile_kv_len; j++)
                    {
                        const int kv_pos = kv_start + j;
                        bool masked = false;

                        if (causal &&
                            kv_pos > my_consumer_q_row +
                                         position_offset_runtime)
                        {
                            masked = true;
                        }
                        if (window_size > 0)
                        {
                            const int q_pos =
                                my_consumer_q_row + position_offset_runtime;
                            if (kv_pos < q_pos - window_size ||
                                kv_pos > q_pos + window_size)
                            {
                                masked = true;
                            }
                        }

                        if (mask)
                        {
                            const float mask_val =
                                mask[(batch_idx * seq_len +
                                      my_consumer_q_row) *
                                         mask_stride +
                                     kv_pos];
                            if (mask_val <= -1.0e20f)
                                masked = true;
                            else
                                my_scores[j] += mask_val;
                        }

                        if (masked)
                            my_scores[j] = -FLT_MAX;
                        else
                            my_scores[j] *= softmax_scale;
                    }
                }
            }

            __syncthreads();

            if (is_active_consumer)
            {
                /*
                 * Physical K/V tiles are a throughput parameter, not an
                 * arithmetic parameter. Consume every 16-key microtile in
                 * ascending order so TILE_KV=16/32/64, query partitioning, TP,
                 * and M all execute the same online-softmax reduction tree.
                 * This preserves byte identity while still allowing wider
                 * vectorized loads and fewer CTA barriers where they win.
                 */
                constexpr int kReductionTileKV = WMMA_N;
#pragma unroll
                for (int reduction_start = 0;
                     reduction_start < TILE_KV;
                     reduction_start += kReductionTileKV)
                {
                    const int reduction_len =
                        min(kReductionTileKV,
                            max(0, actual_tile_kv_len - reduction_start));
                    float m_ij = -FLT_MAX;

                    // Every stripe warp reconstructs the same row maximum from
                    // the published score bytes in canonical key order.
                    if (owns_row && lane_pair_index == 0)
                    {
                        const int local_q_row =
                            q_warp_group * WMMA_M + row_in_warp;
                        const float *my_scores =
                            scores + local_q_row * scores_ld + reduction_start;
#pragma unroll
                        for (int j = 0; j < kReductionTileKV; ++j)
                        {
                            if (j < reduction_len)
                                m_ij = fmaxf(m_ij, my_scores[j]);
                        }
                    }

                    // All lanes participate, including inactive tail-row lanes.
                    m_ij = __shfl_sync(0xFFFFFFFF, m_ij, row_in_warp);

                    /*
                     * A causal/window partition can be wholly invisible to an
                     * earlier query row even when a later row in the same WMMA
                     * tile consumes it.  In that case every score is -FLT_MAX.
                     * Treat the microtile as the neutral online-softmax element;
                     * evaluating exp(-FLT_MAX - -FLT_MAX) would otherwise add
                     * sixteen spurious unit weights.
                     */
                    if (owns_row && reduction_len > 0 && m_ij > -FLT_MAX)
                    {
                        const int local_q_row =
                            q_warp_group * WMMA_M + row_in_warp;
                        const float *my_scores =
                            scores + local_q_row * scores_ld + reduction_start;

                        const float m_i_new = fmaxf(m_i, m_ij);
                        const float scale_old = __expf(m_i - m_i_new);

#pragma unroll
                        for (int d = 0; d < dims_per_lane; ++d)
                            O_acc[d] *= scale_old;
                        l_i *= scale_old;

                        float l_ij = 0.0f;
#pragma unroll
                        for (int j = 0; j < kReductionTileKV; ++j)
                        {
                            if (j < reduction_len)
                            {
                                const float p =
                                    __expf(my_scores[j] - m_i_new);
                                l_ij += p;

                                const half2 *V_row_h2 =
                                    reinterpret_cast<const half2 *>(
                                        V_tile_fp16 +
                                        (reduction_start + j) * smem_stride +
                                        dim_start);
#pragma unroll
                                for (int d = 0; d < dims_per_lane; d += 2)
                                {
                                    const float2 v =
                                        __half22float2(V_row_h2[d >> 1]);
                                    O_acc[d] += p * v.x;
                                    O_acc[d + 1] += p * v.y;
                                }
                            }
                        }

                        l_i += l_ij;
                        m_i = m_i_new;
                    }
                }
            }

            __syncthreads();

            if constexpr (!WRITE_CONTEXT_PARTIAL)
            {
                constexpr int kCanonicalPartitionKeys =
                    llaminar2::cuda::fa2_policy::
                        kFA2CanonicalContextPartitionKeys;
                const int maximum_visible_key =
                    q_block_start + tile_q - 1 + position_offset_runtime;
                const bool next_tile_is_causally_empty =
                    causal && kv_end_tile > maximum_visible_key;
                const bool closes_canonical_partition =
                    kv_end_tile % kCanonicalPartitionKeys == 0;
                const bool closes_scan =
                    kv_tile_iter + 1 == num_kv_tiles ||
                    next_tile_is_causally_empty;

                if ((closes_canonical_partition || closes_scan) && owns_row)
                {
                    float *O_batch =
                        O + batch_idx * seq_len * n_heads * head_dim;
                    float *O_row =
                        O_batch + my_consumer_q_row * n_heads * head_dim +
                        head_idx * head_dim;

                    if (l_i > 0.0f)
                    {
                        if (!(merged_l > 0.0f))
                        {
                            merged_m = m_i;
                            merged_l = l_i;
#pragma unroll
                            for (int d = 0; d < dims_per_lane; ++d)
                            {
                                if (d < active_dims_per_lane)
                                    O_row[dim_start + d] = O_acc[d];
                            }
                        }
                        else
                        {
                            const float next_m = fmaxf(merged_m, m_i);
                            const float previous_scale =
                                __expf(merged_m - next_m);
                            const float partition_scale =
                                __expf(m_i - next_m);
#pragma unroll
                            for (int d = 0; d < dims_per_lane; ++d)
                            {
                                if (d < active_dims_per_lane)
                                {
                                    O_row[dim_start + d] =
                                        O_row[dim_start + d] * previous_scale +
                                        O_acc[d] * partition_scale;
                                }
                            }
                            merged_l = merged_l * previous_scale +
                                       l_i * partition_scale;
                            merged_m = next_m;
                        }
                    }

                    if (!closes_scan)
                    {
#pragma unroll
                        for (int d = 0; d < dims_per_lane; ++d)
                            O_acc[d] = 0.0f;
                        m_i = -FLT_MAX;
                        l_i = 0.0f;
                    }
                }

                // A wholly future tile is the neutral suffix of the current
                // canonical partition. Once its preceding visible summary has
                // been merged, later K/V tiles cannot affect any row in this
                // query block and need not be loaded or scored.
                if (next_tile_is_causally_empty)
                    break;
            }
        }

        // =====================================================================
        // Publish either one context summary or the normalized final output.
        // =====================================================================
        if (is_active_consumer && owns_row)
        {
            if constexpr (WRITE_CONTEXT_PARTIAL)
            {
                if (publish_single_partition_output)
                {
                    const float inverse_l =
                        l_i > 0.0f ? 1.0f / l_i : 0.0f;
                    float *O_batch =
                        O + batch_idx * seq_len * n_heads * head_dim;
                    float *O_row =
                        O_batch + my_consumer_q_row * n_heads * head_dim +
                        head_idx * head_dim;
#pragma unroll
                    for (int d = 0; d < dims_per_lane; ++d)
                    {
                        if (d < active_dims_per_lane)
                            O_row[dim_start + d] = O_acc[d] * inverse_l;
                    }
                }
                else
                {
                    const size_t summary_index =
                        ((static_cast<size_t>(batch_idx) * seq_len +
                          my_consumer_q_row) *
                             n_heads +
                         head_idx) *
                            max_context_partitions +
                        context_partition;
                    float *O_summary =
                        O_partial +
                        summary_index * static_cast<size_t>(head_dim);

#pragma unroll
                    for (int d = 0; d < dims_per_lane; ++d)
                    {
                        if (d < active_dims_per_lane)
                            O_summary[dim_start + d] = O_acc[d];
                    }

                    // All P@V stripes reconstruct identical scalar state. One
                    // named lane publishes it, preserving single-writer ownership.
                    if (pv_warp_in_group == 0 && lane_pair_index == 0)
                    {
                        m_partial[summary_index] = m_i;
                        l_partial[summary_index] = l_i;
                    }
                }
            }
            else
            {
                const float inv_l =
                    (merged_l > 0.0f) ? (1.0f / merged_l) : 0.0f;
                float *O_batch =
                    O + batch_idx * seq_len * n_heads * head_dim;
                float *O_row =
                    O_batch + my_consumer_q_row * n_heads * head_dim +
                    head_idx * head_dim;

#pragma unroll
                for (int d = 0; d < dims_per_lane; d++)
                {
                    if (d < active_dims_per_lane)
                    {
                        O_row[dim_start + d] =
                            merged_l > 0.0f
                                ? O_row[dim_start + d] * inv_l
                                : 0.0f;
                    }
                }
            }
        }

            if constexpr (WRITE_CONTEXT_PARTIAL)
            {
                // Every thread must finish consuming the shared K/V and score
                // tiles before this persistent CTA advances to its next
                // canonical partition and reuses those buffers.
                __syncthreads();
            }
        }
    }

    /**
     * @brief Merge fixed FA2 context summaries in ascending partition order.
     *
     * Phase one publishes an unnormalized online-softmax tuple `(m, l, O)` for
     * every query-row, head, and fixed contiguous K/V partition.  This kernel
     * assigns one or more warps to each query-row/head pair. Every lane
     * traverses the same partition prefix in the same order while owning
     * disjoint output dimensions, so dimension striping changes only physical
     * residency: there are no atomics, inter-block reductions, or order-
     * dependent writers.
     *
     * The launch grid is immutable under graph replay. The reducer derives its
     * active partition prefix directly from device-owned K/V length, so stale
     * summary bytes beyond a shorter replay can never participate. This avoids
     * both a host scalar and hundreds of neutral-tail loads while preserving the
     * exact ascending canonical-partition order.
     *
     * @param O_partial Unnormalized summaries laid out as
     *        `[batch, query, head, partition, head_dim]`.
     * @param m_partial Summary maxima laid out as
     *        `[batch, query, head, partition]`.
     * @param l_partial Summary exponential sums with the same scalar layout.
     * @param O Normalized output `[batch, query, head, head_dim]`.
     * @param seq_len Captured query-row count.
     * @param n_heads Participant-local query-head count.
     * @param head_dim Elements in one output head, at most 256.
     * @param max_context_partitions Fixed captured partition envelope.
     * @param device_direct_partition_limit Largest live partition count already
     *        published directly by phase one; the reducer is then a no-op.
     * @param device_params Device-owned live K/V length and cache stride.
     * @tparam DIMENSION_WARPS_PER_ROW Independent 32-dimension stripes assigned
     *         to each query-row/head reduction.
     */
    template <int DIMENSION_WARPS_PER_ROW>
    __global__ __launch_bounds__(256, 2)
    void flash_attention_2_context_reduce_kernel(
        const float *__restrict__ O_partial,
        const float *__restrict__ m_partial,
        const float *__restrict__ l_partial,
        float *__restrict__ O,
        int seq_len,
        int n_heads,
        int head_dim,
        int max_context_partitions,
        int device_direct_partition_limit,
        const llaminar2::attention::AttentionDeviceParams *__restrict__
            device_params)
    {
        constexpr int kWarpsPerBlock = 8;
        static_assert(
            DIMENSION_WARPS_PER_ROW == 1 ||
                DIMENSION_WARPS_PER_ROW == 2 ||
                DIMENSION_WARPS_PER_ROW == 4 ||
                DIMENSION_WARPS_PER_ROW == 8,
            "FA2 reducer dimension striping supports 1, 2, 4, or 8 warps");
        static_assert(
            kWarpsPerBlock % DIMENSION_WARPS_PER_ROW == 0,
            "FA2 reducer rows must divide the 256-thread block exactly");
        constexpr int kRowsPerBlock =
            kWarpsPerBlock / DIMENSION_WARPS_PER_ROW;
        constexpr int kMaximumDimensionsPerLane =
            8 / DIMENSION_WARPS_PER_ROW;

        const int lane = threadIdx.x & (WARP_SIZE - 1);
        const int warp = threadIdx.x / WARP_SIZE;
        const int row_in_block = warp / DIMENSION_WARPS_PER_ROW;
        const int dimension_warp = warp % DIMENSION_WARPS_PER_ROW;
        const int query_row = blockIdx.y * kRowsPerBlock + row_in_block;
        if (query_row >= seq_len)
            return;

        const int head = blockIdx.x;
        const int batch = blockIdx.z;
        const size_t scalar_base =
            ((static_cast<size_t>(batch) * seq_len + query_row) * n_heads +
             head) *
            max_context_partitions;

        float output_accumulator[kMaximumDimensionsPerLane] = {0.0f};
        float merged_m = -FLT_MAX;
        float merged_l = 0.0f;

        const int live_context_partitions = device_params
                                                ? min(
                                                      max_context_partitions,
                                                      max(
                                                          1,
                                                          (device_params->kv_len +
                                                           llaminar2::cuda::fa2_policy::
                                                               kFA2CanonicalContextPartitionKeys -
                                                           1) /
                                                              llaminar2::cuda::fa2_policy::
                                                                  kFA2CanonicalContextPartitionKeys))
                                                : max_context_partitions;
        if (live_context_partitions <= device_direct_partition_limit)
            return;

        for (int partition = 0;
             partition < live_context_partitions;
             ++partition)
        {
            const size_t scalar_index = scalar_base + partition;
            const float partition_l = l_partial[scalar_index];
            if (!(partition_l > 0.0f))
                continue;

            const float partition_m = m_partial[scalar_index];
            const float *partition_output =
                O_partial + scalar_index * static_cast<size_t>(head_dim);

            if (!(merged_l > 0.0f))
            {
                merged_m = partition_m;
                merged_l = partition_l;
#pragma unroll
                for (int element = 0;
                     element < kMaximumDimensionsPerLane;
                     ++element)
                {
                    const int dimension =
                        dimension_warp * WARP_SIZE + lane +
                        element * WARP_SIZE * DIMENSION_WARPS_PER_ROW;
                    if (dimension < head_dim)
                        output_accumulator[element] =
                            partition_output[dimension];
                }
                continue;
            }

            const float next_m = fmaxf(merged_m, partition_m);
            const float previous_scale = __expf(merged_m - next_m);
            const float partition_scale = __expf(partition_m - next_m);
#pragma unroll
            for (int element = 0;
                 element < kMaximumDimensionsPerLane;
                 ++element)
            {
                const int dimension =
                    dimension_warp * WARP_SIZE + lane +
                    element * WARP_SIZE * DIMENSION_WARPS_PER_ROW;
                if (dimension < head_dim)
                {
                    output_accumulator[element] =
                        output_accumulator[element] * previous_scale +
                        partition_output[dimension] * partition_scale;
                }
            }
            merged_l = merged_l * previous_scale +
                       partition_l * partition_scale;
            merged_m = next_m;
        }

        const float inverse_l =
            merged_l > 0.0f ? 1.0f / merged_l : 0.0f;
        float *output =
            O + (static_cast<size_t>(batch) * seq_len * n_heads +
                 static_cast<size_t>(query_row) * n_heads + head) *
                    head_dim;
#pragma unroll
        for (int element = 0;
             element < kMaximumDimensionsPerLane;
             ++element)
        {
            const int dimension =
                dimension_warp * WARP_SIZE + lane +
                element * WARP_SIZE * DIMENSION_WARPS_PER_ROW;
            if (dimension < head_dim)
                output[dimension] =
                    output_accumulator[element] * inverse_l;
        }
    }

    // Explicit instantiations retain one widest query-group body per head
    // dimension. Runtime active-group selection changes the CTA size and row
    // ownership while the same compiled body preserves one arithmetic implementation.
#define FA2_INSTANTIATE(QW, PVW, HD, TKV, KV16)                                      \
    template __global__ void flash_attention_2_pipelined_kernel<                     \
        QW, PVW, HD, TKV, KV16>(                                                     \
        const float *, const void *, const void *, float *,                           \
        int, int, int, int, int, int, float, bool, int, int,                          \
        const llaminar2::attention::AttentionDeviceParams *, const float *,           \
        int, int, int, int, int, int, float *, float *, float *, int, int, int, int)

#define FA2_INSTANTIATE_TILES(QW, PVW, HD, KV16) \
    FA2_INSTANTIATE(QW, PVW, HD, 16, KV16);      \
    FA2_INSTANTIATE(QW, PVW, HD, 32, KV16);      \
    FA2_INSTANTIATE(QW, PVW, HD, 64, KV16)

    FA2_INSTANTIATE_TILES(6, 1, 64, false);
    FA2_INSTANTIATE_TILES(4, 1, 128, false);
    FA2_INSTANTIATE_TILES(2, 1, 256, false);
    FA2_INSTANTIATE_TILES(2, 2, 256, false);
    FA2_INSTANTIATE_TILES(2, 4, 256, false);

    FA2_INSTANTIATE_TILES(6, 1, 64, true);
    FA2_INSTANTIATE_TILES(4, 1, 128, true);
    FA2_INSTANTIATE_TILES(2, 1, 256, true);
    FA2_INSTANTIATE_TILES(2, 2, 256, true);
    FA2_INSTANTIATE_TILES(2, 4, 256, true);

#undef FA2_INSTANTIATE_TILES
#undef FA2_INSTANTIATE
    // =========================================================================
    // Flash Decoding - Split-K Kernel (FP32)
    // =========================================================================
    //
    // Warp-cooperative design: all 32 lanes in a warp cooperate on each KV
    // position (vs. the old design where each thread owned a full KV position).
    //
    // Key improvements over the previous per-thread design:
    //   - O_lane[4] per thread instead of O_local[128] → ~4 regs vs ~128 regs
    //   - Cooperative dot product: 32 lanes × 4 elements = 128-dim dot in parallel
    //   - No warp-level O shuffle needed (m/l are warp-uniform after reduce)
    //   - Vectorized K/V loads (coalesced across lanes)
    //   - __expf() fast math intrinsic
    // =========================================================================

    /**
     * @brief Warp-level sum reduction (5 shuffle steps for 32 lanes)
     */
    __device__ __forceinline__ float warpReduceSum(float val)
    {
        val += __shfl_xor_sync(0xffffffff, val, 16);
        val += __shfl_xor_sync(0xffffffff, val, 8);
        val += __shfl_xor_sync(0xffffffff, val, 4);
        val += __shfl_xor_sync(0xffffffff, val, 2);
        val += __shfl_xor_sync(0xffffffff, val, 1);
        return val;
    }

    /**
     * @brief Select the live split prefix inside a fixed captured launch envelope.
     *
     * Graph capture records `max_num_splits` physical split planes.  The live KV
     * length is device-owned and may cross several sequence-parallel regimes
     * while that graph is replayed.  Selecting only an active prefix preserves
     * the historical serial-decode partition without changing launch topology.
     */
    __device__ __forceinline__ int activeDecodeSplits(
        int kv_len,
        int max_num_splits,
        int min_kv_per_split)
    {
        return min(
            max_num_splits,
            max(1, kv_len / max(1, min_kv_per_split)));
    }

    /**
     * @brief Flash Decoding kernel for single-query decode (warp-cooperative)
     *
     * Parallelizes over KV cache using split-K pattern.
     * Grid: (n_heads, num_splits, batch_size)
     * Block: (256,) threads = 8 warps
     *
     * Each warp processes KV positions cooperatively: all 32 lanes share the
     * dot product and V accumulation for each position. For head_dim=128,
     * each lane owns 4 output dimensions (128/32 = 4).
     */
    __global__ __launch_bounds__(256, 4) void flash_decoding_fp32_kernel(
        const float *__restrict__ Q,
        const float *__restrict__ K_cache,
        const float *__restrict__ V_cache,
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
        int head_start = 0,
        int gqa_n_rep = 0)
    {
        int kv_stride = kv_len;
        int kv_len_runtime = kv_len;
        int ring_row_origin = 0;
        int ring_row_capacity = 0;
        if (device_params)
        {
            kv_len_runtime = device_params->kv_len;
            kv_stride = device_params->kv_stride;
            ring_row_origin = device_params->ring_row_origin;
            ring_row_capacity = device_params->ring_row_capacity;
        }

        const int head_idx = blockIdx.x;
        const int split_idx = blockIdx.y;
        const int batch_idx = blockIdx.z;

        const int effective_gqa = (gqa_n_rep > 0) ? gqa_n_rep : ((n_heads == n_kv_heads) ? 1 : (n_heads / n_kv_heads));
        const int kv_head_idx = (gqa_n_rep > 0)
                                    ? (head_start + head_idx) / effective_gqa
                                    : head_idx / effective_gqa;

        const int row_num_splits =
            activeDecodeSplits(kv_len_runtime, num_splits, 16);
        if (split_idx >= row_num_splits)
            return;

        const int split_size =
            (kv_len_runtime + row_num_splits - 1) / row_num_splits;
        const int kv_start = split_idx * split_size;
        const int kv_end = min(kv_start + split_size, kv_len_runtime);

        const int partial_idx = (batch_idx * n_heads + head_idx) * num_splits + split_idx;

        if (kv_start >= kv_len_runtime)
        {
            if (threadIdx.x == 0)
            {
                m_partial[partial_idx] = -FLT_MAX;
                l_partial[partial_idx] = 0.0f;
            }
            float *O_out = O_partial + partial_idx * head_dim;
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            {
                O_out[d] = 0.0f;
            }
            return;
        }

        const int tid = threadIdx.x;
        const int num_threads = blockDim.x;
        const int warp_id = tid / WARP_SIZE;
        const int lane_id = tid % WARP_SIZE;
        const int num_warps = num_threads / WARP_SIZE;

        // Load Q into shared memory (all threads cooperate)
        extern __shared__ char smem[];
        float *Q_shared = reinterpret_cast<float *>(smem);

        const float *Q_ptr = Q + (batch_idx * n_heads + head_idx) * head_dim;
        for (int d = tid; d < head_dim; d += num_threads)
        {
            Q_shared[d] = Q_ptr[d];
        }
        __syncthreads();

        // Per-lane O accumulators — only this lane's output dimensions
        // For head_dim=128, WARP_SIZE=32: each lane owns 4 dims
        // Lane i owns dims: i, i+32, i+64, i+96 (strided by WARP_SIZE)
        constexpr int MAX_DIMS_PER_LANE = 8; // supports head_dim up to 256
        float O_lane[MAX_DIMS_PER_LANE] = {0};
        float m_local = -FLT_MAX;
        float l_local = 0.0f;

        const float *K_batch = K_cache + batch_idx * kv_stride * n_kv_heads * head_dim;
        const float *V_batch = V_cache + batch_idx * kv_stride * n_kv_heads * head_dim;

        // =================================================================
        // Main loop: warp-cooperative KV processing
        //
        // Each warp processes a strided subset of KV positions.
        // Within each position, all 32 lanes cooperate on the dot product
        // and V accumulation. K/V loads are coalesced across lanes.
        // =================================================================
        for (int kv_pos = kv_start + warp_id; kv_pos < kv_end; kv_pos += num_warps)
        {
            const int physical_kv_pos = ring_row_capacity > 0
                                            ? (ring_row_origin + kv_pos) %
                                                  ring_row_capacity
                                            : kv_pos;
            const float *K_ptr =
                K_batch + physical_kv_pos * n_kv_heads * head_dim +
                kv_head_idx * head_dim;

            // Cooperative dot product: each lane handles head_dim/32 elements
            float partial_dot = 0.0f;
            for (int d = lane_id; d < head_dim; d += WARP_SIZE)
            {
                partial_dot += Q_shared[d] * K_ptr[d];
            }

            // Warp reduce → all lanes get full dot product (5 shuffles)
            float score = warpReduceSum(partial_dot) * softmax_scale;

            // Online softmax — score is uniform across all lanes,
            // so m_local and l_local stay warp-uniform
            float m_new = fmaxf(m_local, score);
            float scale_old = __expf(m_local - m_new);
            float p = __expf(score - m_new);

            l_local = l_local * scale_old + p;

            // V accumulation: each lane updates only its own output dims
            const float *V_ptr =
                V_batch + physical_kv_pos * n_kv_heads * head_dim +
                kv_head_idx * head_dim;
            int o_idx = 0;
            for (int d = lane_id; d < head_dim; d += WARP_SIZE, o_idx++)
            {
                O_lane[o_idx] = O_lane[o_idx] * scale_old + p * V_ptr[d];
            }

            m_local = m_new;
        }

        // =================================================================
        // Inter-warp reduction using shared memory
        //
        // Each warp has (m_local, l_local) uniform across its lanes,
        // and O distributed across lanes (each lane owns head_dim/32 dims).
        // We merge all warps' results, then write the final partial output.
        // =================================================================
        __shared__ float block_m[8];
        __shared__ float block_l[8];
        __shared__ float block_O[8 * 256]; // 8 warps × max head_dim=256

        if (lane_id == 0)
        {
            block_m[warp_id] = m_local;
            block_l[warp_id] = l_local;
        }
        // Each lane writes its O values to the correct positions in block_O
        {
            int o_idx = 0;
            for (int d = lane_id; d < head_dim; d += WARP_SIZE, o_idx++)
            {
                block_O[warp_id * head_dim + d] = O_lane[o_idx];
            }
        }
        __syncthreads();

        // Thread 0 computes per-warp rescaling factors
        __shared__ float warp_scales[8];
        if (tid == 0)
        {
            float final_m = block_m[0];
            float final_l = block_l[0];
            warp_scales[0] = 1.0f;

            for (int w = 1; w < num_warps; w++)
            {
                float other_m = block_m[w];
                float other_l = block_l[w];

                float m_new = fmaxf(final_m, other_m);
                float scale_self = __expf(final_m - m_new);
                float scale_other = __expf(other_m - m_new);

                for (int prev = 0; prev < w; prev++)
                    warp_scales[prev] *= scale_self;
                warp_scales[w] = scale_other;

                final_l = scale_self * final_l + scale_other * other_l;
                final_m = m_new;
            }

            m_partial[partial_idx] = final_m;
            l_partial[partial_idx] = final_l;
        }
        __syncthreads();

        // Parallel output write — all threads cooperate
        float *O_out = O_partial + partial_idx * head_dim;
        for (int d = tid; d < head_dim; d += num_threads)
        {
            float sum = 0.0f;
            for (int w = 0; w < num_warps; w++)
            {
                sum += warp_scales[w] * block_O[w * head_dim + d];
            }
            O_out[d] = sum;
        }
    }

    /**
     * @brief Flash Decoding reduction kernel
     *
     * Combines partial outputs from all splits using stable softmax merge.
     */
    __global__ void flash_decoding_reduce_fp32_kernel(
        const float *__restrict__ O_partial,
        const float *__restrict__ m_partial,
        const float *__restrict__ l_partial,
        float *__restrict__ O,
        int n_heads,
        int head_dim,
        int max_num_splits,
        int kv_len,
        int min_kv_per_split,
        const llaminar2::attention::AttentionDeviceParams *__restrict__ device_params)
    {
        const int head_idx = blockIdx.x;
        const int batch_idx = blockIdx.y;
        const int output_tile = blockIdx.z;
        const int output_tiles = gridDim.z;

        const int tid = threadIdx.x;
        const int kv_len_runtime =
            device_params ? device_params->kv_len : kv_len;
        const int row_num_splits = activeDecodeSplits(
            kv_len_runtime,
            max_num_splits,
            min_kv_per_split);
        const int base_idx =
            (batch_idx * n_heads + head_idx) * max_num_splits;

        __shared__ float global_l;
        __shared__ float split_scales[32];

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
                float scale = __expf(m_partial[base_idx + s] - m_max);
                split_scales[s] = scale;
                l_sum += scale * l_partial[base_idx + s];
            }
            global_l = l_sum;
        }
        __syncthreads();

        float inv_l = (global_l > 0.0f) ? (1.0f / global_l) : 0.0f;
        float *O_out = O + (batch_idx * n_heads + head_idx) * head_dim;
        const int output_begin =
            (head_dim * output_tile) / output_tiles;
        const int output_end =
            (head_dim * (output_tile + 1)) / output_tiles;

        /*
         * Output tiles duplicate only the tiny scalar softmax merge above.
         * Each output element retains the exact ascending split traversal used
         * by the one-block decoder, so tiling changes occupancy without changing
         * FP32 operation order or byte-level verifier results.
         */
        for (int d = output_begin + tid;
             d < output_end;
             d += blockDim.x)
        {
            float O_sum = 0.0f;
            for (int s = 0; s < row_num_splits; s++)
            {
                const float *O_s = O_partial + (base_idx + s) * head_dim;
                O_sum += split_scales[s] * O_s[d];
            }
            O_out[d] = O_sum * inv_l;
        }
    }

    // =========================================================================
    // Flash Decoding kernel — FP16 KV cache variant
    //
    // Identical to flash_decoding_fp32_kernel but reads K/V from FP16 (half)
    // storage directly, eliminating the FP16→FP32 conversion kernel launches
    // and halving KV cache bandwidth.
    // =========================================================================

    /**
     * @brief Physical K/V ownership represented by one flash-decode grid row.
     */
    enum class FlashDecodeRowMode
    {
        OrdinaryBatch,      ///< One K/V bank per row, one shared parameter row.
        SharedKVVerifier,   ///< All rows share one bank and own row-local params.
        IndependentRequest ///< Rows map to independent fixed-stride request banks.
    };

    /**
     * @brief FP16-KV flash-decode phase shared by every grouped row policy.
     *
     * The arithmetic in an active block is intentionally identical for scalar,
     * shared-cache verifier, and independent-request execution.  The mode only
     * selects the K/V bank and the device-parameter row.  Row-local modes also
     * derive the split count with the exact scalar policy, allowing one fixed
     * launch geometry to remain byte-equivalent across unequal KV lengths.
     */
    template <FlashDecodeRowMode ROW_MODE>
    __global__ __launch_bounds__(256, 4) void flash_decoding_fp16kv_kernel(
        const float *__restrict__ Q,
        const half *__restrict__ K_cache,
        const half *__restrict__ V_cache,
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
        int query_rows_per_request,
        int head_start = 0,
        int gqa_n_rep = 0)
    {
        constexpr bool ROW_LOCAL_PARAMS =
            ROW_MODE != FlashDecodeRowMode::OrdinaryBatch;
        constexpr bool SHARED_KV =
            ROW_MODE == FlashDecodeRowMode::SharedKVVerifier;

        int kv_stride = kv_len;
        int kv_len_runtime = kv_len;
        int ring_row_origin = 0;
        int ring_row_capacity = 0;
        if (device_params)
        {
            const int param_row = ROW_LOCAL_PARAMS ? blockIdx.z : 0;
            kv_len_runtime = device_params[param_row].kv_len;
            kv_stride = device_params[param_row].kv_stride;
            ring_row_origin = device_params[param_row].ring_row_origin;
            ring_row_capacity = device_params[param_row].ring_row_capacity;
        }

        const int head_idx = blockIdx.x;
        const int split_idx = blockIdx.y;
        const int batch_idx = blockIdx.z;

        const int effective_gqa = (gqa_n_rep > 0) ? gqa_n_rep : ((n_heads == n_kv_heads) ? 1 : (n_heads / n_kv_heads));
        const int kv_head_idx = (gqa_n_rep > 0)
                                    ? (head_start + head_idx) / effective_gqa
                                    : head_idx / effective_gqa;

        /*
         * Ordinary and grouped rows share the same device-side policy.  Their
         * physical grid is the graph-stable envelope; the live KV length
         * selects the active prefix and therefore the exact serial partition.
         */
        const int row_num_splits =
            activeDecodeSplits(kv_len_runtime, num_splits, 16);

        const int partial_idx =
            (batch_idx * n_heads + head_idx) * num_splits + split_idx;
        if (split_idx >= row_num_splits)
        {
            return;
        }

        const int split_size =
            (kv_len_runtime + row_num_splits - 1) / row_num_splits;
        const int kv_start = split_idx * split_size;
        const int kv_end = min(kv_start + split_size, kv_len_runtime);

        if (kv_start >= kv_len_runtime)
        {
            if (threadIdx.x == 0)
            {
                m_partial[partial_idx] = -FLT_MAX;
                l_partial[partial_idx] = 0.0f;
            }
            float *O_out = O_partial + partial_idx * head_dim;
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            {
                O_out[d] = 0.0f;
            }
            return;
        }

        const int tid = threadIdx.x;
        const int num_threads = blockDim.x;
        const int warp_id = tid / WARP_SIZE;
        const int lane_id = tid % WARP_SIZE;
        const int num_warps = num_threads / WARP_SIZE;

        extern __shared__ char smem[];
        float *Q_shared = reinterpret_cast<float *>(smem);

        const float *Q_ptr = Q + (batch_idx * n_heads + head_idx) * head_dim;
        for (int d = tid; d < head_dim; d += num_threads)
        {
            Q_shared[d] = Q_ptr[d];
        }
        __syncthreads();

        constexpr int MAX_DIMS_PER_LANE = 8;
        float O_lane[MAX_DIMS_PER_LANE] = {0};
        float m_local = -FLT_MAX;
        float l_local = 0.0f;

        int request_idx = batch_idx;
        if constexpr (ROW_MODE == FlashDecodeRowMode::IndependentRequest)
        {
            request_idx = batch_idx / max(1, query_rows_per_request);
        }
        const size_t kv_batch_offset =
            SHARED_KV
                ? 0
                : static_cast<size_t>(request_idx) *
                      static_cast<size_t>(kv_stride) *
                      static_cast<size_t>(n_kv_heads) *
                      static_cast<size_t>(head_dim);
        const half *K_batch = K_cache + kv_batch_offset;
        const half *V_batch = V_cache + kv_batch_offset;

        for (int kv_pos = kv_start + warp_id; kv_pos < kv_end; kv_pos += num_warps)
        {
            const int physical_kv_pos = ring_row_capacity > 0
                                            ? (ring_row_origin + kv_pos) %
                                                  ring_row_capacity
                                            : kv_pos;
            const half *K_ptr =
                K_batch + physical_kv_pos * n_kv_heads * head_dim + kv_head_idx * head_dim;

            // Cooperative dot product across head_dim
            float partial_dot = 0.0f;
            for (int d = lane_id; d < head_dim; d += WARP_SIZE)
            {
                partial_dot += Q_shared[d] * __half2float(K_ptr[d]);
            }

            float score = warpReduceSum(partial_dot) * softmax_scale;

            float m_new = fmaxf(m_local, score);
            float scale_old = __expf(m_local - m_new);
            float p = __expf(score - m_new);

            l_local = l_local * scale_old + p;

            const half *V_ptr =
                V_batch + physical_kv_pos * n_kv_heads * head_dim + kv_head_idx * head_dim;
            int o_idx = 0;
            for (int d = lane_id; d < head_dim; d += WARP_SIZE, o_idx++)
            {
                O_lane[o_idx] = O_lane[o_idx] * scale_old + p * __half2float(V_ptr[d]);
            }

            m_local = m_new;
        }

        __shared__ float block_m[8];
        __shared__ float block_l[8];
        __shared__ float block_O[8 * 256];

        if (lane_id == 0)
        {
            block_m[warp_id] = m_local;
            block_l[warp_id] = l_local;
        }
        {
            int o_idx = 0;
            for (int d = lane_id; d < head_dim; d += WARP_SIZE, o_idx++)
            {
                block_O[warp_id * head_dim + d] = O_lane[o_idx];
            }
        }
        __syncthreads();

        __shared__ float warp_scales[8];
        if (tid == 0)
        {
            float final_m = block_m[0];
            float final_l = block_l[0];
            warp_scales[0] = 1.0f;

            for (int w = 1; w < num_warps; w++)
            {
                float other_m = block_m[w];
                float other_l = block_l[w];

                float m_new = fmaxf(final_m, other_m);
                float scale_self = __expf(final_m - m_new);
                float scale_other = __expf(other_m - m_new);

                for (int prev = 0; prev < w; prev++)
                    warp_scales[prev] *= scale_self;
                warp_scales[w] = scale_other;

                final_l = scale_self * final_l + scale_other * other_l;
                final_m = m_new;
            }

            m_partial[partial_idx] = final_m;
            l_partial[partial_idx] = final_l;
        }
        __syncthreads();

        float *O_out = O_partial + partial_idx * head_dim;
        for (int d = tid; d < head_dim; d += num_threads)
        {
            float sum = 0.0f;
            for (int w = 0; w < num_warps; w++)
            {
                sum += warp_scales[w] * block_O[w * head_dim + d];
            }
            O_out[d] = sum;
        }
    }

    /**
     * @brief Reduce grouped verifier partials with each row's serial split count.
     *
     * The partial arena has a fixed `max_num_splits` stride so its address and
     * graph geometry remain stable. Only the prefix selected by the row-local
     * KV length participates in the reduction. The scalar decoder's reduction
     * visits the same split indices in the same order and applies the same FP32
     * expressions, which is the byte-equivalence requirement for MTP state
     * publication.
     */
    __global__ void flash_decoding_grouped_row_reduce_fp32_kernel(
        const float *__restrict__ O_partial,
        const float *__restrict__ m_partial,
        const float *__restrict__ l_partial,
        float *__restrict__ O,
        int n_heads,
        int head_dim,
        int max_num_splits,
        const llaminar2::attention::AttentionDeviceParams *__restrict__ device_params)
    {
        const int head_idx = blockIdx.x;
        const int verifier_row = blockIdx.y;
        const int output_tile = blockIdx.z;
        const int output_tiles = gridDim.z;
        const int tid = threadIdx.x;

        const int row_kv_len = device_params[verifier_row].kv_len;
        const int row_num_splits = min(
            max_num_splits,
            max(1, row_kv_len / 16));
        const int base_idx =
            (verifier_row * n_heads + head_idx) * max_num_splits;

        __shared__ float global_l;
        __shared__ float split_scales[32];

        if (tid == 0)
        {
            float m_max = -FLT_MAX;
            for (int split = 0; split < row_num_splits; ++split)
            {
                m_max = fmaxf(m_max, m_partial[base_idx + split]);
            }
            float l_sum = 0.0f;
            for (int split = 0; split < row_num_splits; ++split)
            {
                const float scale =
                    __expf(m_partial[base_idx + split] - m_max);
                split_scales[split] = scale;
                l_sum += scale * l_partial[base_idx + split];
            }
            global_l = l_sum;
        }
        __syncthreads();

        const float inv_l =
            global_l > 0.0f ? (1.0f / global_l) : 0.0f;
        float *O_out =
            O + (verifier_row * n_heads + head_idx) * head_dim;
        const int output_begin =
            (head_dim * output_tile) / output_tiles;
        const int output_end =
            (head_dim * (output_tile + 1)) / output_tiles;
        for (int d = output_begin + tid;
             d < output_end;
             d += blockDim.x)
        {
            float O_sum = 0.0f;
            for (int split = 0; split < row_num_splits; ++split)
            {
                const float *O_s =
                    O_partial + (base_idx + split) * head_dim;
                O_sum += split_scales[split] * O_s[d];
            }
            O_out[d] = O_sum * inv_l;
        }
    }

    // =========================================================================
    // Flash Decoding kernel — Q8_1 KV cache variant (fused inline dequant)
    //
    // Reads K/V directly from Q8_1 block format, performing int8→float
    // dequantization inline in the attention inner loop. This eliminates
    // the separate dequant kernel + FP32 workspace buffer.
    //
    // Q8_1Block layout: { uint16_t d (FP16 scale), int16_t sum_qs, int8_t qs[32] }
    // Total: 36 bytes per block, each block covers 32 elements.
    // =========================================================================

    /**
     * @brief Q8_1 block structure for inline dequantization in attention kernel.
     * Must match host-side Q8_1Block in BlockStructures.h.
     */
    struct GpuQ8_1BlockInline
    {
        uint16_t d;     // FP16 scale factor
        int16_t sum_qs; // pre-computed sum (unused in attention)
        int8_t qs[32];  // 32 quantized int8 values
    };
    static_assert(sizeof(GpuQ8_1BlockInline) == 36, "GpuQ8_1BlockInline must be 36 bytes");

    __global__ __launch_bounds__(256, 4) void flash_decoding_q8kv_kernel(
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
        int head_start = 0,
        int gqa_n_rep = 0)
    {
        // Q8_1 block addressing constants
        constexpr int Q8_BS = 32;

        int kv_stride = kv_len;
        int kv_len_runtime = kv_len;
        if (device_params)
        {
            kv_len_runtime = device_params->kv_len;
            kv_stride = device_params->kv_stride;
        }

        const int head_idx = blockIdx.x;
        const int split_idx = blockIdx.y;
        const int batch_idx = blockIdx.z;

        const int effective_gqa = (gqa_n_rep > 0) ? gqa_n_rep : ((n_heads == n_kv_heads) ? 1 : (n_heads / n_kv_heads));
        const int kv_head_idx = (gqa_n_rep > 0)
                                    ? (head_start + head_idx) / effective_gqa
                                    : head_idx / effective_gqa;

        const int row_num_splits =
            activeDecodeSplits(kv_len_runtime, num_splits, 16);
        if (split_idx >= row_num_splits)
            return;

        const int split_size =
            (kv_len_runtime + row_num_splits - 1) / row_num_splits;
        const int kv_start = split_idx * split_size;
        const int kv_end = min(kv_start + split_size, kv_len_runtime);

        const int partial_idx = (batch_idx * n_heads + head_idx) * num_splits + split_idx;

        if (kv_start >= kv_len_runtime)
        {
            if (threadIdx.x == 0)
            {
                m_partial[partial_idx] = -FLT_MAX;
                l_partial[partial_idx] = 0.0f;
            }
            float *O_out = O_partial + partial_idx * head_dim;
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            {
                O_out[d] = 0.0f;
            }
            return;
        }

        const int tid = threadIdx.x;
        const int num_threads = blockDim.x;
        const int warp_id = tid / WARP_SIZE;
        const int lane_id = tid % WARP_SIZE;
        const int num_warps = num_threads / WARP_SIZE;

        extern __shared__ char smem[];
        float *Q_shared = reinterpret_cast<float *>(smem);

        const float *Q_ptr = Q + (batch_idx * n_heads + head_idx) * head_dim;
        for (int d = tid; d < head_dim; d += num_threads)
        {
            Q_shared[d] = Q_ptr[d];
        }
        __syncthreads();

        constexpr int MAX_DIMS_PER_LANE = 8;
        float O_lane[MAX_DIMS_PER_LANE] = {0};
        float m_local = -FLT_MAX;
        float l_local = 0.0f;

        // Q8_1 block addressing: KV cache is stored as Q8_1Block arrays
        // Layout: [batch, kv_pos, n_kv_heads * blocks_per_head] (row-major blocks)
        const int bph = head_dim / Q8_BS; // blocks per head
        const int bpr = n_kv_heads * bph; // blocks per KV row
        const int row_byte_stride = bpr * static_cast<int>(sizeof(GpuQ8_1BlockInline));

        const char *K_base = static_cast<const char *>(K_cache) + batch_idx * kv_stride * row_byte_stride + kv_head_idx * bph * static_cast<int>(sizeof(GpuQ8_1BlockInline));
        const char *V_base = static_cast<const char *>(V_cache) + batch_idx * kv_stride * row_byte_stride + kv_head_idx * bph * static_cast<int>(sizeof(GpuQ8_1BlockInline));

        for (int kv_pos = kv_start + warp_id; kv_pos < kv_end; kv_pos += num_warps)
        {
            // K dot product: inline Q8_1 dequant
            const GpuQ8_1BlockInline *Kq = reinterpret_cast<const GpuQ8_1BlockInline *>(
                K_base + kv_pos * row_byte_stride);

            float partial_dot = 0.0f;
            for (int d = lane_id; d < head_dim; d += WARP_SIZE)
            {
                const int bi = d / Q8_BS;
                const int bo = d % Q8_BS;
                __half h_scale;
                memcpy(&h_scale, &Kq[bi].d, sizeof(__half));
                float ks = __half2float(h_scale);
                partial_dot += Q_shared[d] * (static_cast<float>(Kq[bi].qs[bo]) * ks);
            }

            float score = warpReduceSum(partial_dot) * softmax_scale;

            float m_new = fmaxf(m_local, score);
            float scale_old = __expf(m_local - m_new);
            float p = __expf(score - m_new);

            l_local = l_local * scale_old + p;

            // V accumulation: inline Q8_1 dequant
            const GpuQ8_1BlockInline *Vq = reinterpret_cast<const GpuQ8_1BlockInline *>(
                V_base + kv_pos * row_byte_stride);

            int o_idx = 0;
            for (int d = lane_id; d < head_dim; d += WARP_SIZE, o_idx++)
            {
                const int bi = d / Q8_BS;
                const int bo = d % Q8_BS;
                __half h_scale;
                memcpy(&h_scale, &Vq[bi].d, sizeof(__half));
                float vs = __half2float(h_scale);
                O_lane[o_idx] = O_lane[o_idx] * scale_old + p * (static_cast<float>(Vq[bi].qs[bo]) * vs);
            }

            m_local = m_new;
        }

        // Inter-warp reduction (identical to FP32/FP16 variants)
        __shared__ float block_m[8];
        __shared__ float block_l[8];
        __shared__ float block_O[8 * 256];

        if (lane_id == 0)
        {
            block_m[warp_id] = m_local;
            block_l[warp_id] = l_local;
        }
        {
            int o_idx = 0;
            for (int d = lane_id; d < head_dim; d += WARP_SIZE, o_idx++)
            {
                block_O[warp_id * head_dim + d] = O_lane[o_idx];
            }
        }
        __syncthreads();

        __shared__ float warp_scales[8];
        if (tid == 0)
        {
            float final_m = block_m[0];
            float final_l = block_l[0];
            warp_scales[0] = 1.0f;

            for (int w = 1; w < num_warps; w++)
            {
                float other_m = block_m[w];
                float other_l = block_l[w];

                float m_new = fmaxf(final_m, other_m);
                float scale_self = __expf(final_m - m_new);
                float scale_other = __expf(other_m - m_new);

                for (int prev = 0; prev < w; prev++)
                    warp_scales[prev] *= scale_self;
                warp_scales[w] = scale_other;

                final_l = scale_self * final_l + scale_other * other_l;
                final_m = m_new;
            }

            m_partial[partial_idx] = final_m;
            l_partial[partial_idx] = final_l;
        }
        __syncthreads();

        float *O_out = O_partial + partial_idx * head_dim;
        for (int d = tid; d < head_dim; d += num_threads)
        {
            float sum = 0.0f;
            for (int w = 0; w < num_warps; w++)
            {
                sum += warp_scales[w] * block_O[w * head_dim + d];
            }
            O_out[d] = sum;
        }
    }

} // anonymous namespace

// =============================================================================
// Fused TQ KV Decode Attention Kernel (TQ8 K + TQ4 V)
//
// Reads TQ8/TQ4 blocks directly from the ring buffer, performing centroid
// lookup and rotation inline. Uses the "rotation trick":
//   dot(Q, dequant(K)) = K.norm/√D · dot(R·Q, centroids[K.indices])
// which reduces per-position cost from O(D²) to O(D).
//
// V accumulation happens in rotated centroid space, with a final Rᵀ multiply.
// =============================================================================

namespace
{
    // TQ codebooks (compile-time Lloyd-Max centroids for N(0,1))
    __constant__ float d_tq8_attn_cents[256];
    __constant__ float d_tq4_attn_cents[16];
    static std::atomic<bool> s_tq_attn_codebooks_uploaded{false};

    void upload_tq_attn_codebooks()
    {
        if (s_tq_attn_codebooks_uploaded.load(std::memory_order_acquire))
            return;
        // TQ4: 16-level Lloyd-Max centroids for N(0,1)
        static constexpr float TQ4_C[16] = {
            -2.732897f, -2.069364f, -1.618400f, -1.256565f,
            -0.942629f, -0.656982f, -0.388189f, -0.128443f,
            0.128443f, 0.388189f, 0.656982f, 0.942629f,
            1.256565f, 1.618400f, 2.069364f, 2.732897f};
        // TQ8: 256-level Lloyd-Max centroids for N(0,1)
        static constexpr float TQ8_C[256] = {
            -3.78920960f, -3.30915570f, -3.01834941f, -2.80658674f, -2.63340139f, -2.48449373f, -2.35561466f, -2.24269128f,
            -2.14211559f, -2.05341673f, -1.97122753f, -1.89669085f, -1.82880616f, -1.76743543f, -1.71260118f, -1.66326785f,
            -1.61619723f, -1.57234073f, -1.53127813f, -1.49353135f, -1.45867503f, -1.42624652f, -1.39503074f, -1.36621320f,
            -1.33910429f, -1.31285179f, -1.28799033f, -1.26373041f, -1.24022365f, -1.21824205f, -1.19704747f, -1.17650998f,
            -1.15636790f, -1.13686514f, -1.11771226f, -1.09937215f, -1.08145535f, -1.06413019f, -1.04700780f, -1.02985537f,
            -1.01289260f, -0.99635839f, -0.97985989f, -0.96346331f, -0.94718844f, -0.93142581f, -0.91591436f, -0.90098518f,
            -0.88661534f, -0.87208009f, -0.85743922f, -0.84298599f, -0.82868934f, -0.81421226f, -0.79995292f, -0.78608519f,
            -0.77253389f, -0.75930405f, -0.74655443f, -0.73399001f, -0.72171885f, -0.70948768f, -0.69729090f, -0.68519193f,
            -0.67294139f, -0.66097361f, -0.64903301f, -0.63699722f, -0.62520957f, -0.61343849f, -0.60164148f, -0.58989501f,
            -0.57817614f, -0.56638294f, -0.55476522f, -0.54346019f, -0.53228927f, -0.52154797f, -0.51064700f, -0.49990472f,
            -0.48919857f, -0.47810543f, -0.46704516f, -0.45603955f, -0.44486380f, -0.43340886f, -0.42221144f, -0.41127491f,
            -0.40069824f, -0.39020312f, -0.37980038f, -0.36931747f, -0.35883522f, -0.34820479f, -0.33763838f, -0.32712156f,
            -0.31680506f, -0.30650702f, -0.29633904f, -0.28598318f, -0.27587256f, -0.26576686f, -0.25534680f, -0.24487814f,
            -0.23456042f, -0.22439688f, -0.21428297f, -0.20450732f, -0.19472016f, -0.18452135f, -0.17464319f, -0.16479470f,
            -0.15506494f, -0.14514914f, -0.13510469f, -0.12490919f, -0.11475672f, -0.10477102f, -0.09481984f, -0.08470155f,
            -0.07488193f, -0.06495188f, -0.05493221f, -0.04509697f, -0.03520501f, -0.02532661f, -0.01568576f, -0.00597589f,
            0.00372220f, 0.01344266f, 0.02319991f, 0.03286659f, 0.04291259f, 0.05293084f, 0.06276978f, 0.07249805f,
            0.08250652f, 0.09261067f, 0.10255274f, 0.11248623f, 0.12212010f, 0.13166577f, 0.14129025f, 0.15095016f,
            0.16084716f, 0.17080866f, 0.18087213f, 0.19098914f, 0.20085125f, 0.21076398f, 0.22071999f, 0.23092821f,
            0.24115537f, 0.25143579f, 0.26179457f, 0.27212587f, 0.28246218f, 0.29251403f, 0.30247530f, 0.31238413f,
            0.32260880f, 0.33276638f, 0.34315947f, 0.35366595f, 0.36426541f, 0.37491897f, 0.38535890f, 0.39584789f,
            0.40643987f, 0.41705394f, 0.42789838f, 0.43888989f, 0.44957283f, 0.46015123f, 0.47093272f, 0.48179030f,
            0.49290875f, 0.50388461f, 0.51482475f, 0.52595741f, 0.53740937f, 0.54880124f, 0.56035239f, 0.57217956f,
            0.58371091f, 0.59516978f, 0.60679603f, 0.61841190f, 0.63014859f, 0.64220595f, 0.65432996f, 0.66648364f,
            0.67858601f, 0.69097805f, 0.70347679f, 0.71617913f, 0.72882068f, 0.74160296f, 0.75484610f, 0.76858562f,
            0.78214169f, 0.79558426f, 0.80941713f, 0.82344604f, 0.83751440f, 0.85179305f, 0.86665273f, 0.88163447f,
            0.89666569f, 0.91165745f, 0.92661709f, 0.94195455f, 0.95747328f, 0.97319126f, 0.98969555f, 1.00626135f,
            1.02299738f, 1.03975272f, 1.05691993f, 1.07431412f, 1.09259737f, 1.11127019f, 1.13069904f, 1.15059435f,
            1.17105842f, 1.19241130f, 1.21456146f, 1.23746312f, 1.26111495f, 1.28548789f, 1.31057048f, 1.33675921f,
            1.36415398f, 1.39277720f, 1.42348731f, 1.45656335f, 1.49078155f, 1.52804673f, 1.56803429f, 1.61146212f,
            1.65781057f, 1.70859325f, 1.76440656f, 1.82453620f, 1.89094412f, 1.96517146f, 2.04702711f, 2.13570380f,
            2.23243070f, 2.34583116f, 2.47412658f, 2.62253070f, 2.79892921f, 3.01957417f, 3.30903292f, 3.74206471f};
        cudaMemcpyToSymbol(d_tq8_attn_cents, TQ8_C, sizeof(TQ8_C));
        cudaMemcpyToSymbol(d_tq4_attn_cents, TQ4_C, sizeof(TQ4_C));
        if (cudaGetLastError() == cudaSuccess)
        {
            s_tq_attn_codebooks_uploaded.store(true, std::memory_order_release);
        }
    }
} // anonymous namespace

namespace
{
    /**
     * @brief Fused TQ8/TQ4 flash decoding kernel.
     *
     * Uses rotation trick: dot(Q, dequant(K)) = norm/√D · dot(R·Q, centroids[indices])
     * V accumulated in rotated centroid space, post-rotated at end.
     *
     * Grid: (n_heads, num_splits, batch_size)
     * Block: 256 threads (8 warps)
     * Shared memory: Q[D] + Q_rot[D] + block_m[8] + block_l[8] + block_O[8*D] + warp_scales[8]
     */
    __global__ __launch_bounds__(256, 2) void flash_decoding_tqkv_kernel(
        const float *__restrict__ Q,
        const void *__restrict__ K_cache,     // TQ8Block<D> ring buffer
        const void *__restrict__ V_cache,     // TQ4Block<D> ring buffer
        const float *__restrict__ rotation,   // R[kv_head][D][D]
        const float *__restrict__ rotation_t, // Rᵀ[kv_head][D][D]
        float *__restrict__ O_partial,
        float *__restrict__ m_partial,
        float *__restrict__ l_partial,
        int kv_count, // actual cached tokens
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int num_splits,
        float softmax_scale,
        int max_seq_len,  // ring buffer capacity (stride)
        int tail,         // ring buffer tail position
        int k_block_size, // sizeof(TQ8Block<D>)
        int v_block_size, // sizeof(TQ4Block<D>)
        const llaminar2::attention::AttentionDeviceParams *__restrict__ device_params,
        int head_start = 0,
        int gqa_n_rep = 0)
    {
        const int head_idx = blockIdx.x;
        const int split_idx = blockIdx.y;
        const int batch_idx = blockIdx.z;

        const int effective_gqa = (gqa_n_rep > 0) ? gqa_n_rep : ((n_heads == n_kv_heads) ? 1 : (n_heads / n_kv_heads));
        const int kv_head_idx = (gqa_n_rep > 0)
                                    ? (head_start + head_idx) / effective_gqa
                                    : head_idx / effective_gqa;

        int kv_count_rt = kv_count;
        if (device_params)
            kv_count_rt = device_params->kv_len;

        const int row_num_splits =
            activeDecodeSplits(kv_count_rt, num_splits, 64);
        if (split_idx >= row_num_splits)
            return;

        const int split_size =
            (kv_count_rt + row_num_splits - 1) / row_num_splits;
        const int kv_start = split_idx * split_size;
        const int kv_end = min(kv_start + split_size, kv_count_rt);

        const int partial_idx = (batch_idx * n_heads + head_idx) * num_splits + split_idx;

        if (kv_start >= kv_count_rt)
        {
            if (threadIdx.x == 0)
            {
                m_partial[partial_idx] = -FLT_MAX;
                l_partial[partial_idx] = 0.0f;
            }
            float *O_out = O_partial + partial_idx * head_dim;
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
                O_out[d] = 0.0f;
            return;
        }

        const int tid = threadIdx.x;
        const int num_threads = blockDim.x;
        const int warp_id = tid / WARP_SIZE;
        const int lane_id = tid % WARP_SIZE;
        const int num_warps = num_threads / WARP_SIZE;

        // Shared memory: Q[D] + Q_rot[D]
        extern __shared__ char smem[];
        float *Q_shared = reinterpret_cast<float *>(smem);
        float *Q_rot = Q_shared + head_dim;

        // Load Q into shared memory
        const float *Q_ptr = Q + (batch_idx * n_heads + head_idx) * head_dim;
        for (int d = tid; d < head_dim; d += num_threads)
            Q_shared[d] = Q_ptr[d];
        __syncthreads();

        // Compute Q_rot = R * Q (rotation matrix multiply, once per head)
        // R layout: [n_kv_heads, D, D]
        const float *R = rotation + static_cast<size_t>(kv_head_idx) * head_dim * head_dim;
        if (tid < head_dim)
        {
            const float *R_row = R + tid * head_dim;
            float sum = 0.0f;
            for (int j = 0; j < head_dim; j++)
                sum += R_row[j] * Q_shared[j];
            Q_rot[tid] = sum;
        }
        __syncthreads();

        // Online softmax + KV loop
        constexpr int MAX_DIMS_PER_LANE = 8; // head_dim/WARP_SIZE = 256/32 = 8
        float O_lane[MAX_DIMS_PER_LANE] = {0};
        float m_local = -FLT_MAX;
        float l_local = 0.0f;

        const float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(head_dim));

        // Ring buffer addressing
        const int k_row_stride = n_kv_heads * k_block_size;
        const int v_row_stride = n_kv_heads * v_block_size;
        const uint8_t *K_base = static_cast<const uint8_t *>(K_cache) + kv_head_idx * k_block_size;
        const uint8_t *V_base = static_cast<const uint8_t *>(V_cache) + kv_head_idx * v_block_size;

        for (int logical_pos = kv_start + warp_id; logical_pos < kv_end; logical_pos += num_warps)
        {
            const int phys_pos = (tail + logical_pos) % max_seq_len;

            // ---- TQ8 K dot product ----
            const uint8_t *k_block = K_base + phys_pos * k_row_stride;
            // Key scoring retains the source norm. The second scalar is fitted
            // for value reconstruction and would bias the QK distribution.
            float k_norm = reinterpret_cast<const float *>(k_block)[0];
            const uint8_t *k_indices = k_block + 2 * sizeof(float); // skip source/reconstruction norms

            float partial_dot = 0.0f;
            for (int d = lane_id; d < head_dim; d += WARP_SIZE)
                partial_dot += Q_rot[d] * d_tq8_attn_cents[k_indices[d]];

            float score = warpReduceSum(partial_dot) * k_norm * inv_sqrt_d * softmax_scale;

            // Online softmax update
            float m_new = fmaxf(m_local, score);
            float scale_old = __expf(m_local - m_new);
            float p = __expf(score - m_new);
            l_local = l_local * scale_old + p;

            // ---- TQ4 V accumulation (in rotated centroid space) ----
            const uint8_t *v_block = V_base + phys_pos * v_row_stride;
            float v_norm = reinterpret_cast<const float *>(v_block)[1];
            const uint8_t *v_mse = v_block + 2 * sizeof(float);
            const uint8_t *v_high = v_mse + head_dim * 3 / 8;
            float weight = p * v_norm * inv_sqrt_d;

            int o_idx = 0;
            for (int d = lane_id; d < head_dim; d += WARP_SIZE, o_idx++)
            {
                // Unpack TQ4 3+1 bit index
                int group8 = d / 8;
                int within = d % 8;
                int byte_off = group8 * 3;
                uint8_t b0 = v_mse[byte_off];
                uint8_t b1 = v_mse[byte_off + 1];
                uint8_t b2 = v_mse[byte_off + 2];

                uint8_t low3;
                switch (within)
                {
                case 0:
                    low3 = b0 & 0x07;
                    break;
                case 1:
                    low3 = (b0 >> 3) & 0x07;
                    break;
                case 2:
                    low3 = ((b0 >> 6) | (b1 << 2)) & 0x07;
                    break;
                case 3:
                    low3 = (b1 >> 1) & 0x07;
                    break;
                case 4:
                    low3 = (b1 >> 4) & 0x07;
                    break;
                case 5:
                    low3 = ((b1 >> 7) | (b2 << 1)) & 0x07;
                    break;
                case 6:
                    low3 = (b2 >> 2) & 0x07;
                    break;
                default:
                    low3 = (b2 >> 5) & 0x07;
                    break;
                }
                uint8_t high1 = (v_high[group8] >> within) & 0x01;
                uint8_t full_idx = low3 | (high1 << 3);

                O_lane[o_idx] = O_lane[o_idx] * scale_old + weight * d_tq4_attn_cents[full_idx];
            }

            m_local = m_new;
        }

        // ---- Inter-warp reduction ----
        // Reuse smem after Q_rot no longer needed
        float *block_m = reinterpret_cast<float *>(smem);
        float *block_l = block_m + 8;
        float *block_O = block_l + 8;

        __syncthreads(); // ensure all warps done before reusing smem

        if (lane_id == 0)
        {
            block_m[warp_id] = m_local;
            block_l[warp_id] = l_local;
        }
        {
            int o_idx = 0;
            for (int d = lane_id; d < head_dim; d += WARP_SIZE, o_idx++)
                block_O[warp_id * head_dim + d] = O_lane[o_idx];
        }
        __syncthreads();

        float *warp_scales = block_O + 8 * head_dim;

        if (tid == 0)
        {
            float final_m = block_m[0];
            float final_l = block_l[0];
            warp_scales[0] = 1.0f;

            for (int w = 1; w < num_warps; w++)
            {
                float other_m = block_m[w];
                float other_l = block_l[w];
                float m_new = fmaxf(final_m, other_m);
                float s_self = __expf(final_m - m_new);
                float s_other = __expf(other_m - m_new);
                for (int prev = 0; prev < w; prev++)
                    warp_scales[prev] *= s_self;
                warp_scales[w] = s_other;
                final_l = s_self * final_l + s_other * other_l;
                final_m = m_new;
            }

            m_partial[partial_idx] = final_m;
            l_partial[partial_idx] = final_l;
        }
        __syncthreads();

        // ---- Post-rotation: output = Rᵀ * V_accum ----
        // Sum warps into first head_dim of block_O (overwrite warp 0)
        for (int d = tid; d < head_dim; d += num_threads)
        {
            float sum = 0.0f;
            for (int w = 0; w < num_warps; w++)
                sum += warp_scales[w] * block_O[w * head_dim + d];
            block_O[d] = sum; // rotated-space output
        }
        __syncthreads();

        // Post-rotate: O_out[d] = Σⱼ Rᵀ[d][j] * block_O[j]
        const float *Rt = rotation_t + static_cast<size_t>(kv_head_idx) * head_dim * head_dim;
        float *O_out = O_partial + partial_idx * head_dim;

        // All 256 threads cooperate: each thread handles one or two output dims
        for (int d = tid; d < head_dim; d += num_threads)
        {
            const float *Rt_row = Rt + d * head_dim;
            float sum = 0.0f;
            for (int j = 0; j < head_dim; j++)
                sum += Rt_row[j] * block_O[j];
            O_out[d] = sum;
        }
    }

} // anonymous namespace

// =============================================================================
// KV Cache Conversion Kernels (FP16→FP32, Q8_1→FP32)
//
// These replace the CPU roundtrip (D2H → convert → H2D) that was used
// previously in compute_tensor(). The CPU path had a bug with head_dim=128
// that produced garbage output on 7B models.
// =============================================================================

/**
 * @brief Convert FP16 (uint16_t) to FP32 on GPU
 *
 * Simple element-wise conversion using CUDA's __half2float intrinsic.
 * One thread per element.
 */
__global__ void convert_fp16_to_fp32_kernel(const uint16_t *__restrict__ src,
                                            float *__restrict__ dst,
                                            int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        __half h;
        memcpy(&h, &src[idx], sizeof(__half));
        dst[idx] = __half2float(h);
    }
}

/**
 * @brief Dynamic FP16→FP32 conversion for CUDA graph replay
 *
 * Reads kv_len from device_params at runtime instead of using a frozen scalar.
 * This is essential for graph capture correctness: the captured graph records
 * a fixed grid size, but the actual element count changes between replay steps
 * as kv_len grows. The kernel computes the actual count from device_params->kv_len
 * and skips threads beyond that count.
 *
 * @param src           FP16 source data (KV cache)
 * @param dst           FP32 destination buffer (workspace)
 * @param cols_per_row  n_kv_heads * head_dim (constant across replays)
 * @param device_params Device-side params containing dynamic kv_len
 */
__global__ void convert_fp16_to_fp32_dynamic_kernel(
    const uint16_t *__restrict__ src,
    float *__restrict__ dst,
    int cols_per_row,
    const llaminar2::attention::AttentionDeviceParams *__restrict__ device_params)
{
    const int kv_len = device_params->kv_len;
    const int count = kv_len * cols_per_row;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        __half h;
        memcpy(&h, &src[idx], sizeof(__half));
        dst[idx] = __half2float(h);
    }
}

/**
 * @brief Q8_1 block structure for GPU dequantization
 *
 * Mirrors the host-side Q8_1Block in BlockStructures.h.
 * Defined locally to avoid including host-only headers in CUDA code.
 */
struct GpuQ8_1Block
{
    uint16_t d;     // FP16 scale factor
    int16_t sum_qs; // pre-computed sum (not used for dequant)
    int8_t qs[32];  // 32 quantized int8 values
};
static_assert(sizeof(GpuQ8_1Block) == 36, "GpuQ8_1Block must be 36 bytes");

/**
 * @brief Dequantize Q8_1 blocks to FP32 on GPU
 *
 * Each thread processes one element within a block.
 * Grid: (blocks_per_row * rows) blocks × 32 threads
 */
__global__ void dequant_q8_1_to_fp32_kernel(const GpuQ8_1Block *__restrict__ src,
                                            float *__restrict__ dst,
                                            int rows,
                                            int cols,
                                            int blocks_per_row)
{
    int block_idx = blockIdx.x;
    int elem_idx = threadIdx.x; // 0..31

    int total_blocks = rows * blocks_per_row;
    if (block_idx >= total_blocks)
        return;

    int row = block_idx / blocks_per_row;
    int block_in_row = block_idx % blocks_per_row;
    int col = block_in_row * 32 + elem_idx;

    if (col >= cols)
        return;

    const GpuQ8_1Block &blk = src[block_idx];
    // Convert FP16 scale to FP32 using CUDA intrinsic
    __half h_scale;
    memcpy(&h_scale, &blk.d, sizeof(__half));
    float scale = __half2float(h_scale);

    dst[row * cols + col] = scale * static_cast<float>(blk.qs[elem_idx]);
}

/**
 * @brief Dynamic Q8_1→FP32 dequantization for CUDA graph replay
 *
 * Same as dequant_q8_1_to_fp32_kernel but reads kv_len from device_params
 * to compute the actual row count at runtime. Essential for graph capture
 * correctness where the frozen row count would be stale on replay.
 */
__global__ void dequant_q8_1_to_fp32_dynamic_kernel(
    const GpuQ8_1Block *__restrict__ src,
    float *__restrict__ dst,
    int cols,
    int blocks_per_row,
    const llaminar2::attention::AttentionDeviceParams *__restrict__ device_params)
{
    const int rows = device_params->kv_len; // Dynamic row count from device memory
    const int block_idx = blockIdx.x;
    const int elem_idx = threadIdx.x; // 0..31

    const int total_blocks = rows * blocks_per_row;
    if (block_idx >= total_blocks)
        return;

    const int row = block_idx / blocks_per_row;
    const int block_in_row = block_idx % blocks_per_row;
    const int col = block_in_row * 32 + elem_idx;

    if (col >= cols)
        return;

    const GpuQ8_1Block &blk = src[block_idx];
    __half h_scale;
    memcpy(&h_scale, &blk.d, sizeof(__half));
    float scale = __half2float(h_scale);

    dst[row * cols + col] = scale * static_cast<float>(blk.qs[elem_idx]);
}

// =============================================================================
// Internal template launcher (cannot have extern "C" linkage)
// =============================================================================

/**
 * @brief Internal templated FA2 launcher — dispatches based on head_dim + KV_FP16 flag.
 * Query groups are selected from every compiled width through 6 (HD64), 4
 * (HD128), or 2 (HD256). P@V striping remains an independent HD256 choice.
 * Returns -1 on invalid input, -2 if GPU doesn't support SM 8.0.
 */
template <bool KV_FP16>
static int fa2_prefill_launch(
    const float *Q, const void *K, const void *V, float *O,
    int batch_size, int seq_len, int kv_len,
    int n_heads, int n_kv_heads, int head_dim,
    bool causal, int window_size, int position_offset,
    const llaminar2::attention::AttentionDeviceParams *device_params,
    const float *mask,
    cudaStream_t cuda_stream,
    int device_idx,
    int head_start,
    int gqa_n_rep)
{
    float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    cudaSetDevice(device_idx);
    const FA2DeviceConfig &dev_cfg = getFA2DeviceConfig(device_idx);

    if (dev_cfg.sm_major < 8)
    {
        printf("[cudaFlashAttn_prefill_fa2] Error: SM %d.%d not supported, requires SM >= 8.0\n",
               dev_cfg.sm_major, dev_cfg.sm_minor);
        return -2;
    }

    const llaminar2::cuda::fa2_policy::FA2HeadMappingGeometry head_mapping{
        .local_query_heads = n_heads,
        .visible_kv_heads = n_kv_heads,
        .head_start = head_start,
        .replicated_gqa_n_rep = gqa_n_rep,
    };
    if (!Q || !K || !V || !O || batch_size <= 0 || seq_len <= 0 ||
        kv_len <= 0 ||
        !llaminar2::cuda::fa2_policy::isValidFA2HeadMapping(head_mapping) ||
        head_dim <= 0 ||
        head_dim % 16 != 0 || head_dim > 256)
    {
        printf("[cudaFlashAttn_prefill_fa2] Error: invalid tensor or launch geometry "
               "(batch=%d, seq=%d, kv=%d, heads=%d, kv_heads=%d, head_dim=%d)\n",
               batch_size, seq_len, kv_len, n_heads, n_kv_heads, head_dim);
        return -1;
    }

    if (!cuda_stream)
    {
        printf("[cudaFlashAttn_prefill_fa2] Error: explicit non-null CUDA stream is required\n");
        return -1;
    }

    const int configured_q_warp_groups =
        llaminar2::debugEnv().attention.cuda_fa2_q_warp_groups;
    const llaminar2::cuda::fa2_policy::FA2QueryPartitionGeometry
        query_partition_geometry{
            .batch_size = batch_size,
            .query_rows = seq_len,
            .local_query_heads = n_heads,
            .head_dim = head_dim,
            .sm_count = dev_cfg.sm_count,
        };
    const int selected_q_warp_groups =
        llaminar2::cuda::fa2_policy::selectFA2QueryWarpGroups(
            query_partition_geometry,
            configured_q_warp_groups);
    if (selected_q_warp_groups == 0)
    {
        printf("[cudaFlashAttn_prefill_fa2] Error: FA2 query partition has no "
               "compiled specialization (head_dim=%d, requested_groups=%d, "
               "maximum_groups=%d, sm_count=%d)\n",
               head_dim,
               configured_q_warp_groups,
               llaminar2::cuda::fa2_policy::maximumFA2QueryWarpGroups(head_dim),
               dev_cfg.sm_count);
        return -1;
    }

    const int configured_hd256_pv_warps =
        llaminar2::debugEnv().attention.cuda_fa2_hd256_pv_warps;
    if (!llaminar2::cuda::fa2_policy::isCompiledFA2PVWarpGeometry(
            head_dim,
            head_dim > 128 ? configured_hd256_pv_warps : 1))
    {
        printf("[cudaFlashAttn_prefill_fa2] Error: LLAMINAR_FA2_HD256_PV_WARPS=%d is invalid for head_dim=%d; HD256 expects 1, 2, or 4\n",
               configured_hd256_pv_warps,
               head_dim);
        return -1;
    }

    FA2KernelConfig cfg = computeFA2Config(
        head_dim,
        dev_cfg.max_smem_optin,
        selected_q_warp_groups);

    if (cfg.tile_q <= 0 ||
        cfg.tile_kv <= 0 ||
        cfg.num_q_warp_groups <= 0 ||
        cfg.pv_warps_per_q_group <= 0 ||
        cfg.block_size <= 0 ||
        cfg.smem_size == 0)
    {
        printf("[cudaFlashAttn_prefill_fa2] Error: selected FA2 capture "
               "geometry cannot fit the device shared-memory contract "
               "(head_dim=%d, q_warp_groups=%d, forced_tile_kv=%d, "
               "max_dynamic_smem=%d)\n",
               head_dim,
               selected_q_warp_groups,
               llaminar2::debugEnv().attention.cuda_fa2_tile_kv,
               dev_cfg.max_smem_optin);
        return -1;
    }

    int num_q_tiles = (seq_len + cfg.tile_q - 1) / cfg.tile_q;
    dim3 grid(n_heads, num_q_tiles, batch_size);

    cudaError_t err;

    // Macro to reduce dispatch boilerplate
#define FA2_LAUNCH(QW, PVW, HD, TKV)                                                                       \
    do                                                                                                     \
    {                                                                                                      \
        auto kernel_fn = flash_attention_2_pipelined_kernel<QW, PVW, HD, TKV, KV_FP16>;                    \
        err = cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, cfg.smem_size); \
        if (err != cudaSuccess)                                                                            \
        {                                                                                                  \
            printf("[cudaFlashAttn_prefill_fa2] cudaFuncSetAttribute<%d,%d,%d,%d,%d>(smem=%zu) FAILED: %s\n", \
                   QW, PVW, HD, TKV, (int)KV_FP16, cfg.smem_size, cudaGetErrorString(err));               \
            return -1;                                                                                     \
        }                                                                                                  \
        kernel_fn<<<grid, cfg.block_size, cfg.smem_size, cuda_stream>>>(                                   \
            Q, K, V, O,                                                                                    \
            batch_size, seq_len, kv_len, n_heads, n_kv_heads, head_dim,                                    \
            softmax_scale, causal, window_size, position_offset,                                           \
            device_params, mask, cfg.tile_q, cfg.tile_kv, cfg.qkv_pad, cfg.scores_pad,                     \
            head_start, gqa_n_rep,                                                                          \
            /*O_partial=*/nullptr, /*m_partial=*/nullptr, /*l_partial=*/nullptr,                             \
            /*context_partition_size=*/0, /*max_context_partitions=*/0,                                    \
            /*device_direct_partition_limit=*/0,                                                            \
            /*explicit_context_partition=*/-1);                                                             \
    } while (0)

#define FA2_LAUNCH_SELECTED_TILE(QW, PVW, HD) \
    do                                        \
    {                                         \
        if (cfg.tile_kv == 64)                \
            FA2_LAUNCH(QW, PVW, HD, 64);      \
        else if (cfg.tile_kv == 16)           \
            FA2_LAUNCH(QW, PVW, HD, 16);      \
        else                                  \
            FA2_LAUNCH(QW, PVW, HD, 32);      \
    } while (0)

    if (head_dim <= 64)
    {
        FA2_LAUNCH_SELECTED_TILE(6, 1, 64);
    }
    else if (head_dim <= 128)
    {
        FA2_LAUNCH_SELECTED_TILE(4, 1, 128);
    }
    else
    {
        if (cfg.pv_warps_per_q_group == 1)
            FA2_LAUNCH_SELECTED_TILE(2, 1, 256);
        else if (cfg.pv_warps_per_q_group == 2)
            FA2_LAUNCH_SELECTED_TILE(2, 2, 256);
        else if (cfg.pv_warps_per_q_group == 4)
            FA2_LAUNCH_SELECTED_TILE(2, 4, 256);
    }

#undef FA2_LAUNCH_SELECTED_TILE
#undef FA2_LAUNCH

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("[cudaFlashAttn_prefill_fa2] CUDA error: %s (smem=%zu bytes, tile_q=%d, tile_kv=%d, head_dim=%d, "
               "q_warp_groups=%d, pv_warps_per_q_group=%d, consumer_warps=%d, kv_fp16=%d, "
               "grid=(%d,%d,%d), block=%d)\n",
               cudaGetErrorString(err), cfg.smem_size, cfg.tile_q, cfg.tile_kv, head_dim,
               cfg.num_q_warp_groups, cfg.pv_warps_per_q_group, cfg.consumerWarps(),
               (int)KV_FP16, grid.x, grid.y, grid.z, cfg.block_size);
        return -1;
    }
    return 0;
}

/**
 * @brief Physical scheduling of fixed logical K/V partitions during capture.
 *
 * Both schedules execute the same compiled phase-one specialization and the
 * same deterministic reducer.  The sequence schedule records one phase-one
 * graph node per partition on the explicit stream; the context schedule records
 * one wider phase-one grid.  Their arithmetic is therefore identical while the
 * benchmark can measure whether additional context-grid residency is economic.
 */
enum class FA2ContextPartitionSchedule
{
    SequenceGraphNodes,
    ContextParallelGrid,
};

/**
 * @brief Immutable arguments shared by one captured partition transaction.
 */
struct FA2ContextTransactionArguments
{
    const float *Q = nullptr;
    const void *K = nullptr;
    const void *V = nullptr;
    float *O = nullptr;
    float *O_partial = nullptr;
    float *m_partial = nullptr;
    float *l_partial = nullptr;
    int batch_size = 0;
    int seq_len = 0;
    int kv_capacity = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    float softmax_scale = 0.0f;
    bool causal = false;
    int window_size = -1;
    int position_offset = 0;
    const llaminar2::attention::AttentionDeviceParams *device_params = nullptr;
    const float *mask = nullptr;
    int head_start = 0;
    int gqa_n_rep = 0;
    int context_partition_size = 0;
    int max_context_partitions = 0;
    int context_partition_slots = 0;
    int device_direct_partition_limit = 0;
    int reducer_dimension_warps = 0;
    void *capture_conditional = nullptr;
};

#if CUDART_VERSION >= 12030
/**
 * @brief Placement of one phase kernel in the adaptive captured transaction.
 */
enum class FA2AdaptivePhasePlacement
{
    ConcurrentGuardedRoot, ///< Parent sibling that retires for context work.
    ContextIfBody,         ///< Non-zero conditional body publishing summaries.
};

/**
 * @brief Append one compiled FA2 phase kernel to its typed adaptive placement.
 *
 * CUDA copies every pointed-to argument value while adding the node. Keeping
 * argument materialization in this helper prevents the direct and context
 * branches from drifting to different ABI order as fields are added.
 *
 * @param transaction Active parent-capture conditional builder.
 * @param placement Guarded parent root or non-zero context body.
 * @param function Exact compiled direct or context kernel specialization.
 * @param grid Immutable physical launch grid.
 * @param arguments Transaction tensor and model geometry.
 * @param config Compiled block/shared-memory geometry.
 * @param output_partial Context-summary output, or null for direct execution.
 * @param m_partial Context-summary maxima, or null for direct execution.
 * @param l_partial Context-summary sums, or null for direct execution.
 * @param explicit_context_partition Diagnostic partition index, normally -1.
 * @param semantic_name Stable node role used in fatal diagnostics.
 */
static bool appendFA2AdaptivePhaseNode(
    llaminar2::CUDAActiveCaptureConditional &transaction,
    FA2AdaptivePhasePlacement placement,
    void *function,
    dim3 grid,
    const FA2ContextTransactionArguments &arguments,
    const FA2KernelConfig &config,
    float *output_partial,
    float *m_partial,
    float *l_partial,
    int explicit_context_partition,
    const char *semantic_name)
{
    const float *Q = arguments.Q;
    const void *K = arguments.K;
    const void *V = arguments.V;
    float *O = arguments.O;
    int batch_size = arguments.batch_size;
    int seq_len = arguments.seq_len;
    int kv_capacity = arguments.kv_capacity;
    int n_heads = arguments.n_heads;
    int n_kv_heads = arguments.n_kv_heads;
    int head_dim = arguments.head_dim;
    float softmax_scale = arguments.softmax_scale;
    bool causal = arguments.causal;
    int window_size = arguments.window_size;
    int position_offset = arguments.position_offset;
    const llaminar2::attention::AttentionDeviceParams *device_params =
        arguments.device_params;
    const float *mask = arguments.mask;
    int tile_q = config.tile_q;
    int tile_kv = config.tile_kv;
    int qkv_pad = config.qkv_pad;
    int scores_pad = config.scores_pad;
    int head_start = arguments.head_start;
    int gqa_n_rep = arguments.gqa_n_rep;
    int context_partition_size = arguments.context_partition_size;
    int max_context_partitions = arguments.max_context_partitions;

    /*
     * The guarded root and IF body are graph siblings. The root therefore keeps
     * the immutable direct-partition threshold and retires uniformly when the
     * context body owns output. The context body is already selected by CUDA's
     * native IF and receives zero to keep its arithmetic unconditional.
     */
    int device_direct_partition_limit =
        placement == FA2AdaptivePhasePlacement::ConcurrentGuardedRoot
            ? arguments.device_direct_partition_limit
            : 0;
    void *kernel_arguments[] = {
        &Q,
        &K,
        &V,
        &O,
        &batch_size,
        &seq_len,
        &kv_capacity,
        &n_heads,
        &n_kv_heads,
        &head_dim,
        &softmax_scale,
        &causal,
        &window_size,
        &position_offset,
        &device_params,
        &mask,
        &tile_q,
        &tile_kv,
        &qkv_pad,
        &scores_pad,
        &head_start,
        &gqa_n_rep,
        &output_partial,
        &m_partial,
        &l_partial,
        &context_partition_size,
        &max_context_partitions,
        &device_direct_partition_limit,
        &explicit_context_partition,
    };
    const cudaKernelNodeParams node_params{
        .func = function,
        .gridDim = grid,
        .blockDim = dim3(config.block_size, 1, 1),
        .sharedMemBytes = static_cast<unsigned int>(config.smem_size),
        .kernelParams = kernel_arguments,
        .extra = nullptr,
    };
    if (placement == FA2AdaptivePhasePlacement::ConcurrentGuardedRoot)
    {
        return transaction.beginBranchesWithConcurrentRootKernel(
            node_params,
            semantic_name);
    }
    return transaction.appendKernel(
        llaminar2::CUDAActiveCaptureConditionalBranch::IfNonZero,
        node_params,
        semantic_name);
}

/**
 * @brief Append the deterministic ascending-partition reducer to one branch.
 * @tparam DIMENSION_WARPS_PER_ROW Independent output strips per query row.
 */
template <int DIMENSION_WARPS_PER_ROW>
static bool appendFA2ReducerConditionalNode(
    llaminar2::CUDAActiveCaptureConditional &transaction,
    llaminar2::CUDAActiveCaptureConditionalBranch branch,
    dim3 grid,
    const FA2ContextTransactionArguments &arguments)
{
    const float *O_partial = arguments.O_partial;
    const float *m_partial = arguments.m_partial;
    const float *l_partial = arguments.l_partial;
    float *O = arguments.O;
    int seq_len = arguments.seq_len;
    int n_heads = arguments.n_heads;
    int head_dim = arguments.head_dim;
    int max_context_partitions = arguments.max_context_partitions;
    int device_direct_partition_limit = 0;
    const llaminar2::attention::AttentionDeviceParams *device_params =
        arguments.device_params;
    void *kernel_arguments[] = {
        &O_partial,
        &m_partial,
        &l_partial,
        &O,
        &seq_len,
        &n_heads,
        &head_dim,
        &max_context_partitions,
        &device_direct_partition_limit,
        &device_params,
    };
    const cudaKernelNodeParams node_params{
        .func = reinterpret_cast<void *>(
            flash_attention_2_context_reduce_kernel<
                DIMENSION_WARPS_PER_ROW>),
        .gridDim = grid,
        .blockDim = dim3(256, 1, 1),
        .sharedMemBytes = 0,
        .kernelParams = kernel_arguments,
        .extra = nullptr,
    };
    return transaction.appendKernel(
        branch,
        node_params,
        "deterministic context reducer");
}
#endif

/**
 * @brief Launch one exact compiled FA2 transaction specialization.
 *
 * @tparam QW Widest query-warp grouping represented by the kernel body.
 * @tparam PVW P@V stripe warps assigned to each query group.
 * @tparam HD Compile-time head-dimension ceiling.
 * @tparam TKV Physical K/V shared-memory tile.
 * @tparam KV_FP16 True when K/V storage is native FP16.
 * @param arguments Immutable transaction tensors and logical geometry.
 * @param config Capture-time physical launch configuration.
 * @param schedule Sequence-node or context-grid physical scheduling. During
 *        adaptive capture, the lean query kernel is a root node whose device
 *        guard retires it for long prefixes; one IF-only child graph owns the
 *        context summary and reduction work. This avoids an ELSE-child dispatch
 *        on the latency-sensitive short-prefix path.
 * @param stream Exact non-null producer stream captured by the caller.
 * @return Zero on successful launch submission, otherwise a fatal launch error.
 */
template <int QW, int PVW, int HD, int TKV, bool KV_FP16>
static int launchFA2ContextTransactionSpecialization(
    const FA2ContextTransactionArguments &arguments,
    const FA2KernelConfig &config,
    FA2ContextPartitionSchedule schedule,
    cudaStream_t stream)
{
    auto context_phase =
        flash_attention_2_pipelined_kernel<QW, PVW, HD, TKV, KV_FP16, true>;
    const cudaError_t attribute_status = cudaFuncSetAttribute(
        context_phase,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        config.smem_size);
    if (attribute_status != cudaSuccess)
    {
        printf("[cudaFlashAttn_context_transaction] Failed to set phase-one "
               "dynamic shared memory: %s\n",
               cudaGetErrorString(attribute_status));
        return -1;
    }

    const int query_tiles =
        (arguments.seq_len + config.tile_q - 1) / config.tile_q;

    constexpr int kReducerThreads = 256;
    constexpr int kReducerWarps = kReducerThreads / WARP_SIZE;
    if (arguments.reducer_dimension_warps <= 0 ||
        arguments.reducer_dimension_warps > kReducerWarps ||
        kReducerWarps % arguments.reducer_dimension_warps != 0)
    {
        printf("[cudaFlashAttn_context_transaction] Invalid reducer dimension "
               "striping: %d\n",
               arguments.reducer_dimension_warps);
        return -1;
    }
    const int reducer_rows_per_block =
        kReducerWarps / arguments.reducer_dimension_warps;
    const dim3 reducer_grid(
        arguments.n_heads,
        (arguments.seq_len + reducer_rows_per_block - 1) /
            reducer_rows_per_block,
        arguments.batch_size);

    auto direct_phase =
        flash_attention_2_pipelined_kernel<
            QW, PVW, HD, TKV, KV_FP16, false>;
    if (arguments.device_direct_partition_limit > 0)
    {
        const cudaError_t direct_attribute_status = cudaFuncSetAttribute(
            direct_phase,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            config.smem_size);
        if (direct_attribute_status != cudaSuccess)
        {
            printf("[cudaFlashAttn_context_transaction] Failed to set adaptive "
                   "direct-node dynamic shared memory: %s\n",
                   cudaGetErrorString(direct_attribute_status));
            return -1;
        }
    }

#if CUDART_VERSION >= 12030
    if (schedule == FA2ContextPartitionSchedule::ContextParallelGrid &&
        arguments.device_direct_partition_limit > 0)
    {
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        const cudaError_t capture_query_status = cudaStreamIsCapturing(
            stream,
            &capture_status);
        if (capture_query_status != cudaSuccess)
        {
            printf("[cudaFlashAttn_context_transaction] Failed to query stream "
                   "capture state for adaptive transaction: %s\n",
                   cudaGetErrorString(capture_query_status));
            return -1;
        }

        if (capture_status == cudaStreamCaptureStatusActive)
        {
            auto *transaction = static_cast<
                llaminar2::CUDAActiveCaptureConditional *>(
                arguments.capture_conditional);
            if (!transaction)
            {
                printf("[cudaFlashAttn_context_transaction] Adaptive capture "
                       "requires a predicate published by the device-parameter "
                       "producer: missing transaction\n");
                return -1;
            }

            const dim3 direct_grid(
                arguments.n_heads,
                query_tiles,
                arguments.batch_size);
            if (!appendFA2AdaptivePhaseNode(
                    *transaction,
                    FA2AdaptivePhasePlacement::ConcurrentGuardedRoot,
                    reinterpret_cast<void *>(direct_phase),
                    direct_grid,
                    arguments,
                    config,
                    /*output_partial=*/nullptr,
                    /*m_partial=*/nullptr,
                    /*l_partial=*/nullptr,
                    /*explicit_context_partition=*/-1,
                    "guarded query-sequence root"))
            {
                printf("[cudaFlashAttn_context_transaction] Failed to fork "
                       "adaptive root and context work: %s\n",
                       transaction->error().c_str());
                return -1;
            }

            const dim3 context_grid(
                arguments.n_heads * arguments.context_partition_slots,
                query_tiles,
                arguments.batch_size);
            if (!appendFA2AdaptivePhaseNode(
                    *transaction,
                    FA2AdaptivePhasePlacement::ContextIfBody,
                    reinterpret_cast<void *>(context_phase),
                    context_grid,
                    arguments,
                    config,
                    arguments.O_partial,
                    arguments.m_partial,
                    arguments.l_partial,
                    /*explicit_context_partition=*/-1,
                    "K/V-context summary phase"))
            {
                printf("[cudaFlashAttn_context_transaction] Failed to append native "
                       "adaptive context branch: %s\n",
                       transaction->error().c_str());
                return -1;
            }

            bool reducer_appended = false;
            switch (arguments.reducer_dimension_warps)
            {
            case 1:
                reducer_appended = appendFA2ReducerConditionalNode<1>(
                    *transaction,
                    llaminar2::CUDAActiveCaptureConditionalBranch::IfNonZero,
                    reducer_grid,
                    arguments);
                break;
            case 2:
                reducer_appended = appendFA2ReducerConditionalNode<2>(
                    *transaction,
                    llaminar2::CUDAActiveCaptureConditionalBranch::IfNonZero,
                    reducer_grid,
                    arguments);
                break;
            case 4:
                reducer_appended = appendFA2ReducerConditionalNode<4>(
                    *transaction,
                    llaminar2::CUDAActiveCaptureConditionalBranch::IfNonZero,
                    reducer_grid,
                    arguments);
                break;
            case 8:
                reducer_appended = appendFA2ReducerConditionalNode<8>(
                    *transaction,
                    llaminar2::CUDAActiveCaptureConditionalBranch::IfNonZero,
                    reducer_grid,
                    arguments);
                break;
            }
            if (!reducer_appended || !transaction->commit())
            {
                printf("[cudaFlashAttn_context_transaction] Failed to complete native "
                       "adaptive context branch: %s\n",
                       transaction->error().c_str());
                return -1;
            }
            return 0;
        }
    }
#endif

    if (arguments.device_direct_partition_limit > 1)
    {
        direct_phase<<<
            dim3(arguments.n_heads, query_tiles, arguments.batch_size),
            config.block_size,
            config.smem_size,
            stream>>>(
            arguments.Q,
            arguments.K,
            arguments.V,
            arguments.O,
            arguments.batch_size,
            arguments.seq_len,
            arguments.kv_capacity,
            arguments.n_heads,
            arguments.n_kv_heads,
            arguments.head_dim,
            arguments.softmax_scale,
            arguments.causal,
            arguments.window_size,
            arguments.position_offset,
            arguments.device_params,
            arguments.mask,
            config.tile_q,
            config.tile_kv,
            config.qkv_pad,
            config.scores_pad,
            arguments.head_start,
            arguments.gqa_n_rep,
            /*O_partial=*/nullptr,
            /*m_partial=*/nullptr,
            /*l_partial=*/nullptr,
            arguments.context_partition_size,
            arguments.max_context_partitions,
            arguments.device_direct_partition_limit,
            /*explicit_context_partition=*/-1);
    }

    const auto launch_partition_grid =
        [&](dim3 grid, int explicit_partition)
    {
        context_phase<<<grid, config.block_size, config.smem_size, stream>>>(
            arguments.Q,
            arguments.K,
            arguments.V,
            arguments.O,
            arguments.batch_size,
            arguments.seq_len,
            arguments.kv_capacity,
            arguments.n_heads,
            arguments.n_kv_heads,
            arguments.head_dim,
            arguments.softmax_scale,
            arguments.causal,
            arguments.window_size,
            arguments.position_offset,
            arguments.device_params,
            arguments.mask,
            config.tile_q,
            config.tile_kv,
            config.qkv_pad,
            config.scores_pad,
            arguments.head_start,
            arguments.gqa_n_rep,
            arguments.O_partial,
            arguments.m_partial,
            arguments.l_partial,
            arguments.context_partition_size,
            arguments.max_context_partitions,
            arguments.device_direct_partition_limit,
            explicit_partition);
    };

    if (schedule == FA2ContextPartitionSchedule::ContextParallelGrid)
    {
        launch_partition_grid(
            dim3(
                arguments.n_heads * arguments.context_partition_slots,
                query_tiles,
                arguments.batch_size),
            /*explicit_partition=*/-1);
    }
    else
    {
        const dim3 partition_grid(
            arguments.n_heads,
            query_tiles,
            arguments.batch_size);
        for (int partition = 0;
             partition < arguments.max_context_partitions;
             ++partition)
        {
            launch_partition_grid(partition_grid, partition);
        }
    }

#define FA2_LAUNCH_CONTEXT_REDUCER(DIMENSION_WARPS)                    \
    flash_attention_2_context_reduce_kernel<DIMENSION_WARPS><<<       \
        reducer_grid, kReducerThreads, 0, stream>>>(                   \
        arguments.O_partial, arguments.m_partial, arguments.l_partial, \
        arguments.O, arguments.seq_len, arguments.n_heads,            \
        arguments.head_dim, arguments.max_context_partitions,          \
        arguments.device_direct_partition_limit,                       \
        arguments.device_params)

    switch (arguments.reducer_dimension_warps)
    {
    case 1:
        FA2_LAUNCH_CONTEXT_REDUCER(1);
        break;
    case 2:
        FA2_LAUNCH_CONTEXT_REDUCER(2);
        break;
    case 4:
        FA2_LAUNCH_CONTEXT_REDUCER(4);
        break;
    case 8:
        FA2_LAUNCH_CONTEXT_REDUCER(8);
        break;
    default:
        printf("[cudaFlashAttn_context_transaction] Reducer dimension "
               "striping escaped validated dispatch\n");
        return -1;
    }

#undef FA2_LAUNCH_CONTEXT_REDUCER

    const cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess)
    {
        printf("[cudaFlashAttn_context_transaction] CUDA launch failed: %s "
               "(schedule=%d, partitions=%d, partition_size=%d, grid_q=%d, "
               "heads=%d, head_dim=%d, reducer_dimension_warps=%d)\n",
               cudaGetErrorString(launch_status),
               static_cast<int>(schedule),
               arguments.max_context_partitions,
               arguments.context_partition_size,
               query_tiles,
               arguments.n_heads,
               arguments.head_dim,
               arguments.reducer_dimension_warps);
        return -1;
    }
    return 0;
}

/**
 * @brief Validate and dispatch one fully device-resident FA2 context transaction.
 *
 * Production selection resolves the immutable geometry before calling this
 * bridge. The bridge does not provide a fallback: every invalid pointer,
 * stream, head mapping, partition envelope, adaptive schedule, or unsupported
 * specialization fails closed.
 *
 * @tparam KV_FP16 True for native FP16 K/V storage.
 */
template <bool KV_FP16>
static int fa2_context_transaction_launch(
    const float *Q,
    const void *K,
    const void *V,
    float *O,
    float *O_partial,
    float *m_partial,
    float *l_partial,
    int batch_size,
    int seq_len,
    int kv_capacity,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    bool causal,
    int window_size,
    int position_offset,
    const llaminar2::attention::AttentionDeviceParams *device_params,
    const float *mask,
    int context_partition_size,
    int max_context_partitions,
    int context_partition_slots,
    int device_direct_partition_limit,
    int reducer_dimension_warps,
    void *capture_conditional,
    FA2ContextPartitionSchedule schedule,
    cudaStream_t stream,
    int device_idx,
    int head_start,
    int gqa_n_rep)
{
    if (!Q || !K || !V || !O || !O_partial || !m_partial || !l_partial ||
        !stream || batch_size <= 0 || seq_len <= 0 || kv_capacity <= 0 ||
        n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 ||
        head_dim % WMMA_K != 0 || head_dim > 256 ||
        context_partition_size !=
            llaminar2::cuda::fa2_policy::
                kFA2CanonicalContextPartitionKeys ||
        max_context_partitions <= 0 ||
        max_context_partitions !=
            (kv_capacity + context_partition_size - 1) /
                context_partition_size ||
        context_partition_slots <= 0 ||
        context_partition_slots > max_context_partitions ||
        device_direct_partition_limit < 0 ||
        device_direct_partition_limit > max_context_partitions ||
        (device_direct_partition_limit > 0 &&
         (!device_params ||
          schedule != FA2ContextPartitionSchedule::ContextParallelGrid)))
    {
        printf("[cudaFlashAttn_context_transaction] Invalid tensor, stream, or "
               "partition geometry\n");
        return -1;
    }

    const llaminar2::cuda::fa2_policy::FA2HeadMappingGeometry head_mapping{
        .local_query_heads = n_heads,
        .visible_kv_heads = n_kv_heads,
        .head_start = head_start,
        .replicated_gqa_n_rep = gqa_n_rep,
    };
    if (!llaminar2::cuda::fa2_policy::isValidFA2HeadMapping(head_mapping))
    {
        printf("[cudaFlashAttn_context_transaction] Invalid GQA head mapping\n");
        return -1;
    }

    if (cudaSetDevice(device_idx) != cudaSuccess)
    {
        printf("[cudaFlashAttn_context_transaction] Failed to select device %d\n",
               device_idx);
        return -1;
    }
    const FA2DeviceConfig &device = getFA2DeviceConfig(device_idx);
    if (device.sm_major < 8)
    {
        printf("[cudaFlashAttn_context_transaction] SM %d.%d is unsupported\n",
               device.sm_major,
               device.sm_minor);
        return -2;
    }

    const llaminar2::cuda::fa2_policy::FA2QueryPartitionGeometry
        query_geometry{
            .batch_size = batch_size,
            .query_rows = seq_len,
            .local_query_heads = n_heads,
            .head_dim = head_dim,
            .sm_count = device.sm_count,
        };
    const int query_warp_groups =
        llaminar2::cuda::fa2_policy::selectFA2QueryWarpGroups(
            query_geometry,
            llaminar2::debugEnv().attention.cuda_fa2_q_warp_groups);
    const int selected_reducer_dimension_warps =
        llaminar2::cuda::fa2_policy::selectFA2ReducerDimensionWarps(
            query_geometry,
            reducer_dimension_warps);
    const FA2KernelConfig config = computeFA2Config(
        head_dim,
        device.max_smem_optin,
        query_warp_groups);
    if (query_warp_groups <= 0 || config.tile_q <= 0 ||
        config.tile_kv <= 0 || config.num_q_warp_groups <= 0 ||
        config.pv_warps_per_q_group <= 0 || config.block_size <= 0 ||
        config.smem_size == 0 || selected_reducer_dimension_warps <= 0)
    {
        printf("[cudaFlashAttn_context_transaction] No valid compiled launch "
               "geometry\n");
        return -1;
    }

    const FA2ContextTransactionArguments arguments{
        .Q = Q,
        .K = K,
        .V = V,
        .O = O,
        .O_partial = O_partial,
        .m_partial = m_partial,
        .l_partial = l_partial,
        .batch_size = batch_size,
        .seq_len = seq_len,
        .kv_capacity = kv_capacity,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .head_dim = head_dim,
        .softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim)),
        .causal = causal,
        .window_size = window_size,
        .position_offset = position_offset,
        .device_params = device_params,
        .mask = mask,
        .head_start = head_start,
        .gqa_n_rep = gqa_n_rep,
        .context_partition_size = context_partition_size,
        .max_context_partitions = max_context_partitions,
        .context_partition_slots = context_partition_slots,
        .device_direct_partition_limit = device_direct_partition_limit,
        .reducer_dimension_warps = selected_reducer_dimension_warps,
        .capture_conditional = capture_conditional,
    };

#define FA2_CONTEXT_DISPATCH(QW, PVW, HD, TKV)                              \
    return launchFA2ContextTransactionSpecialization<                       \
        QW, PVW, HD, TKV, KV_FP16>(arguments, config, schedule, stream)

#define FA2_CONTEXT_DISPATCH_TILE(QW, PVW, HD) \
    do                                          \
    {                                           \
        if (config.tile_kv == 16)               \
            FA2_CONTEXT_DISPATCH(QW, PVW, HD, 16); \
        if (config.tile_kv == 32)               \
            FA2_CONTEXT_DISPATCH(QW, PVW, HD, 32); \
        if (config.tile_kv == 64)               \
            FA2_CONTEXT_DISPATCH(QW, PVW, HD, 64); \
    } while (0)

    if (head_dim <= 64)
        FA2_CONTEXT_DISPATCH_TILE(6, 1, 64);
    if (head_dim <= 128)
        FA2_CONTEXT_DISPATCH_TILE(4, 1, 128);
    if (config.pv_warps_per_q_group == 1)
        FA2_CONTEXT_DISPATCH_TILE(2, 1, 256);
    if (config.pv_warps_per_q_group == 2)
        FA2_CONTEXT_DISPATCH_TILE(2, 2, 256);
    if (config.pv_warps_per_q_group == 4)
        FA2_CONTEXT_DISPATCH_TILE(2, 4, 256);

#undef FA2_CONTEXT_DISPATCH_TILE
#undef FA2_CONTEXT_DISPATCH

    printf("[cudaFlashAttn_context_transaction] Dispatch escaped a validated "
           "specialization\n");
    return -1;
}

// =============================================================================
// Extern "C" Wrapper Functions
// =============================================================================

extern "C"
{
    int cudaFlashAttn_prefill_fa2(
        const float *Q, const float *K, const float *V, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask,
        void *stream,
        int device_idx,
        int head_start,
        int gqa_n_rep)
    {
        return fa2_prefill_launch<false>(
            Q, static_cast<const void *>(K), static_cast<const void *>(V), O,
            batch_size, seq_len, kv_len, n_heads, n_kv_heads, head_dim,
            causal, window_size, position_offset, device_params, mask,
            static_cast<cudaStream_t>(stream), device_idx,
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch FA2 with FP16 K/V inputs (eliminates FP16→FP32→FP16 round-trip)
     *
     * When the KV cache is already FP16, this function passes the half pointers
     * directly to the kernel. The kernel loads K/V as half and writes to shared
     * memory without any FP32 intermediate conversion, saving ~2× bandwidth.
     */
    int cudaFlashAttn_prefill_fa2_fp16kv(
        const float *Q, const void *K_fp16, const void *V_fp16, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask,
        void *stream,
        int device_idx,
        int head_start,
        int gqa_n_rep)
    {
        return fa2_prefill_launch<true>(
            Q, K_fp16, V_fp16, O,
            batch_size, seq_len, kv_len, n_heads, n_kv_heads, head_dim,
            causal, window_size, position_offset, device_params, mask,
            static_cast<cudaStream_t>(stream), device_idx,
            head_start, gqa_n_rep);
    }

    /**
     * @brief Launch canonical fixed-context FA2 as ordered captured graph nodes.
     *
     * One phase-one node is recorded for each contiguous logical K/V partition,
     * followed by the deterministic device reducer.  This is the arithmetic
     * reference for context-parallel scheduling: it uses the same phase-one
     * specialization and summary layout, but exposes no cross-partition grid
     * concurrency.  All workspace is caller-owned and persistent.
     */
    int cudaFlashAttn_prefill_fa2_fp16kv_partitioned_sequence(
        const float *Q,
        const void *K_fp16,
        const void *V_fp16,
        float *O,
        float *O_partial,
        float *m_partial,
        float *l_partial,
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
        int context_partition_size,
        int max_context_partitions,
        int context_partition_slots,
        int device_direct_partition_limit,
        int reducer_dimension_warps,
        void *capture_conditional,
        void *stream,
        int device_idx,
        int head_start,
        int gqa_n_rep)
    {
        return fa2_context_transaction_launch<true>(
            Q,
            K_fp16,
            V_fp16,
            O,
            O_partial,
            m_partial,
            l_partial,
            batch_size,
            seq_len,
            kv_len,
            n_heads,
            n_kv_heads,
            head_dim,
            causal,
            window_size,
            position_offset,
            device_params,
            mask,
            context_partition_size,
            max_context_partitions,
            context_partition_slots,
            device_direct_partition_limit,
            reducer_dimension_warps,
            capture_conditional,
            FA2ContextPartitionSchedule::SequenceGraphNodes,
            static_cast<cudaStream_t>(stream),
            device_idx,
            head_start,
            gqa_n_rep);
    }

    /**
     * @brief Launch canonical fixed-context FA2 with FP32 K/V storage.
     *
     * The arithmetic and persistent summary layout are identical to the FP16
     * entrypoint below. Only the producer's global-memory load/conversion type
     * differs, allowing native FP32 and explicitly converted cache views to use
     * the same capture-stable context graph without an alternate execution path.
     */
    int cudaFlashAttn_prefill_fa2_context_parallel(
        const float *Q,
        const float *K,
        const float *V,
        float *O,
        float *O_partial,
        float *m_partial,
        float *l_partial,
        int batch_size,
        int seq_len,
        int kv_capacity,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        bool causal,
        int window_size,
        int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask,
        int context_partition_size,
        int max_context_partitions,
        int context_partition_slots,
        int device_direct_partition_limit,
        int reducer_dimension_warps,
        void *capture_conditional,
        void *stream,
        int device_idx,
        int head_start,
        int gqa_n_rep)
    {
        return fa2_context_transaction_launch<false>(
            Q,
            K,
            V,
            O,
            O_partial,
            m_partial,
            l_partial,
            batch_size,
            seq_len,
            kv_capacity,
            n_heads,
            n_kv_heads,
            head_dim,
            causal,
            window_size,
            position_offset,
            device_params,
            mask,
            context_partition_size,
            max_context_partitions,
            context_partition_slots,
            device_direct_partition_limit,
            reducer_dimension_warps,
            capture_conditional,
            FA2ContextPartitionSchedule::ContextParallelGrid,
            static_cast<cudaStream_t>(stream),
            device_idx,
            head_start,
            gqa_n_rep);
    }

    /**
     * @brief Launch canonical fixed-context FA2 with a parallel partition grid.
     *
     * Every context partition executes concurrently in one phase-one graph node;
     * the second node merges summaries in ascending logical order entirely on
     * device.  The output must be byte-identical to
     * `cudaFlashAttn_prefill_fa2_fp16kv_partitioned_sequence` before this mode
     * can be admitted to production dispatch.
     */
    int cudaFlashAttn_prefill_fa2_fp16kv_context_parallel(
        const float *Q,
        const void *K_fp16,
        const void *V_fp16,
        float *O,
        float *O_partial,
        float *m_partial,
        float *l_partial,
        int batch_size,
        int seq_len,
        int kv_capacity,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        bool causal,
        int window_size,
        int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask,
        int context_partition_size,
        int max_context_partitions,
        int context_partition_slots,
        int device_direct_partition_limit,
        int reducer_dimension_warps,
        void *capture_conditional,
        void *stream,
        int device_idx,
        int head_start,
        int gqa_n_rep)
    {
        return fa2_context_transaction_launch<true>(
            Q,
            K_fp16,
            V_fp16,
            O,
            O_partial,
            m_partial,
            l_partial,
            batch_size,
            seq_len,
            kv_capacity,
            n_heads,
            n_kv_heads,
            head_dim,
            causal,
            window_size,
            position_offset,
            device_params,
            mask,
            context_partition_size,
            max_context_partitions,
            context_partition_slots,
            device_direct_partition_limit,
            reducer_dimension_warps,
            capture_conditional,
            FA2ContextPartitionSchedule::ContextParallelGrid,
            static_cast<cudaStream_t>(stream),
            device_idx,
            head_start,
            gqa_n_rep);
    }

    /**
     * @brief Launch Flash Decoding kernel (FP32)
     *
     * Workspace layout:
     * - O_partial: [batch, heads, splits, head_dim] - unnormalized weighted sums
     * - m_partial: [batch, heads, splits] - max scores
     * - l_partial: [batch, heads, splits] - sum of exp(score - m)
     */
    int cudaFlashAttn_decode_fp32(
        const float *Q, const float *K_cache, const float *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int batch_size, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int device_idx,
        int head_start,
        int gqa_n_rep)
    {
        // Ensure correct device is active for kernel launch and any implicit allocations
        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

        float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

        // Phase 1: Compute partial attention per split
        {
            dim3 grid(n_heads, num_splits, batch_size);
            int block_size = 256;

            size_t smem_size = head_dim * sizeof(float); // Q_shared

            flash_decoding_fp32_kernel<<<grid, block_size, smem_size, cuda_stream>>>(
                Q, K_cache, V_cache,
                O_partial, m_partial, l_partial,
                kv_len, n_heads, n_kv_heads, head_dim,
                num_splits, softmax_scale, device_params,
                head_start, gqa_n_rep);
        }

        // Phase 2: Reduce partials to final output
        {
            const int output_tiles = head_dim >= 256 ? 2 : 1;
            dim3 grid(n_heads, batch_size, output_tiles);
            const int block_size =
                min((head_dim + output_tiles - 1) / output_tiles, 256);

            flash_decoding_reduce_fp32_kernel<<<grid, block_size, 0, cuda_stream>>>(
                O_partial, m_partial, l_partial, O,
                n_heads, head_dim, num_splits,
                kv_len,
                /*min_kv_per_split=*/16,
                device_params);
        }

        return cudaGetLastError() == cudaSuccess ? 0 : -1;
    }

    /**
     * @brief Launch Flash Decoding kernel with FP16 KV cache (no conversion)
     *
     * Same workspace layout as FP32 variant. K/V are read directly as FP16
     * and converted to FP32 in registers, eliminating the separate conversion
     * kernels and halving KV cache bandwidth.
     */
    int cudaFlashAttn_decode_fp16kv(
        const float *Q, const void *K_cache_fp16, const void *V_cache_fp16, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int batch_size, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int device_idx,
        int head_start,
        int gqa_n_rep)
    {
        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

        float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

        // Phase 1: Compute partial attention per split (FP16 KV)
        {
            dim3 grid(n_heads, num_splits, batch_size);
            int block_size = 256;
            size_t smem_size = head_dim * sizeof(float);

            flash_decoding_fp16kv_kernel<FlashDecodeRowMode::OrdinaryBatch>
                <<<grid, block_size, smem_size, cuda_stream>>>(
                Q,
                static_cast<const half *>(K_cache_fp16),
                static_cast<const half *>(V_cache_fp16),
                O_partial, m_partial, l_partial,
                kv_len, n_heads, n_kv_heads, head_dim,
                num_splits, softmax_scale, device_params,
                /*query_rows_per_request=*/1,
                head_start, gqa_n_rep);
        }

        // Phase 2: Reduce partials (same as FP32 — operates on FP32 partials)
        {
            const int output_tiles = head_dim >= 256 ? 2 : 1;
            dim3 grid(n_heads, batch_size, output_tiles);
            const int block_size =
                min((head_dim + output_tiles - 1) / output_tiles, 256);

            flash_decoding_reduce_fp32_kernel<<<grid, block_size, 0, cuda_stream>>>(
                O_partial, m_partial, l_partial, O,
                n_heads, head_dim, num_splits,
                kv_len,
                /*min_kv_per_split=*/16,
                device_params);
        }

        return cudaGetLastError() == cudaSuccess ? 0 : -1;
    }

    /**
     * @brief Launch an economical, serial-equivalent grouped FP16-KV verifier.
     *
     * This is not an ordinary attention batch. Q/output have one row per MTP
     * verifier position, while K/V name one shared cache whose visible prefix
     * grows by one position per row. A fixed maximum-split grid keeps the launch
     * graph-capturable; row-local device metadata masks surplus split blocks and
     * reproduces the exact scalar split partition.
     */
    int cudaFlashAttn_decode_fp16kv_grouped_verifier_rows(
        const float *Q,
        const void *K_cache_fp16,
        const void *V_cache_fp16,
        float *O,
        float *O_partial,
        float *m_partial,
        float *l_partial,
        int verifier_rows,
        int max_kv_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int max_num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int device_idx,
        int head_start,
        int gqa_n_rep)
    {
        if (!Q || !K_cache_fp16 || !V_cache_fp16 || !O ||
            !O_partial || !m_partial || !l_partial || !device_params ||
            !stream || verifier_rows < 2 ||
            verifier_rows >
                llaminar2::attention::kMaxGroupedVerifierAttentionRows ||
            max_kv_len <= verifier_rows || n_heads <= 0 ||
            n_kv_heads <= 0 || head_dim <= 0 ||
            max_num_splits <= 0 || max_num_splits > 32)
        {
            return -1;
        }

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        const float softmax_scale =
            1.0f / sqrtf(static_cast<float>(head_dim));

        /*
         * Phase 1 uses one z-plane per verifier row. All rows share K/V, but
         * each active block reads that row's logical length and scalar-equivalent
         * split count from stable device metadata.
         */
        {
            const dim3 grid(n_heads, max_num_splits, verifier_rows);
            constexpr int block_size = 256;
            const size_t smem_size =
                static_cast<size_t>(head_dim) * sizeof(float);
            flash_decoding_fp16kv_kernel<FlashDecodeRowMode::SharedKVVerifier>
                <<<grid, block_size, smem_size, cuda_stream>>>(
                    Q,
                    static_cast<const half *>(K_cache_fp16),
                    static_cast<const half *>(V_cache_fp16),
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
                    /*query_rows_per_request=*/1,
                    head_start,
                    gqa_n_rep);
        }

        /* Phase 2 merges each row's active split prefix in serial order. */
        {
            const int output_tiles = head_dim >= 256 ? 2 : 1;
            const dim3 grid(n_heads, verifier_rows, output_tiles);
            const int block_size =
                min((head_dim + output_tiles - 1) / output_tiles, 256);
            flash_decoding_grouped_row_reduce_fp32_kernel
                <<<grid, block_size, 0, cuda_stream>>>(
                    O_partial,
                    m_partial,
                    l_partial,
                    O,
                    n_heads,
                    head_dim,
                    max_num_splits,
                    device_params);
        }

        return cudaGetLastError() == cudaSuccess ? 0 : -1;
    }

    /**
     * @brief Launch serial-equivalent FP16 attention for independent requests.
     *
     * K/V use request-major fixed strides of `max_kv_len`.  The z dimension
     * covers every request/query row, while device parameters select each row's
     * logical prefix and scalar-equivalent split count.  This keeps the launch
     * count constant at two regardless of request count.
     */
    int cudaFlashAttn_decode_fp16kv_grouped_request_rows(
        const float *Q,
        const void *K_cache_fp16,
        const void *V_cache_fp16,
        float *O,
        float *O_partial,
        float *m_partial,
        float *l_partial,
        int request_count,
        int query_rows,
        int max_kv_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int max_num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int device_idx,
        int head_start,
        int gqa_n_rep)
    {
        const int total_rows = request_count * query_rows;
        if (!Q || !K_cache_fp16 || !V_cache_fp16 || !O ||
            !O_partial || !m_partial || !l_partial || !device_params ||
            !stream || request_count < 2 || query_rows <= 0 ||
            total_rows >
                llaminar2::attention::kMaxGroupedVerifierAttentionRows ||
            max_kv_len <= 0 || n_heads <= 0 ||
            n_kv_heads <= 0 || head_dim <= 0 ||
            max_num_splits <= 0 || max_num_splits > 32)
        {
            return -1;
        }

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        const float softmax_scale =
            1.0f / sqrtf(static_cast<float>(head_dim));

        {
            const dim3 grid(n_heads, max_num_splits, total_rows);
            constexpr int block_size = 256;
            const size_t smem_size =
                static_cast<size_t>(head_dim) * sizeof(float);
            flash_decoding_fp16kv_kernel<FlashDecodeRowMode::IndependentRequest>
                <<<grid, block_size, smem_size, cuda_stream>>>(
                    Q,
                    static_cast<const half *>(K_cache_fp16),
                    static_cast<const half *>(V_cache_fp16),
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
                    query_rows,
                    head_start,
                    gqa_n_rep);
        }

        {
            const int output_tiles = head_dim >= 256 ? 2 : 1;
            const dim3 grid(n_heads, total_rows, output_tiles);
            const int block_size =
                min((head_dim + output_tiles - 1) / output_tiles, 256);
            flash_decoding_grouped_row_reduce_fp32_kernel
                <<<grid, block_size, 0, cuda_stream>>>(
                    O_partial,
                    m_partial,
                    l_partial,
                    O,
                    n_heads,
                    head_dim,
                    max_num_splits,
                    device_params);
        }

        return cudaGetLastError() == cudaSuccess ? 0 : -1;
    }

    /**
     * @brief Launch Flash Decoding kernel with Q8_1 KV cache (fused inline dequant)
     *
     * Same workspace layout as FP32/FP16 variants. K/V are read directly as
     * Q8_1 blocks and dequantized inline in the attention inner loop,
     * eliminating the separate dequant kernel and FP32 workspace buffers.
     */
    int cudaFlashAttn_decode_q8_1(
        const float *Q, const void *K_cache_q8, const void *V_cache_q8, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int batch_size, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int device_idx,
        int head_start,
        int gqa_n_rep)
    {
        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

        float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

        // Phase 1: Compute partial attention per split (Q8_1 KV, fused dequant)
        {
            dim3 grid(n_heads, num_splits, batch_size);
            int block_size = 256;
            size_t smem_size = head_dim * sizeof(float);

            flash_decoding_q8kv_kernel<<<grid, block_size, smem_size, cuda_stream>>>(
                Q, K_cache_q8, V_cache_q8,
                O_partial, m_partial, l_partial,
                kv_len, n_heads, n_kv_heads, head_dim,
                num_splits, softmax_scale, device_params,
                head_start, gqa_n_rep);
        }

        // Phase 2: Reduce partials (same as FP32 — operates on FP32 partials)
        {
            const int output_tiles = head_dim >= 256 ? 2 : 1;
            dim3 grid(n_heads, batch_size, output_tiles);
            const int block_size =
                min((head_dim + output_tiles - 1) / output_tiles, 256);

            flash_decoding_reduce_fp32_kernel<<<grid, block_size, 0, cuda_stream>>>(
                O_partial, m_partial, l_partial, O,
                n_heads, head_dim, num_splits,
                kv_len,
                /*min_kv_per_split=*/16,
                device_params);
        }

        return cudaGetLastError() == cudaSuccess ? 0 : -1;
    }

    /**
     * @brief Flash Decoding with TQ8 K / TQ4 V — fused rotation + centroid attention
     *
     * Uses rotation trick to avoid O(D²) per-position dequantization.
     * Pre-rotates Q once per head, then does O(D) centroid lookup per K/V position.
     */
    int cudaFlashAttn_decode_tqkv(
        const float *Q, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        const void *K_cache, const void *V_cache,
        const float *rotation, const float *rotation_t,
        int batch_size, int kv_count,
        int n_heads, int n_kv_heads, int head_dim,
        int num_splits,
        int max_seq_len, int tail,
        int k_block_size, int v_block_size,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int device_idx,
        int head_start,
        int gqa_n_rep)
    {
        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

        // Ensure TQ codebooks are in constant memory
        upload_tq_attn_codebooks();

        float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

        // Phase 1: Fused TQ attention with rotation trick
        {
            dim3 grid(n_heads, num_splits, batch_size);
            int block_size = 256;
            // Shared memory: Q[D] + Q_rot[D] (used during Q rotation),
            // then reused for block_m[8] + block_l[8] + block_O[8*D] + warp_scales[8]
            size_t smem_size = static_cast<size_t>(head_dim) * 2 * sizeof(float); // Q + Q_rot
            size_t reduce_smem = (8 + 8) * sizeof(float) + 8 * head_dim * sizeof(float) + 8 * sizeof(float);
            smem_size = max(smem_size, reduce_smem);

            flash_decoding_tqkv_kernel<<<grid, block_size, smem_size, cuda_stream>>>(
                Q, K_cache, V_cache,
                rotation, rotation_t,
                O_partial, m_partial, l_partial,
                kv_count, n_heads, n_kv_heads, head_dim,
                num_splits, softmax_scale,
                max_seq_len, tail,
                k_block_size, v_block_size,
                device_params,
                head_start, gqa_n_rep);
        }

        // Phase 2: Reduce partials (same reduce kernel as FP32/Q8_1)
        {
            const int output_tiles = head_dim >= 256 ? 2 : 1;
            dim3 grid(n_heads, batch_size, output_tiles);
            const int block_size =
                min((head_dim + output_tiles - 1) / output_tiles, 256);

            flash_decoding_reduce_fp32_kernel<<<grid, block_size, 0, cuda_stream>>>(
                O_partial, m_partial, l_partial, O,
                n_heads, head_dim, num_splits,
                kv_count,
                /*min_kv_per_split=*/64,
                device_params);
        }

        return cudaGetLastError() == cudaSuccess ? 0 : -1;
    }

    /**
     * @brief Derive AttentionDeviceParams from device-owned KV cache count.
     *
     * When @p prefill_branch_condition is non-zero, this same producer also
     * publishes the native graph branch after deriving the live K/V length.
     * The condition and its key threshold must therefore be supplied together.
     */
    int cudaFlashAttn_prepare_device_params_from_count(
        void *device_params,
        const int *post_append_cached_tokens,
        int seq_len,
        int query_rows,
        int kv_stride,
        const int *active_query_rows_device,
        const int *ring_head_device,
        int ring_capacity,
        unsigned long long prefill_branch_condition,
        int direct_kv_limit,
        void *stream)
    {
        if (!device_params || !post_append_cached_tokens || seq_len <= 0 ||
            query_rows <= 0 || kv_stride <= 0 ||
            query_rows > MAX_DYNAMIC_ATTENTION_PARAM_ROWS || !stream ||
            ((ring_head_device == nullptr) != (ring_capacity == 0)) ||
            (ring_capacity > 0 && ring_capacity != kv_stride) ||
            ((prefill_branch_condition == 0) != (direct_kv_limit == 0)))
        {
            return -1;
        }
#if CUDART_VERSION < 12030
        if (prefill_branch_condition != 0)
            return -1;
#endif

        cuda_derive_attention_params_from_cached_tokens_kernel<<<1, query_rows, 0,
                                                                 static_cast<cudaStream_t>(stream)>>>(
            static_cast<llaminar2::attention::AttentionDeviceParams *>(device_params),
            post_append_cached_tokens,
            seq_len,
            query_rows,
            kv_stride,
            active_query_rows_device,
            ring_head_device,
            ring_capacity,
            prefill_branch_condition,
            direct_kv_limit);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("[cudaFlashAttn_prepare_device_params_from_count] Kernel launch failed: %s\n",
                   cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    /**
     * @brief Write explicit decode geometry to the device-owned param block.
     *
     * Isolated graph tests may also use this producer to publish an adaptive
     * prefill branch. Production cache-backed graphs use the count-derived
     * writer above so replay remains device-owned.
     */
    int cudaFlashAttn_prepare_device_params_from_geometry(
        void *device_params,
        int kv_len,
        int kv_stride,
        int position_offset,
        int query_rows,
        unsigned long long prefill_branch_condition,
        int direct_kv_limit,
        void *stream)
    {
        if (!device_params || kv_len <= 0 || kv_stride < kv_len ||
            position_offset < 0 ||
            query_rows <= 0 ||
            query_rows > MAX_DYNAMIC_ATTENTION_PARAM_ROWS || !stream ||
            ((prefill_branch_condition == 0) != (direct_kv_limit == 0)))
        {
            return -1;
        }
#if CUDART_VERSION < 12030
        if (prefill_branch_condition != 0)
            return -1;
#endif

        cuda_write_attention_params_from_geometry_kernel
            <<<1, query_rows, 0, static_cast<cudaStream_t>(stream)>>>(
                static_cast<llaminar2::attention::AttentionDeviceParams *>(device_params),
                kv_len,
                kv_stride,
                position_offset,
                query_rows,
                prefill_branch_condition,
                direct_kv_limit);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("[cudaFlashAttn_prepare_device_params_from_geometry] "
                   "Kernel launch failed: %s\n",
                   cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    /**
     * @brief Derive grouped request-row params from contiguous device counts.
     */
    int cudaFlashAttn_prepare_device_params_from_request_counts(
        void *device_params,
        const int *post_append_cached_tokens,
        int request_count,
        int query_rows,
        int kv_stride,
        void *stream)
    {
        const int total_rows = request_count * query_rows;
        if (!device_params || !post_append_cached_tokens ||
            request_count < 2 || query_rows <= 0 ||
            total_rows >
                llaminar2::attention::kMaxGroupedVerifierAttentionRows ||
            kv_stride <= 0 || !stream)
        {
            return -1;
        }

        cuda_derive_attention_params_from_request_counts_kernel
            <<<1, total_rows, 0, static_cast<cudaStream_t>(stream)>>>(
                static_cast<llaminar2::attention::AttentionDeviceParams *>(device_params),
                post_append_cached_tokens,
                request_count,
                query_rows,
                kv_stride);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("[cudaFlashAttn_prepare_device_params_from_request_counts] "
                   "Kernel launch failed: %s\n",
                   cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    /**
     * @brief Set CUDA device
     */
    int cudaFlashAttn_setDevice(int device_idx)
    {
        return cudaSetDevice(device_idx) == cudaSuccess ? 0 : -1;
    }

    /**
     * @brief Synchronize device
     */
    int cudaFlashAttn_synchronize()
    {
        return cudaDeviceSynchronize() == cudaSuccess ? 0 : -1;
    }

    /**
     * @brief Convert FP16 KV cache data to FP32 on GPU
     *
     * Replaces the previous CPU roundtrip (D2H → fp16_to_fp32 loop → H2D).
     *
     * @param src      Device pointer to FP16 (uint16_t) source data
     * @param dst      Device pointer to FP32 destination buffer
     * @param count    Number of elements to convert
     * @param stream   CUDA stream (nullptr for default stream)
     * @return 0 on success, -1 on error
     */
    int cudaFlashAttn_convert_fp16_to_fp32(const void *src, float *dst, int count, void *stream)
    {
        if (!src || !dst || count <= 0)
            return -1;

        const int threads = 256;
        const int blocks = (count + threads - 1) / threads;
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

        convert_fp16_to_fp32_kernel<<<blocks, threads, 0, cuda_stream>>>(
            static_cast<const uint16_t *>(src), dst, count);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("[cudaFlashAttn_convert_fp16_to_fp32] Kernel launch failed: %s\n",
                   cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    /**
     * @brief Dequantize Q8_1 KV cache data to FP32 on GPU
     *
     * Replaces the previous CPU roundtrip (D2H → Q8_1 dequant loop → H2D).
     *
     * @param src      Device pointer to Q8_1 blocks
     * @param dst      Device pointer to FP32 destination buffer
     * @param rows     Number of rows (batch_size * kv_len)
     * @param cols     Number of columns (n_kv_heads * head_dim)
     * @param stream   CUDA stream (nullptr for default stream)
     * @return 0 on success, -1 on error
     */
    int cudaFlashAttn_dequant_q8_1_to_fp32(const void *src, float *dst,
                                           int rows, int cols, void *stream)
    {
        if (!src || !dst || rows <= 0 || cols <= 0)
            return -1;

        const int blocks_per_row = (cols + 31) / 32;
        const int total_blocks = rows * blocks_per_row;
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

        // One CUDA block per Q8_1 block, 32 threads per block (one per element)
        dequant_q8_1_to_fp32_kernel<<<total_blocks, 32, 0, cuda_stream>>>(
            static_cast<const GpuQ8_1Block *>(src), dst,
            rows, cols, blocks_per_row);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("[cudaFlashAttn_dequant_q8_1_to_fp32] Kernel launch failed: %s\n",
                   cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    /**
     * @brief Dynamic FP16→FP32 conversion for CUDA graph replay
     *
     * Uses a kernel that reads kv_len from device_params at runtime,
     * so the captured graph correctly handles growing kv_len during replay.
     * Grid is oversized for max_kv_len; excess threads return early.
     *
     * @param src           Device pointer to FP16 source data
     * @param dst           Device pointer to FP32 destination buffer
     * @param cols_per_row  n_kv_heads * head_dim (constant)
     * @param max_kv_len    Maximum kv_len for grid sizing (workspace capacity)
     * @param device_params Device pointer to AttentionDeviceParams (has dynamic kv_len)
     * @param stream        CUDA stream
     * @return 0 on success, -1 on error
     */
    int cudaFlashAttn_convert_fp16_to_fp32_dynamic(
        const void *src, float *dst,
        int cols_per_row, int max_kv_len,
        const void *device_params, void *stream)
    {
        if (!src || !dst || !device_params || cols_per_row <= 0 || max_kv_len <= 0)
            return -1;

        const int max_count = max_kv_len * cols_per_row;
        const int threads = 256;
        const int blocks = (max_count + threads - 1) / threads;
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

        convert_fp16_to_fp32_dynamic_kernel<<<blocks, threads, 0, cuda_stream>>>(
            static_cast<const uint16_t *>(src), dst,
            cols_per_row,
            static_cast<const llaminar2::attention::AttentionDeviceParams *>(device_params));

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("[cudaFlashAttn_convert_fp16_to_fp32_dynamic] Kernel launch failed: %s\n",
                   cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

    /**
     * @brief Dynamic Q8_1→FP32 dequantization for CUDA graph replay
     *
     * Uses a kernel that reads kv_len (row count) from device_params at runtime,
     * so the captured graph correctly handles growing kv_len during replay.
     * Grid is oversized for max_kv_len; excess blocks return early.
     *
     * @param src           Device pointer to Q8_1 blocks
     * @param dst           Device pointer to FP32 destination buffer
     * @param cols          n_kv_heads * head_dim (constant)
     * @param max_kv_len    Maximum kv_len for grid sizing (workspace capacity)
     * @param device_params Device pointer to AttentionDeviceParams (has dynamic kv_len)
     * @param stream        CUDA stream
     * @return 0 on success, -1 on error
     */
    int cudaFlashAttn_dequant_q8_1_to_fp32_dynamic(
        const void *src, float *dst,
        int cols, int max_kv_len,
        const void *device_params, void *stream)
    {
        if (!src || !dst || !device_params || cols <= 0 || max_kv_len <= 0)
            return -1;

        const int blocks_per_row = (cols + 31) / 32;
        const int max_total_blocks = max_kv_len * blocks_per_row;
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

        // One CUDA block per Q8_1 block, 32 threads per block
        dequant_q8_1_to_fp32_dynamic_kernel<<<max_total_blocks, 32, 0, cuda_stream>>>(
            static_cast<const GpuQ8_1Block *>(src), dst,
            cols, blocks_per_row,
            static_cast<const llaminar2::attention::AttentionDeviceParams *>(device_params));

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("[cudaFlashAttn_dequant_q8_1_to_fp32_dynamic] Kernel launch failed: %s\n",
                   cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

} // extern "C"
