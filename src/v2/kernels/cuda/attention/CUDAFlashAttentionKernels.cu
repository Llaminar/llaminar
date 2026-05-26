/**
 * @file CUDAFlashAttentionKernels.cu
 * @brief CUDA device kernels for Flash Attention 2 (Ampere) and Flash Decoding
 *
 * This file contains the CUDA device kernels and extern "C" wrapper functions
 * called from the C++ implementation file.
 *
 * Algorithms implemented:
 * - Flash Attention 2 with Pipelined Prefetching: Optimized for Ampere (SM >= 8.0)
 *   - Uses cp.async for overlapped global->shared memory transfers
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
#include <cublas_v2.h>
#include <mma.h> // WMMA (Warp Matrix Multiply-Accumulate) for Tensor Cores
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <atomic>
#include <atomic>

#include "../../attention/AttentionDeviceParams.h"
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
    // Flash Attention 2 - Pipelined Prefill with cp.async (Ampere, SM >= 8.0)
    // =========================================================================
    //
    // This implements Flash Attention 2 (Dao et al., 2023) with software pipelining
    // optimizations enabled by Ampere's cp.async instruction.
    //
    // Key optimizations:
    //   1. cp.async for asynchronous global->shared memory transfers
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
    //   Warps 0-1: Producers - async load K/V tiles via cp.async
    //   Warps 2+:  Consumers - WMMA compute + online softmax
    //
    // =========================================================================

    // Pipeline configuration
    constexpr int FA2_NUM_STAGES = 2;       // Double buffering
    constexpr int FA2_PRODUCER_WARPS = 2;   // Warps dedicated to loading (fixed)
    constexpr int FA2_TILE_KV_DEFAULT = 64; // Default KV tile size
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
            cudaDeviceGetAttribute(&cfg.max_smem_optin,
                                   cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
            cfg.initialized = true;
        }
        return cfg;
    }

    /**
     * @brief Compute shared memory size for a given FA2 configuration
     */
    inline size_t computeFA2SmemSize(int tile_q, int tile_kv, int head_dim,
                                     int qkv_pad, int scores_pad)
    {
        const int qkv_stride = head_dim + qkv_pad;
        const int scores_ld = tile_kv + scores_pad;
        size_t q_size = tile_q * qkv_stride * sizeof(half);
        size_t kv_size = FA2_NUM_STAGES * 2 * tile_kv * qkv_stride * sizeof(half);
        size_t sc_size = tile_q * scores_ld * sizeof(float);
        return q_size + kv_size + sc_size;
    }

    /**
     * @brief Compute FA2 kernel configuration based on head_dim
     *
     * Search strategy (preserving bank-conflict-free padding):
     *   1. Try target tile_q with tile_kv=64,48,32 and full padding (qkv=8, scores=16)
     *   2. Only drop padding as last resort if no tile_kv fits with padding
     *
     * QKV padding is critical: stride(head_dim=128)=256B → 32-way bank conflict.
     * With qkv_pad=8: stride=272B → 4-way conflict (8x better).
     *
     * Override: set LLAMINAR_FA2_TILE_KV=N to force a specific tile_kv value.
     *
     * Configurations:
     *   - head_dim <= 64:  tile_q=96, 6 consumers, 256 threads
     *   - head_dim <= 128: tile_q=64, 4 consumers, 192 threads
     *   - head_dim <= 256: tile_q=32, 2 consumers, 128 threads
     */
    struct FA2KernelConfig
    {
        int tile_q;
        int tile_kv;
        int num_consumer_warps;
        int block_size;
        size_t smem_size;
        int qkv_pad;
        int scores_pad;
    };

    inline FA2KernelConfig computeFA2Config(int head_dim, int max_smem)
    {
        FA2KernelConfig cfg;

        // Target tile_q and consumer warps based on head_dim
        int target_tile_q;
        int target_consumer_warps;
        if (head_dim <= 64)
        {
            target_tile_q = 96; // 6 * 16
            target_consumer_warps = 6;
        }
        else if (head_dim <= 128)
        {
            target_tile_q = 64; // 4 * 16
            target_consumer_warps = 4;
        }
        else
        {
            target_tile_q = 32; // 2 * 16
            target_consumer_warps = 2;
        }

        // Check for env-var tile_kv override (for parameter sweeps)
        const int forced_tile_kv = llaminar2::debugEnv().attention.cuda_fa2_tile_kv;

        // tile_kv candidates: must be multiple of WMMA_N=16.
        // Occupancy-aware selection: prefer tile_kv that maximizes blocks/SM
        // (computed via max_smem / smem_per_block), breaking ties by larger
        // tile_kv (fewer loop iterations). This is critical because shared
        // memory often limits to 1 block/SM with large tiles; smaller tiles
        // allow 2+ concurrent blocks and dramatically reduce barrier and
        // memory-latency stalls via warp-level parallelism.
        const int tile_kv_options[] = {64, 32, 16};
        const int num_tile_kv_options = forced_tile_kv > 0 ? 1 : 3;

        // Evaluate all candidates, pick highest occupancy then largest tile_kv
        int best_tkv = 0;
        size_t best_smem = 0;
        int best_blocks_per_sm = 0;
        bool found = false;

        for (int ti = 0; ti < num_tile_kv_options; ti++)
        {
            int tkv = forced_tile_kv > 0 ? forced_tile_kv : tile_kv_options[ti];
            size_t smem = computeFA2SmemSize(target_tile_q, tkv,
                                             head_dim, FA2_QKV_PAD, FA2_SCORES_LD_PAD);
            if ((int)smem <= max_smem)
            {
                int blocks_per_sm = max_smem / (int)smem;
                if (blocks_per_sm > best_blocks_per_sm ||
                    (blocks_per_sm == best_blocks_per_sm && tkv > best_tkv))
                {
                    best_tkv = tkv;
                    best_smem = smem;
                    best_blocks_per_sm = blocks_per_sm;
                }
            }
        }

        if (best_tkv > 0)
        {
            cfg.tile_q = target_tile_q;
            cfg.tile_kv = best_tkv;
            cfg.num_consumer_warps = target_consumer_warps;
            cfg.qkv_pad = FA2_QKV_PAD;
            cfg.scores_pad = FA2_SCORES_LD_PAD;
            cfg.smem_size = best_smem;
            found = true;
        }

        if (!found)
        {
            // Last resort: drop padding entirely to fit target tile_q
            int tkv = forced_tile_kv > 0 ? forced_tile_kv : 32;
            size_t smem = computeFA2SmemSize(target_tile_q, tkv, head_dim, 0, 0);
            if ((int)smem <= max_smem)
            {
                cfg.tile_q = target_tile_q;
                cfg.tile_kv = tkv;
                cfg.num_consumer_warps = target_consumer_warps;
                cfg.qkv_pad = 0;
                cfg.scores_pad = 0;
                cfg.smem_size = smem;
                found = true;
            }
        }

        if (!found)
        {
            // Even target tile_q doesn't fit — find largest that does
            cfg.qkv_pad = FA2_QKV_PAD;
            cfg.scores_pad = FA2_SCORES_LD_PAD;
            cfg.tile_kv = 32;
            cfg.tile_q = 32; // minimum
            for (int q : {128, 96, 64, 32})
            {
                size_t smem = computeFA2SmemSize(q, 32, head_dim,
                                                 FA2_QKV_PAD, FA2_SCORES_LD_PAD);
                if ((int)smem <= max_smem)
                {
                    cfg.tile_q = q;
                    break;
                }
            }
            cfg.num_consumer_warps = cfg.tile_q / 16;
            if (cfg.num_consumer_warps < 2)
                cfg.num_consumer_warps = 2;
            if (cfg.num_consumer_warps > 6)
                cfg.num_consumer_warps = 6;
            cfg.smem_size = computeFA2SmemSize(cfg.tile_q, cfg.tile_kv,
                                               head_dim, cfg.qkv_pad, cfg.scores_pad);
        }

        cfg.block_size = (FA2_PRODUCER_WARPS + cfg.num_consumer_warps) * WARP_SIZE;
        return cfg;
    }

    /**
     * @brief Flash Attention 2 kernel with pipelined prefetching (Ampere, SM >= 8.0)
     *
     * Uses cp.async for overlapped memory transfers with computation.
     * Double-buffered shared memory for K/V tiles.
     * Producer/consumer warp specialization with WMMA Tensor Core acceleration.
     *
     * Template parameters:
     *   NUM_CONSUMER_WARPS: Number of consumer warps (6 for head_dim=64, 4 for head_dim=128, 2 for head_dim=256)
     *   HEAD_DIM: Compile-time head dimension (64, 128, or 256). Critical for enabling
     *             full unrolling of the P@V accumulation loop and keeping O_acc in
     *             registers. Without this, dims_per_lane is runtime → O_acc spills
     *             to local memory → 10-20x performance loss.
     *   TILE_KV: Compile-time KV tile size (32 or 64). Enables full unrolling of
     *            the P@V outer (j) loop over KV positions, allowing the compiler to
     *            interleave V loads with FMAs. Also removes the warp-divergent
     *            masking branch from the P@V hot loop.
     *
     * Runtime parameters tile_q and tile_kv must be consistent with NUM_CONSUMER_WARPS:
     *   - tile_q should be NUM_CONSUMER_WARPS * 16 (WMMA_M)
     *   KV_FP16: When true, K/V pointers are const half* (FP16) — loaded via
     *            cp.async directly into shared memory with no register staging.
     *            When false, K/V are const float* — converted to FP16 per element.
     *            FP16 path eliminates the FP16→FP32→FP16 round-trip when KV cache
     *            is already FP16, roughly halving K/V global memory bandwidth.
     */
    template <int NUM_CONSUMER_WARPS, int HEAD_DIM, int TILE_KV, bool KV_FP16 = false>
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
        int gqa_n_rep = 0)
    {
        int kv_stride = kv_len;
        int kv_len_runtime = kv_len;
        int position_offset_runtime = position_offset;
        int mask_stride = kv_len;

        if (device_params)
        {
            kv_len_runtime = device_params->kv_len;
            position_offset_runtime = device_params->position_offset;
            mask_stride = device_params->mask_stride;
        }

        // Block/thread indexing
        const int batch_idx = blockIdx.z;
        const int head_idx = blockIdx.x;
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

        // Warp role: producer (warps 0-1) or consumer (warps 2+)
        // Use template parameter for consumer warp count
        const bool is_producer = (warp_id < FA2_PRODUCER_WARPS);
        const int consumer_warp_id = warp_id - FA2_PRODUCER_WARPS;
        const bool is_active_consumer = (!is_producer && consumer_warp_id >= 0 && consumer_warp_id < NUM_CONSUMER_WARPS);

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

        // Consumer warps: warp-cooperative per-row accumulators
        // 2 lanes per Q row: lanes 0-15 handle dims [0, head_dim/2),
        //                    lanes 16-31 handle dims [head_dim/2, head_dim)
        // This halves register pressure (O_acc[64] vs [128]) and doubles
        // throughput of the P@V accumulation loop.
        const int q_tile_rows = min(tile_q, seq_len - q_block_start);
        const int row_in_warp = lane_id & (WMMA_M - 1); // lane_id % 16
        const int dim_half = lane_id >> 4;              // 0 for lanes 0-15, 1 for 16-31
        const int my_consumer_q_row = is_active_consumer
                                          ? q_block_start + consumer_warp_id * WMMA_M + row_in_warp
                                          : -1;
        const bool owns_row = (my_consumer_q_row >= 0 &&
                               my_consumer_q_row < seq_len &&
                               (my_consumer_q_row - q_block_start) < q_tile_rows);

        // O_acc sized at compile-time HEAD_DIM to keep in registers.
        // active_dims_per_lane uses runtime head_dim for correct bounds
        // when head_dim < HEAD_DIM (e.g., head_dim=32 with HEAD_DIM=64).
        constexpr int dims_per_lane = HEAD_DIM >> 1;
        const int active_dims_per_lane = head_dim >> 1;
        const int dim_start = dim_half * active_dims_per_lane;
        float O_acc[dims_per_lane];
        float m_i = -FLT_MAX;
        float l_i = 0.0f;

        if (owns_row)
        {
#pragma unroll
            for (int d = 0; d < dims_per_lane; d++)
                O_acc[d] = 0.0f;
        }

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

        const int num_kv_tiles = (kv_len_runtime + tile_kv - 1) / tile_kv;

        // =====================================================================
        // Pipeline prologue: Start loading first tile(s)
        // =====================================================================
        if (is_producer)
        {
            const int kv_start_0 = 0;
            const int kv_end_0 = min(tile_kv, kv_len_runtime);
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
                    int global_row = kv_start_0 + local_row;
                    int kv_offset = global_row * n_kv_heads * head_dim + kv_head_idx * head_dim + d;
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
                    int global_row = kv_start_0 + local_row;
                    int kv_offset = global_row * n_kv_heads * head_dim + kv_head_idx * head_dim + d;
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

            const int kv_start = kv_tile_iter * tile_kv;
            const int kv_end_tile = min(kv_start + tile_kv, kv_len_runtime);
            const int actual_tile_kv_len = kv_end_tile - kv_start;

            // Early exit for causal
            if (causal && kv_start > (q_block_start + tile_q - 1) + position_offset)
            {
                continue;
            }

            // Get current buffers
            half *K_tile_fp16 = get_K_tile(current_stage);
            half *V_tile_fp16 = get_V_tile(current_stage);

            // ----------------------------------------------------------------
            // PRODUCER WARPS: Start loading next tile into next_stage buffer
            // ----------------------------------------------------------------
            if (is_producer && kv_tile_iter + 1 < num_kv_tiles)
            {
                const int next_kv_start = (kv_tile_iter + 1) * tile_kv;
                const int next_kv_end = min(next_kv_start + tile_kv, kv_len_runtime);
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
                        int global_row = next_kv_start + local_row;
                        int kv_offset = global_row * n_kv_heads * head_dim + kv_head_idx * head_dim + d;
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
                        int global_row = next_kv_start + local_row;
                        int kv_offset = global_row * n_kv_heads * head_dim + kv_head_idx * head_dim + d;
                        K_dst[local_row * smem_stride + d] = __float2half(K_batch_fp32[kv_offset]);
                        V_dst[local_row * smem_stride + d] = __float2half(V_batch_fp32[kv_offset]);
                    }
                }
            }

            // ----------------------------------------------------------------
            // CONSUMER WARPS: Compute Q @ K^T using WMMA
            // ----------------------------------------------------------------
            if (is_active_consumer)
            {
                const int warp_q_start = consumer_warp_id * WMMA_M;

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
            // CONSUMER WARPS: Apply softmax and accumulate P @ V
            // Warp-cooperative: 2 lanes per row, primary (dim_half=0) does
            // masking writes, then both lanes accumulate their dim half.
            //
            // CRITICAL: __syncwarp() and __shfl_sync() are placed at the
            // is_active_consumer level (NOT inside owns_row) so that all 32
            // lanes of the consumer warp participate. When the last Q tile
            // has fewer rows than tile_q, some lanes have owns_row=false;
            // placing warp-collective ops inside owns_row would deadlock.
            // ----------------------------------------------------------------
            if (is_active_consumer)
            {
                float m_ij = -FLT_MAX;

                // Only primary lane (dim_half==0) of rows that exist writes masking
                if (owns_row && dim_half == 0)
                {
                    const int local_q_row = consumer_warp_id * WMMA_M + row_in_warp;
                    float *my_scores = scores + local_q_row * scores_ld;

                    for (int j = 0; j < actual_tile_kv_len; j++)
                    {
                        const int kv_pos = kv_start + j;
                        bool masked = false;

                        if (causal && kv_pos > my_consumer_q_row + position_offset_runtime)
                            masked = true;
                        if (window_size > 0)
                        {
                            int q_pos = my_consumer_q_row + position_offset_runtime;
                            if (kv_pos < q_pos - window_size || kv_pos > q_pos + window_size)
                                masked = true;
                        }

                        if (mask)
                        {
                            float mask_val = mask[(batch_idx * seq_len + my_consumer_q_row) * mask_stride + kv_pos];
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
                            my_scores[j] *= softmax_scale;
                            m_ij = fmaxf(m_ij, my_scores[j]);
                        }
                    }
                }

                // All 32 lanes reach here — safe for warp-collective ops
                __syncwarp();

                // Broadcast m_ij from primary lane to its secondary partner
                // Lanes without rows shuffle -FLT_MAX harmlessly
                m_ij = __shfl_sync(0xFFFFFFFF, m_ij, row_in_warp);

                if (owns_row)
                {
                    const int local_q_row = consumer_warp_id * WMMA_M + row_in_warp;
                    float *my_scores = scores + local_q_row * scores_ld;

                    // Online softmax update
                    float m_i_new = fmaxf(m_i, m_ij);
                    float scale_old = __expf(m_i - m_i_new);

#pragma unroll
                    for (int d = 0; d < dims_per_lane; d++)
                        O_acc[d] *= scale_old;
                    l_i *= scale_old;

                    // Accumulate P @ V (branchless with compile-time unrolling)
                    //
                    // Optimizations over original scalar loop:
                    //   1. TILE_KV compile-time loop bound enables full #pragma unroll,
                    //      letting NVCC interleave V loads with FMAs for ILP.
                    //   2. Removed per-score branch (if s > -FLT_MAX/2): masked scores
                    //      are -FLT_MAX, so exp(-FLT_MAX - m) = 0 → zero contribution.
                    //      Eliminates warp divergence on causal masking boundaries.
                    //   3. j < actual_tile_kv_len is warp-uniform (same for all lanes)
                    //      → no divergence; just skips tail iterations.
                    //   4. Vectorized half2 V loads: 2× shared memory throughput.
                    float l_ij = 0.0f;
#pragma unroll 8
                    for (int j = 0; j < TILE_KV; j++)
                    {
                        if (j < actual_tile_kv_len)
                        {
                            float p = __expf(my_scores[j] - m_i_new);
                            l_ij += p;

                            const half2 *V_row_h2 = reinterpret_cast<const half2 *>(
                                V_tile_fp16 + j * smem_stride + dim_start);
#pragma unroll
                            for (int d = 0; d < dims_per_lane; d += 2)
                            {
                                float2 v = __half22float2(V_row_h2[d >> 1]);
                                O_acc[d] += p * v.x;
                                O_acc[d + 1] += p * v.y;
                            }
                        }
                    }

                    l_i += l_ij;
                    m_i = m_i_new;
                }
            }

            __syncthreads();
        }

        // =====================================================================
        // Write final output (each lane writes its half of dims)
        // =====================================================================
        if (is_active_consumer && owns_row)
        {
            float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;

            float *O_batch = O + batch_idx * seq_len * n_heads * head_dim;
            float *O_row = O_batch + my_consumer_q_row * n_heads * head_dim + head_idx * head_dim;

#pragma unroll
            for (int d = 0; d < dims_per_lane; d++)
            {
                if (d < active_dims_per_lane)
                    O_row[dim_start + d] = O_acc[d] * inv_l;
            }
        }
    }

    // Explicit template instantiations for supported configurations
    // KV_FP16=false (legacy FP32 K/V inputs)
    // head_dim=64: 6 consumer warps, tile_q=96
    template __global__ void flash_attention_2_pipelined_kernel<6, 64, 16, false>(
        const float *, const void *, const void *, float *,
        int, int, int, int, int, int, float, bool, int, int,
        const llaminar2::attention::AttentionDeviceParams *, const float *,
        int, int, int, int, int, int);
    template __global__ void flash_attention_2_pipelined_kernel<6, 64, 32, false>(
        const float *, const void *, const void *, float *,
        int, int, int, int, int, int, float, bool, int, int,
        const llaminar2::attention::AttentionDeviceParams *, const float *,
        int, int, int, int, int, int);
    template __global__ void flash_attention_2_pipelined_kernel<6, 64, 64, false>(
        const float *, const void *, const void *, float *,
        int, int, int, int, int, int, float, bool, int, int,
        const llaminar2::attention::AttentionDeviceParams *, const float *,
        int, int, int, int, int, int);
    // head_dim=128: 4 consumer warps, tile_q=64
    template __global__ void flash_attention_2_pipelined_kernel<4, 128, 16, false>(
        const float *, const void *, const void *, float *,
        int, int, int, int, int, int, float, bool, int, int,
        const llaminar2::attention::AttentionDeviceParams *, const float *,
        int, int, int, int, int, int);
    template __global__ void flash_attention_2_pipelined_kernel<4, 128, 32, false>(
        const float *, const void *, const void *, float *,
        int, int, int, int, int, int, float, bool, int, int,
        const llaminar2::attention::AttentionDeviceParams *, const float *,
        int, int, int, int, int, int);
    template __global__ void flash_attention_2_pipelined_kernel<4, 128, 64, false>(
        const float *, const void *, const void *, float *,
        int, int, int, int, int, int, float, bool, int, int,
        const llaminar2::attention::AttentionDeviceParams *, const float *,
        int, int, int, int, int, int);
    // head_dim=256: 2 consumer warps, tile_q=32
    template __global__ void flash_attention_2_pipelined_kernel<2, 256, 16, false>(
        const float *, const void *, const void *, float *,
        int, int, int, int, int, int, float, bool, int, int,
        const llaminar2::attention::AttentionDeviceParams *, const float *,
        int, int, int, int, int, int);
    template __global__ void flash_attention_2_pipelined_kernel<2, 256, 32, false>(
        const float *, const void *, const void *, float *,
        int, int, int, int, int, int, float, bool, int, int,
        const llaminar2::attention::AttentionDeviceParams *, const float *,
        int, int, int, int, int, int);
    template __global__ void flash_attention_2_pipelined_kernel<2, 256, 64, false>(
        const float *, const void *, const void *, float *,
        int, int, int, int, int, int, float, bool, int, int,
        const llaminar2::attention::AttentionDeviceParams *, const float *,
        int, int, int, int, int, int);

    // KV_FP16=true (FP16 K/V from KV cache — eliminates FP16→FP32→FP16 round-trip)
    template __global__ void flash_attention_2_pipelined_kernel<6, 64, 16, true>(
        const float *, const void *, const void *, float *,
        int, int, int, int, int, int, float, bool, int, int,
        const llaminar2::attention::AttentionDeviceParams *, const float *,
        int, int, int, int, int, int);
    template __global__ void flash_attention_2_pipelined_kernel<6, 64, 32, true>(
        const float *, const void *, const void *, float *,
        int, int, int, int, int, int, float, bool, int, int,
        const llaminar2::attention::AttentionDeviceParams *, const float *,
        int, int, int, int, int, int);
    template __global__ void flash_attention_2_pipelined_kernel<6, 64, 64, true>(
        const float *, const void *, const void *, float *,
        int, int, int, int, int, int, float, bool, int, int,
        const llaminar2::attention::AttentionDeviceParams *, const float *,
        int, int, int, int, int, int);
    template __global__ void flash_attention_2_pipelined_kernel<4, 128, 16, true>(
        const float *, const void *, const void *, float *,
        int, int, int, int, int, int, float, bool, int, int,
        const llaminar2::attention::AttentionDeviceParams *, const float *,
        int, int, int, int, int, int);
    template __global__ void flash_attention_2_pipelined_kernel<4, 128, 32, true>(
        const float *, const void *, const void *, float *,
        int, int, int, int, int, int, float, bool, int, int,
        const llaminar2::attention::AttentionDeviceParams *, const float *,
        int, int, int, int, int, int);
    template __global__ void flash_attention_2_pipelined_kernel<4, 128, 64, true>(
        const float *, const void *, const void *, float *,
        int, int, int, int, int, int, float, bool, int, int,
        const llaminar2::attention::AttentionDeviceParams *, const float *,
        int, int, int, int, int, int);
    // head_dim=256: 2 consumer warps, tile_q=32
    template __global__ void flash_attention_2_pipelined_kernel<2, 256, 16, true>(
        const float *, const void *, const void *, float *,
        int, int, int, int, int, int, float, bool, int, int,
        const llaminar2::attention::AttentionDeviceParams *, const float *,
        int, int, int, int, int, int);
    template __global__ void flash_attention_2_pipelined_kernel<2, 256, 32, true>(
        const float *, const void *, const void *, float *,
        int, int, int, int, int, int, float, bool, int, int,
        const llaminar2::attention::AttentionDeviceParams *, const float *,
        int, int, int, int, int, int);
    template __global__ void flash_attention_2_pipelined_kernel<2, 256, 64, true>(
        const float *, const void *, const void *, float *,
        int, int, int, int, int, int, float, bool, int, int,
        const llaminar2::attention::AttentionDeviceParams *, const float *,
        int, int, int, int, int, int);

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
        if (device_params)
        {
            kv_len_runtime = device_params->kv_len;
        }

        const int head_idx = blockIdx.x;
        const int split_idx = blockIdx.y;
        const int batch_idx = blockIdx.z;

        const int effective_gqa = (gqa_n_rep > 0) ? gqa_n_rep : ((n_heads == n_kv_heads) ? 1 : (n_heads / n_kv_heads));
        const int kv_head_idx = (gqa_n_rep > 0)
                                    ? (head_start + head_idx) / effective_gqa
                                    : head_idx / effective_gqa;

        const int split_size = (kv_len_runtime + num_splits - 1) / num_splits;
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
            const float *K_ptr = K_batch + kv_pos * n_kv_heads * head_dim + kv_head_idx * head_dim;

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
            const float *V_ptr = V_batch + kv_pos * n_kv_heads * head_dim + kv_head_idx * head_dim;
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
        int num_splits)
    {
        const int head_idx = blockIdx.x;
        const int batch_idx = blockIdx.y;

        const int tid = threadIdx.x;
        const int base_idx = (batch_idx * n_heads + head_idx) * num_splits;

        __shared__ float global_m;
        __shared__ float global_l;
        __shared__ float split_scales[32];

        if (tid == 0)
        {
            float m_max = -FLT_MAX;
            for (int s = 0; s < num_splits; s++)
            {
                m_max = fmaxf(m_max, m_partial[base_idx + s]);
            }
            global_m = m_max;

            float l_sum = 0.0f;
            for (int s = 0; s < num_splits; s++)
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

        for (int d = tid; d < head_dim; d += blockDim.x)
        {
            float O_sum = 0.0f;
            for (int s = 0; s < num_splits; s++)
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
        int head_start = 0,
        int gqa_n_rep = 0)
    {
        int kv_stride = kv_len;
        int kv_len_runtime = kv_len;
        if (device_params)
        {
            kv_len_runtime = device_params->kv_len;
        }

        const int head_idx = blockIdx.x;
        const int split_idx = blockIdx.y;
        const int batch_idx = blockIdx.z;

        const int effective_gqa = (gqa_n_rep > 0) ? gqa_n_rep : ((n_heads == n_kv_heads) ? 1 : (n_heads / n_kv_heads));
        const int kv_head_idx = (gqa_n_rep > 0)
                                    ? (head_start + head_idx) / effective_gqa
                                    : head_idx / effective_gqa;

        const int split_size = (kv_len_runtime + num_splits - 1) / num_splits;
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

        const half *K_batch = K_cache + batch_idx * kv_stride * n_kv_heads * head_dim;
        const half *V_batch = V_cache + batch_idx * kv_stride * n_kv_heads * head_dim;

        for (int kv_pos = kv_start + warp_id; kv_pos < kv_end; kv_pos += num_warps)
        {
            const half *K_ptr =
                K_batch + kv_pos * n_kv_heads * head_dim + kv_head_idx * head_dim;

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
                V_batch + kv_pos * n_kv_heads * head_dim + kv_head_idx * head_dim;
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
        }

        const int head_idx = blockIdx.x;
        const int split_idx = blockIdx.y;
        const int batch_idx = blockIdx.z;

        const int effective_gqa = (gqa_n_rep > 0) ? gqa_n_rep : ((n_heads == n_kv_heads) ? 1 : (n_heads / n_kv_heads));
        const int kv_head_idx = (gqa_n_rep > 0)
                                    ? (head_start + head_idx) / effective_gqa
                                    : head_idx / effective_gqa;

        const int split_size = (kv_len_runtime + num_splits - 1) / num_splits;
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

        const int split_size = (kv_count_rt + num_splits - 1) / num_splits;
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
            float k_norm = *reinterpret_cast<const float *>(k_block);
            const uint8_t *k_indices = k_block + 2 * sizeof(float); // skip norm + residual_norm

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
            float v_norm = *reinterpret_cast<const float *>(v_block);
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
 *   - head_dim <= 64:  6 consumer warps, tile_q=96, 256 threads
 *   - head_dim <= 128: 4 consumer warps, tile_q=64, 192 threads
 *   - head_dim <= 256: 2 consumer warps, tile_q=32, 128 threads
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

    if (head_dim <= 0 || head_dim % 16 != 0 || head_dim > 256)
    {
        printf("[cudaFlashAttn_prefill_fa2] Error: head_dim=%d invalid (must be 16-256, multiple of 16)\n", head_dim);
        return -1;
    }

    FA2KernelConfig cfg = computeFA2Config(head_dim, dev_cfg.max_smem_optin);

    int num_q_tiles = (seq_len + cfg.tile_q - 1) / cfg.tile_q;
    dim3 grid(n_heads, num_q_tiles, batch_size);

    cudaError_t err;

    // Macro to reduce dispatch boilerplate
#define FA2_LAUNCH(CW, HD, TKV)                                                                            \
    do                                                                                                     \
    {                                                                                                      \
        auto kernel_fn = flash_attention_2_pipelined_kernel<CW, HD, TKV, KV_FP16>;                         \
        err = cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, cfg.smem_size); \
        if (err != cudaSuccess)                                                                            \
        {                                                                                                  \
            printf("[cudaFlashAttn_prefill_fa2] cudaFuncSetAttribute<%d,%d,%d,%d>(smem=%zu) FAILED: %s\n", \
                   CW, HD, TKV, (int)KV_FP16, cfg.smem_size, cudaGetErrorString(err));                     \
            return -1;                                                                                     \
        }                                                                                                  \
        kernel_fn<<<grid, cfg.block_size, cfg.smem_size, cuda_stream>>>(                                   \
            Q, K, V, O,                                                                                    \
            batch_size, seq_len, kv_len, n_heads, n_kv_heads, head_dim,                                    \
            softmax_scale, causal, window_size, position_offset,                                           \
            device_params, mask, cfg.tile_q, cfg.tile_kv, cfg.qkv_pad, cfg.scores_pad,                     \
            head_start, gqa_n_rep);                                                                        \
    } while (0)

    if (cfg.num_consumer_warps == 6)
    {
        if (cfg.tile_kv == 64)
            FA2_LAUNCH(6, 64, 64);
        else if (cfg.tile_kv == 16)
            FA2_LAUNCH(6, 64, 16);
        else
            FA2_LAUNCH(6, 64, 32);
    }
    else if (cfg.num_consumer_warps == 2)
    {
        if (cfg.tile_kv == 64)
            FA2_LAUNCH(2, 256, 64);
        else if (cfg.tile_kv == 16)
            FA2_LAUNCH(2, 256, 16);
        else
            FA2_LAUNCH(2, 256, 32);
    }
    else
    {
        if (cfg.tile_kv == 64)
            FA2_LAUNCH(4, 128, 64);
        else if (cfg.tile_kv == 16)
            FA2_LAUNCH(4, 128, 16);
        else
            FA2_LAUNCH(4, 128, 32);
    }

#undef FA2_LAUNCH

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("[cudaFlashAttn_prefill_fa2] CUDA error: %s (smem=%zu bytes, tile_q=%d, tile_kv=%d, head_dim=%d, "
               "consumer_warps=%d, kv_fp16=%d, grid=(%d,%d,%d), block=%d)\n",
               cudaGetErrorString(err), cfg.smem_size, cfg.tile_q, cfg.tile_kv, head_dim,
               cfg.num_consumer_warps, (int)KV_FP16, grid.x, grid.y, grid.z, cfg.block_size);
        return -1;
    }
    return 0;
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
            dim3 grid(n_heads, batch_size);
            int block_size = min(head_dim, 256);

            flash_decoding_reduce_fp32_kernel<<<grid, block_size, 0, cuda_stream>>>(
                O_partial, m_partial, l_partial, O,
                n_heads, head_dim, num_splits);
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

            flash_decoding_fp16kv_kernel<<<grid, block_size, smem_size, cuda_stream>>>(
                Q,
                static_cast<const half *>(K_cache_fp16),
                static_cast<const half *>(V_cache_fp16),
                O_partial, m_partial, l_partial,
                kv_len, n_heads, n_kv_heads, head_dim,
                num_splits, softmax_scale, device_params,
                head_start, gqa_n_rep);
        }

        // Phase 2: Reduce partials (same as FP32 — operates on FP32 partials)
        {
            dim3 grid(n_heads, batch_size);
            int block_size = min(head_dim, 256);

            flash_decoding_reduce_fp32_kernel<<<grid, block_size, 0, cuda_stream>>>(
                O_partial, m_partial, l_partial, O,
                n_heads, head_dim, num_splits);
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
            dim3 grid(n_heads, batch_size);
            int block_size = min(head_dim, 256);

            flash_decoding_reduce_fp32_kernel<<<grid, block_size, 0, cuda_stream>>>(
                O_partial, m_partial, l_partial, O,
                n_heads, head_dim, num_splits);
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
            dim3 grid(n_heads, batch_size);
            int block_size = min(head_dim, 256);

            flash_decoding_reduce_fp32_kernel<<<grid, block_size, 0, cuda_stream>>>(
                O_partial, m_partial, l_partial, O,
                n_heads, head_dim, num_splits);
        }

        return cudaGetLastError() == cudaSuccess ? 0 : -1;
    }

    /**
     * @brief Allocate workspace for Flash Decoding
     */
    int cudaFlashAttn_allocWorkspace(
        void **partial_output, void **partial_m, void **partial_l,
        int batch_size, int n_heads, int head_dim, int num_splits)
    {
        size_t output_size = static_cast<size_t>(batch_size) * n_heads * num_splits * head_dim * sizeof(float);
        size_t scalar_size = static_cast<size_t>(batch_size) * n_heads * num_splits * sizeof(float);

        cudaError_t err1 = cudaMalloc(partial_output, output_size);
        cudaError_t err2 = cudaMalloc(partial_m, scalar_size);
        cudaError_t err3 = cudaMalloc(partial_l, scalar_size);

        if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess)
        {
            if (*partial_output)
                cudaFree(*partial_output);
            if (*partial_m)
                cudaFree(*partial_m);
            if (*partial_l)
                cudaFree(*partial_l);
            *partial_output = nullptr;
            *partial_m = nullptr;
            *partial_l = nullptr;
            return -1;
        }

        return 0;
    }

    /**
     * @brief Free workspace
     */
    void cudaFlashAttn_freeWorkspace(void *partial_output, void *partial_m, void *partial_l)
    {
        if (partial_output)
            cudaFree(partial_output);
        if (partial_m)
            cudaFree(partial_m);
        if (partial_l)
            cudaFree(partial_l);
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

// =============================================================================
// cuBLAS Unfused Attention Path
//
// Alternative to Flash Attention 2 for prefill. Uses cuBLAS strided batched
// GEMMs for QK^T and PV with a custom causal softmax kernel between them.
// This trades O(seq_len^2) workspace for much higher compute throughput:
// cuBLAS uses optimized Tensor Core GEMMs vs our FA2's scalar P@V.
//
// Memory layout: [batch, seq_len, n_heads, head_dim] (interleaved).
// GQA: stride=0 broadcasts K/V across query heads within each group.
// =============================================================================

namespace
{
    /**
     * @brief Convert FP32 tensor to FP16 (for Q)
     */
    __global__ void cublas_attn_fp32_to_fp16_kernel(
        const float *__restrict__ src,
        half *__restrict__ dst,
        int count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < count)
            dst[idx] = __float2half(src[idx]);
    }

    /**
     * @brief Warp-level max reduction
     */
    __device__ __forceinline__ float warpReduceMaxAttn(float val)
    {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, 16));
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, 8));
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, 4));
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, 2));
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, 1));
        return val;
    }

    /**
     * @brief Warp-level sum reduction
     */
    __device__ __forceinline__ float warpReduceSumAttn(float val)
    {
        val += __shfl_xor_sync(0xFFFFFFFF, val, 16);
        val += __shfl_xor_sync(0xFFFFFFFF, val, 8);
        val += __shfl_xor_sync(0xFFFFFFFF, val, 4);
        val += __shfl_xor_sync(0xFFFFFFFF, val, 2);
        val += __shfl_xor_sync(0xFFFFFFFF, val, 1);
        return val;
    }

    /**
     * @brief Causal masked softmax: FP32 scores → FP16 probabilities
     *
     * One block per row (seq position × head). Three-pass algorithm:
     * 1. Find max over valid (causal) positions
     * 2. Compute exp and sum
     * 3. Normalize and convert to FP16
     *
     * Causal mask: position i can attend to j where j <= position_offset + i.
     */
    __global__ void causal_softmax_fp32_to_fp16_kernel(
        const float *__restrict__ S,
        half *__restrict__ P,
        int seq_len, int kv_len,
        int position_offset,
        bool causal)
    {
        extern __shared__ float softmax_smem[];

        const int row_idx = blockIdx.x;
        const int head = row_idx / seq_len;
        const int i = row_idx % seq_len;

        const float *S_row = S + (size_t)head * seq_len * kv_len + (size_t)i * kv_len;
        half *P_row = P + (size_t)head * seq_len * kv_len + (size_t)i * kv_len;

        int valid_len = causal ? min(position_offset + i + 1, kv_len) : kv_len;

        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 31;
        const int num_warps = (blockDim.x + 31) / 32;

        // Phase 1: Find max
        float max_val = -FLT_MAX;
        for (int j = threadIdx.x; j < valid_len; j += blockDim.x)
            max_val = fmaxf(max_val, S_row[j]);

        max_val = warpReduceMaxAttn(max_val);
        if (lane_id == 0)
            softmax_smem[warp_id] = max_val;
        __syncthreads();

        if (threadIdx.x < num_warps)
            max_val = softmax_smem[threadIdx.x];
        else
            max_val = -FLT_MAX;
        max_val = warpReduceMaxAttn(max_val);
        // Broadcast from lane 0 to all threads
        max_val = __shfl_sync(0xFFFFFFFF, max_val, 0);
        __syncthreads();

        // Phase 2: Compute exp sum
        float sum = 0.0f;
        for (int j = threadIdx.x; j < valid_len; j += blockDim.x)
            sum += __expf(S_row[j] - max_val);

        sum = warpReduceSumAttn(sum);
        if (lane_id == 0)
            softmax_smem[warp_id] = sum;
        __syncthreads();

        if (threadIdx.x < num_warps)
            sum = softmax_smem[threadIdx.x];
        else
            sum = 0.0f;
        sum = warpReduceSumAttn(sum);
        sum = __shfl_sync(0xFFFFFFFF, sum, 0);
        __syncthreads();

        float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;

        // Phase 3: Normalize and convert to FP16
        for (int j = threadIdx.x; j < valid_len; j += blockDim.x)
            P_row[j] = __float2half(__expf(S_row[j] - max_val) * inv_sum);

        // Zero out masked positions
        for (int j = valid_len + threadIdx.x; j < kv_len; j += blockDim.x)
            P_row[j] = __float2half(0.0f);
    }

    // Persistent workspace for cuBLAS attention (grow-only, never freed)
    struct CuBLASAttnWorkspace
    {
        void *q_fp16 = nullptr;
        void *S = nullptr;
        void *P_fp16 = nullptr;
        size_t q_fp16_bytes = 0;
        size_t S_bytes = 0;
        size_t P_fp16_bytes = 0;
        cublasHandle_t handle = nullptr;

        void ensure(size_t need_q, size_t need_S, size_t need_P, int device_idx)
        {
            if (!handle)
            {
                cudaSetDevice(device_idx);
                cublasCreate(&handle);
            }
            if (need_q > q_fp16_bytes)
            {
                if (q_fp16)
                    cudaFree(q_fp16);
                cudaMalloc(&q_fp16, need_q);
                q_fp16_bytes = need_q;
            }
            if (need_S > S_bytes)
            {
                if (S)
                    cudaFree(S);
                cudaMalloc(&S, need_S);
                S_bytes = need_S;
            }
            if (need_P > P_fp16_bytes)
            {
                if (P_fp16)
                    cudaFree(P_fp16);
                cudaMalloc(&P_fp16, need_P);
                P_fp16_bytes = need_P;
            }
        }
    };

    static CuBLASAttnWorkspace g_cublas_attn_ws;

} // anonymous namespace

extern "C"
{

    /**
     * @brief cuBLAS-based unfused attention for prefill with FP16 KV
     *
     * Algorithm:
     *   1. Convert Q FP32 → FP16
     *   2. For each GQA group: cuBLAS batched GEMM  S = scale * Q @ K^T
     *   3. Causal softmax: S (FP32) → P (FP16)
     *   4. For each GQA group: cuBLAS batched GEMM  O = P @ V
     *
     * @param Q          FP32 query [batch, seq_len, n_heads, head_dim]
     * @param K_fp16     FP16 key   [batch, kv_len, n_kv_heads, head_dim]
     * @param V_fp16     FP16 value [batch, kv_len, n_kv_heads, head_dim]
     * @param O          FP32 output [batch, seq_len, n_heads, head_dim]
     * @param cublas_handle  cuBLAS handle (void* cast of cublasHandle_t)
     */
    int cudaFlashAttn_prefill_cublas_fp16kv(
        const float *Q, const void *K_fp16, const void *V_fp16, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int position_offset,
        void *stream,
        int device_idx)
    {
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

        cudaSetDevice(device_idx);

        const int group_size = n_heads / n_kv_heads;
        const float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

        // Workspace sizing
        const size_t q_fp16_need = (size_t)batch_size * seq_len * n_heads * head_dim * sizeof(half);
        const size_t S_need = (size_t)batch_size * n_heads * seq_len * kv_len * sizeof(float);
        const size_t P_need = (size_t)batch_size * n_heads * seq_len * kv_len * sizeof(half);
        g_cublas_attn_ws.ensure(q_fp16_need, S_need, P_need, device_idx);

        cublasHandle_t handle = g_cublas_attn_ws.handle;
        cublasSetStream(handle, cuda_stream);

        half *Q_fp16 = static_cast<half *>(g_cublas_attn_ws.q_fp16);
        float *S_buf = static_cast<float *>(g_cublas_attn_ws.S);
        half *P_buf = static_cast<half *>(g_cublas_attn_ws.P_fp16);
        const half *K_ptr = static_cast<const half *>(K_fp16);
        const half *V_ptr = static_cast<const half *>(V_fp16);

        // Step 1: Convert Q FP32 → FP16
        {
            const int count = batch_size * seq_len * n_heads * head_dim;
            const int block = 256;
            const int grid = (count + block - 1) / block;
            cublas_attn_fp32_to_fp16_kernel<<<grid, block, 0, cuda_stream>>>(Q, Q_fp16, count);
        }

        for (int b = 0; b < batch_size; b++)
        {
            const half *Q_batch = Q_fp16 + (size_t)b * seq_len * n_heads * head_dim;
            const half *K_batch = K_ptr + (size_t)b * kv_len * n_kv_heads * head_dim;
            const half *V_batch = V_ptr + (size_t)b * kv_len * n_kv_heads * head_dim;
            float *O_batch = O + (size_t)b * seq_len * n_heads * head_dim;
            float *S_batch = S_buf + (size_t)b * n_heads * seq_len * kv_len;
            half *P_batch = P_buf + (size_t)b * n_heads * seq_len * kv_len;

            // Step 2: QK^T per GQA group
            // Row-major S = scale * Q × K^T
            // col-major: S^T[kv_len, seq_len] = scale * op_T(K_cm) × op_N(Q_cm)
            for (int kh = 0; kh < n_kv_heads; kh++)
            {
                float alpha = softmax_scale;
                float beta = 0.0f;

                cublasStatus_t status = cublasGemmStridedBatchedEx(
                    handle,
                    CUBLAS_OP_T, // transa: transpose K from [D, kv_len] → [kv_len, D]
                    CUBLAS_OP_N, // transb: Q as [D, seq_len]
                    kv_len,      // m
                    seq_len,     // n
                    head_dim,    // k
                    &alpha,
                    K_batch + kh * head_dim,              // A (K per KV head)
                    CUDA_R_16F,                           //
                    n_kv_heads * head_dim,                // lda
                    (long long)0,                         // strideA = 0 (broadcast K)
                    Q_batch + kh * group_size * head_dim, // B (Q per group start)
                    CUDA_R_16F,                           //
                    n_heads * head_dim,                   // ldb
                    (long long)head_dim,                  // strideB (step through query heads)
                    &beta,
                    S_batch + (size_t)kh * group_size * seq_len * kv_len, // C
                    CUDA_R_32F,                                           //
                    kv_len,                                               // ldc
                    (long long)seq_len * kv_len,                          // strideC
                    group_size,                                           // batch count
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT);

                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    printf("[cudaFlashAttn_prefill_cublas_fp16kv] cuBLAS QK^T failed: %d (kh=%d)\n",
                           (int)status, kh);
                    return -1;
                }
            }

            // Step 3: Causal softmax S(FP32) → P(FP16) for all heads at once
            {
                const int total_rows = n_heads * seq_len;
                const int block = 256;
                const int smem = sizeof(float) * ((block + 31) / 32);
                causal_softmax_fp32_to_fp16_kernel<<<total_rows, block, smem, cuda_stream>>>(
                    S_batch, P_batch, seq_len, kv_len, position_offset, causal);
            }

            // Step 4: PV per GQA group
            // Row-major O = P × V
            // col-major: O^T[D, seq_len] = op_N(V_cm[D, kv_len]) × op_N(P^T_cm[kv_len, seq_len])
            for (int kh = 0; kh < n_kv_heads; kh++)
            {
                float alpha = 1.0f;
                float beta = 0.0f;

                cublasStatus_t status = cublasGemmStridedBatchedEx(
                    handle,
                    CUBLAS_OP_N, // transa: V as [D, kv_len] in col-major
                    CUBLAS_OP_N, // transb: P^T as [kv_len, seq_len] in col-major
                    head_dim,    // m = D
                    seq_len,     // n
                    kv_len,      // k
                    &alpha,
                    V_batch + kh * head_dim,                              // A (V per KV head)
                    CUDA_R_16F,                                           //
                    n_kv_heads * head_dim,                                // lda
                    (long long)0,                                         // strideA = 0 (broadcast V)
                    P_batch + (size_t)kh * group_size * seq_len * kv_len, // B (P per head)
                    CUDA_R_16F,                                           //
                    kv_len,                                               // ldb
                    (long long)seq_len * kv_len,                          // strideB
                    &beta,
                    O_batch + kh * group_size * head_dim, // C (interleaved output)
                    CUDA_R_32F,                           //
                    n_heads * head_dim,                   // ldc (interleaved stride)
                    (long long)head_dim,                  // strideC
                    group_size,                           // batch count
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT);

                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    printf("[cudaFlashAttn_prefill_cublas_fp16kv] cuBLAS PV failed: %d (kh=%d)\n",
                           (int)status, kh);
                    return -1;
                }
            }
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("[cudaFlashAttn_prefill_cublas_fp16kv] CUDA error: %s\n", cudaGetErrorString(err));
            return -1;
        }
        return 0;
    }

} // extern "C" — cuBLAS attention
