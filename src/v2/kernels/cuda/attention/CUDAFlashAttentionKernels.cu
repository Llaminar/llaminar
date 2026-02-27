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
 *   - Adaptive tile sizing for head_dim=64 and head_dim=128
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

#include "../../attention/AttentionDeviceParams.h"

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
    constexpr int FA2_NUM_STAGES = 2;     // Double buffering
    constexpr int FA2_PRODUCER_WARPS = 2; // Warps dedicated to loading (fixed)
    constexpr int FA2_TILE_KV = 64;       // KV tile size (constant for good K/V reuse)
    // Shared-memory padding for the [tile_q, tile_kv] scores matrix.
    // Note: WMMA store/load paths can have implicit alignment/stride constraints;
    // keep the padded leading dimension a multiple of 16.
    constexpr int FA2_SCORES_LD_PAD = 16;
    constexpr int FA2_QKV_PAD = 8; // Pad Q/K/V leading dimension by 8 halves (16 bytes)

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
     * @brief Calculate optimal tile_q based on head_dim and available shared memory
     *
     * Shared memory layout:
     *   Q_tile: [tile_q, head_dim] in FP16
     *   KV double buffer: 2 stages * (K + V) * [tile_kv, head_dim] in FP16
     *   Scores: [tile_q, tile_kv] in FP32
     *
     * Total = tile_q * head_dim * 2 + 2 * 2 * tile_kv * head_dim * 2 + tile_q * tile_kv * 4
     */
    inline int computeOptimalTileQ(int head_dim, int max_smem)
    {
        const int tile_kv = FA2_TILE_KV;
        const int scores_ld = tile_kv + FA2_SCORES_LD_PAD;
        const int qkv_stride = head_dim + FA2_QKV_PAD;

        // KV buffer is fixed size (doesn't depend on tile_q)
        size_t kv_buffer_size = FA2_NUM_STAGES * 2 * tile_kv * qkv_stride * sizeof(half);

        // Available for Q tile + scores
        if (kv_buffer_size >= max_smem)
            return 32;
        size_t available = max_smem - kv_buffer_size;

        // Q tile + scores = tile_q * (qkv_stride * sizeof(half) + scores_ld * sizeof(float))
        // Note: scores_ld is padded to reduce shared-memory bank conflicts.
        size_t bytes_per_q_row = qkv_stride * sizeof(half) + scores_ld * sizeof(float);

        // Calculate tile_q candidates
        auto get_smem_for_q = [&](int q)
        {
            return kv_buffer_size + q * bytes_per_q_row;
        };

        // We want to maximize a score:
        // Score = OccupancyTier * 1000 + TileQ
        // where OccupancyTier is 2 (fits 2 blocks) or 1 (fits 1 block)
        int best_tile_q = 32;
        int best_score = -1;

        // Iterate standard tile sizes
        // On RTX 3090, 64 is often the sweet spot for occupancy with padding
        for (int q : {32, 64, 96, 128})
        {
            size_t used = get_smem_for_q(q);
            if (used > max_smem)
                continue;

            // Occupancy check: assume 2 blocks if usage <= 50% of max
            // Note: max_smem here is the opt-in limit (e.g. 100KB on Ampere)
            int occupancy = (used <= max_smem / 2) ? 2 : 1;

            // Score prefers higher occupancy first, then larger tile
            int score = occupancy * 1000 + q;

            if (score > best_score)
            {
                best_score = score;
                best_tile_q = q;
            }
        }

        return best_tile_q;
    }

    /**
     * @brief Compute FA2 kernel configuration based on head_dim
     *
     * Configurations:
     *   - head_dim <= 64:  tile_q=96, 6 consumers, 256 threads (~70KB smem)
     *   - head_dim <= 128: tile_q=64, 4 consumers, 192 threads (~98KB smem)
     */
    struct FA2KernelConfig
    {
        int tile_q;
        int tile_kv;
        int num_consumer_warps;
        int block_size;
        size_t smem_size;
    };

    inline FA2KernelConfig computeFA2Config(int head_dim, int max_smem)
    {
        FA2KernelConfig cfg;
        cfg.tile_kv = FA2_TILE_KV; // Always 64

        if (head_dim <= 64)
        {
            // Default config: 6 consumer warps for smaller head_dim
            cfg.tile_q = 96; // 6 * 16
            cfg.num_consumer_warps = 6;
            cfg.block_size = (FA2_PRODUCER_WARPS + 6) * WARP_SIZE; // 256
        }
        else
        {
            // Reduced config for head_dim=128: 4 consumer warps
            cfg.tile_q = 64; // 4 * 16
            cfg.num_consumer_warps = 4;
            cfg.block_size = (FA2_PRODUCER_WARPS + 4) * WARP_SIZE; // 192
        }

        // Verify shared memory fits
        int qkv_stride = head_dim + FA2_QKV_PAD;
        size_t q_tile_size = cfg.tile_q * qkv_stride * sizeof(half);
        size_t kv_buffer_size = FA2_NUM_STAGES * 2 * cfg.tile_kv * qkv_stride * sizeof(half);
        // Pad scores leading dimension to mitigate shared-memory bank conflicts
        size_t scores_size = cfg.tile_q * (cfg.tile_kv + FA2_SCORES_LD_PAD) * sizeof(float);
        cfg.smem_size = q_tile_size + kv_buffer_size + scores_size;

        // If still too large, fall back to even smaller tile_q
        if ((int)cfg.smem_size > max_smem)
        {
            cfg.tile_q = computeOptimalTileQ(head_dim, max_smem);
            cfg.num_consumer_warps = cfg.tile_q / 16;
            if (cfg.num_consumer_warps < 2)
                cfg.num_consumer_warps = 2;
            if (cfg.num_consumer_warps > 6)
                cfg.num_consumer_warps = 6;
            cfg.block_size = (FA2_PRODUCER_WARPS + cfg.num_consumer_warps) * WARP_SIZE;

            // Recompute smem
            q_tile_size = cfg.tile_q * head_dim * sizeof(half);
            scores_size = cfg.tile_q * (cfg.tile_kv + FA2_SCORES_LD_PAD) * sizeof(float);
            cfg.smem_size = q_tile_size + kv_buffer_size + scores_size;
        }

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
     *   NUM_CONSUMER_WARPS: Number of consumer warps (4 for head_dim=128, 6 for head_dim=64)
     *
     * Runtime parameters tile_q and tile_kv must be consistent with NUM_CONSUMER_WARPS:
     *   - tile_q should be NUM_CONSUMER_WARPS * 16 (WMMA_M)
     */
    template <int NUM_CONSUMER_WARPS>
    __global__ void flash_attention_2_pipelined_kernel(
        const float *__restrict__ Q,
        const float *__restrict__ K,
        const float *__restrict__ V,
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
        int tile_kv)
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

        // GQA mapping
        const int kv_head_idx = (n_heads == n_kv_heads) ? head_idx : (head_idx / (n_heads / n_kv_heads));

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

        const int smem_stride = head_dim + FA2_QKV_PAD;
        const int kv_tile_size = tile_kv * smem_stride;
        const int scores_ld = tile_kv + FA2_SCORES_LD_PAD; // padded LD to reduce shared-memory bank conflicts

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

        // Consumer warps: per-row accumulators
        const int q_tile_rows = min(tile_q, seq_len - q_block_start);
        const int my_consumer_q_row = (is_active_consumer && lane_id < WMMA_M)
                                          ? q_block_start + consumer_warp_id * WMMA_M + lane_id
                                          : -1;
        const bool owns_row = (my_consumer_q_row >= 0 &&
                               my_consumer_q_row < seq_len &&
                               (my_consumer_q_row - q_block_start) < q_tile_rows);

        float O_acc[128];
        float m_i = -FLT_MAX;
        float l_i = 0.0f;

        if (owns_row)
        {
            for (int d = 0; d < head_dim; d++)
                O_acc[d] = 0.0f;
        }

        // Pointers for this batch/head
        const float *Q_batch = Q + batch_idx * seq_len * n_heads * head_dim;
        const float *K_batch = K + batch_idx * kv_stride * n_kv_heads * head_dim;
        const float *V_batch = V + batch_idx * kv_stride * n_kv_heads * head_dim;

        // ALL warps: Load Q tile (done once)
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
        // Pipeline prologue: Start loading first tile(s) asynchronously
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

            for (int i = my_start + lane_id; i < my_end; i += WARP_SIZE)
            {
                int local_row = i / head_dim;
                int d = i % head_dim;
                int global_row = kv_start_0 + local_row;
                float k_val = K_batch[global_row * n_kv_heads * head_dim + kv_head_idx * head_dim + d];
                float v_val = V_batch[global_row * n_kv_heads * head_dim + kv_head_idx * head_dim + d];
                K_dst_0[local_row * smem_stride + d] = __float2half(k_val);
                V_dst_0[local_row * smem_stride + d] = __float2half(v_val);
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

                for (int i = my_start + lane_id; i < my_end; i += WARP_SIZE)
                {
                    int local_row = i / head_dim;
                    int d = i % head_dim;
                    int global_row = next_kv_start + local_row;
                    float k_val = K_batch[global_row * n_kv_heads * head_dim + kv_head_idx * head_dim + d];
                    float v_val = V_batch[global_row * n_kv_heads * head_dim + kv_head_idx * head_dim + d];
                    K_dst[local_row * smem_stride + d] = __float2half(k_val);
                    V_dst[local_row * smem_stride + d] = __float2half(v_val);
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
            // ----------------------------------------------------------------
            if (is_active_consumer && owns_row)
            {
                const int local_q_row = consumer_warp_id * WMMA_M + lane_id;
                float *my_scores = scores + local_q_row * scores_ld;

                float m_ij = -FLT_MAX;

                // Apply scaling and masking
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

                // Online softmax update
                float m_i_new = fmaxf(m_i, m_ij);
                float scale_old = expf(m_i - m_i_new);

                for (int d = 0; d < head_dim; d++)
                    O_acc[d] *= scale_old;
                l_i *= scale_old;

                // Accumulate P @ V
                float l_ij = 0.0f;
                for (int j = 0; j < actual_tile_kv_len; j++)
                {
                    float s = my_scores[j];
                    if (s > -FLT_MAX / 2)
                    {
                        float p = expf(s - m_i_new);
                        l_ij += p;

                        const half *V_row = V_tile_fp16 + j * smem_stride;
                        for (int d = 0; d < head_dim; d++)
                        {
                            O_acc[d] += p * __half2float(V_row[d]);
                        }
                    }
                }

                l_i += l_ij;
                m_i = m_i_new;
            }

            __syncthreads();
        }

        // =====================================================================
        // Write final output
        // =====================================================================
        if (is_active_consumer && owns_row)
        {
            float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;

            float *O_batch = O + batch_idx * seq_len * n_heads * head_dim;
            float *O_row = O_batch + my_consumer_q_row * n_heads * head_dim + head_idx * head_dim;

            for (int d = 0; d < head_dim; d++)
            {
                O_row[d] = O_acc[d] * inv_l;
            }
        }
    }

    // Explicit template instantiations for supported configurations
    template __global__ void flash_attention_2_pipelined_kernel<6>(
        const float *, const float *, const float *, float *,
        int, int, int, int, int, int, float, bool, int, int,
        const llaminar2::attention::AttentionDeviceParams *, const float *,
        int, int);
    template __global__ void flash_attention_2_pipelined_kernel<4>(
        const float *, const float *, const float *, float *,
        int, int, int, int, int, int, float, bool, int, int,
        const llaminar2::attention::AttentionDeviceParams *, const float *,
        int, int);

    // =========================================================================
    // Flash Decoding - Split-K Kernel (FP32)
    // =========================================================================
    //
    // The key insight: we store (m, l, O) partials where:
    //   m = max score seen in this split
    //   l = sum of exp(score - m) for all positions in split
    //   O = sum of exp(score - m) * V for all positions (unnormalized)
    //
    // These can be combined across splits using stable softmax merge.
    // =========================================================================

    /**
     * @brief Warp-level reduction of softmax state (m, l, O)
     */
    __device__ void warpReduceSoftmaxState(
        float &m, float &l, float *O, int head_dim)
    {
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        {
            float other_m = __shfl_xor_sync(0xffffffff, m, offset);
            float other_l = __shfl_xor_sync(0xffffffff, l, offset);

            float m_new = fmaxf(m, other_m);
            float scale_self = expf(m - m_new);
            float scale_other = expf(other_m - m_new);

            l = scale_self * l + scale_other * other_l;

            for (int d = 0; d < head_dim; d++)
            {
                float other_O = __shfl_xor_sync(0xffffffff, O[d], offset);
                O[d] = scale_self * O[d] + scale_other * other_O;
            }

            m = m_new;
        }
    }

    /**
     * @brief Flash Decoding kernel for single-query decode
     *
     * Parallelizes over KV cache using split-K pattern.
     * Grid: (n_heads, num_splits, batch_size)
     * Block: (256,) threads
     */
    __global__ void flash_decoding_fp32_kernel(
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
        const llaminar2::attention::AttentionDeviceParams *__restrict__ device_params)
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

        const int kv_head_idx = (n_heads == n_kv_heads) ? head_idx : (head_idx / (n_heads / n_kv_heads));

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

        extern __shared__ char smem[];
        float *Q_shared = reinterpret_cast<float *>(smem);

        const float *Q_ptr = Q + (batch_idx * n_heads + head_idx) * head_dim;
        for (int d = tid; d < head_dim; d += num_threads)
        {
            Q_shared[d] = Q_ptr[d];
        }
        __syncthreads();

        float O_local[128] = {0};
        float m_local = -FLT_MAX;
        float l_local = 0.0f;

        const float *K_batch = K_cache + batch_idx * kv_stride * n_kv_heads * head_dim;
        const float *V_batch = V_cache + batch_idx * kv_stride * n_kv_heads * head_dim;

        for (int kv_pos = kv_start + tid; kv_pos < kv_end; kv_pos += num_threads)
        {
            const float *K_ptr = K_batch + kv_pos * n_kv_heads * head_dim + kv_head_idx * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++)
            {
                score += Q_shared[d] * K_ptr[d];
            }
            score *= softmax_scale;

            float m_new = fmaxf(m_local, score);
            float scale_old = expf(m_local - m_new);
            float p = expf(score - m_new);

            l_local = l_local * scale_old + p;
            for (int d = 0; d < head_dim; d++)
            {
                O_local[d] *= scale_old;
            }

            const float *V_ptr = V_batch + kv_pos * n_kv_heads * head_dim + kv_head_idx * head_dim;
            for (int d = 0; d < head_dim; d++)
            {
                O_local[d] += p * V_ptr[d];
            }

            m_local = m_new;
        }

        warpReduceSoftmaxState(m_local, l_local, O_local, head_dim);

        const int warp_id = tid / WARP_SIZE;
        const int lane_id = tid % WARP_SIZE;
        const int num_warps = num_threads / WARP_SIZE;

        __shared__ float block_m[8];
        __shared__ float block_l[8];
        __shared__ float block_O[8 * 128];

        if (lane_id == 0)
        {
            block_m[warp_id] = m_local;
            block_l[warp_id] = l_local;
            for (int d = 0; d < head_dim; d++)
            {
                block_O[warp_id * head_dim + d] = O_local[d];
            }
        }
        __syncthreads();

        if (warp_id == 0 && lane_id == 0)
        {
            float final_m = block_m[0];
            float final_l = block_l[0];
            float final_O[128];
            for (int d = 0; d < head_dim; d++)
            {
                final_O[d] = block_O[d];
            }

            for (int w = 1; w < num_warps; w++)
            {
                float other_m = block_m[w];
                float other_l = block_l[w];

                float m_new = fmaxf(final_m, other_m);
                float scale_self = expf(final_m - m_new);
                float scale_other = expf(other_m - m_new);

                final_l = scale_self * final_l + scale_other * other_l;

                for (int d = 0; d < head_dim; d++)
                {
                    final_O[d] = scale_self * final_O[d] +
                                 scale_other * block_O[w * head_dim + d];
                }

                final_m = m_new;
            }

            m_partial[partial_idx] = final_m;
            l_partial[partial_idx] = final_l;

            float *O_out = O_partial + partial_idx * head_dim;
            for (int d = 0; d < head_dim; d++)
            {
                O_out[d] = final_O[d];
            }
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
                float scale = expf(m_partial[base_idx + s] - m_max);
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
// Extern "C" Wrapper Functions
// =============================================================================

extern "C"
{
    /**
     * @brief Launch Flash Attention 2 pipelined kernel (SM >= 8.0)
     *
     * Dispatches to different kernel variants based on head_dim:
     *   - head_dim <= 64:  6 consumer warps, tile_q=96, 256 threads
     *   - head_dim <= 128: 4 consumer warps, tile_q=64, 192 threads
     *
     * Returns -1 on invalid input, -2 if GPU doesn't support SM 8.0.
     */
    int cudaFlashAttn_prefill_fa2(
        const float *Q, const float *K, const float *V, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask,
        void *stream)
    {
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

        float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

        // Get cached device config
        int device;
        cudaGetDevice(&device);
        const FA2DeviceConfig &dev_cfg = getFA2DeviceConfig(device);

        if (dev_cfg.sm_major < 8)
        {
            printf("[cudaFlashAttn_prefill_fa2] Error: SM %d.%d not supported, requires SM >= 8.0\n",
                   dev_cfg.sm_major, dev_cfg.sm_minor);
            return -2;
        }

        // Input validation
        if (head_dim <= 0)
        {
            printf("[cudaFlashAttn_prefill_fa2] Error: head_dim=%d must be positive\n", head_dim);
            return -1;
        }
        if (head_dim % 16 != 0)
        {
            printf("[cudaFlashAttn_prefill_fa2] Error: head_dim=%d must be multiple of 16 for WMMA\n", head_dim);
            return -1;
        }
        if (head_dim > 256)
        {
            printf("[cudaFlashAttn_prefill_fa2] Error: head_dim=%d exceeds maximum supported (256)\n", head_dim);
            return -1;
        }

        // Compute kernel configuration based on head_dim
        FA2KernelConfig cfg = computeFA2Config(head_dim, dev_cfg.max_smem_optin);

        // Grid: (n_heads, num_q_tiles, batch_size)
        int num_q_tiles = (seq_len + cfg.tile_q - 1) / cfg.tile_q;
        dim3 grid(n_heads, num_q_tiles, batch_size);

        cudaError_t err;

        // Dispatch to correct kernel variant based on consumer warp count
        if (cfg.num_consumer_warps == 6)
        {
            cudaFuncSetAttribute(flash_attention_2_pipelined_kernel<6>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 cfg.smem_size);

            flash_attention_2_pipelined_kernel<6><<<grid, cfg.block_size, cfg.smem_size, cuda_stream>>>(
                Q, K, V, O,
                batch_size, seq_len, kv_len,
                n_heads, n_kv_heads, head_dim,
                softmax_scale, causal, window_size, position_offset,
                device_params, mask,
                cfg.tile_q, cfg.tile_kv);
        }
        else
        {
            // 4 consumer warps for head_dim=128
            cudaFuncSetAttribute(flash_attention_2_pipelined_kernel<4>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 cfg.smem_size);

            flash_attention_2_pipelined_kernel<4><<<grid, cfg.block_size, cfg.smem_size, cuda_stream>>>(
                Q, K, V, O,
                batch_size, seq_len, kv_len,
                n_heads, n_kv_heads, head_dim,
                softmax_scale, causal, window_size, position_offset,
                device_params, mask,
                cfg.tile_q, cfg.tile_kv);
        }

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("[cudaFlashAttn_prefill_fa2] CUDA error: %s (smem=%zu bytes, tile_q=%d, tile_kv=%d, head_dim=%d, "
                   "consumer_warps=%d, max_smem=%d, grid=(%d,%d,%d), block=%d)\n",
                   cudaGetErrorString(err), cfg.smem_size, cfg.tile_q, cfg.tile_kv, head_dim,
                   cfg.num_consumer_warps, dev_cfg.max_smem_optin,
                   grid.x, grid.y, grid.z, cfg.block_size);
            return -1;
        }
        return 0;
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
        void *stream)
    {
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
                num_splits, softmax_scale, device_params);
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
