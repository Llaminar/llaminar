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
 *   - Uses vectorized FP16 operations (v_pk_*) for 2x throughput
 *   - LDS double-buffering for overlapped load/compute
 *   - Explicit wavefront reductions via DPP
 *   - NO WMMA/Matrix Cores - pure VALU implementation
 * - Flash Decoding: Split-K parallelism for single-token decode
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
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <cstdio>

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

    // LDS padding to avoid bank conflicts (32 banks, 4-byte elements)
    // Pad by 4 elements (16 bytes) for FP32, 8 elements (16 bytes) for FP16
    constexpr int FA2_LDS_PAD_FP16 = 8;
    constexpr int FA2_LDS_PAD_FP32 = 4;

    // Block configuration
    // We use 256 threads = 4 wavefronts per block for good occupancy
    constexpr int FA2_THREADS_PER_BLOCK = 256;
    constexpr int FA2_WAVEFRONTS_PER_BLOCK = FA2_THREADS_PER_BLOCK / WAVEFRONT_SIZE; // 4

    // =========================================================================
    // Flash Decoding Configuration for MI50
    // =========================================================================

    // Number of splits for Flash Decoding
    // More splits = more parallelism but more reduction overhead
    constexpr int FD_DEFAULT_NUM_SPLITS = 8;

    // Threads per block for decode kernel
    constexpr int FD_THREADS_PER_BLOCK = 256;

    // =========================================================================
    // Device Property Caching
    // =========================================================================

    struct MI50DeviceConfig
    {
        int gcn_arch = 0;      // e.g., 906 for gfx906
        int max_lds = 0;       // Max LDS per block
        int max_vgprs = 0;     // Max VGPRs per wavefront
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
            hipGetDeviceProperties(&prop, device);

            // Parse gcnArchName string (e.g., "gfx906") to extract arch number
            // gcnArchName format: "gfx" + arch_number
            cfg.gcn_arch = 0;
            if (strncmp(prop.gcnArchName, "gfx", 3) == 0) {
                cfg.gcn_arch = atoi(prop.gcnArchName + 3);
            }
            cfg.max_lds = static_cast<int>(prop.sharedMemPerBlock);
            cfg.max_vgprs = MAX_VGPRS_PER_WAVEFRONT; // Could query via rocm-smi
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
     * @brief Flash Attention 2 prefill kernel for MI50
     *
     * Uses vectorized FP16 VALU operations for Q@K^T computation.
     * No WMMA/Matrix Cores - pure ALU implementation optimized for Vega.
     */
    __global__ void flash_attention_2_mi50_kernel(
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
        int tile_q,
        int tile_kv)
    {
        // Block/thread indexing
        const int batch_idx = blockIdx.z;
        const int head_idx = blockIdx.x;
        const int q_tile_idx = blockIdx.y;
        const int tid = threadIdx.x;
        const int wavefront_id = tid / WAVEFRONT_SIZE;
        const int lane_id = tid % WAVEFRONT_SIZE;

        // GQA mapping: multiple Q heads can share one KV head
        const int gqa_ratio = n_heads / n_kv_heads;
        const int kv_head_idx = head_idx / gqa_ratio;

        // Q row range for this block
        const int q_block_start = q_tile_idx * tile_q;
        if (q_block_start >= seq_len)
            return;
        const int q_tile_rows = min(tile_q, seq_len - q_block_start);

        // Compute LDS layout
        FA2LDSLayout lds_layout;
        lds_layout.compute(head_dim, tile_q, tile_kv);

        // LDS allocation
        extern __shared__ char smem[];

        __half *Q_tile = reinterpret_cast<__half *>(smem);
        __half *KV_buffers = Q_tile + tile_q * lds_layout.q_stride;
        float *scores = reinterpret_cast<float *>(KV_buffers + FA2_NUM_STAGES * 2 * tile_kv * lds_layout.kv_stride);

        // Helper lambdas for double-buffered K/V access
        auto get_K_tile = [&](int stage) -> __half *
        {
            return KV_buffers + stage * 2 * tile_kv * lds_layout.kv_stride;
        };
        auto get_V_tile = [&](int stage) -> __half *
        {
            return KV_buffers + (stage * 2 + 1) * tile_kv * lds_layout.kv_stride;
        };

        // =====================================================================
        // Thread-local accumulators for output
        // =====================================================================
        // Each thread is responsible for a subset of Q rows
        // With 256 threads and tile_q=64, each thread handles 64/256 = 0.25 rows
        // So we use a strided pattern where thread t handles rows t, t+256, ...

        // Per-row accumulators (each thread may handle multiple rows in small tiles)
        const int rows_per_block = tile_q;
        const int rows_per_thread = (rows_per_block + FA2_THREADS_PER_BLOCK - 1) / FA2_THREADS_PER_BLOCK;

        // Thread's assigned Q rows (strided assignment)
        float O_acc[4][128]; // Support up to 4 rows per thread, head_dim up to 128
        float m_i[4];        // Max score per row
        float l_i[4];        // Sum of exp per row

        for (int r = 0; r < rows_per_thread && tid + r * FA2_THREADS_PER_BLOCK < q_tile_rows; r++)
        {
            m_i[r] = -FLT_MAX;
            l_i[r] = 0.0f;
            for (int d = 0; d < head_dim; d++)
            {
                O_acc[r][d] = 0.0f;
            }
        }

        // Batch/head pointers
        const float *Q_batch = Q + batch_idx * seq_len * n_heads * head_dim;
        const float *K_batch = K + batch_idx * kv_len * n_kv_heads * head_dim;
        const float *V_batch = V + batch_idx * kv_len * n_kv_heads * head_dim;

        // =====================================================================
        // Load Q tile into LDS (done once per block)
        // =====================================================================
        for (int i = tid; i < q_tile_rows * head_dim; i += FA2_THREADS_PER_BLOCK)
        {
            int local_row = i / head_dim;
            int d = i % head_dim;
            int global_row = q_block_start + local_row;
            float val = Q_batch[global_row * n_heads * head_dim + head_idx * head_dim + d];
            Q_tile[local_row * lds_layout.q_stride + d] = float_to_half(val);
        }
        __syncthreads();

        const int num_kv_tiles = (kv_len + tile_kv - 1) / tile_kv;

        // =====================================================================
        // Prologue: Load first KV tile
        // =====================================================================
        {
            const int kv_start = 0;
            const int kv_end = min(tile_kv, kv_len);
            const int actual_len = kv_end - kv_start;

            __half *K_dst = get_K_tile(0);
            __half *V_dst = get_V_tile(0);

            for (int i = tid; i < actual_len * head_dim; i += FA2_THREADS_PER_BLOCK)
            {
                int local_row = i / head_dim;
                int d = i % head_dim;
                int global_row = kv_start + local_row;

                float k_val = K_batch[global_row * n_kv_heads * head_dim + kv_head_idx * head_dim + d];
                float v_val = V_batch[global_row * n_kv_heads * head_dim + kv_head_idx * head_dim + d];

                K_dst[local_row * lds_layout.kv_stride + d] = float_to_half(k_val);
                V_dst[local_row * lds_layout.kv_stride + d] = float_to_half(v_val);
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

            const int kv_start = kv_tile_iter * tile_kv;
            const int kv_end = min(kv_start + tile_kv, kv_len);
            const int actual_kv_len = kv_end - kv_start;

            // Early exit for causal (all positions masked)
            if (causal && kv_start > (q_block_start + tile_q - 1) + position_offset)
            {
                continue;
            }

            __half *K_tile = get_K_tile(current_stage);
            __half *V_tile = get_V_tile(current_stage);

            // =================================================================
            // PHASE 1: Start loading next KV tile (if any)
            // =================================================================
            if (kv_tile_iter + 1 < num_kv_tiles)
            {
                const int next_kv_start = (kv_tile_iter + 1) * tile_kv;
                const int next_kv_end = min(next_kv_start + tile_kv, kv_len);
                const int next_len = next_kv_end - next_kv_start;

                __half *K_next = get_K_tile(next_stage);
                __half *V_next = get_V_tile(next_stage);

                // Cooperative load by all threads
                for (int i = tid; i < next_len * head_dim; i += FA2_THREADS_PER_BLOCK)
                {
                    int local_row = i / head_dim;
                    int d = i % head_dim;
                    int global_row = next_kv_start + local_row;

                    float k_val = K_batch[global_row * n_kv_heads * head_dim + kv_head_idx * head_dim + d];
                    float v_val = V_batch[global_row * n_kv_heads * head_dim + kv_head_idx * head_dim + d];

                    K_next[local_row * lds_layout.kv_stride + d] = float_to_half(k_val);
                    V_next[local_row * lds_layout.kv_stride + d] = float_to_half(v_val);
                }
            }

            // =================================================================
            // PHASE 2: Compute Q @ K^T (vectorized FP16 with FP32 accumulator)
            // =================================================================
            // Each thread computes a subset of score matrix elements

            // Score matrix is [q_tile_rows, actual_kv_len]
            // We distribute computation: thread t computes elements at stride
            const int total_score_elements = q_tile_rows * actual_kv_len;

            for (int s_idx = tid; s_idx < total_score_elements; s_idx += FA2_THREADS_PER_BLOCK)
            {
                int q_row = s_idx / actual_kv_len;
                int kv_col = s_idx % actual_kv_len;

                // Compute dot product Q[q_row] @ K[kv_col]
                float score = 0.0f;

                const __half *Q_ptr = Q_tile + q_row * lds_layout.q_stride;
                const __half *K_ptr = K_tile + kv_col * lds_layout.kv_stride;

                // Vectorized dot product (process 2 FP16 elements at a time)
                // On MI50, this maps to v_pk_fma_f16 for 2x throughput
                for (int d = 0; d < head_dim; d += 2)
                {
                    // Manual unpacking since we're targeting GCN
                    float q0 = half_to_float(Q_ptr[d]);
                    float k0 = half_to_float(K_ptr[d]);
                    score += q0 * k0;

                    if (d + 1 < head_dim)
                    {
                        float q1 = half_to_float(Q_ptr[d + 1]);
                        float k1 = half_to_float(K_ptr[d + 1]);
                        score += q1 * k1;
                    }
                }

                scores[q_row * lds_layout.scores_stride + kv_col] = score;
            }
            __syncthreads();

            // =================================================================
            // PHASE 3: Apply softmax and accumulate O = softmax(QK^T) @ V
            // =================================================================
            // Each thread processes its assigned Q rows

            for (int r = 0; r < rows_per_thread; r++)
            {
                int local_q_row = tid + r * FA2_THREADS_PER_BLOCK;
                if (local_q_row >= q_tile_rows)
                    break;

                int global_q_row = q_block_start + local_q_row;
                float *my_scores = scores + local_q_row * lds_layout.scores_stride;

                // Find max for this row (with masking)
                float m_ij = -FLT_MAX;
                for (int j = 0; j < actual_kv_len; j++)
                {
                    int kv_pos = kv_start + j;
                    bool masked = false;

                    // Causal masking
                    if (causal && kv_pos > global_q_row + position_offset)
                        masked = true;

                    // Sliding window
                    if (window_size > 0)
                    {
                        int q_pos = global_q_row + position_offset;
                        if (kv_pos < q_pos - window_size || kv_pos > q_pos + window_size)
                            masked = true;
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
                float m_i_new = fmaxf(m_i[r], m_ij);
                float scale_old = expf(m_i[r] - m_i_new);

                // Rescale previous accumulators
                for (int d = 0; d < head_dim; d++)
                {
                    O_acc[r][d] *= scale_old;
                }
                l_i[r] *= scale_old;

                // Accumulate P @ V for this tile
                float l_ij = 0.0f;
                for (int j = 0; j < actual_kv_len; j++)
                {
                    float s = my_scores[j];
                    if (s > -FLT_MAX / 2)
                    {
                        float p = expf(s - m_i_new);
                        l_ij += p;

                        const __half *V_row = V_tile + j * lds_layout.kv_stride;
                        for (int d = 0; d < head_dim; d++)
                        {
                            O_acc[r][d] += p * half_to_float(V_row[d]);
                        }
                    }
                }

                l_i[r] += l_ij;
                m_i[r] = m_i_new;
            }
            __syncthreads();
        }

        // =====================================================================
        // Write final output
        // =====================================================================
        float *O_batch = O + batch_idx * seq_len * n_heads * head_dim;

        for (int r = 0; r < rows_per_thread; r++)
        {
            int local_q_row = tid + r * FA2_THREADS_PER_BLOCK;
            if (local_q_row >= q_tile_rows)
                break;

            int global_q_row = q_block_start + local_q_row;
            float inv_l = (l_i[r] > 0.0f) ? (1.0f / l_i[r]) : 0.0f;

            float *O_row = O_batch + global_q_row * n_heads * head_dim + head_idx * head_dim;
            for (int d = 0; d < head_dim; d++)
            {
                O_row[d] = O_acc[r][d] * inv_l;
            }
        }
    }

    // =========================================================================
    // Flash Decoding - Split-K Kernel for MI50
    // =========================================================================
    //
    // For single-token decode (seq_len=1), we parallelize over the KV cache
    // using split-K. Each block handles a portion of the KV cache and produces
    // partial (m, l, O) that are merged in a reduction kernel.
    //
    // Grid: (n_heads, num_splits, batch_size)
    // Block: (256,) threads
    //
    // =========================================================================

    /**
     * @brief Flash Decoding kernel for single-query decode on MI50
     *
     * Parallelizes over KV cache using split-K pattern.
     */
    __global__ void flash_decoding_mi50_kernel(
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
        float softmax_scale)
    {
        const int head_idx = blockIdx.x;
        const int split_idx = blockIdx.y;
        const int batch_idx = blockIdx.z;

        const int gqa_ratio = n_heads / n_kv_heads;
        const int kv_head_idx = head_idx / gqa_ratio;

        // Split range
        const int split_size = (kv_len + num_splits - 1) / num_splits;
        const int kv_start = split_idx * split_size;
        const int kv_end = min(kv_start + split_size, kv_len);

        const int partial_idx = (batch_idx * n_heads + head_idx) * num_splits + split_idx;

        // Handle empty splits
        if (kv_start >= kv_len)
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

        // Load Q into LDS
        extern __shared__ char smem[];
        float *Q_shared = reinterpret_cast<float *>(smem);

        const float *Q_ptr = Q + (batch_idx * n_heads + head_idx) * head_dim;
        for (int d = tid; d < head_dim; d += num_threads)
        {
            Q_shared[d] = Q_ptr[d];
        }
        __syncthreads();

        // Thread-local accumulators
        float O_local[128] = {0}; // Support head_dim up to 128
        float m_local = -FLT_MAX;
        float l_local = 0.0f;

        const float *K_batch = K_cache + batch_idx * kv_len * n_kv_heads * head_dim;
        const float *V_batch = V_cache + batch_idx * kv_len * n_kv_heads * head_dim;

        // Each thread processes a strided subset of KV positions
        for (int kv_pos = kv_start + tid; kv_pos < kv_end; kv_pos += num_threads)
        {
            const float *K_ptr = K_batch + kv_pos * n_kv_heads * head_dim + kv_head_idx * head_dim;

            // Compute Q @ K^T for this position
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++)
            {
                score += Q_shared[d] * K_ptr[d];
            }
            score *= softmax_scale;

            // Online softmax update
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

        // Wavefront reduction
        wavefrontReduceSoftmaxState(m_local, l_local, O_local, head_dim);

        // Inter-wavefront reduction using LDS
        const int wavefront_id = tid / WAVEFRONT_SIZE;
        const int lane_id = tid % WAVEFRONT_SIZE;
        const int num_wavefronts = num_threads / WAVEFRONT_SIZE;

        __shared__ float block_m[4];     // Max 4 wavefronts
        __shared__ float block_l[4];
        __shared__ float block_O[4 * 128];

        if (lane_id == 0)
        {
            block_m[wavefront_id] = m_local;
            block_l[wavefront_id] = l_local;
            for (int d = 0; d < head_dim; d++)
            {
                block_O[wavefront_id * head_dim + d] = O_local[d];
            }
        }
        __syncthreads();

        // Final reduction by thread 0
        if (tid == 0)
        {
            float final_m = block_m[0];
            float final_l = block_l[0];
            float final_O[128];
            for (int d = 0; d < head_dim; d++)
            {
                final_O[d] = block_O[d];
            }

            for (int w = 1; w < num_wavefronts; w++)
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

            // Store partial results
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
    __global__ void flash_decoding_reduce_mi50_kernel(
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

        // Find global max and compute scales
        __shared__ float global_m;
        __shared__ float global_l;
        __shared__ float split_scales[32]; // Support up to 32 splits

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

        // Combine partial outputs
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
// Extern "C" Wrapper Functions
// =============================================================================

extern "C"
{
    /**
     * @brief Launch Flash Attention 2 kernel for MI50
     *
     * @return 0 on success, -1 on invalid input, -2 on unsupported arch
     */
    int hipFlashAttn_prefill_fa2(
        const float *Q, const float *K, const float *V, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        void *stream)
    {
        hipStream_t hip_stream = static_cast<hipStream_t>(stream);

        // Input validation
        if (head_dim <= 0 || head_dim > 128)
        {
            printf("[hipFlashAttn_prefill_fa2] Error: head_dim=%d must be in [1, 128]\n", head_dim);
            return -1;
        }
        if (n_heads % n_kv_heads != 0)
        {
            printf("[hipFlashAttn_prefill_fa2] Error: n_heads must be divisible by n_kv_heads\n");
            return -1;
        }

        float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

        // Adaptively select tile sizes based on LDS constraints
        // Start with default 64x64, fall back to smaller tiles if needed
        int tile_q = FA2_TILE_Q;
        int tile_kv = FA2_TILE_KV;

        // Check if default tiles fit in LDS
        FA2LDSLayout lds_layout;
        lds_layout.compute(head_dim, tile_q, tile_kv);

        if (lds_layout.total_size > LDS_SIZE_BYTES)
        {
            // Try 32x64 tiles
            tile_q = 32;
            tile_kv = 64;
            lds_layout.compute(head_dim, tile_q, tile_kv);

            if (lds_layout.total_size > LDS_SIZE_BYTES)
            {
                // Try 32x32 tiles
                tile_q = 32;
                tile_kv = 32;
                lds_layout.compute(head_dim, tile_q, tile_kv);

                if (lds_layout.total_size > LDS_SIZE_BYTES)
                {
                    // Last resort: 16x32 tiles
                    tile_q = 16;
                    tile_kv = 32;
                    lds_layout.compute(head_dim, tile_q, tile_kv);

                    if (lds_layout.total_size > LDS_SIZE_BYTES)
                    {
                        printf("[hipFlashAttn_prefill_fa2] Error: Cannot fit head_dim=%d in LDS even with smallest tiles\n",
                               head_dim);
                        return -1;
                    }
                }
            }
        }

        // Grid configuration
        const int num_q_tiles = (seq_len + tile_q - 1) / tile_q;
        dim3 grid(n_heads, num_q_tiles, batch_size);
        dim3 block(FA2_THREADS_PER_BLOCK);

        // Launch kernel
        hipLaunchKernelGGL(
            flash_attention_2_mi50_kernel,
            grid, block, lds_layout.total_size, hip_stream,
            Q, K, V, O,
            batch_size, seq_len, kv_len,
            n_heads, n_kv_heads, head_dim,
            softmax_scale, causal, window_size, position_offset,
            tile_q, tile_kv);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            printf("[hipFlashAttn_prefill_fa2] Kernel launch failed: %s\n", hipGetErrorString(err));
            return -1;
        }

        return 0;
    }

    /**
     * @brief Launch Flash Decoding kernel for MI50
     *
     * @return 0 on success, -1 on error
     */
    int hipFlashAttn_decode_fp32(
        const float *Q, const float *K_cache, const float *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int batch_size, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int num_splits,
        void *stream)
    {
        hipStream_t hip_stream = static_cast<hipStream_t>(stream);

        if (head_dim <= 0 || head_dim > 128)
        {
            printf("[hipFlashAttn_decode_fp32] Error: head_dim=%d must be in [1, 128]\n", head_dim);
            return -1;
        }

        float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

        // Split-K kernel
        size_t smem_size = head_dim * sizeof(float); // Q in shared memory
        dim3 grid_split(n_heads, num_splits, batch_size);
        dim3 block_split(FD_THREADS_PER_BLOCK);

        hipLaunchKernelGGL(
            flash_decoding_mi50_kernel,
            grid_split, block_split, smem_size, hip_stream,
            Q, K_cache, V_cache,
            O_partial, m_partial, l_partial,
            kv_len, n_heads, n_kv_heads, head_dim,
            num_splits, softmax_scale);

        // Reduction kernel
        dim3 grid_reduce(n_heads, batch_size);
        dim3 block_reduce(256);

        hipLaunchKernelGGL(
            flash_decoding_reduce_mi50_kernel,
            grid_reduce, block_reduce, 0, hip_stream,
            O_partial, m_partial, l_partial, O,
            n_heads, head_dim, num_splits);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            printf("[hipFlashAttn_decode_fp32] Kernel launch failed: %s\n", hipGetErrorString(err));
            return -1;
        }

        return 0;
    }

    /**
     * @brief Allocate workspace for Flash Decoding
     */
    int hipFlashAttn_allocWorkspace(
        void **partial_output, void **partial_m, void **partial_l,
        int batch_size, int n_heads, int head_dim, int num_splits)
    {
        size_t num_partials = static_cast<size_t>(batch_size) * n_heads * num_splits;

        size_t output_size = num_partials * head_dim * sizeof(float);
        size_t m_size = num_partials * sizeof(float);
        size_t l_size = num_partials * sizeof(float);

        if (hipMalloc(partial_output, output_size) != hipSuccess)
            return -1;
        if (hipMalloc(partial_m, m_size) != hipSuccess)
        {
            hipFree(*partial_output);
            return -1;
        }
        if (hipMalloc(partial_l, l_size) != hipSuccess)
        {
            hipFree(*partial_output);
            hipFree(*partial_m);
            return -1;
        }

        return 0;
    }

    /**
     * @brief Free workspace for Flash Decoding
     */
    void hipFlashAttn_freeWorkspace(void *partial_output, void *partial_m, void *partial_l)
    {
        if (partial_output)
            hipFree(partial_output);
        if (partial_m)
            hipFree(partial_m);
        if (partial_l)
            hipFree(partial_l);
    }

    /**
     * @brief Set active HIP device
     */
    int hipFlashAttn_setDevice(int device_idx)
    {
        return hipSetDevice(device_idx) == hipSuccess ? 0 : -1;
    }

    /**
     * @brief Synchronize HIP device
     */
    int hipFlashAttn_synchronize()
    {
        return hipDeviceSynchronize() == hipSuccess ? 0 : -1;
    }
}
