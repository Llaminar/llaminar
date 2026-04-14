/**
 * @file CUDATurboQuantKernels.cu
 * @brief CUDA kernel implementations for TurboQuant KV cache operations
 * @author David Sanftenberg
 *
 * Implements TQ8/TQ4 quantize and dequantize kernels with optional RoPE fusion.
 * Codebooks are stored in __constant__ memory for fast cache-resident access.
 *
 * Threading strategy:
 * - TQ8 quantize/dequant: 1 thread block per (token, head) pair
 *   Each block has `head_dim` threads (64 or 128)
 * - TQ4 quantize/dequant: Same, with bit-packing in shared memory
 * - RoPE: Element-wise, 2 elements per thread (cos/sin pair)
 */

#include "CUDATurboQuantKernels.h"
#include "../../../kernels/cpu/turboquant/TurboQuantCodebook.h"
#include "../../../kernels/cpu/turboquant/TurboQuantContext.h"
#include "../../../utils/Logger.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <type_traits>
#include <cmath>
#include <cstring>
#include <atomic>

namespace llaminar2
{

    // =========================================================================
    // Constant Memory Codebooks
    // =========================================================================

    __constant__ float d_TQ8_CENTROIDS[256];
    __constant__ float d_TQ8_THRESHOLDS[255];
    __constant__ float d_TQ4_CENTROIDS[16];
    __constant__ float d_TQ4_THRESHOLDS[15];

    // Precomputed RoPE frequency table: freq[i] = 1/theta^(2i/D)
    // Max head_dim = 128, half = 64 pairs
    __constant__ float d_ROPE_FREQS[128];
    static std::atomic<bool> s_rope_freqs_uploaded{false};
    static float s_rope_theta_cached = 0.0f;
    static int s_rope_head_dim_cached = 0;

    static std::atomic<bool> s_codebooks_uploaded{false};

    void cuda_tq_upload_rope_freqs(float rope_theta, int head_dim, cudaStream_t stream)
    {
        // Only recompute if theta or head_dim changed
        if (s_rope_freqs_uploaded.load(std::memory_order_acquire) &&
            s_rope_theta_cached == rope_theta && s_rope_head_dim_cached == head_dim)
            return;

        const int half = head_dim / 2;
        float host_freqs[128] = {};
        for (int i = 0; i < half; ++i)
            host_freqs[i] = 1.0f / powf(rope_theta, static_cast<float>(2 * i) / static_cast<float>(head_dim));

        cudaMemcpyToSymbolAsync(d_ROPE_FREQS, host_freqs,
                                half * sizeof(float), 0, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
        if (cudaGetLastError() == cudaSuccess)
        {
            s_rope_theta_cached = rope_theta;
            s_rope_head_dim_cached = head_dim;
            s_rope_freqs_uploaded.store(true, std::memory_order_release);
        }
    }

    void cuda_tq_upload_codebooks(cudaStream_t stream)
    {
        if (s_codebooks_uploaded.load(std::memory_order_acquire))
            return;

        cudaMemcpyToSymbolAsync(d_TQ8_CENTROIDS, TQ8_CENTROIDS.data(),
                                256 * sizeof(float), 0, cudaMemcpyHostToDevice, stream);
        cudaMemcpyToSymbolAsync(d_TQ8_THRESHOLDS, TQ8_THRESHOLDS.data(),
                                255 * sizeof(float), 0, cudaMemcpyHostToDevice, stream);
        cudaMemcpyToSymbolAsync(d_TQ4_CENTROIDS, TQ4_CENTROIDS.data(),
                                16 * sizeof(float), 0, cudaMemcpyHostToDevice, stream);
        cudaMemcpyToSymbolAsync(d_TQ4_THRESHOLDS, TQ4_THRESHOLDS.data(),
                                15 * sizeof(float), 0, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);

        if (cudaGetLastError() == cudaSuccess)
        {
            s_codebooks_uploaded.store(true, std::memory_order_release);
        }
    }

    // =========================================================================
    // Rotation Matrix Management
    // =========================================================================

    CUDATurboQuantRotations cuda_tq_create_rotations(
        int n_layers, int n_kv_heads, int head_dim,
        uint64_t rotation_seed, int device_id,
        cudaStream_t stream)
    {
        cudaSetDevice(device_id);

        CUDATurboQuantRotations result;
        result.n_layers = n_layers;
        result.n_kv_heads = n_kv_heads;
        result.head_dim = head_dim;

        const size_t mat_size = static_cast<size_t>(head_dim) * head_dim;
        const size_t total_mats = static_cast<size_t>(n_layers) * n_kv_heads;
        const size_t total_floats = total_mats * mat_size;

        // Allocate on GPU
        cudaMalloc(&result.d_rotations, total_floats * sizeof(float));
        cudaMalloc(&result.d_rotations_t, total_floats * sizeof(float));

        // Generate and upload each rotation matrix
        // Use the same derivation as CPU: TurboQuantContext → for_layer(head_idx)
        // The TurboQuantContext for each layer is derived from the base context
        TurboQuantContext base_ctx(head_dim, rotation_seed == 0 ? 31ULL : rotation_seed);

        std::vector<float> host_rot(total_floats);
        std::vector<float> host_rot_t(total_floats);

        for (int layer = 0; layer < n_layers; ++layer)
        {
            const auto &layer_ctx = base_ctx.for_layer(layer);

            for (int head = 0; head < n_kv_heads; ++head)
            {
                const auto &head_ctx = layer_ctx.rotation();
                // Each head within a layer uses for_layer(head) on the layer context
                TurboQuantContext head_derived(head_dim, 0);
                const auto &actual_ctx = layer_ctx.for_layer(head);
                const auto &rot = actual_ctx.rotation();

                const size_t offset = (static_cast<size_t>(layer) * n_kv_heads + head) * mat_size;

                // Copy rotation matrix (row-major)
                std::memcpy(host_rot.data() + offset, rot.matrix.data(), mat_size * sizeof(float));

                // Transpose for dequant
                for (int r = 0; r < head_dim; ++r)
                {
                    for (int c = 0; c < head_dim; ++c)
                    {
                        host_rot_t[offset + r * head_dim + c] = rot.matrix[c * head_dim + r];
                    }
                }
            }
        }

        cudaMemcpyAsync(result.d_rotations, host_rot.data(),
                        total_floats * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(result.d_rotations_t, host_rot_t.data(),
                        total_floats * sizeof(float), cudaMemcpyHostToDevice, stream);

        return result;
    }

    void cuda_tq_free_rotations(CUDATurboQuantRotations &rotations)
    {
        if (rotations.d_rotations)
        {
            cudaFree(rotations.d_rotations);
            rotations.d_rotations = nullptr;
        }
        if (rotations.d_rotations_t)
        {
            cudaFree(rotations.d_rotations_t);
            rotations.d_rotations_t = nullptr;
        }
    }

    // =========================================================================
    // Device Helpers
    // =========================================================================

    /**
     * @brief Warp-level reduction for sum.
     */
    __device__ inline float warp_reduce_sum(float val)
    {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        return val;
    }

    /**
     * @brief Block-level reduction for sum using shared memory.
     * Assumes blockDim.x threads participate.
     */
    __device__ float block_reduce_sum(float val, float *shared)
    {
        const int lane = threadIdx.x % warpSize;
        const int warp_id = threadIdx.x / warpSize;
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;

        val = warp_reduce_sum(val);

        if (lane == 0)
            shared[warp_id] = val;
        __syncthreads();

        val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
        if (warp_id == 0)
            val = warp_reduce_sum(val);

        return val;
    }

    // =========================================================================
    // TQ8 Quantize Kernel
    // =========================================================================

    /**
     * @brief Quantize one head of one token: FP32 → TQ8Block
     *
     * Grid:  (num_tokens, n_kv_heads)
     * Block: (head_dim) — one thread per element
     *
     * Steps per block:
     * 1. Load head_dim floats from input
     * 2. Compute L2 norm (block reduction)
     * 3. Normalize × √D
     * 4. Apply rotation Π (matrix-vector: each thread computes one output)
     * 5. Find nearest TQ8 centroid (binary search per thread)
     * 6. Write TQ8Block (norm + indices)
     */
    template <int D>
    __global__ void tq8_quantize_kernel(
        const float *__restrict__ d_input,     // [num_tokens, n_kv_heads * D]
        const float *__restrict__ d_rotations, // [n_kv_heads * D * D]
        uint8_t *__restrict__ d_output_bytes,  // raw bytes for TQ8Block<D> array
        int n_kv_heads)
    {
        const int token = blockIdx.x;
        const int head = blockIdx.y;
        const int tid = threadIdx.x;

        if (tid >= D)
            return;

        // Load input element
        const int kv_dim = n_kv_heads * D;
        const float x = d_input[token * kv_dim + head * D + tid];

        // Compute L2 norm via block reduction
        __shared__ float s_reduce[32]; // for warp reduce
        __shared__ float s_norm;
        __shared__ float s_combined_scale;

        float norm_sq = block_reduce_sum(x * x, s_reduce);

        if (tid == 0)
        {
            float norm = sqrtf(norm_sq);
            s_norm = norm;
            s_combined_scale = (norm > 1e-30f) ? (sqrtf(static_cast<float>(D)) / norm) : 0.0f;
        }
        __syncthreads();

        // Normalize and scale
        float scaled = x * s_combined_scale;

        // Rotation: out[tid] = Σ_j Π[tid][j] * scaled_input[j]
        // Each thread computes one output element via dot product with one row of Π
        __shared__ float s_scaled[128]; // max head_dim = 128
        s_scaled[tid] = scaled;
        __syncthreads();

        const float *rot_row = d_rotations + static_cast<size_t>(head) * D * D + tid * D;
        float rotated = 0.0f;
        for (int j = 0; j < D; ++j)
        {
            rotated += rot_row[j] * s_scaled[j];
        }

        // Binary search for nearest TQ8 centroid
        int idx = 0;
// 7 binary search steps for 255 thresholds
#define TQ8_GPU_BSEARCH(STEP, OFFSET)             \
    if (rotated > d_TQ8_THRESHOLDS[idx + OFFSET]) \
        idx += STEP;
        TQ8_GPU_BSEARCH(128, 127)
        TQ8_GPU_BSEARCH(64, 63)
        TQ8_GPU_BSEARCH(32, 31)
        TQ8_GPU_BSEARCH(16, 15)
        TQ8_GPU_BSEARCH(8, 7)
        TQ8_GPU_BSEARCH(4, 3)
        TQ8_GPU_BSEARCH(2, 1)
        if (rotated > d_TQ8_THRESHOLDS[idx])
            idx += 1;
#undef TQ8_GPU_BSEARCH

        // Write TQ8Block: [norm, residual_norm, indices[D]]
        // Layout: norm(4B) + residual_norm(4B) + indices(D bytes)
        const size_t block_size = sizeof(TQ8Block<D>);
        uint8_t *block_ptr = d_output_bytes + (static_cast<size_t>(token) * n_kv_heads + head) * block_size;

        if (tid == 0)
        {
            // Write norm and residual_norm
            float *norms = reinterpret_cast<float *>(block_ptr);
            norms[0] = s_norm;
            norms[1] = -1.0f; // sentinel for scalar-full mode
        }

        // Write index
        block_ptr[2 * sizeof(float) + tid] = static_cast<uint8_t>(idx);
    }

    // =========================================================================
    // TQ4 Quantize Kernel
    // =========================================================================

    /**
     * @brief Quantize one head of one token: FP32 → TQ4Block
     *
     * Grid:  (num_tokens, n_kv_heads)
     * Block: (head_dim) — one thread per element
     *
     * TQ4 uses 4-bit indices packed as 3-bit MSE + 1 high bit.
     * The packing is done via shared memory.
     */
    template <int D>
    __global__ void tq4_quantize_kernel(
        const float *__restrict__ d_input,
        const float *__restrict__ d_rotations,
        uint8_t *__restrict__ d_output_bytes,
        int n_kv_heads)
    {
        const int token = blockIdx.x;
        const int head = blockIdx.y;
        const int tid = threadIdx.x;

        if (tid >= D)
            return;

        const int kv_dim = n_kv_heads * D;
        const float x = d_input[token * kv_dim + head * D + tid];

        // Compute norm
        __shared__ float s_reduce[32];
        __shared__ float s_norm;
        __shared__ float s_combined_scale;

        float norm_sq = block_reduce_sum(x * x, s_reduce);

        if (tid == 0)
        {
            float norm = sqrtf(norm_sq);
            s_norm = norm;
            s_combined_scale = (norm > 1e-30f) ? (sqrtf(static_cast<float>(D)) / norm) : 0.0f;
        }
        __syncthreads();

        float scaled = x * s_combined_scale;

        // Rotation
        __shared__ float s_scaled[128];
        s_scaled[tid] = scaled;
        __syncthreads();

        const float *rot_row = d_rotations + static_cast<size_t>(head) * D * D + tid * D;
        float rotated = 0.0f;
        for (int j = 0; j < D; ++j)
            rotated += rot_row[j] * s_scaled[j];

        // Binary search for nearest TQ4 centroid (15 thresholds)
        int idx = 0;
#define TQ4_GPU_BSEARCH(STEP, OFFSET)             \
    if (rotated > d_TQ4_THRESHOLDS[idx + OFFSET]) \
        idx += STEP;
        TQ4_GPU_BSEARCH(8, 7)
        TQ4_GPU_BSEARCH(4, 3)
        TQ4_GPU_BSEARCH(2, 1)
        if (rotated > d_TQ4_THRESHOLDS[idx])
            idx += 1;
#undef TQ4_GPU_BSEARCH

        // TQ4 packing: 3 low bits go to mse_indices (packed 8→3 bytes),
        // 1 high bit goes to high_bits (packed 8→1 byte)
        // idx is 0-15, so low3 = idx & 0x7, high1 = idx >> 3
        __shared__ uint8_t s_indices[128]; // full 4-bit indices
        s_indices[tid] = static_cast<uint8_t>(idx);
        __syncthreads();

        // Write TQ4Block: [norm(4B), residual_norm(4B), mse_indices[D*3/8], high_bits[D/8]]
        constexpr size_t MSE_BYTES = D * 3 / 8;
        constexpr size_t HIGH_BIT_BYTES = D / 8;
        const size_t block_size = sizeof(TQ4Block<D>);
        uint8_t *block_ptr = d_output_bytes + (static_cast<size_t>(token) * n_kv_heads + head) * block_size;

        if (tid == 0)
        {
            float *norms = reinterpret_cast<float *>(block_ptr);
            norms[0] = s_norm;
            norms[1] = -1.0f;

            uint8_t *mse_ptr = block_ptr + 2 * sizeof(float);
            uint8_t *high_ptr = mse_ptr + MSE_BYTES;

            // Pack 3-bit indices (8 indices → 3 bytes)
            for (int i = 0; i < D; i += 8)
            {
                uint8_t low3[8];
                uint8_t high1 = 0;
                for (int j = 0; j < 8; ++j)
                {
                    low3[j] = s_indices[i + j] & 0x07;
                    high1 |= ((s_indices[i + j] >> 3) & 0x01) << j;
                }

                // Pack 8 × 3-bit into 3 bytes (same as tq3_pack_8)
                int byte_idx = i * 3 / 8;
                mse_ptr[byte_idx + 0] = static_cast<uint8_t>(low3[0] | (low3[1] << 3) | (low3[2] << 6));
                mse_ptr[byte_idx + 1] = static_cast<uint8_t>((low3[2] >> 2) | (low3[3] << 1) | (low3[4] << 4) | (low3[5] << 7));
                mse_ptr[byte_idx + 2] = static_cast<uint8_t>((low3[5] >> 1) | (low3[6] << 2) | (low3[7] << 5));

                high_ptr[i / 8] = high1;
            }
        }
    }

    // =========================================================================
    // Fused Quantize-to-Ring Kernel (K TQ8 + V TQ4, single token decode)
    // =========================================================================
    // Grid: (n_kv_heads, 2)  — blockIdx.y=0 → K(TQ8), blockIdx.y=1 → V(TQ4)
    // Block: (D)
    //
    // Writes quantized blocks directly to the ring buffer position, eliminating
    // the temp buffer + D2D memcpy. For decode where num_tokens=1.

    template <int D>
    __global__ void tq_quantize_fused_ring_kernel(
        const float *__restrict__ d_K_input,   // [n_kv_heads * D]
        const float *__restrict__ d_V_input,   // [n_kv_heads * D]
        const float *__restrict__ d_rotations, // [n_kv_heads * D * D]
        uint8_t *__restrict__ d_K_ring,        // K ring buffer
        uint8_t *__restrict__ d_V_ring,        // V ring buffer
        int ring_pos_scalar,                   // Used when d_ring_pos_ptr is null
        const int *d_ring_pos_ptr,             // When non-null, read ring_pos from device memory
        int n_kv_heads)
    {
        const int ring_pos = d_ring_pos_ptr ? *d_ring_pos_ptr : ring_pos_scalar;
        const int head = blockIdx.x;
        const int phase = blockIdx.y; // 0=K(TQ8), 1=V(TQ4)
        const int tid = threadIdx.x;

        if (tid >= D)
            return;

        const int kv_dim = n_kv_heads * D;
        const float x = (phase == 0)
                            ? d_K_input[head * D + tid]
                            : d_V_input[head * D + tid];

        // Block reduction for L2 norm
        __shared__ float s_reduce[32];
        __shared__ float s_norm;
        __shared__ float s_combined_scale;

        float norm_sq = block_reduce_sum(x * x, s_reduce);

        if (tid == 0)
        {
            float norm = sqrtf(norm_sq);
            s_norm = norm;
            s_combined_scale = (norm > 1e-30f) ? (sqrtf(static_cast<float>(D)) / norm) : 0.0f;
        }
        __syncthreads();

        // Normalize + scale
        float scaled = x * s_combined_scale;

        // Rotation: out[tid] = Σ_j Π[tid][j] * scaled[j]
        __shared__ float s_scaled[128];
        s_scaled[tid] = scaled;
        __syncthreads();

        const float *rot_row = d_rotations + static_cast<size_t>(head) * D * D + tid * D;
        float rotated = 0.0f;
        for (int j = 0; j < D; ++j)
            rotated += rot_row[j] * s_scaled[j];

        if (phase == 0)
        {
            // ---- TQ8: 8-bit binary search + write to ring ----
            int idx = 0;
#define TQ8_GPU_BSEARCH_F(STEP, OFFSET)           \
    if (rotated > d_TQ8_THRESHOLDS[idx + OFFSET]) \
        idx += STEP;
            TQ8_GPU_BSEARCH_F(128, 127)
            TQ8_GPU_BSEARCH_F(64, 63)
            TQ8_GPU_BSEARCH_F(32, 31)
            TQ8_GPU_BSEARCH_F(16, 15)
            TQ8_GPU_BSEARCH_F(8, 7)
            TQ8_GPU_BSEARCH_F(4, 3)
            TQ8_GPU_BSEARCH_F(2, 1)
            if (rotated > d_TQ8_THRESHOLDS[idx])
                idx += 1;
#undef TQ8_GPU_BSEARCH_F

            constexpr size_t block_size = sizeof(TQ8Block<D>);
            uint8_t *block_ptr = d_K_ring +
                                 (static_cast<size_t>(ring_pos) * n_kv_heads + head) * block_size;

            if (tid == 0)
            {
                float *norms = reinterpret_cast<float *>(block_ptr);
                norms[0] = s_norm;
                norms[1] = -1.0f;
            }
            block_ptr[2 * sizeof(float) + tid] = static_cast<uint8_t>(idx);
        }
        else
        {
            // ---- TQ4: 4-bit binary search + parallel packing + write to ring ----
            int idx = 0;
#define TQ4_GPU_BSEARCH_F(STEP, OFFSET)           \
    if (rotated > d_TQ4_THRESHOLDS[idx + OFFSET]) \
        idx += STEP;
            TQ4_GPU_BSEARCH_F(8, 7)
            TQ4_GPU_BSEARCH_F(4, 3)
            TQ4_GPU_BSEARCH_F(2, 1)
            if (rotated > d_TQ4_THRESHOLDS[idx])
                idx += 1;
#undef TQ4_GPU_BSEARCH_F

            // Store full index to shared memory for packing
            __shared__ uint8_t s_indices[128];
            s_indices[tid] = static_cast<uint8_t>(idx);
            __syncthreads();

            constexpr size_t MSE_BYTES = D * 3 / 8;
            constexpr size_t block_size = sizeof(TQ4Block<D>);
            uint8_t *block_ptr = d_V_ring +
                                 (static_cast<size_t>(ring_pos) * n_kv_heads + head) * block_size;

            if (tid == 0)
            {
                float *norms = reinterpret_cast<float *>(block_ptr);
                norms[0] = s_norm;
                norms[1] = -1.0f;
            }

            // Parallel packing: each thread in [0, D/8) packs one group of 8
            const int pack_tid = tid;
            if (pack_tid < D / 8)
            {
                const int base = pack_tid * 8;
                uint8_t *mse_ptr = block_ptr + 2 * sizeof(float);
                uint8_t *high_ptr = mse_ptr + MSE_BYTES;

                uint8_t low3[8];
                uint8_t high1 = 0;
                for (int j = 0; j < 8; ++j)
                {
                    low3[j] = s_indices[base + j] & 0x07;
                    high1 |= ((s_indices[base + j] >> 3) & 0x01) << j;
                }

                int byte_idx = pack_tid * 3;
                mse_ptr[byte_idx + 0] = static_cast<uint8_t>(low3[0] | (low3[1] << 3) | (low3[2] << 6));
                mse_ptr[byte_idx + 1] = static_cast<uint8_t>((low3[2] >> 2) | (low3[3] << 1) | (low3[4] << 4) | (low3[5] << 7));
                mse_ptr[byte_idx + 2] = static_cast<uint8_t>((low3[5] >> 1) | (low3[6] << 2) | (low3[7] << 5));
                high_ptr[pack_tid] = high1;
            }
        }
    }

    // =========================================================================
    // TQ8 Dequantize Kernel
    // =========================================================================

    /**
     * @brief Dequantize TQ8 blocks to FP16 with optional RoPE.
     *
     * Grid:  (count, n_kv_heads)
     * Block: (head_dim)
     *
     * Each block processes one (position, head) pair.
     */
    template <int D>
    __global__ void tq8_dequantize_kernel(
        const uint8_t *__restrict__ d_tq8_bytes, // raw TQ8Block<D> array
        const float *__restrict__ d_rotations_t, // [n_kv_heads * D * D] transposed
        float *__restrict__ d_output,            // [count, n_kv_heads * D]
        int count, int n_kv_heads,
        float rope_theta, int position_start,
        int max_seq_len, int tail)
    {
        const int pos = blockIdx.x; // position in linearized output
        const int head = blockIdx.y;
        const int tid = threadIdx.x;

        if (pos >= count || tid >= D)
            return;

        // Read TQ8Block for this (position, head)
        // Ring buffer: source position = (tail + pos) % max_seq_len
        // Blocks are stored at [ring_pos * n_kv_heads + head]
        const int ring_pos = (tail + pos) % max_seq_len;
        const size_t block_size = sizeof(TQ8Block<D>);
        const uint8_t *block_ptr = d_tq8_bytes + (static_cast<size_t>(ring_pos) * n_kv_heads + head) * block_size;

        // Read norm
        const float *norms = reinterpret_cast<const float *>(block_ptr);
        const float norm = norms[0];

        // Read index and look up centroid
        const uint8_t idx = block_ptr[2 * sizeof(float) + tid];
        const float inv_scale = 1.0f / sqrtf(static_cast<float>(D));
        float centroid_val = d_TQ8_CENTROIDS[idx] * inv_scale;

        // Inverse rotation: out[tid] = Σ_j Πᵀ[tid][j] * centroid[j]
        __shared__ float s_centroid[128];
        s_centroid[tid] = centroid_val;
        __syncthreads();

        const float *rot_t_row = d_rotations_t + static_cast<size_t>(head) * D * D + tid * D;
        float val = 0.0f;
        for (int j = 0; j < D; ++j)
            val += rot_t_row[j] * s_centroid[j];

        // Scale by norm
        val *= norm;

        // Optional RoPE using precomputed frequency table (no powf!)
        if (rope_theta > 0.0f)
        {
            constexpr int HALF = D / 2;
            const int actual_pos = position_start + pos;
            const int pair_idx = (tid < HALF) ? tid : (tid - HALF);
            const float angle = static_cast<float>(actual_pos) * d_ROPE_FREQS[pair_idx];
            float cos_val, sin_val;
            __sincosf(angle, &sin_val, &cos_val);

            s_centroid[tid] = val;
            __syncthreads();

            if (tid < HALF)
            {
                float partner = s_centroid[tid + HALF];
                val = val * cos_val - partner * sin_val;
            }
            else
            {
                float partner = s_centroid[tid - HALF];
                val = partner * sin_val + val * cos_val;
            }
        }

        // Write FP32 output
        const int kv_dim = n_kv_heads * D;
        d_output[pos * kv_dim + head * D + tid] = val;
    }

    // =========================================================================
    // TQ4 Dequantize Kernel
    // =========================================================================

    /**
     * @brief Dequantize TQ4 blocks to FP32 (no RoPE — V doesn't need it).
     *
     * Grid:  (count, n_kv_heads)
     * Block: (head_dim)
     */
    template <int D>
    __global__ void tq4_dequantize_kernel(
        const uint8_t *__restrict__ d_tq4_bytes,
        const float *__restrict__ d_rotations_t,
        float *__restrict__ d_output,
        int count, int n_kv_heads,
        int max_seq_len, int tail)
    {
        const int pos = blockIdx.x;
        const int head = blockIdx.y;
        const int tid = threadIdx.x;

        if (pos >= count || tid >= D)
            return;

        const int ring_pos = (tail + pos) % max_seq_len;
        const size_t block_size = sizeof(TQ4Block<D>);
        const uint8_t *block_ptr = d_tq4_bytes + (static_cast<size_t>(ring_pos) * n_kv_heads + head) * block_size;

        const float *norms = reinterpret_cast<const float *>(block_ptr);
        const float norm = norms[0];

        // Unpack TQ4 index for this element
        // Layout: [norm(4B)][residual_norm(4B)][mse_indices[D*3/8]][high_bits[D/8]]
        constexpr size_t MSE_BYTES = D * 3 / 8;
        const uint8_t *mse_ptr = block_ptr + 2 * sizeof(float);
        const uint8_t *high_ptr = mse_ptr + MSE_BYTES;

        // Unpack 3-bit low index for element `tid`
        // 8 elements pack into 3 bytes
        const int group8 = tid / 8;
        const int within = tid % 8;
        const int byte_idx = group8 * 3;

        // Unpack 3 bytes → 8 indices
        // We only need the one at position `within`
        const uint8_t b0 = mse_ptr[byte_idx + 0];
        const uint8_t b1 = mse_ptr[byte_idx + 1];
        const uint8_t b2 = mse_ptr[byte_idx + 2];

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
        case 7:
            low3 = (b2 >> 5) & 0x07;
            break;
        default:
            low3 = 0;
            break;
        }

        // Unpack high bit
        const uint8_t high_byte = high_ptr[tid / 8];
        const uint8_t high1 = (high_byte >> (tid % 8)) & 0x01;

        // Reconstruct 4-bit index
        const uint8_t full_idx = low3 | (high1 << 3);

        // Centroid lookup
        const float inv_scale = 1.0f / sqrtf(static_cast<float>(D));
        float centroid_val = d_TQ4_CENTROIDS[full_idx] * inv_scale;

        // Inverse rotation
        __shared__ float s_centroid[128];
        s_centroid[tid] = centroid_val;
        __syncthreads();

        const float *rot_t_row = d_rotations_t + static_cast<size_t>(head) * D * D + tid * D;
        float val = 0.0f;
        for (int j = 0; j < D; ++j)
            val += rot_t_row[j] * s_centroid[j];

        val *= norm;

        const int kv_dim = n_kv_heads * D;
        d_output[pos * kv_dim + head * D + tid] = val;
    }

    // =========================================================================
    // =========================================================================
    // Batched Incremental Dequant Kernels (all layers in 1 launch)
    // =========================================================================
    //
    // During decode, each layer needs exactly 1 new position dequantized.
    // Instead of 28 separate kernel launches (one per layer), these kernels
    // process ALL layers in a single launch with grid.y = n_layers.
    //
    // Grid:  (n_kv_heads, n_layers)
    // Block: (D)
    //
    // Each block handles one (head, layer) pair for a single position.

    // =========================================================================
    // Fused Single-Position Incremental Kernel (K TQ8 + V TQ4 in one launch)
    // =========================================================================
    // Grid: (n_kv_heads)    Block: (D)
    //
    // Processes both K (TQ8→FP16) and V (TQ4→FP16) for a single newly-appended
    // position in one kernel launch. Halves per-layer launch overhead from 2→1.
    // The K path includes optional RoPE; V has no RoPE.
    // Shared memory is reused between K and V phases.

    template <int D>
    __global__ void tq_incremental_fused_kernel(
        const uint8_t *__restrict__ k_cache,
        const uint8_t *__restrict__ v_cache,
        __half *__restrict__ k_output,
        __half *__restrict__ v_output,
        const float *__restrict__ k_rotation,
        const float *__restrict__ v_rotation,
        int ring_pos_scalar, int out_offset_elems_scalar,
        int n_kv_heads,
        float rope_theta, int rope_position_scalar,
        const TQDequantDynamicParams *d_params) // When non-null, override scalars
    {
        const int ring_pos = d_params ? d_params->ring_pos : ring_pos_scalar;
        const int out_offset_elems = d_params ? d_params->out_offset_elems : out_offset_elems_scalar;
        const int rope_position = d_params ? d_params->rope_position : rope_position_scalar;

        const int head = blockIdx.x;
        const int tid = threadIdx.x;
        if (tid >= D)
            return;

        const float inv_scale = rsqrtf(static_cast<float>(D));
        __shared__ float s_cent[D];

        // ---- Phase 1: K (TQ8) ----
        {
            constexpr size_t block_size = sizeof(TQ8Block<D>);
            const uint8_t *block_ptr = k_cache +
                                       (static_cast<size_t>(ring_pos) * n_kv_heads + head) * block_size;
            const float norm = reinterpret_cast<const float *>(block_ptr)[0];
            const uint8_t idx = block_ptr[2 * sizeof(float) + tid];

            s_cent[tid] = d_TQ8_CENTROIDS[idx] * inv_scale;
            __syncthreads();

            const float *rot = k_rotation + static_cast<size_t>(head) * D * D;
            float val = 0.0f;
            for (int j = 0; j < D; ++j)
                val += rot[j * D + tid] * s_cent[j];
            val *= norm;

            // Optional RoPE (reuses s_cent for pair exchange)
            if (rope_theta > 0.0f)
            {
                s_cent[tid] = val;
                __syncthreads();

                constexpr int HALF = D / 2;
                const int pair_idx = (tid < HALF) ? tid : (tid - HALF);
                const float angle = static_cast<float>(rope_position) * d_ROPE_FREQS[pair_idx];
                float cos_val, sin_val;
                __sincosf(angle, &sin_val, &cos_val);
                const float partner = s_cent[(tid < HALF) ? (tid + HALF) : (tid - HALF)];
                if (tid < HALF)
                    val = val * cos_val - partner * sin_val;
                else
                    val = partner * sin_val + val * cos_val;
            }

            k_output[out_offset_elems + head * D + tid] = __float2half(val);
        }

        __syncthreads(); // Barrier before reusing shared memory for V

        // ---- Phase 2: V (TQ4, no RoPE) ----
        {
            constexpr size_t block_size = sizeof(TQ4Block<D>);
            const uint8_t *block_ptr = v_cache +
                                       (static_cast<size_t>(ring_pos) * n_kv_heads + head) * block_size;
            const float norm = reinterpret_cast<const float *>(block_ptr)[0];

            constexpr size_t MSE_BYTES = D * 3 / 8;
            const uint8_t *mse_ptr = block_ptr + 2 * sizeof(float);
            const uint8_t *high_ptr = mse_ptr + MSE_BYTES;

            const int group8 = tid / 8;
            const int within = tid % 8;
            const int byte_idx = group8 * 3;

            const uint8_t b0 = mse_ptr[byte_idx + 0];
            const uint8_t b1 = mse_ptr[byte_idx + 1];
            const uint8_t b2 = mse_ptr[byte_idx + 2];

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
            case 7:
                low3 = (b2 >> 5) & 0x07;
                break;
            default:
                low3 = 0;
                break;
            }
            const uint8_t high1 = (high_ptr[tid / 8] >> (tid % 8)) & 0x01;
            const uint8_t full_idx = low3 | (high1 << 3);

            s_cent[tid] = d_TQ4_CENTROIDS[full_idx] * inv_scale;
            __syncthreads();

            const float *rot = v_rotation + static_cast<size_t>(head) * D * D;
            float val = 0.0f;
            for (int j = 0; j < D; ++j)
                val += rot[j * D + tid] * s_cent[j];

            v_output[out_offset_elems + head * D + tid] = __float2half(val * norm);
        }
    }

    template <int D>
    __global__ void tq8_incremental_batch_kernel(
        const IncrementalDequantParam *__restrict__ params,
        int n_kv_heads,
        float rope_theta)
    {
        const int head = blockIdx.x;
        const int layer = blockIdx.y;
        const int tid = threadIdx.x;

        if (tid >= D)
            return;

        const auto &p = params[layer];
        const size_t block_size = sizeof(TQ8Block<D>);
        const float inv_scale = 1.0f / sqrtf(static_cast<float>(D));
        const int kv_dim = n_kv_heads * D;

        // Load centroid for the single new position
        const uint8_t *block_ptr = p.cache +
                                   (static_cast<size_t>(p.ring_pos) * n_kv_heads + head) * block_size;
        const float norm = reinterpret_cast<const float *>(block_ptr)[0];
        const uint8_t idx = block_ptr[2 * sizeof(float) + tid];

        __shared__ float s_cent[D];
        s_cent[tid] = d_TQ8_CENTROIDS[idx] * inv_scale;
        __syncthreads();

        // Matmul: output[tid] = Σ_j R[j][tid] × centroid[j]
        const float *rot = p.rotation + static_cast<size_t>(head) * D * D;
        float val = 0.0f;
        for (int j = 0; j < D; ++j)
            val += rot[j * D + tid] * s_cent[j];

        val *= norm;

        // Optional RoPE
        if (rope_theta > 0.0f)
        {
            s_cent[tid] = val;
            __syncthreads();

            constexpr int HALF = D / 2;
            const int pair_idx = (tid < HALF) ? tid : (tid - HALF);
            const float angle = static_cast<float>(p.rope_position) * d_ROPE_FREQS[pair_idx];
            float cos_val, sin_val;
            __sincosf(angle, &sin_val, &cos_val);
            const float partner = s_cent[(tid < HALF) ? (tid + HALF) : (tid - HALF)];
            if (tid < HALF)
                val = val * cos_val - partner * sin_val;
            else
                val = partner * sin_val + val * cos_val;
        }

        p.output[p.out_offset + head * D + tid] = __float2half(val);
    }

    template <int D>
    __global__ void tq4_incremental_batch_kernel(
        const IncrementalDequantParam *__restrict__ params,
        int n_kv_heads)
    {
        const int head = blockIdx.x;
        const int layer = blockIdx.y;
        const int tid = threadIdx.x;

        if (tid >= D)
            return;

        const auto &p = params[layer];
        const size_t block_size = sizeof(TQ4Block<D>);
        const float inv_scale = 1.0f / sqrtf(static_cast<float>(D));
        const int kv_dim = n_kv_heads * D;

        // Load TQ4 centroid for the single new position
        const uint8_t *block_ptr = p.cache +
                                   (static_cast<size_t>(p.ring_pos) * n_kv_heads + head) * block_size;
        const float norm = reinterpret_cast<const float *>(block_ptr)[0];

        constexpr size_t MSE_BYTES = D * 3 / 8;
        const uint8_t *mse_ptr = block_ptr + 2 * sizeof(float);
        const uint8_t *high_ptr = mse_ptr + MSE_BYTES;

        const int group8 = tid / 8;
        const int within = tid % 8;
        const int byte_idx = group8 * 3;

        const uint8_t b0 = mse_ptr[byte_idx + 0];
        const uint8_t b1 = mse_ptr[byte_idx + 1];
        const uint8_t b2 = mse_ptr[byte_idx + 2];

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
        case 7:
            low3 = (b2 >> 5) & 0x07;
            break;
        default:
            low3 = 0;
            break;
        }
        const uint8_t high1 = (high_ptr[tid / 8] >> (tid % 8)) & 0x01;
        const uint8_t full_idx = low3 | (high1 << 3);

        __shared__ float s_cent[D];
        s_cent[tid] = d_TQ4_CENTROIDS[full_idx] * inv_scale;
        __syncthreads();

        // Matmul: output[tid] = Σ_j R[j][tid] × centroid[j]
        const float *rot = p.rotation + static_cast<size_t>(head) * D * D;
        float val = 0.0f;
        for (int j = 0; j < D; ++j)
            val += rot[j * D + tid] * s_cent[j];

        p.output[p.out_offset + head * D + tid] = __float2half(val * norm);
    }

    // Tiled TQ8 Dequantize Kernel — L1-cached rotation matrix
    // =========================================================================
    // Loop-interchanged tiled TQ8/TQ4 dequantize kernels.
    //
    // Grid:  (n_kv_heads, ceil(count / TILE))
    // Block: (D)
    //
    // Previous kernels had position-outer, dimension-inner loops, causing
    // the rotation matrix to be read D×TILE times per thread. The loop
    // interchange restructures to j-outer (dimension), t-inner (position):
    //
    //   Phase 1: Load ALL TILE positions' centroids into transposed shared
    //            memory: s_cents[dim * TILE + pos] — broadcast reads in the
    //            inner loop (all threads read same address = 1 cycle).
    //
    //   Phase 2: j-outer loop: load rotation element ONCE per j (coalesced),
    //            then TILE independent FMAs. Result: 16× fewer global memory
    //            reads + full TILE-way ILP on the FMA chain.
    //
    // Shared memory: D × TILE × 4B = 128 × 16 × 4 = 8KB (same as before)
    // Registers: TILE floats for accumulators + TILE floats for norms

    /// Store computed FP32 value to output buffer (float or __half)
    template <typename OutT>
    __device__ __forceinline__ void tq_store(OutT *dst, int idx, float val)
    {
        if constexpr (std::is_same_v<OutT, __half>)
            dst[idx] = __float2half(val);
        else
            dst[idx] = val;
    }

    template <int D, int TILE, typename OutT = float>
    __global__ void tq8_dequantize_tiled_kernel(
        const uint8_t *__restrict__ d_tq8_bytes,
        const float *__restrict__ d_rotations, // R (non-transposed, row-major) for coalesced access
        OutT *__restrict__ d_output,
        int count, int n_kv_heads,
        float rope_theta, int position_start,
        int max_seq_len, int tail)
    {
        const int head = blockIdx.x;
        const int tile_start = blockIdx.y * TILE;
        const int tid = threadIdx.x;

        if (tid >= D)
            return;

        const float *rot_head = d_rotations + static_cast<size_t>(head) * D * D;
        const float inv_scale = 1.0f / sqrtf(static_cast<float>(D));
        const size_t block_size = sizeof(TQ8Block<D>);
        const int kv_dim = n_kv_heads * D;

        // Transposed layout: s_cents[dim * TILE + pos]
        // Inner loop reads s_cents[j * TILE + t] — all threads read same address (broadcast)
        __shared__ float s_cents[D * TILE];

        // Phase 1: Load all TILE positions' centroid values into transposed shared memory
        float norms[TILE];
        int tile_count = 0;
#pragma unroll
        for (int t = 0; t < TILE; ++t)
        {
            const int pos = tile_start + t;
            if (pos < count)
            {
                tile_count = t + 1;

                const int ring_pos = (tail + pos) % max_seq_len;
                const uint8_t *block_ptr = d_tq8_bytes +
                                           (static_cast<size_t>(ring_pos) * n_kv_heads + head) * block_size;

                norms[t] = reinterpret_cast<const float *>(block_ptr)[0];
                const uint8_t idx = block_ptr[2 * sizeof(float) + tid];
                s_cents[tid * TILE + t] = d_TQ8_CENTROIDS[idx] * inv_scale;
            }
            else
            {
                norms[t] = 0.0f;
                s_cents[tid * TILE + t] = 0.0f;
            }
        }
        __syncthreads();

        // Phase 2: Loop-interchanged matmul — j outer, t inner
        // Rotation loaded ONCE per j (coalesced), TILE independent FMAs per j
        float vals[TILE];
#pragma unroll
        for (int t = 0; t < TILE; ++t)
            vals[t] = 0.0f;

        for (int j = 0; j < D; ++j)
        {
            const float r = rot_head[j * D + tid]; // Coalesced: consecutive threads read consecutive addresses
#pragma unroll
            for (int t = 0; t < TILE; ++t)            // TILE is compile-time → fully unrolls for 16-way ILP
                vals[t] += r * s_cents[j * TILE + t]; // Broadcast: all threads read same address
        }

        // Phase 3: Scale by norms and write output (with optional RoPE)
        if (rope_theta > 0.0f)
        {
            // Store all values into shared memory for cross-thread RoPE exchange
            // (partner thread is at ±D/2, which spans warps — can't use __shfl)
            __syncthreads(); // ensure Phase 2 is done before reusing s_cents

            constexpr int HALF = D / 2;
            const int pair_idx = (tid < HALF) ? tid : (tid - HALF);

            for (int t = 0; t < tile_count; ++t)
            {
                const float val = vals[t] * norms[t];
                s_cents[t * D + tid] = val; // reuse s_cents for RoPE exchange
                __syncthreads();

                const int actual_pos = position_start + tile_start + t;
                const float angle = static_cast<float>(actual_pos) * d_ROPE_FREQS[pair_idx];
                float cos_val, sin_val;
                __sincosf(angle, &sin_val, &cos_val);

                const float partner = s_cents[t * D + ((tid < HALF) ? (tid + HALF) : (tid - HALF))];
                if (tid < HALF)
                    vals[t] = val * cos_val - partner * sin_val;
                else
                    vals[t] = partner * sin_val + val * cos_val;

                tq_store(d_output, (tile_start + t) * kv_dim + head * D + tid, vals[t]);
                __syncthreads();
            }
        }
        else
        {
            // Fast path: no RoPE — just scale and write
            for (int t = 0; t < tile_count; ++t)
                tq_store(d_output, (tile_start + t) * kv_dim + head * D + tid, vals[t] * norms[t]);
        }
    }

    // =========================================================================
    // Loop-interchanged Tiled TQ4 Dequantize Kernel
    // =========================================================================

    template <int D, int TILE, typename OutT = float>
    __global__ void tq4_dequantize_tiled_kernel(
        const uint8_t *__restrict__ d_tq4_bytes,
        const float *__restrict__ d_rotations, // R (non-transposed, row-major) for coalesced access
        OutT *__restrict__ d_output,
        int count, int n_kv_heads,
        int max_seq_len, int tail)
    {
        const int head = blockIdx.x;
        const int tile_start = blockIdx.y * TILE;
        const int tid = threadIdx.x;

        if (tid >= D)
            return;

        const float *rot_head = d_rotations + static_cast<size_t>(head) * D * D;
        const float inv_scale = 1.0f / sqrtf(static_cast<float>(D));
        const size_t block_size = sizeof(TQ4Block<D>);
        const int kv_dim = n_kv_heads * D;

        // Transposed layout: s_cents[dim * TILE + pos]
        __shared__ float s_cents[D * TILE];

        // Phase 1: Load all TILE positions' TQ4 centroid values
        float norms[TILE];
        int tile_count = 0;

        constexpr size_t MSE_BYTES = D * 3 / 8;
        const int group8 = tid / 8;
        const int within = tid % 8;
        const int byte_idx = group8 * 3;

#pragma unroll
        for (int t = 0; t < TILE; ++t)
        {
            const int pos = tile_start + t;
            if (pos < count)
            {
                tile_count = t + 1;

                const int ring_pos = (tail + pos) % max_seq_len;
                const uint8_t *block_ptr = d_tq4_bytes +
                                           (static_cast<size_t>(ring_pos) * n_kv_heads + head) * block_size;

                norms[t] = reinterpret_cast<const float *>(block_ptr)[0];

                const uint8_t *mse_ptr = block_ptr + 2 * sizeof(float);
                const uint8_t *high_ptr = mse_ptr + MSE_BYTES;

                const uint8_t b0 = mse_ptr[byte_idx + 0];
                const uint8_t b1 = mse_ptr[byte_idx + 1];
                const uint8_t b2 = mse_ptr[byte_idx + 2];

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
                case 7:
                    low3 = (b2 >> 5) & 0x07;
                    break;
                default:
                    low3 = 0;
                    break;
                }

                const uint8_t high1 = (high_ptr[tid / 8] >> (tid % 8)) & 0x01;
                const uint8_t full_idx = low3 | (high1 << 3);

                s_cents[tid * TILE + t] = d_TQ4_CENTROIDS[full_idx] * inv_scale;
            }
            else
            {
                norms[t] = 0.0f;
                s_cents[tid * TILE + t] = 0.0f;
            }
        }
        __syncthreads();

        // Phase 2: Loop-interchanged matmul
        float vals[TILE];
#pragma unroll
        for (int t = 0; t < TILE; ++t)
            vals[t] = 0.0f;

        for (int j = 0; j < D; ++j)
        {
            const float r = rot_head[j * D + tid];
#pragma unroll
            for (int t = 0; t < TILE; ++t) // TILE is compile-time → fully unrolls for 16-way ILP
                vals[t] += r * s_cents[j * TILE + t];
        }

        // Phase 3: Scale by norms and write output
        for (int t = 0; t < tile_count; ++t)
            tq_store(d_output, (tile_start + t) * kv_dim + head * D + tid, vals[t] * norms[t]);
    }

    // =========================================================================
    // RoPE Kernels (for non-TQ caches)
    // =========================================================================

    __global__ void rope_apply_fp16_kernel(
        __half *__restrict__ d_K,
        int count, int n_kv_heads, int head_dim,
        float rope_theta, int position_start, int rope_dim)
    {
        // Each thread handles one (cos, sin) pair using half-split convention
        // Pair i: elements (i, i + half_dim) within each head
        // rope_dim: number of dimensions to rotate (partial RoPE). 0 = full head_dim.
        const int effective_rope_dim = (rope_dim > 0) ? rope_dim : head_dim;
        const int kv_dim = n_kv_heads * head_dim;
        const int half_dim = effective_rope_dim / 2;
        const int total_pairs = count * n_kv_heads * half_dim;
        const int pair_global = blockIdx.x * blockDim.x + threadIdx.x;

        if (pair_global >= total_pairs)
            return;

        // Decompose into (position, head, pair_within_head)
        const int pairs_per_head = half_dim;
        const int pairs_per_pos = n_kv_heads * pairs_per_head;

        const int pos = pair_global / pairs_per_pos;
        const int remaining = pair_global % pairs_per_pos;
        const int head = remaining / pairs_per_head;
        const int pair_idx = remaining % pairs_per_head;

        const int actual_pos = position_start + pos;
        const float freq = 1.0f / powf(rope_theta, static_cast<float>(2 * pair_idx) / static_cast<float>(effective_rope_dim));
        const float angle = static_cast<float>(actual_pos) * freq;
        const float cos_val = cosf(angle);
        const float sin_val = sinf(angle);

        // Half-split: pair (i, i + half_dim)
        const int head_base = pos * kv_dim + head * head_dim;
        const int idx0 = head_base + pair_idx;
        const int idx1 = head_base + pair_idx + half_dim;
        float x = __half2float(d_K[idx0]);
        float y = __half2float(d_K[idx1]);

        d_K[idx0] = __float2half_rn(x * cos_val - y * sin_val);
        d_K[idx1] = __float2half_rn(x * sin_val + y * cos_val);
    }

    __global__ void rope_apply_fp32_kernel(
        float *__restrict__ d_K,
        int count, int n_kv_heads, int head_dim,
        float rope_theta, int position_start, int rope_dim)
    {
        const int effective_rope_dim = (rope_dim > 0) ? rope_dim : head_dim;
        const int kv_dim = n_kv_heads * head_dim;
        const int half_dim = effective_rope_dim / 2;
        const int total_pairs = count * n_kv_heads * half_dim;
        const int pair_global = blockIdx.x * blockDim.x + threadIdx.x;

        if (pair_global >= total_pairs)
            return;

        const int pairs_per_head = half_dim;
        const int pairs_per_pos = n_kv_heads * pairs_per_head;

        const int pos = pair_global / pairs_per_pos;
        const int remaining = pair_global % pairs_per_pos;
        const int head = remaining / pairs_per_head;
        const int pair_idx = remaining % pairs_per_head;

        const int actual_pos = position_start + pos;
        const float freq = 1.0f / powf(rope_theta, static_cast<float>(2 * pair_idx) / static_cast<float>(effective_rope_dim));
        const float angle = static_cast<float>(actual_pos) * freq;
        const float cos_val = cosf(angle);
        const float sin_val = sinf(angle);

        // Half-split: pair (i, i + half_dim)
        const int head_base = pos * kv_dim + head * head_dim;
        const int idx0 = head_base + pair_idx;
        const int idx1 = head_base + pair_idx + half_dim;
        float x = d_K[idx0];
        float y = d_K[idx1];

        d_K[idx0] = x * cos_val - y * sin_val;
        d_K[idx1] = x * sin_val + y * cos_val;
    }

    // =========================================================================
    // Extern "C" Wrappers
    // =========================================================================

    extern "C" bool cuda_tq8_quantize(
        const float *d_input,
        const float *d_rotations,
        void *d_output,
        int num_tokens, int n_kv_heads, int head_dim,
        cudaStream_t stream)
    {
        if (!d_input || !d_rotations || !d_output || num_tokens <= 0)
            return false;

        cuda_tq_upload_codebooks(stream);

        const dim3 grid(num_tokens, n_kv_heads);

        if (head_dim == 64)
        {
            const dim3 block(64);
            tq8_quantize_kernel<64><<<grid, block, 0, stream>>>(
                d_input, d_rotations, static_cast<uint8_t *>(d_output), n_kv_heads);
        }
        else if (head_dim == 128)
        {
            const dim3 block(128);
            tq8_quantize_kernel<128><<<grid, block, 0, stream>>>(
                d_input, d_rotations, static_cast<uint8_t *>(d_output), n_kv_heads);
        }
        else
        {
            return false;
        }

        return cudaGetLastError() == cudaSuccess;
    }

    extern "C" bool cuda_tq4_quantize(
        const float *d_input,
        const float *d_rotations,
        void *d_output,
        int num_tokens, int n_kv_heads, int head_dim,
        cudaStream_t stream)
    {
        if (!d_input || !d_rotations || !d_output || num_tokens <= 0)
            return false;

        cuda_tq_upload_codebooks(stream);

        const dim3 grid(num_tokens, n_kv_heads);

        if (head_dim == 64)
        {
            const dim3 block(64);
            tq4_quantize_kernel<64><<<grid, block, 0, stream>>>(
                d_input, d_rotations, static_cast<uint8_t *>(d_output), n_kv_heads);
        }
        else if (head_dim == 128)
        {
            const dim3 block(128);
            tq4_quantize_kernel<128><<<grid, block, 0, stream>>>(
                d_input, d_rotations, static_cast<uint8_t *>(d_output), n_kv_heads);
        }
        else
        {
            return false;
        }

        return cudaGetLastError() == cudaSuccess;
    }

    extern "C" bool cuda_tq_quantize_fused_ring(
        const float *d_K_input, const float *d_V_input,
        const float *d_rotations,
        void *d_K_ring, void *d_V_ring,
        int ring_pos, int n_kv_heads, int head_dim,
        cudaStream_t stream)
    {
        cuda_tq_upload_codebooks(stream);

        // Grid: (n_kv_heads, 2) — y=0 for K(TQ8), y=1 for V(TQ4)
        const dim3 grid(n_kv_heads, 2);

        if (head_dim == 64)
        {
            const dim3 block(64);
            tq_quantize_fused_ring_kernel<64><<<grid, block, 0, stream>>>(
                d_K_input, d_V_input, d_rotations,
                static_cast<uint8_t *>(d_K_ring),
                static_cast<uint8_t *>(d_V_ring),
                ring_pos, nullptr, n_kv_heads);
        }
        else if (head_dim == 128)
        {
            const dim3 block(128);
            tq_quantize_fused_ring_kernel<128><<<grid, block, 0, stream>>>(
                d_K_input, d_V_input, d_rotations,
                static_cast<uint8_t *>(d_K_ring),
                static_cast<uint8_t *>(d_V_ring),
                ring_pos, nullptr, n_kv_heads);
        }
        else
        {
            return false;
        }

        return cudaGetLastError() == cudaSuccess;
    }

    extern "C" bool cuda_tq_quantize_fused_ring_dynamic(
        const float *d_K_input, const float *d_V_input,
        const float *d_rotations,
        void *d_K_ring, void *d_V_ring,
        const int *d_ring_pos, int n_kv_heads, int head_dim,
        cudaStream_t stream)
    {
        cuda_tq_upload_codebooks(stream);

        const dim3 grid(n_kv_heads, 2);

        if (head_dim == 64)
        {
            const dim3 block(64);
            tq_quantize_fused_ring_kernel<64><<<grid, block, 0, stream>>>(
                d_K_input, d_V_input, d_rotations,
                static_cast<uint8_t *>(d_K_ring),
                static_cast<uint8_t *>(d_V_ring),
                0, d_ring_pos, n_kv_heads);
        }
        else if (head_dim == 128)
        {
            const dim3 block(128);
            tq_quantize_fused_ring_kernel<128><<<grid, block, 0, stream>>>(
                d_K_input, d_V_input, d_rotations,
                static_cast<uint8_t *>(d_K_ring),
                static_cast<uint8_t *>(d_V_ring),
                0, d_ring_pos, n_kv_heads);
        }
        else
        {
            return false;
        }

        return cudaGetLastError() == cudaSuccess;
    }

    extern "C" bool cuda_tq8_dequantize_fp32(
        const void *d_tq8_blocks,
        const float *d_rotations_t,
        float *d_output,
        int count, int n_kv_heads, int head_dim,
        float rope_theta, int position_start,
        int max_seq_len, int tail,
        cudaStream_t stream)
    {
        if (!d_tq8_blocks || !d_rotations_t || !d_output || count <= 0)
            return false;

        cuda_tq_upload_codebooks(stream);
        if (rope_theta > 0.0f)
            cuda_tq_upload_rope_freqs(rope_theta, head_dim, stream);

        const dim3 grid(count, n_kv_heads);

        if (head_dim == 64)
        {
            const dim3 block(64);
            tq8_dequantize_kernel<64><<<grid, block, 0, stream>>>(
                static_cast<const uint8_t *>(d_tq8_blocks), d_rotations_t, d_output,
                count, n_kv_heads, rope_theta, position_start, max_seq_len, tail);
        }
        else if (head_dim == 128)
        {
            const dim3 block(128);
            tq8_dequantize_kernel<128><<<grid, block, 0, stream>>>(
                static_cast<const uint8_t *>(d_tq8_blocks), d_rotations_t, d_output,
                count, n_kv_heads, rope_theta, position_start, max_seq_len, tail);
        }
        else
        {
            return false;
        }

        return cudaGetLastError() == cudaSuccess;
    }

    extern "C" bool cuda_tq4_dequantize_fp32(
        const void *d_tq4_blocks,
        const float *d_rotations_t,
        float *d_output,
        int count, int n_kv_heads, int head_dim,
        cudaStream_t stream)
    {
        if (!d_tq4_blocks || !d_rotations_t || !d_output || count <= 0)
            return false;

        cuda_tq_upload_codebooks(stream);

        const dim3 grid(count, n_kv_heads);

        if (head_dim == 64)
        {
            const dim3 block(64);
            tq4_dequantize_kernel<64><<<grid, block, 0, stream>>>(
                static_cast<const uint8_t *>(d_tq4_blocks), d_rotations_t, d_output,
                count, n_kv_heads, 0, 0); // max_seq_len=0, tail=0 (direct array, not ring)
        }
        else if (head_dim == 128)
        {
            const dim3 block(128);
            tq4_dequantize_kernel<128><<<grid, block, 0, stream>>>(
                static_cast<const uint8_t *>(d_tq4_blocks), d_rotations_t, d_output,
                count, n_kv_heads, 0, 0);
        }
        else
        {
            return false;
        }

        return cudaGetLastError() == cudaSuccess;
    }

    extern "C" bool cuda_tq_ring_append(
        void *d_K_cache, void *d_V_cache,
        const float *d_K_new, const float *d_V_new,
        const float *d_K_rotations, const float *d_V_rotations,
        int head, int max_seq_len,
        int n_kv_heads, int head_dim, int num_tokens,
        cudaStream_t stream)
    {
        if (num_tokens <= 0)
            return true;

        // Quantize K → TQ8 into scratch, then copy to ring position
        // For simplicity, quantize directly into ring buffer at correct position
        // We need a intermediate buffer for the quantized blocks
        // Actually, we can use a 2-pass approach:
        // 1. Allocate temp TQ8/TQ4 blocks for num_tokens
        // 2. Quantize into temp
        // 3. Ring-append temp blocks

        // Determine block sizes
        size_t k_block_size, v_block_size;
        if (head_dim == 64)
        {
            k_block_size = sizeof(TQ8Block<64>);
            v_block_size = sizeof(TQ4Block<64>);
        }
        else if (head_dim == 128)
        {
            k_block_size = sizeof(TQ8Block<128>);
            v_block_size = sizeof(TQ4Block<128>);
        }
        else
        {
            return false;
        }

        const size_t k_temp_bytes = static_cast<size_t>(num_tokens) * n_kv_heads * k_block_size;
        const size_t v_temp_bytes = static_cast<size_t>(num_tokens) * n_kv_heads * v_block_size;

        // Allocate temp buffers
        void *d_k_temp = nullptr;
        void *d_v_temp = nullptr;
        cudaMalloc(&d_k_temp, k_temp_bytes);
        cudaMalloc(&d_v_temp, v_temp_bytes);

        // Quantize
        bool ok = cuda_tq8_quantize(d_K_new, d_K_rotations, d_k_temp,
                                    num_tokens, n_kv_heads, head_dim, stream);
        if (ok)
        {
            ok = cuda_tq4_quantize(d_V_new, d_V_rotations, d_v_temp,
                                   num_tokens, n_kv_heads, head_dim, stream);
        }

        if (ok)
        {
            // Copy quantized blocks to ring buffer positions
            // Each position stores n_kv_heads blocks
            for (int t = 0; t < num_tokens; ++t)
            {
                int dst_pos = (head + t) % max_seq_len;
                size_t k_dst_offset = static_cast<size_t>(dst_pos) * n_kv_heads * k_block_size;
                size_t k_src_offset = static_cast<size_t>(t) * n_kv_heads * k_block_size;
                cudaMemcpyAsync(static_cast<uint8_t *>(d_K_cache) + k_dst_offset,
                                static_cast<uint8_t *>(d_k_temp) + k_src_offset,
                                n_kv_heads * k_block_size, cudaMemcpyDeviceToDevice, stream);

                size_t v_dst_offset = static_cast<size_t>(dst_pos) * n_kv_heads * v_block_size;
                size_t v_src_offset = static_cast<size_t>(t) * n_kv_heads * v_block_size;
                cudaMemcpyAsync(static_cast<uint8_t *>(d_V_cache) + v_dst_offset,
                                static_cast<uint8_t *>(d_v_temp) + v_src_offset,
                                n_kv_heads * v_block_size, cudaMemcpyDeviceToDevice, stream);
            }
        }

        cudaFree(d_k_temp);
        cudaFree(d_v_temp);

        return ok;
    }

    extern "C" bool cuda_tq_ring_linearize_dequant(
        float *d_K_out, float *d_V_out,
        const void *d_K_cache, const void *d_V_cache,
        const float *d_K_rotations_t, const float *d_V_rotations_t,
        const float *d_K_rotations, const float *d_V_rotations,
        int tail, int count, int max_seq_len,
        int n_kv_heads, int head_dim,
        float rope_theta, int position_start,
        cudaStream_t stream)
    {
        if (count <= 0)
            return true;

        cuda_tq_upload_codebooks(stream);
        if (rope_theta > 0.0f)
            cuda_tq_upload_rope_freqs(rope_theta, head_dim, stream);

        // Tiled kernels use R (non-transposed, row-major) for coalesced access:
        // R[j][tid] at offset j*D + tid → consecutive threads read consecutive addresses.
        // Math: output[tid] = Σ_j R^T[tid][j] * centroid[j] = Σ_j R[j][tid] * centroid[j]
        constexpr int TILE = 16;

        if (head_dim == 64)
        {
            const dim3 grid(n_kv_heads, (count + TILE - 1) / TILE);
            const dim3 block(64);
            tq8_dequantize_tiled_kernel<64, TILE><<<grid, block, 0, stream>>>(
                static_cast<const uint8_t *>(d_K_cache), d_K_rotations, d_K_out,
                count, n_kv_heads, rope_theta, position_start, max_seq_len, tail);
            tq4_dequantize_tiled_kernel<64, TILE><<<grid, block, 0, stream>>>(
                static_cast<const uint8_t *>(d_V_cache), d_V_rotations, d_V_out,
                count, n_kv_heads, max_seq_len, tail);
        }
        else if (head_dim == 128)
        {
            const dim3 grid(n_kv_heads, (count + TILE - 1) / TILE);
            const dim3 block(128);
            tq8_dequantize_tiled_kernel<128, TILE><<<grid, block, 0, stream>>>(
                static_cast<const uint8_t *>(d_K_cache), d_K_rotations, d_K_out,
                count, n_kv_heads, rope_theta, position_start, max_seq_len, tail);
            tq4_dequantize_tiled_kernel<128, TILE><<<grid, block, 0, stream>>>(
                static_cast<const uint8_t *>(d_V_cache), d_V_rotations, d_V_out,
                count, n_kv_heads, max_seq_len, tail);
        }
        else
        {
            return false;
        }

        return cudaGetLastError() == cudaSuccess;
    }

    extern "C" bool cuda_tq_ring_linearize_dequant_fp16(
        __half *d_K_out, __half *d_V_out,
        const void *d_K_cache, const void *d_V_cache,
        const float *d_K_rotations_t, const float *d_V_rotations_t,
        const float *d_K_rotations, const float *d_V_rotations,
        int tail, int count, int max_seq_len,
        int n_kv_heads, int head_dim,
        float rope_theta, int position_start,
        cudaStream_t stream)
    {
        if (count <= 0)
            return true;

        cuda_tq_upload_codebooks(stream);
        if (rope_theta > 0.0f)
            cuda_tq_upload_rope_freqs(rope_theta, head_dim, stream);

        constexpr int TILE = 16;

        if (head_dim == 64)
        {
            const dim3 grid(n_kv_heads, (count + TILE - 1) / TILE);
            const dim3 block(64);
            tq8_dequantize_tiled_kernel<64, TILE, __half><<<grid, block, 0, stream>>>(
                static_cast<const uint8_t *>(d_K_cache), d_K_rotations, d_K_out,
                count, n_kv_heads, rope_theta, position_start, max_seq_len, tail);
            tq4_dequantize_tiled_kernel<64, TILE, __half><<<grid, block, 0, stream>>>(
                static_cast<const uint8_t *>(d_V_cache), d_V_rotations, d_V_out,
                count, n_kv_heads, max_seq_len, tail);
        }
        else if (head_dim == 128)
        {
            const dim3 grid(n_kv_heads, (count + TILE - 1) / TILE);
            const dim3 block(128);
            tq8_dequantize_tiled_kernel<128, TILE, __half><<<grid, block, 0, stream>>>(
                static_cast<const uint8_t *>(d_K_cache), d_K_rotations, d_K_out,
                count, n_kv_heads, rope_theta, position_start, max_seq_len, tail);
            tq4_dequantize_tiled_kernel<128, TILE, __half><<<grid, block, 0, stream>>>(
                static_cast<const uint8_t *>(d_V_cache), d_V_rotations, d_V_out,
                count, n_kv_heads, max_seq_len, tail);
        }
        else
        {
            return false;
        }

        return cudaGetLastError() == cudaSuccess;
    }

    extern "C" bool cuda_tq_incremental_single_fp16(
        __half *d_K_out, __half *d_V_out,
        const void *d_K_cache, const void *d_V_cache,
        const float *d_K_rotation, const float *d_V_rotation,
        int ring_pos, int out_offset_elems,
        int n_kv_heads, int head_dim,
        float rope_theta, int rope_position,
        cudaStream_t stream)
    {
        cuda_tq_upload_codebooks(stream);
        if (rope_theta > 0.0f)
            cuda_tq_upload_rope_freqs(rope_theta, head_dim, stream);

        const dim3 grid(n_kv_heads);

        if (head_dim == 64)
        {
            const dim3 block(64);
            tq_incremental_fused_kernel<64><<<grid, block, 0, stream>>>(
                static_cast<const uint8_t *>(d_K_cache),
                static_cast<const uint8_t *>(d_V_cache),
                d_K_out, d_V_out,
                d_K_rotation, d_V_rotation,
                ring_pos, out_offset_elems, n_kv_heads,
                rope_theta, rope_position, nullptr);
        }
        else if (head_dim == 128)
        {
            const dim3 block(128);
            tq_incremental_fused_kernel<128><<<grid, block, 0, stream>>>(
                static_cast<const uint8_t *>(d_K_cache),
                static_cast<const uint8_t *>(d_V_cache),
                d_K_out, d_V_out,
                d_K_rotation, d_V_rotation,
                ring_pos, out_offset_elems, n_kv_heads,
                rope_theta, rope_position, nullptr);
        }
        else
        {
            return false;
        }

        return cudaGetLastError() == cudaSuccess;
    }

    extern "C" bool cuda_tq_incremental_single_fp16_dynamic(
        __half *d_K_base, __half *d_V_base,
        const void *d_K_cache, const void *d_V_cache,
        const float *d_K_rotation, const float *d_V_rotation,
        const TQDequantDynamicParams *d_params,
        int n_kv_heads, int head_dim,
        float rope_theta,
        cudaStream_t stream)
    {
        cuda_tq_upload_codebooks(stream);
        if (rope_theta > 0.0f)
            cuda_tq_upload_rope_freqs(rope_theta, head_dim, stream);

        const dim3 grid(n_kv_heads);

        if (head_dim == 64)
        {
            const dim3 block(64);
            tq_incremental_fused_kernel<64><<<grid, block, 0, stream>>>(
                static_cast<const uint8_t *>(d_K_cache),
                static_cast<const uint8_t *>(d_V_cache),
                d_K_base, d_V_base,
                d_K_rotation, d_V_rotation,
                0, 0, n_kv_heads,
                rope_theta, 0, d_params);
        }
        else if (head_dim == 128)
        {
            const dim3 block(128);
            tq_incremental_fused_kernel<128><<<grid, block, 0, stream>>>(
                static_cast<const uint8_t *>(d_K_cache),
                static_cast<const uint8_t *>(d_V_cache),
                d_K_base, d_V_base,
                d_K_rotation, d_V_rotation,
                0, 0, n_kv_heads,
                rope_theta, 0, d_params);
        }
        else
        {
            return false;
        }

        return cudaGetLastError() == cudaSuccess;
    }

    extern "C" bool cuda_tq_batch_incremental_dequant_fp16(
        const IncrementalDequantParam *d_K_params,
        const IncrementalDequantParam *d_V_params,
        int n_layers, int n_kv_heads, int head_dim,
        float rope_theta,
        cudaStream_t stream)
    {
        if (n_layers <= 0)
            return true;

        cuda_tq_upload_codebooks(stream);
        if (rope_theta > 0.0f)
            cuda_tq_upload_rope_freqs(rope_theta, head_dim, stream);

        // Grid: (n_kv_heads, n_layers) — all layers in one launch
        const dim3 grid(n_kv_heads, n_layers);

        if (head_dim == 64)
        {
            const dim3 block(64);
            tq8_incremental_batch_kernel<64><<<grid, block, 0, stream>>>(
                d_K_params, n_kv_heads, rope_theta);
            tq4_incremental_batch_kernel<64><<<grid, block, 0, stream>>>(
                d_V_params, n_kv_heads);
        }
        else if (head_dim == 128)
        {
            const dim3 block(128);
            tq8_incremental_batch_kernel<128><<<grid, block, 0, stream>>>(
                d_K_params, n_kv_heads, rope_theta);
            tq4_incremental_batch_kernel<128><<<grid, block, 0, stream>>>(
                d_V_params, n_kv_heads);
        }
        else
        {
            return false;
        }

        return cudaGetLastError() == cudaSuccess;
    }

    extern "C" bool cuda_rope_apply_fp16(
        __half *d_K, int count,
        int n_kv_heads, int head_dim,
        float rope_theta, int position_start,
        cudaStream_t stream, int rope_dim)
    {
        if (!d_K || count <= 0 || rope_theta <= 0.0f)
            return false;

        const int effective_rope_dim = (rope_dim > 0) ? rope_dim : head_dim;
        const int total_pairs = count * n_kv_heads * (effective_rope_dim / 2);
        const dim3 block(256);
        const dim3 grid((total_pairs + 255) / 256);

        rope_apply_fp16_kernel<<<grid, block, 0, stream>>>(
            d_K, count, n_kv_heads, head_dim, rope_theta, position_start, rope_dim);

        return cudaGetLastError() == cudaSuccess;
    }

    extern "C" bool cuda_rope_apply_fp32(
        float *d_K, int count,
        int n_kv_heads, int head_dim,
        float rope_theta, int position_start,
        cudaStream_t stream, int rope_dim)
    {
        if (!d_K || count <= 0 || rope_theta <= 0.0f)
            return false;

        const int effective_rope_dim = (rope_dim > 0) ? rope_dim : head_dim;
        const int total_pairs = count * n_kv_heads * (effective_rope_dim / 2);
        const dim3 block(256);
        const dim3 grid((total_pairs + 255) / 256);

        rope_apply_fp32_kernel<<<grid, block, 0, stream>>>(
            d_K, count, n_kv_heads, head_dim, rope_theta, position_start, rope_dim);

        return cudaGetLastError() == cudaSuccess;
    }

} // namespace llaminar2
