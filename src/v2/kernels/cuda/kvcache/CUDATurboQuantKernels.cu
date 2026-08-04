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
#include "../../../backends/BackendManager.h"
#include "../../../kernels/cpu/turboquant/TurboQuantCodebook.h"
#include "../../../kernels/cpu/turboquant/TurboQuantContext.h"
#include "../../../utils/Logger.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <type_traits>
#include <cmath>
#include <cstring>
#include <atomic>
#include <stdexcept>

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

    bool cuda_tq_upload_codebooks(cudaStream_t stream)
    {
        if (!stream)
            return false;

        /*
         * Constant memory is device-local. Publishing these small immutable
         * tables for every cache construction avoids a process-global bit that
         * incorrectly aliases multiple CUDA devices. Concurrent identical
         * writes are harmless; the cache's construction fence establishes
         * completion before any production launch or graph capture.
         */
        return cudaMemcpyToSymbolAsync(
                   d_TQ8_CENTROIDS, TQ8_CENTROIDS.data(),
                   256 * sizeof(float), 0, cudaMemcpyHostToDevice, stream) == cudaSuccess &&
               cudaMemcpyToSymbolAsync(
                   d_TQ8_THRESHOLDS, TQ8_THRESHOLDS.data(),
                   255 * sizeof(float), 0, cudaMemcpyHostToDevice, stream) == cudaSuccess &&
               cudaMemcpyToSymbolAsync(
                   d_TQ4_CENTROIDS, TQ4_CENTROIDS.data(),
                   16 * sizeof(float), 0, cudaMemcpyHostToDevice, stream) == cudaSuccess &&
               cudaMemcpyToSymbolAsync(
                   d_TQ4_THRESHOLDS, TQ4_THRESHOLDS.data(),
                   15 * sizeof(float), 0, cudaMemcpyHostToDevice, stream) == cudaSuccess;
    }

    // =========================================================================
    // Rotation Matrix Management
    // =========================================================================

    CUDATurboQuantRotations cuda_tq_create_rotations(
        int n_layers, int n_kv_heads, int head_dim,
        uint64_t rotation_seed, int device_id,
        cudaStream_t stream,
        int kv_head_start)
    {
        cudaSetDevice(device_id);

        CUDATurboQuantRotations result;
        result.n_layers = n_layers;
        result.n_kv_heads = n_kv_heads;
        result.head_dim = head_dim;
        result.device_id = device_id;

        const size_t mat_size = static_cast<size_t>(head_dim) * head_dim;
        const size_t total_mats = static_cast<size_t>(n_layers) * n_kv_heads;
        const size_t total_floats = total_mats * mat_size;

        // Rotation storage is a model-lifetime resource, owned by the backend.
        auto *backend = getCUDABackend();
        if (!backend)
            throw std::runtime_error("[TurboQuant CUDA] CUDA backend unavailable");
        const size_t rotation_bytes = total_floats * sizeof(float);
        result.d_rotations =
            static_cast<float *>(backend->allocate(rotation_bytes, device_id));
        result.d_rotations_t =
            static_cast<float *>(backend->allocate(rotation_bytes, device_id));
        if (!result.d_rotations || !result.d_rotations_t)
        {
            if (result.d_rotations)
                backend->free(result.d_rotations, device_id);
            if (result.d_rotations_t)
                backend->free(result.d_rotations_t, device_id);
            throw std::runtime_error(
                "[TurboQuant CUDA] Failed to allocate model-lifetime rotation storage");
        }

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
                const auto &actual_ctx = layer_ctx.for_layer(kv_head_start + head);
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
        auto *backend = getCUDABackend();
        if (!backend && (rotations.d_rotations || rotations.d_rotations_t))
            throw std::runtime_error(
                "[TurboQuant CUDA] CUDA backend unavailable during rotation teardown");
        if (rotations.d_rotations)
        {
            backend->free(rotations.d_rotations, rotations.device_id);
            rotations.d_rotations = nullptr;
        }
        if (rotations.d_rotations_t)
        {
            backend->free(rotations.d_rotations_t, rotations.device_id);
            rotations.d_rotations_t = nullptr;
        }
        rotations.device_id = -1;
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
        __shared__ float s_scaled[D]; // max head_dim = 128
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
        __shared__ float s_scaled[D];
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
        __shared__ uint8_t s_indices[D]; // full 4-bit indices
        s_indices[tid] = static_cast<uint8_t>(idx);
        __syncthreads();

        // Write TQ4Block: [norm(4B), residual_norm(4B), mse_indices[D*3/8], high_bits[D/8]]
        constexpr size_t MSE_BYTES = D * 3 / 8;
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

    template <int D, bool VUsesTQ8>
    __global__ void tq_quantize_fused_ring_kernel(
        const float *__restrict__ d_K_input,   // position-major or verifier head-major rows
        const float *__restrict__ d_V_input,   // position-major or verifier head-major rows
        const float *__restrict__ d_rotations, // [n_kv_heads * D * D]
        uint8_t *__restrict__ d_K_ring,        // K ring buffer
        uint8_t *__restrict__ d_V_ring,        // V ring buffer
        int ring_pos_scalar,                   // Used when d_ring_pos_ptr is null
        const int *d_ring_pos_ptr,             // When non-null, read ring_pos from device memory
        const int *d_row_count,                // Optional real rows inside fixed bucket geometry
        int max_seq_len,
        int verifier_rows,
        int n_kv_heads,
        bool k_head_major,
        bool v_head_major)
    {
        const int verifier_row = blockIdx.z;
        const int rows_to_write = d_row_count ? *d_row_count : verifier_rows;
        const int source_start = rows_to_write > max_seq_len
                                     ? rows_to_write - max_seq_len
                                     : 0;
        if (rows_to_write <= 0 || rows_to_write > verifier_rows ||
            verifier_row < source_start || verifier_row >= rows_to_write)
            return;

        const int first_ring_pos = d_ring_pos_ptr ? *d_ring_pos_ptr : ring_pos_scalar;
        const int ring_pos = max_seq_len > 0
                                 ? (first_ring_pos + verifier_row) % max_seq_len
                                 : first_ring_pos;
        const int head = blockIdx.x;
        const int phase = blockIdx.y; // 0=K(TQ8), 1=V(TQ4)
        const int tid = threadIdx.x;

        if (tid >= D)
            return;

        const size_t k_index = k_head_major
                                   ? (static_cast<size_t>(head) * verifier_rows + verifier_row) * D + tid
                                   : (static_cast<size_t>(verifier_row) * n_kv_heads + head) * D + tid;
        const size_t v_index = v_head_major
                                   ? (static_cast<size_t>(head) * verifier_rows + verifier_row) * D + tid
                                   : (static_cast<size_t>(verifier_row) * n_kv_heads + head) * D + tid;
        const float x = (phase == 0) ? d_K_input[k_index] : d_V_input[v_index];

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
        __shared__ float s_scaled[D];
        s_scaled[tid] = scaled;
        __syncthreads();

        const float *rot_row = d_rotations + static_cast<size_t>(head) * D * D + tid * D;
        float rotated = 0.0f;
        for (int j = 0; j < D; ++j)
            rotated += rot_row[j] * s_scaled[j];

        if (phase == 0 || VUsesTQ8)
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
            uint8_t *ring = phase == 0 ? d_K_ring : d_V_ring;
            uint8_t *block_ptr = ring +
                                 (static_cast<size_t>(ring_pos) * n_kv_heads + head) *
                                     block_size;

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
            __shared__ uint8_t s_indices[D];
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

    /**
     * @brief Copy device-resident prepared TQ rows into wrapped ring slots.
     *
     * `blockIdx.y` selects K or V while the linear X dimension spans every
     * byte of every row.  This replaces the old host loop of per-row H2D
     * copies with one graph-capturable D2D launch.
     */
    __global__ void tq_copy_prepared_rows_ring_kernel(
        const uint8_t *__restrict__ source_k,
        const uint8_t *__restrict__ source_v,
        uint8_t *__restrict__ ring_k,
        uint8_t *__restrict__ ring_v,
        int ring_head_scalar,
        const int *__restrict__ d_ring_head,
        const int *__restrict__ d_row_count,
        int max_seq_len,
        int rows,
        size_t k_row_bytes,
        size_t v_row_bytes,
        bool k_head_major,
        bool v_head_major,
        int n_kv_heads)
    {
        const bool copy_v = blockIdx.y != 0;
        const size_t row_bytes = copy_v ? v_row_bytes : k_row_bytes;
        const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t total_bytes = static_cast<size_t>(rows) * row_bytes;
        if (index >= total_bytes)
            return;

        const int source_row = static_cast<int>(index / row_bytes);
        const int rows_to_write = d_row_count ? *d_row_count : rows;
        const int source_start = rows_to_write > max_seq_len
                                     ? rows_to_write - max_seq_len
                                     : 0;
        if (rows_to_write <= 0 || rows_to_write > rows ||
            source_row < source_start || source_row >= rows_to_write)
            return;
        const size_t byte_in_row = index - static_cast<size_t>(source_row) * row_bytes;
        const int first_ring_row = d_ring_head ? *d_ring_head : ring_head_scalar;
        const int destination_row = (first_ring_row + source_row) % max_seq_len;
        const bool head_major = copy_v ? v_head_major : k_head_major;
        const size_t head_bytes = row_bytes / static_cast<size_t>(n_kv_heads);
        const int head = static_cast<int>(byte_in_row / head_bytes);
        const size_t byte_in_head = byte_in_row - static_cast<size_t>(head) * head_bytes;
        const size_t source_index = head_major
                                        ? (static_cast<size_t>(head) * rows + source_row) * head_bytes + byte_in_head
                                        : index;
        const uint8_t *source = copy_v ? source_v : source_k;
        uint8_t *destination = copy_v ? ring_v : ring_k;
        destination[static_cast<size_t>(destination_row) * row_bytes + byte_in_row] =
            source[source_index];
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
        __shared__ float s_centroid[D];
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
        __shared__ float s_centroid[D];
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
        const int *__restrict__ d_ring_head,
        const int *__restrict__ d_cached_count,
        int max_seq_len, int kv_dim, int position_start)
    {
        int ring_pos = ring_pos_scalar;
        int out_offset_elems = out_offset_elems_scalar;
        int rope_position = rope_position_scalar;
        if (d_ring_head && d_cached_count)
        {
            const int cached_count = *d_cached_count;
            const int ring_head = *d_ring_head;
            if (cached_count <= 0 || cached_count > max_seq_len ||
                ring_head < 0 || ring_head >= max_seq_len || kv_dim <= 0)
            {
                return;
            }

            /*
             * Append advances device head/count before this kernel executes on
             * the same stream. The newest compressed row is therefore one slot
             * behind head, and its linear scratch row is count - 1. Deriving
             * these values here removes the former pinned-host dequant mailbox
             * without adding a planning kernel or another graph node.
             */
            ring_pos = (ring_head + max_seq_len - 1) % max_seq_len;
            out_offset_elems = (cached_count - 1) * kv_dim;
            rope_position =
                rope_theta > 0.0f ? position_start + cached_count - 1 : 0;
        }

        const int head = blockIdx.x;
        const int tid = threadIdx.x;
        if (tid >= D)
            return;

        const float inv_scale = 1.0f / sqrtf(static_cast<float>(D));
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
                /*
                 * Every warp must finish reading the centroid vector for its
                 * inverse-rotation dot products before any lane repurposes
                 * s_cent for RoPE pair exchange. Without this phase boundary,
                 * the four-warp D=128 specialization intermittently let an
                 * early warp overwrite values still consumed by a later warp,
                 * making graph replay neither deterministic nor serial-decode
                 * equivalent.
                 */
                __syncthreads();
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

    /**
     * @brief Canonical RoPE pair arithmetic shared by every CUDA cache format.
     *
     * Explicit `fmaf` calls fix the arithmetic contraction and rounding contract.
     * That contract is used by serial TQ dequantization, grouped TQ
     * dequantization, and ordinary converted cache reads.
     */
    __device__ __forceinline__ void canonical_rope_pair(
        float x,
        float y,
        float rope_theta,
        int pair,
        int rope_dim,
        int position,
        float *rotated_x,
        float *rotated_y)
    {
        const float frequency = 1.0f / powf(
            rope_theta,
            static_cast<float>(2 * pair) / static_cast<float>(rope_dim));
        const float angle = static_cast<float>(position) * frequency;
        const float cosine = cosf(angle);
        const float sine = sinf(angle);
        const float x_cosine = x * cosine;
        const float y_cosine = y * cosine;
        *rotated_x = fmaf(-y, sine, x_cosine);
        *rotated_y = fmaf(x, sine, y_cosine);
    }

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
        float rope_theta, int position_start, int rope_dim,
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

            const int effective_rope_dim =
                rope_dim > 0 ? min(rope_dim, D) : D;
            const int half_dim = effective_rope_dim / 2;

            for (int t = 0; t < tile_count; ++t)
            {
                const float val = vals[t] * norms[t];
                s_cents[t * D + tid] = val; // reuse s_cents for RoPE exchange
                __syncthreads();

                vals[t] = val;
                if (tid < effective_rope_dim)
                {
                    const int pair = tid < half_dim ? tid : tid - half_dim;
                    const float partner = s_cents[
                        t * D +
                        (tid < half_dim ? tid + half_dim : tid - half_dim)];
                    float rotated_x = 0.0f;
                    float rotated_y = 0.0f;
                    if (tid < half_dim)
                    {
                        canonical_rope_pair(
                            val, partner, rope_theta, pair,
                            effective_rope_dim,
                            position_start + tile_start + t,
                            &rotated_x, &rotated_y);
                        vals[t] = rotated_x;
                    }
                    else
                    {
                        canonical_rope_pair(
                            partner, val, rope_theta, pair,
                            effective_rope_dim,
                            position_start + tile_start + t,
                            &rotated_x, &rotated_y);
                        vals[t] = rotated_y;
                    }
                }

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

    /**
     * @brief Grouped TQ8 ring dequantization driven by resident cache metadata.
     *
     * The inner centroid load, loop-interchanged rotation multiply, and FP16
     * rounding intentionally match `tq8_dequantize_tiled_kernel`. The only new
     * dimension is `blockIdx.z`, which selects an independent request ring.
     */
    template <int D, int TILE>
    __global__ void tq8_batched_ring_dequant_fp16_device_state_kernel(
        __half *__restrict__ output,
        const void *const *__restrict__ entry_table,
        const int *__restrict__ heads,
        const int *__restrict__ counts,
        const float *__restrict__ rotations,
        int entry_offset,
        int request_count,
        int max_kv_len,
        int max_seq_len,
        int n_kv_heads,
        float rope_theta,
        int position_start,
        int rope_dim)
    {
        const int head = static_cast<int>(blockIdx.x);
        const int request = static_cast<int>(blockIdx.z);
        const int tid = static_cast<int>(threadIdx.x);
        if (request >= request_count || head >= n_kv_heads || tid >= D)
            return;

        const int entry = entry_offset + request;
        int ring_count = counts[entry];
        ring_count = ring_count < 0
                         ? 0
                         : (ring_count > max_seq_len ? max_seq_len : ring_count);
        const int visible_count =
            ring_count < max_kv_len ? ring_count : max_kv_len;
        const int skipped_rows = ring_count - visible_count;
        int tail = heads[entry] - ring_count;
        tail %= max_seq_len;
        if (tail < 0)
            tail += max_seq_len;

        const auto *cache =
            static_cast<const uint8_t *>(entry_table[entry]);
        const float *rotation =
            rotations + static_cast<size_t>(head) * D * D;
        constexpr size_t block_size = sizeof(TQ8Block<D>);
        const float inv_scale = 1.0f / sqrtf(static_cast<float>(D));
        const int kv_dim = n_kv_heads * D;
        __shared__ float centroids[D * TILE];
        for (int tile_start = static_cast<int>(blockIdx.y) * TILE;
             tile_start < max_kv_len;
             tile_start += static_cast<int>(gridDim.y) * TILE)
        {
            float norms[TILE];
            const int output_tile_count =
                min(TILE, max_kv_len - tile_start);

#pragma unroll
        for (int tile = 0; tile < TILE; ++tile)
        {
            const int output_token = tile_start + tile;
            if (tile < output_tile_count && output_token < visible_count)
            {
                const int source_token =
                    (tail + skipped_rows + output_token) % max_seq_len;
                const uint8_t *block = cache +
                    (static_cast<size_t>(source_token) * n_kv_heads + head) *
                        block_size;
                norms[tile] = reinterpret_cast<const float *>(block)[0];
                const uint8_t index = block[2 * sizeof(float) + tid];
                centroids[tid * TILE + tile] =
                    d_TQ8_CENTROIDS[index] * inv_scale;
            }
            else
            {
                norms[tile] = 0.0f;
                centroids[tid * TILE + tile] = 0.0f;
            }
        }
        __syncthreads();

        float values[TILE];
#pragma unroll
        for (int tile = 0; tile < TILE; ++tile)
            values[tile] = 0.0f;
        for (int dimension = 0; dimension < D; ++dimension)
        {
            const float coefficient = rotation[dimension * D + tid];
#pragma unroll
            for (int tile = 0; tile < TILE; ++tile)
                values[tile] +=
                    coefficient * centroids[dimension * TILE + tile];
        }

        if (rope_theta > 0.0f)
        {
            __syncthreads();
            const int effective_rope_dim =
                rope_dim > 0 ? min(rope_dim, D) : D;
            const int half_dim = effective_rope_dim / 2;
            for (int tile = 0; tile < output_tile_count; ++tile)
            {
                const int output_token = tile_start + tile;
                const size_t output_index =
                    (static_cast<size_t>(request) * max_kv_len +
                     static_cast<size_t>(output_token)) *
                        static_cast<size_t>(kv_dim) +
                    static_cast<size_t>(head * D + tid);
                if (output_token >= visible_count)
                {
                    output[output_index] = __float2half_rn(0.0f);
                    __syncthreads();
                    continue;
                }

                float value = values[tile] * norms[tile];
                centroids[tile * D + tid] = value;
                __syncthreads();
                if (tid < effective_rope_dim)
                {
                    const int pair = tid < half_dim ? tid : tid - half_dim;
                    const float partner = centroids[
                        tile * D +
                        (tid < half_dim ? tid + half_dim : tid - half_dim)];
                    float rotated_x = 0.0f;
                    float rotated_y = 0.0f;
                    if (tid < half_dim)
                    {
                        canonical_rope_pair(
                            value, partner, rope_theta, pair,
                            effective_rope_dim,
                            position_start + output_token,
                            &rotated_x, &rotated_y);
                        value = rotated_x;
                    }
                    else
                    {
                        canonical_rope_pair(
                            partner, value, rope_theta, pair,
                            effective_rope_dim,
                            position_start + output_token,
                            &rotated_x, &rotated_y);
                        value = rotated_y;
                    }
                }
                output[output_index] = __float2half(value);
                __syncthreads();
            }
        }
        else
        {
            for (int tile = 0; tile < output_tile_count; ++tile)
            {
                const int output_token = tile_start + tile;
                const size_t output_index =
                    (static_cast<size_t>(request) * max_kv_len +
                     static_cast<size_t>(output_token)) *
                        static_cast<size_t>(kv_dim) +
                    static_cast<size_t>(head * D + tid);
                output[output_index] = output_token < visible_count
                                           ? __float2half(values[tile] * norms[tile])
                                           : __float2half_rn(0.0f);
            }
        }
            __syncthreads();
        }
    }

    /**
     * @brief Grouped TQ4 ring dequantization driven by resident cache metadata.
     */
    template <int D, int TILE>
    __global__ void tq4_batched_ring_dequant_fp16_device_state_kernel(
        __half *__restrict__ output,
        const void *const *__restrict__ entry_table,
        const int *__restrict__ heads,
        const int *__restrict__ counts,
        const float *__restrict__ rotations,
        int entry_offset,
        int request_count,
        int max_kv_len,
        int max_seq_len,
        int n_kv_heads)
    {
        const int head = static_cast<int>(blockIdx.x);
        const int request = static_cast<int>(blockIdx.z);
        const int tid = static_cast<int>(threadIdx.x);
        if (request >= request_count || head >= n_kv_heads || tid >= D)
            return;

        const int entry = entry_offset + request;
        int ring_count = counts[entry];
        ring_count = ring_count < 0
                         ? 0
                         : (ring_count > max_seq_len ? max_seq_len : ring_count);
        const int visible_count =
            ring_count < max_kv_len ? ring_count : max_kv_len;
        const int skipped_rows = ring_count - visible_count;
        int tail = heads[entry] - ring_count;
        tail %= max_seq_len;
        if (tail < 0)
            tail += max_seq_len;

        const auto *cache =
            static_cast<const uint8_t *>(entry_table[entry]);
        const float *rotation =
            rotations + static_cast<size_t>(head) * D * D;
        constexpr size_t block_size = sizeof(TQ4Block<D>);
        constexpr size_t mse_bytes = D * 3 / 8;
        const float inv_scale = rsqrtf(static_cast<float>(D));
        const int kv_dim = n_kv_heads * D;
        const int group8 = tid / 8;
        const int within = tid % 8;
        const int byte_index = group8 * 3;
        __shared__ float centroids[D * TILE];
        for (int tile_start = static_cast<int>(blockIdx.y) * TILE;
             tile_start < max_kv_len;
             tile_start += static_cast<int>(gridDim.y) * TILE)
        {
            float norms[TILE];
            const int output_tile_count =
                min(TILE, max_kv_len - tile_start);

#pragma unroll
        for (int tile = 0; tile < TILE; ++tile)
        {
            const int output_token = tile_start + tile;
            if (tile < output_tile_count && output_token < visible_count)
            {
                const int source_token =
                    (tail + skipped_rows + output_token) % max_seq_len;
                const uint8_t *block = cache +
                    (static_cast<size_t>(source_token) * n_kv_heads + head) *
                        block_size;
                norms[tile] = reinterpret_cast<const float *>(block)[0];
                const uint8_t *mse = block + 2 * sizeof(float);
                const uint8_t *high = mse + mse_bytes;
                const uint8_t byte0 = mse[byte_index];
                const uint8_t byte1 = mse[byte_index + 1];
                const uint8_t byte2 = mse[byte_index + 2];
                uint8_t low3 = 0;
                switch (within)
                {
                case 0: low3 = byte0 & 0x07; break;
                case 1: low3 = (byte0 >> 3) & 0x07; break;
                case 2: low3 = ((byte0 >> 6) | (byte1 << 2)) & 0x07; break;
                case 3: low3 = (byte1 >> 1) & 0x07; break;
                case 4: low3 = (byte1 >> 4) & 0x07; break;
                case 5: low3 = ((byte1 >> 7) | (byte2 << 1)) & 0x07; break;
                case 6: low3 = (byte2 >> 2) & 0x07; break;
                case 7: low3 = (byte2 >> 5) & 0x07; break;
                }
                const uint8_t high1 =
                    (high[tid / 8] >> (tid % 8)) & 0x01;
                const uint8_t index = low3 | (high1 << 3);
                centroids[tid * TILE + tile] =
                    d_TQ4_CENTROIDS[index] * inv_scale;
            }
            else
            {
                norms[tile] = 0.0f;
                centroids[tid * TILE + tile] = 0.0f;
            }
        }
        __syncthreads();

        float values[TILE];
#pragma unroll
        for (int tile = 0; tile < TILE; ++tile)
            values[tile] = 0.0f;
        for (int dimension = 0; dimension < D; ++dimension)
        {
            const float coefficient = rotation[dimension * D + tid];
#pragma unroll
            for (int tile = 0; tile < TILE; ++tile)
                values[tile] +=
                    coefficient * centroids[dimension * TILE + tile];
        }

        for (int tile = 0; tile < output_tile_count; ++tile)
        {
            const int output_token = tile_start + tile;
            const size_t output_index =
                (static_cast<size_t>(request) * max_kv_len +
                 static_cast<size_t>(output_token)) *
                    static_cast<size_t>(kv_dim) +
                static_cast<size_t>(head * D + tid);
            output[output_index] = output_token < visible_count
                                       ? __float2half(values[tile] * norms[tile])
                                       : __float2half_rn(0.0f);
        }
            __syncthreads();
        }
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

        // Half-split: pair (i, i + half_dim)
        const int head_base = pos * kv_dim + head * head_dim;
        const int idx0 = head_base + pair_idx;
        const int idx1 = head_base + pair_idx + half_dim;
        float x = __half2float(d_K[idx0]);
        float y = __half2float(d_K[idx1]);

        float rotated_x = 0.0f;
        float rotated_y = 0.0f;
        canonical_rope_pair(
            x, y, rope_theta, pair_idx, effective_rope_dim,
            position_start + pos, &rotated_x, &rotated_y);
        d_K[idx0] = __float2half_rn(rotated_x);
        d_K[idx1] = __float2half_rn(rotated_y);
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

        // Half-split: pair (i, i + half_dim)
        const int head_base = pos * kv_dim + head * head_dim;
        const int idx0 = head_base + pair_idx;
        const int idx1 = head_base + pair_idx + half_dim;
        float x = d_K[idx0];
        float y = d_K[idx1];

        canonical_rope_pair(
            x, y, rope_theta, pair_idx, effective_rope_dim,
            position_start + pos, &d_K[idx0], &d_K[idx1]);
    }

    __global__ void rope_apply_batched_fp16_device_state_kernel(
        __half *__restrict__ d_K,
        const int *__restrict__ counts,
        int entry_offset,
        int request_count,
        int max_kv_len,
        int max_seq_len,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        int position_start,
        int rope_dim)
    {
        const int effective_rope_dim = rope_dim > 0 ? rope_dim : head_dim;
        const int half_dim = effective_rope_dim / 2;
        const int request = static_cast<int>(blockIdx.y);
        if (request >= request_count)
            return;

        const int pairs_per_token = n_kv_heads * half_dim;
        const int ring_count = min(max(counts[entry_offset + request], 0),
                                   max_seq_len);
        const int visible_count = min(ring_count, max_kv_len);
        const int live_pairs = visible_count * pairs_per_token;
        const int kv_dim = n_kv_heads * head_dim;
        for (int request_pair =
                 static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
             request_pair < live_pairs;
             request_pair += static_cast<int>(gridDim.x * blockDim.x))
        {
            const int token = request_pair / pairs_per_token;
            const int token_pair = request_pair % pairs_per_token;
            const int head = token_pair / half_dim;
            const int pair = token_pair % half_dim;
            const size_t base =
                (static_cast<size_t>(request) * max_kv_len + token) * kv_dim +
                static_cast<size_t>(head) * head_dim;
            const size_t first = base + pair;
            const size_t second = base + pair + half_dim;
            float rotated_x = 0.0f;
            float rotated_y = 0.0f;
            canonical_rope_pair(
                __half2float(d_K[first]), __half2float(d_K[second]),
                rope_theta, pair, effective_rope_dim,
                position_start + token, &rotated_x, &rotated_y);
            d_K[first] = __float2half_rn(rotated_x);
            d_K[second] = __float2half_rn(rotated_y);
        }
    }

    __global__ void rope_apply_batched_fp32_ring_to_fp16_device_state_kernel(
        __half *__restrict__ d_K_out,
        const float *const *__restrict__ entry_table,
        const int *__restrict__ heads,
        const int *__restrict__ counts,
        int entry_offset,
        int request_count,
        int max_kv_len,
        int max_seq_len,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        int position_start,
        int rope_dim)
    {
        const int effective_rope_dim = rope_dim > 0 ? rope_dim : head_dim;
        const int half_dim = effective_rope_dim / 2;
        const int request = static_cast<int>(blockIdx.y);
        if (request >= request_count)
            return;

        const int pairs_per_token = n_kv_heads * half_dim;
        const int entry = entry_offset + request;
        const int ring_count = min(max(counts[entry], 0), max_seq_len);
        const int visible_count = min(ring_count, max_kv_len);
        const int skipped_rows = ring_count - visible_count;
        const int tail = (heads[entry] - ring_count + max_seq_len) % max_seq_len;
        const int kv_dim = n_kv_heads * head_dim;
        const int live_pairs = visible_count * pairs_per_token;
        for (int request_pair =
                 static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
             request_pair < live_pairs;
             request_pair += static_cast<int>(gridDim.x * blockDim.x))
        {
            const int output_token = request_pair / pairs_per_token;
            const int token_pair = request_pair % pairs_per_token;
            const int head = token_pair / half_dim;
            const int pair = token_pair % half_dim;
            const int source_token =
                (tail + skipped_rows + output_token) % max_seq_len;
            const size_t source_base =
                static_cast<size_t>(source_token) * kv_dim +
                static_cast<size_t>(head) * head_dim;
            float rotated_x = 0.0f;
            float rotated_y = 0.0f;
            canonical_rope_pair(
                entry_table[entry][source_base + pair],
                entry_table[entry][source_base + pair + half_dim],
                rope_theta, pair, effective_rope_dim,
                position_start + output_token, &rotated_x, &rotated_y);
            const size_t output_base =
                (static_cast<size_t>(request) * max_kv_len + output_token) *
                    kv_dim +
                static_cast<size_t>(head) * head_dim;
            d_K_out[output_base + pair] = __float2half_rn(rotated_x);
            d_K_out[output_base + pair + half_dim] =
                __float2half_rn(rotated_y);
        }
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

    extern "C" bool cuda_tq_quantize_grouped_ring(
        const float *d_K_input, const float *d_V_input,
        const float *d_rotations,
        void *d_K_ring, void *d_V_ring,
        int ring_head, int max_seq_len,
        int verifier_rows, int n_kv_heads, int head_dim,
        bool k_head_major, bool v_head_major,
        TurboQuantKVMode mode,
        cudaStream_t stream)
    {
        if (!d_K_input || !d_V_input || !d_rotations || !d_K_ring || !d_V_ring ||
            !stream || max_seq_len <= 0 || verifier_rows <= 0 || verifier_rows > 65535 ||
            n_kv_heads <= 0)
        {
            return false;
        }

        const dim3 grid(n_kv_heads, 2, verifier_rows);
        if (head_dim == 64)
        {
            if (mode == TurboQuantKVMode::TQ8_K_TQ8_V)
                tq_quantize_fused_ring_kernel<64, true><<<grid, dim3(64), 0, stream>>>(
                    d_K_input, d_V_input, d_rotations,
                    static_cast<uint8_t *>(d_K_ring), static_cast<uint8_t *>(d_V_ring),
                    ring_head, nullptr, nullptr, max_seq_len, verifier_rows, n_kv_heads,
                    k_head_major, v_head_major);
            else
                tq_quantize_fused_ring_kernel<64, false><<<grid, dim3(64), 0, stream>>>(
                    d_K_input, d_V_input, d_rotations,
                    static_cast<uint8_t *>(d_K_ring), static_cast<uint8_t *>(d_V_ring),
                    ring_head, nullptr, nullptr, max_seq_len, verifier_rows, n_kv_heads,
                    k_head_major, v_head_major);
        }
        else if (head_dim == 128)
        {
            if (mode == TurboQuantKVMode::TQ8_K_TQ8_V)
                tq_quantize_fused_ring_kernel<128, true><<<grid, dim3(128), 0, stream>>>(
                    d_K_input, d_V_input, d_rotations,
                    static_cast<uint8_t *>(d_K_ring), static_cast<uint8_t *>(d_V_ring),
                    ring_head, nullptr, nullptr, max_seq_len, verifier_rows, n_kv_heads,
                    k_head_major, v_head_major);
            else
                tq_quantize_fused_ring_kernel<128, false><<<grid, dim3(128), 0, stream>>>(
                    d_K_input, d_V_input, d_rotations,
                    static_cast<uint8_t *>(d_K_ring), static_cast<uint8_t *>(d_V_ring),
                    ring_head, nullptr, nullptr, max_seq_len, verifier_rows, n_kv_heads,
                    k_head_major, v_head_major);
        }
        else if (head_dim == 256)
        {
            if (mode == TurboQuantKVMode::TQ8_K_TQ8_V)
                tq_quantize_fused_ring_kernel<256, true><<<grid, dim3(256), 0, stream>>>(
                    d_K_input, d_V_input, d_rotations,
                    static_cast<uint8_t *>(d_K_ring), static_cast<uint8_t *>(d_V_ring),
                    ring_head, nullptr, nullptr, max_seq_len, verifier_rows, n_kv_heads,
                    k_head_major, v_head_major);
            else
                tq_quantize_fused_ring_kernel<256, false><<<grid, dim3(256), 0, stream>>>(
                    d_K_input, d_V_input, d_rotations,
                    static_cast<uint8_t *>(d_K_ring), static_cast<uint8_t *>(d_V_ring),
                    ring_head, nullptr, nullptr, max_seq_len, verifier_rows, n_kv_heads,
                    k_head_major, v_head_major);
        }
        else
        {
            return false;
        }
        return cudaGetLastError() == cudaSuccess;
    }

    extern "C" bool cuda_tq_quantize_grouped_ring_dynamic(
        const float *d_K_input, const float *d_V_input,
        const float *d_rotations,
        void *d_K_ring, void *d_V_ring,
        const int *d_ring_head, const int *d_row_count, int max_seq_len,
        int verifier_rows, int n_kv_heads, int head_dim,
        bool k_head_major, bool v_head_major,
        TurboQuantKVMode mode,
        cudaStream_t stream)
    {
        if (!d_ring_head || !d_K_input || !d_V_input || !d_rotations ||
            !d_K_ring || !d_V_ring || !stream || max_seq_len <= 0 ||
            verifier_rows <= 0 || verifier_rows > 65535 || n_kv_heads <= 0)
        {
            return false;
        }

        const dim3 grid(n_kv_heads, 2, verifier_rows);
        if (head_dim == 64)
        {
            if (mode == TurboQuantKVMode::TQ8_K_TQ8_V)
                tq_quantize_fused_ring_kernel<64, true><<<grid, dim3(64), 0, stream>>>(
                    d_K_input, d_V_input, d_rotations,
                    static_cast<uint8_t *>(d_K_ring), static_cast<uint8_t *>(d_V_ring),
                    0, d_ring_head, d_row_count, max_seq_len, verifier_rows, n_kv_heads,
                    k_head_major, v_head_major);
            else
                tq_quantize_fused_ring_kernel<64, false><<<grid, dim3(64), 0, stream>>>(
                    d_K_input, d_V_input, d_rotations,
                    static_cast<uint8_t *>(d_K_ring), static_cast<uint8_t *>(d_V_ring),
                    0, d_ring_head, d_row_count, max_seq_len, verifier_rows, n_kv_heads,
                    k_head_major, v_head_major);
        }
        else if (head_dim == 128)
        {
            if (mode == TurboQuantKVMode::TQ8_K_TQ8_V)
                tq_quantize_fused_ring_kernel<128, true><<<grid, dim3(128), 0, stream>>>(
                    d_K_input, d_V_input, d_rotations,
                    static_cast<uint8_t *>(d_K_ring), static_cast<uint8_t *>(d_V_ring),
                    0, d_ring_head, d_row_count, max_seq_len, verifier_rows, n_kv_heads,
                    k_head_major, v_head_major);
            else
                tq_quantize_fused_ring_kernel<128, false><<<grid, dim3(128), 0, stream>>>(
                    d_K_input, d_V_input, d_rotations,
                    static_cast<uint8_t *>(d_K_ring), static_cast<uint8_t *>(d_V_ring),
                    0, d_ring_head, d_row_count, max_seq_len, verifier_rows, n_kv_heads,
                    k_head_major, v_head_major);
        }
        else if (head_dim == 256)
        {
            if (mode == TurboQuantKVMode::TQ8_K_TQ8_V)
                tq_quantize_fused_ring_kernel<256, true><<<grid, dim3(256), 0, stream>>>(
                    d_K_input, d_V_input, d_rotations,
                    static_cast<uint8_t *>(d_K_ring), static_cast<uint8_t *>(d_V_ring),
                    0, d_ring_head, d_row_count, max_seq_len, verifier_rows, n_kv_heads,
                    k_head_major, v_head_major);
            else
                tq_quantize_fused_ring_kernel<256, false><<<grid, dim3(256), 0, stream>>>(
                    d_K_input, d_V_input, d_rotations,
                    static_cast<uint8_t *>(d_K_ring), static_cast<uint8_t *>(d_V_ring),
                    0, d_ring_head, d_row_count, max_seq_len, verifier_rows, n_kv_heads,
                    k_head_major, v_head_major);
        }
        else
        {
            return false;
        }
        return cudaGetLastError() == cudaSuccess;
    }

    namespace
    {
        /** @brief Shared prepared-row launcher for static and device-resident heads. */
        bool launch_cuda_tq_copy_prepared_rows_ring(
            const void *source_k, const void *source_v,
            void *ring_k, void *ring_v,
            int ring_head, const int *d_ring_head,
            const int *d_row_count,
            int max_seq_len, int rows,
            size_t k_row_bytes, size_t v_row_bytes,
            bool k_head_major, bool v_head_major,
            int n_kv_heads,
            cudaStream_t stream)
        {
            if (!source_k || !source_v || !ring_k || !ring_v || !stream ||
                max_seq_len <= 0 || rows <= 0 || k_row_bytes == 0 || v_row_bytes == 0 ||
                n_kv_heads <= 0 || k_row_bytes % static_cast<size_t>(n_kv_heads) != 0 ||
                v_row_bytes % static_cast<size_t>(n_kv_heads) != 0)
            {
                return false;
            }
            const size_t largest_row_bytes = k_row_bytes > v_row_bytes
                                                 ? k_row_bytes
                                                 : v_row_bytes;
            const size_t largest_bytes = static_cast<size_t>(rows) * largest_row_bytes;
            constexpr int threads = 256;
            const dim3 grid(static_cast<unsigned>((largest_bytes + threads - 1) / threads), 2);
            tq_copy_prepared_rows_ring_kernel<<<grid, threads, 0, stream>>>(
                static_cast<const uint8_t *>(source_k),
                static_cast<const uint8_t *>(source_v),
                static_cast<uint8_t *>(ring_k), static_cast<uint8_t *>(ring_v),
                ring_head, d_ring_head, d_row_count, max_seq_len, rows,
                k_row_bytes, v_row_bytes, k_head_major, v_head_major, n_kv_heads);
            return cudaGetLastError() == cudaSuccess;
        }
    } // namespace

    extern "C" bool cuda_tq_copy_prepared_rows_ring(
        const void *source_k, const void *source_v,
        void *ring_k, void *ring_v,
        int ring_head, int max_seq_len, int rows,
        size_t k_row_bytes, size_t v_row_bytes,
        bool k_head_major, bool v_head_major,
        int n_kv_heads,
        cudaStream_t stream)
    {
        return launch_cuda_tq_copy_prepared_rows_ring(
            source_k, source_v, ring_k, ring_v,
            ring_head, nullptr, nullptr, max_seq_len, rows,
            k_row_bytes, v_row_bytes, k_head_major, v_head_major,
            n_kv_heads, stream);
    }

    extern "C" bool cuda_tq_copy_prepared_rows_ring_dynamic(
        const void *source_k, const void *source_v,
        void *ring_k, void *ring_v,
        const int *d_ring_head, const int *d_row_count,
        int max_seq_len, int rows,
        size_t k_row_bytes, size_t v_row_bytes,
        bool k_head_major, bool v_head_major,
        int n_kv_heads,
        cudaStream_t stream)
    {
        if (!d_ring_head)
            return false;
        return launch_cuda_tq_copy_prepared_rows_ring(
            source_k, source_v, ring_k, ring_v,
            0, d_ring_head, d_row_count, max_seq_len, rows,
            k_row_bytes, v_row_bytes, k_head_major, v_head_major,
            n_kv_heads, stream);
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
                count, n_kv_heads, rope_theta, position_start, head_dim,
                max_seq_len, tail);
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
                count, n_kv_heads, rope_theta, position_start, head_dim,
                max_seq_len, tail);
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
        float rope_theta, int position_start, int rope_dim,
        TurboQuantKVMode mode,
        cudaStream_t stream)
    {
        if (count <= 0)
            return true;

        constexpr int TILE = 16;

        if (head_dim == 64)
        {
            const dim3 grid(n_kv_heads, (count + TILE - 1) / TILE);
            const dim3 block(64);
            tq8_dequantize_tiled_kernel<64, TILE, __half><<<grid, block, 0, stream>>>(
                static_cast<const uint8_t *>(d_K_cache), d_K_rotations, d_K_out,
                count, n_kv_heads, rope_theta, position_start, rope_dim,
                max_seq_len, tail);
            if (mode == TurboQuantKVMode::TQ8_K_TQ8_V)
                tq8_dequantize_tiled_kernel<64, TILE, __half><<<grid, block, 0, stream>>>(
                    static_cast<const uint8_t *>(d_V_cache), d_V_rotations, d_V_out,
                    count, n_kv_heads, 0.0f, 0, 0, max_seq_len, tail);
            else
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
                count, n_kv_heads, rope_theta, position_start, rope_dim,
                max_seq_len, tail);
            if (mode == TurboQuantKVMode::TQ8_K_TQ8_V)
                tq8_dequantize_tiled_kernel<128, TILE, __half><<<grid, block, 0, stream>>>(
                    static_cast<const uint8_t *>(d_V_cache), d_V_rotations, d_V_out,
                    count, n_kv_heads, 0.0f, 0, 0, max_seq_len, tail);
            else
                tq4_dequantize_tiled_kernel<128, TILE, __half><<<grid, block, 0, stream>>>(
                    static_cast<const uint8_t *>(d_V_cache), d_V_rotations, d_V_out,
                    count, n_kv_heads, max_seq_len, tail);
        }
        else if (head_dim == 256)
        {
            const dim3 grid(n_kv_heads, (count + TILE - 1) / TILE);
            const dim3 block(256);
            tq8_dequantize_tiled_kernel<256, TILE, __half><<<grid, block, 0, stream>>>(
                static_cast<const uint8_t *>(d_K_cache), d_K_rotations, d_K_out,
                count, n_kv_heads, rope_theta, position_start, rope_dim,
                max_seq_len, tail);
            if (mode == TurboQuantKVMode::TQ8_K_TQ8_V)
                tq8_dequantize_tiled_kernel<256, TILE, __half><<<grid, block, 0, stream>>>(
                    static_cast<const uint8_t *>(d_V_cache), d_V_rotations, d_V_out,
                    count, n_kv_heads, 0.0f, 0, 0, max_seq_len, tail);
            else
                tq4_dequantize_tiled_kernel<256, TILE, __half><<<grid, block, 0, stream>>>(
                    static_cast<const uint8_t *>(d_V_cache), d_V_rotations, d_V_out,
                    count, n_kv_heads, max_seq_len, tail);
        }
        else
        {
            return false;
        }

        return cudaGetLastError() == cudaSuccess;
    }

    extern "C" bool cuda_tq_batched_ring_dequant_fp16_device_state(
        __half *d_K_out,
        __half *d_V_out,
        const void *const *d_K_entry_table,
        const void *const *d_V_entry_table,
        const int *d_heads,
        const int *d_counts,
        const float *d_rotations,
        int entry_offset,
        int request_count,
        int max_kv_len,
        int max_seq_len,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        int position_start,
        int rope_dim,
        TurboQuantKVMode mode,
        cudaStream_t stream)
    {
        if (!d_K_out || !d_V_out || !d_K_entry_table || !d_V_entry_table ||
            !d_heads || !d_counts || !d_rotations || !stream ||
            entry_offset < 0 || request_count <= 0 || max_kv_len <= 0 ||
            max_seq_len <= 0 || n_kv_heads <= 0)
        {
            return false;
        }

        constexpr int tile = 16;
        constexpr unsigned int resident_tile_blocks = 8;
        const dim3 grid(
            static_cast<unsigned int>(n_kv_heads),
            min(
                static_cast<unsigned int>((max_kv_len + tile - 1) / tile),
                resident_tile_blocks),
            static_cast<unsigned int>(request_count));
        if (head_dim == 64)
        {
            tq8_batched_ring_dequant_fp16_device_state_kernel<64, tile>
                <<<grid, dim3(64), 0, stream>>>(
                    d_K_out, d_K_entry_table, d_heads, d_counts, d_rotations,
                    entry_offset, request_count, max_kv_len, max_seq_len,
                    n_kv_heads, rope_theta, position_start, rope_dim);
            if (mode == TurboQuantKVMode::TQ8_K_TQ8_V)
                tq8_batched_ring_dequant_fp16_device_state_kernel<64, tile>
                    <<<grid, dim3(64), 0, stream>>>(
                        d_V_out, d_V_entry_table, d_heads, d_counts, d_rotations,
                        entry_offset, request_count, max_kv_len, max_seq_len,
                        n_kv_heads, 0.0f, 0, 0);
            else
                tq4_batched_ring_dequant_fp16_device_state_kernel<64, tile>
                    <<<grid, dim3(64), 0, stream>>>(
                        d_V_out, d_V_entry_table, d_heads, d_counts, d_rotations,
                        entry_offset, request_count, max_kv_len, max_seq_len,
                        n_kv_heads);
        }
        else if (head_dim == 128)
        {
            tq8_batched_ring_dequant_fp16_device_state_kernel<128, tile>
                <<<grid, dim3(128), 0, stream>>>(
                    d_K_out, d_K_entry_table, d_heads, d_counts, d_rotations,
                    entry_offset, request_count, max_kv_len, max_seq_len,
                    n_kv_heads, rope_theta, position_start, rope_dim);
            if (mode == TurboQuantKVMode::TQ8_K_TQ8_V)
                tq8_batched_ring_dequant_fp16_device_state_kernel<128, tile>
                    <<<grid, dim3(128), 0, stream>>>(
                        d_V_out, d_V_entry_table, d_heads, d_counts, d_rotations,
                        entry_offset, request_count, max_kv_len, max_seq_len,
                        n_kv_heads, 0.0f, 0, 0);
            else
                tq4_batched_ring_dequant_fp16_device_state_kernel<128, tile>
                    <<<grid, dim3(128), 0, stream>>>(
                        d_V_out, d_V_entry_table, d_heads, d_counts, d_rotations,
                        entry_offset, request_count, max_kv_len, max_seq_len,
                        n_kv_heads);
        }
        else if (head_dim == 256)
        {
            tq8_batched_ring_dequant_fp16_device_state_kernel<256, tile>
                <<<grid, dim3(256), 0, stream>>>(
                    d_K_out, d_K_entry_table, d_heads, d_counts, d_rotations,
                    entry_offset, request_count, max_kv_len, max_seq_len,
                    n_kv_heads, rope_theta, position_start, rope_dim);
            if (mode == TurboQuantKVMode::TQ8_K_TQ8_V)
                tq8_batched_ring_dequant_fp16_device_state_kernel<256, tile>
                    <<<grid, dim3(256), 0, stream>>>(
                        d_V_out, d_V_entry_table, d_heads, d_counts, d_rotations,
                        entry_offset, request_count, max_kv_len, max_seq_len,
                        n_kv_heads, 0.0f, 0, 0);
            else
                tq4_batched_ring_dequant_fp16_device_state_kernel<256, tile>
                    <<<grid, dim3(256), 0, stream>>>(
                        d_V_out, d_V_entry_table, d_heads, d_counts, d_rotations,
                        entry_offset, request_count, max_kv_len, max_seq_len,
                        n_kv_heads);
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
                rope_theta, rope_position,
                nullptr, nullptr, 0, 0, 0);
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
                rope_theta, rope_position,
                nullptr, nullptr, 0, 0, 0);
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
        const int *d_ring_head, const int *d_cached_count,
        int max_seq_len, int kv_dim, int position_start,
        int n_kv_heads, int head_dim,
        float rope_theta,
        cudaStream_t stream)
    {
        if (!d_K_base || !d_V_base || !d_K_cache || !d_V_cache ||
            !d_K_rotation || !d_V_rotation ||
            !d_ring_head || !d_cached_count ||
            max_seq_len <= 0 || kv_dim <= 0 || n_kv_heads <= 0)
        {
            return false;
        }
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
                rope_theta, 0,
                d_ring_head, d_cached_count,
                max_seq_len, kv_dim, position_start);
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
                rope_theta, 0,
                d_ring_head, d_cached_count,
                max_seq_len, kv_dim, position_start);
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

    extern "C" bool cuda_rope_apply_batched_fp16_device_state(
        __half *d_K,
        const int *d_counts,
        int entry_offset,
        int request_count,
        int max_kv_len,
        int max_seq_len,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        int position_start,
        int rope_dim,
        cudaStream_t stream)
    {
        const int effective_rope_dim = rope_dim > 0 ? rope_dim : head_dim;
        if (!d_K || !d_counts || !stream || entry_offset < 0 ||
            request_count <= 0 || max_kv_len <= 0 || max_seq_len <= 0 ||
            n_kv_heads <= 0 || head_dim <= 0 || rope_theta <= 0.0f ||
            effective_rope_dim <= 0 || effective_rope_dim > head_dim ||
            (effective_rope_dim % 2) != 0)
        {
            return false;
        }
        const int pairs_per_request =
            max_kv_len * n_kv_heads * (effective_rope_dim / 2);
        constexpr unsigned int resident_rope_blocks = 64;
        const unsigned int blocks = min(
            static_cast<unsigned int>((pairs_per_request + 255) / 256),
            resident_rope_blocks);
        rope_apply_batched_fp16_device_state_kernel<<<
            dim3(blocks, static_cast<unsigned int>(request_count)),
            256, 0, stream>>>(
                d_K, d_counts, entry_offset, request_count, max_kv_len,
                max_seq_len, n_kv_heads, head_dim, rope_theta,
                position_start, effective_rope_dim);
        return cudaGetLastError() == cudaSuccess;
    }

    extern "C" bool cuda_rope_apply_batched_fp32_ring_to_fp16_device_state(
        __half *d_K_out,
        const float *const *d_K_entry_table,
        const int *d_heads,
        const int *d_counts,
        int entry_offset,
        int request_count,
        int max_kv_len,
        int max_seq_len,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        int position_start,
        int rope_dim,
        cudaStream_t stream)
    {
        const int effective_rope_dim = rope_dim > 0 ? rope_dim : head_dim;
        if (!d_K_out || !d_K_entry_table || !d_heads || !d_counts || !stream ||
            entry_offset < 0 || request_count <= 0 || max_kv_len <= 0 ||
            max_seq_len <= 0 || n_kv_heads <= 0 || head_dim <= 0 ||
            rope_theta <= 0.0f || effective_rope_dim <= 0 ||
            effective_rope_dim > head_dim || (effective_rope_dim % 2) != 0)
        {
            return false;
        }
        const int pairs_per_request =
            max_kv_len * n_kv_heads * (effective_rope_dim / 2);
        constexpr unsigned int resident_rope_blocks = 64;
        const unsigned int blocks = min(
            static_cast<unsigned int>((pairs_per_request + 255) / 256),
            resident_rope_blocks);
        rope_apply_batched_fp32_ring_to_fp16_device_state_kernel<<<
            dim3(blocks, static_cast<unsigned int>(request_count)),
            256, 0, stream>>>(
                d_K_out, d_K_entry_table, d_heads, d_counts, entry_offset,
                request_count, max_kv_len, max_seq_len, n_kv_heads, head_dim,
                rope_theta, position_start, effective_rope_dim);
        return cudaGetLastError() == cudaSuccess;
    }

} // namespace llaminar2
