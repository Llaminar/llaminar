/**
 * @file ROCmTurboQuantKernels.hip
 * @brief HIP kernel implementations for TurboQuant KV cache operations
 * @author David Sanftenberg
 *
 * HIP mirror of CUDATurboQuantKernels.cu for AMD GPUs.
 * Uses hipLaunchKernelGGL() launch syntax and HIP FP16 types.
 */

#include "ROCmTurboQuantKernels.h"
#include "../../../kernels/cpu/turboquant/TurboQuantCodebook.h"
#include "../../../kernels/cpu/turboquant/TurboQuantContext.h"
#include "../../../utils/Logger.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstring>
#include <atomic>
#include <vector>

namespace llaminar2
{

    // =========================================================================
    // Constant Memory Codebooks
    // =========================================================================

    __constant__ float d_TQ8_CENTROIDS_HIP[256];
    __constant__ float d_TQ8_THRESHOLDS_HIP[255];
    __constant__ float d_TQ4_CENTROIDS_HIP[16];
    __constant__ float d_TQ4_THRESHOLDS_HIP[15];

    static std::atomic<bool> s_hip_codebooks_uploaded{false};

    // Precomputed RoPE frequencies: freq[i] = 1/theta^(2i/D) for i in [0, D/2)
    __constant__ float d_HIP_ROPE_FREQS[64]; // max head_dim/2 = 128/2 = 64
    static std::atomic<int> s_hip_rope_freqs_head_dim{0};

    void hip_tq_upload_codebooks(hipStream_t stream)
    {
        if (s_hip_codebooks_uploaded.load(std::memory_order_acquire))
            return;

        (void)hipMemcpyToSymbolAsync(HIP_SYMBOL(d_TQ8_CENTROIDS_HIP), TQ8_CENTROIDS.data(),
                               256 * sizeof(float), 0, hipMemcpyHostToDevice, stream);
        (void)hipMemcpyToSymbolAsync(HIP_SYMBOL(d_TQ8_THRESHOLDS_HIP), TQ8_THRESHOLDS.data(),
                               255 * sizeof(float), 0, hipMemcpyHostToDevice, stream);
        (void)hipMemcpyToSymbolAsync(HIP_SYMBOL(d_TQ4_CENTROIDS_HIP), TQ4_CENTROIDS.data(),
                               16 * sizeof(float), 0, hipMemcpyHostToDevice, stream);
        (void)hipMemcpyToSymbolAsync(HIP_SYMBOL(d_TQ4_THRESHOLDS_HIP), TQ4_THRESHOLDS.data(),
                               15 * sizeof(float), 0, hipMemcpyHostToDevice, stream);
        (void)hipStreamSynchronize(stream);

        if (hipGetLastError() == hipSuccess)
            s_hip_codebooks_uploaded.store(true, std::memory_order_release);
    }

    void hip_tq_upload_rope_freqs(float rope_theta, int head_dim, hipStream_t stream)
    {
        if (s_hip_rope_freqs_head_dim.load(std::memory_order_acquire) == head_dim)
            return;

        const int half_dim = head_dim / 2;
        std::vector<float> freqs(half_dim);
        for (int i = 0; i < half_dim; ++i)
            freqs[i] = 1.0f / powf(rope_theta, static_cast<float>(2 * i) / static_cast<float>(head_dim));

        (void)hipMemcpyToSymbolAsync(HIP_SYMBOL(d_HIP_ROPE_FREQS), freqs.data(),
                               half_dim * sizeof(float), 0, hipMemcpyHostToDevice, stream);
        (void)hipStreamSynchronize(stream);

        if (hipGetLastError() == hipSuccess)
            s_hip_rope_freqs_head_dim.store(head_dim, std::memory_order_release);
    }

    // =========================================================================
    // Rotation Matrix Management
    // =========================================================================

    ROCmTurboQuantRotations hip_tq_create_rotations(
        int n_layers, int n_kv_heads, int head_dim,
        uint64_t rotation_seed, int device_id,
        hipStream_t stream)
    {
        (void)hipSetDevice(device_id);

        ROCmTurboQuantRotations result;
        result.n_layers = n_layers;
        result.n_kv_heads = n_kv_heads;
        result.head_dim = head_dim;

        const size_t mat_size = static_cast<size_t>(head_dim) * head_dim;
        const size_t total_mats = static_cast<size_t>(n_layers) * n_kv_heads;
        const size_t total_floats = total_mats * mat_size;

        (void)hipMalloc(&result.d_rotations, total_floats * sizeof(float));
        (void)hipMalloc(&result.d_rotations_t, total_floats * sizeof(float));

        TurboQuantContext base_ctx(head_dim, rotation_seed == 0 ? 31ULL : rotation_seed);

        std::vector<float> host_rot(total_floats);
        std::vector<float> host_rot_t(total_floats);

        for (int layer = 0; layer < n_layers; ++layer)
        {
            const auto &layer_ctx = base_ctx.for_layer(layer);

            for (int head = 0; head < n_kv_heads; ++head)
            {
                const auto &actual_ctx = layer_ctx.for_layer(head);
                const auto &rot = actual_ctx.rotation();

                const size_t offset = (static_cast<size_t>(layer) * n_kv_heads + head) * mat_size;
                std::memcpy(host_rot.data() + offset, rot.matrix.data(), mat_size * sizeof(float));

                for (int r = 0; r < head_dim; ++r)
                    for (int c = 0; c < head_dim; ++c)
                        host_rot_t[offset + r * head_dim + c] = rot.matrix[c * head_dim + r];
            }
        }

        (void)hipMemcpyAsync(result.d_rotations, host_rot.data(),
                       total_floats * sizeof(float), hipMemcpyHostToDevice, stream);
        (void)hipMemcpyAsync(result.d_rotations_t, host_rot_t.data(),
                       total_floats * sizeof(float), hipMemcpyHostToDevice, stream);

        return result;
    }

    void hip_tq_free_rotations(ROCmTurboQuantRotations &rotations)
    {
        if (rotations.d_rotations)
        {
            (void)hipFree(rotations.d_rotations);
            rotations.d_rotations = nullptr;
        }
        if (rotations.d_rotations_t)
        {
            (void)hipFree(rotations.d_rotations_t);
            rotations.d_rotations_t = nullptr;
        }
    }

    // =========================================================================
    // Device Helpers
    // =========================================================================

    __device__ inline float hip_warp_reduce_sum(float val)
    {
        // AMD wavefront is 64-wide
        for (int offset = 32; offset > 0; offset >>= 1)
            val += __shfl_down(val, offset);
        return val;
    }

    __device__ float hip_block_reduce_sum(float val, float *shared)
    {
        const int lane = threadIdx.x % 64; // AMD wavefront size
        const int warp_id = threadIdx.x / 64;
        const int num_warps = (blockDim.x + 63) / 64;

        val = hip_warp_reduce_sum(val);

        if (lane == 0)
            shared[warp_id] = val;
        __syncthreads();

        val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
        if (warp_id == 0)
        {
            for (int offset = 32; offset > 0; offset >>= 1)
                val += __shfl_down(val, offset);
        }

        return val;
    }

    // =========================================================================
    // TQ8 Quantize Kernel
    // =========================================================================

    template <int D>
    __global__ void hip_tq8_quantize_kernel(
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

        __shared__ float s_reduce[4]; // max 4 wavefronts for 256 threads
        __shared__ float s_norm;
        __shared__ float s_combined_scale;

        float norm_sq = hip_block_reduce_sum(x * x, s_reduce);

        if (tid == 0)
        {
            float norm = sqrtf(norm_sq);
            s_norm = norm;
            s_combined_scale = (norm > 1e-30f) ? (sqrtf(static_cast<float>(D)) / norm) : 0.0f;
        }
        __syncthreads();

        float scaled = x * s_combined_scale;

        __shared__ float s_scaled[128];
        s_scaled[tid] = scaled;
        __syncthreads();

        const float *rot_row = d_rotations + static_cast<size_t>(head) * D * D + tid * D;
        float rotated = 0.0f;
        for (int j = 0; j < D; ++j)
            rotated += rot_row[j] * s_scaled[j];

        // Binary search for nearest TQ8 centroid
        int idx = 0;
        if (rotated > d_TQ8_THRESHOLDS_HIP[idx + 127]) idx += 128;
        if (rotated > d_TQ8_THRESHOLDS_HIP[idx + 63]) idx += 64;
        if (rotated > d_TQ8_THRESHOLDS_HIP[idx + 31]) idx += 32;
        if (rotated > d_TQ8_THRESHOLDS_HIP[idx + 15]) idx += 16;
        if (rotated > d_TQ8_THRESHOLDS_HIP[idx + 7]) idx += 8;
        if (rotated > d_TQ8_THRESHOLDS_HIP[idx + 3]) idx += 4;
        if (rotated > d_TQ8_THRESHOLDS_HIP[idx + 1]) idx += 2;
        if (rotated > d_TQ8_THRESHOLDS_HIP[idx]) idx += 1;

        const size_t block_size = sizeof(TQ8Block<D>);
        uint8_t *block_ptr = d_output_bytes + (static_cast<size_t>(token) * n_kv_heads + head) * block_size;

        if (tid == 0)
        {
            float *norms = reinterpret_cast<float *>(block_ptr);
            norms[0] = s_norm;
            norms[1] = -1.0f;
        }
        block_ptr[2 * sizeof(float) + tid] = static_cast<uint8_t>(idx);
    }

    // =========================================================================
    // TQ4 Quantize Kernel
    // =========================================================================

    template <int D>
    __global__ void hip_tq4_quantize_kernel(
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

        __shared__ float s_reduce[4];
        __shared__ float s_norm;
        __shared__ float s_combined_scale;

        float norm_sq = hip_block_reduce_sum(x * x, s_reduce);

        if (tid == 0)
        {
            float norm = sqrtf(norm_sq);
            s_norm = norm;
            s_combined_scale = (norm > 1e-30f) ? (sqrtf(static_cast<float>(D)) / norm) : 0.0f;
        }
        __syncthreads();

        float scaled = x * s_combined_scale;

        __shared__ float s_scaled[128];
        s_scaled[tid] = scaled;
        __syncthreads();

        const float *rot_row = d_rotations + static_cast<size_t>(head) * D * D + tid * D;
        float rotated = 0.0f;
        for (int j = 0; j < D; ++j)
            rotated += rot_row[j] * s_scaled[j];

        // Binary search for TQ4 centroid
        int idx = 0;
        if (rotated > d_TQ4_THRESHOLDS_HIP[idx + 7]) idx += 8;
        if (rotated > d_TQ4_THRESHOLDS_HIP[idx + 3]) idx += 4;
        if (rotated > d_TQ4_THRESHOLDS_HIP[idx + 1]) idx += 2;
        if (rotated > d_TQ4_THRESHOLDS_HIP[idx]) idx += 1;

        __shared__ uint8_t s_indices[128];
        s_indices[tid] = static_cast<uint8_t>(idx);
        __syncthreads();

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

            for (int i = 0; i < D; i += 8)
            {
                uint8_t low3[8];
                uint8_t high1 = 0;
                for (int j = 0; j < 8; ++j)
                {
                    low3[j] = s_indices[i + j] & 0x07;
                    high1 |= ((s_indices[i + j] >> 3) & 0x01) << j;
                }

                int byte_idx = i * 3 / 8;
                mse_ptr[byte_idx + 0] = static_cast<uint8_t>(low3[0] | (low3[1] << 3) | (low3[2] << 6));
                mse_ptr[byte_idx + 1] = static_cast<uint8_t>((low3[2] >> 2) | (low3[3] << 1) | (low3[4] << 4) | (low3[5] << 7));
                mse_ptr[byte_idx + 2] = static_cast<uint8_t>((low3[5] >> 1) | (low3[6] << 2) | (low3[7] << 5));

                high_ptr[i / 8] = high1;
            }
        }
    }

    // =========================================================================
    // TQ8 Dequantize Kernel
    // =========================================================================

    template <int D>
    __global__ void hip_tq8_dequantize_fp16_kernel(
        const uint8_t *__restrict__ d_tq8_bytes,
        const float *__restrict__ d_rotations_t,
        float *__restrict__ d_output,
        int count, int n_kv_heads,
        float rope_theta, int position_start,
        int max_seq_len, int tail)
    {
        const int pos = blockIdx.x;
        const int head = blockIdx.y;
        const int tid = threadIdx.x;

        if (pos >= count || tid >= D)
            return;

        const int ring_pos = (tail + pos) % max_seq_len;
        const size_t block_size = sizeof(TQ8Block<D>);
        const uint8_t *block_ptr = d_tq8_bytes + (static_cast<size_t>(ring_pos) * n_kv_heads + head) * block_size;

        const float *norms = reinterpret_cast<const float *>(block_ptr);
        const float norm = norms[0];

        const uint8_t idx = block_ptr[2 * sizeof(float) + tid];
        const float inv_scale = 1.0f / sqrtf(static_cast<float>(D));
        float centroid_val = d_TQ8_CENTROIDS_HIP[idx] * inv_scale;

        __shared__ float s_centroid[128];
        s_centroid[tid] = centroid_val;
        __syncthreads();

        const float *rot_t_row = d_rotations_t + static_cast<size_t>(head) * D * D + tid * D;
        float val = 0.0f;
        for (int j = 0; j < D; ++j)
            val += rot_t_row[j] * s_centroid[j];

        val *= norm;

        // Optional RoPE (half-split convention: pairs (i, i+D/2))
        if (rope_theta > 0.0f)
        {
            constexpr int HALF = D / 2;
            const int actual_pos = position_start + pos;
            const int pair_idx = (tid < HALF) ? tid : (tid - HALF);
            const float angle = static_cast<float>(actual_pos) * d_HIP_ROPE_FREQS[pair_idx];
            float cos_val, sin_val;
            __sincosf(angle, &sin_val, &cos_val);

            __shared__ float s_vals[128];
            s_vals[tid] = val;
            __syncthreads();

            if (tid < HALF)
            {
                // First half: x' = x * cos - x_partner * sin
                float partner = s_vals[tid + HALF];
                val = val * cos_val - partner * sin_val;
            }
            else
            {
                // Second half: x' = x_partner * sin + x * cos
                float partner = s_vals[tid - HALF];
                val = partner * sin_val + val * cos_val;
            }
        }

        const int kv_dim = n_kv_heads * D;
        d_output[pos * kv_dim + head * D + tid] = val;
    }

    // =========================================================================
    // TQ4 Dequantize Kernel
    // =========================================================================

    template <int D>
    __global__ void hip_tq4_dequantize_fp16_kernel(
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
        case 0: low3 = b0 & 0x07; break;
        case 1: low3 = (b0 >> 3) & 0x07; break;
        case 2: low3 = ((b0 >> 6) | (b1 << 2)) & 0x07; break;
        case 3: low3 = (b1 >> 1) & 0x07; break;
        case 4: low3 = (b1 >> 4) & 0x07; break;
        case 5: low3 = ((b1 >> 7) | (b2 << 1)) & 0x07; break;
        case 6: low3 = (b2 >> 2) & 0x07; break;
        case 7: low3 = (b2 >> 5) & 0x07; break;
        default: low3 = 0; break;
        }

        const uint8_t high_byte = high_ptr[tid / 8];
        const uint8_t high1 = (high_byte >> (tid % 8)) & 0x01;
        const uint8_t full_idx = low3 | (high1 << 3);

        const float inv_scale = 1.0f / sqrtf(static_cast<float>(D));
        float centroid_val = d_TQ4_CENTROIDS_HIP[full_idx] * inv_scale;

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
    // Tiled Dequant Kernels (coalesced rotation matrix access)
    // =========================================================================

    template <int D, int TILE>
    __global__ void hip_tq8_dequantize_tiled_kernel(
        const uint8_t *__restrict__ d_tq8_bytes,
        const float *__restrict__ d_rotations,    // R (non-transposed, row-major) for coalesced access
        float *__restrict__ d_output,
        int count, int n_kv_heads,
        float rope_theta, int position_start,
        int max_seq_len, int tail)
    {
        const int head = blockIdx.x;
        const int tile_start = blockIdx.y * TILE;
        const int tid = threadIdx.x;

        if (tid >= D)
            return;

        // R is stored row-major: R[j][tid] at offset head*D*D + j*D + tid
        // Consecutive threads read consecutive addresses (coalesced!)
        // Math: output[tid] = sum_j R^T[tid][j] * centroid[j] = sum_j R[j][tid] * centroid[j]
        const float *rot_head = d_rotations + static_cast<size_t>(head) * D * D;
        const float inv_scale = 1.0f / sqrtf(static_cast<float>(D));
        const size_t block_size = sizeof(TQ8Block<D>);
        const int kv_dim = n_kv_heads * D;

        __shared__ float s_centroid[D];

        for (int t = 0; t < TILE; ++t)
        {
            const int pos = tile_start + t;
            if (pos >= count)
                return;

            const int ring_pos = (tail + pos) % max_seq_len;
            const uint8_t *block_ptr = d_tq8_bytes +
                (static_cast<size_t>(ring_pos) * n_kv_heads + head) * block_size;

            const float norm = reinterpret_cast<const float *>(block_ptr)[0];
            const uint8_t idx = block_ptr[2 * sizeof(float) + tid];
            s_centroid[tid] = d_TQ8_CENTROIDS_HIP[idx] * inv_scale;
            __syncthreads();

            // Coalesced dot product: R[j][tid] × centroid[j]
            float val = 0.0f;
            for (int j = 0; j < D; ++j)
                val += rot_head[j * D + tid] * s_centroid[j];
            val *= norm;

            if (rope_theta > 0.0f)
            {
                constexpr int HALF = D / 2;
                const int actual_pos = position_start + pos;
                const int pair_idx = (tid < HALF) ? tid : (tid - HALF);
                const float angle = static_cast<float>(actual_pos) * d_HIP_ROPE_FREQS[pair_idx];
                float cos_val, sin_val;
                __sincosf(angle, &sin_val, &cos_val);

                s_centroid[tid] = val;
                __syncthreads();

                if (tid < HALF)
                    val = val * cos_val - s_centroid[tid + HALF] * sin_val;
                else
                    val = s_centroid[tid - HALF] * sin_val + val * cos_val;
            }

            d_output[pos * kv_dim + head * D + tid] = val;
            __syncthreads();
        }
    }

    template <int D, int TILE>
    __global__ void hip_tq4_dequantize_tiled_kernel(
        const uint8_t *__restrict__ d_tq4_bytes,
        const float *__restrict__ d_rotations,    // R (non-transposed, row-major) for coalesced access
        float *__restrict__ d_output,
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

        __shared__ float s_centroid[D];

        for (int t = 0; t < TILE; ++t)
        {
            const int pos = tile_start + t;
            if (pos >= count)
                return;

            const int ring_pos = (tail + pos) % max_seq_len;
            const uint8_t *block_ptr = d_tq4_bytes +
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
            case 0: low3 = b0 & 0x07; break;
            case 1: low3 = (b0 >> 3) & 0x07; break;
            case 2: low3 = ((b0 >> 6) | (b1 << 2)) & 0x07; break;
            case 3: low3 = (b1 >> 1) & 0x07; break;
            case 4: low3 = (b1 >> 4) & 0x07; break;
            case 5: low3 = ((b1 >> 7) | (b2 << 1)) & 0x07; break;
            case 6: low3 = (b2 >> 2) & 0x07; break;
            case 7: low3 = (b2 >> 5) & 0x07; break;
            default: low3 = 0; break;
            }

            const uint8_t high_byte = high_ptr[tid / 8];
            const uint8_t high1 = (high_byte >> (tid % 8)) & 0x01;
            const uint8_t full_idx = low3 | (high1 << 3);

            s_centroid[tid] = d_TQ4_CENTROIDS_HIP[full_idx] * inv_scale;
            __syncthreads();

            // Coalesced dot product: R[j][tid] × centroid[j]
            float val = 0.0f;
            for (int j = 0; j < D; ++j)
                val += rot_head[j * D + tid] * s_centroid[j];
            val *= norm;

            d_output[pos * kv_dim + head * D + tid] = val;
            __syncthreads();
        }
    }

    // =========================================================================
    // FP16 Tiled Dequant Kernels (prefill path — half bandwidth)
    // =========================================================================

    template <int D, int TILE>
    __global__ void hip_tq8_dequantize_tiled_fp16_kernel(
        const uint8_t *__restrict__ d_tq8_bytes,
        const float *__restrict__ d_rotations,
        _Float16 *__restrict__ d_output,
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

        __shared__ float s_centroid[D];

        for (int t = 0; t < TILE; ++t)
        {
            const int pos = tile_start + t;
            if (pos >= count)
                return;

            const int ring_pos = (tail + pos) % max_seq_len;
            const uint8_t *block_ptr = d_tq8_bytes +
                (static_cast<size_t>(ring_pos) * n_kv_heads + head) * block_size;

            const float norm = reinterpret_cast<const float *>(block_ptr)[0];
            const uint8_t idx = block_ptr[2 * sizeof(float) + tid];
            s_centroid[tid] = d_TQ8_CENTROIDS_HIP[idx] * inv_scale;
            __syncthreads();

            float val = 0.0f;
            for (int j = 0; j < D; ++j)
                val += rot_head[j * D + tid] * s_centroid[j];
            val *= norm;

            if (rope_theta > 0.0f)
            {
                constexpr int HALF = D / 2;
                const int actual_pos = position_start + pos;
                const int pair_idx = (tid < HALF) ? tid : (tid - HALF);
                const float angle = static_cast<float>(actual_pos) * d_HIP_ROPE_FREQS[pair_idx];
                float cos_val, sin_val;
                __sincosf(angle, &sin_val, &cos_val);

                s_centroid[tid] = val;
                __syncthreads();

                if (tid < HALF)
                    val = val * cos_val - s_centroid[tid + HALF] * sin_val;
                else
                    val = s_centroid[tid - HALF] * sin_val + val * cos_val;
            }

            d_output[pos * kv_dim + head * D + tid] = static_cast<_Float16>(val);
            __syncthreads();
        }
    }

    template <int D, int TILE>
    __global__ void hip_tq4_dequantize_tiled_fp16_kernel(
        const uint8_t *__restrict__ d_tq4_bytes,
        const float *__restrict__ d_rotations,
        _Float16 *__restrict__ d_output,
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

        __shared__ float s_centroid[D];

        for (int t = 0; t < TILE; ++t)
        {
            const int pos = tile_start + t;
            if (pos >= count)
                return;

            const int ring_pos = (tail + pos) % max_seq_len;
            const uint8_t *block_ptr = d_tq4_bytes +
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
            case 0: low3 = b0 & 0x07; break;
            case 1: low3 = (b0 >> 3) & 0x07; break;
            case 2: low3 = ((b0 >> 6) | (b1 << 2)) & 0x07; break;
            case 3: low3 = (b1 >> 1) & 0x07; break;
            case 4: low3 = (b1 >> 4) & 0x07; break;
            case 5: low3 = ((b1 >> 7) | (b2 << 1)) & 0x07; break;
            case 6: low3 = (b2 >> 2) & 0x07; break;
            case 7: low3 = (b2 >> 5) & 0x07; break;
            default: low3 = 0; break;
            }

            const uint8_t high_byte = high_ptr[tid / 8];
            const uint8_t high1 = (high_byte >> (tid % 8)) & 0x01;
            const uint8_t full_idx = low3 | (high1 << 3);

            s_centroid[tid] = d_TQ4_CENTROIDS_HIP[full_idx] * inv_scale;
            __syncthreads();

            float val = 0.0f;
            for (int j = 0; j < D; ++j)
                val += rot_head[j * D + tid] * s_centroid[j];
            val *= norm;

            d_output[pos * kv_dim + head * D + tid] = static_cast<_Float16>(val);
            __syncthreads();
        }
    }

    // =========================================================================
    // FP16 Fused Incremental Kernel (decode path — K+V in one launch)
    // =========================================================================

    template <int D>
    __global__ void hip_tq_incremental_fused_kernel(
        const uint8_t *__restrict__ k_cache,
        const uint8_t *__restrict__ v_cache,
        _Float16 *__restrict__ k_output,
        _Float16 *__restrict__ v_output,
        const float *__restrict__ k_rotation,
        const float *__restrict__ v_rotation,
        int ring_pos_scalar, int out_offset_elems_scalar,
        int n_kv_heads,
        float rope_theta, int rope_position_scalar,
        const HIPTQDequantDynamicParams *d_params)
    {
        const int ring_pos = d_params ? d_params->ring_pos : ring_pos_scalar;
        const int out_offset_elems = d_params ? d_params->out_offset_elems : out_offset_elems_scalar;
        const int rope_position = d_params ? d_params->rope_position : rope_position_scalar;

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

            s_cent[tid] = d_TQ8_CENTROIDS_HIP[idx] * inv_scale;
            __syncthreads();

            const float *rot = k_rotation + static_cast<size_t>(head) * D * D;
            float val = 0.0f;
            for (int j = 0; j < D; ++j)
                val += rot[j * D + tid] * s_cent[j];
            val *= norm;

            if (rope_theta > 0.0f)
            {
                s_cent[tid] = val;
                __syncthreads();

                constexpr int HALF = D / 2;
                const int pair_idx = (tid < HALF) ? tid : (tid - HALF);
                const float angle = static_cast<float>(rope_position) * d_HIP_ROPE_FREQS[pair_idx];
                float cos_val, sin_val;
                __sincosf(angle, &sin_val, &cos_val);
                const float partner = s_cent[(tid < HALF) ? (tid + HALF) : (tid - HALF)];
                if (tid < HALF)
                    val = val * cos_val - partner * sin_val;
                else
                    val = partner * sin_val + val * cos_val;
            }

            k_output[out_offset_elems + head * D + tid] = static_cast<_Float16>(val);
        }

        __syncthreads();

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
            case 0: low3 = b0 & 0x07; break;
            case 1: low3 = (b0 >> 3) & 0x07; break;
            case 2: low3 = ((b0 >> 6) | (b1 << 2)) & 0x07; break;
            case 3: low3 = (b1 >> 1) & 0x07; break;
            case 4: low3 = (b1 >> 4) & 0x07; break;
            case 5: low3 = ((b1 >> 7) | (b2 << 1)) & 0x07; break;
            case 6: low3 = (b2 >> 2) & 0x07; break;
            case 7: low3 = (b2 >> 5) & 0x07; break;
            default: low3 = 0; break;
            }
            const uint8_t high1 = (high_ptr[tid / 8] >> (tid % 8)) & 0x01;
            const uint8_t full_idx = low3 | (high1 << 3);

            s_cent[tid] = d_TQ4_CENTROIDS_HIP[full_idx] * inv_scale;
            __syncthreads();

            const float *rot = v_rotation + static_cast<size_t>(head) * D * D;
            float val = 0.0f;
            for (int j = 0; j < D; ++j)
                val += rot[j * D + tid] * s_cent[j];

            v_output[out_offset_elems + head * D + tid] = static_cast<_Float16>(val * norm);
        }
    }

    // =========================================================================
    // RoPE Kernels
    // =========================================================================

    __global__ void hip_rope_apply_fp16_kernel(
        _Float16 *__restrict__ d_K,
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
        float x = static_cast<float>(d_K[idx0]);
        float y = static_cast<float>(d_K[idx1]);

        d_K[idx0] = static_cast<_Float16>(x * cos_val - y * sin_val);
        d_K[idx1] = static_cast<_Float16>(x * sin_val + y * cos_val);
    }

    __global__ void hip_rope_apply_fp32_kernel(
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

    extern "C" bool hip_tq8_quantize(
        const float *d_input, const float *d_rotations, void *d_output,
        int num_tokens, int n_kv_heads, int head_dim, hipStream_t stream)
    {
        if (!d_input || !d_rotations || !d_output || num_tokens <= 0)
            return false;

        hip_tq_upload_codebooks(stream);

        const dim3 grid(num_tokens, n_kv_heads);

        if (head_dim == 64)
        {
            hipLaunchKernelGGL(hip_tq8_quantize_kernel<64>, grid, dim3(64), 0, stream,
                               d_input, d_rotations, static_cast<uint8_t *>(d_output), n_kv_heads);
        }
        else if (head_dim == 128)
        {
            hipLaunchKernelGGL(hip_tq8_quantize_kernel<128>, grid, dim3(128), 0, stream,
                               d_input, d_rotations, static_cast<uint8_t *>(d_output), n_kv_heads);
        }
        else
            return false;

        return hipGetLastError() == hipSuccess;
    }

    extern "C" bool hip_tq4_quantize(
        const float *d_input, const float *d_rotations, void *d_output,
        int num_tokens, int n_kv_heads, int head_dim, hipStream_t stream)
    {
        if (!d_input || !d_rotations || !d_output || num_tokens <= 0)
            return false;

        hip_tq_upload_codebooks(stream);

        const dim3 grid(num_tokens, n_kv_heads);

        if (head_dim == 64)
        {
            hipLaunchKernelGGL(hip_tq4_quantize_kernel<64>, grid, dim3(64), 0, stream,
                               d_input, d_rotations, static_cast<uint8_t *>(d_output), n_kv_heads);
        }
        else if (head_dim == 128)
        {
            hipLaunchKernelGGL(hip_tq4_quantize_kernel<128>, grid, dim3(128), 0, stream,
                               d_input, d_rotations, static_cast<uint8_t *>(d_output), n_kv_heads);
        }
        else
            return false;

        return hipGetLastError() == hipSuccess;
    }

    extern "C" bool hip_tq8_dequantize_fp32(
        const void *d_tq8_blocks, const float *d_rotations_t,
        float *d_output,
        int count, int n_kv_heads, int head_dim,
        float rope_theta, int position_start,
        int max_seq_len, int tail, hipStream_t stream)
    {
        if (!d_tq8_blocks || !d_rotations_t || !d_output || count <= 0)
            return false;

        hip_tq_upload_codebooks(stream);

        const dim3 grid(count, n_kv_heads);

        if (head_dim == 64)
        {
            hipLaunchKernelGGL(hip_tq8_dequantize_fp16_kernel<64>, grid, dim3(64), 0, stream,
                               static_cast<const uint8_t *>(d_tq8_blocks), d_rotations_t, d_output,
                               count, n_kv_heads, rope_theta, position_start, max_seq_len, tail);
        }
        else if (head_dim == 128)
        {
            hipLaunchKernelGGL(hip_tq8_dequantize_fp16_kernel<128>, grid, dim3(128), 0, stream,
                               static_cast<const uint8_t *>(d_tq8_blocks), d_rotations_t, d_output,
                               count, n_kv_heads, rope_theta, position_start, max_seq_len, tail);
        }
        else
            return false;

        return hipGetLastError() == hipSuccess;
    }

    extern "C" bool hip_tq4_dequantize_fp32(
        const void *d_tq4_blocks, const float *d_rotations_t,
        float *d_output,
        int count, int n_kv_heads, int head_dim, hipStream_t stream)
    {
        if (!d_tq4_blocks || !d_rotations_t || !d_output || count <= 0)
            return false;

        hip_tq_upload_codebooks(stream);

        const dim3 grid(count, n_kv_heads);

        if (head_dim == 64)
        {
            hipLaunchKernelGGL(hip_tq4_dequantize_fp16_kernel<64>, grid, dim3(64), 0, stream,
                               static_cast<const uint8_t *>(d_tq4_blocks), d_rotations_t, d_output,
                               count, n_kv_heads, 0, 0);
        }
        else if (head_dim == 128)
        {
            hipLaunchKernelGGL(hip_tq4_dequantize_fp16_kernel<128>, grid, dim3(128), 0, stream,
                               static_cast<const uint8_t *>(d_tq4_blocks), d_rotations_t, d_output,
                               count, n_kv_heads, 0, 0);
        }
        else
            return false;

        return hipGetLastError() == hipSuccess;
    }

    extern "C" bool hip_tq_ring_append(
        void *d_K_cache, void *d_V_cache,
        const float *d_K_new, const float *d_V_new,
        const float *d_K_rotations, const float *d_V_rotations,
        int head, int max_seq_len,
        int n_kv_heads, int head_dim, int num_tokens,
        hipStream_t stream)
    {
        if (num_tokens <= 0)
            return true;

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
            return false;

        const size_t k_temp_bytes = static_cast<size_t>(num_tokens) * n_kv_heads * k_block_size;
        const size_t v_temp_bytes = static_cast<size_t>(num_tokens) * n_kv_heads * v_block_size;

        void *d_k_temp = nullptr;
        void *d_v_temp = nullptr;
        (void)hipMalloc(&d_k_temp, k_temp_bytes);
        (void)hipMalloc(&d_v_temp, v_temp_bytes);

        bool ok = hip_tq8_quantize(d_K_new, d_K_rotations, d_k_temp,
                                    num_tokens, n_kv_heads, head_dim, stream);
        if (ok)
            ok = hip_tq4_quantize(d_V_new, d_V_rotations, d_v_temp,
                                   num_tokens, n_kv_heads, head_dim, stream);

        if (ok)
        {
            for (int t = 0; t < num_tokens; ++t)
            {
                int dst_pos = (head + t) % max_seq_len;
                size_t k_dst_offset = static_cast<size_t>(dst_pos) * n_kv_heads * k_block_size;
                size_t k_src_offset = static_cast<size_t>(t) * n_kv_heads * k_block_size;
                (void)hipMemcpyAsync(static_cast<uint8_t *>(d_K_cache) + k_dst_offset,
                               static_cast<uint8_t *>(d_k_temp) + k_src_offset,
                               n_kv_heads * k_block_size, hipMemcpyDeviceToDevice, stream);

                size_t v_dst_offset = static_cast<size_t>(dst_pos) * n_kv_heads * v_block_size;
                size_t v_src_offset = static_cast<size_t>(t) * n_kv_heads * v_block_size;
                (void)hipMemcpyAsync(static_cast<uint8_t *>(d_V_cache) + v_dst_offset,
                               static_cast<uint8_t *>(d_v_temp) + v_src_offset,
                               n_kv_heads * v_block_size, hipMemcpyDeviceToDevice, stream);
            }
        }

        (void)hipFree(d_k_temp);
        (void)hipFree(d_v_temp);

        return ok;
    }

    extern "C" bool hip_tq_ring_linearize_dequant(
        float *d_K_out, float *d_V_out,
        const void *d_K_cache, const void *d_V_cache,
        const float *d_K_rotations_t, const float *d_V_rotations_t,
        const float *d_K_rotations, const float *d_V_rotations,
        int tail, int count, int max_seq_len,
        int n_kv_heads, int head_dim,
        float rope_theta, int position_start, hipStream_t stream)
    {
        if (count <= 0)
            return true;

        hip_tq_upload_codebooks(stream);
        if (rope_theta > 0.0f)
            hip_tq_upload_rope_freqs(rope_theta, head_dim, stream);

        // Tiled kernels use R (non-transposed, row-major) for coalesced access:
        // R[j][tid] at offset j*D + tid → consecutive threads read consecutive addresses.
        constexpr int TILE = 16;

        if (head_dim == 64)
        {
            const dim3 grid(n_kv_heads, (count + TILE - 1) / TILE);
            hipLaunchKernelGGL((hip_tq8_dequantize_tiled_kernel<64, TILE>), grid, dim3(64), 0, stream,
                               static_cast<const uint8_t *>(d_K_cache), d_K_rotations, d_K_out,
                               count, n_kv_heads, rope_theta, position_start, max_seq_len, tail);
            hipLaunchKernelGGL((hip_tq4_dequantize_tiled_kernel<64, TILE>), grid, dim3(64), 0, stream,
                               static_cast<const uint8_t *>(d_V_cache), d_V_rotations, d_V_out,
                               count, n_kv_heads, max_seq_len, tail);
        }
        else if (head_dim == 128)
        {
            const dim3 grid(n_kv_heads, (count + TILE - 1) / TILE);
            hipLaunchKernelGGL((hip_tq8_dequantize_tiled_kernel<128, TILE>), grid, dim3(128), 0, stream,
                               static_cast<const uint8_t *>(d_K_cache), d_K_rotations, d_K_out,
                               count, n_kv_heads, rope_theta, position_start, max_seq_len, tail);
            hipLaunchKernelGGL((hip_tq4_dequantize_tiled_kernel<128, TILE>), grid, dim3(128), 0, stream,
                               static_cast<const uint8_t *>(d_V_cache), d_V_rotations, d_V_out,
                               count, n_kv_heads, max_seq_len, tail);
        }
        else
            return false;

        return hipGetLastError() == hipSuccess;
    }

    extern "C" bool hip_rope_apply_fp16(
        _Float16 *d_K, int count,
        int n_kv_heads, int head_dim,
        float rope_theta, int position_start,
        hipStream_t stream, int rope_dim)
    {
        if (!d_K || count <= 0 || rope_theta <= 0.0f)
            return false;

        const int effective_rope_dim = (rope_dim > 0) ? rope_dim : head_dim;
        const int total_pairs = count * n_kv_heads * (effective_rope_dim / 2);
        hipLaunchKernelGGL(hip_rope_apply_fp16_kernel, dim3((total_pairs + 255) / 256), dim3(256), 0, stream,
                           d_K, count, n_kv_heads, head_dim, rope_theta, position_start, rope_dim);
        return hipGetLastError() == hipSuccess;
    }

    extern "C" bool hip_rope_apply_fp32(
        float *d_K, int count,
        int n_kv_heads, int head_dim,
        float rope_theta, int position_start,
        hipStream_t stream, int rope_dim)
    {
        if (!d_K || count <= 0 || rope_theta <= 0.0f)
            return false;

        const int effective_rope_dim = (rope_dim > 0) ? rope_dim : head_dim;
        const int total_pairs = count * n_kv_heads * (effective_rope_dim / 2);
        hipLaunchKernelGGL(hip_rope_apply_fp32_kernel, dim3((total_pairs + 255) / 256), dim3(256), 0, stream,
                           d_K, count, n_kv_heads, head_dim, rope_theta, position_start, rope_dim);
        return hipGetLastError() == hipSuccess;
    }

    // =========================================================================
    // FP16 Linearize + Dequant (full sequence, for prefill)
    // =========================================================================

    extern "C" bool hip_tq_ring_linearize_dequant_fp16(
        _Float16 *d_K_out, _Float16 *d_V_out,
        const void *d_K_cache, const void *d_V_cache,
        const float *d_K_rotations_t, const float *d_V_rotations_t,
        const float *d_K_rotations, const float *d_V_rotations,
        int tail, int count, int max_seq_len,
        int n_kv_heads, int head_dim,
        float rope_theta, int position_start, hipStream_t stream)
    {
        if (count <= 0)
            return true;

        hip_tq_upload_codebooks(stream);
        if (rope_theta > 0.0f)
            hip_tq_upload_rope_freqs(rope_theta, head_dim, stream);

        constexpr int TILE = 16;

        if (head_dim == 64)
        {
            const dim3 grid(n_kv_heads, (count + TILE - 1) / TILE);
            hipLaunchKernelGGL((hip_tq8_dequantize_tiled_fp16_kernel<64, TILE>), grid, dim3(64), 0, stream,
                               static_cast<const uint8_t *>(d_K_cache), d_K_rotations, d_K_out,
                               count, n_kv_heads, rope_theta, position_start, max_seq_len, tail);
            hipLaunchKernelGGL((hip_tq4_dequantize_tiled_fp16_kernel<64, TILE>), grid, dim3(64), 0, stream,
                               static_cast<const uint8_t *>(d_V_cache), d_V_rotations, d_V_out,
                               count, n_kv_heads, max_seq_len, tail);
        }
        else if (head_dim == 128)
        {
            const dim3 grid(n_kv_heads, (count + TILE - 1) / TILE);
            hipLaunchKernelGGL((hip_tq8_dequantize_tiled_fp16_kernel<128, TILE>), grid, dim3(128), 0, stream,
                               static_cast<const uint8_t *>(d_K_cache), d_K_rotations, d_K_out,
                               count, n_kv_heads, rope_theta, position_start, max_seq_len, tail);
            hipLaunchKernelGGL((hip_tq4_dequantize_tiled_fp16_kernel<128, TILE>), grid, dim3(128), 0, stream,
                               static_cast<const uint8_t *>(d_V_cache), d_V_rotations, d_V_out,
                               count, n_kv_heads, max_seq_len, tail);
        }
        else
            return false;

        return hipGetLastError() == hipSuccess;
    }

    // =========================================================================
    // FP16 Fused Incremental Dequant (single position, for decode)
    // =========================================================================

    extern "C" bool hip_tq_incremental_single_fp16(
        _Float16 *d_K_out, _Float16 *d_V_out,
        const void *d_K_cache, const void *d_V_cache,
        const float *d_K_rotation, const float *d_V_rotation,
        int ring_pos, int out_offset_elems,
        int n_kv_heads, int head_dim,
        float rope_theta, int rope_position,
        hipStream_t stream)
    {
        if (!d_K_out || !d_V_out || !d_K_cache || !d_V_cache)
            return false;

        hip_tq_upload_codebooks(stream);
        if (rope_theta > 0.0f)
            hip_tq_upload_rope_freqs(rope_theta, head_dim, stream);

        const dim3 grid(n_kv_heads);

        if (head_dim == 64)
        {
            hipLaunchKernelGGL((hip_tq_incremental_fused_kernel<64>), grid, dim3(64), 0, stream,
                               static_cast<const uint8_t *>(d_K_cache),
                               static_cast<const uint8_t *>(d_V_cache),
                               d_K_out, d_V_out,
                               d_K_rotation, d_V_rotation,
                               ring_pos, out_offset_elems,
                               n_kv_heads, rope_theta, rope_position,
                               static_cast<const HIPTQDequantDynamicParams *>(nullptr));
        }
        else if (head_dim == 128)
        {
            hipLaunchKernelGGL((hip_tq_incremental_fused_kernel<128>), grid, dim3(128), 0, stream,
                               static_cast<const uint8_t *>(d_K_cache),
                               static_cast<const uint8_t *>(d_V_cache),
                               d_K_out, d_V_out,
                               d_K_rotation, d_V_rotation,
                               ring_pos, out_offset_elems,
                               n_kv_heads, rope_theta, rope_position,
                               static_cast<const HIPTQDequantDynamicParams *>(nullptr));
        }
        else
            return false;

        return hipGetLastError() == hipSuccess;
    }

    extern "C" bool hip_tq_incremental_single_fp16_dynamic(
        _Float16 *d_K_base, _Float16 *d_V_base,
        const void *d_K_cache, const void *d_V_cache,
        const float *d_K_rotation, const float *d_V_rotation,
        const HIPTQDequantDynamicParams *d_params,
        int n_kv_heads, int head_dim,
        float rope_theta,
        hipStream_t stream)
    {
        if (!d_K_base || !d_V_base || !d_K_cache || !d_V_cache || !d_params)
            return false;

        hip_tq_upload_codebooks(stream);
        if (rope_theta > 0.0f)
            hip_tq_upload_rope_freqs(rope_theta, head_dim, stream);

        const dim3 grid(n_kv_heads);

        if (head_dim == 64)
        {
            hipLaunchKernelGGL((hip_tq_incremental_fused_kernel<64>), grid, dim3(64), 0, stream,
                               static_cast<const uint8_t *>(d_K_cache),
                               static_cast<const uint8_t *>(d_V_cache),
                               d_K_base, d_V_base,
                               d_K_rotation, d_V_rotation,
                               0, 0, n_kv_heads, rope_theta, 0,
                               d_params);
        }
        else if (head_dim == 128)
        {
            hipLaunchKernelGGL((hip_tq_incremental_fused_kernel<128>), grid, dim3(128), 0, stream,
                               static_cast<const uint8_t *>(d_K_cache),
                               static_cast<const uint8_t *>(d_V_cache),
                               d_K_base, d_V_base,
                               d_K_rotation, d_V_rotation,
                               0, 0, n_kv_heads, rope_theta, 0,
                               d_params);
        }
        else
            return false;

        return hipGetLastError() == hipSuccess;
    }

} // namespace llaminar2
