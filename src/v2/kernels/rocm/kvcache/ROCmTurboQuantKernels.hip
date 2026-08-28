/**
 * @file ROCmTurboQuantKernels.hip
 * @brief HIP kernel implementations for TurboQuant KV cache operations
 * @author David Sanftenberg
 *
 * HIP mirror of CUDATurboQuantKernels.cu for AMD GPUs.
 * Uses hipLaunchKernelGGL() launch syntax and HIP FP16 types.
 */

#include "ROCmTurboQuantKernels.h"
#include "../../../backends/BackendManager.h"
#include "../../../kernels/cpu/turboquant/TurboQuantCodebook.h"
#include "../../../kernels/cpu/turboquant/TurboQuantContext.h"
#include "../../../utils/Logger.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <stdexcept>

namespace llaminar2
{

    // =========================================================================
    // Constant Memory Codebooks
    // =========================================================================

    __constant__ float d_TQ8_CENTROIDS_HIP[256];
    __constant__ float d_TQ8_THRESHOLDS_HIP[255];
    __constant__ float d_TQ4_CENTROIDS_HIP[16];
    __constant__ float d_TQ4_THRESHOLDS_HIP[15];

    // Precomputed RoPE frequencies: freq[i] = 1/theta^(2i/D) for i in [0, D/2)
    __constant__ float d_HIP_ROPE_FREQS[64]; // max head_dim/2 = 128/2 = 64
    /** Host mirror identity for one HIP module's uploaded RoPE constants. */
    struct HIPRopeConstantGeneration
    {
        std::uint64_t runtime_generation = 0u;
        float rope_theta = 0.0f;
        int head_dim = 0;
    };

    static std::mutex s_hip_rope_freqs_mutex;
    static std::unordered_map<int, HIPRopeConstantGeneration>
        s_hip_rope_freqs_by_device;

    bool hip_tq_upload_codebooks(hipStream_t stream)
    {
        if (!stream)
            return false;

        /*
         * HIP constant memory belongs to the active device. Each cache queues
         * these immutable tables on its own initialization stream so a global
         * host flag cannot suppress publication on another ROCm participant.
         * The cache's final construction fence establishes readiness.
         */
        return hipMemcpyToSymbolAsync(
                   HIP_SYMBOL(d_TQ8_CENTROIDS_HIP), TQ8_CENTROIDS.data(),
                   256 * sizeof(float), 0, hipMemcpyHostToDevice, stream) == hipSuccess &&
               hipMemcpyToSymbolAsync(
                   HIP_SYMBOL(d_TQ8_THRESHOLDS_HIP), TQ8_THRESHOLDS.data(),
                   255 * sizeof(float), 0, hipMemcpyHostToDevice, stream) == hipSuccess &&
               hipMemcpyToSymbolAsync(
                   HIP_SYMBOL(d_TQ4_CENTROIDS_HIP), TQ4_CENTROIDS.data(),
                   16 * sizeof(float), 0, hipMemcpyHostToDevice, stream) == hipSuccess &&
               hipMemcpyToSymbolAsync(
                   HIP_SYMBOL(d_TQ4_THRESHOLDS_HIP), TQ4_THRESHOLDS.data(),
                   15 * sizeof(float), 0, hipMemcpyHostToDevice, stream) == hipSuccess;
    }

    void hip_tq_upload_rope_freqs(float rope_theta, int head_dim, hipStream_t stream)
    {
        int device_id = -1;
        if (hipGetDevice(&device_id) != hipSuccess)
            return;
        IBackend *const backend = getROCmBackend();
        const std::uint64_t generation = backend
                                             ? backend->deviceRuntimeGeneration(
                                                   device_id)
                                             : 0u;
        if (generation == 0u)
            return;

        std::lock_guard<std::mutex> lock(s_hip_rope_freqs_mutex);
        const auto existing = s_hip_rope_freqs_by_device.find(device_id);
        if (existing != s_hip_rope_freqs_by_device.end() &&
            existing->second.runtime_generation == generation &&
            existing->second.rope_theta == rope_theta &&
            existing->second.head_dim == head_dim)
        {
            return;
        }

        const int half_dim = head_dim / 2;
        std::vector<float> freqs(half_dim);
        for (int i = 0; i < half_dim; ++i)
            freqs[i] = 1.0f / powf(rope_theta, static_cast<float>(2 * i) / static_cast<float>(head_dim));

        (void)hipMemcpyToSymbolAsync(HIP_SYMBOL(d_HIP_ROPE_FREQS), freqs.data(),
                               half_dim * sizeof(float), 0, hipMemcpyHostToDevice, stream);
        (void)hipStreamSynchronize(stream);

        if (hipGetLastError() == hipSuccess)
        {
            s_hip_rope_freqs_by_device[device_id] = {
                .runtime_generation = generation,
                .rope_theta = rope_theta,
                .head_dim = head_dim,
            };
        }
    }

    // =========================================================================
    // Rotation Matrix Management
    // =========================================================================

    ROCmTurboQuantRotations hip_tq_create_rotations(
        int n_layers, int n_kv_heads, int head_dim,
        uint64_t rotation_seed, int device_id,
        hipStream_t stream,
        int kv_head_start)
    {
        (void)hipSetDevice(device_id);

        ROCmTurboQuantRotations result;
        result.n_layers = n_layers;
        result.n_kv_heads = n_kv_heads;
        result.head_dim = head_dim;
        result.device_id = device_id;

        const size_t mat_size = static_cast<size_t>(head_dim) * head_dim;
        const size_t total_mats = static_cast<size_t>(n_layers) * n_kv_heads;
        const size_t total_floats = total_mats * mat_size;

        auto *backend = getROCmBackend();
        if (!backend)
            throw std::runtime_error("[TurboQuant ROCm] ROCm backend unavailable");
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
                "[TurboQuant ROCm] Failed to allocate model-lifetime rotation storage");
        }

        TurboQuantContext base_ctx(head_dim, rotation_seed == 0 ? 31ULL : rotation_seed);

        std::vector<float> host_rot(total_floats);
        std::vector<float> host_rot_t(total_floats);

        for (int layer = 0; layer < n_layers; ++layer)
        {
            const auto &layer_ctx = base_ctx.for_layer(layer);

            for (int head = 0; head < n_kv_heads; ++head)
            {
                const auto &actual_ctx = layer_ctx.for_layer(kv_head_start + head);
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
        auto *backend = getROCmBackend();
        if (!backend && (rotations.d_rotations || rotations.d_rotations_t))
            throw std::runtime_error(
                "[TurboQuant ROCm] ROCm backend unavailable during rotation teardown");
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

    __device__ inline float hip_warp_reduce_sum(float val)
    {
        // AMD wavefront is 64-wide
        for (int offset = 32; offset > 0; offset >>= 1)
            val += __shfl_down(val, offset);
        return val;
    }

    /**
     * @brief Workgroup fixed-order sum with safely reusable LDS workspace.
     *
     * Every work item must participate. Only work item zero receives the full
     * sum. The final barrier closes the read epoch before a following radial-fit
     * reduction reuses `shared`, preventing a faster wave from overwriting LDS
     * while wave zero still reads the previous result.
     *
     * @param val Calling work item's contribution.
     * @param shared LDS workspace covering the workgroup's wave count.
     * @return Complete sum in work item zero.
     */
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

        __syncthreads();

        return val;
    }

    /** @brief Fixed-order gfx wave maximum for AQ8 key encoding. */
    __device__ __forceinline__ float hip_wave_reduce_max(float value)
    {
        for (int offset = 32; offset > 0; offset >>= 1)
            value = fmaxf(value, __shfl_down(value, offset, 64));
        return value;
    }

    /** @brief Fixed-order workgroup maximum for supported attention widths. */
    __device__ __forceinline__ float hip_block_reduce_max(
        float value,
        float *shared)
    {
        const int lane = threadIdx.x % 64;
        const int wave = threadIdx.x / 64;
        const int wave_count = (blockDim.x + 63) / 64;
        value = hip_wave_reduce_max(value);
        if (lane == 0)
            shared[wave] = value;
        __syncthreads();
        value = threadIdx.x < wave_count ? shared[threadIdx.x] : 0.0f;
        if (wave == 0)
            value = hip_wave_reduce_max(value);
        return value;
    }

    /**
     * @brief Return a fixed-order maximum within one 32-lane half-wave.
     *
     * ROCm executes the supported kernels with 64-lane wavefronts, while a
     * physical Q8_1 block contains exactly 32 coordinates. Supplying an
     * explicit shuffle width keeps the two adjacent Q8_1 blocks independent;
     * a wave-wide reduction would silently give both blocks the same scale.
     */
    __device__ __forceinline__ float hip_half_wave_reduce_max(float value)
    {
#pragma unroll
        for (int offset = Q8_1Block::BLOCK_SIZE / 2; offset > 0; offset >>= 1)
            value = fmaxf(
                value,
                __shfl_xor(value, offset, Q8_1Block::BLOCK_SIZE));
        return value;
    }

    /** @brief Return a fixed-order integer sum within one Q8_1 half-wave. */
    __device__ __forceinline__ int hip_half_wave_reduce_sum(int value)
    {
#pragma unroll
        for (int offset = Q8_1Block::BLOCK_SIZE / 2; offset > 0; offset >>= 1)
            value += __shfl_xor(value, offset, Q8_1Block::BLOCK_SIZE);
        return value;
    }

    /** @brief Device mirror of the scalar AQ8 rational interval search. */
    __device__ __forceinline__ int8_t hip_attention_key_q8_encode_magnitude(
        float magnitude,
        float max_abs)
    {
        constexpr int denominator = 254;
        constexpr int denominator_squared = denominator * denominator;
        const float scaled_magnitude =
            magnitude * static_cast<float>(denominator_squared);
        int lower = 0;
        int upper = AttentionKeyQ8Block<64>::MAX_CODE;
#pragma unroll
        for (int step = 0; step < 7; ++step)
        {
            const int midpoint = (lower + upper) / 2;
            const int odd_boundary = 2 * midpoint + 1;
            const int odd_boundary_squared = odd_boundary * odd_boundary;
            const float scaled_boundary =
                max_abs * static_cast<float>(odd_boundary_squared);
            if (scaled_magnitude < scaled_boundary)
                upper = midpoint;
            else
                lower = midpoint + 1;
        }
        return static_cast<int8_t>(lower);
    }

    // =========================================================================
    // TQ8 Quantize Kernel
    // =========================================================================

    /**
     * @brief Canonical ROCm RoPE arithmetic shared by TQ and ordinary caches.
     *
     * Explicit fused operations establish one byte-stable rounding contract for
     * scalar decode and grouped verifier launches.
     */
    __device__ __forceinline__ void canonical_hip_rope_pair(
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

    struct HipFP32RoPEPair
    {
        float first;
        float second;
    };

    /**
     * @brief Materialize the canonical FP32 pair behind one shared call boundary.
     *
     * The HIP optimizer can otherwise see through the grouped kernel's immediate
     * FP16 conversion while the serial kernel publishes FP32 to global memory.
     * A shared noinline function gives both paths identical FP32 contraction and
     * return-value rounding without allocating a request-sized FP32 staging tensor.
     */
    __device__ __attribute__((noinline)) HipFP32RoPEPair
    canonical_hip_fp32_rope_pair(
        float x,
        float y,
        float rope_theta,
        int pair,
        int rope_dim,
        int position)
    {
        HipFP32RoPEPair result{};
        canonical_hip_rope_pair(
            x, y, rope_theta, pair, rope_dim, position,
            &result.first, &result.second);
        return result;
    }

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

        __shared__ float s_scaled[D];
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

        const float selected_centroid = d_TQ8_CENTROIDS_HIP[idx];
        const float source_dot_centroids = hip_block_reduce_sum(
            rotated * selected_centroid, s_reduce);
        const float centroid_norm_sq = hip_block_reduce_sum(
            selected_centroid * selected_centroid, s_reduce);

        const size_t block_size = sizeof(TQ8Block<D>);
        uint8_t *block_ptr = d_output_bytes + (static_cast<size_t>(token) * n_kv_heads + head) * block_size;

        if (tid == 0)
        {
            float *norms = reinterpret_cast<float *>(block_ptr);
            norms[0] = s_norm;
            norms[1] = centroid_norm_sq > 0.0f
                           ? s_norm * source_dot_centroids / centroid_norm_sq
                           : 0.0f;
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

        __shared__ float s_scaled[D];
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

        const float selected_centroid = d_TQ4_CENTROIDS_HIP[idx];
        const float source_dot_centroids = hip_block_reduce_sum(
            rotated * selected_centroid, s_reduce);
        const float centroid_norm_sq = hip_block_reduce_sum(
            selected_centroid * selected_centroid, s_reduce);

        __shared__ uint8_t s_indices[D];
        s_indices[tid] = static_cast<uint8_t>(idx);
        __syncthreads();

        constexpr size_t MSE_BYTES = D * 3 / 8;
        const size_t block_size = sizeof(TQ4Block<D>);
        uint8_t *block_ptr = d_output_bytes + (static_cast<size_t>(token) * n_kv_heads + head) * block_size;

        if (tid == 0)
        {
            float *norms = reinterpret_cast<float *>(block_ptr);
            norms[0] = s_norm;
            norms[1] = centroid_norm_sq > 0.0f
                           ? s_norm * source_dot_centroids / centroid_norm_sq
                           : 0.0f;

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
    // Fused grouped verifier quantize-to-ring kernel
    // =========================================================================

    /**
     * @brief Quantize K/AQ8 and selectable TurboQuant V rows into the ring.
     *
     * The block math intentionally mirrors `hip_tq8_quantize_kernel()` and
     * `hip_tq4_quantize_kernel()` instruction-for-instruction.  Adding the
     * verifier row as `blockIdx.z` changes only independent block scheduling;
     * it does not change the reduction or dot-product order within a row/head,
     * which is the requirement for serial-decode byte equivalence.
     */
    /**
     * @brief Publish a deterministic mean-key basis for an empty AQ8 cache.
     *
     * Each thread owns one coordinate and accumulates retained input rows in a
     * fixed ascending order. The following quantization grid is ordered behind
     * this kernel by the caller's exact HIP stream.
     */
    template <int D>
    __global__ void hip_prepare_attention_key_anchor_kernel(
        const float *__restrict__ key_input,
        float *__restrict__ key_anchor,
        const int *__restrict__ cached_count,
        const int *__restrict__ row_count,
        int max_seq_len,
        int verifier_rows,
        int n_kv_heads,
        bool head_major,
        AttentionKeyAnchorPolicy anchor_policy)
    {
        const int head = static_cast<int>(blockIdx.x);
        const int coordinate = static_cast<int>(threadIdx.x);
        if (head >= n_kv_heads || coordinate >= D || *cached_count != 0)
            return;

        const int rows_to_write = row_count ? *row_count : verifier_rows;
        const int source_start = rows_to_write > max_seq_len
                                     ? rows_to_write - max_seq_len
                                     : 0;
        if (rows_to_write <= source_start || rows_to_write > verifier_rows)
            return;

        if (anchor_policy == AttentionKeyAnchorPolicy::FirstRetained)
        {
            const size_t index = head_major
                                     ? (static_cast<size_t>(head) * verifier_rows + source_start) * D + coordinate
                                     : (static_cast<size_t>(source_start) * n_kv_heads + head) * D + coordinate;
            key_anchor[static_cast<size_t>(head) * D + coordinate] =
                key_input[index];
            return;
        }

        float sum = 0.0f;
        for (int row = source_start; row < rows_to_write; ++row)
        {
            const size_t index = head_major
                                     ? (static_cast<size_t>(head) * verifier_rows + row) * D + coordinate
                                     : (static_cast<size_t>(row) * n_kv_heads + head) * D + coordinate;
            sum += key_input[index];
        }
        key_anchor[static_cast<size_t>(head) * D + coordinate] =
            sum / static_cast<float>(rows_to_write - source_start);
    }

    template <int D, bool VUsesTQ8, bool VUsesQ8_1>
    __global__ void hip_tq_quantize_grouped_ring_kernel(
        const float *__restrict__ d_k_input,
        const float *__restrict__ d_v_input,
        const float *__restrict__ d_rotations,
        uint8_t *__restrict__ d_k_ring,
        uint8_t *__restrict__ d_v_ring,
        float *__restrict__ d_k_anchor,
        int ring_head_scalar,
        const int *__restrict__ d_ring_head,
        const int *__restrict__ d_cached_count,
        const int *__restrict__ d_row_count,
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

        const int first_ring_row = d_ring_head ? *d_ring_head : ring_head_scalar;
        const int ring_row = (first_ring_row + verifier_row) % max_seq_len;
        const int head = blockIdx.x;
        const int phase = blockIdx.y;
        const int tid = threadIdx.x;
        if (tid >= D)
            return;

        const size_t k_index = k_head_major
                                   ? (static_cast<size_t>(head) * verifier_rows + verifier_row) * D + tid
                                   : (static_cast<size_t>(verifier_row) * n_kv_heads + head) * D + tid;
        const size_t v_index = v_head_major
                                   ? (static_cast<size_t>(head) * verifier_rows + verifier_row) * D + tid
                                   : (static_cast<size_t>(verifier_row) * n_kv_heads + head) * D + tid;
        const float x = phase == 0 ? d_k_input[k_index] : d_v_input[v_index];

        /*
         * The key phase is a head-wide quadratic compander. It remains in this
         * two-phase launch so HIP graph capture adds no per-layer kernel node,
         * but it skips all value-only L2 normalization and matrix rotation.
         */
        if (phase == 0)
        {
            /*
             * The preceding stream-ordered kernel publishes the exact mean of
             * the initial retained pre-RoPE keys. Projection bias is largely
             * shared between tokens, so AQ8 spends its codes on the much
             * smaller token residual without a cross-block publication race.
             */
            float key_anchor = 0.0f;
            if (d_k_anchor)
                key_anchor = d_k_anchor[static_cast<size_t>(head) * D + tid];
            const float key_residual = x - key_anchor;
            __shared__ float s_key_wave_maxima[4];
            __shared__ float s_key_maximum;
            const float maximum =
                hip_block_reduce_max(fabsf(key_residual), s_key_wave_maxima);
            auto *block = reinterpret_cast<AttentionKeyQ8Block<D> *>(d_k_ring) +
                          (static_cast<size_t>(ring_row) * n_kv_heads + head);
            if (tid == 0)
            {
                s_key_maximum = maximum;
                block->quadratic_scale =
                    maximum * AttentionKeyQ8Block<D>::INVERSE_MAX_CODE_SQUARED;
            }
            __syncthreads();
            int8_t code = 0;
            if (s_key_maximum != 0.0f)
            {
                const int8_t magnitude =
                    hip_attention_key_q8_encode_magnitude(
                        fabsf(key_residual), s_key_maximum);
                code = key_residual < 0.0f
                           ? static_cast<int8_t>(-magnitude)
                           : magnitude;
            }
            block->codes[tid] = code;
            return;
        }

        if constexpr (VUsesQ8_1)
        {
            /*
             * A wave64 owns two independent Q8_1 blocks. The explicit
             * 32-lane shuffle width is part of the format contract: it gives
             * every block its own FP16 scale and raw integer sum while all
             * head coordinates are encoded in parallel in this same launch.
             */
            constexpr int q8_block_width = Q8_1Block::BLOCK_SIZE;
            constexpr int blocks_per_head = D / q8_block_width;
            const int lane = tid % q8_block_width;
            const int block_in_head = tid / q8_block_width;
            const float abs_max = hip_half_wave_reduce_max(fabsf(x));
            const float scale = abs_max > 0.0f ? abs_max / 127.0f : 0.0f;
            int q = scale > 0.0f ? __float2int_rn(x / scale) : 0;
            q = max(-127, min(127, q));
            const int sum = hip_half_wave_reduce_sum(q);
            auto *block = reinterpret_cast<Q8_1Block *>(d_v_ring) +
                          (static_cast<size_t>(ring_row) * n_kv_heads + head) *
                              blocks_per_head +
                          block_in_head;
            if (lane == 0)
            {
                block->d = __half_as_ushort(__float2half_rn(scale));
                block->sum_qs = static_cast<int16_t>(sum);
            }
            block->qs[lane] = static_cast<int8_t>(q);
            return;
        }

        // TurboQuant value phase: deterministic L2 norm and rotation.
        __shared__ float s_reduce[4];
        __shared__ float s_norm;
        __shared__ float s_combined_scale;
        float norm_sq = hip_block_reduce_sum(x * x, s_reduce);
        if (tid == 0)
        {
            const float norm = sqrtf(norm_sq);
            s_norm = norm;
            s_combined_scale = norm > 1e-30f
                                   ? sqrtf(static_cast<float>(D)) / norm
                                   : 0.0f;
        }
        __syncthreads();

        __shared__ float s_scaled[D];
        s_scaled[tid] = x * s_combined_scale;
        __syncthreads();

        const float *rot_row = d_rotations +
                               static_cast<size_t>(head) * D * D + tid * D;
        float rotated = 0.0f;
        for (int j = 0; j < D; ++j)
            rotated += rot_row[j] * s_scaled[j];

        if (VUsesTQ8)
        {
            int idx = 0;
            if (rotated > d_TQ8_THRESHOLDS_HIP[idx + 127]) idx += 128;
            if (rotated > d_TQ8_THRESHOLDS_HIP[idx + 63]) idx += 64;
            if (rotated > d_TQ8_THRESHOLDS_HIP[idx + 31]) idx += 32;
            if (rotated > d_TQ8_THRESHOLDS_HIP[idx + 15]) idx += 16;
            if (rotated > d_TQ8_THRESHOLDS_HIP[idx + 7]) idx += 8;
            if (rotated > d_TQ8_THRESHOLDS_HIP[idx + 3]) idx += 4;
            if (rotated > d_TQ8_THRESHOLDS_HIP[idx + 1]) idx += 2;
            if (rotated > d_TQ8_THRESHOLDS_HIP[idx]) idx += 1;

            const float selected_centroid = d_TQ8_CENTROIDS_HIP[idx];
            const float source_dot_centroids = hip_block_reduce_sum(
                rotated * selected_centroid, s_reduce);
            const float centroid_norm_sq = hip_block_reduce_sum(
                selected_centroid * selected_centroid, s_reduce);

            constexpr size_t block_size = sizeof(TQ8Block<D>);
            uint8_t *block = d_v_ring +
                             (static_cast<size_t>(ring_row) * n_kv_heads + head) *
                                 block_size;
            if (tid == 0)
            {
                auto *norms = reinterpret_cast<float *>(block);
                norms[0] = s_norm;
                norms[1] = centroid_norm_sq > 0.0f
                               ? s_norm * source_dot_centroids / centroid_norm_sq
                               : 0.0f;
            }
            block[2 * sizeof(float) + tid] = static_cast<uint8_t>(idx);
            return;
        }

        int idx = 0;
        if (rotated > d_TQ4_THRESHOLDS_HIP[idx + 7]) idx += 8;
        if (rotated > d_TQ4_THRESHOLDS_HIP[idx + 3]) idx += 4;
        if (rotated > d_TQ4_THRESHOLDS_HIP[idx + 1]) idx += 2;
        if (rotated > d_TQ4_THRESHOLDS_HIP[idx]) idx += 1;

        const float selected_centroid = d_TQ4_CENTROIDS_HIP[idx];
        const float source_dot_centroids = hip_block_reduce_sum(
            rotated * selected_centroid, s_reduce);
        const float centroid_norm_sq = hip_block_reduce_sum(
            selected_centroid * selected_centroid, s_reduce);

        __shared__ uint8_t s_indices[D];
        s_indices[tid] = static_cast<uint8_t>(idx);
        __syncthreads();

        constexpr size_t mse_bytes = D * 3 / 8;
        constexpr size_t block_size = sizeof(TQ4Block<D>);
        uint8_t *block = d_v_ring +
                         (static_cast<size_t>(ring_row) * n_kv_heads + head) * block_size;
        if (tid == 0)
        {
            auto *norms = reinterpret_cast<float *>(block);
            norms[0] = s_norm;
            norms[1] = centroid_norm_sq > 0.0f
                           ? s_norm * source_dot_centroids / centroid_norm_sq
                           : 0.0f;
            uint8_t *mse = block + 2 * sizeof(float);
            uint8_t *high = mse + mse_bytes;
            for (int i = 0; i < D; i += 8)
            {
                uint8_t low3[8];
                uint8_t high1 = 0;
                for (int j = 0; j < 8; ++j)
                {
                    low3[j] = s_indices[i + j] & 0x07;
                    high1 |= ((s_indices[i + j] >> 3) & 0x01) << j;
                }
                const int byte = i * 3 / 8;
                mse[byte + 0] = static_cast<uint8_t>(low3[0] | (low3[1] << 3) | (low3[2] << 6));
                mse[byte + 1] = static_cast<uint8_t>((low3[2] >> 2) | (low3[3] << 1) |
                                                     (low3[4] << 4) | (low3[5] << 7));
                mse[byte + 2] = static_cast<uint8_t>((low3[5] >> 1) | (low3[6] << 2) |
                                                     (low3[7] << 5));
                high[i / 8] = high1;
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
        const float norm = norms[1];

        const uint8_t idx = block_ptr[2 * sizeof(float) + tid];
        const float inv_scale = 1.0f / sqrtf(static_cast<float>(D));
        float centroid_val = d_TQ8_CENTROIDS_HIP[idx] * inv_scale;

        __shared__ float s_centroid[D];
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

            __shared__ float s_vals[D];
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
        const float norm = norms[1];

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

            const float norm = reinterpret_cast<const float *>(block_ptr)[1];
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

            const float norm = reinterpret_cast<const float *>(block_ptr)[1];

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

    /**
     * @brief Execute the canonical ordered inverse rotation accumulation.
     *
     * This routine is intentionally not inlined. HIP otherwise specializes
     * the same 128-term expression differently when a request dimension is
     * added to the caller's grid, producing occasional one-ULP FP16 drift.
     * Scalar and grouped dequant now call one compiled arithmetic body while
     * retaining full request/head/token parallelism around it. The FP32 result
     * remains available for the scalar K path's optional RoPE before the caller
     * performs its established FP16 rounding step.
     */
    template <int D>
    __device__ __attribute__((noinline)) float
    hip_tq_inverse_rotate_canonical(
        const float *__restrict__ rotation,
        const float *__restrict__ centroid,
        float norm,
        int output_dimension)
    {
        float value = 0.0f;
        for (int dimension = 0; dimension < D; ++dimension)
        {
            value = fmaf(
                rotation[dimension * D + output_dimension],
                centroid[dimension],
                value);
        }
        return value * norm;
    }

    /**
     * @brief Reconstruct one AQ8 key coordinate with a fixed rounding order.
     *
     * Serial and request-batched graph paths share this helper so the final
     * anchor addition cannot be contracted with the residual polynomial in
     * only one path. That fixed order preserves FP16 byte equivalence around
     * cancellation points.
     */
    template <int D>
    __device__ __forceinline__ float hip_reconstruct_attention_key_q8_coordinate(
        const AttentionKeyQ8Block<D> &block,
        const float *anchor,
        int head,
        int coordinate)
    {
        const float q = static_cast<float>(block.codes[coordinate]);
        const float scaled_code = block.quadratic_scale * q;
        const float residual = scaled_code * fabsf(q);
        if (!anchor)
            return residual;
        return fmaf(
            1.0f,
            residual,
            anchor[static_cast<size_t>(head) * D + coordinate]);
    }

    /**
     * @brief Decode AQ8 key rings without an intermediate FP32 tensor.
     *
     * The quadratic polynomial preserves the scalar oracle's multiplication order.
     * Optional RoPE uses the same canonical pair helper as every other ROCm
     * cache path, so scalar and grouped reads retain identical FP16 bytes.
     */
    template <int D, int TILE, typename OutT>
    __global__ void hip_attention_key_q8_dequantize_tiled_kernel(
        const AttentionKeyQ8Block<D> *__restrict__ cache,
        const float *__restrict__ anchor,
        OutT *__restrict__ output,
        int count,
        int n_kv_heads,
        float rope_theta,
        int position_start,
        int rope_dim,
        int max_seq_len,
        int tail)
    {
        const int head = static_cast<int>(blockIdx.x);
        const int tile_start = static_cast<int>(blockIdx.y) * TILE;
        const int coordinate = static_cast<int>(threadIdx.x);
        if (coordinate >= D)
            return;

        const int kv_dim = n_kv_heads * D;
        __shared__ float rope_row[D];
        const int tile_count = min(TILE, count - tile_start);
        for (int tile = 0; tile < tile_count; ++tile)
        {
            const int position = tile_start + tile;
            const int ring_position = (tail + position) % max_seq_len;
            const auto &block = cache[
                static_cast<size_t>(ring_position) * n_kv_heads + head];
            float value = hip_reconstruct_attention_key_q8_coordinate(
                block, anchor, head, coordinate);

            if (rope_theta > 0.0f)
            {
                rope_row[coordinate] = value;
                __syncthreads();
                const int effective_rope_dim =
                    rope_dim > 0 ? min(rope_dim, D) : D;
                if (coordinate < effective_rope_dim)
                {
                    const int half_dim = effective_rope_dim / 2;
                    const int pair = coordinate < half_dim
                                         ? coordinate
                                         : coordinate - half_dim;
                    const float partner = rope_row[
                        coordinate < half_dim
                            ? coordinate + half_dim
                            : coordinate - half_dim];
                    float rotated_x = 0.0f;
                    float rotated_y = 0.0f;
                    if (coordinate < half_dim)
                        canonical_hip_rope_pair(
                            value, partner, rope_theta, pair,
                            effective_rope_dim, position_start + position,
                            &rotated_x, &rotated_y);
                    else
                        canonical_hip_rope_pair(
                            partner, value, rope_theta, pair,
                            effective_rope_dim, position_start + position,
                            &rotated_x, &rotated_y);
                    value = coordinate < half_dim ? rotated_x : rotated_y;
                }
                __syncthreads();
            }
            output[position * kv_dim + head * D + coordinate] =
                static_cast<OutT>(value);
        }
    }

    /**
     * @brief Decode independent AQ8 rings from device-owned request metadata.
     */
    template <int D, int TILE>
    __global__ void hip_attention_key_q8_batched_ring_dequant_fp16_device_state_kernel(
        _Float16 *__restrict__ output,
        const void *const *__restrict__ entry_table,
        const void *const *__restrict__ anchor_table,
        const int *__restrict__ heads,
        const int *__restrict__ counts,
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
        const int coordinate = static_cast<int>(threadIdx.x);
        if (request >= request_count || head >= n_kv_heads || coordinate >= D)
            return;

        const int entry = entry_offset + request;
        int ring_count = counts[entry];
        ring_count = ring_count < 0
                         ? 0
                         : (ring_count > max_seq_len ? max_seq_len : ring_count);
        const int visible_count = min(ring_count, max_kv_len);
        const int skipped_rows = ring_count - visible_count;
        int tail = (heads[entry] - ring_count) % max_seq_len;
        if (tail < 0)
            tail += max_seq_len;
        const auto *cache = static_cast<const AttentionKeyQ8Block<D> *>(
            entry_table[entry]);
        const auto *anchor = static_cast<const float *>(anchor_table[entry]);
        const int kv_dim = n_kv_heads * D;
        __shared__ float rope_row[D];

        for (int tile_start = static_cast<int>(blockIdx.y) * TILE;
             tile_start < max_kv_len;
             tile_start += static_cast<int>(gridDim.y) * TILE)
        {
            const int output_tile_count = min(TILE, max_kv_len - tile_start);
            for (int tile = 0; tile < output_tile_count; ++tile)
            {
                const int output_token = tile_start + tile;
                const bool visible = output_token < visible_count;
                float value = 0.0f;
                if (visible)
                {
                    const int source_token =
                        (tail + skipped_rows + output_token) % max_seq_len;
                    const auto &block = cache[
                        static_cast<size_t>(source_token) * n_kv_heads + head];
                    value = hip_reconstruct_attention_key_q8_coordinate(
                        block, anchor, head, coordinate);
                    if (rope_theta > 0.0f)
                    {
                        rope_row[coordinate] = value;
                        __syncthreads();
                        const int effective_rope_dim =
                            rope_dim > 0 ? min(rope_dim, D) : D;
                        if (coordinate < effective_rope_dim)
                        {
                            const int half_dim = effective_rope_dim / 2;
                            const int pair = coordinate < half_dim
                                                 ? coordinate
                                                 : coordinate - half_dim;
                            const float partner = rope_row[
                                coordinate < half_dim
                                    ? coordinate + half_dim
                                    : coordinate - half_dim];
                            float rotated_x = 0.0f;
                            float rotated_y = 0.0f;
                            if (coordinate < half_dim)
                                canonical_hip_rope_pair(
                                    value, partner, rope_theta, pair,
                                    effective_rope_dim,
                                    position_start + output_token,
                                    &rotated_x, &rotated_y);
                            else
                                canonical_hip_rope_pair(
                                    partner, value, rope_theta, pair,
                                    effective_rope_dim,
                                    position_start + output_token,
                                    &rotated_x, &rotated_y);
                            value = coordinate < half_dim ? rotated_x : rotated_y;
                        }
                        __syncthreads();
                    }
                }
                const size_t output_index =
                    (static_cast<size_t>(request) * max_kv_len + output_token) *
                        static_cast<size_t>(kv_dim) +
                    static_cast<size_t>(head * D + coordinate);
                output[output_index] = static_cast<_Float16>(value);
            }
        }
    }

    /** @brief Decode block-linear Q8_1 value rings to a contiguous view. */
    template <int D, int TILE, typename OutT = float>
    __global__ void hip_q8_1_value_dequantize_tiled_kernel(
        const Q8_1Block *__restrict__ cache,
        OutT *__restrict__ output,
        int count,
        int n_kv_heads,
        int max_seq_len,
        int tail)
    {
        const int head = static_cast<int>(blockIdx.x);
        const int tile_start = static_cast<int>(blockIdx.y) * TILE;
        const int coordinate = static_cast<int>(threadIdx.x);
        if (coordinate >= D)
            return;

        constexpr int blocks_per_head = D / Q8_1Block::BLOCK_SIZE;
        const int kv_dim = n_kv_heads * D;
        const int tile_count = min(TILE, count - tile_start);
        for (int tile = 0; tile < tile_count; ++tile)
        {
            const int position = tile_start + tile;
            const int ring_position = (tail + position) % max_seq_len;
            const auto &block = cache[
                (static_cast<size_t>(ring_position) * n_kv_heads + head) *
                    blocks_per_head +
                coordinate / Q8_1Block::BLOCK_SIZE];
            const float scale = __half2float(__ushort_as_half(block.d));
            const float value = scale * static_cast<float>(
                block.qs[coordinate % Q8_1Block::BLOCK_SIZE]);
            output[position * kv_dim + head * D + coordinate] =
                static_cast<OutT>(value);
        }
    }

    /** @brief Decode request-batched Q8_1 value rings from resident metadata. */
    template <int D, int TILE>
    __global__ void hip_q8_1_value_batched_ring_dequant_fp16_device_state_kernel(
        _Float16 *__restrict__ output,
        const void *const *__restrict__ entry_table,
        const int *__restrict__ heads,
        const int *__restrict__ counts,
        int entry_offset,
        int request_count,
        int max_kv_len,
        int max_seq_len,
        int n_kv_heads)
    {
        const int head = static_cast<int>(blockIdx.x);
        const int request = static_cast<int>(blockIdx.z);
        const int coordinate = static_cast<int>(threadIdx.x);
        if (request >= request_count || head >= n_kv_heads || coordinate >= D)
            return;

        const int entry = entry_offset + request;
        int ring_count = counts[entry];
        ring_count = ring_count < 0
                         ? 0
                         : (ring_count > max_seq_len ? max_seq_len : ring_count);
        const int visible_count = min(ring_count, max_kv_len);
        const int skipped_rows = ring_count - visible_count;
        int tail = (heads[entry] - ring_count) % max_seq_len;
        if (tail < 0)
            tail += max_seq_len;
        const auto *cache = static_cast<const Q8_1Block *>(entry_table[entry]);
        constexpr int blocks_per_head = D / Q8_1Block::BLOCK_SIZE;
        const int kv_dim = n_kv_heads * D;

        for (int tile_start = static_cast<int>(blockIdx.y) * TILE;
             tile_start < max_kv_len;
             tile_start += static_cast<int>(gridDim.y) * TILE)
        {
            const int output_tile_count = min(TILE, max_kv_len - tile_start);
            for (int tile = 0; tile < output_tile_count; ++tile)
            {
                const int output_token = tile_start + tile;
                float value = 0.0f;
                if (output_token < visible_count)
                {
                    const int source_token =
                        (tail + skipped_rows + output_token) % max_seq_len;
                    const auto &block = cache[
                        (static_cast<size_t>(source_token) * n_kv_heads + head) *
                            blocks_per_head +
                        coordinate / Q8_1Block::BLOCK_SIZE];
                    const float scale =
                        __half2float(__ushort_as_half(block.d));
                    value = scale * static_cast<float>(
                        block.qs[coordinate % Q8_1Block::BLOCK_SIZE]);
                }
                const size_t output_index =
                    (static_cast<size_t>(request) * max_kv_len + output_token) *
                        static_cast<size_t>(kv_dim) +
                    static_cast<size_t>(head * D + coordinate);
                output[output_index] = static_cast<_Float16>(value);
            }
        }
    }

    template <int D, int TILE>
    __global__ void hip_tq8_dequantize_tiled_fp16_kernel(
        const uint8_t *__restrict__ d_tq8_bytes,
        const float *__restrict__ d_rotations,
        _Float16 *__restrict__ d_output,
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

        __shared__ float s_centroid[D];

        for (int t = 0; t < TILE; ++t)
        {
            const int pos = tile_start + t;
            if (pos >= count)
                return;

            const int ring_pos = (tail + pos) % max_seq_len;
            const uint8_t *block_ptr = d_tq8_bytes +
                (static_cast<size_t>(ring_pos) * n_kv_heads + head) * block_size;

            // TQ8 serial and grouped reads must consume the encoder's fitted
            // reconstruction radius, not the historical source norm.
            const float norm = reinterpret_cast<const float *>(block_ptr)[1];
            const uint8_t idx = block_ptr[2 * sizeof(float) + tid];
            s_centroid[tid] = d_TQ8_CENTROIDS_HIP[idx] * inv_scale;
            __syncthreads();

            float val = hip_tq_inverse_rotate_canonical<D>(
                rot_head, s_centroid, norm, tid);

            if (rope_theta > 0.0f)
            {
                // Every lane above reads the complete centroid vector.  No
                // lane may reuse its shared slot for the inverse-rotated
                // value until every other lane has finished those reads.  The
                // grouped request-batch kernel has the same barrier; keeping
                // it here makes scalar decode and grouped verification use an
                // identical, race-free floating-point operation order.
                __syncthreads();
                const int effective_rope_dim =
                    rope_dim > 0 ? min(rope_dim, D) : D;
                const int half_dim = effective_rope_dim / 2;
                s_centroid[tid] = val;
                __syncthreads();
                if (tid < effective_rope_dim)
                {
                    const int pair = tid < half_dim ? tid : tid - half_dim;
                    const float partner = s_centroid[
                        tid < half_dim ? tid + half_dim : tid - half_dim];
                    float rotated_x = 0.0f;
                    float rotated_y = 0.0f;
                    if (tid < half_dim)
                    {
                        canonical_hip_rope_pair(
                            val, partner, rope_theta, pair,
                            effective_rope_dim, position_start + pos,
                            &rotated_x, &rotated_y);
                        val = rotated_x;
                    }
                    else
                    {
                        canonical_hip_rope_pair(
                            partner, val, rope_theta, pair,
                            effective_rope_dim, position_start + pos,
                            &rotated_x, &rotated_y);
                        val = rotated_y;
                    }
                }
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

            const float norm = reinterpret_cast<const float *>(block_ptr)[1];

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

            d_output[pos * kv_dim + head * D + tid] =
                static_cast<_Float16>(hip_tq_inverse_rotate_canonical<D>(
                    rot_head, s_centroid, norm, tid));
            __syncthreads();
        }
    }

    /**
     * @brief Grouped ROCm TQ8 dequantization using resident ring metadata.
     *
     * The per-position operation order mirrors
     * `hip_tq8_dequantize_tiled_fp16_kernel`; `blockIdx.z` merely selects the
     * request-local ring and output span.
     */
    template <int D, int TILE>
    __global__ void hip_tq8_batched_ring_dequant_fp16_device_state_kernel(
        _Float16 *__restrict__ output,
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
        __shared__ float centroid[D];

        for (int tile_start = static_cast<int>(blockIdx.y) * TILE;
             tile_start < max_kv_len;
             tile_start += static_cast<int>(gridDim.y) * TILE)
        {
        for (int tile = 0; tile < TILE; ++tile)
        {
            const int output_token = tile_start + tile;
            if (output_token >= max_kv_len)
                break;
            const size_t output_index =
                (static_cast<size_t>(request) * max_kv_len + output_token) *
                    static_cast<size_t>(kv_dim) +
                static_cast<size_t>(head * D + tid);
            if (output_token >= visible_count)
            {
                output[output_index] = static_cast<_Float16>(0.0f);
                __syncthreads();
                continue;
            }
            const int source_token =
                (tail + skipped_rows + output_token) % max_seq_len;
            const uint8_t *block = cache +
                (static_cast<size_t>(source_token) * n_kv_heads + head) *
                    block_size;
            const float norm = reinterpret_cast<const float *>(block)[1];
            const uint8_t index = block[2 * sizeof(float) + tid];
            centroid[tid] = d_TQ8_CENTROIDS_HIP[index] * inv_scale;
            __syncthreads();

            float value = hip_tq_inverse_rotate_canonical<D>(
                rotation, centroid, norm, tid);
            if (rope_theta > 0.0f)
            {
                __syncthreads();
                centroid[tid] = value;
                __syncthreads();
                const int effective_rope_dim =
                    rope_dim > 0 ? min(rope_dim, D) : D;
                if (tid < effective_rope_dim)
                {
                    const int half_dim = effective_rope_dim / 2;
                    const int pair = tid < half_dim ? tid : tid - half_dim;
                    const float partner = centroid[
                        tid < half_dim ? tid + half_dim : tid - half_dim];
                    float rotated_x = 0.0f;
                    float rotated_y = 0.0f;
                    if (tid < half_dim)
                    {
                        canonical_hip_rope_pair(
                            value, partner, rope_theta, pair,
                            effective_rope_dim,
                            position_start + output_token,
                            &rotated_x, &rotated_y);
                        value = rotated_x;
                    }
                    else
                    {
                        canonical_hip_rope_pair(
                            partner, value, rope_theta, pair,
                            effective_rope_dim,
                            position_start + output_token,
                            &rotated_x, &rotated_y);
                        value = rotated_y;
                    }
                }
            }
            output[output_index] = static_cast<_Float16>(value);
            __syncthreads();
        }
        }
    }

    /**
     * @brief Grouped ROCm TQ4 dequantization using resident ring metadata.
     */
    template <int D, int TILE>
    __global__ void hip_tq4_batched_ring_dequant_fp16_device_state_kernel(
        _Float16 *__restrict__ output,
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
        const float inv_scale = 1.0f / sqrtf(static_cast<float>(D));
        const int kv_dim = n_kv_heads * D;
        const int group8 = tid / 8;
        const int within = tid % 8;
        const int byte_index = group8 * 3;
        __shared__ float centroid[D];

        for (int tile_start = static_cast<int>(blockIdx.y) * TILE;
             tile_start < max_kv_len;
             tile_start += static_cast<int>(gridDim.y) * TILE)
        {
        for (int tile = 0; tile < TILE; ++tile)
        {
            const int output_token = tile_start + tile;
            if (output_token >= max_kv_len)
                break;
            const size_t output_index =
                (static_cast<size_t>(request) * max_kv_len + output_token) *
                    static_cast<size_t>(kv_dim) +
                static_cast<size_t>(head * D + tid);
            if (output_token >= visible_count)
            {
                output[output_index] = static_cast<_Float16>(0.0f);
                __syncthreads();
                continue;
            }
            const int source_token =
                (tail + skipped_rows + output_token) % max_seq_len;
            const uint8_t *block = cache +
                (static_cast<size_t>(source_token) * n_kv_heads + head) *
                    block_size;
            const float norm = reinterpret_cast<const float *>(block)[1];
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
            centroid[tid] = d_TQ4_CENTROIDS_HIP[index] * inv_scale;
            __syncthreads();

            output[output_index] =
                static_cast<_Float16>(hip_tq_inverse_rotate_canonical<D>(
                    rotation, centroid, norm, tid));
            __syncthreads();
        }
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
             * The append's sequence-state kernel is ordered before dequant on
             * this stream. Derive the newest compressed row and its linear
             * scratch destination directly from that canonical device state;
             * no host dequant mailbox or extra planning launch is required.
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
            const float norm = reinterpret_cast<const float *>(block_ptr)[1];
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
                /*
                 * Do not reuse the centroid tile until every wave has
                 * completed every inverse-rotation read. This explicit phase
                 * barrier mirrors CUDA and prevents D=128 RoPE from racing a
                 * later wave's dot product during captured replay.
                 */
                __syncthreads();
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
            const float norm = reinterpret_cast<const float *>(block_ptr)[1];

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

        // Half-split: pair (i, i + half_dim)
        const int head_base = pos * kv_dim + head * head_dim;
        const int idx0 = head_base + pair_idx;
        const int idx1 = head_base + pair_idx + half_dim;
        float x = static_cast<float>(d_K[idx0]);
        float y = static_cast<float>(d_K[idx1]);

        float rotated_x = 0.0f;
        float rotated_y = 0.0f;
        canonical_hip_rope_pair(
            x, y, rope_theta, pair_idx, effective_rope_dim,
            position_start + pos, &rotated_x, &rotated_y);
        d_K[idx0] = static_cast<_Float16>(rotated_x);
        d_K[idx1] = static_cast<_Float16>(rotated_y);
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

        // Half-split: pair (i, i + half_dim)
        const int head_base = pos * kv_dim + head * head_dim;
        const int idx0 = head_base + pair_idx;
        const int idx1 = head_base + pair_idx + half_dim;
        float x = d_K[idx0];
        float y = d_K[idx1];

        const HipFP32RoPEPair rotated = canonical_hip_fp32_rope_pair(
            x, y, rope_theta, pair_idx, effective_rope_dim,
            position_start + pos);
        d_K[idx0] = rotated.first;
        d_K[idx1] = rotated.second;
    }

    __global__ void hip_rope_apply_batched_fp16_device_state_kernel(
        _Float16 *__restrict__ d_K,
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
        int ring_count = counts[entry_offset + request];
        ring_count = ring_count < 0
                         ? 0
                         : (ring_count > max_seq_len ? max_seq_len : ring_count);
        const int visible_count =
            ring_count < max_kv_len ? ring_count : max_kv_len;
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
            canonical_hip_rope_pair(
                static_cast<float>(d_K[first]),
                static_cast<float>(d_K[second]),
                rope_theta, pair, effective_rope_dim,
                position_start + token, &rotated_x, &rotated_y);
            d_K[first] = static_cast<_Float16>(rotated_x);
            d_K[second] = static_cast<_Float16>(rotated_y);
        }
    }

    __global__ void hip_rope_apply_batched_fp32_ring_to_fp16_device_state_kernel(
        _Float16 *__restrict__ d_K_out,
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
        int ring_count = counts[entry];
        ring_count = ring_count < 0
                         ? 0
                         : (ring_count > max_seq_len ? max_seq_len : ring_count);
        const int visible_count =
            ring_count < max_kv_len ? ring_count : max_kv_len;
        const int skipped_rows = ring_count - visible_count;
        int tail = (heads[entry] - ring_count) % max_seq_len;
        if (tail < 0)
            tail += max_seq_len;
        const int kv_dim = n_kv_heads * head_dim;
        auto *half_output = reinterpret_cast<half *>(d_K_out);
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
            const HipFP32RoPEPair rotated = canonical_hip_fp32_rope_pair(
                entry_table[entry][source_base + pair],
                entry_table[entry][source_base + pair + half_dim],
                rope_theta, pair, effective_rope_dim,
                position_start + output_token);
            const size_t output_base =
                (static_cast<size_t>(request) * max_kv_len + output_token) *
                    kv_dim +
                static_cast<size_t>(head) * head_dim;
            half_output[output_base + pair] = __float2half(rotated.first);
            half_output[output_base + pair + half_dim] =
                __float2half(rotated.second);
        }
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

    namespace
    {
        /** @brief Shared launcher for static-head and dynamic-head grouped TQ publication. */
        bool launch_hip_tq_quantize_grouped_ring(
            const float *d_k_input, const float *d_v_input,
            const float *d_rotations,
            void *d_k_ring, void *d_v_ring, float *d_k_anchor,
            int ring_head, const int *d_ring_head,
            const int *d_cached_count,
            const int *d_row_count,
            int max_seq_len, int verifier_rows,
            int n_kv_heads, int head_dim,
            bool k_head_major, bool v_head_major,
            TurboQuantKVMode mode,
            AttentionKeyAnchorPolicy anchor_policy,
            hipStream_t stream)
        {
            if (!d_k_input || !d_v_input ||
                (turboQuantValueUsesRotation(mode) && !d_rotations) ||
                !d_k_ring || !d_v_ring ||
                !stream || max_seq_len <= 0 || verifier_rows <= 0 || verifier_rows > 65535 ||
                n_kv_heads <= 0 || (d_ring_head == nullptr && ring_head < 0))
            {
                return false;
            }

            if ((d_k_anchor == nullptr) != (d_cached_count == nullptr))
                return false;
            if (d_k_anchor)
            {
                if (head_dim == 64)
                    hipLaunchKernelGGL(
                        hip_prepare_attention_key_anchor_kernel<64>,
                        dim3(n_kv_heads), dim3(64), 0, stream,
                        d_k_input, d_k_anchor, d_cached_count, d_row_count,
                        max_seq_len, verifier_rows, n_kv_heads, k_head_major,
                        anchor_policy);
                else if (head_dim == 128)
                    hipLaunchKernelGGL(
                        hip_prepare_attention_key_anchor_kernel<128>,
                        dim3(n_kv_heads), dim3(128), 0, stream,
                        d_k_input, d_k_anchor, d_cached_count, d_row_count,
                        max_seq_len, verifier_rows, n_kv_heads, k_head_major,
                        anchor_policy);
                else if (head_dim == 256)
                    hipLaunchKernelGGL(
                        hip_prepare_attention_key_anchor_kernel<256>,
                        dim3(n_kv_heads), dim3(256), 0, stream,
                        d_k_input, d_k_anchor, d_cached_count, d_row_count,
                        max_seq_len, verifier_rows, n_kv_heads, k_head_major,
                        anchor_policy);
                else
                    return false;
            }

            const dim3 grid(n_kv_heads, 2, verifier_rows);
            if (head_dim == 64)
            {
                if (mode == TurboQuantKVMode::AQ8_K_Q8_1_V)
                    hipLaunchKernelGGL(
                        (hip_tq_quantize_grouped_ring_kernel<64, false, true>), grid, dim3(64), 0, stream,
                        d_k_input, d_v_input, d_rotations,
                        static_cast<uint8_t *>(d_k_ring), static_cast<uint8_t *>(d_v_ring),
                        d_k_anchor, ring_head, d_ring_head, d_cached_count, d_row_count,
                        max_seq_len, verifier_rows, n_kv_heads,
                        k_head_major, v_head_major);
                else if (mode == TurboQuantKVMode::AQ8_K_TQ8_V)
                    hipLaunchKernelGGL(
                        (hip_tq_quantize_grouped_ring_kernel<64, true, false>), grid, dim3(64), 0, stream,
                        d_k_input, d_v_input, d_rotations,
                        static_cast<uint8_t *>(d_k_ring), static_cast<uint8_t *>(d_v_ring),
                        d_k_anchor, ring_head, d_ring_head, d_cached_count, d_row_count,
                        max_seq_len, verifier_rows, n_kv_heads,
                        k_head_major, v_head_major);
                else
                    hipLaunchKernelGGL(
                        (hip_tq_quantize_grouped_ring_kernel<64, false, false>), grid, dim3(64), 0, stream,
                        d_k_input, d_v_input, d_rotations,
                        static_cast<uint8_t *>(d_k_ring), static_cast<uint8_t *>(d_v_ring),
                        d_k_anchor, ring_head, d_ring_head, d_cached_count, d_row_count,
                        max_seq_len, verifier_rows, n_kv_heads,
                        k_head_major, v_head_major);
            }
            else if (head_dim == 128)
            {
                if (mode == TurboQuantKVMode::AQ8_K_Q8_1_V)
                    hipLaunchKernelGGL(
                        (hip_tq_quantize_grouped_ring_kernel<128, false, true>), grid, dim3(128), 0, stream,
                        d_k_input, d_v_input, d_rotations,
                        static_cast<uint8_t *>(d_k_ring), static_cast<uint8_t *>(d_v_ring),
                        d_k_anchor, ring_head, d_ring_head, d_cached_count, d_row_count,
                        max_seq_len, verifier_rows, n_kv_heads,
                        k_head_major, v_head_major);
                else if (mode == TurboQuantKVMode::AQ8_K_TQ8_V)
                    hipLaunchKernelGGL(
                        (hip_tq_quantize_grouped_ring_kernel<128, true, false>), grid, dim3(128), 0, stream,
                        d_k_input, d_v_input, d_rotations,
                        static_cast<uint8_t *>(d_k_ring), static_cast<uint8_t *>(d_v_ring),
                        d_k_anchor, ring_head, d_ring_head, d_cached_count, d_row_count,
                        max_seq_len, verifier_rows, n_kv_heads,
                        k_head_major, v_head_major);
                else
                    hipLaunchKernelGGL(
                        (hip_tq_quantize_grouped_ring_kernel<128, false, false>), grid, dim3(128), 0, stream,
                        d_k_input, d_v_input, d_rotations,
                        static_cast<uint8_t *>(d_k_ring), static_cast<uint8_t *>(d_v_ring),
                        d_k_anchor, ring_head, d_ring_head, d_cached_count, d_row_count,
                        max_seq_len, verifier_rows, n_kv_heads,
                        k_head_major, v_head_major);
            }
            else if (head_dim == 256)
            {
                if (mode == TurboQuantKVMode::AQ8_K_Q8_1_V)
                    hipLaunchKernelGGL(
                        (hip_tq_quantize_grouped_ring_kernel<256, false, true>), grid, dim3(256), 0, stream,
                        d_k_input, d_v_input, d_rotations,
                        static_cast<uint8_t *>(d_k_ring), static_cast<uint8_t *>(d_v_ring),
                        d_k_anchor, ring_head, d_ring_head, d_cached_count, d_row_count,
                        max_seq_len, verifier_rows, n_kv_heads,
                        k_head_major, v_head_major);
                else if (mode == TurboQuantKVMode::AQ8_K_TQ8_V)
                    hipLaunchKernelGGL(
                        (hip_tq_quantize_grouped_ring_kernel<256, true, false>), grid, dim3(256), 0, stream,
                        d_k_input, d_v_input, d_rotations,
                        static_cast<uint8_t *>(d_k_ring), static_cast<uint8_t *>(d_v_ring),
                        d_k_anchor, ring_head, d_ring_head, d_cached_count, d_row_count,
                        max_seq_len, verifier_rows, n_kv_heads,
                        k_head_major, v_head_major);
                else
                    hipLaunchKernelGGL(
                        (hip_tq_quantize_grouped_ring_kernel<256, false, false>), grid, dim3(256), 0, stream,
                        d_k_input, d_v_input, d_rotations,
                        static_cast<uint8_t *>(d_k_ring), static_cast<uint8_t *>(d_v_ring),
                        d_k_anchor, ring_head, d_ring_head, d_cached_count, d_row_count,
                        max_seq_len, verifier_rows, n_kv_heads,
                        k_head_major, v_head_major);
            }
            else
            {
                return false;
            }
            return hipGetLastError() == hipSuccess;
        }
    } // namespace

    extern "C" bool hip_tq_quantize_grouped_ring(
        const float *d_k_input, const float *d_v_input,
        const float *d_rotations,
        void *d_k_ring, void *d_v_ring,
        int ring_head, int max_seq_len,
        int verifier_rows, int n_kv_heads, int head_dim,
        bool k_head_major, bool v_head_major,
        TurboQuantKVMode mode,
        hipStream_t stream)
    {
        return launch_hip_tq_quantize_grouped_ring(
            d_k_input, d_v_input, d_rotations, d_k_ring, d_v_ring,
            nullptr, ring_head, nullptr, nullptr, nullptr,
            max_seq_len, verifier_rows, n_kv_heads, head_dim,
            k_head_major, v_head_major, mode,
            AttentionKeyAnchorPolicy::FirstRetained, stream);
    }

    extern "C" bool hip_tq_quantize_grouped_ring_dynamic(
        const float *d_k_input, const float *d_v_input,
        const float *d_rotations,
        void *d_k_ring, void *d_v_ring, float *d_k_anchor,
        const int *d_ring_head, const int *d_cached_count,
        const int *d_row_count, int max_seq_len,
        int verifier_rows, int n_kv_heads, int head_dim,
        bool k_head_major, bool v_head_major,
        TurboQuantKVMode mode,
        AttentionKeyAnchorPolicy anchor_policy,
        hipStream_t stream)
    {
        if (!d_ring_head || !d_cached_count || !d_k_anchor)
            return false;
        return launch_hip_tq_quantize_grouped_ring(
            d_k_input, d_v_input, d_rotations, d_k_ring, d_v_ring,
            d_k_anchor, 0, d_ring_head, d_cached_count, d_row_count,
            max_seq_len, verifier_rows, n_kv_heads, head_dim,
            k_head_major, v_head_major, mode, anchor_policy, stream);
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

        if (rope_theta > 0.0f)
            hip_tq_upload_rope_freqs(rope_theta, head_dim, stream);

        // Tiled kernels use R (non-transposed, row-major) for coalesced access:
        // R[j][tid] at offset j*D + tid → consecutive threads read consecutive addresses.
        constexpr int TILE = 16;

        if (head_dim == 64)
        {
            const dim3 grid(n_kv_heads, (count + TILE - 1) / TILE);
            hipLaunchKernelGGL((hip_attention_key_q8_dequantize_tiled_kernel<64, TILE, float>), grid, dim3(64), 0, stream,
                               static_cast<const AttentionKeyQ8Block<64> *>(d_K_cache), nullptr, d_K_out,
                               count, n_kv_heads, rope_theta, position_start, 64, max_seq_len, tail);
            hipLaunchKernelGGL((hip_tq4_dequantize_tiled_kernel<64, TILE>), grid, dim3(64), 0, stream,
                               static_cast<const uint8_t *>(d_V_cache), d_V_rotations, d_V_out,
                               count, n_kv_heads, max_seq_len, tail);
        }
        else if (head_dim == 128)
        {
            const dim3 grid(n_kv_heads, (count + TILE - 1) / TILE);
            hipLaunchKernelGGL((hip_attention_key_q8_dequantize_tiled_kernel<128, TILE, float>), grid, dim3(128), 0, stream,
                               static_cast<const AttentionKeyQ8Block<128> *>(d_K_cache), nullptr, d_K_out,
                               count, n_kv_heads, rope_theta, position_start, 128, max_seq_len, tail);
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

    extern "C" bool hip_rope_apply_batched_fp16_device_state(
        _Float16 *d_K,
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
        hipStream_t stream)
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
        const unsigned int blocks = std::min(
            static_cast<unsigned int>((pairs_per_request + 255) / 256),
            resident_rope_blocks);
        hipLaunchKernelGGL(
            hip_rope_apply_batched_fp16_device_state_kernel,
            dim3(blocks, static_cast<unsigned int>(request_count)),
            dim3(256), 0, stream,
            d_K, d_counts, entry_offset, request_count, max_kv_len,
            max_seq_len, n_kv_heads, head_dim, rope_theta,
            position_start, effective_rope_dim);
        return hipGetLastError() == hipSuccess;
    }

    extern "C" bool hip_rope_apply_batched_fp32_ring_to_fp16_device_state(
        _Float16 *d_K_out,
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
        hipStream_t stream)
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
        const unsigned int blocks = std::min(
            static_cast<unsigned int>((pairs_per_request + 255) / 256),
            resident_rope_blocks);
        hipLaunchKernelGGL(
            hip_rope_apply_batched_fp32_ring_to_fp16_device_state_kernel,
            dim3(blocks, static_cast<unsigned int>(request_count)),
            dim3(256), 0, stream,
            d_K_out, d_K_entry_table, d_heads, d_counts, entry_offset,
            request_count, max_kv_len, max_seq_len, n_kv_heads, head_dim,
            rope_theta, position_start, effective_rope_dim);
        return hipGetLastError() == hipSuccess;
    }

    // =========================================================================
    // FP16 Linearize + Dequant (full sequence, for prefill)
    // =========================================================================

    extern "C" bool hip_tq_ring_linearize_dequant_fp16(
        _Float16 *d_K_out, _Float16 *d_V_out,
        const void *d_K_cache, const void *d_V_cache,
        const float *d_K_anchor,
        const float *d_K_rotations_t, const float *d_V_rotations_t,
        const float *d_K_rotations, const float *d_V_rotations,
        int tail, int count, int max_seq_len,
        int n_kv_heads, int head_dim,
        float rope_theta, int position_start, int rope_dim,
        TurboQuantKVMode mode,
        hipStream_t stream)
    {
        if (count <= 0)
            return true;

        constexpr int TILE = 16;

        if (head_dim == 64)
        {
            const dim3 grid(n_kv_heads, (count + TILE - 1) / TILE);
            hipLaunchKernelGGL((hip_attention_key_q8_dequantize_tiled_kernel<64, TILE, _Float16>), grid, dim3(64), 0, stream,
                               static_cast<const AttentionKeyQ8Block<64> *>(d_K_cache), d_K_anchor, d_K_out,
                               count, n_kv_heads, rope_theta, position_start,
                               rope_dim, max_seq_len, tail);
            if (mode == TurboQuantKVMode::AQ8_K_Q8_1_V)
                hipLaunchKernelGGL((hip_q8_1_value_dequantize_tiled_kernel<64, TILE, _Float16>), grid, dim3(64), 0, stream,
                                   static_cast<const Q8_1Block *>(d_V_cache), d_V_out,
                                   count, n_kv_heads, max_seq_len, tail);
            else if (mode == TurboQuantKVMode::AQ8_K_TQ8_V)
                hipLaunchKernelGGL((hip_tq8_dequantize_tiled_fp16_kernel<64, TILE>), grid, dim3(64), 0, stream,
                                   static_cast<const uint8_t *>(d_V_cache), d_V_rotations, d_V_out,
                                   count, n_kv_heads, 0.0f, 0, 0, max_seq_len, tail);
            else
                hipLaunchKernelGGL((hip_tq4_dequantize_tiled_fp16_kernel<64, TILE>), grid, dim3(64), 0, stream,
                                   static_cast<const uint8_t *>(d_V_cache), d_V_rotations, d_V_out,
                                   count, n_kv_heads, max_seq_len, tail);
        }
        else if (head_dim == 128)
        {
            const dim3 grid(n_kv_heads, (count + TILE - 1) / TILE);
            hipLaunchKernelGGL((hip_attention_key_q8_dequantize_tiled_kernel<128, TILE, _Float16>), grid, dim3(128), 0, stream,
                               static_cast<const AttentionKeyQ8Block<128> *>(d_K_cache), d_K_anchor, d_K_out,
                               count, n_kv_heads, rope_theta, position_start,
                               rope_dim, max_seq_len, tail);
            if (mode == TurboQuantKVMode::AQ8_K_Q8_1_V)
                hipLaunchKernelGGL((hip_q8_1_value_dequantize_tiled_kernel<128, TILE, _Float16>), grid, dim3(128), 0, stream,
                                   static_cast<const Q8_1Block *>(d_V_cache), d_V_out,
                                   count, n_kv_heads, max_seq_len, tail);
            else if (mode == TurboQuantKVMode::AQ8_K_TQ8_V)
                hipLaunchKernelGGL((hip_tq8_dequantize_tiled_fp16_kernel<128, TILE>), grid, dim3(128), 0, stream,
                                   static_cast<const uint8_t *>(d_V_cache), d_V_rotations, d_V_out,
                                   count, n_kv_heads, 0.0f, 0, 0, max_seq_len, tail);
            else
                hipLaunchKernelGGL((hip_tq4_dequantize_tiled_fp16_kernel<128, TILE>), grid, dim3(128), 0, stream,
                                   static_cast<const uint8_t *>(d_V_cache), d_V_rotations, d_V_out,
                                   count, n_kv_heads, max_seq_len, tail);
        }
        else if (head_dim == 256)
        {
            const dim3 grid(n_kv_heads, (count + TILE - 1) / TILE);
            hipLaunchKernelGGL((hip_attention_key_q8_dequantize_tiled_kernel<256, TILE, _Float16>), grid, dim3(256), 0, stream,
                               static_cast<const AttentionKeyQ8Block<256> *>(d_K_cache), d_K_anchor, d_K_out,
                               count, n_kv_heads, rope_theta, position_start,
                               rope_dim, max_seq_len, tail);
            if (mode == TurboQuantKVMode::AQ8_K_Q8_1_V)
                hipLaunchKernelGGL((hip_q8_1_value_dequantize_tiled_kernel<256, TILE, _Float16>), grid, dim3(256), 0, stream,
                                   static_cast<const Q8_1Block *>(d_V_cache), d_V_out,
                                   count, n_kv_heads, max_seq_len, tail);
            else if (mode == TurboQuantKVMode::AQ8_K_TQ8_V)
                hipLaunchKernelGGL((hip_tq8_dequantize_tiled_fp16_kernel<256, TILE>), grid, dim3(256), 0, stream,
                                   static_cast<const uint8_t *>(d_V_cache), d_V_rotations, d_V_out,
                                   count, n_kv_heads, 0.0f, 0, 0, max_seq_len, tail);
            else
                hipLaunchKernelGGL((hip_tq4_dequantize_tiled_fp16_kernel<256, TILE>), grid, dim3(256), 0, stream,
                                   static_cast<const uint8_t *>(d_V_cache), d_V_rotations, d_V_out,
                                   count, n_kv_heads, max_seq_len, tail);
        }
        else
            return false;

        return hipGetLastError() == hipSuccess;
    }

    extern "C" bool hip_tq_batched_ring_dequant_fp16_device_state(
        _Float16 *d_K_out,
        _Float16 *d_V_out,
        const void *const *d_K_entry_table,
        const void *const *d_V_entry_table,
        const void *const *d_K_anchor_table,
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
        hipStream_t stream)
    {
        if (!d_K_out || !d_V_out || !d_K_entry_table || !d_V_entry_table ||
            !d_K_anchor_table ||
            !d_heads || !d_counts ||
            (turboQuantValueUsesRotation(mode) && !d_rotations) || !stream ||
            entry_offset < 0 || request_count <= 0 || max_kv_len <= 0 ||
            max_seq_len <= 0 || n_kv_heads <= 0)
        {
            return false;
        }

        constexpr int tile = 16;
        constexpr unsigned int resident_tile_blocks = 8;
        const dim3 grid(
            static_cast<unsigned int>(n_kv_heads),
            std::min(
                static_cast<unsigned int>((max_kv_len + tile - 1) / tile),
                resident_tile_blocks),
            static_cast<unsigned int>(request_count));
        if (head_dim == 64)
        {
            hipLaunchKernelGGL(
                (hip_attention_key_q8_batched_ring_dequant_fp16_device_state_kernel<64, tile>),
                grid, dim3(64), 0, stream,
                d_K_out, d_K_entry_table, d_K_anchor_table, d_heads, d_counts,
                entry_offset, request_count, max_kv_len, max_seq_len,
                n_kv_heads, rope_theta, position_start, rope_dim);
            if (mode == TurboQuantKVMode::AQ8_K_Q8_1_V)
                hipLaunchKernelGGL(
                    (hip_q8_1_value_batched_ring_dequant_fp16_device_state_kernel<64, tile>),
                    grid, dim3(64), 0, stream,
                    d_V_out, d_V_entry_table, d_heads, d_counts,
                    entry_offset, request_count, max_kv_len, max_seq_len,
                    n_kv_heads);
            else if (mode == TurboQuantKVMode::AQ8_K_TQ8_V)
                hipLaunchKernelGGL(
                    (hip_tq8_batched_ring_dequant_fp16_device_state_kernel<64, tile>),
                    grid, dim3(64), 0, stream,
                    d_V_out, d_V_entry_table, d_heads, d_counts, d_rotations,
                    entry_offset, request_count, max_kv_len, max_seq_len,
                    n_kv_heads, 0.0f, 0, 0);
            else
                hipLaunchKernelGGL(
                    (hip_tq4_batched_ring_dequant_fp16_device_state_kernel<64, tile>),
                    grid, dim3(64), 0, stream,
                    d_V_out, d_V_entry_table, d_heads, d_counts, d_rotations,
                    entry_offset, request_count, max_kv_len, max_seq_len,
                    n_kv_heads);
        }
        else if (head_dim == 128)
        {
            hipLaunchKernelGGL(
                (hip_attention_key_q8_batched_ring_dequant_fp16_device_state_kernel<128, tile>),
                grid, dim3(128), 0, stream,
                d_K_out, d_K_entry_table, d_K_anchor_table, d_heads, d_counts,
                entry_offset, request_count, max_kv_len, max_seq_len,
                n_kv_heads, rope_theta, position_start, rope_dim);
            if (mode == TurboQuantKVMode::AQ8_K_Q8_1_V)
                hipLaunchKernelGGL(
                    (hip_q8_1_value_batched_ring_dequant_fp16_device_state_kernel<128, tile>),
                    grid, dim3(128), 0, stream,
                    d_V_out, d_V_entry_table, d_heads, d_counts,
                    entry_offset, request_count, max_kv_len, max_seq_len,
                    n_kv_heads);
            else if (mode == TurboQuantKVMode::AQ8_K_TQ8_V)
                hipLaunchKernelGGL(
                    (hip_tq8_batched_ring_dequant_fp16_device_state_kernel<128, tile>),
                    grid, dim3(128), 0, stream,
                    d_V_out, d_V_entry_table, d_heads, d_counts, d_rotations,
                    entry_offset, request_count, max_kv_len, max_seq_len,
                    n_kv_heads, 0.0f, 0, 0);
            else
                hipLaunchKernelGGL(
                    (hip_tq4_batched_ring_dequant_fp16_device_state_kernel<128, tile>),
                    grid, dim3(128), 0, stream,
                    d_V_out, d_V_entry_table, d_heads, d_counts, d_rotations,
                    entry_offset, request_count, max_kv_len, max_seq_len,
                    n_kv_heads);
        }
        else if (head_dim == 256)
        {
            hipLaunchKernelGGL(
                (hip_attention_key_q8_batched_ring_dequant_fp16_device_state_kernel<256, tile>),
                grid, dim3(256), 0, stream,
                d_K_out, d_K_entry_table, d_K_anchor_table, d_heads, d_counts,
                entry_offset, request_count, max_kv_len, max_seq_len,
                n_kv_heads, rope_theta, position_start, rope_dim);
            if (mode == TurboQuantKVMode::AQ8_K_Q8_1_V)
                hipLaunchKernelGGL(
                    (hip_q8_1_value_batched_ring_dequant_fp16_device_state_kernel<256, tile>),
                    grid, dim3(256), 0, stream,
                    d_V_out, d_V_entry_table, d_heads, d_counts,
                    entry_offset, request_count, max_kv_len, max_seq_len,
                    n_kv_heads);
            else if (mode == TurboQuantKVMode::AQ8_K_TQ8_V)
                hipLaunchKernelGGL(
                    (hip_tq8_batched_ring_dequant_fp16_device_state_kernel<256, tile>),
                    grid, dim3(256), 0, stream,
                    d_V_out, d_V_entry_table, d_heads, d_counts, d_rotations,
                    entry_offset, request_count, max_kv_len, max_seq_len,
                    n_kv_heads, 0.0f, 0, 0);
            else
                hipLaunchKernelGGL(
                    (hip_tq4_batched_ring_dequant_fp16_device_state_kernel<256, tile>),
                    grid, dim3(256), 0, stream,
                    d_V_out, d_V_entry_table, d_heads, d_counts, d_rotations,
                    entry_offset, request_count, max_kv_len, max_seq_len,
                    n_kv_heads);
        }
        else
        {
            return false;
        }
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
                               nullptr, nullptr, 0, 0, 0);
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
                               nullptr, nullptr, 0, 0, 0);
        }
        else
            return false;

        return hipGetLastError() == hipSuccess;
    }

    extern "C" bool hip_tq_incremental_single_fp16_dynamic(
        _Float16 *d_K_base, _Float16 *d_V_base,
        const void *d_K_cache, const void *d_V_cache,
        const float *d_K_rotation, const float *d_V_rotation,
        const int *d_ring_head, const int *d_cached_count,
        int max_seq_len, int kv_dim, int position_start,
        int n_kv_heads, int head_dim,
        float rope_theta,
        hipStream_t stream)
    {
        if (!d_K_base || !d_V_base || !d_K_cache || !d_V_cache ||
            !d_K_rotation || !d_V_rotation ||
            !d_ring_head || !d_cached_count ||
            max_seq_len <= 0 || kv_dim <= 0 || n_kv_heads <= 0)
            return false;

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
                               d_ring_head, d_cached_count,
                               max_seq_len, kv_dim, position_start);
        }
        else if (head_dim == 128)
        {
            hipLaunchKernelGGL((hip_tq_incremental_fused_kernel<128>), grid, dim3(128), 0, stream,
                               static_cast<const uint8_t *>(d_K_cache),
                               static_cast<const uint8_t *>(d_V_cache),
                               d_K_base, d_V_base,
                               d_K_rotation, d_V_rotation,
                               0, 0, n_kv_heads, rope_theta, 0,
                               d_ring_head, d_cached_count,
                               max_seq_len, kv_dim, position_start);
        }
        else
            return false;

        return hipGetLastError() == hipSuccess;
    }

} // namespace llaminar2
