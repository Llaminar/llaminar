/**
 * @file CUDAQuantisedGemmKernel_CUTLASS.cu
 * @brief CUDA utility kernels and memory management for CUDAQuantisedGemmKernel
 *
 * After the NativeVNNI-only transition, this file retains:
 * - Blockwise activation quantization (FP32→INT8 per-block-of-32)
 * - Work buffer management
 * - Device memory utilities (upload, alloc, copy, free)
 * - Stream/event management
 *
 * The CUTLASS INT8 GEMM, row-wise quantization, output scaling, and
 * blockwise dp4a GEMM kernels have been removed — NativeVNNI is now
 * the sole CUDA GEMM execution path.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <algorithm>

// =========================================================================
// CUDA Error Checking Macros
// =========================================================================

#include <stdexcept>
#include <sstream>

/**
 * @brief CUDA error check that throws on failure
 *
 * CUDA errors should never be silently ignored - they indicate serious problems
 * that will cascade if execution continues. Throwing ensures:
 * 1. Immediate failure with clear error message
 * 2. Stack trace in debug builds
 * 3. No corrupted state from partial execution
 */
#define CUDA_CHECK_THROW(call)                                               \
    do                                                                       \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            std::ostringstream oss;                                          \
            oss << "[CUDAQuantGemm] CUDA error: " << cudaGetErrorString(err) \
                << " at " << __FILE__ << ":" << __LINE__;                    \
            throw std::runtime_error(oss.str());                             \
        }                                                                    \
    } while (0)

// Legacy macro - returns false (being phased out)
#define CUDA_CHECK(call)                                                           \
    do                                                                             \
    {                                                                              \
        cudaError_t err = call;                                                    \
        if (err != cudaSuccess)                                                    \
        {                                                                          \
            std::cerr << "[CUDAQuantGemm] CUDA error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";            \
            return false;                                                          \
        }                                                                          \
    } while (0)

#define CUDA_CHECK_VOID(call)                                                      \
    do                                                                             \
    {                                                                              \
        cudaError_t err = call;                                                    \
        if (err != cudaSuccess)                                                    \
        {                                                                          \
            std::cerr << "[CUDAQuantGemm] CUDA error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";            \
            return;                                                                \
        }                                                                          \
    } while (0)

// =========================================================================
// CUDA Kernels
// =========================================================================

namespace
{
    static constexpr int BLOCKWISE_BLOCK_SIZE = 32;                        // Elements per quantization block

    /**
     * @brief Quantize FP32 activations to INT8 with per-block scales
     *
     * Warp-cooperative design: each warp of 32 threads processes one
     * quantization block of 32 elements with fully coalesced access.
     * Uses warp shuffle for max reduction (no shared memory needed).
     *
     * Uses 2D grid for K-parallel execution:
     * Grid:  (grid_x, M) — multiple CUDA blocks share K-blocks of each row
     * Block: (128, 1, 1) — 4 warps per block
     *
     * Each 32-element K-block is independently quantized, so K-parallelism is trivial.
     * Critical for decode (M=1) where a 1D grid wastes 81 of 82 SMs.
     */
    __global__ void quantize_activations_blockwise_kernel(
        const float *__restrict__ A_fp32,       // [M × K]
        int8_t *__restrict__ A_int8,            // [M × K] output
        float *__restrict__ scales_A_blockwise, // [M × num_blocks] output
        int M, int K)
    {
        const int row = blockIdx.y;
        if (row >= M)
            return;

        const int num_blocks = K / BLOCKWISE_BLOCK_SIZE;
        const int lane = threadIdx.x & 31;
        const int warp_id = threadIdx.x >> 5;
        const int num_warps = blockDim.x >> 5;

        // Global warp index across all blocks in this row
        const int global_warp = blockIdx.x * num_warps + warp_id;
        const int total_warps = gridDim.x * num_warps;

        const float *row_fp32 = A_fp32 + row * K;
        int8_t *row_int8 = A_int8 + row * K;
        float *row_scales = scales_A_blockwise + row * num_blocks;

        for (int b = global_warp; b < num_blocks; b += total_warps)
        {
            const int k_start = b * BLOCKWISE_BLOCK_SIZE;

            // Coalesced load: each lane reads one element
            float val = row_fp32[k_start + lane];

            // Warp-level max_abs reduction via shuffle
            float abs_val = fabsf(val);
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1)
            {
                abs_val = fmaxf(abs_val, __shfl_xor_sync(0xFFFFFFFF, abs_val, mask));
            }

            float scale = (abs_val > 0.0f) ? (abs_val / 127.0f) : 1.0f;
            float inv_scale = 1.0f / scale;

            if (lane == 0)
                row_scales[b] = scale;

            // Quantize and coalesced write
            float qval = val * inv_scale;
            row_int8[k_start + lane] = static_cast<int8_t>(rintf(fminf(127.0f, fmaxf(-127.0f, qval))));
        }
    }

    // NOTE: blockwise_gemm_dp4a_kernel, quantize_activations_kernel, and
    // apply_scaling_kernel have been removed — NativeVNNI-only mode.

} // anonymous namespace

// =========================================================================
// Extern "C" API for .cpp adapter
// =========================================================================

extern "C"
{

    // NOTE: cudaQuantGemm_uploadWeights removed — NativeVNNI-only mode,
    // no Int8Expanded weight upload needed.

    /**
     * @brief Ensure work buffers are allocated for given M
     */
    bool cudaQuantGemm_ensureWorkBuffers(
        int8_t **d_A_int8,
        float **d_scales_A,
        int32_t **d_C_int32,
        int *work_buffer_M,
        int M, int K, int N,
        int cuda_device_id)
    {
        if (M <= *work_buffer_M)
        {
            return true; // Already have enough capacity
        }

        CUDA_CHECK(cudaSetDevice(cuda_device_id));

        // Free existing
        if (*d_A_int8)
        {
            cudaFree(*d_A_int8);
        }
        if (*d_scales_A)
        {
            cudaFree(*d_scales_A);
        }
        if (*d_C_int32)
        {
            cudaFree(*d_C_int32);
        }

        // Allocate new buffers with 2x headroom for growth
        int alloc_M = M * 2;

        CUDA_CHECK(cudaMalloc(d_A_int8, static_cast<size_t>(alloc_M) * K * sizeof(int8_t)));
        CUDA_CHECK(cudaMalloc(d_scales_A, static_cast<size_t>(alloc_M) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(d_C_int32, static_cast<size_t>(alloc_M) * N * sizeof(int32_t)));

        *work_buffer_M = alloc_M;
        return true;
    }

    // NOTE: cudaQuantGemm_execute, cudaQuantGemm_applyScaling,
    // cudaQuantGemm_quantizeActivations (row-wise), and cudaQuantGemm_blockwiseGemm
    // have been removed — NativeVNNI is now the sole CUDA GEMM execution path.
    // Only cudaQuantGemm_quantizeActivationsBlockwise is retained (used by NativeVNNI).

    /**
     * @brief Quantize FP32 activations to INT8 with per-block-of-32 scales
     */
    bool cudaQuantGemm_quantizeActivationsBlockwise(
        const float *d_A_fp32,
        int8_t *d_A_int8,
        float *d_scales_A_blockwise,
        int M, int K,
        int cuda_device_id,
        void *stream)
    {
        if (!d_A_fp32 || !d_A_int8 || !d_scales_A_blockwise)
        {
            std::ostringstream oss;
            oss << "[CUDAQuantGemm::quantizeActivationsBlockwise] Null pointer: "
                << "d_A_fp32=" << (void *)d_A_fp32
                << " d_A_int8=" << (void *)d_A_int8
                << " d_scales_A_blockwise=" << (void *)d_scales_A_blockwise;
            throw std::runtime_error(oss.str());
        }

        CUDA_CHECK_THROW(cudaSetDevice(cuda_device_id));

        // 2D grid: blockIdx.x distributes K-blocks, blockIdx.y distributes rows.
        // For decode (M=1), this ensures all SMs participate instead of just 1.
        constexpr int NUM_WARPS = 4;
        constexpr int BLOCK_SIZE = NUM_WARPS * 32; // 128 threads
        const int num_k_blocks = K / BLOCKWISE_BLOCK_SIZE;
        int grid_x = (num_k_blocks + NUM_WARPS - 1) / NUM_WARPS;
        if (grid_x > 256)
            grid_x = 256;

        dim3 grid(grid_x, M);
        dim3 block(BLOCK_SIZE);

        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        quantize_activations_blockwise_kernel<<<grid, block, 0, cuda_stream>>>(
            d_A_fp32, d_A_int8, d_scales_A_blockwise, M, K);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::ostringstream oss;
            oss << "[CUDAQuantGemm] blockwise quantize kernel launch failed: "
                << cudaGetErrorString(err)
                << " (M=" << M << ", K=" << K << ")";
            throw std::runtime_error(oss.str());
        }

        return true;
    }

    // NOTE: cudaQuantGemm_blockwiseGemm removed — NativeVNNI-only mode.

    /**
     * @brief Free device memory
     * @note Handles CUDA runtime shutdown gracefully during static destruction
     */
    void cudaQuantGemm_freeDevice(void *d_ptr)
    {
        if (d_ptr)
        {
            cudaPointerAttributes attr;
            cudaError_t pe = cudaPointerGetAttributes(&attr, d_ptr);
            cudaError_t err = cudaFree(d_ptr);
            // During static destruction at program exit, CUDA runtime may already
            // be torn down. These error codes indicate this harmless condition.
            if (err != cudaSuccess && err != cudaErrorCudartUnloading && err != cudaErrorNoDevice)
            {
                // Only log actual errors, not shutdown-related ones
                fprintf(stderr, "WARNING: cudaFree failed: %s ptr=%p attr_err=%d type=%d device=%d\n",
                        cudaGetErrorString(err), d_ptr, (int)pe, (int)attr.type, attr.device);
            }
        }
    }

    /**
     * @brief Allocate raw bytes on device and copy from host
     * @note Must be compiled by nvcc to ensure CUDA runtime context consistency
     */
    bool cudaQuantGemm_uploadRawBytes(
        const void *h_src,
        void **d_dst,
        size_t bytes,
        int cuda_device_id)
    {
        *d_dst = nullptr;
        if (bytes == 0)
        {
            return true;
        }
        CUDA_CHECK(cudaSetDevice(cuda_device_id));
        CUDA_CHECK(cudaMalloc(d_dst, bytes));
        CUDA_CHECK(cudaMemcpy(*d_dst, h_src, bytes, cudaMemcpyHostToDevice));
        return true;
    }

    /**
     * @brief Allocate float array on device
     */
    bool cudaQuantGemm_allocFloat(float **d_ptr, size_t count, int cuda_device_id)
    {
        CUDA_CHECK(cudaSetDevice(cuda_device_id));
        CUDA_CHECK(cudaMalloc(d_ptr, count * sizeof(float)));
        return true;
    }

    /**
     * @brief Copy floats from host to device
     */
    bool cudaQuantGemm_copyHostToDevice(float *d_dst, const float *h_src, size_t count, int cuda_device_id)
    {
        CUDA_CHECK(cudaSetDevice(cuda_device_id));
        CUDA_CHECK(cudaMemcpy(d_dst, h_src, count * sizeof(float), cudaMemcpyHostToDevice));
        return true;
    }

    /**
     * @brief Copy floats from device to host
     */
    bool cudaQuantGemm_copyDeviceToHost(float *h_dst, const float *d_src, size_t count, int cuda_device_id)
    {
        CUDA_CHECK(cudaSetDevice(cuda_device_id));
        CUDA_CHECK(cudaMemcpy(h_dst, d_src, count * sizeof(float), cudaMemcpyDeviceToHost));
        return true;
    }

    /**
     * @brief Copy int32 from device to host
     */
    bool cudaQuantGemm_copyInt32DeviceToHost(int32_t *h_dst, const int32_t *d_src, size_t count, int cuda_device_id)
    {
        CUDA_CHECK(cudaSetDevice(cuda_device_id));
        CUDA_CHECK(cudaMemcpy(h_dst, d_src, count * sizeof(int32_t), cudaMemcpyDeviceToHost));
        return true;
    }

    /**
     * @brief Async device-to-device float copy (for mapped output redirect)
     */
    bool cudaQuantGemm_copyDeviceToDeviceAsync(float *d_dst, const float *d_src, size_t count, int cuda_device_id, void *stream)
    {
        if (!stream)
        {
            std::cerr << "[CUDAQuantGemm] Refusing async D2D copy on the CUDA default stream; "
                      << "callers must bind an explicit stream\n";
            return false;
        }
        CUDA_CHECK(cudaSetDevice(cuda_device_id));
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        CUDA_CHECK(cudaMemcpyAsync(d_dst, d_src, count * sizeof(float),
                                   cudaMemcpyDeviceToDevice, cuda_stream));
        return true;
    }

    /**
     * @brief Set active CUDA device
     */
    bool cudaQuantGemm_setDevice(int cuda_device_id)
    {
        CUDA_CHECK(cudaSetDevice(cuda_device_id));
        return true;
    }

    /**
     * @brief Synchronize a CUDA stream (for diagnostics)
     */
    bool cudaQuantGemm_streamSync(int cuda_device_id, void *stream)
    {
        CUDA_CHECK(cudaSetDevice(cuda_device_id));
        if (stream)
        {
            CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
        }
        else
        {
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        return true;
    }

    // ── Concurrent prefill stream/event helpers ─────────────────────────────

    bool cudaQuantGemm_createStream(void **out_stream, int cuda_device_id)
    {
        CUDA_CHECK(cudaSetDevice(cuda_device_id));
        cudaStream_t s;
        CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        *out_stream = static_cast<void *>(s);
        return true;
    }

    void cudaQuantGemm_destroyStream(void *stream)
    {
        if (stream)
            cudaStreamDestroy(static_cast<cudaStream_t>(stream));
    }

    bool cudaQuantGemm_createEvent(void **out_event, int cuda_device_id)
    {
        CUDA_CHECK(cudaSetDevice(cuda_device_id));
        cudaEvent_t e;
        CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        *out_event = static_cast<void *>(e);
        return true;
    }

    void cudaQuantGemm_destroyEvent(void *event)
    {
        if (event)
            cudaEventDestroy(static_cast<cudaEvent_t>(event));
    }

    bool cudaQuantGemm_recordEvent(void *event, void *stream)
    {
        CUDA_CHECK(cudaEventRecord(
            static_cast<cudaEvent_t>(event),
            static_cast<cudaStream_t>(stream)));
        return true;
    }

    bool cudaQuantGemm_streamWaitEvent(void *stream, void *event)
    {
        CUDA_CHECK(cudaStreamWaitEvent(
            static_cast<cudaStream_t>(stream),
            static_cast<cudaEvent_t>(event), 0));
        return true;
    }

    // NOTE: repack_weights_tc_blocked_kernel and cudaQuantGemm_prepareTensorCoreBlockedWeights
    // removed — TC-blocked weight format is no longer used (NativeVNNI-only mode).

} // extern "C"
