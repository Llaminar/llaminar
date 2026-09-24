/**
 * @file CUDAQuantisedGemmKernel_CUTLASS.cu
 * @brief CUDA utility kernels and setup-time transfers for CUDAQuantisedGemmKernel
 *
 * After the NativeVNNI-only transition, this file retains:
 * - Blockwise activation quantization (FP32→INT8 per-block-of-32)
 * - Setup-time packed-weight upload through the canonical CUDA backend
 * - Device copy utilities
 * - Stream/event management
 *
 * The CUTLASS INT8 GEMM, row-wise quantization, output scaling, and
 * blockwise dp4a GEMM kernels have been removed — NativeVNNI is now
 * the sole CUDA GEMM execution path.
 * Activation scales remain row-major. Optional INT32 block sums are stored
 * block-major so a tensor-core consumer reads adjacent rows coalescently;
 * the producer writes that single representation directly, without a transpose.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <algorithm>

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"

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
     *
     * @param A_fp32 Row-major FP32 input, M by K.
     * @param A_int8 Row-major INT8 output with the same extent.
     * @param scales_A_blockwise Row-major FP32 scales, M by K/32.
     * @param sums_A_blockwise Optional block-major INT32 sums, K/32 by M.
     * @param M Positive physical row count, also the sum buffer's row pitch.
     * @param K Positive input width divisible by 32.
     */
    __global__ void quantize_activations_blockwise_kernel(
        const float *__restrict__ A_fp32,       // [M × K]
        int8_t *__restrict__ A_int8,            // [M × K] output
        float *__restrict__ scales_A_blockwise, // [M × num_blocks] output
        int32_t *__restrict__ sums_A_blockwise, // [num_blocks × M] optional quantized activation sums
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

            // Quantize, coalesced write, and optionally record the quantized
            // block sum. Asymmetric NativeVNNI prefill uses this sum for the
            // min correction instead of recomputing it for every output tile.
            float qval = val * inv_scale;
            const int32_t q = static_cast<int32_t>(rintf(fminf(127.0f, fmaxf(-127.0f, qval))));
            row_int8[k_start + lane] = static_cast<int8_t>(q);

            if (sums_A_blockwise)
            {
                int32_t sum_q = q;
#pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1)
                    sum_q += __shfl_xor_sync(0xFFFFFFFF, sum_q, mask);
                if (lane == 0)
                    // The captured producer and consumer share the same M.
                    // Adjacent consumer rows now share cache sectors instead
                    // of gathering one distant sum from each activation row.
                    sums_A_blockwise[b * M + row] = sum_q;
            }
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
            d_A_fp32, d_A_int8, d_scales_A_blockwise, nullptr, M, K);

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

    /**
     * @brief Quantize FP32 activations and also emit per-32-block INT8 sums.
     * @param d_A_fp32 Row-major M by K source.
     * @param d_A_int8 Row-major quantized destination.
     * @param d_scales_A_blockwise Row-major M by K/32 FP32 scale destination.
     * @param d_sums_A_blockwise Optional block-major K/32 by M INT32 destination.
     * @param M Captured physical row count; consumers must use this same pitch.
     * @param K Input width divisible by 32.
     * @param cuda_device_id Owning CUDA device ordinal.
     * @param stream Exact caller-owned execution stream.
     * @return True after successfully submitting the quantization kernel.
     */
    bool cudaQuantGemm_quantizeActivationsBlockwiseWithSums(
        const float *d_A_fp32,
        int8_t *d_A_int8,
        float *d_scales_A_blockwise,
        int32_t *d_sums_A_blockwise,
        int M, int K,
        int cuda_device_id,
        void *stream)
    {
        if (!d_sums_A_blockwise)
        {
            return cudaQuantGemm_quantizeActivationsBlockwise(
                d_A_fp32, d_A_int8, d_scales_A_blockwise,
                M, K, cuda_device_id, stream);
        }
        if (!d_A_fp32 || !d_A_int8 || !d_scales_A_blockwise)
        {
            std::ostringstream oss;
            oss << "[CUDAQuantGemm::quantizeActivationsBlockwiseWithSums] Null pointer: "
                << "d_A_fp32=" << (void *)d_A_fp32
                << " d_A_int8=" << (void *)d_A_int8
                << " d_scales_A_blockwise=" << (void *)d_scales_A_blockwise
                << " d_sums_A_blockwise=" << (void *)d_sums_A_blockwise;
            throw std::runtime_error(oss.str());
        }
        if ((K % BLOCKWISE_BLOCK_SIZE) != 0)
        {
            std::ostringstream oss;
            oss << "[CUDAQuantGemm::quantizeActivationsBlockwiseWithSums] K=" << K
                << " is not divisible by " << BLOCKWISE_BLOCK_SIZE;
            throw std::runtime_error(oss.str());
        }

        CUDA_CHECK_THROW(cudaSetDevice(cuda_device_id));

        constexpr int NUM_WARPS = 4;
        constexpr int BLOCK_SIZE = NUM_WARPS * 32;
        const int num_k_blocks = K / BLOCKWISE_BLOCK_SIZE;
        int grid_x = (num_k_blocks + NUM_WARPS - 1) / NUM_WARPS;
        if (grid_x > 256)
            grid_x = 256;

        dim3 grid(grid_x, M);
        dim3 block(BLOCK_SIZE);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        quantize_activations_blockwise_kernel<<<grid, block, 0, cuda_stream>>>(
            d_A_fp32, d_A_int8, d_scales_A_blockwise, d_sums_A_blockwise, M, K);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::ostringstream oss;
            oss << "[CUDAQuantGemm] blockwise quantize-with-sums kernel launch failed: "
                << cudaGetErrorString(err)
                << " (M=" << M << ", K=" << K << ")";
            throw std::runtime_error(oss.str());
        }

        return true;
    }

    // NOTE: cudaQuantGemm_blockwiseGemm removed — NativeVNNI-only mode.

    /**
     * @brief Release setup-owned packed-weight storage through the CUDA backend.
     *
     * The explicit device ordinal is part of the ownership contract.  Teardown
     * must not infer ownership from whichever CUDA context happens to be active.
     */
    void cudaQuantGemm_freeDevice(void *d_ptr, int cuda_device_id)
    {
        if (!d_ptr)
            return;

        auto *backend = llaminar2::getCUDABackend();
        if (!backend)
            throw std::runtime_error(
                "[CUDAQuantGemm] CUDA backend unavailable while releasing packed weights");
        backend->free(d_ptr, cuda_device_id);
    }

    /**
     * @brief Allocate and upload setup-owned packed weights through the backend.
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
        auto *backend = llaminar2::getCUDABackend();
        if (!backend)
            return false;

        *d_dst = backend->allocate(bytes, cuda_device_id);
        if (!*d_dst)
            return false;
        void *const setup_stream =
            llaminar2::GPUDeviceContextPool::instance()
                .getNvidiaContext(cuda_device_id)
                .defaultStream();
        if (!setup_stream)
        {
            backend->free(*d_dst, cuda_device_id);
            *d_dst = nullptr;
            throw std::runtime_error(
                "[CUDAQuantGemm] Packed-weight upload has no explicit setup stream");
        }
        if (!backend->hostToDevice(
                *d_dst,
                h_src,
                bytes,
                cuda_device_id,
                setup_stream))
        {
            backend->free(*d_dst, cuda_device_id);
            *d_dst = nullptr;
            return false;
        }
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
