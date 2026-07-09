/**
 * @file CUDACastKernels.cu
 * @brief FP32 ↔ FP16 conversion kernels for mixed-precision allreduce (CUDA)
 *
 * CUDA equivalent of ROCmCastKernels.hip. Converts between FP32 and FP16
 * on-device for bandwidth optimization in TP allreduce.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// FP32 → FP16 conversion kernel
// ============================================================================

static __global__ void fp32_to_fp16_kernel(const float *__restrict__ input,
                                           __half *__restrict__ output,
                                           const size_t count)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        output[idx] = __float2half(input[idx]);
    }
}

// ============================================================================
// FP16 → FP32 conversion kernel
// ============================================================================

static __global__ void fp16_to_fp32_kernel(const __half *__restrict__ input,
                                           float *__restrict__ output,
                                           const size_t count)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        output[idx] = __half2float(input[idx]);
    }
}

static __global__ void fp32_peer_add_kernel(float *__restrict__ dst,
                                            const float *__restrict__ peer,
                                            const size_t count)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        dst[idx] += peer[idx];
    }
}

// ============================================================================
// Host API — called from LocalTPContext
// ============================================================================

extern "C"
{
    cudaError_t cudaCastFP32ToFP16(const float *fp32_input, void *fp16_output,
                                   size_t count, int ordinal, cudaStream_t stream)
    {
        if (count == 0)
            return cudaSuccess;
        if (!fp32_input || !fp16_output || !stream)
            return cudaErrorInvalidValue;

        cudaError_t err = cudaSetDevice(ordinal);
        if (err != cudaSuccess)
            return err;
        (void)cudaGetLastError();

        constexpr int BLOCK_SIZE = 256;
        const int grid_size = static_cast<int>((count + BLOCK_SIZE - 1) / BLOCK_SIZE);

        fp32_to_fp16_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
            fp32_input, static_cast<__half *>(fp16_output), count);

        return cudaGetLastError();
    }

    cudaError_t cudaCastFP16ToFP32(const void *fp16_input, float *fp32_output,
                                   size_t count, int ordinal, cudaStream_t stream)
    {
        if (count == 0)
            return cudaSuccess;
        if (!fp16_input || !fp32_output || !stream)
            return cudaErrorInvalidValue;

        cudaError_t err = cudaSetDevice(ordinal);
        if (err != cudaSuccess)
            return err;
        (void)cudaGetLastError();

        constexpr int BLOCK_SIZE = 256;
        const int grid_size = static_cast<int>((count + BLOCK_SIZE - 1) / BLOCK_SIZE);

        fp16_to_fp32_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
            static_cast<const __half *>(fp16_input), fp32_output, count);

        return cudaGetLastError();
    }

    /**
     * Allocate device memory and set active device (for FP16 scratch buffers).
     * @param buf     Output pointer to allocated memory
     * @param bytes   Bytes to allocate
     * @param ordinal GPU device ordinal
     * @return 0 (cudaSuccess) on success
     */
    int cudaFP16ScratchAlloc(void **buf, size_t bytes, int ordinal)
    {
        cudaError_t err = cudaSetDevice(ordinal);
        if (err != cudaSuccess)
            return static_cast<int>(err);
        return static_cast<int>(cudaMalloc(buf, bytes));
    }

    /**
     * Free device memory on specified device.
     * @param buf     Pointer to free
     * @param ordinal GPU device ordinal
     */
    void cudaFP16ScratchFree(void *buf, int ordinal)
    {
        cudaSetDevice(ordinal);
        cudaFree(buf);
    }

    int cudaLocalTPSmallFP32AllreduceScratchAlloc(void **buf, size_t bytes, int ordinal)
    {
        if (!buf || bytes == 0)
            return static_cast<int>(cudaErrorInvalidValue);
        cudaError_t err = cudaSetDevice(ordinal);
        if (err != cudaSuccess)
            return static_cast<int>(err);
        return static_cast<int>(cudaMalloc(buf, bytes));
    }

    void cudaLocalTPSmallFP32AllreduceScratchFree(void *buf, int ordinal)
    {
        if (!buf)
            return;
        (void)cudaSetDevice(ordinal);
        (void)cudaFree(buf);
    }

    int cudaLocalTPSmallFP32AllreduceLaunch(float *dst, const float *peer,
                                            size_t count, int ordinal, void *stream)
    {
        if (!dst || !peer || !stream)
            return static_cast<int>(cudaErrorInvalidValue);
        if (count == 0)
            return static_cast<int>(cudaSuccess);
        cudaError_t err = cudaSetDevice(ordinal);
        if (err != cudaSuccess)
            return static_cast<int>(err);

        constexpr int BLOCK_SIZE = 256;
        const int grid_size = static_cast<int>((count + BLOCK_SIZE - 1) / BLOCK_SIZE);
        fp32_peer_add_kernel<<<grid_size, BLOCK_SIZE, 0, static_cast<cudaStream_t>(stream)>>>(
            dst, peer, count);
        return static_cast<int>(cudaGetLastError());
    }
}
