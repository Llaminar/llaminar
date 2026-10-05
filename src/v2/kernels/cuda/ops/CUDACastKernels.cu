/**
 * @file CUDACastKernels.cu
 * @brief FP32 ↔ FP16 conversion kernels for mixed-precision allreduce (CUDA)
 *
 * CUDA equivalent of ROCmCastKernels.hip. Converts between FP32 and FP16
 * on-device for bandwidth optimization in TP allreduce.
 * Captured device-prefix conversions leave every inactive FP32 value unchanged;
 * their borrowed row authority is consumed on the exact collective stream.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "kernels/common/DeviceRowRange.h"
#include <limits>

// ============================================================================
// FP32 → FP16 conversion kernel
// ============================================================================

/**
 * @brief Convert a fixed FP32 to FP16 bank using the established per-element rounding.
 * @param input Ordered admitted source bank.
 * @param output Disjoint persistent destination bank.
 * @param count Exact immutable scalar extent for the fixed operand contract.
 */
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

/**
 * @brief Convert a fixed FP16 to FP32 bank using the established per-element rounding.
 * @param input Ordered admitted source bank.
 * @param output Disjoint persistent destination bank.
 * @param count Exact immutable scalar extent for the fixed operand contract.
 */
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

// ============================================================================
// Device-owned live-prefix conversions
// ============================================================================

/**
 * @brief Convert only a device-owned live prefix using the fixed FP16 rounding.
 * @param input Admitted FP32 bank, ordered before this exact-stream consumer.
 * @param output Disjoint persistent FP16 collective scratch.
 * @param rows Captured whole-prefix geometry and borrowed device count.
 * @param columns Exact scalar width of one logical row.
 * @note Inactive inputs are neither read nor converted; allocation capacity is
 *       launch geometry only. The existing conversion arithmetic is unchanged.
 */
static __global__ void fp32_to_fp16_live_rows_kernel(
    const float *__restrict__ input, __half *__restrict__ output,
    llaminar2::DeviceRowRange rows, size_t columns)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t live_elements = static_cast<size_t>(rows.activeRows()) * columns;
    if (index < live_elements) output[index] = __float2half(input[index]);
}

/**
 * @brief Publish converted collective values without touching inactive FP32 rows.
 * @param input Ordered native FP16 result in admitted persistent scratch.
 * @param output Original FP32 bank; its inactive tail remains byte-identical.
 * @param rows Same immutable borrowed prefix used by conversion and transport.
 * @param columns Exact scalar row width.
 */
static __global__ void fp16_to_fp32_live_rows_kernel(
    const __half *__restrict__ input, float *__restrict__ output,
    llaminar2::DeviceRowRange rows, size_t columns)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t live_elements = static_cast<size_t>(rows.activeRows()) * columns;
    if (index < live_elements) output[index] = __half2float(input[index]);
}

/**
 * @brief Validate retained prefix geometry before recording a fixed-size launch.
 * @param rows Whole-prefix descriptor; sliced origins are not representable here.
 * @param columns Positive exact scalar width, independent of allocation capacity.
 * @return Positive 256-thread grid width, or zero for invalid/overflowing geometry.
 */
static int live_cast_grid(llaminar2::DeviceRowRange rows, size_t columns)
{
    constexpr size_t threads = 256;
    if (rows.physicalRows() != rows.capacity() || columns == 0 ||
        columns > std::numeric_limits<size_t>::max() / static_cast<size_t>(rows.capacity()))
        return 0;
    const size_t elements = columns * static_cast<size_t>(rows.capacity());
    const size_t blocks = 1 + (elements - 1) / threads;
    return blocks <= static_cast<size_t>(std::numeric_limits<int>::max())
        ? static_cast<int>(blocks) : 0;
}

// ============================================================================
// Host API — called from LocalTPContext
// ============================================================================

extern "C"
{
    /**
     * @brief Enqueue fixed-bank FP32ToFP16 conversion without changing arithmetic.
     * @param fp32_input Stable admitted source bank on the selected device.
     * @param fp16_output Disjoint persistent destination bank.
     * @param count Exact fixed scalar extent, not a device-owned prefix capacity.
     * @param ordinal Physical participant owning the stream and both banks.
     * @param stream Non-null ordered stream.
     * @return Native enqueue error; this fixed contract performs no count readback.
     */
    cudaError_t cudaCastFP32ToFP16(const float *fp32_input, void *fp16_output,
                                   size_t count, int ordinal, cudaStream_t stream)
    {
        if (!fp32_input || !fp16_output || !stream)
            return cudaErrorInvalidValue;
        if (count == 0)
            return cudaSuccess;

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

    /**
     * @brief Enqueue fixed-bank FP16ToFP32 conversion without changing arithmetic.
     * @param fp16_input Stable admitted source bank on the selected device.
     * @param fp32_output Disjoint persistent destination bank.
     * @param count Exact fixed scalar extent, not a device-owned prefix capacity.
     * @param ordinal Physical participant owning the stream and both banks.
     * @param stream Non-null ordered stream.
     * @return Native enqueue error; this fixed contract performs no count readback.
     */
    cudaError_t cudaCastFP16ToFP32(const void *fp16_input, float *fp32_output,
                                   size_t count, int ordinal, cudaStream_t stream)
    {
        if (!fp16_input || !fp32_output || !stream)
            return cudaErrorInvalidValue;
        if (count == 0)
            return cudaSuccess;

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
     * @brief Enqueue FP32ToFP16 conversion over the collective's exact live prefix.
     * @param fp32_input Stable admitted source bank on the selected participant.
     * @param fp16_output Disjoint admitted destination; inactive elements are untouched.
     * @param rows Complete captured row geometry and canonical device count owner.
     * @param columns Positive scalar width of each row.
     * @param ordinal Exact physical device owning both banks and the stream.
     * @param stream Non-null ordered producer/consumer stream.
     * @return Native enqueue error; no allocation, count readback or synchronization.
     */
    cudaError_t cudaCastLiveFP32ToFP16(const float *fp32_input, void *fp16_output,
        llaminar2::DeviceRowRange rows, size_t columns, int ordinal, cudaStream_t stream)
    {
        if (!fp32_input || !fp16_output || !stream || ordinal < 0)
            return cudaErrorInvalidValue;
        const int grid_size = live_cast_grid(rows, columns);
        if (grid_size == 0) return cudaErrorInvalidValue;
        const cudaError_t bound = cudaSetDevice(ordinal);
        if (bound != cudaSuccess) return bound;
        (void)cudaGetLastError();
        fp32_to_fp16_live_rows_kernel<<<grid_size, 256, 0, stream>>>(
            fp32_input, static_cast<__half *>(fp16_output), rows, columns);
        return cudaGetLastError();
    }

    /**
     * @brief Enqueue FP16ToFP32 conversion over the collective's exact live prefix.
     * @param fp16_input Stable admitted source bank on the selected participant.
     * @param fp32_output Disjoint admitted destination; inactive elements are untouched.
     * @param rows Complete captured row geometry and canonical device count owner.
     * @param columns Positive scalar width of each row.
     * @param ordinal Exact physical device owning both banks and the stream.
     * @param stream Non-null ordered producer/consumer stream.
     * @return Native enqueue error; no allocation, count readback or synchronization.
     */
    cudaError_t cudaCastLiveFP16ToFP32(const void *fp16_input, float *fp32_output,
        llaminar2::DeviceRowRange rows, size_t columns, int ordinal, cudaStream_t stream)
    {
        if (!fp16_input || !fp32_output || !stream || ordinal < 0)
            return cudaErrorInvalidValue;
        const int grid_size = live_cast_grid(rows, columns);
        if (grid_size == 0) return cudaErrorInvalidValue;
        const cudaError_t bound = cudaSetDevice(ordinal);
        if (bound != cudaSuccess) return bound;
        (void)cudaGetLastError();
        fp16_to_fp32_live_rows_kernel<<<grid_size, 256, 0, stream>>>(
            static_cast<const __half *>(fp16_input), fp32_output, rows, columns);
        return cudaGetLastError();
    }

}
