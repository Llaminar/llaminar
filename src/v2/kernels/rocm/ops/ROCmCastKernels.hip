/**
 * @file ROCmCastKernels.hip
 * @brief FP32 ↔ FP16 conversion kernels for mixed-precision allreduce
 *
 * These kernels convert between FP32 and FP16 on-device, used for bandwidth
 * optimization in TP allreduce. Instead of transferring FP32 data across PCIe,
 * we cast to FP16 first, allreduce in FP16, then cast back to FP32.
 *
 * For 8.5 MB allreduces on PCIe 3.0 x4 (MI50), this roughly halves transfer
 * time (~7.6ms → ~3.8ms per allreduce, saving ~213ms across 56 allreduces
 * in a 7B model forward pass).
 *
 * Conversion cost: ~0.05ms for 2M elements on MI50 — negligible vs transfer.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ============================================================================
// FP32 → FP16 conversion kernel
// ============================================================================

static __global__ void fp32_to_fp16_kernel(const float* __restrict__ input,
                                     __half* __restrict__ output,
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

static __global__ void fp16_to_fp32_kernel(const __half* __restrict__ input,
                                     float* __restrict__ output,
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
    /**
     * Cast FP32 buffer to FP16 on the specified HIP stream.
     * @param fp32_input  Source FP32 device buffer
     * @param fp16_output Destination FP16 device buffer (must have count * 2 bytes)
     * @param count       Number of elements
     * @param stream      HIP stream (must not be null)
     * @return hipSuccess on success
     */
    hipError_t rocmCastFP32ToFP16(const float *fp32_input, void *fp16_output,
                                   size_t count, hipStream_t stream)
    {
        if (count == 0) return hipSuccess;

        constexpr int BLOCK_SIZE = 256;
        const int grid_size = static_cast<int>((count + BLOCK_SIZE - 1) / BLOCK_SIZE);

        hipLaunchKernelGGL(fp32_to_fp16_kernel, dim3(grid_size), dim3(BLOCK_SIZE), 0, stream,
                           fp32_input, static_cast<__half *>(fp16_output), count);

        return hipGetLastError();
    }

    /**
     * Cast FP16 buffer to FP32 on the specified HIP stream.
     * @param fp16_input  Source FP16 device buffer
     * @param fp32_output Destination FP32 device buffer (must have count * 4 bytes)
     * @param count       Number of elements
     * @param stream      HIP stream (must not be null)
     * @return hipSuccess on success
     */
    hipError_t rocmCastFP16ToFP32(const void *fp16_input, float *fp32_output,
                                   size_t count, hipStream_t stream)
    {
        if (count == 0) return hipSuccess;

        constexpr int BLOCK_SIZE = 256;
        const int grid_size = static_cast<int>((count + BLOCK_SIZE - 1) / BLOCK_SIZE);

        hipLaunchKernelGGL(fp16_to_fp32_kernel, dim3(grid_size), dim3(BLOCK_SIZE), 0, stream,
                           static_cast<const __half *>(fp16_input), fp32_output, count);

        return hipGetLastError();
    }

    /**
     * Allocate device memory and set active device (for FP16 scratch buffers).
     * @param buf     Output pointer to allocated memory
     * @param bytes   Bytes to allocate
     * @param ordinal GPU device ordinal
     * @return hipSuccess on success
     */
    int rocmFP16ScratchAlloc(void **buf, size_t bytes, int ordinal)
    {
        hipError_t err = hipSetDevice(ordinal);
        if (err != hipSuccess) return static_cast<int>(err);
        return static_cast<int>(hipMalloc(buf, bytes));
    }

    /**
     * Free device memory on specified device.
     * @param buf     Pointer to free
     * @param ordinal GPU device ordinal
     */
    void rocmFP16ScratchFree(void *buf, int ordinal)
    {
        (void)hipSetDevice(ordinal);
        (void)hipFree(buf);
    }

    int rocmLocalTPSmallFP32AllreduceCreateEvent(void **event_out, int ordinal)
    {
        if (!event_out)
            return static_cast<int>(hipErrorInvalidValue);
        hipError_t err = hipSetDevice(ordinal);
        if (err != hipSuccess)
            return static_cast<int>(err);
        hipEvent_t event;
        err = hipEventCreateWithFlags(&event, hipEventDisableTiming);
        if (err != hipSuccess)
        {
            *event_out = nullptr;
            return static_cast<int>(err);
        }
        *event_out = static_cast<void *>(event);
        return static_cast<int>(hipSuccess);
    }

    void rocmLocalTPSmallFP32AllreduceDestroyEvent(void *event, int ordinal)
    {
        if (!event)
            return;
        (void)hipSetDevice(ordinal);
        (void)hipEventDestroy(static_cast<hipEvent_t>(event));
    }

    int rocmLocalTPSmallFP32AllreduceRecordEvent(void *event, int ordinal, void *stream)
    {
        if (!event || !stream)
            return static_cast<int>(hipErrorInvalidValue);
        hipError_t err = hipSetDevice(ordinal);
        if (err != hipSuccess)
            return static_cast<int>(err);
        err = hipEventRecord(static_cast<hipEvent_t>(event), static_cast<hipStream_t>(stream));
        return static_cast<int>(err);
    }

    int rocmLocalTPSmallFP32AllreduceWaitEvent(void *event, int ordinal, void *stream)
    {
        if (!event || !stream)
            return static_cast<int>(hipErrorInvalidValue);
        hipError_t err = hipSetDevice(ordinal);
        if (err != hipSuccess)
            return static_cast<int>(err);
        err = hipStreamWaitEvent(static_cast<hipStream_t>(stream), static_cast<hipEvent_t>(event), 0);
        return static_cast<int>(err);
    }

    int rocmLocalTPSmallFP32AllreduceCanAccessPeer(int ordinal, int peer_ordinal)
    {
        int can_access = 0;
        hipError_t err = hipDeviceCanAccessPeer(&can_access, ordinal, peer_ordinal);
        if (err != hipSuccess)
            return 0;
        return can_access != 0;
    }

    int rocmLocalTPSmallFP32AllreduceEnablePeerAccess(int ordinal, int peer_ordinal)
    {
        hipError_t err = hipSetDevice(ordinal);
        if (err != hipSuccess)
            return static_cast<int>(err);
        err = hipDeviceEnablePeerAccess(peer_ordinal, 0);
        if (err == hipErrorPeerAccessAlreadyEnabled)
            return static_cast<int>(hipSuccess);
        return static_cast<int>(err);
    }

    int rocmLocalTPSmallFP32AllreduceLaunch(float *dst, const float *peer,
                                            size_t count, int ordinal, void *stream)
    {
        if (!dst || !peer || !stream)
            return static_cast<int>(hipErrorInvalidValue);
        if (count == 0)
            return static_cast<int>(hipSuccess);
        hipError_t err = hipSetDevice(ordinal);
        if (err != hipSuccess)
            return static_cast<int>(err);

        constexpr int BLOCK_SIZE = 256;
        const int grid_size = static_cast<int>((count + BLOCK_SIZE - 1) / BLOCK_SIZE);
        hipLaunchKernelGGL(fp32_peer_add_kernel, dim3(grid_size), dim3(BLOCK_SIZE), 0,
                           static_cast<hipStream_t>(stream), dst, peer, count);
        return static_cast<int>(hipGetLastError());
    }
}
