/**
 * @file CUDALiveLaunchProof.cu
 * @brief Observe physically submitted CTAs; no bounds guard hides an oversized launch.
 */
#include "backends/cuda/CUDADeferredKernelLaunch.h"

#if CUDART_VERSION >= 13000
namespace
{
    /** @brief Configure the exact live grid, including disablement on an empty replay. */
    __global__ void selectGrid(cudaGraphDeviceNode_t handle, const unsigned *live, unsigned *failure)
    {
        const auto count = *live;
        auto error = cudaGraphKernelNodeSetEnabled(handle, count != 0);
        if (error == cudaSuccess && count != 0)
            error = cudaGraphKernelNodeSetGridDim(handle, dim3(count));
        *failure = static_cast<unsigned>(error);
    }

    /** @brief All submitted blocks execute this observable work. */
    __global__ void countBlocks(unsigned *count) { atomicAdd(count, 1u); }
}
#endif

/** @brief Record a real launch binding into the production capture owner's frontier. */
cudaError_t recordCUDALiveLaunchProof(cudaStream_t stream, const unsigned *live, unsigned *count, unsigned *failure)
{
#if CUDART_VERSION < 13000
    (void)stream; (void)live; (void)count; (void)failure;
    return cudaErrorNotSupported;
#else
    cudaGraphDeviceNode_t unbound{};
    void *control_args[]{&unbound, &live, &failure};
    cudaKernelNodeParams control{};
    control.func = reinterpret_cast<void *>(selectGrid);
    control.gridDim = control.blockDim = dim3(1);
    control.kernelParams = control_args;
    void *work_args[]{&count};
    cudaKernelNodeParams work{};
    work.func = reinterpret_cast<void *>(countBlocks);
    work.gridDim = work.blockDim = dim3(1);
    work.kernelParams = work_args;
    return llaminar2::appendCUDADeferredKernelLaunch(stream, control, 3u, work);
#endif
}
