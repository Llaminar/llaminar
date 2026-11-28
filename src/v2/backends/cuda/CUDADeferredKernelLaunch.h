/**
 * @file CUDADeferredKernelLaunch.h
 * @brief Record a grid-update producer and its kernel as an importable CUDA definition.
 *
 * Neither node is updatable in the source. The final executable binds a fresh
 * handle, ensuring imported copies never share a mutable CUDA update target.
 */
#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace llaminar2
{
    /**
     * @brief Insert one unbound grid control/work pair at the exact capture frontier.
     * @param stream Exact stream participating in an owned native capture.
     * @param control Full-completion predecessor; argument zero must be a zero cudaGraphDeviceNode_t.
     * @param control_argument_count Exact count of the argument array, including the handle.
     * @param work Launch parameters with a nonzero definition grid; control replaces it before every launch.
     * @return CUDA's construction status. No out-of-capture launch is permitted.
     */
    [[nodiscard]] cudaError_t appendCUDADeferredKernelLaunch(cudaStream_t stream,
        const cudaKernelNodeParams &control, std::size_t control_argument_count,
        const cudaKernelNodeParams &work);
}
