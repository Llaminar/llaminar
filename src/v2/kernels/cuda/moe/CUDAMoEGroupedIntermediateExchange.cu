/**
 * @file CUDAMoEGroupedIntermediateExchange.cu
 * @brief CUDA recording bridge for lossless grouped expert intermediate packets.
 *
 * The shared kernels copy only existing representation bits. This bridge binds
 * the exact producer/consumer stream without allocating, reading device routing
 * state, or inserting synchronization into a captured transaction.
 */
#include <cuda_runtime.h>
#include <algorithm>

#include "kernels/common/MoEGroupedIntermediateExchangeKernels.h"
#include "kernels/common/MoEGroupedIntermediateExchangeDevice.inl"

namespace llaminar2::cuda
{
    namespace
    {
        /** @return Fixed offsets, computed once while recording the operation. */
        moe_intermediate_exchange_device::Geometry geometry(const MoEGroupedIntermediateLayout &layout)
        {
            return {layout.valueWords(), layout.scaleWords(), layout.routeWords(),
                layout.packetBytes() / sizeof(std::uint32_t)};
        }

        /** @return Whether the caller named one exact explicit stream. */
        bool explicitStream(void *stream)
        {
            return stream && stream != cudaStreamLegacy && stream != cudaStreamPerThread;
        }
    }

    bool packGroupedIntermediate(const MoEGroupedIntermediatePackLaunch &launch, void *stream)
    {
        if (!explicitStream(stream) || !launch.valid()) return false;
        moe_intermediate_exchange_device::packGroupedIntermediateKernel
            <<<std::min(launch.layout.route_capacity, 65535u), 128, 0,
                static_cast<cudaStream_t>(stream)>>>(launch, geometry(launch.layout));
        return cudaGetLastError() == cudaSuccess;
    }

    bool consumeGroupedIntermediate(const MoEGroupedIntermediateConsumeLaunch &launch, void *stream)
    {
        if (!explicitStream(stream) || !launch.valid()) return false;
        moe_intermediate_exchange_device::consumeGroupedIntermediateKernel
            <<<std::min(launch.layout.route_capacity, 65535u), 128, 0,
                static_cast<cudaStream_t>(stream)>>>(launch, geometry(launch.layout));
        return cudaGetLastError() == cudaSuccess;
    }

    bool packCompactIntermediate(const MoECompactIntermediatePackLaunch &launch, void *stream)
    {
        if (!explicitStream(stream) || !launch.valid()) return false;
        moe_intermediate_exchange_device::packCompactIntermediateKernel
            <<<std::min(launch.payload.layout.route_capacity, 65535u), 128, 0,
                static_cast<cudaStream_t>(stream)>>>(launch, geometry(launch.payload.layout));
        return cudaGetLastError() == cudaSuccess;
    }

    bool consumeCompactIntermediate(const MoECompactIntermediateConsumeLaunch &launch, void *stream)
    {
        if (!explicitStream(stream) || !launch.valid()) return false;
        moe_intermediate_exchange_device::consumeCompactIntermediateKernel
            <<<std::min(launch.layout.route_capacity, 65535u), 128, 0,
                static_cast<cudaStream_t>(stream)>>>(launch, geometry(launch.layout));
        return cudaGetLastError() == cudaSuccess;
    }
} // namespace llaminar2::cuda
