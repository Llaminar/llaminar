/**
 * @file ROCmMoEGroupedIntermediateExchange.hip
 * @brief HIP recording bridge for lossless grouped expert intermediate packets.
 *
 * All geometry is fixed before graph capture; ownership/grouping remains in
 * device memory and may change on every replay. Native collective stream edges,
 * not a host-visible epoch or blocking wait, publish the resulting bytes.
 */
#include <hip/hip_runtime.h>
#include <algorithm>

#include "kernels/common/MoEGroupedIntermediateExchangeKernels.h"
#include "kernels/common/MoEGroupedIntermediateExchangeDevice.inl"

namespace llaminar2::rocm
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
            return stream && stream != hipStreamLegacy && stream != hipStreamPerThread;
        }
    }

    bool packGroupedIntermediate(const MoEGroupedIntermediatePackLaunch &launch, void *stream)
    {
        if (!explicitStream(stream) || !launch.valid()) return false;
        hipLaunchKernelGGL(moe_intermediate_exchange_device::packGroupedIntermediateKernel,
            dim3(std::min(launch.layout.route_capacity, 65535u)), dim3(128), 0,
            static_cast<hipStream_t>(stream), launch, geometry(launch.layout));
        return hipGetLastError() == hipSuccess;
    }

    bool consumeGroupedIntermediate(const MoEGroupedIntermediateConsumeLaunch &launch, void *stream)
    {
        if (!explicitStream(stream) || !launch.valid()) return false;
        hipLaunchKernelGGL(moe_intermediate_exchange_device::consumeGroupedIntermediateKernel,
            dim3(std::min(launch.layout.route_capacity, 65535u)), dim3(128), 0,
            static_cast<hipStream_t>(stream), launch, geometry(launch.layout));
        return hipGetLastError() == hipSuccess;
    }

    bool packCompactIntermediate(const MoECompactIntermediatePackLaunch &launch, void *stream)
    {
        if (!explicitStream(stream) || !launch.valid()) return false;
        hipLaunchKernelGGL(moe_intermediate_exchange_device::packCompactIntermediateKernel,
            dim3(std::min(launch.payload.layout.route_capacity, 65535u)), dim3(128), 0,
            static_cast<hipStream_t>(stream), launch, geometry(launch.payload.layout));
        return hipGetLastError() == hipSuccess;
    }

    bool consumeCompactIntermediate(const MoECompactIntermediateConsumeLaunch &launch, void *stream)
    {
        if (!explicitStream(stream) || !launch.valid()) return false;
        hipLaunchKernelGGL(moe_intermediate_exchange_device::consumeCompactIntermediateKernel,
            dim3(std::min(launch.layout.route_capacity, 65535u)), dim3(128), 0,
            static_cast<hipStream_t>(stream), launch, geometry(launch.layout));
        return hipGetLastError() == hipSuccess;
    }
} // namespace llaminar2::rocm
