/**
 * @file CUDAMoEOverlayActivationPacketKernels.cu
 * @brief CUDA launches for device-owned sparse ExpertOverlay activation epochs.
 *
 * Every function enqueues only on the caller's exact non-default stream.
 * Metadata and compact payload kernels access one setup-owned mapped channel.
 * The adjacent mapped 64-bit timeline is the exact producer/consumer edge for
 * every byte; no host wait, default stream, or fixed-capacity DMA is involved.
 */

#include "CUDAMoEOverlayActivationPacketKernels.h"

#include <cuda_runtime.h>

#include "../../common/MoEOverlayActivationPacketDevice.inl"
#include "../../common/MoEOverlayNodeLocalRouteExchangeDevice.inl"

#include <algorithm>
#include <cstddef>

namespace
{
    constexpr unsigned int kThreads = 256u;
    /**
     * One-row packets carry a full hidden-width payload through mapped system
     * memory.  Extra warps expose enough independent PCIe reads/writes to hide
     * system-memory latency; ordinary device-resident packet kernels retain the
     * smaller block because they scale through their grid instead.
     */
    constexpr unsigned int kSingleRowThreads = 512u;
    /** Bounded route-slot blocks; each block walks additional slots grid-stride. */
    constexpr unsigned int kSparseRoutePayloadBlocks = 256u;

    /** @return Number of blocks needed for one positive element count. */
    unsigned int blocksFor(std::size_t elements) noexcept
    {
        return static_cast<unsigned int>(
            (elements + kThreads - 1u) / kThreads);
    }

    /** @return Whether the exact CUDA device and non-default stream are usable. */
    bool selectLaunchContext(int device_ordinal, void *stream) noexcept
    {
        return device_ordinal >= 0 && stream != nullptr &&
               cudaSetDevice(device_ordinal) == cudaSuccess;
    }

    /** @return Whether every launch submitted so far is accepted by CUDA. */
    bool launchAccepted() noexcept
    {
        return cudaGetLastError() == cudaSuccess;
    }
} // namespace

extern "C" bool cudaMoEOverlayActivationPackDispatch(
    const llaminar2::MoEOverlayActivationDispatchPackLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    llaminar2::moe_activation_packet_device::packDispatchMetadataKernel
        <<<1u,
           llaminar2::moe_activation_packet_device::
               kDispatchMetadataBlockThreads,
           0u,
           cuda_stream>>>(*launch);
    if (launch->hidden_payload_layout ==
        llaminar2::MoEOverlayActivationHiddenPayloadLayout::CompactRows)
    {
        const std::size_t hidden_elements =
            static_cast<std::size_t>(launch->physical_rows) *
            static_cast<std::size_t>(launch->packet.d_model);
        llaminar2::moe_activation_packet_device::packDispatchHiddenKernel
            <<<blocksFor(hidden_elements), kThreads, 0u, cuda_stream>>>(*launch);
    }
    return launchAccepted();
}

extern "C" bool cudaMoEOverlayActivationConsumeDispatch(
    const llaminar2::MoEOverlayActivationDispatchConsumeLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    llaminar2::moe_activation_packet_device::validateDispatchKernel
        <<<1u, kThreads, 0u, cuda_stream>>>(*launch);
    const std::size_t hidden_elements =
        static_cast<std::size_t>(launch->physical_rows) *
        static_cast<std::size_t>(launch->packet.d_model);
    const std::size_t route_elements =
        static_cast<std::size_t>(launch->physical_rows) *
        static_cast<std::size_t>(launch->packet.top_k);
    const std::size_t elements = std::max(hidden_elements, route_elements);
    llaminar2::moe_activation_packet_device::materializeMappedDispatchKernel
        <<<blocksFor(elements), kThreads, 0u, cuda_stream>>>(*launch);
    return launchAccepted();
}

extern "C" bool cudaMoEOverlayActivationPackReturn(
    const llaminar2::MoEOverlayActivationReturnPackLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    llaminar2::moe_activation_packet_device::packReturnMetadataKernel
        <<<1u, kThreads, 0u, cuda_stream>>>(*launch);
    const std::size_t output_elements =
        static_cast<std::size_t>(launch->physical_rows) *
        static_cast<std::size_t>(launch->returned.d_model);
    llaminar2::moe_activation_packet_device::packMappedReturnPayloadKernel
        <<<blocksFor(output_elements), kThreads, 0u, cuda_stream>>>(*launch);
    return launchAccepted();
}

extern "C" bool cudaMoEOverlayActivationConsumeReturn(
    const llaminar2::MoEOverlayActivationReturnConsumeLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    llaminar2::moe_activation_packet_device::validateReturnKernel
        <<<1u, kThreads, 0u, cuda_stream>>>(*launch);
    const std::size_t output_elements =
        static_cast<std::size_t>(launch->physical_rows) *
        static_cast<std::size_t>(launch->returned.d_model);
    llaminar2::moe_activation_packet_device::accumulateMappedReturnKernel
        <<<blocksFor(output_elements), kThreads, 0u, cuda_stream>>>(*launch);
    return launchAccepted();
}

extern "C" bool cudaMoEOverlayActivationPackSingleRowDispatch(
    const llaminar2::MoEOverlayActivationSingleRowDispatchPackLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    llaminar2::moe_activation_packet_device::packSingleRowDispatchKernel
        <<<1u, kSingleRowThreads, 0u, cuda_stream>>>(*launch);
    return launchAccepted();
}

extern "C" bool cudaMoEOverlayActivationConsumeSingleRowDispatch(
    const llaminar2::MoEOverlayActivationSingleRowDispatchConsumeLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    llaminar2::moe_activation_packet_device::consumeSingleRowDispatchKernel
        <<<1u, kSingleRowThreads, 0u, cuda_stream>>>(*launch);
    return launchAccepted();
}

extern "C" bool cudaMoEOverlayActivationPackSingleRowReturn(
    const llaminar2::MoEOverlayActivationSingleRowReturnPackLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    llaminar2::moe_activation_packet_device::packSingleRowReturnKernel
        <<<1u, kSingleRowThreads, 0u, cuda_stream>>>(*launch);
    return launchAccepted();
}

extern "C" bool cudaMoEOverlayActivationConsumeSingleRowReturn(
    const llaminar2::MoEOverlayActivationSingleRowReturnConsumeLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    llaminar2::moe_activation_packet_device::consumeSingleRowReturnKernel
        <<<1u, kSingleRowThreads, 0u, cuda_stream>>>(*launch);
    return launchAccepted();
}

extern "C" bool cudaMoEOverlayActivationPackSingleRowDispatchBatch(
    const llaminar2::MoEOverlayActivationSingleRowDispatchBatchLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    llaminar2::moe_activation_packet_device::packSingleRowDispatchBatchKernel
        <<<launch->lane_count, kSingleRowThreads, 0u, cuda_stream>>>(*launch);
    return launchAccepted();
}

extern "C" bool cudaMoEOverlayActivationConsumeSingleRowReturnBatch(
    const llaminar2::MoEOverlayActivationSingleRowReturnBatchLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    llaminar2::moe_activation_packet_device::gatherSingleRowReturnBatchKernel
        <<<launch->lane_count, kSingleRowThreads, 0u, cuda_stream>>>(*launch);
    llaminar2::moe_activation_packet_device::foldSingleRowReturnBatchKernel
        <<<blocksFor(static_cast<std::size_t>(launch->d_model)),
           kThreads, 0u, cuda_stream>>>(*launch);
    return launchAccepted();
}

extern "C" bool cudaMoEOverlayActivationConsumeMultiRowReturnBatch(
    const llaminar2::MoEOverlayActivationMultiRowReturnBatchLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    llaminar2::moe_activation_packet_device::
        validateMultiRowReturnBatchKernel
        <<<launch->lane_count, kThreads, 0u, cuda_stream>>>(*launch);
    const std::size_t output_elements =
        static_cast<std::size_t>(launch->physical_rows) *
        static_cast<std::size_t>(launch->d_model);
    llaminar2::moe_activation_packet_device::foldMultiRowReturnBatchKernel
        <<<blocksFor(output_elements), kThreads, 0u, cuda_stream>>>(*launch);
    return launchAccepted();
}

extern "C" bool cudaMoEOverlayPublishNodeLocalCanonicalRoutes(
    const llaminar2::MoENodeLocalRoutePublishLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    llaminar2::moe_node_local_route_device::beginRoutePublishKernel
        <<<1u, 1u, 0u, cuda_stream>>>(*launch);
    const unsigned int payload_blocks = std::min(
        kSparseRoutePayloadBlocks,
        static_cast<unsigned int>(launch->live_route_slots));
    llaminar2::moe_node_local_route_device::publishSparseRoutePayloadKernel
        <<<payload_blocks, kThreads, 0u, cuda_stream>>>(*launch);
    llaminar2::moe_node_local_route_device::finishRoutePublishKernel
        <<<1u, 1u, 0u, cuda_stream>>>(*launch);
    return launchAccepted();
}

extern "C" bool cudaMoEOverlayAcquireNodeLocalCanonicalRoutes(
    const llaminar2::MoENodeLocalRouteConsumeLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    llaminar2::moe_node_local_route_device::beginRouteConsumeKernel
        <<<1u, 1u, 0u, cuda_stream>>>(*launch);
    return launchAccepted();
}

extern "C" bool cudaMoEOverlayStageNodeLocalCanonicalRoutes(
    const llaminar2::MoENodeLocalRouteConsumeLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const std::size_t route_slots =
        static_cast<std::size_t>(launch->physical_rows) * launch->top_k;
    const unsigned int payload_blocks = std::min(
        kSparseRoutePayloadBlocks,
        static_cast<unsigned int>(route_slots));
    const dim3 stage_grid(payload_blocks, launch->peer_count);
    llaminar2::moe_node_local_route_device::stageSparseRoutePayloadKernel
        <<<stage_grid, kThreads, 0u, cuda_stream>>>(*launch);
    return launchAccepted();
}

extern "C" bool cudaMoEOverlayFoldNodeLocalCanonicalRoutes(
    const llaminar2::MoENodeLocalRouteConsumeLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const std::size_t route_slots =
        static_cast<std::size_t>(launch->physical_rows) * launch->top_k;
    llaminar2::moe_node_local_route_device::validateSparseRouteSlotsKernel
        <<<blocksFor(route_slots), kThreads, 0u, cuda_stream>>>(*launch);
    const dim3 fold_grid(
        blocksFor(static_cast<std::size_t>(launch->d_model)),
        launch->physical_rows);
    llaminar2::moe_node_local_route_device::foldSparseCanonicalRoutesKernel
        <<<fold_grid, kThreads, 0u, cuda_stream>>>(*launch);
    llaminar2::moe_node_local_route_device::finishRouteConsumeKernel
        <<<launch->peer_count, 1u, 0u, cuda_stream>>>(*launch);
    return launchAccepted();
}
