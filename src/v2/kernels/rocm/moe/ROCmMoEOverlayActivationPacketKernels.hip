/**
 * @file ROCmMoEOverlayActivationPacketKernels.hip
 * @brief HIP launches for device-owned sparse ExpertOverlay activation epochs.
 *
 * Every function enqueues only on the caller's exact non-default stream.
 * Metadata and compact payload kernels access one setup-owned mapped channel.
 * The adjacent mapped 64-bit timeline is the exact producer/consumer edge for
 * every byte; no host wait, default stream, or fixed-capacity DMA is involved.
 */

#include "ROCmMoEOverlayActivationPacketKernels.h"

#include <hip/hip_runtime.h>

#include "../../common/MoEOverlayActivationPacketDevice.inl"
#include "../../common/MoEOverlayNodeLocalRouteExchangeDevice.inl"

#include <algorithm>
#include <cstddef>

namespace
{
    constexpr unsigned int kThreads = 256u;
    /**
     * One-row packets carry a full hidden-width payload through mapped system
     * memory.  Eight AMD wavefronts expose enough independent PCIe operations
     * to hide that latency; ordinary packet kernels retain the smaller block
     * because their element grid already supplies the required concurrency.
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

    /** @return Whether the exact HIP device and non-default stream are usable. */
    bool selectLaunchContext(int device_ordinal, void *stream) noexcept
    {
        return device_ordinal >= 0 && stream != nullptr &&
               hipSetDevice(device_ordinal) == hipSuccess;
    }

    /** @return Whether every launch submitted so far is accepted by HIP. */
    bool launchAccepted() noexcept
    {
        return hipGetLastError() == hipSuccess;
    }
} // namespace

extern "C" bool hipMoEOverlayActivationPackDispatch(
    const llaminar2::MoEOverlayActivationDispatchPackLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto hip_stream = reinterpret_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(
        llaminar2::moe_activation_packet_device::packDispatchMetadataKernel,
        dim3(1u),
        dim3(llaminar2::moe_activation_packet_device::
                 kDispatchMetadataBlockThreads),
        0u,
        hip_stream,
        *launch);
    if (launch->hidden_payload_layout ==
        llaminar2::MoEOverlayActivationHiddenPayloadLayout::CompactRows)
    {
        const std::size_t hidden_elements =
            static_cast<std::size_t>(launch->physical_rows) *
            static_cast<std::size_t>(launch->packet.d_model);
        hipLaunchKernelGGL(
            llaminar2::moe_activation_packet_device::packDispatchHiddenKernel,
            dim3(blocksFor(hidden_elements)), dim3(kThreads), 0u, hip_stream,
            *launch);
    }
    return launchAccepted();
}

extern "C" bool hipMoEOverlayActivationConsumeDispatch(
    const llaminar2::MoEOverlayActivationDispatchConsumeLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto hip_stream = reinterpret_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(
        llaminar2::moe_activation_packet_device::validateDispatchKernel,
        dim3(1u), dim3(kThreads), 0u, hip_stream, *launch);
    const std::size_t hidden_elements =
        static_cast<std::size_t>(launch->physical_rows) *
        static_cast<std::size_t>(launch->packet.d_model);
    const std::size_t route_elements =
        static_cast<std::size_t>(launch->physical_rows) *
        static_cast<std::size_t>(launch->packet.top_k);
    const std::size_t elements = std::max(hidden_elements, route_elements);
    hipLaunchKernelGGL(
        llaminar2::moe_activation_packet_device::materializeMappedDispatchKernel,
        dim3(blocksFor(elements)), dim3(kThreads), 0u, hip_stream, *launch);
    return launchAccepted();
}

extern "C" bool hipMoEOverlayActivationPackReturn(
    const llaminar2::MoEOverlayActivationReturnPackLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto hip_stream = reinterpret_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(
        llaminar2::moe_activation_packet_device::packReturnMetadataKernel,
        dim3(1u), dim3(kThreads), 0u, hip_stream, *launch);
    const std::size_t output_elements =
        static_cast<std::size_t>(launch->physical_rows) *
        static_cast<std::size_t>(launch->dispatch.top_k) *
        static_cast<std::size_t>(launch->returned.d_model);
    hipLaunchKernelGGL(
        llaminar2::moe_activation_packet_device::packMappedReturnPayloadKernel,
        dim3(blocksFor(output_elements)), dim3(kThreads), 0u, hip_stream,
        *launch);
    return launchAccepted();
}

extern "C" bool hipMoEOverlayActivationConsumeReturn(
    const llaminar2::MoEOverlayActivationReturnConsumeLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto hip_stream = reinterpret_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(
        llaminar2::moe_activation_packet_device::validateReturnKernel,
        dim3(1u), dim3(kThreads), 0u, hip_stream, *launch);
    const std::size_t output_elements =
        static_cast<std::size_t>(launch->physical_rows) *
        static_cast<std::size_t>(launch->dispatch.top_k) *
        static_cast<std::size_t>(launch->returned.d_model);
    hipLaunchKernelGGL(
        llaminar2::moe_activation_packet_device::materializeMappedCanonicalReturnKernel,
        dim3(blocksFor(output_elements)), dim3(kThreads), 0u, hip_stream,
        *launch);
    return launchAccepted();
}

extern "C" bool hipMoEOverlayConsumeCanonicalRouteTicket(
    const llaminar2::MoEOverlayCanonicalRouteTicketConsumeLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    // Route-row tiles amortize mapped metadata; the grid strides over rows.
    const unsigned int blocks = static_cast<unsigned int>(std::min<std::size_t>(
        kSparseRoutePayloadBlocks, launch->route_capacity));
    const auto hip_stream = reinterpret_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(
        llaminar2::moe_activation_packet_device::awaitCanonicalRouteTicketKernel,
        dim3(1u), dim3(1u), 0u, hip_stream, *launch);
    hipLaunchKernelGGL(
        llaminar2::moe_activation_packet_device::
            materializeCanonicalRouteTicketKernel,
        dim3(blocks), dim3(kThreads), 0u, hip_stream, *launch);
    /* A separate one-thread node cannot race the materialization blocks. Its
     * exact-stream release is the sole authority permitting host payload reuse. */
    hipLaunchKernelGGL(
        llaminar2::moe_activation_packet_device::
            acknowledgeCanonicalRouteTicketKernel,
        dim3(1u), dim3(1u), 0u, hip_stream, *launch);
    return launchAccepted();
}

extern "C" bool hipMoEOverlayActivationPackSingleRowDispatch(
    const llaminar2::MoEOverlayActivationSingleRowDispatchPackLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto hip_stream = reinterpret_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(
        llaminar2::moe_activation_packet_device::packSingleRowDispatchKernel,
        dim3(1u), dim3(kSingleRowThreads), 0u, hip_stream, *launch);
    return launchAccepted();
}

extern "C" bool hipMoEOverlayActivationConsumeSingleRowDispatch(
    const llaminar2::MoEOverlayActivationSingleRowDispatchConsumeLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto hip_stream = reinterpret_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(
        llaminar2::moe_activation_packet_device::consumeSingleRowDispatchKernel,
        dim3(1u), dim3(kSingleRowThreads), 0u, hip_stream, *launch);
    return launchAccepted();
}

extern "C" bool hipMoEOverlayActivationPackSingleRowReturn(
    const llaminar2::MoEOverlayActivationSingleRowReturnPackLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto hip_stream = reinterpret_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(
        llaminar2::moe_activation_packet_device::packSingleRowReturnKernel,
        dim3(1u), dim3(kSingleRowThreads), 0u, hip_stream, *launch);
    return launchAccepted();
}

extern "C" bool hipMoEOverlayActivationConsumeSingleRowReturn(
    const llaminar2::MoEOverlayActivationSingleRowReturnConsumeLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto hip_stream = reinterpret_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(
        llaminar2::moe_activation_packet_device::consumeSingleRowReturnKernel,
        dim3(1u), dim3(kSingleRowThreads), 0u, hip_stream, *launch);
    return launchAccepted();
}

extern "C" bool hipMoEOverlayActivationPackSingleRowDispatchBatch(
    const llaminar2::MoEOverlayActivationSingleRowDispatchBatchLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto hip_stream = reinterpret_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(
        llaminar2::moe_activation_packet_device::
            packSingleRowDispatchBatchKernel,
        dim3(launch->lane_count), dim3(kSingleRowThreads), 0u, hip_stream,
        *launch);
    return launchAccepted();
}

extern "C" bool hipMoEOverlayActivationConsumeSingleRowReturnBatch(
    const llaminar2::MoEOverlayActivationSingleRowReturnBatchLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto hip_stream = reinterpret_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(
        llaminar2::moe_activation_packet_device::
            validateSingleRowReturnBatchKernel,
        dim3(launch->lane_count), dim3(kSingleRowThreads), 0u, hip_stream,
        *launch);
    const std::size_t route_elements =
        static_cast<std::size_t>(launch->top_k) *
        static_cast<std::size_t>(launch->d_model);
    hipLaunchKernelGGL(
        llaminar2::moe_activation_packet_device::
            materializeSingleRowCanonicalReturnBatchKernel,
        dim3(blocksFor(route_elements), launch->lane_count, 1u),
        dim3(kThreads), 0u, hip_stream, *launch);
    return launchAccepted();
}

extern "C" bool hipMoEOverlayActivationConsumeMultiRowReturnBatch(
    const llaminar2::MoEOverlayActivationMultiRowReturnBatchLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto hip_stream = reinterpret_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(
        llaminar2::moe_activation_packet_device::
            validateMultiRowReturnBatchKernel,
        dim3(launch->lane_count), dim3(kThreads), 0u, hip_stream, *launch);
    const std::size_t output_elements =
        static_cast<std::size_t>(launch->physical_rows) *
        static_cast<std::size_t>(launch->top_k) *
        static_cast<std::size_t>(launch->d_model);
    // Each lane copies its compact live entries with unique writers. Bound
    // the grid independently of capacity so sparse lanes do not schedule
    // thousands of idle workgroups; the shared loop covers every grid stride.
    const unsigned int payload_blocks = std::min(
        kSparseRoutePayloadBlocks, blocksFor(output_elements));
    hipLaunchKernelGGL(
        llaminar2::moe_activation_packet_device::
            materializeMultiRowCanonicalReturnBatchKernel,
        dim3(payload_blocks, launch->lane_count, 1u),
        dim3(kThreads), 0u, hip_stream,
        *launch);
    return launchAccepted();
}

extern "C" bool hipMoEOverlayPublishNodeLocalCanonicalRoutes(
    const llaminar2::MoENodeLocalRoutePublishLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto hip_stream = reinterpret_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(
        llaminar2::moe_node_local_route_device::beginRoutePublishKernel,
        dim3(1u), dim3(1u), 0u, hip_stream, *launch);
    const unsigned int payload_blocks = std::min(
        kSparseRoutePayloadBlocks,
        static_cast<unsigned int>(launch->live_route_slots));
    hipLaunchKernelGGL(
        llaminar2::moe_node_local_route_device::publishSparseRoutePayloadKernel,
        dim3(payload_blocks), dim3(kThreads), 0u, hip_stream, *launch);
    hipLaunchKernelGGL(
        llaminar2::moe_node_local_route_device::finishRoutePublishKernel,
        dim3(1u), dim3(1u), 0u, hip_stream, *launch);
    return launchAccepted();
}

extern "C" bool hipMoEOverlayAcquireNodeLocalCanonicalRoutes(
    const llaminar2::MoENodeLocalRouteConsumeLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto hip_stream = reinterpret_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(
        llaminar2::moe_node_local_route_device::beginRouteConsumeKernel,
        dim3(1u), dim3(1u), 0u, hip_stream, *launch);
    return launchAccepted();
}

extern "C" bool hipMoEOverlayStageNodeLocalCanonicalRoutes(
    const llaminar2::MoENodeLocalRouteConsumeLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto hip_stream = reinterpret_cast<hipStream_t>(stream);
    const std::size_t route_slots =
        static_cast<std::size_t>(launch->physical_rows) * launch->top_k;
    const unsigned int payload_blocks = std::min(
        kSparseRoutePayloadBlocks,
        static_cast<unsigned int>(route_slots));
    const dim3 stage_grid(payload_blocks, launch->peer_count);
    hipLaunchKernelGGL(
        llaminar2::moe_node_local_route_device::stageSparseRoutePayloadKernel,
        stage_grid, dim3(kThreads), 0u, hip_stream, *launch);
    return launchAccepted();
}

extern "C" bool hipMoEOverlayFoldNodeLocalCanonicalRoutes(
    const llaminar2::MoENodeLocalRouteConsumeLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto hip_stream = reinterpret_cast<hipStream_t>(stream);
    const std::size_t route_slots =
        static_cast<std::size_t>(launch->physical_rows) * launch->top_k;
    hipLaunchKernelGGL(
        llaminar2::moe_node_local_route_device::validateSparseRouteSlotsKernel,
        dim3(blocksFor(route_slots)), dim3(kThreads), 0u, hip_stream, *launch);
    const dim3 fold_grid(
        blocksFor(static_cast<std::size_t>(launch->d_model)),
        launch->physical_rows);
    hipLaunchKernelGGL(
        llaminar2::moe_node_local_route_device::foldSparseCanonicalRoutesKernel,
        fold_grid, dim3(kThreads), 0u, hip_stream, *launch);
    hipLaunchKernelGGL(
        llaminar2::moe_node_local_route_device::finishRouteConsumeKernel,
        dim3(launch->peer_count), dim3(1u), 0u, hip_stream, *launch);
    return launchAccepted();
}

extern "C" bool hipMoEOverlayBeginNodeLocalDensePublication(
    const llaminar2::MoENodeLocalDensePublicationLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() || !launch->binding.isRoot() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto hip_stream = reinterpret_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(
        llaminar2::moe_node_local_route_device::beginDensePublicationKernel,
        dim3(1u), dim3(1u), 0u, hip_stream, *launch);
    return launchAccepted();
}

extern "C" bool hipMoEOverlayFinishNodeLocalDensePublication(
    const llaminar2::MoENodeLocalDensePublicationLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() || !launch->binding.isRoot() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto hip_stream = reinterpret_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(
        llaminar2::moe_node_local_route_device::finishDensePublicationKernel,
        dim3(1u), dim3(1u), 0u, hip_stream, *launch);
    return launchAccepted();
}

extern "C" bool hipMoEOverlayBeginNodeLocalDensePublicationConsume(
    const llaminar2::MoENodeLocalDensePublicationLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() || launch->binding.isRoot() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto hip_stream = reinterpret_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(
        llaminar2::moe_node_local_route_device::beginDensePublicationConsumeKernel,
        dim3(1u), dim3(1u), 0u, hip_stream, *launch);
    return launchAccepted();
}

extern "C" bool hipMoEOverlayFinishNodeLocalDensePublicationConsume(
    const llaminar2::MoENodeLocalDensePublicationLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() || launch->binding.isRoot() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto hip_stream = reinterpret_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(
        llaminar2::moe_node_local_route_device::finishDensePublicationConsumeKernel,
        dim3(1u), dim3(1u), 0u, hip_stream, *launch);
    return launchAccepted();
}
