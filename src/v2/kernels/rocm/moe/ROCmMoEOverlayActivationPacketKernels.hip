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
#include "../../common/MoEOverlayNativeInvocationDevice.inl"
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
    constexpr unsigned int kSingleRowThreads =
        llaminar2::moe_activation_packet_device::kSingleRowDispatchThreads;
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
        hipLaunchKernelGGL(
            llaminar2::moe_activation_packet_device::packDispatchHiddenKernel,
            dim3(static_cast<unsigned int>(launch->physical_rows)),
            dim3(kThreads), 0u, hip_stream, *launch);
    }
    if (launch->shared_row_writer)
        hipLaunchKernelGGL(llaminar2::moe_activation_packet_device::packSharedDispatchUnionKernel,
            dim3(launch->physical_rows),dim3(kThreads),0u,hip_stream,*launch);
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
    const std::size_t route_elements =
        static_cast<std::size_t>(launch->physical_rows) *
        static_cast<std::size_t>(launch->packet.top_k);
    const unsigned int dispatch_blocks = std::max(
        static_cast<unsigned int>(launch->physical_rows),
        blocksFor(route_elements));
    hipLaunchKernelGGL(
        llaminar2::moe_activation_packet_device::materializeMappedDispatchKernel,
        dim3(dispatch_blocks), dim3(kThreads), 0u, hip_stream, *launch);
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
    // Each block owns a route, amortizing mapped metadata across its width.
    const auto payload_blocks = static_cast<unsigned int>(std::min<std::size_t>(
        kSparseRoutePayloadBlocks, launch->returned.route_slot_capacity));
    hipLaunchKernelGGL(
        llaminar2::moe_activation_packet_device::packMappedReturnPayloadKernel,
        dim3(payload_blocks), dim3(kThreads), 0u, hip_stream,
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
    // One block owns a row; payload width no longer multiplies the grid.
    const unsigned int payload_blocks = static_cast<unsigned int>(std::min<std::size_t>(
        kSparseRoutePayloadBlocks, launch->returned.route_slot_capacity));
    hipLaunchKernelGGL(
        llaminar2::moe_activation_packet_device::materializeMappedCanonicalReturnKernel,
        dim3(payload_blocks), dim3(kThreads), 0u, hip_stream,
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
    if (!launch->metadataOnly() && launch->route_capacity <= 32u)
    {
        hipLaunchKernelGGL(
            llaminar2::moe_activation_packet_device::
                materializeAndAcknowledgeSingleRowCanonicalRouteTicketKernel,
            dim3(1u), dim3(512u), 0u, hip_stream, *launch);
        return launchAccepted();
    }
    if (!launch->metadataOnly())
    {
        hipLaunchKernelGGL(
            llaminar2::moe_activation_packet_device::
                materializeCanonicalRouteTicketKernel,
            dim3(blocks), dim3(kThreads), 0u, hip_stream, *launch);
    }
    /* A separate wavefront-coalesced node cannot race multi-row materialization blocks. */
    hipLaunchKernelGGL(
        llaminar2::moe_activation_packet_device::
            acknowledgeCanonicalRouteTicketKernel,
        dim3(1u), dim3(64u), 0u, hip_stream, *launch);
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
    const unsigned int blocks =
        (launch->shared_hidden_rows_fp32 != nullptr &&
         launch->lane_count > 1u && launch->lane_count <= 32u)
            ? 1u
            : launch->lane_count;
    hipLaunchKernelGGL(
        llaminar2::moe_activation_packet_device::
            packSingleRowDispatchBatchKernel,
        dim3(blocks), dim3(kSingleRowThreads), 0u, hip_stream,
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
    // Single-row canonical materialization is fused into each lane's wait/validate
    // block so fast followers copy over PCIe while slower followers still run.
    hipLaunchKernelGGL(
        llaminar2::moe_activation_packet_device::
            validateSingleRowReturnBatchKernel,
        dim3(launch->lane_count), dim3(kSingleRowThreads), 0u, hip_stream,
        *launch);
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
    const std::size_t route_capacity =
        static_cast<std::size_t>(launch->physical_rows) *
        static_cast<std::size_t>(launch->top_k);
    // Each lane copies its compact live entries with unique writers. Bound
    // the grid independently of capacity so sparse lanes do not schedule
    // thousands of idle workgroups; the shared loop covers every grid stride.
    const unsigned int payload_blocks = static_cast<unsigned int>(std::min<std::size_t>(
        kSparseRoutePayloadBlocks, route_capacity));
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
    if (launch->live_route_slots <= 32u)
    {
        hipLaunchKernelGGL(
            llaminar2::moe_node_local_route_device::publishSingleRowSparseRouteKernel,
            dim3(1u), dim3(512u), 0u, hip_stream, *launch);
        return launchAccepted();
    }
    hipLaunchKernelGGL(
        llaminar2::moe_node_local_route_device::beginRoutePublishKernel,
        dim3(1u), dim3(64u), 0u, hip_stream, *launch);
    const unsigned int payload_blocks = std::min(
        kSparseRoutePayloadBlocks,
        static_cast<unsigned int>(launch->live_route_slots));
    hipLaunchKernelGGL(
        llaminar2::moe_node_local_route_device::publishSparseRoutePayloadKernel,
        dim3(payload_blocks), dim3(kThreads), 0u, hip_stream, *launch);
    hipLaunchKernelGGL(
        llaminar2::moe_node_local_route_device::finishRoutePublishKernel,
        dim3(1u), dim3(64u), 0u, hip_stream, *launch);
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
        dim3(1u), dim3(64u), 0u, hip_stream, *launch);
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

extern "C" bool hipMoEOverlayConsumeSingleRowNodeLocalCanonicalRoutes(
    const llaminar2::MoENodeLocalRouteConsumeLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() || launch->physical_rows != 1u ||
        launch->top_k > 32u || launch->peer_count > 8u ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto hip_stream = reinterpret_cast<hipStream_t>(stream);
    const unsigned vecs = (launch->d_model + 3u) / 4u;
    const unsigned rounded = ((vecs + 63u) / 64u) * 64u;
    const unsigned threads = rounded < 256u ? 256u : (rounded > 1024u ? 1024u : rounded);
    hipLaunchKernelGGL(
        llaminar2::moe_node_local_route_device::consumeSingleRowSparseRoutesKernel,
        dim3(1u), dim3(threads), 0u, hip_stream, *launch);
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


/** @brief Submit the original-slot begin without a host state observation. */
extern "C" bool hipMoEOverlayOriginalNativeBegin(
    const llaminar2::MoEOverlayOriginalNativeBeginLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        reinterpret_cast<std::uintptr_t>(stream) <= 2u ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    hipLaunchKernelGGL(
        (llaminar2::moe_activation_packet_device::beginOriginalNativeInvocationKernel),
        dim3(1u), dim3(64u), 0u,
        reinterpret_cast<hipStream_t>(stream), *launch);
    return launchAccepted();
}

/** @brief Submit the original-slot retire without a host state observation. */
extern "C" bool hipMoEOverlayOriginalNativeRetire(
    const llaminar2::MoEOverlayOriginalNativeRetireLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        reinterpret_cast<std::uintptr_t>(stream) <= 2u ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    hipLaunchKernelGGL(
        (llaminar2::moe_activation_packet_device::retireOriginalNativeInvocationKernel),
        dim3(1u), dim3(1u), 0u,
        reinterpret_cast<hipStream_t>(stream), *launch);
    return launchAccepted();
}

/** @brief Submit the original-slot retiresinglerowbatch without a host state observation. */
extern "C" bool hipMoEOverlayOriginalNativeRetireSingleRowBatch(
    const llaminar2::MoEOverlayActivationSingleRowReturnBatchLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        reinterpret_cast<std::uintptr_t>(stream) <= 2u ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    hipLaunchKernelGGL(
        (llaminar2::moe_activation_packet_device::retireOriginalNativeInvocationBatchKernel<llaminar2::MoEOverlayActivationSingleRowReturnBatchLaunch>),
        dim3(launch->lane_count), dim3(1u), 0u,
        reinterpret_cast<hipStream_t>(stream), *launch);
    return launchAccepted();
}

/** @brief Submit the original-slot retiremultirowbatch without a host state observation. */
extern "C" bool hipMoEOverlayOriginalNativeRetireMultiRowBatch(
    const llaminar2::MoEOverlayActivationMultiRowReturnBatchLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        reinterpret_cast<std::uintptr_t>(stream) <= 2u ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    hipLaunchKernelGGL(
        (llaminar2::moe_activation_packet_device::retireOriginalNativeInvocationBatchKernel<llaminar2::MoEOverlayActivationMultiRowReturnBatchLaunch>),
        dim3(launch->lane_count), dim3(1u), 0u,
        reinterpret_cast<hipStream_t>(stream), *launch);
    return launchAccepted();
}
