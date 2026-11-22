/**
 * @file CUDAMoEOverlayActivationPacketKernels.cu
 * @brief CUDA launches for device-owned sparse ExpertOverlay activation epochs.
 *
 * Every function enqueues only on the caller's exact non-default stream.
 * Metadata and compact payload kernels access one setup-owned mapped channel.
 * The adjacent mapped 64-bit timeline is the exact producer/consumer edge for
 * every byte; no host wait, default stream, or fixed-capacity DMA is involved.
 * Sparse return payload grids are bounded independently of their capacity:
 * grid-stride kernels cover every live element without scheduling an idle
 * full-capacity grid for each participant lane.
 */

#include "CUDAMoEOverlayActivationPacketKernels.h"
#include "backends/cuda/CUDAHardwareMetadata.h"

#include <cuda_runtime.h>

#include "../../common/MoEOverlayActivationPacketDevice.inl"
#include "../../common/MoEOverlayNativeInvocationDevice.inl"
#include "../../common/MoEOverlayNodeLocalRouteExchangeDevice.inl"
#include "backends/cuda/CUDAGraphCapture.h"
#include "backends/cuda/MappedTransferServiceDevice.cuh"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

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
    /**
     * @brief Threads for one single-row dispatch kernel on the current device.
     * @param kernel The exact dispatch kernel about to be launched.
     * @return The 512-thread schedule capped by that kernel's compiled bound,
     *         or 0 when CUDA cannot report the bound.
     *
     * The shared device bound (kSingleRowDispatchThreads) is 256 in sm_120+
     * code and 512 elsewhere. The cubin or JIT image the driver selects for
     * this device owns the real limit, so ask the kernel itself rather than
     * inferring the image from the device's compute capability. The kernel
     * bodies stride by blockDim.x, so either width covers every element.
     * The query enqueues nothing and is legal during stream capture.
     */
    template <typename Kernel>
    unsigned int singleRowDispatchThreads(Kernel kernel) noexcept
    {
        cudaFuncAttributes attributes{};
        if (cudaFuncGetAttributes(&attributes, kernel) != cudaSuccess ||
            attributes.maxThreadsPerBlock <= 0)
            return 0u;
        return std::min(kSingleRowThreads,
                        static_cast<unsigned int>(attributes.maxThreadsPerBlock));
    }
    /** Bounded route-slot blocks; each block walks additional slots grid-stride. */
    constexpr unsigned int kSparseRoutePayloadBlocks = 256u;

    /**
     * @brief Expose more PCIe reads once the row-owned grid reaches its bound.
     * @param routes Captured route capacity, independent of current live routing.
     * @return Threads per mapped return-copy block.
     *
     * Small verifier packets retain the lower-latency 256-thread block. Bulk
     * packets cannot add blocks beyond the bounded row grid, so doubling its
     * warps hides dependent mapped-read latency without changing storage.
     */
    constexpr unsigned int returnPayloadThreads(std::size_t routes) noexcept
    {
        return routes >= kSparseRoutePayloadBlocks ? kSingleRowThreads : kThreads;
    }

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

namespace llaminar2::cuda_moe_packet
{
    namespace
    {
    using namespace moe_activation_packet_device;
    constexpr unsigned kDispatchMetadataThreads = 32;
    /**
     * @brief Authenticate the original lane and publish its exact active payload predicate.
     * @tparam AcquireInMetadata This is the unique consumer of the resident route word.
     * @param launch Original grant, placement and packet operands.
     * @param handle Native condition for the exact live mapped row transfer.
     * @param ready Original same-device word when this warp owns its acquire.
     */
    template<bool AcquireInMetadata>
    __global__ __launch_bounds__(32) void prepareResidentDmaMetadata(
        const __grid_constant__ MoEOverlayActivationDispatchPackLaunch launch,
        cudaGraphConditionalHandle handle, std::uint64_t* ready)
    {
        // Lane zero authenticates the acquired grant and placement. The warp
        // then compacts one original router slot per lane in canonical order.
        constexpr unsigned warp = 0xffffffffu;
        const unsigned slot = threadIdx.x;
        DispatchMetadataPreparation preparation{};
        if (slot == 0)
        {
            preparation = prepareDispatchMetadata(launch);
            if constexpr (AcquireInMetadata)
                consumeResidentPacketAcquire64(ready, 1u);
        }
        // Grant/placement validation needs no router output. Finish the one
        // acquire before any lane reads the selected route and its weight.
        if constexpr (AcquireInMetadata)
            __syncwarp(warp);
        const auto placement_ptr = __shfl_sync(warp,
            reinterpret_cast<unsigned long long>(preparation.route_participants), 0);
        const auto* placement = reinterpret_cast<const int*>(placement_ptr);
        const auto rows = __shfl_sync(warp, preparation.active_rows, 0);
        if (!placement)
        {
            if (slot == 0)
                cudaGraphSetConditional(handle, 0);
            return;
        }

        int expert = -1;
        int participant = -1;
        float weight = 0.f;
        const bool real = rows == 1 && slot < static_cast<unsigned>(launch.packet.top_k);
        const bool valid = !real || decodeDispatchRoute(launch, placement, slot, &expert, &weight, &participant);
        if (__ballot_sync(warp, !valid))
        {
            if (slot == 0)
            {
                publishFailure(launch.control, launch.grant,
                    MoEOverlayActivationEndpoint::Continuation, MoEOverlayActivationOperation::PublishDispatch,
                    MoEOverlayActivationStatusCode::PayloadMismatch, launch.stage_ordinal, launch.model_layer_index, 0);
                cudaGraphSetConditional(handle, 0);
            }
            return;
        }

        const bool selected = real && participant == launch.target_participant_id;
        const auto mask = __ballot_sync(warp, selected);
        const unsigned entries = __popc(mask);
        if (entries > launch.packet.entry_capacity)
        {
            if (slot == 0)
            {
                publishFailure(launch.control, launch.grant,
                    MoEOverlayActivationEndpoint::Continuation, MoEOverlayActivationOperation::PublishDispatch,
                    MoEOverlayActivationStatusCode::PayloadMismatch, launch.stage_ordinal, launch.model_layer_index, 0);
                cudaGraphSetConditional(handle, 0);
            }
            return;
        }

        if (selected)
        {
            const auto index = __popc(mask & ((1u << slot) - 1u));
            launch.packet.expert_ids[index] = expert;
            launch.packet.route_weights[index] = weight;
            launch.packet.original_route_slots[index] = static_cast<int>(slot);
            launch.packet.compact_route_slots[index] = static_cast<int>(index);
        }
        __syncwarp(warp);
        if (slot == 0)
        {
            launch.packet.entry_offsets[0] = 0;
            if (entries)
            {
                launch.packet.entry_offsets[1] = static_cast<int>(entries);
                launch.packet.row_ids[0] = 0;
            }
            finishDispatchMetadata(launch, entries != 0, entries);
            const auto& grant = *launch.grant;
            cudaGraphSetConditional(handle, grant.code == raw(MoEOverlayActivationStatusCode::Success) && entries != 0);
        }
    }

    /** @brief Publish the original lease once the validated exact transfer has retired. */
    __global__ void publishResidentDmaPacket(MoEOverlayActivationTimelinePublishDeviceBinding publication)
    {
        storeSystemRelease64(publication);
    }

    /**
     * @brief Record one lane's metadata, live DMA, and ordered lease release.
     * @tparam AcquireInMetadata Whether the metadata warp acquires the route once.
     * @param lane Original lane and TransferEngine-bound storage.
     * @param stream Exact captured stream with the early router dependency.
     * @param ready Same-device release word when this warp owns the acquire.
     * @return Whether CUDA accepted the complete publication.
     */
    template<bool AcquireInMetadata = false>
    bool recordResidentDispatchLane(const ResidentMappedDispatch& lane, cudaStream_t stream,
        std::uint64_t* ready = nullptr)
    {
        const auto& launch = lane.launch;
        CUDAActiveCaptureConditional branch(stream, CUDAActiveCaptureConditionalKind::IfOnly);
        if (!branch.publishPredicate([&](cudaGraphConditionalHandle condition) {
                prepareResidentDmaMetadata<AcquireInMetadata><<<1, kDispatchMetadataThreads, 0, stream>>>(
                    launch.packet, condition, ready);
                return launchAccepted();
            }) || !branch.beginBranches() ||
            !branch.appendMappedHostCopy(CUDAActiveCaptureConditionalBranch::IfNonZero, lane.copy) ||
            !branch.commit())
            throw std::runtime_error(branch.error());

        // Empty packets still publish their authenticated timeline so the CPU
        // endpoint can retire the lease without waiting for an absent copy.
        publishResidentDmaPacket<<<1, 1, 0, stream>>>(launch.publication);
        return launchAccepted();
    }
    } // namespace

    /** @copydoc prepareResidentDispatchPacket */
    bool prepareResidentDispatchPacket(int device_ordinal)
    {
        const auto& hardware = cudaHardwareMetadata().at(device_ordinal);
        int current = -1;
        if (hardware.sm_major != 8 || hardware.sm_minor != 6 ||
            cudaGetDevice(&current) != cudaSuccess || current != device_ordinal)
            return false;

        using Peer = ResidentDispatchPeerBudget;
        constexpr unsigned peer_registers = Peer::blocks_per_sm * Peer::block_threads * Peer::registers_per_thread;
        const auto fits = [&](const auto kernel) {
            cudaFuncAttributes attributes{};
            if (cudaFuncGetAttributes(&attributes, kernel) != cudaSuccess)
                return false;
            const unsigned packet_registers = (unsigned(attributes.numRegs) + 7u) / 8u * 8u;
            // The device graph conditional API owns a linked call frame.
            // Register, thread and shared limits determine concurrent residency.
            return attributes.maxThreadsPerBlock >= int(kDispatchMetadataThreads) &&
                unsigned(hardware.registers_per_sm) >= peer_registers + kDispatchMetadataThreads * packet_registers &&
                unsigned(hardware.maximum_threads_per_sm) >= Peer::blocks_per_sm * Peer::block_threads + kDispatchMetadataThreads &&
                unsigned(hardware.maximum_blocks_per_sm) > Peer::blocks_per_sm &&
                hardware.shared_memory_per_sm >= Peer::blocks_per_sm * Peer::shared_bytes + attributes.sharedSizeBytes;
        };
        return fits(prepareResidentDmaMetadata<false>) && fits(prepareResidentDmaMetadata<true>);
    }

    /** @copydoc recordResidentDispatchPackets */
    bool recordResidentDispatchPackets(std::span<const ResidentMappedDispatch> lanes,
        int device_ordinal, void* stream, const CUDAResidentPacketAcquire* acquire)
    {
        if (lanes.empty() || (acquire && lanes.size() != 1) || stream == cudaStreamLegacy || stream == cudaStreamPerThread ||
            !selectLaunchContext(device_ordinal, stream))
            return false;
        for (const auto& lane : lanes)
        {
            const auto& launch = lane.launch;
            if (!launch.valid() || launch.packet.packet.top_k > int(kDispatchMetadataThreads) ||
                launch.packet.hidden_payload_layout != MoEOverlayActivationHiddenPayloadLayout::CompactRows ||
                lane.copy.device() != DeviceId::cuda(device_ordinal) ||
                lane.copy.source() != launch.packet.hidden_rows_fp32 ||
                lane.copy.mappedAlias() != launch.packet.packet.hidden_rows_fp32 ||
                lane.copy.bytes() != std::size_t(launch.packet.packet.d_model) * sizeof(float))
                return false;
        }

        const auto native = static_cast<cudaStream_t>(stream);
        if (acquire)
            return recordResidentDispatchLane<true>(lanes.front(), native, acquire->claimSignal(stream));
        if (lanes.size() == 1)
            return recordResidentDispatchLane(lanes.front(), native);

        // All admitted lanes borrow the same router frontier. Preserve each
        // native edge and join their individual releases without serializing
        // the independent host packets behind one another.
        cudaStreamCaptureStatus state{};
        unsigned long long capture_id = 0;
        cudaGraph_t graph = nullptr;
        const cudaGraphNode_t* incoming = nullptr;
        const cudaGraphEdgeData* incoming_edges = nullptr;
        std::size_t count = 0;
        if (cudaStreamGetCaptureInfo(native, &state, &capture_id, &graph,
                &incoming, &incoming_edges, &count) != cudaSuccess ||
            state != cudaStreamCaptureStatusActive || !capture_id || !graph ||
            count == 0 || !incoming || !incoming_edges)
            return false;
        std::vector<cudaGraphNode_t> fork(incoming, incoming + count);
        const std::vector<cudaGraphEdgeData> edges(incoming_edges, incoming_edges + count);
        std::vector<cudaGraphNode_t> completions;
        completions.reserve(lanes.size());
        for (const auto& lane : lanes)
        {
            if (cudaStreamUpdateCaptureDependencies(native, fork.data(), edges.data(), fork.size(),
                    cudaStreamSetCaptureDependencies) != cudaSuccess ||
                !recordResidentDispatchLane(lane, native))
                return false;

            cudaGraph_t current_graph = nullptr;
            unsigned long long current_capture = 0;
            const cudaGraphNode_t* tail = nullptr;
            std::size_t tail_size = 0;
            if (cudaStreamGetCaptureInfo(native, &state, &current_capture, &current_graph,
                    &tail, nullptr, &tail_size) != cudaSuccess || state != cudaStreamCaptureStatusActive ||
                current_capture != capture_id || current_graph != graph || tail_size != 1 || !tail)
                return false;
            completions.push_back(*tail);
        }
        return cudaStreamUpdateCaptureDependencies(native, completions.data(), nullptr, completions.size(),
            cudaStreamSetCaptureDependencies) == cudaSuccess;
    }
}

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
    if (launch->shared_row_writer)
        llaminar2::moe_activation_packet_device::packSharedDispatchUnionKernel
            <<<launch->physical_rows,kThreads,0u,cuda_stream>>>(*launch);
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
    // Each block owns a route, amortizing mapped metadata across its width.
    const auto payload_blocks = static_cast<unsigned int>(std::min<std::size_t>(
        kSparseRoutePayloadBlocks, launch->returned.route_slot_capacity));
    llaminar2::moe_activation_packet_device::packMappedReturnPayloadKernel
        <<<payload_blocks, kThreads, 0u, cuda_stream>>>(*launch);
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
    // One block owns a row; payload width no longer multiplies the grid.
    const unsigned int payload_blocks = static_cast<unsigned int>(std::min<std::size_t>(
        kSparseRoutePayloadBlocks, launch->returned.route_slot_capacity));
    llaminar2::moe_activation_packet_device::materializeMappedCanonicalReturnKernel
        <<<payload_blocks, returnPayloadThreads(launch->returned.route_slot_capacity), 0u, cuda_stream>>>(*launch);
    return launchAccepted();
}

extern "C" bool cudaMoEOverlayConsumeCanonicalRouteTicket(
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
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    llaminar2::moe_activation_packet_device::awaitCanonicalRouteTicketKernel
        <<<1u, 1u, 0u, cuda_stream>>>(*launch);
    if (!launch->metadataOnly())
    {
        llaminar2::moe_activation_packet_device::
            materializeCanonicalRouteTicketKernel
            <<<blocks, kThreads, 0u, cuda_stream>>>(*launch);
    }
    /* A separate one-thread node cannot race the materialization blocks. Its
     * exact-stream release is the sole authority permitting host payload reuse. */
    llaminar2::moe_activation_packet_device::
        acknowledgeCanonicalRouteTicketKernel
        <<<1u, 1u, 0u, cuda_stream>>>(*launch);
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
    const unsigned int threads = singleRowDispatchThreads(
        llaminar2::moe_activation_packet_device::packSingleRowDispatchKernel);
    if (threads == 0u)
        return false;
    llaminar2::moe_activation_packet_device::packSingleRowDispatchKernel
        <<<1u, threads, 0u, cuda_stream>>>(*launch);
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
    const unsigned int threads = singleRowDispatchThreads(
        llaminar2::moe_activation_packet_device::consumeSingleRowDispatchKernel);
    if (threads == 0u)
        return false;
    llaminar2::moe_activation_packet_device::consumeSingleRowDispatchKernel
        <<<1u, threads, 0u, cuda_stream>>>(*launch);
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
    const unsigned int threads = singleRowDispatchThreads(
        llaminar2::moe_activation_packet_device::packSingleRowDispatchBatchKernel);
    if (threads == 0u)
        return false;
    llaminar2::moe_activation_packet_device::packSingleRowDispatchBatchKernel
        <<<launch->lane_count, threads, 0u, cuda_stream>>>(*launch);
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
    llaminar2::moe_activation_packet_device::validateSingleRowReturnBatchKernel
        <<<launch->lane_count, kSingleRowThreads, 0u, cuda_stream>>>(*launch);
    llaminar2::moe_activation_packet_device::materializeSingleRowCanonicalReturnBatchKernel
        <<<dim3(static_cast<unsigned int>(launch->top_k), launch->lane_count, 1u),
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
    const std::size_t route_capacity =
        static_cast<std::size_t>(launch->physical_rows) *
        static_cast<std::size_t>(launch->top_k);
    // A lane owns only its compact live entries, not every possible route.
    // Launching a capacity-sized grid per lane multiplies idle blocks and
    // system-memory latency. The existing grid-stride loop retains unique
    // writers and full coverage while reusing the bounded payload geometry.
    const unsigned int payload_blocks = static_cast<unsigned int>(std::min<std::size_t>(
        kSparseRoutePayloadBlocks, route_capacity));
    llaminar2::moe_activation_packet_device::materializeMultiRowCanonicalReturnBatchKernel
        <<<dim3(payload_blocks, launch->lane_count, 1u),
           returnPayloadThreads(route_capacity), 0u, cuda_stream>>>(*launch);
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

extern "C" bool cudaMoEOverlayBeginNodeLocalDensePublication(
    const llaminar2::MoENodeLocalDensePublicationLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() || !launch->binding.isRoot() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    llaminar2::moe_node_local_route_device::beginDensePublicationKernel
        <<<1u, 1u, 0u, cuda_stream>>>(*launch);
    return launchAccepted();
}

extern "C" bool cudaMoEOverlayFinishNodeLocalDensePublication(
    const llaminar2::MoENodeLocalDensePublicationLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() || !launch->binding.isRoot() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    llaminar2::moe_node_local_route_device::finishDensePublicationKernel
        <<<1u, 1u, 0u, cuda_stream>>>(*launch);
    return launchAccepted();
}

extern "C" bool cudaMoEOverlayBeginNodeLocalDensePublicationConsume(
    const llaminar2::MoENodeLocalDensePublicationLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() || launch->binding.isRoot() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    llaminar2::moe_node_local_route_device::beginDensePublicationConsumeKernel
        <<<1u, 1u, 0u, cuda_stream>>>(*launch);
    return launchAccepted();
}

extern "C" bool cudaMoEOverlayFinishNodeLocalDensePublicationConsume(
    const llaminar2::MoENodeLocalDensePublicationLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() || launch->binding.isRoot() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    llaminar2::moe_node_local_route_device::finishDensePublicationConsumeKernel
        <<<1u, 1u, 0u, cuda_stream>>>(*launch);
    return launchAccepted();
}


/** @brief Submit the original-slot begin without a host state observation. */
extern "C" bool cudaMoEOverlayOriginalNativeBegin(
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
    llaminar2::moe_activation_packet_device::beginOriginalNativeInvocationKernel
        <<<1u, 64u, 0u, reinterpret_cast<cudaStream_t>(stream)>>>(*launch);
    return launchAccepted();
}

/** @brief Submit the original-slot retire without a host state observation. */
extern "C" bool cudaMoEOverlayOriginalNativeRetire(
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
    llaminar2::moe_activation_packet_device::retireOriginalNativeInvocationKernel
        <<<1u, 1u, 0u, reinterpret_cast<cudaStream_t>(stream)>>>(*launch);
    return launchAccepted();
}

/** @brief Submit the original-slot retiresinglerowbatch without a host state observation. */
extern "C" bool cudaMoEOverlayOriginalNativeRetireSingleRowBatch(
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
    llaminar2::moe_activation_packet_device::retireOriginalNativeInvocationBatchKernel<llaminar2::MoEOverlayActivationSingleRowReturnBatchLaunch>
        <<<launch->lane_count, 1u, 0u, reinterpret_cast<cudaStream_t>(stream)>>>(*launch);
    return launchAccepted();
}

/** @brief Submit the original-slot retiremultirowbatch without a host state observation. */
extern "C" bool cudaMoEOverlayOriginalNativeRetireMultiRowBatch(
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
    llaminar2::moe_activation_packet_device::retireOriginalNativeInvocationBatchKernel<llaminar2::MoEOverlayActivationMultiRowReturnBatchLaunch>
        <<<launch->lane_count, 1u, 0u, reinterpret_cast<cudaStream_t>(stream)>>>(*launch);
    return launchAccepted();
}
