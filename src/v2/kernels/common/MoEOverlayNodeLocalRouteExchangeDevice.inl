/**
 * @file MoEOverlayNodeLocalRouteExchangeDevice.inl
 * @brief Shared CUDA/HIP kernels for sparse node-local canonical-route lanes.
 *
 * Included by the CUDA and HIP activation-packet bridge translation units.
 * Producer and root graphs exchange monotonic epochs through mapped system
 * memory.  Payload work stays sparse: a producer writes only route slots whose
 * device-owned runtime assignment names it, and the root reads mapped memory
 * only for those same slots while retaining the exact original top-k fold.
 */

#pragma once

#include "execution/moe/MoEOverlayNodeLocalRouteExchangeABI.h"
#include "MoEOverlayActivationPacketDevice.inl"

#include <cstddef>
#include <cstdint>

namespace llaminar2::moe_node_local_route_device
{
    using moe_activation_packet_device::loadPeerPublished;
    using moe_activation_packet_device::loadSystemAcquire64;

    /** @return Raw fixed-width value of one ABI enum. */
    template <typename Enum>
    __device__ __forceinline__ std::uint32_t raw(Enum value) noexcept
    {
        return static_cast<std::uint32_t>(value);
    }

    /** @brief System-release store for a dynamically selected epoch value. */
    __device__ __forceinline__ void storeSystemRelease64(
        std::uint64_t *address,
        std::uint64_t value) noexcept
    {
#if defined(__CUDA_ARCH__)
        cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> reference(
            *address);
        reference.store(value, cuda::memory_order_release);
#elif defined(__HIP_DEVICE_COMPILE__)
        __hip_atomic_store(
            address,
            value,
            __ATOMIC_RELEASE,
            __HIP_MEMORY_SCOPE_SYSTEM);
#else
        *address = value;
#endif
    }

    /** @return Whether immutable lane identity matches one launch descriptor. */
    __device__ __forceinline__ bool validControl(
        const MoENodeLocalRoutePeerDeviceBinding &lane) noexcept
    {
        if (!lane.valid())
            return false;
        const auto *control = lane.control;
        return loadPeerPublished(&control->magic) ==
                   kMoENodeLocalRouteExchangeMagic &&
               loadPeerPublished(&control->version) ==
                   kMoENodeLocalRouteExchangeVersion &&
               loadPeerPublished(&control->producer_participant) ==
                   lane.producer_participant &&
               loadPeerPublished(&control->route_capacity) ==
                   lane.route_capacity &&
               loadPeerPublished(&control->d_model) == lane.d_model;
    }

    /** @brief Publish the first terminal lane fault and release both endpoints. */
    __device__ __forceinline__ void abortLane(
        MoENodeLocalRoutePeerDeviceBinding lane,
        MoENodeLocalRouteExchangeCode code,
        std::uint64_t failed_epoch) noexcept
    {
        if (!lane.control)
            return;
        lane.control->code = raw(code);
        lane.control->failed_epoch = failed_epoch;
        lane.control->state = raw(MoENodeLocalRouteExchangeState::Aborted);
        __threadfence_system();
        storeSystemRelease64(
            &lane.control->produced_epoch,
            kMoENodeLocalRouteExchangeAbortEpoch);
        storeSystemRelease64(
            &lane.control->consumed_epoch,
            kMoENodeLocalRouteExchangeAbortEpoch);
    }

    /** @return Whether a peer lane already carries a terminal failure. */
    __device__ __forceinline__ bool laneAborted(
        const MoENodeLocalRoutePeerDeviceBinding &lane) noexcept
    {
        return !validControl(lane) ||
               loadPeerPublished(&lane.control->state) !=
                   raw(MoENodeLocalRouteExchangeState::Ready) ||
               loadPeerPublished(&lane.control->code) !=
                   raw(MoENodeLocalRouteExchangeCode::Success);
    }

    /**
     * @brief Wait until the root has consumed the producer's prior payload.
     *
     * One thread owns the dynamic epoch decision.  The following payload
     * kernel remains fixed-shape and reads `produced_epoch + 1`; stream order
     * makes that value stable until the final publication kernel advances it.
     */
    static __global__ void beginRoutePublishKernel(
        MoENodeLocalRoutePublishLaunch launch)
    {
        if (blockIdx.x != 0u || threadIdx.x != 0u)
            return;
        auto lane = launch.lane;
        if (!launch.valid() || !validControl(lane))
        {
            abortLane(
                lane, MoENodeLocalRouteExchangeCode::InvalidControl, 0u);
            return;
        }
        while (true)
        {
            const std::uint64_t produced =
                loadSystemAcquire64(&lane.control->produced_epoch);
            const std::uint64_t consumed =
                loadSystemAcquire64(&lane.control->consumed_epoch);
            if (produced == kMoENodeLocalRouteExchangeAbortEpoch ||
                consumed == kMoENodeLocalRouteExchangeAbortEpoch ||
                laneAborted(lane))
            {
                abortLane(
                    lane, MoENodeLocalRouteExchangeCode::PeerAborted,
                    produced);
                return;
            }
            if (produced == consumed)
            {
                if (produced + 1u ==
                    kMoENodeLocalRouteExchangeAbortEpoch)
                {
                    abortLane(
                        lane,
                        MoENodeLocalRouteExchangeCode::EpochOverflow,
                        produced);
                }
                return;
            }
#if defined(__CUDA_ARCH__)
            __nanosleep(64u);
#elif defined(__HIP_DEVICE_COMPILE__)
            __builtin_amdgcn_s_sleep(1u);
#endif
        }
    }

    /**
     * @brief Copy only this producer's original route slots into mapped pages.
     *
     * A bounded block set walks route slots in a grid-stride loop.  This avoids
     * launching tens of thousands of empty blocks when a participant owns only
     * a small fraction of the selected experts.  One block owns each visited
     * slot, all threads stripe the hidden width, and thread zero tags the slot
     * with the still-unpublished next epoch.
     */
    static __global__ void publishSparseRoutePayloadKernel(
        MoENodeLocalRoutePublishLaunch launch)
    {
        __shared__ std::uint32_t lane_ready;
        __shared__ std::uint64_t next_epoch;
        if (threadIdx.x == 0u)
        {
            /*
             * Mapped control reads are PCIe transactions. Performing the full
             * validation and epoch load in every payload thread turns a small
             * control ticket into hundreds of thousands of serialized remote
             * reads. The preceding one-thread begin kernel authenticated the
             * immutable lane; one thread per block now checks only the mutable
             * terminal state and broadcasts the epoch through shared memory.
             */
            lane_ready =
                launch.valid() &&
                loadPeerPublished(&launch.lane.control->state) ==
                    raw(MoENodeLocalRouteExchangeState::Ready) &&
                loadPeerPublished(&launch.lane.control->code) ==
                    raw(MoENodeLocalRouteExchangeCode::Success);
            next_epoch = lane_ready
                             ? loadSystemAcquire64(
                                   &launch.lane.control->produced_epoch) +
                                   1u
                             : 0u;
        }
        __syncthreads();
        if (lane_ready == 0u)
            return;
        for (std::uint32_t slot = blockIdx.x;
             slot < launch.live_route_slots;
             slot += gridDim.x)
        {
            if (launch.route_participant_ids[slot] !=
                launch.lane.producer_participant)
            {
                continue;
            }
            const std::size_t base =
                static_cast<std::size_t>(slot) *
                static_cast<std::size_t>(launch.lane.d_model);
            for (std::uint32_t column = threadIdx.x;
                 column < launch.lane.d_model;
                 column += blockDim.x)
            {
                launch.lane.route_payload[base + column] =
                    launch.canonical_route_contributions[base + column];
            }
            if (threadIdx.x == 0u)
                launch.lane.slot_epochs[slot] = next_epoch;
        }

        /*
         * Do not fence mapped system memory from every payload block. The
         * fixed-stream finalizer cannot begin until this entire kernel has
         * completed, and its one system fence orders every payload and slot
         * tag before the release-store of `produced_epoch`. Per-block system
         * fences are therefore both redundant and catastrophically expensive
         * over PCIe (hundreds of serialized flushes for one publication).
         */
    }

    /** @brief Release the complete sparse payload as one monotonic epoch. */
    static __global__ void finishRoutePublishKernel(
        MoENodeLocalRoutePublishLaunch launch)
    {
        if (blockIdx.x != 0u || threadIdx.x != 0u ||
            !launch.valid() || laneAborted(launch.lane))
        {
            return;
        }
        const std::uint64_t next_epoch =
            loadSystemAcquire64(&launch.lane.control->produced_epoch) + 1u;
        __threadfence_system();
        storeSystemRelease64(
            &launch.lane.control->produced_epoch, next_epoch);
    }

    /**
     * @brief Acquire one fresh epoch from every non-root continuation peer.
     *
     * `validation_status` is root-device storage.  Zero means the following
     * validation/fold may proceed; a non-zero ABI code makes the fold publish
     * NaNs and the finalizer abort every lane so no producer remains resident
     * in a wait.
     */
    static __global__ void beginRouteConsumeKernel(
        MoENodeLocalRouteConsumeLaunch launch)
    {
        if (blockIdx.x != 0u || threadIdx.x != 0u)
            return;
        auto *const validation_status = launch.validation_status;
        if (!validation_status)
            return;
        *validation_status = raw(MoENodeLocalRouteExchangeCode::Success);
        if (!launch.valid())
        {
            *validation_status =
                raw(MoENodeLocalRouteExchangeCode::InvalidControl);
            return;
        }
        for (std::uint32_t peer_index = 0u;
             peer_index < launch.peer_count;
             ++peer_index)
        {
            const auto lane = launch.peers[peer_index];
            if (!validControl(lane) || laneAborted(lane))
            {
                *validation_status =
                    raw(MoENodeLocalRouteExchangeCode::PeerAborted);
                return;
            }
            while (true)
            {
                const std::uint64_t produced =
                    loadSystemAcquire64(&lane.control->produced_epoch);
                const std::uint64_t consumed =
                    loadSystemAcquire64(&lane.control->consumed_epoch);
                if (produced == kMoENodeLocalRouteExchangeAbortEpoch ||
                    consumed == kMoENodeLocalRouteExchangeAbortEpoch ||
                    laneAborted(lane))
                {
                    *validation_status =
                        raw(MoENodeLocalRouteExchangeCode::PeerAborted);
                    return;
                }
                if (produced == consumed + 1u)
                    break;
#if defined(__CUDA_ARCH__)
                __nanosleep(64u);
#elif defined(__HIP_DEVICE_COMPILE__)
                __builtin_amdgcn_s_sleep(1u);
#endif
            }
        }
    }

    /** @return Peer binding whose logical id matches `participant`, if local. */
    __device__ __forceinline__ const MoENodeLocalRoutePeerDeviceBinding *
    peerForParticipant(
        MoENodeLocalRouteConsumeLaunch launch,
        std::int32_t participant) noexcept
    {
        for (std::uint32_t peer = 0u; peer < launch.peer_count; ++peer)
        {
            if (launch.peers[peer].producer_participant == participant)
                return &launch.peers[peer];
        }
        return nullptr;
    }

    /**
     * @brief Pull only peer-owned route rows into root-device scratch.
     *
     * One bounded block family is assigned to each peer lane. A block examines
     * one route slot at a time and issues a fully coalesced hidden-row copy only
     * when the authoritative runtime assignment names that peer. This keeps
     * mapped PCIe reads out of the arithmetic kernel without copying the large
     * zero/stale portion of the lane.
     */
    static __global__ void stageSparseRoutePayloadKernel(
        MoENodeLocalRouteConsumeLaunch launch)
    {
        const std::uint32_t peer_index = blockIdx.y;
        if (!launch.valid() || peer_index >= launch.peer_count ||
            *launch.validation_status !=
                static_cast<std::int32_t>(
                    raw(MoENodeLocalRouteExchangeCode::Success)))
        {
            return;
        }
        const auto lane = launch.peers[peer_index];
        const std::uint32_t live_slots =
            launch.physical_rows * launch.top_k;
        for (std::uint32_t slot = blockIdx.x;
             slot < live_slots;
             slot += gridDim.x)
        {
            if (launch.route_participant_ids[slot] !=
                lane.producer_participant)
            {
                continue;
            }
            const std::size_t base =
                static_cast<std::size_t>(slot) * launch.d_model;
            for (std::uint32_t column = threadIdx.x;
                 column < launch.d_model;
                 column += blockDim.x)
            {
                lane.route_payload[base + column] = loadPeerPublished(
                    lane.mapped_route_payload + base + column);
            }
        }
    }

    /**
     * @brief Validate that every root-selected peer slot belongs to this epoch.
     *
     * Participants outside the local continuation domain are deliberately
     * ignored: their rows arrive through the separate heterogeneous sparse
     * return lane.  A local peer selected by the root must have tagged the slot
     * in the acquired epoch, otherwise the participant graphs disagree about
     * device-owned routing state and the transaction is terminally invalid.
     */
    static __global__ void validateSparseRouteSlotsKernel(
        MoENodeLocalRouteConsumeLaunch launch)
    {
        auto *const validation_status = launch.validation_status;
        const std::uint32_t slot =
            static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const std::uint32_t live_slots = launch.physical_rows * launch.top_k;
        if (slot >= live_slots ||
            *validation_status !=
                static_cast<std::int32_t>(
                    raw(MoENodeLocalRouteExchangeCode::Success)))
        {
            return;
        }
        const std::int32_t participant = launch.route_participant_ids[slot];
        if (participant < 0 || participant == launch.root_participant)
            return;
        const auto *peer = peerForParticipant(launch, participant);
        if (!peer)
            return; // A heterogeneous/rank-remote participant is folded later.
        const std::uint64_t expected =
            loadSystemAcquire64(&peer->control->consumed_epoch) + 1u;
        const std::uint64_t observed =
            loadPeerPublished(peer->slot_epochs + slot);
        if (observed != expected)
        {
#if defined(__CUDA_ARCH__)
            atomicCAS(
                reinterpret_cast<unsigned int *>(validation_status),
                raw(MoENodeLocalRouteExchangeCode::Success),
                raw(MoENodeLocalRouteExchangeCode::RouteAssignmentMismatch));
#elif defined(__HIP_DEVICE_COMPILE__)
            atomicCAS(
                reinterpret_cast<unsigned int *>(validation_status),
                raw(MoENodeLocalRouteExchangeCode::Success),
                raw(MoENodeLocalRouteExchangeCode::RouteAssignmentMismatch));
#endif
        }
    }

    /**
     * @brief Select local/mapped route sources and fold top-k in exact order.
     *
     * Each `(row, hidden-column)` element has one writer.  Remote-tier routes
     * contribute exact +0 here and are added later by their canonical sparse
     * return stage.  Local peer completion order cannot affect source choice or
     * the FP32 arithmetic sequence.
     */
    static __global__ void foldSparseCanonicalRoutesKernel(
        MoENodeLocalRouteConsumeLaunch launch)
    {
        const auto *const validation_status = launch.validation_status;
        const std::uint32_t column =
            static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const std::uint32_t row = blockIdx.y;
        if (row >= launch.physical_rows || column >= launch.d_model)
            return;
        const std::size_t output_index =
            static_cast<std::size_t>(row) * launch.d_model + column;
        if (*validation_status !=
            static_cast<std::int32_t>(
                raw(MoENodeLocalRouteExchangeCode::Success)))
        {
            launch.dense_output[output_index] = __int_as_float(0x7fc00000);
            return;
        }

        float sum = 0.0f;
#pragma unroll 1
        for (std::uint32_t route = 0u; route < launch.top_k; ++route)
        {
            const std::uint32_t slot = row * launch.top_k + route;
            const std::int32_t participant =
                launch.route_participant_ids[slot];
            float contribution = 0.0f;
            if (participant == launch.root_participant)
            {
                contribution = launch.root_canonical_route_contributions[
                    static_cast<std::size_t>(slot) * launch.d_model + column];
            }
            else if (participant >= 0)
            {
                const auto *peer = peerForParticipant(launch, participant);
                if (peer)
                {
                    contribution = loadPeerPublished(
                        peer->route_payload +
                        static_cast<std::size_t>(slot) * launch.d_model +
                        column);
                }
            }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            sum = __fadd_rn(sum, contribution);
#else
            sum += contribution;
#endif
        }
        launch.dense_output[output_index] = sum;
    }

    /** @brief Acknowledge the acquired payload or abort all lanes on failure. */
    static __global__ void finishRouteConsumeKernel(
        MoENodeLocalRouteConsumeLaunch launch)
    {
        const auto *const validation_status = launch.validation_status;
        const std::uint32_t peer_index = blockIdx.x;
        if (peer_index >= launch.peer_count || threadIdx.x != 0u)
            return;
        auto lane = launch.peers[peer_index];
        const auto status = static_cast<MoENodeLocalRouteExchangeCode>(
            *validation_status);
        if (status != MoENodeLocalRouteExchangeCode::Success)
        {
            abortLane(
                lane,
                status,
                loadSystemAcquire64(&lane.control->produced_epoch));
            return;
        }
        const std::uint64_t produced =
            loadSystemAcquire64(&lane.control->produced_epoch);
        __threadfence_system();
        storeSystemRelease64(&lane.control->consumed_epoch, produced);
    }
} // namespace llaminar2::moe_node_local_route_device
