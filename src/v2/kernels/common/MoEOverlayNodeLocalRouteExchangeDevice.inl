/**
 * @file MoEOverlayNodeLocalRouteExchangeDevice.inl
 * @brief Shared CUDA/HIP kernels for sparse node-local canonical-route lanes.
 *
 * Included by the CUDA and HIP activation-packet bridge translation units.
 * Producer and root graphs exchange monotonic epochs through mapped system
 * memory.  Payload work stays sparse: a producer writes only route slots whose
 * device-owned runtime assignment names it, and the root reads mapped memory
 * only for those same slots while retaining the exact original top-k fold.
 * Raw continuation shards complete their participant sum before the router
 * multiply, using the same arithmetic order as the CPU route return.
 */

#pragma once

#include "execution/moe/MoEOverlayNodeLocalRouteExchangeABI.h"
#include "DeviceFP32NumericalContract.h"
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

    /**
     * @brief Read one original route slot from the final assignment ledger.
     *
     * The upstream route transaction has already resolved Static, Dynamic, or
     * LLEP policy against its pinned residency epoch. Transport is deliberately
     * policy-blind and consumes that exact assignment without reconstruction.
     */
    __device__ __forceinline__ std::int32_t routeParticipantForSlot(
        const MoEDomainRouteAssignmentLedger &assignment,
        std::uint32_t slot) noexcept
    {
        return assignment.participant_ids[slot];
    }

    /** @return Whether the original participant contributes to this device-published slot. */
    __device__ __forceinline__ bool ownsRouteSlot(
        MoENodeLocalRouteOwnership ownership, std::int32_t assignment, std::int32_t participant) noexcept
    {
        return ownership == MoENodeLocalRouteOwnership::ContinuationCohort
            ? assignment >= 0 : assignment == participant;
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

    /** @return Whether immutable lane identity matches an already-snapshotted control record. */
    __device__ __forceinline__ bool validControlSnapshot(
        const MoENodeLocalRouteLaneControl &control,
        const MoENodeLocalRoutePeerDeviceBinding &lane) noexcept
    {
        return lane.valid() &&
               control.magic == kMoENodeLocalRouteExchangeMagic &&
               control.version == kMoENodeLocalRouteExchangeVersion &&
               control.producer_participant == lane.producer_participant &&
               control.route_capacity == lane.route_capacity &&
               control.d_model == lane.d_model;
    }

    /** @return Whether an already-snapshotted control record carries a terminal fault. */
    __device__ __forceinline__ bool laneAbortedSnapshot(
        const MoENodeLocalRouteLaneControl &control) noexcept
    {
        return control.state != raw(MoENodeLocalRouteExchangeState::Ready) ||
               control.code != raw(MoENodeLocalRouteExchangeCode::Success);
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

    /**
     * @return Whether an authenticated peer lane carries a terminal failure.
     *
     * The acquire kernel authenticates immutable identity once before polling.
     * Only the state and status may change while it waits for the next epoch;
     * rereading five immutable fields each poll adds serial PCIe transactions.
     */
    __device__ __forceinline__ bool laneAborted(
        const MoENodeLocalRoutePeerDeviceBinding &lane) noexcept
    {
        return loadPeerPublished(&lane.control->state) !=
                   raw(MoENodeLocalRouteExchangeState::Ready) ||
               loadPeerPublished(&lane.control->code) !=
                   raw(MoENodeLocalRouteExchangeCode::Success);
    }

    /**
     * @brief Wait until the root has consumed the producer's prior payload.
     *
     * Block-coalesces the 64-byte control cache line into one PCIe read before
     * thread zero owns the dynamic epoch decision.
     */
    static __global__ void beginRoutePublishKernel(
        MoENodeLocalRoutePublishLaunch launch)
    {
        if (blockIdx.x != 0u)
            return;
        auto lane = launch.lane;
        if (!launch.valid())
        {
            if (threadIdx.x == 0u)
                abortLane(lane, MoENodeLocalRouteExchangeCode::InvalidControl, 0u);
            return;
        }
        const auto initial =
            moe_activation_packet_device::snapshotPeerPublishedBlock(lane.control);
        if (threadIdx.x != 0u)
            return;
        if (!validControlSnapshot(initial, lane))
        {
            abortLane(lane, MoENodeLocalRouteExchangeCode::InvalidControl, 0u);
            return;
        }
        const std::uint64_t produced = initial.produced_epoch;
        std::uint64_t consumed = initial.consumed_epoch;
        if (produced == kMoENodeLocalRouteExchangeAbortEpoch ||
            consumed == kMoENodeLocalRouteExchangeAbortEpoch ||
            laneAbortedSnapshot(initial))
        {
            abortLane(lane, MoENodeLocalRouteExchangeCode::PeerAborted, produced);
            return;
        }
        if (produced == consumed)
        {
            if (produced + 1u == kMoENodeLocalRouteExchangeAbortEpoch)
                abortLane(lane, MoENodeLocalRouteExchangeCode::EpochOverflow, produced);
            return;
        }
        while (true)
        {
            consumed = loadSystemAcquire64(&lane.control->consumed_epoch);
            if (consumed == kMoENodeLocalRouteExchangeAbortEpoch)
            {
                abortLane(lane, MoENodeLocalRouteExchangeCode::PeerAborted, produced);
                return;
            }
            if (produced == consumed)
            {
                if (produced + 1u == kMoENodeLocalRouteExchangeAbortEpoch)
                    abortLane(lane, MoENodeLocalRouteExchangeCode::EpochOverflow, produced);
                return;
            }
            if (laneAborted(lane))
            {
                abortLane(lane, MoENodeLocalRouteExchangeCode::PeerAborted, produced);
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
     * @brief Fused single-block producer begin + sparse copy + epoch release
     *        for one-row decode (`live_route_slots <= 32`).
     */
    static __global__ void publishSingleRowSparseRouteKernel(
        MoENodeLocalRoutePublishLaunch launch)
    {
        if (blockIdx.x != 0u)
            return;
        auto lane = launch.lane;
        __shared__ std::uint32_t lane_ready;
        __shared__ std::uint64_t next_epoch;
        if (!launch.valid())
        {
            if (threadIdx.x == 0u)
                abortLane(lane, MoENodeLocalRouteExchangeCode::InvalidControl, 0u);
            return;
        }
        const auto initial =
            moe_activation_packet_device::snapshotPeerPublishedBlock(lane.control);
        if (threadIdx.x == 0u)
        {
            lane_ready = 0u;
            next_epoch = 0u;
            if (!validControlSnapshot(initial, lane))
            {
                abortLane(lane, MoENodeLocalRouteExchangeCode::InvalidControl, 0u);
            }
            else
            {
                const std::uint64_t produced = initial.produced_epoch;
                std::uint64_t consumed = initial.consumed_epoch;
                if (produced == kMoENodeLocalRouteExchangeAbortEpoch ||
                    consumed == kMoENodeLocalRouteExchangeAbortEpoch ||
                    laneAbortedSnapshot(initial))
                {
                    abortLane(lane, MoENodeLocalRouteExchangeCode::PeerAborted, produced);
                }
                else
                {
                    bool ok = (produced == consumed);
                    while (!ok)
                    {
                        consumed = loadSystemAcquire64(&lane.control->consumed_epoch);
                        if (consumed == kMoENodeLocalRouteExchangeAbortEpoch || laneAborted(lane))
                        {
                            abortLane(lane, MoENodeLocalRouteExchangeCode::PeerAborted, produced);
                            break;
                        }
                        if (produced == consumed)
                        {
                            ok = true;
                            break;
                        }
#if defined(__CUDA_ARCH__)
                        __nanosleep(64u);
#elif defined(__HIP_DEVICE_COMPILE__)
                        __builtin_amdgcn_s_sleep(1u);
#endif
                    }
                    if (ok)
                    {
                        if (produced + 1u == kMoENodeLocalRouteExchangeAbortEpoch)
                        {
                            abortLane(lane, MoENodeLocalRouteExchangeCode::EpochOverflow, produced);
                        }
                        else
                        {
                            lane_ready = 1u;
                            next_epoch = produced + 1u;
                        }
                    }
                }
            }
        }
        __syncthreads();
        if (lane_ready == 0u)
            return;

        __shared__ std::int32_t slot_owned[32];
        if (threadIdx.x < launch.live_route_slots)
        {
            slot_owned[threadIdx.x] = ownsRouteSlot(
                launch.ownership,
                routeParticipantForSlot(launch.domain_assignment, threadIdx.x),
                lane.producer_participant) ? 1 : 0;
        }
        __syncthreads();

        const std::size_t width = static_cast<std::size_t>(lane.d_model);
        for (std::uint32_t slot = 0u; slot < launch.live_route_slots; ++slot)
        {
            if (slot_owned[slot] == 0)
                continue;
            const std::size_t base = static_cast<std::size_t>(slot) * width;
            const float *src = launch.canonical_route_contributions + base;
            float *dst = lane.route_payload + base;
            std::size_t copied = 0u;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            if (((reinterpret_cast<std::uintptr_t>(src) |
                  reinterpret_cast<std::uintptr_t>(dst)) & (sizeof(float4) - 1u)) == 0u)
            {
                const std::size_t vecs = width / 4u;
                for (std::size_t v = threadIdx.x; v < vecs; v += blockDim.x)
                    reinterpret_cast<float4 *>(dst)[v] =
                        reinterpret_cast<const float4 *>(src)[v];
                copied = vecs * 4u;
            }
#endif
            for (std::size_t col = copied + threadIdx.x; col < width; col += blockDim.x)
                dst[col] = src[col];
            if (threadIdx.x == 0u)
                lane.slot_epochs[slot] = next_epoch;
        }
        __syncthreads();
        if (threadIdx.x == 0u)
        {
            __threadfence_system();
            storeSystemRelease64(&lane.control->produced_epoch, next_epoch);
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
        if (!launch.valid())
            return;
        bool epoch_acquired = false;
        for (std::uint32_t slot = blockIdx.x;
             slot < launch.live_route_slots;
             slot += gridDim.x)
        {
            if (!ownsRouteSlot(launch.ownership, routeParticipantForSlot(launch.domain_assignment, slot),
                               launch.lane.producer_participant))
            {
                continue;
            }
            if (!epoch_acquired)
            {
                /* The assignment is device-local and block-uniform. Empty
                 * blocks must not issue a fresh PCIe epoch read: a retained
                 * verifier envelope often selects only a handful of routes
                 * on this peer. Acquire once at the first owned row, then
                 * retain the epoch for all remaining rows of this block.
                 * The stream-ordered begin/finalizer still authenticate the
                 * lane, protect reuse and own terminal-state publication. */
                if (threadIdx.x == 0u)
                {
                    const auto produced = loadSystemAcquire64(
                        &launch.lane.control->produced_epoch);
                    lane_ready = produced != kMoENodeLocalRouteExchangeAbortEpoch;
                    next_epoch = lane_ready ? produced + 1u : 0u;
                }
                __syncthreads();
                if (lane_ready == 0u)
                    return;
                epoch_acquired = true;
            }
            const std::size_t base =
                static_cast<std::size_t>(slot) *
                static_cast<std::size_t>(launch.lane.d_model);
            const std::size_t width = static_cast<std::size_t>(launch.lane.d_model);
            const float *__restrict__ src = launch.canonical_route_contributions + base;
            std::uint64_t slot_epoch_word = next_epoch;
            if (launch.payload_precision ==
                MoENodeLocalRoutePayloadPrecision::FixedPoint32Adaptive16)
            {
                const bool wide = moe_activation_packet_device::storeRowFP32ToAdaptiveFixed16Or32(
                    src, launch.lane.route_payload + base, width);
                if (wide)
                    slot_epoch_word |= (std::uint64_t{1} << 63u);
            }
            else
            {
                float *__restrict__ dst = launch.lane.route_payload + base;
                std::size_t copied = 0u;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
                if ((width & 3u) == 0u &&
                    (((reinterpret_cast<std::uintptr_t>(dst) |
                       reinterpret_cast<std::uintptr_t>(src)) &
                      (sizeof(float4) - 1u)) == 0u))
                {
                    const std::size_t vecs = width >> 2u;
                    auto *__restrict__ dst4 = reinterpret_cast<float4 *>(dst);
                    const auto *__restrict__ src4 = reinterpret_cast<const float4 *>(src);
                    for (std::size_t v = threadIdx.x; v < vecs; v += blockDim.x)
                        dst4[v] = src4[v];
                    copied = vecs << 2u;
                }
#endif
                for (std::size_t column = copied + threadIdx.x;
                     column < width;
                     column += blockDim.x)
                {
                    dst[column] = src[column];
                }
            }
            if (threadIdx.x == 0u)
                launch.lane.slot_epochs[slot] = slot_epoch_word;
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
        if (blockIdx.x != 0u || !launch.valid())
            return;
        const auto snap =
            moe_activation_packet_device::snapshotPeerPublishedBlock(launch.lane.control);
        if (threadIdx.x != 0u)
            return;
        if (laneAbortedSnapshot(snap))
            return;
        const std::uint64_t produced = snap.produced_epoch;
        if (produced == kMoENodeLocalRouteExchangeAbortEpoch)
            return;
        const std::uint64_t next_epoch = produced + 1u;
        __threadfence_system();
        storeSystemRelease64(
            &launch.lane.control->produced_epoch, next_epoch);
    }

    /**
     * @brief Acquire one fresh epoch from every non-root continuation peer.
     *
     * Block-coalesces each peer's 64-byte control record into one PCIe read.
     */
    static __global__ void beginRouteConsumeKernel(
        MoENodeLocalRouteConsumeLaunch launch)
    {
        if (blockIdx.x != 0u)
            return;
        auto *const validation_status = launch.validation_status;
        if (!validation_status)
            return;
        if (threadIdx.x == 0u)
            *validation_status = raw(MoENodeLocalRouteExchangeCode::Success);
        if (!launch.valid())
        {
            if (threadIdx.x == 0u)
                *validation_status =
                    raw(MoENodeLocalRouteExchangeCode::InvalidControl);
            return;
        }
        __shared__ std::uint32_t peer_failed;
        if (threadIdx.x == 0u)
            peer_failed = 0u;
        __syncthreads();
        for (std::uint32_t peer_index = 0u;
             peer_index < launch.peer_count;
             ++peer_index)
        {
            const auto lane = launch.peers[peer_index];
            const auto snap =
                moe_activation_packet_device::snapshotPeerPublishedBlock(lane.control);
            if (threadIdx.x == 0u)
            {
                if (!validControlSnapshot(snap, lane) || laneAbortedSnapshot(snap) ||
                    snap.produced_epoch == kMoENodeLocalRouteExchangeAbortEpoch ||
                    snap.consumed_epoch == kMoENodeLocalRouteExchangeAbortEpoch)
                {
                    *validation_status =
                        raw(MoENodeLocalRouteExchangeCode::PeerAborted);
                    peer_failed = 1u;
                }
                else
                {
                    const std::uint64_t consumed = snap.consumed_epoch;
                    std::uint64_t produced = snap.produced_epoch;
                    while (produced != consumed + 1u)
                    {
#if defined(__CUDA_ARCH__)
                        __nanosleep(64u);
#elif defined(__HIP_DEVICE_COMPILE__)
                        __builtin_amdgcn_s_sleep(1u);
#endif
                        produced = loadSystemAcquire64(&lane.control->produced_epoch);
                        if (produced == kMoENodeLocalRouteExchangeAbortEpoch ||
                            laneAborted(lane))
                        {
                            *validation_status =
                                raw(MoENodeLocalRouteExchangeCode::PeerAborted);
                            peer_failed = 1u;
                            break;
                        }
                    }
                }
            }
            __syncthreads();
            if (peer_failed != 0u)
                return;
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
        if (lane.mapped_route_payload == lane.route_payload)
        {
            // Direct P2P peer already wrote its sparse live slots into Root's
            // local VRAM slice; foldSparseCanonicalRoutesKernel reads directly.
            return;
        }
        const std::uint32_t live_slots =
            launch.physical_rows * launch.top_k;
        __shared__ std::uint32_t s_slot_wide;
        for (std::uint32_t slot = blockIdx.x;
             slot < live_slots;
             slot += gridDim.x)
        {
            if (!ownsRouteSlot(launch.ownership, routeParticipantForSlot(launch.domain_assignment, slot),
                               lane.producer_participant))
            {
                continue;
            }
            const std::size_t base =
                static_cast<std::size_t>(slot) * launch.d_model;
            // The release/acquire above owns source visibility. Stripe fresh
            // vector loads where alignment allows it; prefill adaptive fixed-point
            // halves mapped PCIe traffic while materializing exact int32 bit-patterns.
            if (launch.payload_precision ==
                MoENodeLocalRoutePayloadPrecision::FixedPoint32Adaptive16)
            {
                if (threadIdx.x == 0u)
                {
                    const std::uint64_t tag =
                        loadPeerPublished(lane.slot_epochs + slot);
                    s_slot_wide = ((tag >> 63u) != 0u) ? 1u : 0u;
                }
                __syncthreads();
                if (s_slot_wide == 0u)
                {
                    moe_activation_packet_device::copyPeerPublishedFixed16ToInt32Bits(
                        reinterpret_cast<const std::int16_t *>(
                            lane.mapped_route_payload + base),
                        lane.route_payload + base,
                        launch.d_model);
                }
                else
                {
                    moe_activation_packet_device::copyPeerPublishedRow(
                        lane.mapped_route_payload + base,
                        lane.route_payload + base,
                        launch.d_model);
                }
                __syncthreads();
            }
            else
            {
                moe_activation_packet_device::copyPeerPublishedRow(
                    lane.mapped_route_payload + base,
                    lane.route_payload + base,
                    launch.d_model);
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
        constexpr std::uint32_t kCachedExpectedPeers = 32u;
        auto *const validation_status = launch.validation_status;
        __shared__ std::uint64_t cached_expected[kCachedExpectedPeers];
        __shared__ std::uint32_t status_ok;
        if (threadIdx.x == 0u)
        {
            status_ok = (*validation_status ==
                         static_cast<std::int32_t>(
                             raw(MoENodeLocalRouteExchangeCode::Success)))
                            ? 1u
                            : 0u;
        }
        if (threadIdx.x < launch.peer_count && threadIdx.x < kCachedExpectedPeers)
        {
            cached_expected[threadIdx.x] =
                loadSystemAcquire64(&launch.peers[threadIdx.x].control->consumed_epoch) + 1u;
        }
        __syncthreads();
        const std::uint32_t slot =
            static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const std::uint32_t live_slots = launch.physical_rows * launch.top_k;
        if (slot >= live_slots || status_ok == 0u)
        {
            return;
        }
        const std::int32_t participant =
            routeParticipantForSlot(launch.domain_assignment, slot);
        const bool joint = launch.ownership == MoENodeLocalRouteOwnership::ContinuationCohort;
        std::uint32_t first_peer = 0u;
        std::uint32_t peer_end = launch.peer_count;
        if (!joint)
        {
            if (participant < 0 || participant == launch.root_participant)
                return;
            bool found = false;
            for (std::uint32_t p = 0u; p < launch.peer_count; ++p)
            {
                if (launch.peers[p].producer_participant == participant)
                {
                    first_peer = p;
                    peer_end = p + 1u;
                    found = true;
                    break;
                }
            }
            if (!found)
                return;
        }
        for (std::uint32_t p = first_peer; p < peer_end; ++p)
        {
            const auto &peer = launch.peers[p];
            const std::uint64_t expected = p < kCachedExpectedPeers
                ? cached_expected[p]
                : (loadSystemAcquire64(&peer.control->consumed_epoch) + 1u);
            const std::uint64_t observed =
                loadPeerPublished(peer.slot_epochs + slot) &
                ~(std::uint64_t{1} << 63u);
            const bool published = observed == expected;
            if (published == (participant >= 0))
                continue;
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
     * @brief Select root-local staged route sources and fold top-k in exact order.
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
        if (row >= launch.physical_rows)
            return;
        const bool fixed_point_fold =
            launch.payload_precision ==
                MoENodeLocalRoutePayloadPrecision::FixedPoint32Adaptive16 &&
            launch.ownership != MoENodeLocalRouteOwnership::ContinuationCohort;
        const bool validation_ok =
            *validation_status ==
            static_cast<std::int32_t>(
                raw(MoENodeLocalRouteExchangeCode::Success));
        __shared__ unsigned int s_slot_wide_mask;
        if (fixed_point_fold && validation_ok && launch.top_k <= 32u)
        {
            // Cache direct-P2P adaptive 16/32-bit slot tags once per block
            // before any per-thread column return so __syncthreads() stays uniform.
            if (threadIdx.x == 0u)
                s_slot_wide_mask = 0u;
            __syncthreads();
            if (threadIdx.x < launch.top_k)
            {
                const std::uint32_t slot = row * launch.top_k + threadIdx.x;
                const std::int32_t participant =
                    routeParticipantForSlot(launch.domain_assignment, slot);
                if (participant >= 0 && participant != launch.root_participant)
                {
                    const auto *peer = peerForParticipant(launch, participant);
                    if (peer && peer->mapped_route_payload == peer->route_payload)
                    {
                        const std::uint64_t tag =
                            loadPeerPublished(peer->slot_epochs + slot);
                        if ((tag >> 63u) != 0u)
                            atomicOr(&s_slot_wide_mask, 1u << threadIdx.x);
                    }
                }
            }
            __syncthreads();
        }
        if (column >= launch.d_model)
            return;
        const std::size_t output_index =
            static_cast<std::size_t>(row) * launch.d_model + column;
        if (!validation_ok)
        {
            launch.dense_output[output_index] = __int_as_float(0x7fc00000);
            return;
        }

        if (fixed_point_fold)
        {
            // Placement-invariant exact integer accumulation across all local,
            // node-local peer, and heterogeneous remote GPU participants.
            // Remote slots and staged peers carry bit-cast int32 fixed-point
            // values (scale 2^12); direct-P2P peers carry adaptive int16/int32
            // directly in Root VRAM and are read with L1-bypassing loads.
            std::int32_t acc = 0;
#pragma unroll 1
            for (std::uint32_t route = 0u; route < launch.top_k; ++route)
            {
                const std::uint32_t slot = row * launch.top_k + route;
                const std::int32_t participant =
                    routeParticipantForSlot(launch.domain_assignment, slot);
                const std::size_t slot_base =
                    static_cast<std::size_t>(slot) * launch.d_model;
                const std::size_t elem = slot_base + column;
                if (participant == launch.root_participant)
                {
                    acc += moe_activation_packet_device::quantizePrefillFixed32(
                        launch.root_canonical_route_contributions[elem]);
                }
                else if (participant < 0 &&
                         launch.external_route_source ==
                             MoEExternalCanonicalRouteSource::RootCanonicalRouteBank)
                {
                    std::int32_t q = 0;
                    const float raw = launch.root_canonical_route_contributions[elem];
                    __builtin_memcpy(&q, &raw, sizeof(q));
                    acc += q;
                }
                else if (participant >= 0)
                {
                    const auto *peer = peerForParticipant(launch, participant);
                    if (peer)
                    {
                        if (peer->mapped_route_payload == peer->route_payload)
                        {
                            const bool is_wide = (launch.top_k <= 32u)
                                ? ((s_slot_wide_mask & (1u << route)) != 0u)
                                : ((loadPeerPublished(peer->slot_epochs + slot) >> 63u) != 0u);
                            if (!is_wide)
                            {
                                const auto *src16 = reinterpret_cast<const std::int16_t *>(
                                    peer->route_payload + slot_base);
                                acc += static_cast<std::int32_t>(
                                    loadPeerPublished(src16 + column));
                            }
                            else
                            {
                                std::int32_t q = 0;
                                const float raw = loadPeerPublished(peer->route_payload + elem);
                                __builtin_memcpy(&q, &raw, sizeof(q));
                                acc += q;
                            }
                        }
                        else
                        {
                            std::int32_t q = 0;
                            const float raw = peer->route_payload[elem];
                            __builtin_memcpy(&q, &raw, sizeof(q));
                            acc += q;
                        }
                    }
                }
            }
            launch.dense_output[output_index] =
                static_cast<float>(acc) *
                moe_activation_packet_device::kMoEPrefillFixedPointInvScale;
            return;
        }

        float sum = 0.0f;
#pragma unroll 1
        for (std::uint32_t route = 0u; route < launch.top_k; ++route)
        {
            const std::uint32_t slot = row * launch.top_k + route;
            const std::int32_t participant =
                routeParticipantForSlot(launch.domain_assignment, slot);
            float contribution = 0.0f;
            if (launch.ownership == MoENodeLocalRouteOwnership::ContinuationCohort && participant >= 0)
            {
                const auto element = static_cast<std::size_t>(slot) * launch.d_model + column;
                const float local = launch.root_canonical_route_contributions[element];
                // The peer binding is in admitted participant order, excluding
                // the root. Insert the root at its original rank coordinate.
                bool added_root = false;
                bool added_operand = false;
                for (std::uint32_t index = 0u; index < launch.peer_count; ++index)
                {
                    const auto &peer = launch.peers[index];
                    if (!added_root && launch.root_participant < peer.producer_participant)
                    {
                        contribution = added_operand ? device_fp32_contract::add(contribution, local) : local;
                        added_root = true;
                        added_operand = true;
                    }
                    const float partial =
                        (peer.mapped_route_payload == peer.route_payload)
                            ? loadPeerPublished(peer.route_payload + element)
                            : peer.route_payload[element];
                    contribution = added_operand ? device_fp32_contract::add(contribution, partial) : partial;
                    added_operand = true;
                }
                if (!added_root)
                    contribution = device_fp32_contract::add(contribution, local);
                contribution = device_fp32_contract::multiply(contribution, launch.cohort_route_weights[slot]);
            }
            else if (participant == launch.root_participant ||
                (participant < 0 &&
                 launch.external_route_source ==
                     MoEExternalCanonicalRouteSource::
                         RootCanonicalRouteBank))
            {
                contribution = launch.root_canonical_route_contributions[
                    static_cast<std::size_t>(slot) * launch.d_model + column];
            }
            else if (participant >= 0)
            {
                const auto *peer = peerForParticipant(launch, participant);
                if (peer)
                {
                    // Staged peers were written on this stream (normal caching);
                    // direct-P2P peers wrote into Root VRAM from a peer GPU, so
                    // bypass L1 via loadPeerPublished to observe coherent L2/VRAM.
                    const std::size_t elem =
                        static_cast<std::size_t>(slot) * launch.d_model + column;
                    contribution =
                        (peer->mapped_route_payload == peer->route_payload)
                            ? loadPeerPublished(peer->route_payload + elem)
                            : peer->route_payload[elem];
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

    /**
     * @brief Single-block fused acquire + validate + direct-mapped fold +
     *        release-acknowledge for single-row decode (`physical_rows == 1`,
     *        `top_k <= 32`, arbitrary `peer_count`).
     *
     * Eliminates four intermediate kernel launches and intermediate VRAM
     * staging while preserving identical protocol validation, exact ascending
     * route FP32 arithmetic order, and system-release epoch acknowledgement.
     * The first 32 peers are cached in shared memory; any additional peers are
     * read directly from the device binding array.
     */
    static __global__ void consumeSingleRowSparseRoutesKernel(
        MoENodeLocalRouteConsumeLaunch launch)
    {
        if (blockIdx.x != 0u)
            return;
        auto *const validation_status = launch.validation_status;
        if (!validation_status)
            return;
        constexpr std::uint32_t kCachedSingleRowPeers = 32u;
        __shared__ std::uint32_t shared_status;
        __shared__ std::uint64_t peer_produced[kCachedSingleRowPeers];
        alignas(MoENodeLocalRoutePeerDeviceBinding) __shared__ unsigned char
            cached_peers_storage[kCachedSingleRowPeers * sizeof(MoENodeLocalRoutePeerDeviceBinding)];
        auto *const cached_peers =
            reinterpret_cast<MoENodeLocalRoutePeerDeviceBinding *>(cached_peers_storage);
        __shared__ std::int32_t slot_participants[32];
        __shared__ std::int32_t slot_peer_idx[32];
        __shared__ float slot_cohort_weight[32];

        auto get_peer = [&](std::uint32_t p) -> const MoENodeLocalRoutePeerDeviceBinding &
        {
            return (p < kCachedSingleRowPeers) ? cached_peers[p] : launch.peers[p];
        };

        if (threadIdx.x == 0u)
        {
            shared_status = raw(MoENodeLocalRouteExchangeCode::Success);
            *validation_status = raw(MoENodeLocalRouteExchangeCode::Success);
        }
        if (!launch.valid() || launch.physical_rows != 1u ||
            launch.top_k > 32u)
        {
            if (threadIdx.x == 0u)
            {
                *validation_status =
                    raw(MoENodeLocalRouteExchangeCode::InvalidControl);
            }
            return;
        }
        if (threadIdx.x < launch.peer_count && threadIdx.x < kCachedSingleRowPeers)
            cached_peers[threadIdx.x] = launch.peers[threadIdx.x];
        __syncthreads();

        // Step 1: Acquire every peer using one coalesced 64-byte control snapshot.
        for (std::uint32_t p = 0u; p < launch.peer_count; ++p)
        {
            const auto lane = get_peer(p);
            const auto snap =
                moe_activation_packet_device::snapshotPeerPublishedBlock(lane.control);
            if (threadIdx.x == 0u &&
                shared_status == raw(MoENodeLocalRouteExchangeCode::Success))
            {
                if (!validControlSnapshot(snap, lane) || laneAbortedSnapshot(snap) ||
                    snap.produced_epoch == kMoENodeLocalRouteExchangeAbortEpoch ||
                    snap.consumed_epoch == kMoENodeLocalRouteExchangeAbortEpoch)
                {
                    shared_status = raw(MoENodeLocalRouteExchangeCode::PeerAborted);
                    if (p < kCachedSingleRowPeers)
                        peer_produced[p] = snap.produced_epoch;
                }
                else
                {
                    const std::uint64_t consumed = snap.consumed_epoch;
                    std::uint64_t produced = snap.produced_epoch;
                    while (produced != consumed + 1u)
                    {
#if defined(__CUDA_ARCH__)
                        __nanosleep(64u);
#elif defined(__HIP_DEVICE_COMPILE__)
                        __builtin_amdgcn_s_sleep(1u);
#endif
                        produced = loadSystemAcquire64(&lane.control->produced_epoch);
                        if (produced == kMoENodeLocalRouteExchangeAbortEpoch ||
                            laneAborted(lane))
                        {
                            shared_status =
                                raw(MoENodeLocalRouteExchangeCode::PeerAborted);
                            break;
                        }
                    }
                    if (p < kCachedSingleRowPeers)
                        peer_produced[p] = produced;
                }
            }
            __syncthreads();
            if (shared_status != raw(MoENodeLocalRouteExchangeCode::Success))
                break;
        }

        // Step 2: Validate all top_k slot tags in parallel and cache route metadata.
        const bool joint =
            launch.ownership == MoENodeLocalRouteOwnership::ContinuationCohort;
        if (shared_status == raw(MoENodeLocalRouteExchangeCode::Success) &&
            threadIdx.x < launch.top_k)
        {
            const std::uint32_t slot = threadIdx.x;
            const std::int32_t participant =
                routeParticipantForSlot(launch.domain_assignment, slot);
            slot_participants[slot] = participant;
            if (joint)
                slot_cohort_weight[slot] = launch.cohort_route_weights[slot];

            std::int32_t matched_peer = -1;
            if (!joint && participant >= 0 && participant != launch.root_participant)
            {
                for (std::uint32_t p = 0u; p < launch.peer_count; ++p)
                {
                    if (get_peer(p).producer_participant == participant)
                    {
                        matched_peer = static_cast<std::int32_t>(p);
                        break;
                    }
                }
            }
            slot_peer_idx[slot] = matched_peer;

            if (joint)
            {
                for (std::uint32_t p = 0u; p < launch.peer_count; ++p)
                {
                    const auto &peer = get_peer(p);
                    const std::uint64_t expected_produced =
                        (p < kCachedSingleRowPeers)
                            ? peer_produced[p]
                            : (loadSystemAcquire64(&peer.control->consumed_epoch) + 1u);
                    const bool published =
                        loadPeerPublished(peer.slot_epochs + slot) == expected_produced;
                    if (published != (participant >= 0))
                    {
                        atomicCAS(
                            &shared_status,
                            raw(MoENodeLocalRouteExchangeCode::Success),
                            raw(MoENodeLocalRouteExchangeCode::RouteAssignmentMismatch));
                    }
                }
            }
            else if (matched_peer >= 0)
            {
                const std::uint32_t p = static_cast<std::uint32_t>(matched_peer);
                const auto &peer = get_peer(p);
                const std::uint64_t expected_produced =
                    (p < kCachedSingleRowPeers)
                        ? peer_produced[p]
                        : (loadSystemAcquire64(&peer.control->consumed_epoch) + 1u);
                const bool published =
                    loadPeerPublished(peer.slot_epochs + slot) == expected_produced;
                if (!published)
                {
                    atomicCAS(
                        &shared_status,
                        raw(MoENodeLocalRouteExchangeCode::Success),
                        raw(MoENodeLocalRouteExchangeCode::RouteAssignmentMismatch));
                }
            }
        }
        __syncthreads();

        if (threadIdx.x == 0u)
            *validation_status = static_cast<std::int32_t>(shared_status);

        if (shared_status != raw(MoENodeLocalRouteExchangeCode::Success))
        {
            const float nan_val = __int_as_float(0x7fc00000);
            for (std::uint32_t col = threadIdx.x; col < launch.d_model; col += blockDim.x)
                launch.dense_output[col] = nan_val;
            __syncthreads();
            for (std::uint32_t p = threadIdx.x; p < launch.peer_count; p += blockDim.x)
            {
                const auto &peer = get_peer(p);
                abortLane(
                    peer,
                    static_cast<MoENodeLocalRouteExchangeCode>(shared_status),
                    loadSystemAcquire64(&peer.control->produced_epoch));
            }
            return;
        }

        // Step 3: Directly read peer-owned mapped rows and fold in exact route order.
        const std::size_t width = static_cast<std::size_t>(launch.d_model);
        const bool external_in_root =
            launch.external_route_source ==
            MoEExternalCanonicalRouteSource::RootCanonicalRouteBank;
        std::size_t folded_columns = 0u;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        bool vec_aligned = (width % 4u == 0u) &&
            (((reinterpret_cast<std::uintptr_t>(launch.dense_output) |
               reinterpret_cast<std::uintptr_t>(launch.root_canonical_route_contributions) |
               reinterpret_cast<std::uintptr_t>(launch.gated_shared_residual)) &
              (sizeof(float4) - 1u)) == 0u);
        for (std::uint32_t p = 0u; vec_aligned && p < launch.peer_count; ++p)
        {
            vec_aligned =
                (reinterpret_cast<std::uintptr_t>(get_peer(p).mapped_route_payload) &
                 (sizeof(float4) - 1u)) == 0u;
        }
        if (vec_aligned)
        {
            const std::size_t vecs = width / 4u;
            const auto *root_v = reinterpret_cast<const float4 *>(
                launch.root_canonical_route_contributions);
            for (std::size_t v = threadIdx.x; v < vecs; v += blockDim.x)
            {
                float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (joint && launch.peer_count == 1u)
                {
                    // Overlap up to 4 independent peer PCIe reads per batch before
                    // folding in exact ascending route order (0..top_k-1).
                    const auto *peer_v = reinterpret_cast<const float4 *>(
                        cached_peers[0].mapped_route_payload);
                    const bool root_first =
                        launch.root_participant < cached_peers[0].producer_participant;
                    for (std::uint32_t base = 0u; base < launch.top_k; base += 4u)
                    {
                        const std::uint32_t count =
                            (launch.top_k - base < 4u) ? (launch.top_k - base) : 4u;
                        float4 loc[4]{};
                        float4 par[4]{};
#pragma unroll
                        for (std::uint32_t k = 0u; k < 4u; ++k)
                        {
                            if (k < count)
                            {
                                const std::uint32_t r = base + k;
                                const std::int32_t part = slot_participants[r];
                                const std::size_t off =
                                    static_cast<std::size_t>(r) * vecs + v;
                                if (part >= 0)
                                {
                                    loc[k] = root_v[off];
                                    par[k] = moe_activation_packet_device::loadPeerPublishedFloat4(peer_v + off);
                                }
                                else if (external_in_root)
                                {
                                    loc[k] = root_v[off];
                                }
                            }
                        }
#pragma unroll
                        for (std::uint32_t k = 0u; k < 4u; ++k)
                        {
                            if (k < count)
                            {
                                const std::uint32_t r = base + k;
                                const std::int32_t part = slot_participants[r];
                                float4 c = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                                if (part >= 0)
                                {
                                    const float4 a = root_first ? loc[k] : par[k];
                                    const float4 b = root_first ? par[k] : loc[k];
                                    const float w = slot_cohort_weight[r];
                                    c.x = device_fp32_contract::multiply(
                                        device_fp32_contract::add(a.x, b.x), w);
                                    c.y = device_fp32_contract::multiply(
                                        device_fp32_contract::add(a.y, b.y), w);
                                    c.z = device_fp32_contract::multiply(
                                        device_fp32_contract::add(a.z, b.z), w);
                                    c.w = device_fp32_contract::multiply(
                                        device_fp32_contract::add(a.w, b.w), w);
                                }
                                else if (external_in_root)
                                {
                                    c = loc[k];
                                }
                                sum.x = __fadd_rn(sum.x, c.x);
                                sum.y = __fadd_rn(sum.y, c.y);
                                sum.z = __fadd_rn(sum.z, c.z);
                                sum.w = __fadd_rn(sum.w, c.w);
                            }
                        }
                    }
                }
                else
                {
#pragma unroll 1
                    for (std::uint32_t route = 0u; route < launch.top_k; ++route)
                    {
                        const std::int32_t participant = slot_participants[route];
                        const std::size_t vec_offset =
                            static_cast<std::size_t>(route) * vecs + v;
                        float4 c = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                        if (joint && participant >= 0)
                        {
                            const float4 local = root_v[vec_offset];
                            bool added_root = false;
                            bool added_operand = false;
                            for (std::uint32_t p = 0u; p < launch.peer_count; ++p)
                            {
                                const auto &peer = get_peer(p);
                                if (!added_root &&
                                    launch.root_participant < peer.producer_participant)
                                {
                                    if (added_operand)
                                    {
                                        c.x = device_fp32_contract::add(c.x, local.x);
                                        c.y = device_fp32_contract::add(c.y, local.y);
                                        c.z = device_fp32_contract::add(c.z, local.z);
                                        c.w = device_fp32_contract::add(c.w, local.w);
                                    }
                                    else
                                    {
                                        c = local;
                                    }
                                    added_root = true;
                                    added_operand = true;
                                }
                                const float4 partial = moe_activation_packet_device::loadPeerPublishedFloat4(
                                    reinterpret_cast<const float4 *>(
                                        peer.mapped_route_payload) + vec_offset);
                                if (added_operand)
                                {
                                    c.x = device_fp32_contract::add(c.x, partial.x);
                                    c.y = device_fp32_contract::add(c.y, partial.y);
                                    c.z = device_fp32_contract::add(c.z, partial.z);
                                    c.w = device_fp32_contract::add(c.w, partial.w);
                                }
                                else
                                {
                                    c = partial;
                                }
                                added_operand = true;
                            }
                            if (!added_root)
                            {
                                c.x = device_fp32_contract::add(c.x, local.x);
                                c.y = device_fp32_contract::add(c.y, local.y);
                                c.z = device_fp32_contract::add(c.z, local.z);
                                c.w = device_fp32_contract::add(c.w, local.w);
                            }
                            const float w = slot_cohort_weight[route];
                            c.x = device_fp32_contract::multiply(c.x, w);
                            c.y = device_fp32_contract::multiply(c.y, w);
                            c.z = device_fp32_contract::multiply(c.z, w);
                            c.w = device_fp32_contract::multiply(c.w, w);
                        }
                        else if (participant == launch.root_participant ||
                                 (participant < 0 && external_in_root))
                        {
                            c = root_v[vec_offset];
                        }
                        else if (slot_peer_idx[route] >= 0)
                        {
                            c = moe_activation_packet_device::loadPeerPublishedFloat4(
                                reinterpret_cast<const float4 *>(
                                    get_peer(static_cast<std::uint32_t>(slot_peer_idx[route])).mapped_route_payload) +
                                vec_offset);
                        }
                        sum.x = __fadd_rn(sum.x, c.x);
                        sum.y = __fadd_rn(sum.y, c.y);
                        sum.z = __fadd_rn(sum.z, c.z);
                        sum.w = __fadd_rn(sum.w, c.w);
                    }
                }
                if (launch.gated_shared_residual)
                {
                    const float4 shared_v =
                        reinterpret_cast<const float4 *>(launch.gated_shared_residual)[v];
                    sum.x = device_fp32_contract::add(sum.x, shared_v.x);
                    sum.y = device_fp32_contract::add(sum.y, shared_v.y);
                    sum.z = device_fp32_contract::add(sum.z, shared_v.z);
                    sum.w = device_fp32_contract::add(sum.w, shared_v.w);
                }
                reinterpret_cast<float4 *>(launch.dense_output)[v] = sum;
            }
            folded_columns = vecs * 4u;
        }
#endif
        for (std::size_t col = folded_columns + threadIdx.x; col < width; col += blockDim.x)
        {
            float sum = 0.0f;
#pragma unroll 1
            for (std::uint32_t route = 0u; route < launch.top_k; ++route)
            {
                const std::int32_t participant = slot_participants[route];
                const std::size_t element = static_cast<std::size_t>(route) * width + col;
                float contribution = 0.0f;
                if (joint && participant >= 0)
                {
                    const float local = launch.root_canonical_route_contributions[element];
                    bool added_root = false;
                    bool added_operand = false;
                    for (std::uint32_t p = 0u; p < launch.peer_count; ++p)
                    {
                        const auto &peer = get_peer(p);
                        if (!added_root &&
                            launch.root_participant < peer.producer_participant)
                        {
                            contribution = added_operand
                                ? device_fp32_contract::add(contribution, local)
                                : local;
                            added_root = true;
                            added_operand = true;
                        }
                        const float partial = loadPeerPublished(
                            peer.mapped_route_payload + element);
                        contribution = added_operand
                            ? device_fp32_contract::add(contribution, partial)
                            : partial;
                        added_operand = true;
                    }
                    if (!added_root)
                        contribution = device_fp32_contract::add(contribution, local);
                    contribution =
                        device_fp32_contract::multiply(contribution, slot_cohort_weight[route]);
                }
                else if (participant == launch.root_participant ||
                         (participant < 0 && external_in_root))
                {
                    contribution = launch.root_canonical_route_contributions[element];
                }
                else if (slot_peer_idx[route] >= 0)
                {
                    contribution = loadPeerPublished(
                        get_peer(static_cast<std::uint32_t>(slot_peer_idx[route])).mapped_route_payload + element);
                }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
                sum = __fadd_rn(sum, contribution);
#else
                sum += contribution;
#endif
            }
            if (launch.gated_shared_residual)
                sum = device_fp32_contract::add(sum, launch.gated_shared_residual[col]);
            launch.dense_output[col] = sum;
        }

        // Step 4: Release-acknowledge every peer once all threads finish reading mapped rows.
        __syncthreads();
        for (std::uint32_t p = threadIdx.x; p < launch.peer_count; p += blockDim.x)
        {
            const auto &peer = get_peer(p);
            const std::uint64_t expected_produced =
                (p < kCachedSingleRowPeers)
                    ? peer_produced[p]
                    : (loadSystemAcquire64(&peer.control->consumed_epoch) + 1u);
            __threadfence_system();
            storeSystemRelease64(
                &peer.control->consumed_epoch,
                expected_produced);
        }
    }

    /** @return Whether dense publication control matches one endpoint binding. */
    __device__ __forceinline__ bool validDensePublicationControl(
        const MoENodeLocalDensePublicationDeviceBinding &binding) noexcept
    {
        if (!binding.valid())
            return false;
        const auto *const control = binding.control;
        return loadPeerPublished(&control->magic) ==
                   kMoENodeLocalDensePublicationMagic &&
               loadPeerPublished(&control->version) ==
                   kMoENodeLocalDensePublicationVersion &&
               loadPeerPublished(&control->peer_count) ==
                   binding.peer_count &&
               loadPeerPublished(&control->element_capacity) ==
                   binding.element_capacity;
    }

    /** @brief Publish a terminal dense-channel fault and release every waiter. */
    __device__ __forceinline__ void abortDensePublication(
        MoENodeLocalDensePublicationDeviceBinding binding,
        MoENodeLocalDensePublicationCode code,
        std::uint64_t failed_epoch) noexcept
    {
        if (!binding.control)
            return;
        binding.control->code = raw(code);
        binding.control->failed_epoch = failed_epoch;
        binding.control->state =
            raw(MoENodeLocalDensePublicationState::Aborted);
        __threadfence_system();
        storeSystemRelease64(
            &binding.control->produced_epoch,
            kMoENodeLocalRouteExchangeAbortEpoch);
        if (binding.local_peer)
        {
            binding.local_peer->state =
                raw(MoENodeLocalDensePublicationState::Aborted);
            storeSystemRelease64(
                &binding.local_peer->consumed_epoch,
                kMoENodeLocalRouteExchangeAbortEpoch);
        }
    }

    /** @return Whether root or peer state makes the channel terminal. */
    __device__ __forceinline__ bool densePublicationAborted(
        const MoENodeLocalDensePublicationDeviceBinding &binding) noexcept
    {
        if (!validDensePublicationControl(binding) ||
            loadPeerPublished(&binding.control->state) !=
                raw(MoENodeLocalDensePublicationState::Ready) ||
            loadPeerPublished(&binding.control->code) !=
                raw(MoENodeLocalDensePublicationCode::Success))
        {
            return true;
        }
        return binding.local_peer &&
               loadPeerPublished(&binding.local_peer->state) !=
                   raw(MoENodeLocalDensePublicationState::Ready);
    }

    /**
     * @brief Wait until every peer has finished reading the prior dense bank.
     *
     * The root submits this one-thread node before its captured D2H payload
     * copy. It examines only cache-line-isolated progress words, so payload
     * movement remains a single copy-engine transaction rather than a mapped
     * scalar-store kernel.
     */
    static __global__ void beginDensePublicationKernel(
        MoENodeLocalDensePublicationLaunch launch)
    {
        if (blockIdx.x != 0u || threadIdx.x != 0u)
            return;
        auto binding = launch.binding;
        if (!launch.valid() || !binding.isRoot() ||
            !validDensePublicationControl(binding))
        {
            abortDensePublication(
                binding,
                MoENodeLocalDensePublicationCode::InvalidControl,
                0u);
            return;
        }

        while (true)
        {
            const std::uint64_t produced =
                loadSystemAcquire64(&binding.control->produced_epoch);
            if (produced == kMoENodeLocalRouteExchangeAbortEpoch ||
                densePublicationAborted(binding))
            {
                abortDensePublication(
                    binding,
                    MoENodeLocalDensePublicationCode::PeerAborted,
                    produced);
                return;
            }

            bool all_consumed = true;
            for (std::uint32_t peer_index = 0u;
                 peer_index < binding.peer_count;
                 ++peer_index)
            {
                const auto *const peer = binding.peers + peer_index;
                const std::uint64_t consumed =
                    loadSystemAcquire64(&peer->consumed_epoch);
                if (loadPeerPublished(&peer->participant_id) < 0 ||
                    loadPeerPublished(&peer->state) !=
                        raw(MoENodeLocalDensePublicationState::Ready) ||
                    consumed == kMoENodeLocalRouteExchangeAbortEpoch)
                {
                    abortDensePublication(
                        binding,
                        MoENodeLocalDensePublicationCode::PeerAborted,
                        produced);
                    return;
                }
                all_consumed = all_consumed && consumed == produced;
            }
            if (all_consumed)
            {
                if (produced + 1u ==
                    kMoENodeLocalRouteExchangeAbortEpoch)
                {
                    abortDensePublication(
                        binding,
                        MoENodeLocalDensePublicationCode::EpochOverflow,
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

    /** @brief Release the D2H-complete dense payload as the next root epoch. */
    static __global__ void finishDensePublicationKernel(
        MoENodeLocalDensePublicationLaunch launch)
    {
        if (blockIdx.x != 0u || threadIdx.x != 0u)
            return;
        auto binding = launch.binding;
        if (!launch.valid() || !binding.isRoot() ||
            densePublicationAborted(binding))
        {
            abortDensePublication(
                binding,
                MoENodeLocalDensePublicationCode::InvalidControl,
                0u);
            return;
        }
        const std::uint64_t produced =
            loadSystemAcquire64(&binding.control->produced_epoch);
        __threadfence_system();
        storeSystemRelease64(
            &binding.control->produced_epoch, produced + 1u);
    }

    /**
     * @brief Acquire the next root epoch before a peer's captured H2D copy.
     *
     * A peer may submit before the root. The wait remains device resident and
     * consumes no host progress thread; stream order prevents the following
     * copy-engine node from reading a partially published matrix.
     */
    static __global__ void beginDensePublicationConsumeKernel(
        MoENodeLocalDensePublicationLaunch launch)
    {
        if (blockIdx.x != 0u || threadIdx.x != 0u)
            return;
        auto binding = launch.binding;
        if (!launch.valid() || binding.isRoot() || !binding.local_peer ||
            !validDensePublicationControl(binding) ||
            loadPeerPublished(&binding.local_peer->participant_id) !=
                binding.participant_id)
        {
            abortDensePublication(
                binding,
                MoENodeLocalDensePublicationCode::InvalidControl,
                0u);
            return;
        }

        while (true)
        {
            const std::uint64_t produced =
                loadSystemAcquire64(&binding.control->produced_epoch);
            const std::uint64_t consumed =
                loadSystemAcquire64(&binding.local_peer->consumed_epoch);
            if (produced == kMoENodeLocalRouteExchangeAbortEpoch ||
                consumed == kMoENodeLocalRouteExchangeAbortEpoch ||
                densePublicationAborted(binding))
            {
                abortDensePublication(
                    binding,
                    MoENodeLocalDensePublicationCode::PeerAborted,
                    produced);
                return;
            }
            if (produced == consumed + 1u)
                return;
            if (produced < consumed || produced > consumed + 1u)
            {
                abortDensePublication(
                    binding,
                    MoENodeLocalDensePublicationCode::InvalidControl,
                    produced);
                return;
            }
#if defined(__CUDA_ARCH__)
            __nanosleep(64u);
#elif defined(__HIP_DEVICE_COMPILE__)
            __builtin_amdgcn_s_sleep(1u);
#endif
        }
    }

    /** @brief Acknowledge that the peer's H2D copy consumed the current epoch. */
    static __global__ void finishDensePublicationConsumeKernel(
        MoENodeLocalDensePublicationLaunch launch)
    {
        if (blockIdx.x != 0u || threadIdx.x != 0u)
            return;
        auto binding = launch.binding;
        if (!launch.valid() || binding.isRoot() || !binding.local_peer ||
            densePublicationAborted(binding))
        {
            abortDensePublication(
                binding,
                MoENodeLocalDensePublicationCode::InvalidControl,
                0u);
            return;
        }
        const std::uint64_t produced =
            loadSystemAcquire64(&binding.control->produced_epoch);
        __threadfence_system();
        storeSystemRelease64(
            &binding.local_peer->consumed_epoch, produced);
    }
} // namespace llaminar2::moe_node_local_route_device
