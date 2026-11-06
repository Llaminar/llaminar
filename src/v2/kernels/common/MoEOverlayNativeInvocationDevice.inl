/**
 * @file MoEOverlayNativeInvocationDevice.inl
 * @brief Original-slot native source handoff and exact post-payload receipts.
 *
 * Only the original continuation stream rearms a shared packet lane. It waits
 * for both actual post-payload endpoint markers and final bank leases before
 * installing the next immutable accepted record. Original failure is absorbing.
 * Slot snapshots retain completed traffic until the real aggregate frontier;
 * current lane markers survive later ring clearing and protect next-request reuse.
 */
#pragma once
#include "execution/moe/MoEOverlayNativeInvocationABI.h"
#include "execution/moe/MoEOverlayActivationAdmissionPublication.h"

#if defined(__CUDACC__) && !defined(__HIPCC__)
#define LLAMINAR_MOE_NATIVE_GRID_CONSTANT const __grid_constant__
#else
#define LLAMINAR_MOE_NATIVE_GRID_CONSTANT const
#endif

namespace llaminar2::moe_activation_packet_device
{
    /** @return System-acquired original absorbing state. */
    __device__ __forceinline__ std::uint32_t nativeAcquire32(const std::uint32_t* address) noexcept
    {
#if defined(__CUDA_ARCH__)
        cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> reference(*const_cast<std::uint32_t*>(address));
        return reference.load(cuda::memory_order_acquire);
#elif defined(__HIP_DEVICE_COMPILE__)
        return __hip_atomic_load(address, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
        return *address;
#endif
    }

    /** @brief Replace only the expected original lifecycle state, preserving concurrent Abort. */
    __device__ __forceinline__ bool nativeCompare32(std::uint32_t* address, std::uint32_t expected,
                                                   std::uint32_t desired) noexcept
    {
#if defined(__CUDA_ARCH__)
        cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> reference(*address);
        return reference.compare_exchange_strong(expected, desired, cuda::memory_order_acq_rel, cuda::memory_order_acquire);
#elif defined(__HIP_DEVICE_COMPILE__)
        return __hip_atomic_compare_exchange_strong(address, &expected, desired,
            __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
        if (*address != expected) return false;
        *address = desired;
        return true;
#endif
    }

    /**
     * @brief Change only the exact current publication word, never overwrite Abort.
     * @param address Same original mapped ready word.
     * @param expected Publication value authenticated before this transition.
     * @param desired Closed zero or the original ready timeline.
     * @return Whether the transition won before any concurrent cancellation.
     * The system-scoped CAS preserves the release edge on success and leaves
     * cancellation's absorbing word intact when it wins first.
     */
    __device__ __forceinline__ bool nativeCompare64(
        std::uint64_t *address, std::uint64_t expected,
        std::uint64_t desired) noexcept
    {
#if defined(__CUDA_ARCH__)
        cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> reference(*address);
        return reference.compare_exchange_strong(
            expected, desired, cuda::memory_order_acq_rel, cuda::memory_order_acquire);
#elif defined(__HIP_DEVICE_COMPILE__)
        return __hip_atomic_compare_exchange_strong(address, &expected, desired,
            __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
        if (*address != expected)
            return false;
        *address = desired;
        return true;
#endif
    }

    /** @brief Publish one original mapped 64-bit protocol word. */
    __device__ __forceinline__ void nativeRelease64(std::uint64_t* address, std::uint64_t value) noexcept
    {
        storeSystemRelease64({.signal = address, .value = value});
    }

    /** @return Whether the original channel has terminally aborted, even without a successor record. */
    __device__ __forceinline__ bool nativeLaneAborted(const MoEOverlayActivationEpochControl* control) noexcept
    {
        if (nativeAcquire32(&control->admission.state) == raw(MoEOverlayActivationAdmissionState::Failed))
            return true;
        for (std::uint32_t bank = 0; bank < kMoEOverlayActivationBufferCount; ++bank)
            if (loadSystemAcquire64(&control->buffers[bank].dispatch_signal.value) == kMoEOverlayActivationAbortTimeline ||
                loadSystemAcquire64(&control->buffers[bank].return_signal.value) == kMoEOverlayActivationAbortTimeline)
                return true;
        return false;
    }

    /** @brief Preserve failure and release every already-captured wait; never publish successful completion. */
    __device__ __forceinline__ void nativeAbortLane(MoEOverlayActivationEpochControl* control,
                                                   MoEOverlayActivationStatusCode code) noexcept
    {
        const auto current = nativeAcquire32(&control->admission.state);
        if (current != raw(MoEOverlayActivationAdmissionState::Failed))
        {
            const auto old_code = nativeAcquire32(&control->admission.code);
            if (old_code == raw(MoEOverlayActivationStatusCode::Idle) || old_code == raw(MoEOverlayActivationStatusCode::Success))
                (void)nativeCompare32(&control->admission.code, old_code, raw(code));
            (void)nativeCompare32(&control->admission.state, current, raw(MoEOverlayActivationAdmissionState::Failed));
        }
        for (std::uint32_t bank = 0; bank < kMoEOverlayActivationBufferCount; ++bank)
        {
            nativeRelease64(&control->buffers[bank].dispatch_signal.value, kMoEOverlayActivationAbortTimeline);
            nativeRelease64(&control->buffers[bank].return_signal.value, kMoEOverlayActivationAbortTimeline);
        }
        nativeRelease64(&control->admission.ready_signal, kMoEOverlayActivationAbortTimeline);
    }

    /** @return Both actual post-payload owners and all final original bank leases have completed. */
    __device__ __forceinline__ bool nativePreviousPayloadRetired(
        const MoEOverlayActivationEpochControl* control, std::uint64_t generation) noexcept
    {
        if (generation == 0)
            return loadSystemAcquire64(&control->identity.epoch_generation) == 0;
        if (loadSystemAcquire64(&control->identity.epoch_generation) != generation ||
            loadSystemAcquire64(&control->continuation_status.payload_retired_generation) != generation ||
            loadSystemAcquire64(&control->follower_status.payload_retired_generation) != generation)
            return false;
        const auto stages = loadPeerPublished(&control->identity.stage_count);
        if (stages == 0) return false;
        for (std::uint32_t bank = 0; bank < kMoEOverlayActivationBufferCount; ++bank)
        {
            if (bank >= stages) continue;
            auto last = stages - 1;
            if (moeOverlayActivationBufferIndex(last) != bank) --last;
            const auto expected = moeOverlayActivationLeasedTimelineValue(moeOverlayActivationBufferVisit(last));
            if (loadSystemAcquire64(&control->buffers[bank].dispatch_signal.value) != expected ||
                loadSystemAcquire64(&control->buffers[bank].return_signal.value) != expected)
                return false;
        }
        return true;
    }

    /** @brief Initialize only actual protocol fields after the previous two owners retired. */
    __device__ __forceinline__ void nativeArmEndpoint(MoEOverlayActivationEndpointStatus& status,
                                                      const MoEOverlayActivationEpochIdentity& identity) noexcept
    {
        status.digest = identity.digest;
        status.generation = identity.epoch_generation;
        status.observed_timeline = 0;
        status.state = raw(MoEOverlayActivationEndpointState::Armed);
        status.code = raw(MoEOverlayActivationStatusCode::Success);
        status.operation = raw(MoEOverlayActivationOperation::None);
        status.last_published_stage = -1;
        status.last_consumed_stage = -1;
        status.last_model_layer = -1;
        status.failure_diagnostic = 0;
        status.failure_auxiliary = 0;
        status.published_payload_bytes = 0;
        status.published_live_rows = 0;
        status.published_live_entries = 0;
        status.published_stage_count = 0;
        status.published_shared_hidden_bytes = 0;
        status.payload_retired_generation = 0;
    }

    /**
     * @brief Reuse the SAME retained native graph/lane for its next original accepted slot.
     * All threads cooperate only in the immutable identity copy. One original
     * continuation writer owns reset/arm; no host state query or extra ring exists.
     */
    static __global__ void beginOriginalNativeInvocationKernel(LLAMINAR_MOE_NATIVE_GRID_CONSTANT MoEOverlayOriginalNativeBeginLaunch launch)
    {
        __shared__ std::uint32_t selected;
        __shared__ bool accepted;
        __shared__ alignas(MoEOverlayActivationEpochIdentity) unsigned char identity_storage[sizeof(MoEOverlayActivationEpochIdentity)];
        auto &identity = *reinterpret_cast<MoEOverlayActivationEpochIdentity *>(identity_storage);
        if (threadIdx.x == 0)
        {
            selected = kMoEOverlayNativeInvocationNoSlot;
            accepted = false;
            if (launch.valid() && !nativeLaneAborted(launch.control))
            {
                const auto previous = launch.grant->generation;
                std::uint64_t next_generation = kMoEOverlayActivationMaxGeneration + 1;
                for (std::uint32_t index = 0; index < launch.original_invocations.original_slot_count; ++index)
                {
                    const auto& slot = launch.original_invocations.original_slots[index];
                    const auto generation = loadSystemAcquire64(&slot.published_generation);
                    if (generation == 0 || generation <= previous || generation >= next_generation) continue;
                    if (loadPeerPublished(&slot.identity.channel_nonce) != launch.control->channel.channel_nonce) continue;
                    const auto rows = std::uint64_t(loadPeerPublished(&slot.identity.request_count)) *
                                      loadPeerPublished(&slot.identity.physical_rows_per_request);
                    if (rows != std::uint64_t(launch.physical_rows)) continue;
                    next_generation = generation;
                    selected = index;
                }
                accepted = selected != kMoEOverlayNativeInvocationNoSlot;
            }
            if (!accepted && launch.control)
                nativeAbortLane(launch.control, MoEOverlayActivationStatusCode::InvalidIdentity);
        }
        __syncthreads();
        if (!accepted) return;
        const auto* record = launch.original_invocations.original_slots + selected;
        const auto* source = reinterpret_cast<const std::uint64_t*>(&record->identity);
        auto* destination = reinterpret_cast<std::uint64_t*>(&identity);
        for (std::uint32_t word = threadIdx.x; word < sizeof(identity) / sizeof(std::uint64_t); word += blockDim.x)
            destination[word] = loadPeerPublished(source + word);
        __syncthreads();
        if (threadIdx.x == 0)
        {
            const auto generation = loadSystemAcquire64(&record->published_generation);
            accepted = record->abi_version == kMoEOverlayNativeInvocationABIVersion && record->reserved == 0 &&
                record->original_slot_index == selected && identity.original_slot_index == selected &&
                identity.stage_count == launch.retained_stage_count && record->timeout_not_before_ns != 0 &&
                loadSystemAcquire64(&record->cancelled_generation) == 0 &&
                moeOverlayNativeInvocationAdmissionMatches(identity, generation, launch.control->channel, launch.physical_rows);
            while (accepted && !nativePreviousPayloadRetired(launch.control, launch.grant->generation))
            {
                if (nativeLaneAborted(launch.control) || loadSystemAcquire64(&record->cancelled_generation) != 0)
                {
                    accepted = false;
                    break;
                }
#if defined(__CUDA_ARCH__)
                __nanosleep(64u);
#elif defined(__HIP_DEVICE_COMPILE__)
                __builtin_amdgcn_s_sleep(1u);
#endif
            }
            if (accepted)
            {
                const auto state = nativeAcquire32(&launch.control->admission.state);
                accepted = moeOverlayNativeInvocationRearmStateAllowed(state) &&
                    nativeCompare32(&launch.control->admission.state, state, raw(MoEOverlayActivationAdmissionState::Idle));
            }
            if (accepted)
            {
                accepted = closeMoEOverlayActivationAdmissionPublication(
                    [&]() { return loadSystemAcquire64(&launch.control->admission.ready_signal); },
                    [&](std::uint64_t expected, std::uint64_t desired) {
                        return nativeCompare64(&launch.control->admission.ready_signal, expected, desired);
                    });
            }
            if (accepted)
            {
                for (auto& bank : launch.control->buffers)
                {
                    bank.dispatch_descriptor = {};
                    bank.return_descriptor = {};
                    nativeRelease64(&bank.dispatch_signal.value, 0);
                    nativeRelease64(&bank.return_signal.value, 0);
                }
            }
            else
                nativeAbortLane(launch.control, MoEOverlayActivationStatusCode::ExplicitAbort);
        }
        __syncthreads();
        if (!accepted) return;
        auto* control_words = reinterpret_cast<std::uint64_t*>(&launch.control->identity);
        const auto* identity_words = reinterpret_cast<const std::uint64_t*>(&identity);
        for (std::uint32_t word = threadIdx.x; word < sizeof(identity) / sizeof(std::uint64_t); word += blockDim.x)
            control_words[word] = identity_words[word];
        __syncthreads();
        if (threadIdx.x == 0)
        {
            nativeArmEndpoint(launch.control->continuation_status, identity);
            nativeArmEndpoint(launch.control->follower_status, identity);
            launch.control->admission.timeout_not_before_ns = record->timeout_not_before_ns;
            launch.control->admission.observed_timeout_ns = 0;
            launch.control->admission.digest = identity.digest;
            nativeRelease64(&launch.control->admission.last_generation, identity.epoch_generation);
            const auto code = nativeAcquire32(&launch.control->admission.code);
            if ((code != raw(MoEOverlayActivationStatusCode::Idle) && code != raw(MoEOverlayActivationStatusCode::Success)) ||
                !nativeCompare32(&launch.control->admission.code, code, raw(MoEOverlayActivationStatusCode::Success)) ||
                loadSystemAcquire64(&record->cancelled_generation) != 0 ||
                !nativeCompare32(&launch.control->admission.state, raw(MoEOverlayActivationAdmissionState::Idle),
                                 raw(MoEOverlayActivationAdmissionState::Armed)))
                nativeAbortLane(launch.control, MoEOverlayActivationStatusCode::ExplicitAbort);
            else if (!publishMoEOverlayActivationAdmissionPublication(
                [&](std::uint64_t expected, std::uint64_t desired) {
                    return nativeCompare64(&launch.control->admission.ready_signal, expected, desired);
                }))
            {
                nativeAbortLane(launch.control, MoEOverlayActivationStatusCode::ExplicitAbort);
            }
        }
    }

    /** @brief Preserve the exact completed endpoint status without copying unused padding. */
    __device__ __forceinline__ void copyNativeEndpointStatus(MoEOverlayActivationEndpointStatus& destination,
        const MoEOverlayActivationEndpointStatus& source, std::uint64_t generation) noexcept
    {
        destination.digest = source.digest;
        destination.generation = source.generation;
        destination.observed_timeline = source.observed_timeline;
        destination.state = source.state;
        destination.code = source.code;
        destination.operation = source.operation;
        destination.last_published_stage = source.last_published_stage;
        destination.last_consumed_stage = source.last_consumed_stage;
        destination.last_model_layer = source.last_model_layer;
        destination.failure_diagnostic = source.failure_diagnostic;
        destination.failure_auxiliary = source.failure_auxiliary;
        destination.published_payload_bytes = source.published_payload_bytes;
        destination.published_live_rows = source.published_live_rows;
        destination.published_live_entries = source.published_live_entries;
        destination.published_stage_count = source.published_stage_count;
        destination.published_shared_hidden_bytes = source.published_shared_hidden_bytes;
        destination.payload_retired_generation = generation;
    }

    /** @brief One final, original-stream publication AFTER every payload operation and signal. */
    __device__ __forceinline__ void retireOriginalNativeInvocation(const MoEOverlayOriginalNativeRetireLaunch& launch) noexcept
    {
        // Mixed source batches retain their original host-ordered and non-final
        // descriptors. Only a selected native terminal owes this publication.
        if (!launch.original_invocations.deviceOwned() ||
            launch.stage_ordinal + 1u != launch.retained_stage_count)
        {
            return;
        }
        if (!launch.control)
        {
            return; // Host launch validation rejects an absent channel.
        }
        if (nativeLaneAborted(launch.control))
        {
            return; // Physical graph drain never fabricates successful receipt.
        }
        if (!launch.valid())
        {
            nativeAbortLane(launch.control, MoEOverlayActivationStatusCode::InvalidIdentity);
            return;
        }
        const auto index = launch.grant->original_slot_index;
        if (index >= launch.original_invocations.original_slot_count ||
            launch.grant->state != raw(MoEOverlayActivationEndpointState::Complete) ||
            launch.grant->code != raw(MoEOverlayActivationStatusCode::Success))
        {
            nativeAbortLane(launch.control, MoEOverlayActivationStatusCode::InvalidIdentity);
            return;
        }
        auto &record = launch.original_invocations.original_slots[index];
        const auto generation = launch.grant->generation;
        if (loadSystemAcquire64(&record.published_generation) != generation ||
            loadSystemAcquire64(&record.cancelled_generation) != 0 ||
            record.identity.original_slot_index != index ||
            !sameDigest(record.identity.digest, launch.grant->digest) ||
            launch.control->identity.epoch_generation != generation)
        {
            nativeAbortLane(launch.control, MoEOverlayActivationStatusCode::InvalidIdentity);
            return;
        }
        auto *status = statusFor(launch.control, launch.endpoint);
        if (status->generation != generation ||
            !sameDigest(status->digest, launch.grant->digest) ||
            status->state != raw(MoEOverlayActivationEndpointState::Complete) ||
            status->code != raw(MoEOverlayActivationStatusCode::Success) ||
            status->last_published_stage != static_cast<std::int32_t>(launch.stage_ordinal) ||
            status->last_consumed_stage != static_cast<std::int32_t>(launch.stage_ordinal) ||
            status->published_stage_count != launch.retained_stage_count)
        {
            nativeAbortLane(launch.control, MoEOverlayActivationStatusCode::InvalidIdentity);
            return;
        }
        auto& receipt = launch.endpoint == MoEOverlayActivationEndpoint::Continuation ? record.continuation : record.follower;
        copyNativeEndpointStatus(receipt.status, *status, generation);
        nativeRelease64(&receipt.retired_generation, generation);
        // This last write is the only permission for the next native rearm.
        // The previous immutable slot may later be cleared at aggregate retirement.
        nativeRelease64(&status->payload_retired_generation, generation);
    }

    /** @brief Native single-endpoint post-payload epilogue. */
    static __global__ void retireOriginalNativeInvocationKernel(LLAMINAR_MOE_NATIVE_GRID_CONSTANT MoEOverlayOriginalNativeRetireLaunch launch)
    {
        if (threadIdx.x == 0) retireOriginalNativeInvocation(launch);
    }

    /** @brief Return the original source packet from the existing typed batch descriptor. */
    template<class Packet> __device__ __forceinline__ const auto& nativeReturnPacket(const Packet& packet) noexcept
    {
        if constexpr (requires { packet.packet; }) return packet.packet;
        else return packet;
    }

    /** @brief Actual lane count, with one protocol-only block per final source lane. */
    template<class Batch> static __global__ void retireOriginalNativeInvocationBatchKernel(LLAMINAR_MOE_NATIVE_GRID_CONSTANT Batch batch)
    {
        if (threadIdx.x != 0 || blockIdx.x >= batch.lane_count) return;
        const auto& packet = nativeReturnPacket(batch.lanes[blockIdx.x]);
        retireOriginalNativeInvocation({.original_invocations = packet.original_invocations,
            .control = const_cast<MoEOverlayActivationEpochControl*>(packet.control), .grant = packet.grant,
            .endpoint = MoEOverlayActivationEndpoint::Continuation, .physical_rows = packet.physical_rows,
            .stage_ordinal = packet.stage_ordinal, .retained_stage_count = packet.retained_stage_count});
    }
} // namespace llaminar2::moe_activation_packet_device

#undef LLAMINAR_MOE_NATIVE_GRID_CONSTANT
