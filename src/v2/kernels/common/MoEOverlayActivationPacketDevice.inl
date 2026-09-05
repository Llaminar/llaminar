/**
 * @file MoEOverlayActivationPacketDevice.inl
 * @brief Shared CUDA/HIP kernels for device-owned sparse activation packets.
 *
 * This implementation is included by one CUDA and one HIP bridge translation
 * unit.  It contains no backend-role assumptions: every launch receives the
 * exact continuation/follower lane and logical participant chosen by planning.
 * Packet metadata uses a deterministic block scan that preserves original
 * row/router-slot order without serializing a whole prefill packet onto one GPU
 * thread. Bulk FP32 movement is parallel and remains ordered by the caller's
 * exact stream and TransferEngine timeline publication.
 */

#pragma once

#include "execution/moe/MoEOverlayActivationPacketABI.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#include <cuda/atomic>
#endif

namespace llaminar2::moe_activation_packet_device
{
    /** Threads in the portable CUDA/HIP deterministic metadata compactor. */
    inline constexpr std::uint32_t kDispatchMetadataBlockThreads = 256u;

    /**
     * @brief Load one peer-published scalar without accepting a cached line.
     *
     * Node-local heterogeneous participants do not share one GPU L2. The
     * timeline acquire orders peer writes, while these backend cache-volatile
     * loads force the payload read itself to observe system memory. Ordinary
     * volatile C++ loads are insufficient because they constrain compiler
     * motion but do not select the required GPU cache policy.
     */
    template <typename Value>
    __device__ __forceinline__ Value loadPeerPublished(
        const Value *address) noexcept
    {
#if defined(__CUDA_ARCH__)
        return __ldcv(address);
#elif defined(__HIP_DEVICE_COMPILE__)
        return __builtin_nontemporal_load(address);
#else
        return *address;
#endif
    }

    /**
     * @brief Acquire one peer-owned publication word at system scope.
     *
     * A retained timeline wait gates graph scheduling, but a captured copy-
     * engine node may sit between that wait and the packet validation kernel.
     * Re-acquiring the exact publication word here carries peer descriptor and
     * metadata visibility into this compute kernel across that engine boundary.
     */
    __device__ __forceinline__ std::uint64_t loadSystemAcquire64(
        const std::uint64_t *address) noexcept
    {
#if defined(__CUDA_ARCH__)
        cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> reference(
            *const_cast<std::uint64_t *>(address));
        return reference.load(cuda::memory_order_acquire);
#elif defined(__HIP_DEVICE_COMPILE__)
        return __hip_atomic_load(
            address,
            __ATOMIC_ACQUIRE,
            __HIP_MEMORY_SCOPE_SYSTEM);
#else
        return *address;
#endif
    }

    /** @brief Device-side unsigned-GEQ wait for a TransferEngine-bound edge. */
    __device__ __forceinline__ void waitSystemAcquire64(
        const MoEOverlayActivationTimelineWaitDeviceBinding &binding) noexcept
    {
        while (loadSystemAcquire64(binding.signal) < binding.value)
        {
#if defined(__CUDA_ARCH__)
            __nanosleep(64u);
#elif defined(__HIP_DEVICE_COMPILE__)
            __builtin_amdgcn_s_sleep(1u);
#endif
        }
    }

    /** @brief Publish a TransferEngine-bound edge with system-release ordering. */
    __device__ __forceinline__ void storeSystemRelease64(
        const MoEOverlayActivationTimelinePublishDeviceBinding &binding) noexcept
    {
#if defined(__CUDA_ARCH__)
        cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> reference(
            *binding.signal);
        reference.store(binding.value, cuda::memory_order_release);
#elif defined(__HIP_DEVICE_COMPILE__)
        __hip_atomic_store(
            binding.signal,
            binding.value,
            __ATOMIC_RELEASE,
            __HIP_MEMORY_SCOPE_SYSTEM);
#else
        *binding.signal = binding.value;
#endif
    }

    /**
     * @brief Snapshot a naturally aligned protocol record with fresh loads.
     *
     * Protocol records are fixed-width, 64-byte-aligned ABI objects. Loading
     * each 64-bit lane through @ref loadPeerPublished avoids mixing fields from
     * stale and current cache lines before validation computes the digest.
     */
    template <typename Record>
    __device__ __forceinline__ Record snapshotPeerPublished(
        const Record *record) noexcept
    {
        static_assert(sizeof(Record) % sizeof(std::uint64_t) == 0u);
        Record snapshot{};
        const auto *source = reinterpret_cast<const std::uint64_t *>(record);
        auto *destination = reinterpret_cast<std::uint64_t *>(&snapshot);
        for (std::size_t lane = 0u;
             lane < sizeof(Record) / sizeof(std::uint64_t);
             ++lane)
        {
            destination[lane] = loadPeerPublished(source + lane);
        }
        return snapshot;
    }

    /** @return Raw fixed-width value for one strongly typed protocol enum. */
    template <typename Enum>
    __device__ __forceinline__ std::uint32_t raw(Enum value) noexcept
    {
        return static_cast<std::uint32_t>(value);
    }

    /** @return Whether two digest lanes are exactly equal without host operators. */
    __device__ __forceinline__ bool sameDigest(
        const MoEOverlayActivationDigest &left,
        const MoEOverlayActivationDigest &right) noexcept
    {
        return left.low == right.low && left.high == right.high;
    }

    /**
     * @brief Pack the exact placement-acquire result into terminal evidence.
     *
     * Activation endpoint status already carries one 64-bit diagnostic word.
     * On a stage-zero placement failure it records the last completed epoch
     * operation, the request ticket visible to the failing packet, and the
     * epoch currently published in that ticket's selected placement bank. The
     * three 16-bit epoch lanes are diagnostic modulo counters; operation and
     * result retain their full ABI widths. This remains entirely device-owned
     * and lets the bounded host watchdog distinguish a missing acquire, a
     * prematurely reused placement bank, and an ordinary semantic rejection
     * without adding a synchronization or another mapped diagnostic field.
     */
    __device__ __forceinline__ std::uint64_t placementAcquireEvidence(
        MoEOverlayRoutePlacementDeviceBinding placement) noexcept
    {
        if (!placement.status)
            return 0u;
        const DeviceMoEOverlayEpochStatus status = *placement.status;
        const std::uint64_t ticket_epoch =
            placement.ticket ? placement.ticket->epoch : 0u;
        const std::uint32_t ticket_bank =
            placement.ticket
                ? static_cast<std::uint32_t>(
                      placement.ticket->selector & 1u)
                : kDeviceMoEOverlayInvalidBank;
        const std::uint64_t selected_bank_epoch =
            ticket_bank < kDeviceMoEOverlayEpochBankCount &&
                    placement.banks[ticket_bank].epoch
                ? static_cast<std::uint64_t>(
                      *placement.banks[ticket_bank].epoch)
                : 0u;
        return ((status.epoch & 0xffffull) << 48u) |
               ((ticket_epoch & 0xffffull) << 32u) |
               ((selected_bank_epoch & 0xffffull) << 16u) |
               (static_cast<std::uint64_t>(status.operation & 0xffu) << 8u) |
               static_cast<std::uint64_t>(status.code & 0xffu);
    }

    /** @return Expected physical rows encoded by the immutable admission ticket. */
    __device__ __forceinline__ std::uint64_t admittedPhysicalRows(
        const MoEOverlayActivationEpochIdentity &identity) noexcept
    {
        return static_cast<std::uint64_t>(identity.request_count) *
               static_cast<std::uint64_t>(
                   identity.physical_rows_per_request);
    }

    /** @return Status record exclusively owned by one protocol endpoint. */
    __device__ __forceinline__ MoEOverlayActivationEndpointStatus *statusFor(
        MoEOverlayActivationEpochControl *control,
        MoEOverlayActivationEndpoint endpoint) noexcept
    {
        return endpoint == MoEOverlayActivationEndpoint::Continuation
                   ? &control->continuation_status
                   : &control->follower_status;
    }

    /** @brief Publish one endpoint-owned successful operation. */
    __device__ __forceinline__ void publishSuccess(
        MoEOverlayActivationEpochControl *control,
        MoEOverlayActivationDeviceEpochGrant *grant,
        MoEOverlayActivationEndpoint endpoint,
        MoEOverlayActivationOperation operation,
        std::uint32_t stage_ordinal,
        std::int32_t model_layer_index,
        std::uint64_t timeline,
        bool consumed,
        bool published,
        std::uint64_t payload_bytes = 0u,
        std::uint64_t live_rows = 0u,
        std::uint64_t live_entries = 0u) noexcept
    {
        if (!control || !grant)
            return;
        grant->code = raw(MoEOverlayActivationStatusCode::Success);
        if (consumed)
            grant->last_consumed_stage =
                static_cast<std::int32_t>(stage_ordinal);
        if (published)
            grant->last_published_stage =
                static_cast<std::int32_t>(stage_ordinal);

        /*
         * Payload geometry is bounded by the authenticated graph family. Each
         * endpoint is the sole writer and retained stages are stream-ordered,
         * so the hot counters need neither atomics nor mapped-memory traffic.
         */
        if (published &&
            (operation == MoEOverlayActivationOperation::PublishDispatch ||
             operation == MoEOverlayActivationOperation::PublishReturn))
        {
            grant->published_payload_bytes += payload_bytes;
            grant->published_live_rows += live_rows;
            grant->published_live_entries += live_entries;
            ++grant->published_stage_count;
        }

        if (stage_ordinal + 1u != grant->stage_count || !consumed ||
            !published)
        {
            return;
        }

        /* Only the terminal stage exports device-local evidence. The
         * scheduler acquires this Complete state after the graph's terminal
         * event, so it can read the preceding counters without a host shadow. */
        grant->state = raw(MoEOverlayActivationEndpointState::Complete);
        auto *status = statusFor(control, endpoint);
        status->digest = grant->digest;
        status->generation = grant->generation;
        status->observed_timeline = timeline;
        status->operation = raw(operation);
        status->code = raw(MoEOverlayActivationStatusCode::Success);
        status->last_model_layer = model_layer_index;
        status->last_consumed_stage = grant->last_consumed_stage;
        status->last_published_stage = grant->last_published_stage;
        status->published_payload_bytes = grant->published_payload_bytes;
        status->published_live_rows = grant->published_live_rows;
        status->published_live_entries = grant->published_live_entries;
        status->published_stage_count = grant->published_stage_count;
        __threadfence_system();
        status->state = raw(MoEOverlayActivationEndpointState::Complete);
    }

    /** @brief Publish a terminal semantic failure without trapping peer streams. */
    __device__ __forceinline__ void publishFailure(
        MoEOverlayActivationEpochControl *control,
        MoEOverlayActivationDeviceEpochGrant *grant,
        MoEOverlayActivationEndpoint endpoint,
        MoEOverlayActivationOperation operation,
        MoEOverlayActivationStatusCode code,
        std::uint32_t stage_ordinal,
        std::int32_t model_layer_index,
        std::uint64_t observed_timeline,
        std::uint32_t failure_diagnostic = 0u,
        std::uint32_t failure_auxiliary = 0u) noexcept
    {
        /* A retained graph must drain after a terminal packet fault so its
         * unconditional timeline nodes can release the peer. Preserve the
         * first semantic failure while those later stages no-op: overwriting
         * it with the final cascading rejection hides the actual broken edge. */
        if (grant &&
            grant->state ==
                raw(MoEOverlayActivationEndpointState::Aborted))
        {
            return;
        }
        if (grant)
        {
            grant->state = raw(MoEOverlayActivationEndpointState::Aborted);
            grant->code = raw(code);
        }
        auto *status = statusFor(control, endpoint);
        const auto identity = snapshotPeerPublished(&control->identity);
        status->digest = identity.digest;
        status->generation = identity.epoch_generation;
        status->observed_timeline = observed_timeline;
        status->operation = raw(operation);
        status->code = raw(code);
        status->last_model_layer = model_layer_index;
        status->failure_diagnostic = failure_diagnostic;
        status->failure_auxiliary = failure_auxiliary;
        status->last_consumed_stage =
            grant ? grant->last_consumed_stage : -1;
        status->last_published_stage =
            grant ? grant->last_published_stage : -1;
        status->published_payload_bytes =
            grant ? grant->published_payload_bytes : 0u;
        status->published_live_rows =
            grant ? grant->published_live_rows : 0u;
        status->published_live_entries =
            grant ? grant->published_live_entries : 0u;
        status->published_stage_count =
            grant ? grant->published_stage_count : 0u;
        __threadfence_system();
        status->state = raw(MoEOverlayActivationEndpointState::Aborted);
    }

    __device__ __forceinline__ bool validControl(
        const MoEOverlayActivationEpochControl *control,
        MoEOverlayActivationEndpoint endpoint,
        std::uint32_t stage_ordinal,
        std::int32_t physical_rows,
        std::int32_t expected_target_participant) noexcept;

    /**
     * @brief Capture every stage-zero continuation admission predicate.
     *
     * A set bit means the named predicate succeeded.  The mask is emitted only
     * after admission rejects the retained packet, so these redundant reads do
     * not add traffic to successful inference.  Bits 0 through 16 cover the
     * complete placement/grant expression in @ref prepareDispatchMetadata;
     * bits 17 through 20 distinguish the mapped-control identity edge that is
     * otherwise hidden inside @ref acquireOrValidateGrant.
     */
    __device__ __forceinline__ std::uint32_t dispatchAdmissionDiagnostic(
        MoEOverlayActivationDispatchPackLaunch launch,
        bool grant_acquired) noexcept
    {
        const DeviceMoEOverlayEpochTicket ticket = launch.placement.ticket
            ? *launch.placement.ticket
            : DeviceMoEOverlayEpochTicket{};
        const std::uint32_t bank =
            static_cast<std::uint32_t>(ticket.selector & 1u);
        const std::uint64_t selector_generation = ticket.selector >> 1u;
        const bool bank_valid =
            bank < kDeviceMoEOverlayEpochBankCount;
        const auto selected = bank_valid
            ? launch.placement.banks[bank]
            : MoEOverlayRoutePlacementBankDeviceView{};
        const auto identity = launch.control
            ? snapshotPeerPublished(&launch.control->identity)
            : MoEOverlayActivationEpochIdentity{};
        const auto endpoint_status = launch.control
            ? snapshotPeerPublished(
                  &launch.control->continuation_status)
            : MoEOverlayActivationEndpointStatus{};
        const auto endpoint_state =
            static_cast<MoEOverlayActivationEndpointState>(
                endpoint_status.state);

        std::uint32_t mask = 0u;
        const auto record = [&](std::uint32_t bit, bool passed)
        {
            if (passed)
                mask |= std::uint32_t{1} << bit;
        };
        record(0u, grant_acquired);
        record(1u, launch.placement.valid());
        record(2u, launch.grant && launch.grant->digest.valid());
        record(3u, launch.grant && launch.grant->generation != 0u);
        record(4u, launch.grant && launch.grant->placement_epoch != 0u);
        record(
            5u,
            launch.grant &&
                launch.grant->state ==
                    raw(MoEOverlayActivationEndpointState::Active));
        record(
            6u,
            launch.grant &&
                launch.grant->code ==
                    raw(MoEOverlayActivationStatusCode::Success));
        record(
            7u,
            launch.grant &&
                launch.grant->stage_count > launch.stage_ordinal);
        record(
            8u,
            launch.grant &&
                launch.grant->physical_rows == launch.physical_rows);
        record(
            9u,
            launch.grant &&
                launch.grant->endpoint ==
                    raw(MoEOverlayActivationEndpoint::Continuation));
        record(
            10u,
            launch.grant &&
                launch.grant->graph_role !=
                    MoEOverlayInferenceGraphRole::None);
        record(11u, ticket.epoch != 0u);
        record(12u, selector_generation != 0u);
        record(
            13u,
            launch.grant &&
                ticket.epoch == launch.grant->placement_epoch);
        record(14u, bank_valid);
        record(15u, selected.valid());
        record(
            16u,
            selected.valid() &&
                static_cast<std::uint64_t>(*selected.epoch) == ticket.epoch);
        record(
            17u,
            endpoint_state == MoEOverlayActivationEndpointState::Armed ||
                endpoint_state == MoEOverlayActivationEndpointState::Active);
        record(
            18u,
            sameDigest(endpoint_status.digest, identity.digest));
        record(
            19u,
            endpoint_status.generation == identity.epoch_generation);
        record(
            20u,
            validControl(
                launch.control,
                MoEOverlayActivationEndpoint::Continuation,
                launch.stage_ordinal,
                launch.physical_rows,
                launch.target_participant_id));
        return mask;
    }

    /** @return Whether immutable mapped control bytes authenticate this graph. */
    __device__ __forceinline__ bool validControl(
        const MoEOverlayActivationEpochControl *control,
        MoEOverlayActivationEndpoint endpoint,
        std::uint32_t stage_ordinal,
        std::int32_t physical_rows,
        std::int32_t expected_target_participant = -1) noexcept
    {
        if (!control || physical_rows <= 0)
            return false;
        const auto channel = snapshotPeerPublished(&control->channel);
        const auto identity = snapshotPeerPublished(&control->identity);
        const std::uint64_t admission_timeline =
            loadSystemAcquire64(&control->admission.ready_signal);
        const auto admission = snapshotPeerPublished(&control->admission);
        if (channel.magic != kMoEOverlayActivationMagic ||
            channel.abi_version != kMoEOverlayActivationABIVersion ||
            channel.channel_nonce == 0u ||
            channel.buffer_count != kMoEOverlayActivationBufferCount ||
            channel.stage_count == 0u ||
            stage_ordinal >= channel.stage_count ||
            admission_timeline !=
                kMoEOverlayActivationAdmissionTimeline ||
            admission.state != raw(MoEOverlayActivationAdmissionState::Armed) ||
            identity.epoch_generation == 0u ||
            identity.stage_count != channel.stage_count ||
            identity.lane_ordinal != channel.lane_ordinal ||
            identity.channel_nonce != channel.channel_nonce ||
            identity.workspace_generation != channel.workspace_generation ||
            identity.stage_manifest_digest != channel.stage_manifest_digest ||
            identity.source_world_rank != channel.source_world_rank ||
            identity.target_world_rank != channel.target_world_rank ||
            identity.source_participant_id != channel.source_participant_id ||
            identity.target_participant_id != channel.target_participant_id ||
            identity.source_tier_priority != channel.source_tier_priority ||
            identity.target_tier_priority != channel.target_tier_priority ||
            identity.source_domain_ordinal != channel.source_domain_ordinal ||
            identity.target_domain_ordinal != channel.target_domain_ordinal ||
            identity.graph_role >= 32u ||
            (channel.graph_role_mask &
             (std::uint32_t{1} << identity.graph_role)) == 0u ||
            admittedPhysicalRows(identity) !=
                static_cast<std::uint64_t>(physical_rows) ||
            (expected_target_participant >= 0 &&
             channel.target_participant_id != expected_target_participant) ||
            !sameDigest(identity.digest, admission.digest) ||
            !sameDigest(
                identity.digest,
                moeOverlayActivationIdentityDigest(identity)))
        {
            return false;
        }

        const auto status = snapshotPeerPublished(
            endpoint == MoEOverlayActivationEndpoint::Continuation
                ? &control->continuation_status
                : &control->follower_status);
        const auto state = static_cast<MoEOverlayActivationEndpointState>(
            status.state);
        return sameDigest(status.digest, identity.digest) &&
               status.generation == identity.epoch_generation &&
               (state == MoEOverlayActivationEndpointState::Armed ||
                state == MoEOverlayActivationEndpointState::Active);
    }

    /**
     * @brief Acquire mapped admission once, then validate only local epoch state.
     *
     * Stage zero is ordered after the scheduler's mapped admission signal and
     * authenticates the complete channel/identity/status record. Every later
     * stage is already part of that same immutable retained executable, so it
     * checks the endpoint-private grant instead of traversing the PCIe mapping.
     */
    __device__ __forceinline__ bool acquireOrValidateGrant(
        MoEOverlayActivationEpochControl *control,
        MoEOverlayActivationDeviceEpochGrant *grant,
        MoEOverlayActivationEndpoint endpoint,
        MoEOverlayActivationOperation operation,
        std::uint32_t stage_ordinal,
        std::int32_t physical_rows,
        std::int32_t expected_target_participant = -1,
        MoEOverlayRoutePlacementDeviceBinding placement = {}) noexcept
    {
        if (!control || !grant || physical_rows <= 0)
            return false;
        const bool begins_epoch =
            operation == MoEOverlayActivationOperation::PublishDispatch ||
            operation == MoEOverlayActivationOperation::ConsumeDispatch;
        if (stage_ordinal == 0u && begins_epoch)
        {
            /*
             * The capture-stable admission wait uses a reusable non-zero
             * threshold. It can therefore pass on transaction N's still-live
             * publication when transaction N+1 has already been submitted by
             * the continuation host. The endpoint-private grant is the device
             * anti-ABA authority: an epoch-opening operation must observe a
             * fully authenticated scheduler identity strictly newer than the
             * generation it last consumed. Later stage-zero operations reuse
             * the active grant and never enter this loop.
             *
             * This wait is normally one iteration. Under intentional host
             * submission overlap it suspends only this endpoint's graph block;
             * the peer device retires N and publishes N+1 independently. No
             * host fence, graph mutation, or mutable execution-state mirror is
             * introduced.
             */
            const std::uint64_t previous_generation = grant->generation;
            MoEOverlayActivationEpochIdentity identity{};
            for (;;)
            {
                if (validControl(
                        control,
                        endpoint,
                        stage_ordinal,
                        physical_rows,
                        expected_target_participant))
                {
                    identity = snapshotPeerPublished(&control->identity);
                    if (identity.epoch_generation > previous_generation)
                        break;
                }
#if defined(__CUDA_ARCH__)
                __nanosleep(64u);
#elif defined(__HIP_DEVICE_COMPILE__)
                __builtin_amdgcn_s_sleep(1u);
#endif
            }
            if (!placement.valid())
                return false;

            /*
             * The host ticket carries only a lower bound because background
             * device maintenance may publish after scheduler admission. Read
             * the endpoint's request ticket on its exact graph stream and
             * authenticate the selected durable bank before making that exact
             * epoch part of the private grant. Continuation and follower use
             * the same device-owned admission epoch, while the mapped packet
             * descriptor proves cross-device equality without a host shadow.
             */
            const DeviceMoEOverlayEpochTicket placement_ticket{
                .epoch = placement.ticket->epoch,
                .selector = placement.ticket->selector,
            };
            const std::uint32_t placement_bank =
                static_cast<std::uint32_t>(
                    placement_ticket.selector & 1u);
            const std::uint64_t placement_generation =
                placement_ticket.selector >> 1u;
            if (placement_ticket.epoch == 0u ||
                placement_ticket.epoch < identity.placement_epoch_floor ||
                placement_generation == 0u ||
                placement_bank >= kDeviceMoEOverlayEpochBankCount ||
                !placement.banks[placement_bank].valid() ||
                static_cast<std::uint64_t>(
                    *placement.banks[placement_bank].epoch) !=
                    placement_ticket.epoch)
            {
                return false;
            }
            *grant = MoEOverlayActivationDeviceEpochGrant{
                .digest = identity.digest,
                .generation = identity.epoch_generation,
                .placement_epoch = placement_ticket.epoch,
                .published_payload_bytes = 0u,
                .published_live_rows = 0u,
                .published_live_entries = 0u,
                .published_stage_count = 0u,
                .live_rows = 0u,
                .live_entries = 0u,
                .stage_count = identity.stage_count,
                .physical_rows = physical_rows,
                .last_published_stage = -1,
                .last_consumed_stage = -1,
                .single_row_id = -1,
                .endpoint = raw(endpoint),
                .state = raw(MoEOverlayActivationEndpointState::Active),
                .code = raw(MoEOverlayActivationStatusCode::Success),
                .graph_role = static_cast<MoEOverlayInferenceGraphRole>(
                    identity.graph_role),
            };
        }
        return grant->digest.valid() && grant->generation != 0u &&
               grant->placement_epoch != 0u &&
               grant->stage_count > stage_ordinal &&
               grant->physical_rows == physical_rows &&
               grant->endpoint == raw(endpoint) &&
               grant->graph_role != MoEOverlayInferenceGraphRole::None &&
               grant->state ==
                   raw(MoEOverlayActivationEndpointState::Active) &&
               grant->code == raw(MoEOverlayActivationStatusCode::Success);
    }

    /** @return Whether a descriptor names the exact retained stage and payload. */
    __device__ __forceinline__ bool validDescriptor(
        const MoEOverlayActivationPayloadDescriptor &descriptor,
        const MoEOverlayActivationDeviceEpochGrant &grant,
        std::uint32_t stage_ordinal,
        std::int32_t model_layer_index,
        std::uint64_t timeline) noexcept
    {
        return sameDigest(descriptor.digest, grant.digest) &&
               descriptor.timeline == timeline &&
               descriptor.placement_epoch == grant.placement_epoch &&
               descriptor.stage_ordinal == stage_ordinal &&
               descriptor.model_layer_index == model_layer_index;
    }

    /** Device-local result of validating one dispatch metadata launch. */
    struct DispatchMetadataPreparation
    {
        const std::int32_t *route_participants = nullptr;
        std::int32_t active_rows = 0;

        /** @return Whether validation selected a live placement bank. */
        __device__ __forceinline__ bool valid() const noexcept
        {
            return route_participants != nullptr;
        }
    };

    /**
     * @brief Validate captured control state before any packet byte is written.
     *
     * Exactly one thread calls this helper. Failures are published immediately
     * so every peer retained graph drains through the ordinary protocol edge.
     */
    __device__ __forceinline__ DispatchMetadataPreparation
    prepareDispatchMetadata(MoEOverlayActivationDispatchPackLaunch launch)
    {
        auto *control = launch.control;
        constexpr auto endpoint = MoEOverlayActivationEndpoint::Continuation;
        constexpr auto operation =
            MoEOverlayActivationOperation::PublishDispatch;
        const DeviceMoEOverlayEpochTicket placement_ticket{
            .epoch = launch.placement.ticket->epoch,
            .selector = launch.placement.ticket->selector,
        };
        const std::uint32_t placement_bank =
            static_cast<std::uint32_t>(placement_ticket.selector & 1u);
        const std::uint64_t placement_generation =
            placement_ticket.selector >> 1u;
        const auto selected_placement =
            launch.placement.banks[placement_bank];
        const bool grant_acquired = acquireOrValidateGrant(
                control,
                launch.grant,
                endpoint,
                operation,
                launch.stage_ordinal,
                launch.physical_rows,
                launch.target_participant_id,
                launch.placement);
        if (!grant_acquired ||
            placement_ticket.epoch == 0u ||
            placement_generation == 0u ||
            placement_ticket.epoch != launch.grant->placement_epoch ||
            placement_bank >= kDeviceMoEOverlayEpochBankCount ||
            !selected_placement.valid() ||
            static_cast<std::uint64_t>(*selected_placement.epoch) !=
                placement_ticket.epoch)
        {
            publishFailure(
                control,
                launch.grant,
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::InvalidControl,
                launch.stage_ordinal,
                launch.model_layer_index,
                placementAcquireEvidence(launch.placement),
                dispatchAdmissionDiagnostic(launch, grant_acquired),
                launch.grant
                    ? (static_cast<std::uint32_t>(
                           launch.grant->generation & 0xffffu) << 16u) |
                          static_cast<std::uint32_t>(
                              launch.grant->placement_epoch & 0xffffu)
                    : 0u);
            return {};
        }
        const std::int32_t *const overlay_route_participants =
            selected_placement.route_participants;

        const std::int32_t expected_previous =
            launch.stage_ordinal == 0u
                ? -1
                : static_cast<std::int32_t>(launch.stage_ordinal - 1u);
        if (launch.grant->last_published_stage != expected_previous ||
            (launch.stage_ordinal >= kMoEOverlayActivationBufferCount &&
             launch.grant->last_consumed_stage <
                 static_cast<std::int32_t>(
                     launch.stage_ordinal -
                     kMoEOverlayActivationBufferCount)))
        {
            publishFailure(
                control,
                launch.grant,
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::OutOfOrder,
                launch.stage_ordinal,
                launch.model_layer_index,
                0u);
            return {};
        }

        const std::int32_t active_rows =
            launch.active_row_count_device
                ? *launch.active_row_count_device
                : launch.physical_rows;
        if (active_rows < 0 || active_rows > launch.physical_rows)
        {
            publishFailure(
                control,
                launch.grant,
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::PayloadMismatch,
                launch.stage_ordinal,
                launch.model_layer_index,
                0u);
            return {};
        }
        return {
            .route_participants = overlay_route_participants,
            .active_rows = active_rows,
        };
    }

    /**
     * @brief Decode and validate one canonical route slot.
     * @return Whether the expert, weight, and placement entry are valid.
     */
    __device__ __forceinline__ bool decodeDispatchRoute(
        MoEOverlayActivationDispatchPackLaunch launch,
        const std::int32_t *route_participants,
        std::size_t route_slot,
        std::int32_t *expert,
        float *weight,
        std::int32_t *participant) noexcept
    {
        const float expert_fp32 = launch.route_expert_ids_fp32[route_slot];
        const float route_weight = launch.route_weights[route_slot];
        if (!isfinite(expert_fp32) || expert_fp32 < 0.0f ||
            expert_fp32 >= static_cast<float>(launch.placement.expert_count) ||
            truncf(expert_fp32) != expert_fp32 || !isfinite(route_weight))
        {
            return false;
        }
        const auto expert_id = static_cast<std::int32_t>(expert_fp32);
        const std::int32_t route_participant =
            route_participants[expert_id];
        if (route_participant < -1)
            return false;
        *expert = expert_id;
        *weight = route_weight;
        *participant = route_participant;
        return true;
    }

    /** @brief Publish the authenticated descriptor for one compact packet. */
    __device__ __forceinline__ void finishDispatchMetadata(
        MoEOverlayActivationDispatchPackLaunch launch,
        std::uint64_t compact_rows,
        std::uint64_t compact_entries)
    {
        auto *control = launch.control;
        constexpr auto endpoint = MoEOverlayActivationEndpoint::Continuation;
        constexpr auto operation =
            MoEOverlayActivationOperation::PublishDispatch;
        const std::uint64_t timeline =
            moeOverlayActivationLeasedTimelineValue(
                moeOverlayActivationBufferVisit(launch.stage_ordinal));
        const std::uint32_t bank =
            moeOverlayActivationBufferIndex(launch.stage_ordinal);
        auto &buffer = control->buffers[bank];
        if (timeline == 0u)
        {
            publishFailure(
                control,
                launch.grant,
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::StaleGeneration,
                launch.stage_ordinal,
                launch.model_layer_index,
                0u);
            return;
        }
        launch.grant->live_rows = compact_rows;
        launch.grant->live_entries = compact_entries;
        launch.grant->single_row_id =
            compact_rows == 1u ? launch.packet.row_ids[0] : -1;
        buffer.dispatch_descriptor = {
            .digest = launch.grant->digest,
            .timeline = timeline,
            .placement_epoch = launch.grant->placement_epoch,
            .live_rows = compact_rows,
            .live_entries = compact_entries,
            .payload_bytes = moeOverlayDispatchPayloadBytes(
                compact_rows,
                compact_entries,
                static_cast<std::uint32_t>(launch.packet.d_model)),
            .stage_ordinal = launch.stage_ordinal,
            .model_layer_index = launch.model_layer_index,
        };
        publishSuccess(
            control,
            launch.grant,
            endpoint,
            operation,
            launch.stage_ordinal,
            launch.model_layer_index,
            timeline,
            /*consumed=*/false,
            /*published=*/true,
            buffer.dispatch_descriptor.payload_bytes,
            compact_rows,
            compact_entries);
    }

    /** @brief Deterministically compact one target's row-major route metadata. */
    __device__ __forceinline__ void packDispatchMetadata(
        MoEOverlayActivationDispatchPackLaunch launch)
    {
        constexpr auto endpoint = MoEOverlayActivationEndpoint::Continuation;
        constexpr auto operation =
            MoEOverlayActivationOperation::PublishDispatch;
        const auto preparation = prepareDispatchMetadata(launch);
        if (!preparation.valid())
            return;

        std::uint64_t compact_rows = 0u;
        std::uint64_t compact_entries = 0u;
        launch.packet.entry_offsets[0] = 0;
        for (std::int32_t row = 0; row < preparation.active_rows; ++row)
        {
            const std::uint64_t row_begin = compact_entries;
            for (std::int32_t slot = 0; slot < launch.packet.top_k; ++slot)
            {
                const std::size_t route_slot =
                    static_cast<std::size_t>(row) *
                        static_cast<std::size_t>(launch.packet.top_k) +
                    static_cast<std::size_t>(slot);
                std::int32_t expert = -1;
                std::int32_t route_participant = -1;
                float weight = 0.0f;
                if (!decodeDispatchRoute(
                        launch,
                        preparation.route_participants,
                        route_slot,
                        &expert,
                        &weight,
                        &route_participant))
                {
                    publishFailure(
                        launch.control,
                        launch.grant,
                        endpoint,
                        operation,
                        MoEOverlayActivationStatusCode::PayloadMismatch,
                        launch.stage_ordinal,
                        launch.model_layer_index,
                        0u);
                    return;
                }
                if (route_participant != launch.target_participant_id)
                    continue;
                if (compact_entries >= launch.packet.entry_capacity)
                {
                    publishFailure(
                        launch.control,
                        launch.grant,
                        endpoint,
                        operation,
                        MoEOverlayActivationStatusCode::PayloadMismatch,
                        launch.stage_ordinal,
                        launch.model_layer_index,
                        0u);
                    return;
                }
                launch.packet.expert_ids[compact_entries] = expert;
                launch.packet.route_weights[compact_entries] = weight;
                launch.packet.original_route_slots[compact_entries] =
                    static_cast<std::int32_t>(route_slot);
                launch.packet.compact_route_slots[compact_entries] =
                    static_cast<std::int32_t>(
                        compact_rows *
                            static_cast<std::uint64_t>(
                                launch.packet.top_k) +
                        (compact_entries - row_begin));
                ++compact_entries;
            }
            if (compact_entries == row_begin)
                continue;
            if (compact_rows >= launch.packet.row_capacity)
            {
                publishFailure(
                    launch.control,
                    launch.grant,
                    endpoint,
                    operation,
                    MoEOverlayActivationStatusCode::PayloadMismatch,
                    launch.stage_ordinal,
                    launch.model_layer_index,
                    0u);
                return;
            }
            launch.packet.row_ids[compact_rows] = row;
            launch.packet.entry_offsets[compact_rows] =
                static_cast<std::int32_t>(row_begin);
            ++compact_rows;
            launch.packet.entry_offsets[compact_rows] =
                static_cast<std::int32_t>(compact_entries);
        }
        finishDispatchMetadata(launch, compact_rows, compact_entries);
    }

    /** Two-lane scan value: matching entries and non-empty rows. */
    struct DispatchMetadataScanValue
    {
        std::uint32_t entries;
        std::uint32_t rows;
    };

    /**
     * @brief Deterministically compact multi-row metadata with one GPU block.
     *
     * A thread owns one row in each fixed-size chunk. The inclusive block scan
     * assigns stable row and entry offsets; a second slot walk writes matches in
     * original router order. Chunk carries make the same algorithm total for
     * graph buckets larger than one block without atomics or reordered output.
     */
    static __global__ void packDispatchMetadataKernel(
        MoEOverlayActivationDispatchPackLaunch launch)
    {
        __shared__ DispatchMetadataScanValue
            scan[kDispatchMetadataBlockThreads];
        __shared__ const std::int32_t *route_participants;
        __shared__ std::int32_t active_rows;
        __shared__ std::uint64_t compact_rows;
        __shared__ std::uint64_t compact_entries;
        __shared__ std::uint32_t packet_valid;
        __shared__ std::uint32_t chunk_invalid;

        if (blockIdx.x != 0u ||
            blockDim.x != kDispatchMetadataBlockThreads)
        {
            return;
        }
        if (threadIdx.x == 0u)
        {
            const auto preparation = prepareDispatchMetadata(launch);
            route_participants = preparation.route_participants;
            active_rows = preparation.active_rows;
            compact_rows = 0u;
            compact_entries = 0u;
            packet_valid = preparation.valid() ? 1u : 0u;
            launch.packet.entry_offsets[0] = 0;
        }
        __syncthreads();
        if (packet_valid == 0u)
            return;

        for (std::uint32_t chunk_begin = 0u;
             chunk_begin < static_cast<std::uint32_t>(active_rows);
             chunk_begin += blockDim.x)
        {
            if (threadIdx.x == 0u)
                chunk_invalid = 0u;
            __syncthreads();

            const std::uint32_t row = chunk_begin + threadIdx.x;
            std::uint32_t matching_entries = 0u;
            if (row < static_cast<std::uint32_t>(active_rows))
            {
                for (std::int32_t slot = 0;
                     slot < launch.packet.top_k;
                     ++slot)
                {
                    const std::size_t route_slot =
                        static_cast<std::size_t>(row) *
                            static_cast<std::size_t>(launch.packet.top_k) +
                        static_cast<std::size_t>(slot);
                    std::int32_t expert = -1;
                    std::int32_t participant = -1;
                    float weight = 0.0f;
                    if (!decodeDispatchRoute(
                            launch,
                            route_participants,
                            route_slot,
                            &expert,
                            &weight,
                            &participant))
                    {
                        atomicExch(&chunk_invalid, 1u);
                        continue;
                    }
                    matching_entries +=
                        participant == launch.target_participant_id ? 1u : 0u;
                }
            }

            scan[threadIdx.x] = {
                .entries = matching_entries,
                .rows = matching_entries == 0u ? 0u : 1u,
            };
            __syncthreads();
            if (chunk_invalid != 0u)
            {
                if (threadIdx.x == 0u)
                {
                    publishFailure(
                        launch.control,
                        launch.grant,
                        MoEOverlayActivationEndpoint::Continuation,
                        MoEOverlayActivationOperation::PublishDispatch,
                        MoEOverlayActivationStatusCode::PayloadMismatch,
                        launch.stage_ordinal,
                        launch.model_layer_index,
                        0u);
                }
                return;
            }

            /* Hillis-Steele is deliberately fixed-order. It scans two small
             * integer counters together; no floating-point arithmetic or
             * scheduling-dependent atomic contributes to packet ordering. */
            for (std::uint32_t offset = 1u;
                 offset < blockDim.x;
                 offset <<= 1u)
            {
                DispatchMetadataScanValue prior{};
                if (threadIdx.x >= offset)
                    prior = scan[threadIdx.x - offset];
                __syncthreads();
                scan[threadIdx.x].entries += prior.entries;
                scan[threadIdx.x].rows += prior.rows;
                __syncthreads();
            }

            const auto chunk_total = scan[blockDim.x - 1u];
            const std::uint64_t entry_base = compact_entries;
            const std::uint64_t row_base = compact_rows;
            if (threadIdx.x == 0u &&
                (entry_base + chunk_total.entries >
                     launch.packet.entry_capacity ||
                 row_base + chunk_total.rows > launch.packet.row_capacity))
            {
                chunk_invalid = 1u;
            }
            __syncthreads();
            if (chunk_invalid != 0u)
            {
                if (threadIdx.x == 0u)
                {
                    publishFailure(
                        launch.control,
                        launch.grant,
                        MoEOverlayActivationEndpoint::Continuation,
                        MoEOverlayActivationOperation::PublishDispatch,
                        MoEOverlayActivationStatusCode::PayloadMismatch,
                        launch.stage_ordinal,
                        launch.model_layer_index,
                        0u);
                }
                return;
            }

            const std::uint64_t row_offset =
                row_base + scan[threadIdx.x].rows -
                (matching_entries == 0u ? 0u : 1u);
            const std::uint64_t entry_offset =
                entry_base + scan[threadIdx.x].entries - matching_entries;
            if (matching_entries != 0u)
            {
                launch.packet.row_ids[row_offset] =
                    static_cast<std::int32_t>(row);
                launch.packet.entry_offsets[row_offset] =
                    static_cast<std::int32_t>(entry_offset);

                std::uint32_t row_match = 0u;
                for (std::int32_t slot = 0;
                     slot < launch.packet.top_k;
                     ++slot)
                {
                    const std::size_t route_slot =
                        static_cast<std::size_t>(row) *
                            static_cast<std::size_t>(launch.packet.top_k) +
                        static_cast<std::size_t>(slot);
                    std::int32_t expert = -1;
                    std::int32_t participant = -1;
                    float weight = 0.0f;
                    if (!decodeDispatchRoute(
                            launch,
                            route_participants,
                            route_slot,
                            &expert,
                            &weight,
                            &participant) ||
                        participant != launch.target_participant_id)
                    {
                        continue;
                    }
                    const std::uint64_t output_entry =
                        entry_offset + row_match++;
                    launch.packet.expert_ids[output_entry] = expert;
                    launch.packet.route_weights[output_entry] = weight;
                    launch.packet.original_route_slots[output_entry] =
                        static_cast<std::int32_t>(route_slot);
                    launch.packet.compact_route_slots[output_entry] =
                        static_cast<std::int32_t>(
                            row_offset *
                                static_cast<std::uint64_t>(
                                    launch.packet.top_k) +
                            (row_match - 1u));
                }
            }
            __syncthreads();
            if (threadIdx.x == 0u)
            {
                compact_entries += chunk_total.entries;
                compact_rows += chunk_total.rows;
            }
            __syncthreads();
        }

        if (threadIdx.x == 0u)
        {
            launch.packet.entry_offsets[compact_rows] =
                static_cast<std::int32_t>(compact_entries);
            finishDispatchMetadata(launch, compact_rows, compact_entries);
        }
    }

    /**
     * @brief Gather compact live hidden rows directly into the mapped packet.
     *
     * The mapped timeline publication that follows this kernel is the release
     * edge for every payload byte. One thread caches the compact-to-source row
     * for blocks wholly contained in one row; adversarial narrow widths retain
     * the exact general lookup. This path deliberately avoids a fixed-capacity
     * copy-engine transaction at every MoE layer, which is more expensive than
     * direct mapped access for decode-sized packets on the measured topology.
     * Shared-physical bulk launches deliberately skip this kernel because the
     * transfer engine publishes their complete source matrix once per channel.
     */
    static __global__ void packDispatchHiddenKernel(
        MoEOverlayActivationDispatchPackLaunch launch)
    {
        __shared__ std::uint64_t block_live_rows;
        __shared__ std::uint32_t block_endpoint_state;
        __shared__ std::uint32_t block_has_single_row;
        __shared__ std::int32_t block_source_row;
        if (threadIdx.x == 0u)
        {
            block_live_rows = launch.grant->live_rows;
            block_endpoint_state = launch.grant->state;
            block_has_single_row = 0u;

            /* A normal hidden width is much larger than one block. Cache the
             * compact-to-source row mapping once for that whole block. The
             * uncommon block that crosses a row boundary retains the exact
             * per-thread lookup below. */
            const std::uint64_t live_elements =
                block_live_rows *
                static_cast<std::uint64_t>(launch.packet.d_model);
            const std::uint64_t block_begin =
                static_cast<std::uint64_t>(blockIdx.x) * blockDim.x;
            if (block_endpoint_state ==
                    raw(MoEOverlayActivationEndpointState::Active) &&
                block_begin < live_elements)
            {
                const std::uint64_t block_end =
                    min(block_begin + blockDim.x, live_elements) - 1u;
                const std::uint64_t first_row =
                    block_begin /
                    static_cast<std::uint64_t>(launch.packet.d_model);
                const std::uint64_t last_row =
                    block_end /
                    static_cast<std::uint64_t>(launch.packet.d_model);
                if (first_row == last_row)
                {
                    block_source_row =
                        block_live_rows == 1u
                            ? launch.grant->single_row_id
                            : loadPeerPublished(
                                  launch.packet.row_ids + first_row);
                    block_has_single_row = 1u;
                }
            }
        }
        __syncthreads();

        const std::size_t element =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (block_endpoint_state !=
                raw(MoEOverlayActivationEndpointState::Active) ||
            element >= block_live_rows *
                           static_cast<std::uint64_t>(launch.packet.d_model))
        {
            return;
        }
        const std::size_t compact_row =
            element / static_cast<std::size_t>(launch.packet.d_model);
        const std::size_t column =
            element % static_cast<std::size_t>(launch.packet.d_model);
        const std::int32_t source_row =
            block_has_single_row != 0u
                ? block_source_row
                : loadPeerPublished(launch.packet.row_ids + compact_row);
        launch.packet.hidden_rows_fp32[element] =
            launch.hidden_rows_fp32[
                static_cast<std::size_t>(source_row) *
                    static_cast<std::size_t>(launch.packet.d_model) +
                column];
    }

    /** @brief Validate the exact dispatch descriptor before follower reads. */
    __device__ __forceinline__ void validateDispatch(
        MoEOverlayActivationDispatchConsumeLaunch launch)
    {
        *launch.active_row_count_device = 0;
        auto *control = const_cast<MoEOverlayActivationEpochControl *>(
            launch.control);
        constexpr auto endpoint = MoEOverlayActivationEndpoint::Follower;
        constexpr auto operation =
            MoEOverlayActivationOperation::ConsumeDispatch;
        if (!acquireOrValidateGrant(
                control,
                launch.grant,
                endpoint,
                operation,
                launch.stage_ordinal,
                launch.physical_rows,
                /*expected_target_participant=*/-1,
                launch.placement))
        {
            publishFailure(
                control,
                launch.grant,
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::InvalidControl,
                launch.stage_ordinal,
                launch.model_layer_index,
                placementAcquireEvidence(launch.placement));
            return;
        }
        const std::int32_t expected_previous =
            launch.stage_ordinal == 0u
                ? -1
                : static_cast<std::int32_t>(launch.stage_ordinal - 1u);
        if (launch.grant->last_consumed_stage != expected_previous)
        {
            publishFailure(
                control,
                launch.grant,
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::OutOfOrder,
                launch.stage_ordinal,
                launch.model_layer_index,
                0u);
            return;
        }

        const std::uint64_t timeline =
            moeOverlayActivationLeasedTimelineValue(
                moeOverlayActivationBufferVisit(launch.stage_ordinal));
        const auto &buffer = control->buffers[
            moeOverlayActivationBufferIndex(launch.stage_ordinal)];
        const auto descriptor = snapshotPeerPublished(
            &buffer.dispatch_descriptor);
        const std::uint64_t dispatch_timeline =
            loadSystemAcquire64(&buffer.dispatch_signal.value);
        bool valid = dispatch_timeline == timeline &&
                     validDescriptor(
                         descriptor,
                         *launch.grant,
                         launch.stage_ordinal,
                         launch.model_layer_index,
                         timeline) &&
                     descriptor.live_rows <=
                         static_cast<std::uint64_t>(launch.physical_rows) &&
                     descriptor.live_rows <= launch.packet.row_capacity &&
                     descriptor.live_entries <= launch.packet.entry_capacity &&
                     descriptor.live_entries <=
                         descriptor.live_rows *
                             static_cast<std::uint64_t>(launch.packet.top_k) &&
                     descriptor.payload_bytes ==
                         moeOverlayDispatchPayloadBytes(
                             descriptor.live_rows,
                             descriptor.live_entries,
                             static_cast<std::uint32_t>(
                                 launch.packet.d_model));
        std::int32_t prior_row = -1;
        std::int32_t prior_offset = 0;
        if (valid &&
            loadPeerPublished(launch.packet.entry_offsets) != 0)
            valid = false;
        for (std::uint64_t row = 0u; valid && row < descriptor.live_rows; ++row)
        {
            const std::int32_t row_id =
                loadPeerPublished(launch.packet.row_ids + row);
            const std::int32_t begin =
                loadPeerPublished(launch.packet.entry_offsets + row);
            const std::int32_t end =
                loadPeerPublished(launch.packet.entry_offsets + row + 1u);
            if (row_id <= prior_row || row_id < 0 ||
                row_id >= launch.physical_rows || begin != prior_offset ||
                end <= begin || end - begin > launch.packet.top_k ||
                end > static_cast<std::int32_t>(descriptor.live_entries))
            {
                valid = false;
                break;
            }
            for (std::int32_t entry = begin; entry < end; ++entry)
            {
                const std::int32_t original_slot = loadPeerPublished(
                    launch.packet.original_route_slots + entry);
                const std::int32_t compact_slot = loadPeerPublished(
                    launch.packet.compact_route_slots + entry);
                const std::int32_t expected_compact_slot =
                    static_cast<std::int32_t>(row) * launch.packet.top_k +
                    (entry - begin);
                const std::int32_t original_row =
                    original_slot >= 0
                        ? original_slot / launch.packet.top_k
                        : -1;
                const std::int32_t original_route =
                    original_slot >= 0
                        ? original_slot % launch.packet.top_k
                        : -1;
                const std::int32_t prior_original_slot =
                    entry == begin
                        ? row_id * launch.packet.top_k - 1
                        : loadPeerPublished(
                              launch.packet.original_route_slots +
                              entry - 1);
                if (loadPeerPublished(
                        launch.packet.expert_ids + entry) < 0 ||
                    !isfinite(loadPeerPublished(
                        launch.packet.route_weights + entry)) ||
                    original_row != row_id || original_route < 0 ||
                    original_route >= launch.packet.top_k ||
                    original_slot <= prior_original_slot ||
                    compact_slot != expected_compact_slot)
                {
                    valid = false;
                    break;
                }
            }
            prior_row = row_id;
            prior_offset = end;
        }
        if (valid &&
            prior_offset != static_cast<std::int32_t>(
                                descriptor.live_entries))
        {
            valid = false;
        }
        if (!valid)
        {
            publishFailure(
                control,
                launch.grant,
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::PayloadMismatch,
                launch.stage_ordinal,
                launch.model_layer_index,
                dispatch_timeline);
            return;
        }

        *launch.active_row_count_device =
            static_cast<std::int32_t>(descriptor.live_rows);
        launch.grant->live_rows = descriptor.live_rows;
        launch.grant->live_entries = descriptor.live_entries;
        launch.grant->single_row_id =
            descriptor.live_rows == 1u
                ? loadPeerPublished(launch.packet.row_ids)
                : -1;
        publishSuccess(
            control,
            launch.grant,
            endpoint,
            operation,
            launch.stage_ordinal,
            launch.model_layer_index,
            timeline,
            /*consumed=*/true,
            /*published=*/false);
    }

    /**
     * @brief Cooperatively validate one multi-row dispatch packet.
     *
     * The descriptor and endpoint lifecycle remain single-authority work for
     * thread zero. Row/entry evidence lives in node-local mapped memory,
     * however, so reading thousands of values serially turns PCIe latency into
     * a per-layer bubble. One block stripes those independent checks while a
     * shared validity word records the first failure. Thread zero publishes
     * the exact same endpoint transition only after every checker arrives at
     * the final barrier.
     */
    static __global__ void validateDispatchKernel(
        MoEOverlayActivationDispatchConsumeLaunch launch)
    {
        if (blockIdx.x != 0u)
            return;

        __shared__ std::uint32_t block_header_ready;
        __shared__ std::uint32_t block_valid;
        __shared__ std::uint64_t block_live_rows;
        __shared__ std::uint64_t block_live_entries;
        __shared__ std::uint64_t block_timeline;
        __shared__ std::uint64_t block_dispatch_timeline;

        auto *control = const_cast<MoEOverlayActivationEpochControl *>(
            launch.control);
        constexpr auto endpoint = MoEOverlayActivationEndpoint::Follower;
        constexpr auto operation =
            MoEOverlayActivationOperation::ConsumeDispatch;
        if (threadIdx.x == 0u)
        {
            *launch.active_row_count_device = 0;
            block_header_ready = 0u;
            block_valid = 0u;
            block_live_rows = 0u;
            block_live_entries = 0u;
            block_timeline = moeOverlayActivationLeasedTimelineValue(
                moeOverlayActivationBufferVisit(launch.stage_ordinal));
            block_dispatch_timeline = 0u;

            if (!acquireOrValidateGrant(
                    control,
                    launch.grant,
                    endpoint,
                    operation,
                    launch.stage_ordinal,
                    launch.physical_rows,
                    /*expected_target_participant=*/-1,
                    launch.placement))
            {
                publishFailure(
                    control,
                    launch.grant,
                    endpoint,
                    operation,
                    MoEOverlayActivationStatusCode::InvalidControl,
                    launch.stage_ordinal,
                    launch.model_layer_index,
                    placementAcquireEvidence(launch.placement));
            }
            else
            {
                const std::int32_t expected_previous =
                    launch.stage_ordinal == 0u
                        ? -1
                        : static_cast<std::int32_t>(
                              launch.stage_ordinal - 1u);
                if (launch.grant->last_consumed_stage != expected_previous)
                {
                    publishFailure(
                        control,
                        launch.grant,
                        endpoint,
                        operation,
                        MoEOverlayActivationStatusCode::OutOfOrder,
                        launch.stage_ordinal,
                        launch.model_layer_index,
                        0u);
                }
                else
                {
                    const auto &buffer = control->buffers[
                        moeOverlayActivationBufferIndex(
                            launch.stage_ordinal)];
                    const auto descriptor = snapshotPeerPublished(
                        &buffer.dispatch_descriptor);
                    block_dispatch_timeline = loadSystemAcquire64(
                        &buffer.dispatch_signal.value);
                    block_live_rows = descriptor.live_rows;
                    block_live_entries = descriptor.live_entries;
                    block_header_ready = 1u;
                    block_valid =
                        block_dispatch_timeline == block_timeline &&
                                validDescriptor(
                                    descriptor,
                                    *launch.grant,
                                    launch.stage_ordinal,
                                    launch.model_layer_index,
                                    block_timeline) &&
                                descriptor.live_rows <=
                                    static_cast<std::uint64_t>(
                                        launch.physical_rows) &&
                                descriptor.live_rows <=
                                    launch.packet.row_capacity &&
                                descriptor.live_entries <=
                                    launch.packet.entry_capacity &&
                                descriptor.live_entries <=
                                    descriptor.live_rows *
                                        static_cast<std::uint64_t>(
                                            launch.packet.top_k) &&
                                descriptor.payload_bytes ==
                                    moeOverlayDispatchPayloadBytes(
                                        descriptor.live_rows,
                                        descriptor.live_entries,
                                        static_cast<std::uint32_t>(
                                            launch.packet.d_model))
                            ? 1u
                            : 0u;
                }
            }
        }
        __syncthreads();

        if (block_header_ready != 0u && block_valid != 0u)
        {
            for (std::uint64_t row = threadIdx.x;
                 row < block_live_rows;
                 row += blockDim.x)
            {
                const std::int32_t row_id =
                    loadPeerPublished(launch.packet.row_ids + row);
                const std::int32_t begin =
                    loadPeerPublished(launch.packet.entry_offsets + row);
                const std::int32_t end =
                    loadPeerPublished(
                        launch.packet.entry_offsets + row + 1u);
                const std::int32_t prior_row =
                    row == 0u
                        ? -1
                        : loadPeerPublished(
                              launch.packet.row_ids + row - 1u);
                const bool row_valid =
                    row_id > prior_row && row_id >= 0 &&
                    row_id < launch.physical_rows && begin >= 0 &&
                    (row != 0u || begin == 0) && end > begin &&
                    end - begin <= launch.packet.top_k &&
                    end <= static_cast<std::int32_t>(block_live_entries) &&
                    (row + 1u != block_live_rows ||
                     end == static_cast<std::int32_t>(
                                block_live_entries));
                bool route_slots_valid = row_valid;
                std::int32_t prior_original_slot =
                    row_id * launch.packet.top_k - 1;
                for (std::int32_t entry = begin;
                     route_slots_valid && entry < end;
                     ++entry)
                {
                    const std::int32_t original_slot = loadPeerPublished(
                        launch.packet.original_route_slots + entry);
                    const std::int32_t compact_slot = loadPeerPublished(
                        launch.packet.compact_route_slots + entry);
                    route_slots_valid =
                        original_slot / launch.packet.top_k == row_id &&
                        original_slot % launch.packet.top_k >= 0 &&
                        original_slot > prior_original_slot &&
                        compact_slot ==
                            static_cast<std::int32_t>(row) *
                                launch.packet.top_k +
                            (entry - begin);
                    prior_original_slot = original_slot;
                }
                if (!route_slots_valid)
                    atomicExch(&block_valid, 0u);
            }
            for (std::uint64_t entry = threadIdx.x;
                 entry < block_live_entries;
                 entry += blockDim.x)
            {
                if (loadPeerPublished(
                        launch.packet.expert_ids + entry) < 0 ||
                    !isfinite(loadPeerPublished(
                        launch.packet.route_weights + entry)))
                {
                    atomicExch(&block_valid, 0u);
                }
            }
        }
        __syncthreads();

        if (threadIdx.x != 0u || block_header_ready == 0u)
            return;
        if (block_valid == 0u)
        {
            publishFailure(
                control,
                launch.grant,
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::PayloadMismatch,
                launch.stage_ordinal,
                launch.model_layer_index,
                block_dispatch_timeline);
            return;
        }

        *launch.active_row_count_device =
            static_cast<std::int32_t>(block_live_rows);
        launch.grant->live_rows = block_live_rows;
        launch.grant->live_entries = block_live_entries;
        launch.grant->single_row_id =
            block_live_rows == 1u
                ? loadPeerPublished(launch.packet.row_ids)
                : -1;
        publishSuccess(
            control,
            launch.grant,
            endpoint,
            operation,
            launch.stage_ordinal,
            launch.model_layer_index,
            block_timeline,
            /*consumed=*/true,
            /*published=*/false);
    }

    /**
     * @brief Materialize one validated mapped dispatch into follower tensors.
     *
     * Active hidden rows and route metadata are read directly from the mapped
     * packet after its timeline acquire; inactive rows and slots are cleared.
     * Grant state is endpoint-private device memory, so one thread snapshots it
     * for the block without issuing redundant mapped control loads.
     */
    static __global__ void materializeMappedDispatchKernel(
        MoEOverlayActivationDispatchConsumeLaunch launch)
    {
        __shared__ std::uint32_t block_endpoint_state;
        __shared__ std::int32_t block_live_rows;
        __shared__ std::uint32_t block_has_hidden_payload_row;
        __shared__ std::uint64_t block_hidden_payload_row;
        if (threadIdx.x == 0u)
        {
            block_endpoint_state = launch.grant->state;
            block_live_rows =
                block_endpoint_state ==
                        raw(MoEOverlayActivationEndpointState::Active)
                    ? static_cast<std::int32_t>(launch.grant->live_rows)
                    : 0;
            block_has_hidden_payload_row = 0u;

            /* Normal hidden widths span several whole blocks. Resolve their
             * compact-to-physical row once per block instead of making every
             * activation element issue the same mapped PCIe/UPI load. The
             * uncommon block crossing a row boundary uses the exact per-thread
             * lookup below. */
            const std::uint64_t hidden_elements =
                static_cast<std::uint64_t>(launch.physical_rows) *
                static_cast<std::uint64_t>(launch.packet.d_model);
            const std::uint64_t block_begin =
                static_cast<std::uint64_t>(blockIdx.x) * blockDim.x;
            if (block_begin < hidden_elements)
            {
                const std::uint64_t block_end =
                    min(block_begin + blockDim.x, hidden_elements) - 1u;
                const std::uint64_t first_row =
                    block_begin /
                    static_cast<std::uint64_t>(launch.packet.d_model);
                const std::uint64_t last_row =
                    block_end /
                    static_cast<std::uint64_t>(launch.packet.d_model);
                if (first_row == last_row &&
                    first_row < static_cast<std::uint64_t>(block_live_rows))
                {
                    block_hidden_payload_row =
                        launch.hidden_payload_layout ==
                                MoEOverlayActivationHiddenPayloadLayout::
                                    SharedPhysicalRows
                            ? static_cast<std::uint64_t>(loadPeerPublished(
                                  launch.packet.row_ids + first_row))
                            : first_row;
                    block_has_hidden_payload_row = 1u;
                }
            }
        }
        __syncthreads();

        const std::size_t element =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const std::size_t hidden_elements =
            static_cast<std::size_t>(launch.physical_rows) *
            static_cast<std::size_t>(launch.packet.d_model);
        const std::size_t route_elements =
            static_cast<std::size_t>(launch.physical_rows) *
            static_cast<std::size_t>(launch.packet.top_k);
        if (element < hidden_elements)
        {
            const std::size_t row =
                element / static_cast<std::size_t>(launch.packet.d_model);
            if (row < static_cast<std::size_t>(block_live_rows))
            {
                const std::size_t column =
                    element %
                    static_cast<std::size_t>(launch.packet.d_model);
                const std::size_t payload_row =
                    block_has_hidden_payload_row != 0u
                        ? static_cast<std::size_t>(
                              block_hidden_payload_row)
                        : (launch.hidden_payload_layout ==
                                   MoEOverlayActivationHiddenPayloadLayout::
                                       SharedPhysicalRows
                               ? static_cast<std::size_t>(loadPeerPublished(
                                     launch.packet.row_ids + row))
                               : row);
                launch.hidden_rows_fp32[element] = loadPeerPublished(
                    launch.packet.hidden_rows_fp32 +
                        payload_row *
                            static_cast<std::size_t>(launch.packet.d_model) +
                        column);
            }
            else
            {
                launch.hidden_rows_fp32[element] = 0.0f;
            }
        }
        if (element < route_elements)
        {
            const std::size_t row =
                element / static_cast<std::size_t>(launch.packet.top_k);
            const std::int32_t slot = static_cast<std::int32_t>(
                element % static_cast<std::size_t>(launch.packet.top_k));
            std::int32_t expert = -1;
            float weight = 0.0f;
            if (row < static_cast<std::size_t>(block_live_rows))
            {
                const std::int32_t begin =
                    loadPeerPublished(
                        launch.packet.entry_offsets + row);
                const std::int32_t end =
                    loadPeerPublished(
                        launch.packet.entry_offsets + row + 1u);
                if (begin + slot < end)
                {
                    expert = loadPeerPublished(
                        launch.packet.expert_ids + begin + slot);
                    weight = loadPeerPublished(
                        launch.packet.route_weights + begin + slot);
                }
            }
            launch.routing_indices_fp32[element] =
                static_cast<float>(expert);
            launch.routing_weights_fp32[element] = weight;
        }
    }

    /** @brief Validate follower state and publish the compact return descriptor. */
    __device__ __forceinline__ void packReturnMetadata(
        MoEOverlayActivationReturnPackLaunch launch)
    {
        auto *control = launch.control;
        constexpr auto endpoint = MoEOverlayActivationEndpoint::Follower;
        constexpr auto operation = MoEOverlayActivationOperation::PublishReturn;
        if (!acquireOrValidateGrant(
                control,
                launch.grant,
                endpoint,
                operation,
                launch.stage_ordinal,
                launch.physical_rows))
        {
            publishFailure(
                control,
                launch.grant,
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::InvalidControl,
                launch.stage_ordinal,
                launch.model_layer_index,
                0u);
            return;
        }
        const std::int32_t expected_previous =
            launch.stage_ordinal == 0u
                ? -1
                : static_cast<std::int32_t>(launch.stage_ordinal - 1u);
        if (launch.grant->last_consumed_stage !=
                static_cast<std::int32_t>(launch.stage_ordinal) ||
            launch.grant->last_published_stage != expected_previous)
        {
            publishFailure(
                control,
                launch.grant,
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::OutOfOrder,
                launch.stage_ordinal,
                launch.model_layer_index,
                0u);
            return;
        }

        const std::uint64_t timeline =
            moeOverlayActivationLeasedTimelineValue(
                moeOverlayActivationBufferVisit(launch.stage_ordinal));
        const std::uint32_t bank =
            moeOverlayActivationBufferIndex(launch.stage_ordinal);
        auto &buffer = control->buffers[bank];
        if (timeline == 0u ||
            launch.grant->live_entries >
                launch.returned.route_slot_capacity ||
            launch.grant->live_entries >
                launch.grant->live_rows *
                    static_cast<std::uint64_t>(launch.dispatch.top_k))
        {
            publishFailure(
                control,
                launch.grant,
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::PayloadMismatch,
                launch.stage_ordinal,
                launch.model_layer_index,
                0u);
            return;
        }
        const MoEOverlayActivationPayloadDescriptor descriptor{
            .digest = launch.grant->digest,
            .timeline = timeline,
            .placement_epoch = launch.grant->placement_epoch,
            .live_rows = launch.grant->live_rows,
            .live_entries = launch.grant->live_entries,
            .payload_bytes = moeOverlayReturnPayloadBytes(
                launch.grant->live_entries,
                static_cast<std::uint32_t>(launch.returned.d_model)),
            .stage_ordinal = launch.stage_ordinal,
            .model_layer_index = launch.model_layer_index,
        };
        buffer.return_descriptor = descriptor;
        publishSuccess(
            control,
            launch.grant,
            endpoint,
            operation,
            launch.stage_ordinal,
            launch.model_layer_index,
            timeline,
            /*consumed=*/true,
            /*published=*/true,
            descriptor.payload_bytes,
            descriptor.live_rows,
            descriptor.live_entries);
    }

    /**
     * @brief Publish multi-row return metadata with coalesced row-id stores.
     *
     * Lifecycle validation and descriptor publication stay on thread zero.
     * The independent row-id copy is striped across the block so a wide
     * prefill packet does not serialize mapped reads and writes on one lane.
     */
    static __global__ void packReturnMetadataKernel(
        MoEOverlayActivationReturnPackLaunch launch)
    {
        if (blockIdx.x != 0u)
            return;

        __shared__ std::uint32_t block_ready;
        __shared__ std::uint64_t block_live_rows;
        __shared__ std::uint64_t block_timeline;
        auto *control = launch.control;
        constexpr auto endpoint = MoEOverlayActivationEndpoint::Follower;
        constexpr auto operation =
            MoEOverlayActivationOperation::PublishReturn;
        if (threadIdx.x == 0u)
        {
            block_ready = 0u;
            block_live_rows = 0u;
            block_timeline = moeOverlayActivationLeasedTimelineValue(
                moeOverlayActivationBufferVisit(launch.stage_ordinal));
            if (!acquireOrValidateGrant(
                    control,
                    launch.grant,
                    endpoint,
                    operation,
                    launch.stage_ordinal,
                    launch.physical_rows))
            {
                publishFailure(
                    control,
                    launch.grant,
                    endpoint,
                    operation,
                    MoEOverlayActivationStatusCode::InvalidControl,
                    launch.stage_ordinal,
                    launch.model_layer_index,
                    0u);
            }
            else
            {
                const std::int32_t expected_previous =
                    launch.stage_ordinal == 0u
                        ? -1
                        : static_cast<std::int32_t>(
                              launch.stage_ordinal - 1u);
                if (launch.grant->last_consumed_stage !=
                        static_cast<std::int32_t>(
                            launch.stage_ordinal) ||
                    launch.grant->last_published_stage !=
                        expected_previous)
                {
                    publishFailure(
                        control,
                        launch.grant,
                        endpoint,
                        operation,
                        MoEOverlayActivationStatusCode::OutOfOrder,
                        launch.stage_ordinal,
                        launch.model_layer_index,
                        0u);
                }
                else if (block_timeline == 0u ||
                         launch.grant->live_entries >
                             launch.returned.route_slot_capacity ||
                         launch.grant->live_entries >
                             launch.grant->live_rows *
                                 static_cast<std::uint64_t>(
                                     launch.dispatch.top_k))
                {
                    publishFailure(
                        control,
                        launch.grant,
                        endpoint,
                        operation,
                        MoEOverlayActivationStatusCode::PayloadMismatch,
                        launch.stage_ordinal,
                        launch.model_layer_index,
                        0u);
                }
                else
                {
                    block_live_rows = launch.grant->live_rows;
                    block_ready = 1u;
                }
            }
        }
        __syncthreads();

        if (threadIdx.x != 0u || block_ready == 0u)
            return;
        const std::uint32_t bank =
            moeOverlayActivationBufferIndex(launch.stage_ordinal);
        auto &buffer = control->buffers[bank];
        const MoEOverlayActivationPayloadDescriptor descriptor{
            .digest = launch.grant->digest,
            .timeline = block_timeline,
            .placement_epoch = launch.grant->placement_epoch,
            .live_rows = block_live_rows,
            .live_entries = launch.grant->live_entries,
            .payload_bytes = moeOverlayReturnPayloadBytes(
                launch.grant->live_entries,
                static_cast<std::uint32_t>(
                    launch.returned.d_model)),
            .stage_ordinal = launch.stage_ordinal,
            .model_layer_index = launch.model_layer_index,
        };
        buffer.return_descriptor = descriptor;
        publishSuccess(
            control,
            launch.grant,
            endpoint,
            operation,
            launch.stage_ordinal,
            launch.model_layer_index,
            block_timeline,
            /*consumed=*/true,
            /*published=*/true,
            descriptor.payload_bytes,
            descriptor.live_rows,
            descriptor.live_entries);
    }

    /**
     * @brief Publish follower-local expert output directly into mapped pages.
     *
     * Metadata runs first on the same stream and records the validated live-row
     * prefix in the endpoint-private grant. Every payload thread then performs
     * one ordered mapped store. The subsequent timeline publication is the only
     * peer-visible release edge, so a continuation cannot observe a partial
     * return even though metadata and payload use separate kernels.
     */
    static __global__ void packMappedReturnPayloadKernel(
        MoEOverlayActivationReturnPackLaunch launch)
    {
        __shared__ std::uint32_t block_endpoint_state;
        __shared__ std::uint32_t block_status_code;
        __shared__ std::int32_t block_last_published_stage;
        __shared__ std::uint64_t block_live_entries;
        if (threadIdx.x == 0u)
        {
            block_endpoint_state = launch.grant->state;
            block_status_code = launch.grant->code;
            block_last_published_stage = launch.grant->last_published_stage;
            block_live_entries = launch.grant->live_entries;
        }
        __syncthreads();

        const auto state = static_cast<MoEOverlayActivationEndpointState>(
            block_endpoint_state);
        const std::size_t element =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if ((state != MoEOverlayActivationEndpointState::Active &&
             state != MoEOverlayActivationEndpointState::Complete) ||
            block_status_code !=
                raw(MoEOverlayActivationStatusCode::Success) ||
            block_last_published_stage !=
                static_cast<std::int32_t>(launch.stage_ordinal) ||
            block_live_entries >
                launch.returned.route_slot_capacity ||
            element >= block_live_entries *
                           static_cast<std::uint64_t>(
                               launch.returned.d_model))
        {
            return;
        }
        const std::size_t compact_entry =
            element / static_cast<std::size_t>(launch.returned.d_model);
        const std::size_t column =
            element % static_cast<std::size_t>(launch.returned.d_model);
        const std::int32_t original_slot = loadPeerPublished(
            launch.dispatch.original_route_slots + compact_entry);
        const std::int32_t compact_slot = loadPeerPublished(
            launch.dispatch.compact_route_slots + compact_entry);
        const std::size_t route_slot_limit =
            static_cast<std::size_t>(launch.physical_rows) *
            static_cast<std::size_t>(launch.dispatch.top_k);
        if (original_slot < 0 || compact_slot < 0 ||
            static_cast<std::size_t>(original_slot) >= route_slot_limit ||
            static_cast<std::size_t>(compact_slot) >= route_slot_limit)
        {
            return;
        }
        launch.returned.canonical_route_contributions_fp32[
            static_cast<std::size_t>(original_slot) *
                static_cast<std::size_t>(launch.returned.d_model) +
            column] =
            launch.local_canonical_route_contributions_fp32[
                static_cast<std::size_t>(compact_slot) *
                    static_cast<std::size_t>(launch.returned.d_model) +
                column];
    }

    /** @brief Validate one participant return before deterministic accumulation. */
    __device__ __forceinline__ void validateReturn(
        MoEOverlayActivationReturnConsumeLaunch launch)
    {
        auto *control = const_cast<MoEOverlayActivationEpochControl *>(
            launch.control);
        constexpr auto endpoint = MoEOverlayActivationEndpoint::Continuation;
        constexpr auto operation = MoEOverlayActivationOperation::ConsumeReturn;
        if (!acquireOrValidateGrant(
                control,
                launch.grant,
                endpoint,
                operation,
                launch.stage_ordinal,
                launch.physical_rows))
        {
            publishFailure(
                control,
                launch.grant,
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::InvalidControl,
                launch.stage_ordinal,
                launch.model_layer_index,
                0u);
            return;
        }
        const std::int32_t expected_previous =
            launch.stage_ordinal == 0u
                ? -1
                : static_cast<std::int32_t>(launch.stage_ordinal - 1u);
        if (launch.grant->last_published_stage <
                static_cast<std::int32_t>(launch.stage_ordinal) ||
            launch.grant->last_consumed_stage != expected_previous)
        {
            publishFailure(
                control,
                launch.grant,
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::OutOfOrder,
                launch.stage_ordinal,
                launch.model_layer_index,
                0u);
            return;
        }

        const std::uint64_t timeline =
            moeOverlayActivationLeasedTimelineValue(
                moeOverlayActivationBufferVisit(launch.stage_ordinal));
        const auto &buffer = control->buffers[
            moeOverlayActivationBufferIndex(launch.stage_ordinal)];
        const std::uint64_t return_timeline =
            loadSystemAcquire64(&buffer.return_signal.value);
        const auto descriptor = snapshotPeerPublished(
            &buffer.return_descriptor);
        bool valid = return_timeline == timeline &&
                     validDescriptor(
                         descriptor,
                         *launch.grant,
                         launch.stage_ordinal,
                         launch.model_layer_index,
                         timeline) &&
                     descriptor.live_entries ==
                         launch.grant->live_entries &&
                     descriptor.live_rows == launch.grant->live_rows &&
                     descriptor.live_entries <=
                         launch.returned.route_slot_capacity &&
                     descriptor.live_entries <=
                         descriptor.live_rows *
                             static_cast<std::uint64_t>(
                                 launch.dispatch.top_k) &&
                     descriptor.payload_bytes ==
                         moeOverlayReturnPayloadBytes(
                             descriptor.live_entries,
                             static_cast<std::uint32_t>(
                                 launch.returned.d_model));
        std::int32_t prior_original_slot = -1;
        const std::uint64_t route_slot_limit =
            static_cast<std::uint64_t>(launch.physical_rows) *
            static_cast<std::uint64_t>(launch.dispatch.top_k);
        for (std::uint64_t entry = 0u;
             valid && entry < descriptor.live_entries;
             ++entry)
        {
            const std::int32_t original_slot = loadPeerPublished(
                launch.dispatch.original_route_slots + entry);
            const std::int32_t compact_slot = loadPeerPublished(
                launch.dispatch.compact_route_slots + entry);
            if (original_slot <= prior_original_slot || original_slot < 0 ||
                compact_slot < 0 ||
                static_cast<std::uint64_t>(original_slot) >=
                    route_slot_limit ||
                static_cast<std::uint64_t>(compact_slot) >=
                    route_slot_limit)
            {
                valid = false;
            }
            prior_original_slot = original_slot;
        }
        if (!valid)
        {
            publishFailure(
                control,
                launch.grant,
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::PayloadMismatch,
                launch.stage_ordinal,
                launch.model_layer_index,
                return_timeline);
            return;
        }
        launch.grant->live_rows = descriptor.live_rows;
        launch.grant->live_entries = descriptor.live_entries;
        launch.grant->single_row_id =
            descriptor.live_rows == 1u
                ? loadPeerPublished(launch.dispatch.row_ids)
                : -1;
        publishSuccess(
            control,
            launch.grant,
            endpoint,
            operation,
            launch.stage_ordinal,
            launch.model_layer_index,
            timeline,
            /*consumed=*/true,
            /*published=*/true);
    }

    /**
     * @brief Cooperatively validate one multi-row return packet.
     *
     * Return row identities are independent evidence after the acquire edge.
     * Striping them across one block removes another serial mapped-memory walk
     * while thread zero remains the sole grant/state publication authority.
     */
    static __global__ void validateReturnKernel(
        MoEOverlayActivationReturnConsumeLaunch launch)
    {
        if (blockIdx.x != 0u)
            return;

        __shared__ std::uint32_t block_header_ready;
        __shared__ std::uint32_t block_valid;
        __shared__ std::uint64_t block_live_rows;
        __shared__ std::uint64_t block_timeline;
        __shared__ std::uint64_t block_return_timeline;
        auto *control = const_cast<MoEOverlayActivationEpochControl *>(
            launch.control);
        constexpr auto endpoint =
            MoEOverlayActivationEndpoint::Continuation;
        constexpr auto operation =
            MoEOverlayActivationOperation::ConsumeReturn;
        if (threadIdx.x == 0u)
        {
            block_header_ready = 0u;
            block_valid = 0u;
            block_live_rows = 0u;
            block_timeline = moeOverlayActivationLeasedTimelineValue(
                moeOverlayActivationBufferVisit(launch.stage_ordinal));
            block_return_timeline = 0u;
            if (!acquireOrValidateGrant(
                    control,
                    launch.grant,
                    endpoint,
                    operation,
                    launch.stage_ordinal,
                    launch.physical_rows))
            {
                publishFailure(
                    control,
                    launch.grant,
                    endpoint,
                    operation,
                    MoEOverlayActivationStatusCode::InvalidControl,
                    launch.stage_ordinal,
                    launch.model_layer_index,
                    0u);
            }
            else
            {
                const std::int32_t expected_previous =
                    launch.stage_ordinal == 0u
                        ? -1
                        : static_cast<std::int32_t>(
                              launch.stage_ordinal - 1u);
                if (launch.grant->last_published_stage <
                        static_cast<std::int32_t>(
                            launch.stage_ordinal) ||
                    launch.grant->last_consumed_stage !=
                        expected_previous)
                {
                    publishFailure(
                        control,
                        launch.grant,
                        endpoint,
                        operation,
                        MoEOverlayActivationStatusCode::OutOfOrder,
                        launch.stage_ordinal,
                        launch.model_layer_index,
                        0u);
                }
                else
                {
                    const auto &buffer = control->buffers[
                        moeOverlayActivationBufferIndex(
                            launch.stage_ordinal)];
                    block_return_timeline = loadSystemAcquire64(
                        &buffer.return_signal.value);
                    const auto descriptor = snapshotPeerPublished(
                        &buffer.return_descriptor);
                    block_live_rows = descriptor.live_rows;
                    block_header_ready = 1u;
                    block_valid =
                        block_return_timeline == block_timeline &&
                                validDescriptor(
                                    descriptor,
                                    *launch.grant,
                                    launch.stage_ordinal,
                                    launch.model_layer_index,
                                    block_timeline) &&
                                descriptor.live_entries ==
                                    launch.grant->live_entries &&
                                descriptor.live_rows ==
                                    launch.grant->live_rows &&
                                descriptor.live_entries <=
                                    launch.returned.route_slot_capacity &&
                                descriptor.live_entries <=
                                    descriptor.live_rows *
                                        static_cast<std::uint64_t>(
                                            launch.dispatch.top_k) &&
                                descriptor.payload_bytes ==
                                    moeOverlayReturnPayloadBytes(
                                        descriptor.live_entries,
                                        static_cast<std::uint32_t>(
                                            launch.returned.d_model))
                            ? 1u
                            : 0u;
                }
            }
        }
        __syncthreads();

        if (block_header_ready != 0u && block_valid != 0u)
        {
            const std::uint64_t route_slot_limit =
                static_cast<std::uint64_t>(launch.physical_rows) *
                static_cast<std::uint64_t>(launch.dispatch.top_k);
            for (std::uint64_t entry = threadIdx.x;
                 entry < launch.grant->live_entries;
                 entry += blockDim.x)
            {
                const std::int32_t original_slot = loadPeerPublished(
                    launch.dispatch.original_route_slots + entry);
                const std::int32_t compact_slot = loadPeerPublished(
                    launch.dispatch.compact_route_slots + entry);
                const std::int32_t prior_original_slot =
                    entry == 0u
                        ? -1
                        : loadPeerPublished(
                              launch.dispatch.original_route_slots +
                              entry - 1u);
                if (original_slot <= prior_original_slot ||
                    original_slot < 0 || compact_slot < 0 ||
                    static_cast<std::uint64_t>(original_slot) >=
                        route_slot_limit ||
                    static_cast<std::uint64_t>(compact_slot) >=
                        route_slot_limit)
                {
                    atomicExch(&block_valid, 0u);
                }
            }
        }
        __syncthreads();

        if (threadIdx.x != 0u || block_header_ready == 0u)
            return;
        if (block_valid == 0u)
        {
            publishFailure(
                control,
                launch.grant,
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::PayloadMismatch,
                launch.stage_ordinal,
                launch.model_layer_index,
                block_return_timeline);
            return;
        }
        launch.grant->live_rows = block_live_rows;
        /* Descriptor equality above authenticated the already-admitted entry
         * count; retain it as the payload extent for materialization. */
        launch.grant->single_row_id =
            block_live_rows == 1u
                ? loadPeerPublished(launch.dispatch.row_ids)
                : -1;
        publishSuccess(
            control,
            launch.grant,
            endpoint,
            operation,
            launch.stage_ordinal,
            launch.model_layer_index,
            block_timeline,
            /*consumed=*/true,
            /*published=*/true);
    }

    /**
     * @brief Materialize one authenticated participant's original route rows.
     *
     * Each compact entry names one unique original route slot.  The follower
     * already wrote that slot into the rank-pair shared matrix; this kernel
     * copies it into continuation VRAM without performing an addition.  The
     * later canonical reducer consequently sees the same slot tensor for every
     * placement and owns the only floating-point fold.
     */
    static __global__ void materializeMappedCanonicalReturnKernel(
        MoEOverlayActivationReturnConsumeLaunch launch)
    {
        __shared__ std::uint32_t block_endpoint_state;
        __shared__ std::uint32_t block_status_code;
        __shared__ std::int32_t block_last_consumed_stage;
        __shared__ std::uint64_t block_live_entries;
        if (threadIdx.x == 0u)
        {
            block_endpoint_state = launch.grant->state;
            block_status_code = launch.grant->code;
            block_last_consumed_stage = launch.grant->last_consumed_stage;
            block_live_entries = launch.grant->live_entries;
        }
        __syncthreads();

        const std::size_t element =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const auto state = static_cast<MoEOverlayActivationEndpointState>(
            block_endpoint_state);
        if ((state != MoEOverlayActivationEndpointState::Active &&
             state != MoEOverlayActivationEndpointState::Complete) ||
            block_status_code !=
                raw(MoEOverlayActivationStatusCode::Success) ||
            block_last_consumed_stage !=
                static_cast<std::int32_t>(launch.stage_ordinal))
        {
            return;
        }
        if (block_live_entries >
                launch.returned.route_slot_capacity ||
            element >= block_live_entries *
                           static_cast<std::uint64_t>(
                               launch.returned.d_model))
        {
            return;
        }
        const std::size_t compact_entry =
            element / static_cast<std::size_t>(launch.returned.d_model);
        const std::size_t column =
            element % static_cast<std::size_t>(launch.returned.d_model);
        const std::int32_t original_slot = loadPeerPublished(
            launch.dispatch.original_route_slots + compact_entry);
        if (original_slot < 0 ||
            static_cast<std::size_t>(original_slot) >=
                launch.returned.route_slot_capacity)
        {
            return;
        }
        const std::size_t destination =
            static_cast<std::size_t>(original_slot) *
                static_cast<std::size_t>(launch.returned.d_model) +
            column;
        const float returned_value = loadPeerPublished(
            launch.returned.canonical_route_contributions_fp32 +
            destination);
        launch.canonical_route_contributions_fp32[destination] =
            returned_value;
    }

    /**
     * @brief Acquire and materialize one colocated CPU canonical-route ticket.
     *
     * The CPU has already authenticated the sparse packet, computed one raw
     * expert result per compact route, multiplied each row by its router
     * weight, and release-published a new sequence. Every block waits for
     * exactly one sequence beyond the GPU-owned consumed cursor before reading
     * mapped payload bytes. The kernel performs only a route-slot permutation
     * into continuation VRAM; the following canonical reducer remains the sole
     * floating-point fold authority.
     */
    static __global__ void materializeCanonicalRouteTicketKernel(
        MoEOverlayCanonicalRouteTicketConsumeLaunch launch)
    {
        __shared__ std::uint64_t block_live_entries;
        __shared__ std::uint32_t block_ticket_valid;
        if (threadIdx.x == 0u)
        {
            std::uint64_t published = 0u;
            std::uint64_t consumed = 0u;
            do
            {
                consumed = loadSystemAcquire64(
                    &launch.control->consumed_sequence);
                published = loadSystemAcquire64(
                    &launch.control->published_sequence);
                if (published > consumed && published - consumed == 1u)
                    break;
#if defined(__CUDA_ARCH__)
                __nanosleep(64u);
#elif defined(__HIP_DEVICE_COMPILE__)
                __builtin_amdgcn_s_sleep(1u);
#endif
            } while (true);
            const auto control = snapshotPeerPublished(launch.control);
            const auto publication_status =
                static_cast<MoEOverlayCanonicalRouteTicketStatus>(
                    control.publication_status);
            const bool success_payload =
                publication_status ==
                MoEOverlayCanonicalRouteTicketStatus::Success;
            const bool abort_payload =
                publication_status ==
                MoEOverlayCanonicalRouteTicketStatus::Aborted;
            block_ticket_valid =
                success_payload && control.valid() &&
                        control.publicationPending() &&
                        control.residency_epoch != 0u &&
                        control.route_capacity ==
                            static_cast<std::int32_t>(
                                launch.route_capacity) &&
                        control.d_model == launch.d_model &&
                        control.live_entry_count <= launch.route_capacity
                    ? 1u
                    : 0u;
            /* An authenticated abort deliberately materializes zero rows. The
             * following acknowledgement kernel still advances the sequence,
             * allowing every later wait in a failed retained parent to drain. */
            block_live_entries =
                abort_payload && control.valid() &&
                        control.publicationPending() &&
                        control.live_entry_count == 0u
                    ? 0u
                    : control.live_entry_count;
        }
        __syncthreads();
        if (block_ticket_valid == 0u)
            return;

        const std::size_t element_count =
            static_cast<std::size_t>(block_live_entries) *
            static_cast<std::size_t>(launch.d_model);
        for (std::size_t element =
                 static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                 threadIdx.x;
             element < element_count;
             element += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        {
            const std::size_t compact_entry =
                element / static_cast<std::size_t>(launch.d_model);
            const std::size_t column =
                element % static_cast<std::size_t>(launch.d_model);
            const std::int32_t original_slot = loadPeerPublished(
                launch.original_route_slots + compact_entry);
            const std::int32_t compact_slot = loadPeerPublished(
                launch.compact_route_slots + compact_entry);
            if (original_slot < 0 || compact_slot < 0 ||
                static_cast<std::size_t>(original_slot) >=
                    launch.route_capacity ||
                static_cast<std::size_t>(compact_slot) >=
                    launch.route_capacity)
            {
                continue;
            }
            const std::size_t source =
                static_cast<std::size_t>(compact_slot) *
                    static_cast<std::size_t>(launch.d_model) +
                column;
            const std::size_t destination =
                static_cast<std::size_t>(original_slot) *
                    static_cast<std::size_t>(launch.d_model) +
                column;
            launch.canonical_route_contributions_fp32[destination] =
                loadPeerPublished(
                    launch.compact_preweighted_contributions_fp32 + source);
        }
    }

    /**
     * @brief Release-acknowledge a completely materialized CPU route payload.
     *
     * Backends enqueue this one-thread kernel immediately after every
     * @ref materializeCanonicalRouteTicketKernel launch on the same exact
     * stream. The stream edge guarantees every block has finished reading the
     * mapped payload and writing continuation VRAM before the acknowledgement
     * becomes system-visible. Only then may the CPU producer arm and overwrite
     * the single reusable ticket for another retained graph replay.
     */
    static __global__ void acknowledgeCanonicalRouteTicketKernel(
        MoEOverlayCanonicalRouteTicketConsumeLaunch launch)
    {
        if (blockIdx.x != 0u || threadIdx.x != 0u)
            return;
        const auto control = snapshotPeerPublished(launch.control);
        const auto publication_status =
            static_cast<MoEOverlayCanonicalRouteTicketStatus>(
                control.publication_status);
        const bool success_payload =
            publication_status ==
            MoEOverlayCanonicalRouteTicketStatus::Success;
        const bool abort_payload =
            publication_status ==
            MoEOverlayCanonicalRouteTicketStatus::Aborted;
        if (!control.valid() || !control.publicationPending() ||
            control.route_capacity !=
                static_cast<std::int32_t>(launch.route_capacity) ||
            control.d_model != launch.d_model ||
            control.live_entry_count > launch.route_capacity ||
            (success_payload && control.residency_epoch == 0u) ||
            (abort_payload &&
             (control.residency_epoch != 0u ||
              control.live_entry_count != 0u)) ||
            (!success_payload && !abort_payload))
        {
            return;
        }
        storeSystemRelease64(
            MoEOverlayActivationTimelinePublishDeviceBinding{
                .signal = &launch.control->consumed_sequence,
                .value = control.published_sequence,
            });
    }

    /**
     * @brief Authenticate independent multi-row lanes and build dense row maps.
     *
     * One block owns one lane, so descriptor arrival and validation never
     * serialize unrelated participants.  The row lookup is published only
     * through @ref lane_valid after the complete descriptor has passed the same
     * checks as @ref validateReturnKernel.  A bad lane can therefore leave
     * speculative lookup values behind without exposing them to the fold.
     */
    static __global__ void validateMultiRowReturnBatchKernel(
        MoEOverlayActivationMultiRowReturnBatchLaunch launch)
    {
        const std::uint32_t lane = blockIdx.x;
        if (lane >= launch.lane_count)
            return;

        const auto packet = launch.lanes[lane];

        __shared__ std::uint32_t block_header_ready;
        __shared__ std::uint32_t block_valid;
        __shared__ std::uint64_t block_live_rows;
        __shared__ std::uint64_t block_live_entries;
        __shared__ std::uint64_t block_timeline;
        __shared__ std::uint64_t block_return_timeline;
        auto *const control =
            const_cast<MoEOverlayActivationEpochControl *>(packet.control);
        constexpr auto endpoint =
            MoEOverlayActivationEndpoint::Continuation;
        constexpr auto operation =
            MoEOverlayActivationOperation::ConsumeReturn;
        if (threadIdx.x == 0u)
        {
            launch.lane_valid[lane] = 0;
            block_header_ready = 0u;
            block_valid = 0u;
            block_live_rows = 0u;
            block_live_entries = 0u;
            block_timeline = moeOverlayActivationLeasedTimelineValue(
                moeOverlayActivationBufferVisit(packet.stage_ordinal));
            block_return_timeline = 0u;
            if (!acquireOrValidateGrant(
                    control,
                    packet.grant,
                    endpoint,
                    operation,
                    packet.stage_ordinal,
                    packet.physical_rows))
            {
                publishFailure(
                    control,
                    packet.grant,
                    endpoint,
                    operation,
                    MoEOverlayActivationStatusCode::InvalidControl,
                    packet.stage_ordinal,
                    packet.model_layer_index,
                    0u);
            }
            else
            {
                const std::int32_t expected_previous =
                    packet.stage_ordinal == 0u
                        ? -1
                        : static_cast<std::int32_t>(
                              packet.stage_ordinal - 1u);
                if (packet.grant->last_published_stage <
                        static_cast<std::int32_t>(packet.stage_ordinal) ||
                    packet.grant->last_consumed_stage != expected_previous)
                {
                    publishFailure(
                        control,
                        packet.grant,
                        endpoint,
                        operation,
                        MoEOverlayActivationStatusCode::OutOfOrder,
                        packet.stage_ordinal,
                        packet.model_layer_index,
                        0u);
                }
                else
                {
                    const auto &buffer = control->buffers[
                        moeOverlayActivationBufferIndex(
                            packet.stage_ordinal)];
                    block_return_timeline = loadSystemAcquire64(
                        &buffer.return_signal.value);
                    const auto descriptor = snapshotPeerPublished(
                        &buffer.return_descriptor);
                    block_live_rows = descriptor.live_rows;
                    block_live_entries = descriptor.live_entries;
                    block_header_ready = 1u;
                    block_valid =
                        block_return_timeline == block_timeline &&
                                validDescriptor(
                                    descriptor,
                                    *packet.grant,
                                    packet.stage_ordinal,
                                    packet.model_layer_index,
                                    block_timeline) &&
                                descriptor.live_entries ==
                                    packet.grant->live_entries &&
                                descriptor.live_rows ==
                                    packet.grant->live_rows &&
                                descriptor.live_entries <=
                                    packet.returned.route_slot_capacity &&
                                descriptor.live_entries <=
                                    descriptor.live_rows *
                                        static_cast<std::uint64_t>(
                                            packet.dispatch.top_k) &&
                                descriptor.live_rows <=
                                    static_cast<std::uint64_t>(
                                        launch.physical_rows) &&
                                descriptor.payload_bytes ==
                                    moeOverlayReturnPayloadBytes(
                                        descriptor.live_entries,
                                        static_cast<std::uint32_t>(
                                            packet.returned.d_model))
                            ? 1u
                            : 0u;
                }
            }
        }
        __syncthreads();

        if (block_header_ready != 0u && block_valid != 0u)
        {
            const std::uint64_t route_slot_limit =
                static_cast<std::uint64_t>(launch.physical_rows) *
                static_cast<std::uint64_t>(packet.dispatch.top_k);
            for (std::uint64_t entry = threadIdx.x;
                 entry < block_live_entries;
                 entry += blockDim.x)
            {
                const std::int32_t original_slot = loadPeerPublished(
                    packet.dispatch.original_route_slots + entry);
                const std::int32_t compact_slot = loadPeerPublished(
                    packet.dispatch.compact_route_slots + entry);
                const std::int32_t prior_original_slot =
                    entry == 0u
                        ? -1
                        : loadPeerPublished(
                              packet.dispatch.original_route_slots +
                              entry - 1u);
                if (original_slot <= prior_original_slot ||
                    original_slot < 0 || compact_slot < 0 ||
                    static_cast<std::uint64_t>(original_slot) >=
                        route_slot_limit ||
                    static_cast<std::uint64_t>(compact_slot) >=
                        route_slot_limit)
                {
                    atomicExch(&block_valid, 0u);
                }
            }
        }
        __syncthreads();

        if (threadIdx.x != 0u || block_header_ready == 0u)
            return;
        if (block_valid == 0u)
        {
            publishFailure(
                control,
                packet.grant,
                endpoint,
                operation,
                MoEOverlayActivationStatusCode::PayloadMismatch,
                packet.stage_ordinal,
                packet.model_layer_index,
                block_return_timeline);
            return;
        }
        packet.grant->live_rows = block_live_rows;
        packet.grant->live_entries = block_live_entries;
        packet.grant->single_row_id =
            block_live_rows == 1u
                ? loadPeerPublished(packet.dispatch.row_ids)
                : -1;
        publishSuccess(
            control,
            packet.grant,
            endpoint,
            operation,
            packet.stage_ordinal,
            packet.model_layer_index,
            block_timeline,
            /*consumed=*/true,
            /*published=*/true);
        launch.lane_valid[lane] = 1;
    }

    /**
     * @brief Copy authenticated lane contributions into original route slots.
     *
     * Grid Y owns a participant lane and grid X stripes that lane's compact
     * contribution payload. Dynamic placement guarantees one authoritative
     * participant per original slot, so stores are disjoint and require no
     * atomics or cross-lane ordering.
     */
    static __global__ void materializeMultiRowCanonicalReturnBatchKernel(
        MoEOverlayActivationMultiRowReturnBatchLaunch launch)
    {
        const std::uint32_t lane = blockIdx.y;
        if (lane >= launch.lane_count || launch.lane_valid[lane] == 0)
            return;
        const auto packet = launch.lanes[lane];
        const std::size_t output_elements =
            static_cast<std::size_t>(packet.grant->live_entries) *
            static_cast<std::size_t>(launch.d_model);
        for (std::size_t element =
                 static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                 threadIdx.x;
             element < output_elements;
             element += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        {
            const std::size_t compact_entry =
                element / static_cast<std::size_t>(launch.d_model);
            const std::size_t column =
                element % static_cast<std::size_t>(launch.d_model);
            const std::int32_t original_slot = loadPeerPublished(
                packet.dispatch.original_route_slots + compact_entry);
            if (original_slot >= 0 &&
                static_cast<std::size_t>(original_slot) <
                    packet.returned.route_slot_capacity)
            {
                const std::size_t destination =
                    static_cast<std::size_t>(original_slot) *
                        static_cast<std::size_t>(launch.d_model) +
                    column;
                launch.canonical_route_contributions_fp32[destination] =
                    loadPeerPublished(
                        packet.returned
                            .canonical_route_contributions_fp32 +
                        destination);
            }
        }
    }

    /**
     * @brief Compact, copy, and release one direct-mapped dispatch in one node.
     *
     * One block is sufficient because physical row count is exactly one. Thread
     * zero retains the scalar protocol arithmetic; all threads stripe the hidden
     * width, meet at a block barrier, and only then publish the system timeline.
     * The release is unconditional so a semantic packet failure still drains the
     * peer graph instead of leaving it resident in a permanent wait.
     */
    __device__ __forceinline__ void packSingleRowDispatchBody(
        MoEOverlayActivationSingleRowDispatchPackLaunch launch)
    {
        auto &packet = launch.packet;
        __shared__ std::uint64_t live_rows;
        __shared__ std::int32_t source_row;
        __shared__ std::uint32_t packet_active;
        if (threadIdx.x == 0u)
        {
            packDispatchMetadata(packet);
            live_rows = packet.grant->live_rows;
            source_row = packet.grant->single_row_id;
            packet_active =
                packet.grant->state ==
                    raw(MoEOverlayActivationEndpointState::Active) &&
                packet.grant->code ==
                    raw(MoEOverlayActivationStatusCode::Success)
                    ? 1u
                    : 0u;
        }
        __syncthreads();

        if (packet_active != 0u && live_rows == 1u && source_row >= 0)
        {
            for (std::size_t column = threadIdx.x;
                 column < static_cast<std::size_t>(packet.packet.d_model);
                 column += blockDim.x)
            {
                packet.packet.hidden_rows_fp32[column] =
                    packet.hidden_rows_fp32[
                        static_cast<std::size_t>(source_row) *
                            static_cast<std::size_t>(packet.packet.d_model) +
                        column];
            }
        }
        __syncthreads();
        if (threadIdx.x == 0u)
        {
            __threadfence_system();
            storeSystemRelease64(launch.publication);
        }
    }

    /** @brief Launch-compatible wrapper for one direct-mapped dispatch lane. */
    static __global__ void packSingleRowDispatchKernel(
        MoEOverlayActivationSingleRowDispatchPackLaunch launch)
    {
        if (blockIdx.x == 0u)
            packSingleRowDispatchBody(launch);
    }

    /**
     * @brief Compact and publish every independent continuation lane in parallel.
     *
     * Lane descriptors are already ordered by the topology planner. Dispatch
     * has no cross-lane arithmetic, so assigning one block to each descriptor
     * removes serial launch and mapped-store latency without changing any lane's
     * packet bytes or timeline protocol.
     */
    static __global__ void packSingleRowDispatchBatchKernel(
        MoEOverlayActivationSingleRowDispatchBatchLaunch launch)
    {
        const std::uint32_t lane = blockIdx.x;
        if (lane < launch.lane_count)
            packSingleRowDispatchBody(launch.lanes[lane]);
    }

    /**
     * @brief Acquire and materialize one direct-mapped dispatch in one node.
     *
     * Thread zero owns the potentially long peer wait and authenticated scalar
     * validation. Once it publishes the local live-row snapshot through shared
     * memory, the block copies hidden columns and initializes every route slot.
     * Thus downstream expert compute observes the same tensors as the ordinary
     * validator/materializer pair, with two graph nodes removed.
     */
    static __global__ void consumeSingleRowDispatchKernel(
        MoEOverlayActivationSingleRowDispatchConsumeLaunch launch)
    {
        auto &packet = launch.packet;
        __shared__ std::int32_t live_rows;
        if (threadIdx.x == 0u)
        {
            waitSystemAcquire64(launch.acquire);
            validateDispatch(packet);
            live_rows =
                packet.grant->state ==
                        raw(MoEOverlayActivationEndpointState::Active) &&
                    packet.grant->code ==
                        raw(MoEOverlayActivationStatusCode::Success)
                    ? static_cast<std::int32_t>(packet.grant->live_rows)
                    : 0;
        }
        __syncthreads();

        for (std::size_t column = threadIdx.x;
             column < static_cast<std::size_t>(packet.packet.d_model);
             column += blockDim.x)
        {
            packet.hidden_rows_fp32[column] =
                live_rows == 1
                    ? loadPeerPublished(
                          packet.packet.hidden_rows_fp32 + column)
                    : 0.0f;
        }
        for (std::int32_t slot = static_cast<std::int32_t>(threadIdx.x);
             slot < packet.packet.top_k;
             slot += static_cast<std::int32_t>(blockDim.x))
        {
            std::int32_t expert = -1;
            float weight = 0.0f;
            if (live_rows == 1)
            {
                const std::int32_t begin = loadPeerPublished(
                    packet.packet.entry_offsets);
                const std::int32_t end = loadPeerPublished(
                    packet.packet.entry_offsets + 1u);
                if (begin + slot < end)
                {
                    expert = loadPeerPublished(
                        packet.packet.expert_ids + begin + slot);
                    weight = loadPeerPublished(
                        packet.packet.route_weights + begin + slot);
                }
            }
            packet.routing_indices_fp32[slot] = static_cast<float>(expert);
            packet.routing_weights_fp32[slot] = weight;
        }
    }

    /**
     * @brief Pack and release one direct-mapped participant return in one node.
     *
     * Metadata is finalized first, the block stripes live output columns into
     * mapped pages, and a system fence plus release store exposes the complete
     * packet to the continuation endpoint. Empty routes still publish their
     * descriptor and timeline so graph topology is independent of routing skew.
     */
    static __global__ void packSingleRowReturnKernel(
        MoEOverlayActivationSingleRowReturnPackLaunch launch)
    {
        auto &packet = launch.packet;
        __shared__ std::uint64_t live_entries;
        __shared__ std::uint32_t packet_valid;
        if (threadIdx.x == 0u)
        {
            packReturnMetadata(packet);
            const auto state = static_cast<MoEOverlayActivationEndpointState>(
                packet.grant->state);
            packet_valid =
                (state == MoEOverlayActivationEndpointState::Active ||
                 state == MoEOverlayActivationEndpointState::Complete) &&
                    packet.grant->code ==
                        raw(MoEOverlayActivationStatusCode::Success) &&
                    packet.grant->last_published_stage ==
                        static_cast<std::int32_t>(packet.stage_ordinal)
                    ? 1u
                    : 0u;
            live_entries = packet_valid != 0u
                               ? packet.grant->live_entries
                               : 0u;
        }
        __syncthreads();

        for (std::size_t element = threadIdx.x;
             element <
                 static_cast<std::size_t>(live_entries) *
                     static_cast<std::size_t>(packet.returned.d_model);
             element += blockDim.x)
        {
            const std::size_t compact_entry =
                element /
                static_cast<std::size_t>(packet.returned.d_model);
            const std::size_t column =
                element %
                static_cast<std::size_t>(packet.returned.d_model);
            const std::int32_t original_slot = loadPeerPublished(
                packet.dispatch.original_route_slots + compact_entry);
            const std::int32_t compact_slot = loadPeerPublished(
                packet.dispatch.compact_route_slots + compact_entry);
            if (original_slot >= 0 && compact_slot >= 0 &&
                static_cast<std::size_t>(original_slot) <
                    packet.returned.route_slot_capacity)
            {
                packet.returned.canonical_route_contributions_fp32[
                    static_cast<std::size_t>(original_slot) *
                        static_cast<std::size_t>(
                            packet.returned.d_model) +
                    column] =
                    packet.local_canonical_route_contributions_fp32[
                        static_cast<std::size_t>(compact_slot) *
                            static_cast<std::size_t>(
                                packet.returned.d_model) +
                        column];
            }
        }
        __syncthreads();
        if (threadIdx.x == 0u)
        {
            __threadfence_system();
            storeSystemRelease64(launch.publication);
        }
    }

    /**
     * @brief Acquire, validate, and materialize one direct-mapped return.
     */
    static __global__ void consumeSingleRowReturnKernel(
        MoEOverlayActivationSingleRowReturnConsumeLaunch launch)
    {
        auto &packet = launch.packet;
        __shared__ std::uint64_t live_entries;
        if (threadIdx.x == 0u)
        {
            waitSystemAcquire64(launch.acquire);
            validateReturn(packet);
            const auto state = static_cast<MoEOverlayActivationEndpointState>(
                packet.grant->state);
            const bool valid =
                (state == MoEOverlayActivationEndpointState::Active ||
                 state == MoEOverlayActivationEndpointState::Complete) &&
                packet.grant->code ==
                    raw(MoEOverlayActivationStatusCode::Success) &&
                packet.grant->last_consumed_stage ==
                    static_cast<std::int32_t>(packet.stage_ordinal);
            live_entries = valid ? packet.grant->live_entries : 0u;
        }
        __syncthreads();

        for (std::size_t element = threadIdx.x;
             element <
                 static_cast<std::size_t>(live_entries) *
                     static_cast<std::size_t>(packet.returned.d_model);
             element += blockDim.x)
        {
            const std::size_t compact_entry =
                element /
                static_cast<std::size_t>(packet.returned.d_model);
            const std::size_t column =
                element %
                static_cast<std::size_t>(packet.returned.d_model);
            const std::int32_t original_slot = loadPeerPublished(
                packet.dispatch.original_route_slots + compact_entry);
            if (original_slot >= 0 &&
                static_cast<std::size_t>(original_slot) <
                    packet.returned.route_slot_capacity)
            {
                const std::size_t destination =
                    static_cast<std::size_t>(original_slot) *
                        static_cast<std::size_t>(packet.returned.d_model) +
                    column;
                packet.canonical_route_contributions_fp32[destination] =
                    loadPeerPublished(
                        packet.returned
                            .canonical_route_contributions_fp32 +
                        destination);
            }
        }
    }

    /**
     * @brief Wait for and authenticate independent one-row returns concurrently.
     *
     * Each block touches one mapped lane, its private grant, and one exclusive
     * row in device scratch. The kernel as a whole is the device-side join: the
     * following fold cannot begin until every lane has either validated or
     * published its terminal protocol failure.
     */
    static __global__ void validateSingleRowReturnBatchKernel(
        MoEOverlayActivationSingleRowReturnBatchLaunch launch)
    {
        const std::uint32_t lane = blockIdx.x;
        if (lane >= launch.lane_count)
            return;

        const auto lane_launch = launch.lanes[lane];
        auto packet = lane_launch.packet;
        if (threadIdx.x == 0u)
        {
            waitSystemAcquire64(lane_launch.acquire);
            validateReturn(packet);
            const auto state = static_cast<MoEOverlayActivationEndpointState>(
                packet.grant->state);
            const bool valid =
                (state == MoEOverlayActivationEndpointState::Active ||
                 state == MoEOverlayActivationEndpointState::Complete) &&
                packet.grant->code ==
                    raw(MoEOverlayActivationStatusCode::Success) &&
                packet.grant->last_consumed_stage ==
                    static_cast<std::int32_t>(packet.stage_ordinal) &&
                packet.grant->live_rows <= 1u &&
                (packet.grant->live_rows == 0u ||
                 packet.grant->single_row_id == 0);
            launch.lane_valid[lane] = valid ? 1 : 0;
        }
    }

    /**
     * @brief Materialize authenticated one-row lane contributions in parallel.
     */
    static __global__ void materializeSingleRowCanonicalReturnBatchKernel(
        MoEOverlayActivationSingleRowReturnBatchLaunch launch)
    {
        const std::uint32_t lane = blockIdx.y;
        if (lane >= launch.lane_count || launch.lane_valid[lane] == 0)
            return;
        const auto packet = launch.lanes[lane].packet;
        const std::size_t live_elements =
            static_cast<std::size_t>(packet.grant->live_entries) *
            static_cast<std::size_t>(launch.d_model);
        for (std::size_t element =
                 static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                 threadIdx.x;
             element < live_elements;
             element += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        {
            const std::size_t compact_entry =
                element / static_cast<std::size_t>(launch.d_model);
            const std::size_t column =
                element % static_cast<std::size_t>(launch.d_model);
            const std::int32_t original_slot = loadPeerPublished(
                packet.dispatch.original_route_slots + compact_entry);
            if (original_slot >= 0 && original_slot < launch.top_k)
            {
                const std::size_t destination =
                    static_cast<std::size_t>(original_slot) *
                        static_cast<std::size_t>(launch.d_model) +
                    column;
                launch.canonical_route_contributions_fp32[destination] =
                    loadPeerPublished(
                        packet.returned
                            .canonical_route_contributions_fp32 +
                        destination);
            }
        }
    }
} // namespace llaminar2::moe_activation_packet_device
