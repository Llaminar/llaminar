/**
 * @file CUDAMoEOverlayEpochKernels.cu
 * @brief CUDA RCU primitives for captured ExpertOverlay placement epochs.
 *
 * Inference performs one bounded acquire before the first heterogeneous MoE
 * boundary and one bounded release after the final continuation segment.  A
 * background maintenance stream prepares and publishes the other bank without
 * waiting for those readers.  The global acquisition guard closes the only
 * reuse race: maintenance cannot reclaim an old bank while a reader has loaded
 * its selector but has not yet incremented the per-bank count.
 */

#include "CUDAMoEOverlayEpochKernels.h"

#include "execution/moe/MoEOverlayDeviceControllerABI.h"

#include <cuda/atomic>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

namespace
{
    using llaminar2::DeviceMoEOverlayEpochBankState;
    using llaminar2::DeviceMoEOverlayEpochControl;
    using llaminar2::DeviceMoEOverlayEpochOperation;
    using llaminar2::DeviceMoEOverlayEpochStatus;
    using llaminar2::DeviceMoEOverlayEpochStatusCode;
    using llaminar2::DeviceMoEOverlayEpochTicket;
    using llaminar2::DeviceMoEOverlayEpochAdmissionBarrierBinding;
    using llaminar2::MoEOverlayPeerPlacementEpochBinding;
    using llaminar2::MoEOverlayDeviceControllerInferenceEpochRecord;

    constexpr std::uint32_t kBankCount =
        llaminar2::kDeviceMoEOverlayEpochBankCount;
    constexpr std::uint32_t kInvalidBank =
        llaminar2::kDeviceMoEOverlayInvalidBank;

    /** @return Raw fixed-width value for a typed lifecycle. */
    __device__ __forceinline__ std::uint32_t rawState(
        DeviceMoEOverlayEpochBankState state)
    {
        return static_cast<std::uint32_t>(state);
    }

    /** @return Raw fixed-width value for a typed operation. */
    __device__ __forceinline__ std::uint32_t rawOperation(
        DeviceMoEOverlayEpochOperation operation)
    {
        return static_cast<std::uint32_t>(operation);
    }

    /** @return Raw fixed-width value for a typed result. */
    __device__ __forceinline__ std::uint32_t rawCode(
        DeviceMoEOverlayEpochStatusCode code)
    {
        return static_cast<std::uint32_t>(code);
    }

    /** @return Bank encoded in a selector without calling host-only helpers. */
    __device__ __forceinline__ std::uint32_t selectorBank(
        std::uint64_t selector)
    {
        return static_cast<std::uint32_t>(selector & 1u);
    }

    /** @return Publication generation encoded in a selector. */
    __device__ __forceinline__ std::uint64_t selectorGeneration(
        std::uint64_t selector)
    {
        return selector >> 1u;
    }

    /** @return One atomically publishable selector. */
    __device__ __forceinline__ std::uint64_t makeSelector(
        std::uint64_t generation,
        std::uint32_t bank)
    {
        return (generation << 1u) | static_cast<std::uint64_t>(bank & 1u);
    }

    /** @return Sequentially consistent-enough device load for one 64-bit word. */
    __device__ __forceinline__ std::uint64_t atomicLoad64(
        const std::uint64_t *value)
    {
        return static_cast<std::uint64_t>(atomicAdd(
            reinterpret_cast<unsigned long long *>(
                const_cast<std::uint64_t *>(value)),
            0ull));
    }

    /** @return System-scope acquire load for a mapped controller epoch. */
    __device__ __forceinline__ std::uint64_t systemAcquire64(
        const std::uint64_t *value)
    {
        cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> reference(
            *const_cast<std::uint64_t *>(value));
        return reference.load(cuda::memory_order_acquire);
    }

    /** @brief System-scope release store into one node-local mapped ABI word. */
    __device__ __forceinline__ void systemRelease64(
        std::uint64_t *value,
        std::uint64_t published)
    {
        cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> reference(
            *value);
        reference.store(published, cuda::memory_order_release);
    }

    /** @return Atomic device load for one 32-bit lifecycle/status word. */
    __device__ __forceinline__ std::uint32_t atomicLoad32(
        const std::uint32_t *value)
    {
        return atomicAdd(
            const_cast<unsigned int *>(
                reinterpret_cast<const unsigned int *>(value)),
            0u);
    }

    /** @brief Atomically replace one 64-bit ABI word. */
    __device__ __forceinline__ void atomicStore64(
        std::uint64_t *destination,
        std::uint64_t value)
    {
        atomicExch(
            reinterpret_cast<unsigned long long *>(destination),
            static_cast<unsigned long long>(value));
    }

    /** @brief Atomically replace one 32-bit ABI word. */
    __device__ __forceinline__ void atomicStore32(
        std::uint32_t *destination,
        std::uint32_t value)
    {
        atomicExch(reinterpret_cast<unsigned int *>(destination), value);
    }

    /** @brief Begin a status publication; final result code is written last. */
    __device__ __forceinline__ void beginStatus(
        DeviceMoEOverlayEpochStatus *status,
        DeviceMoEOverlayEpochOperation operation)
    {
        atomicStore32(
            &status->code,
            rawCode(DeviceMoEOverlayEpochStatusCode::Idle));
        status->epoch = 0u;
        status->selector = 0u;
        status->operation = rawOperation(operation);
        status->bank = kInvalidBank;
        status->observed_state =
            rawState(DeviceMoEOverlayEpochBankState::Empty);
    }

    /**
     * @brief Publish a complete operation result to later stream/device consumers.
     *
     * The status code is the release word.  Pollers that see a non-Idle value
     * are therefore guaranteed to see all accompanying provenance fields.
     */
    __device__ __forceinline__ void finishStatus(
        DeviceMoEOverlayEpochStatus *status,
        DeviceMoEOverlayEpochOperation operation,
        DeviceMoEOverlayEpochStatusCode code,
        std::uint64_t epoch,
        std::uint64_t selector,
        std::uint32_t bank,
        std::uint32_t observed_state)
    {
        status->epoch = epoch;
        status->selector = selector;
        status->operation = rawOperation(operation);
        status->bank = bank;
        status->observed_state = observed_state;
        __threadfence();
        atomicStore32(&status->code, rawCode(code));
    }

    /**
     * @brief Emit one failure-only epoch snapshot before the downstream guard aborts.
     *
     * Successful hot-path operations never call this helper. It keeps semantic
     * acquire/release failures attributable to their actual boundary instead of
     * letting a later MoE descriptor assertion obscure the causal state.
     */
    __device__ __forceinline__ void reportBoundaryFailure(
        const char *reason,
        const DeviceMoEOverlayEpochControl *control,
        const DeviceMoEOverlayEpochTicket *ticket,
        std::uint64_t required_epoch = 0u)
    {
        printf("overlay_epoch_boundary_failure reason=%s ticket=%p "
               "ticket_epoch=%llu ticket_selector=%llu required_epoch=%llu "
               "published_selector=%llu bank0_epoch=%llu bank1_epoch=%llu "
               "bank0_readers=%llu bank1_readers=%llu "
               "bank0_state=%u bank1_state=%u acquisitions=%llu\n",
               reason,
               ticket,
               static_cast<unsigned long long>(ticket->epoch),
               static_cast<unsigned long long>(ticket->selector),
               static_cast<unsigned long long>(required_epoch),
               static_cast<unsigned long long>(
                   atomicLoad64(&control->published_selector)),
               static_cast<unsigned long long>(
                   atomicLoad64(&control->bank_epochs[0])),
               static_cast<unsigned long long>(
                   atomicLoad64(&control->bank_epochs[1])),
               static_cast<unsigned long long>(
                   atomicLoad64(&control->bank_readers[0])),
               static_cast<unsigned long long>(
                   atomicLoad64(&control->bank_readers[1])),
               atomicLoad32(&control->bank_states[0]),
               atomicLoad32(&control->bank_states[1]),
               static_cast<unsigned long long>(
                   atomicLoad64(&control->acquisitions_in_flight)));
    }

    /** @return Bank retaining @p epoch, or the explicit invalid sentinel. */
    __device__ __forceinline__ std::uint32_t findEpochBank(
        const DeviceMoEOverlayEpochControl *control,
        std::uint64_t epoch)
    {
        if (epoch == 0u)
            return kInvalidBank;
        for (std::uint32_t bank = 0u; bank < kBankCount; ++bank)
        {
            if (atomicLoad64(&control->bank_epochs[bank]) == epoch)
                return bank;
        }
        return kInvalidBank;
    }

    /**
     * @brief Freeze one topology admission after every continuation GPU is guarded.
     *
     * The caller has already incremented its local `acquisitions_in_flight`.
     * Publishing arrival only after that increment prevents maintenance on this
     * participant from reclaiming either candidate bank while it waits for the
     * symmetric transaction epoch selected by the logical root.
     */
    __device__ __forceinline__ bool resolveRequiredEpoch(
        const std::uint64_t *external_admission_epoch,
        DeviceMoEOverlayEpochAdmissionBarrierBinding barrier,
        std::uint64_t *required_epoch)
    {
        if (!required_epoch)
            return false;
        if (!barrier.record)
        {
            if (barrier.participant_id != 0xffffffffu)
                return false;
            *required_epoch = external_admission_epoch
                                  ? systemAcquire64(external_admission_epoch)
                                  : 0u;
            return true;
        }

        MoEOverlayDeviceControllerInferenceEpochRecord *const record =
            barrier.record;
        constexpr std::uint32_t kParticipantCount =
            llaminar2::kMoEOverlayDeviceControllerInferenceEpochMaxParticipants;
        constexpr std::uint32_t kValidMask =
            (1u << kParticipantCount) - 1u;
        const std::uint32_t participant = barrier.participant_id;
        const std::uint32_t mask = record->participant_mask;
        const std::uint32_t publisher = record->publisher_participant_id;
        if (!external_admission_epoch ||
            record->magic !=
                llaminar2::kMoEOverlayDeviceControllerInferenceEpochMagic ||
            record->version !=
                llaminar2::kMoEOverlayDeviceControllerInferenceEpochVersion ||
            record->topology_fingerprint == 0u || mask == 0u ||
            (mask & ~kValidMask) != 0u || participant >= kParticipantCount ||
            publisher >= kParticipantCount ||
            (mask & (1u << participant)) == 0u ||
            (mask & (1u << publisher)) == 0u)
        {
            return false;
        }

        const std::uint64_t previous =
            systemAcquire64(&record->arrival_sequence[participant]);
        if (previous == UINT64_MAX)
            return false;
        const std::uint64_t sequence = previous + 1u;
        systemRelease64(&record->arrival_sequence[participant], sequence);

        if (participant == publisher)
        {
            for (std::uint32_t member = 0u; member < kParticipantCount; ++member)
            {
                if ((mask & (1u << member)) == 0u)
                    continue;
                while (systemAcquire64(&record->arrival_sequence[member]) <
                       sequence)
                {
                    // One control thread waits; all compute SMs remain available.
                }
            }

            const std::uint64_t published =
                systemAcquire64(&record->publication_sequence);
            if (published == UINT64_MAX || published + 1u != sequence)
            {
                /* Release a poison epoch when possible so followers terminate
                 * semantically instead of remaining in a stale wait forever. */
                if (published < sequence)
                {
                    systemRelease64(&record->epoch, 0u);
                    systemRelease64(
                        &record->publication_sequence, sequence);
                }
                return false;
            }
            const std::uint64_t frozen =
                systemAcquire64(external_admission_epoch);
            systemRelease64(&record->epoch, frozen);
            systemRelease64(&record->publication_sequence, sequence);
        }
        else
        {
            std::uint64_t published =
                systemAcquire64(&record->publication_sequence);
            while (published < sequence)
            {
                published = systemAcquire64(&record->publication_sequence);
            }
            if (published != sequence)
                return false;
        }

        *required_epoch = systemAcquire64(&record->epoch);
        return *required_epoch != 0u;
    }

    /**
     * @brief Wait for the exact new activation descriptor, then read its epoch.
     *
     * Bank-local packet timelines deliberately reuse small captured values.
     * The descriptor digest and strictly increasing activation generation are
     * therefore the anti-ABA authority; observing `dispatch_signal >= 1` alone
     * is insufficient after a channel lease has been reset.
     */
    __device__ __forceinline__ bool resolvePeerPlacementEpoch(
        MoEOverlayPeerPlacementEpochBinding binding,
        std::uint64_t *required_epoch)
    {
        if (!binding.valid() || !required_epoch)
            return false;

        const std::uint64_t previous_generation =
            atomicLoad64(&binding.grant->generation);
        const std::uint32_t bank =
            llaminar2::moeOverlayActivationBufferIndex(
                binding.stage_ordinal);
        const std::uint64_t expected_timeline =
            llaminar2::moeOverlayActivationLeasedTimelineValue(
                llaminar2::moeOverlayActivationBufferVisit(
                    binding.stage_ordinal));
        if (bank >= llaminar2::kMoEOverlayActivationBufferCount ||
            expected_timeline == 0u)
        {
            return false;
        }

        const auto *const control = binding.control;
        const auto *const descriptor =
            &control->buffers[bank].dispatch_descriptor;
        const auto *const signal =
            &control->buffers[bank].dispatch_signal.value;
        for (;;)
        {
            const std::uint64_t ready =
                systemAcquire64(&control->admission.ready_signal);
            const std::uint64_t generation =
                systemAcquire64(&control->identity.epoch_generation);
            if (ready < llaminar2::kMoEOverlayActivationAdmissionTimeline ||
                generation <= previous_generation)
            {
                __nanosleep(64u);
                continue;
            }

            const std::uint64_t observed_timeline =
                systemAcquire64(signal);
            if (observed_timeline ==
                llaminar2::kMoEOverlayActivationAbortTimeline)
            {
                return false;
            }
            if (observed_timeline < expected_timeline)
            {
                __nanosleep(64u);
                continue;
            }

            /* The signal is a system-release publication for the descriptor.
             * Recheck generation after copying its identity fields so a host
             * reset/rearm cannot splice two transactions into one snapshot. */
            const std::uint64_t descriptor_epoch =
                systemAcquire64(&descriptor->placement_epoch);
            const std::uint64_t descriptor_digest_low =
                systemAcquire64(&descriptor->digest.low);
            const std::uint64_t descriptor_digest_high =
                systemAcquire64(&descriptor->digest.high);
            const std::uint64_t identity_digest_low =
                systemAcquire64(&control->identity.digest.low);
            const std::uint64_t identity_digest_high =
                systemAcquire64(&control->identity.digest.high);
            const std::uint64_t generation_after =
                systemAcquire64(&control->identity.epoch_generation);
            if (generation == generation_after && descriptor_epoch != 0u &&
                descriptor->timeline == expected_timeline &&
                descriptor->stage_ordinal == binding.stage_ordinal &&
                descriptor_digest_low == identity_digest_low &&
                descriptor_digest_high == identity_digest_high)
            {
                *required_epoch = descriptor_epoch;
                return true;
            }
            __nanosleep(64u);
        }
    }

    /** @brief Install a request reader in the exact selector it observes. */
    __global__ void acquireEpochKernel(
        DeviceMoEOverlayEpochControl *control,
        DeviceMoEOverlayEpochTicket *ticket,
        DeviceMoEOverlayEpochStatus *status,
        const std::uint64_t *external_admission_epoch,
        DeviceMoEOverlayEpochAdmissionBarrierBinding admission_barrier,
        MoEOverlayPeerPlacementEpochBinding peer_placement_epoch)
    {
        if (blockIdx.x != 0u || threadIdx.x != 0u)
            return;
        beginStatus(status, DeviceMoEOverlayEpochOperation::Acquire);

        /* A live ticket here means the preceding request failed to release. */
        if (ticket->epoch != 0u || ticket->selector != 0u)
        {
            reportBoundaryFailure(
                "acquire_live_ticket", control, ticket);
            finishStatus(
                status,
                DeviceMoEOverlayEpochOperation::Acquire,
                DeviceMoEOverlayEpochStatusCode::InvalidTicket,
                ticket->epoch,
                ticket->selector,
                kInvalidBank,
                rawState(DeviceMoEOverlayEpochBankState::Empty));
            return;
        }

        atomicAdd(
            reinterpret_cast<unsigned long long *>(
                &control->acquisitions_in_flight),
            1ull);
        const std::uint64_t selector =
            atomicLoad64(&control->published_selector);
        const std::uint32_t published_bank = selectorBank(selector);
        const std::uint64_t generation = selectorGeneration(selector);
        if (generation == 0u || published_bank >= kBankCount)
        {
            atomicAdd(
                reinterpret_cast<unsigned long long *>(
                    &control->acquisitions_in_flight),
                ~0ull);
            reportBoundaryFailure(
                "acquire_invalid_selector", control, ticket);
            finishStatus(
                status,
                DeviceMoEOverlayEpochOperation::Acquire,
                DeviceMoEOverlayEpochStatusCode::InvalidControl,
                0u,
                selector,
                kInvalidBank,
                rawState(DeviceMoEOverlayEpochBankState::Empty));
            return;
        }

        std::uint64_t required_epoch = 0u;
        bool resolved = true;
        if (peer_placement_epoch.valid())
        {
            resolved = !external_admission_epoch &&
                       !admission_barrier.record &&
                       admission_barrier.participant_id == 0xffffffffu &&
                       resolvePeerPlacementEpoch(
                           peer_placement_epoch, &required_epoch);
        }
        else if (admission_barrier.record)
        {
            resolved = resolveRequiredEpoch(
                external_admission_epoch,
                admission_barrier,
                &required_epoch);
        }
        else if (admission_barrier.participant_id != 0xffffffffu)
        {
            resolved = false;
        }
        else
        {
            required_epoch = external_admission_epoch
                                 ? systemAcquire64(external_admission_epoch)
                                 : atomicLoad64(
                                       &control->bank_epochs[published_bank]);
        }
        const std::uint32_t bank = findEpochBank(control, required_epoch);
        const std::uint64_t ticket_generation =
            bank == published_bank
                ? generation
                : (generation > 1u ? generation - 1u : 0u);
        if (!resolved || required_epoch == 0u || bank >= kBankCount ||
            ticket_generation == 0u)
        {
            atomicAdd(
                reinterpret_cast<unsigned long long *>(
                    &control->acquisitions_in_flight),
                ~0ull);
            reportBoundaryFailure(
                "acquire_unresolved_epoch",
                control,
                ticket,
                required_epoch);
            finishStatus(
                status,
                DeviceMoEOverlayEpochOperation::Acquire,
                DeviceMoEOverlayEpochStatusCode::InvalidControl,
                required_epoch,
                selector,
                kInvalidBank,
                rawState(DeviceMoEOverlayEpochBankState::Empty));
            return;
        }

        const std::uint32_t state =
            atomicLoad32(&control->bank_states[bank]);
        const std::uint64_t epoch =
            atomicLoad64(&control->bank_epochs[bank]);
        if (epoch == 0u || epoch != required_epoch ||
            (state != rawState(DeviceMoEOverlayEpochBankState::Published) &&
             state != rawState(DeviceMoEOverlayEpochBankState::Retiring)))
        {
            atomicAdd(
                reinterpret_cast<unsigned long long *>(
                    &control->acquisitions_in_flight),
                ~0ull);
            reportBoundaryFailure(
                "acquire_unpublished_bank",
                control,
                ticket,
                required_epoch);
            finishStatus(
                status,
                DeviceMoEOverlayEpochOperation::Acquire,
                DeviceMoEOverlayEpochStatusCode::InvalidControl,
                epoch,
                selector,
                bank,
                state);
            return;
        }

        atomicAdd(
            reinterpret_cast<unsigned long long *>(
                &control->bank_readers[bank]),
            1ull);
        ticket->epoch = epoch;
        ticket->selector = makeSelector(ticket_generation, bank);
        __threadfence();
        atomicAdd(
            reinterpret_cast<unsigned long long *>(
                &control->acquisitions_in_flight),
            ~0ull);
        finishStatus(
            status,
            DeviceMoEOverlayEpochOperation::Acquire,
            DeviceMoEOverlayEpochStatusCode::Success,
            epoch,
            selector,
            bank,
            state);
    }

    /** @brief Drop exactly one reader and clear its reusable ticket storage. */
    __global__ void releaseEpochKernel(
        DeviceMoEOverlayEpochControl *control,
        DeviceMoEOverlayEpochTicket *ticket,
        DeviceMoEOverlayEpochStatus *status)
    {
        if (blockIdx.x != 0u || threadIdx.x != 0u)
            return;
        beginStatus(status, DeviceMoEOverlayEpochOperation::Release);

        const std::uint64_t epoch = ticket->epoch;
        const std::uint64_t selector = ticket->selector;
        const std::uint32_t bank = selectorBank(selector);
        if (epoch == 0u || selectorGeneration(selector) == 0u ||
            bank >= kBankCount ||
            atomicLoad64(&control->bank_epochs[bank]) != epoch)
        {
            reportBoundaryFailure(
                "release_invalid_ticket", control, ticket, epoch);
            finishStatus(
                status,
                DeviceMoEOverlayEpochOperation::Release,
                DeviceMoEOverlayEpochStatusCode::InvalidTicket,
                epoch,
                selector,
                bank < kBankCount ? bank : kInvalidBank,
                bank < kBankCount
                    ? atomicLoad32(&control->bank_states[bank])
                    : rawState(DeviceMoEOverlayEpochBankState::Empty));
            return;
        }

        auto *readers = reinterpret_cast<unsigned long long *>(
            &control->bank_readers[bank]);
        unsigned long long observed = atomicAdd(readers, 0ull);
        while (observed != 0ull)
        {
            const unsigned long long previous =
                atomicCAS(readers, observed, observed - 1ull);
            if (previous == observed)
            {
                ticket->epoch = 0u;
                ticket->selector = 0u;
                __threadfence();
                finishStatus(
                    status,
                    DeviceMoEOverlayEpochOperation::Release,
                    DeviceMoEOverlayEpochStatusCode::Success,
                    epoch,
                    selector,
                    bank,
                    atomicLoad32(&control->bank_states[bank]));
                return;
            }
            observed = previous;
        }

        finishStatus(
            status,
            DeviceMoEOverlayEpochOperation::Release,
            DeviceMoEOverlayEpochStatusCode::ReaderUnderflow,
            epoch,
            selector,
            bank,
            atomicLoad32(&control->bank_states[bank]));
        reportBoundaryFailure(
            "release_reader_underflow", control, ticket, epoch);
    }

    /** @brief Claim the non-published empty bank for background preparation. */
    __global__ void reserveCandidateKernel(
        DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch_ptr,
        DeviceMoEOverlayEpochStatus *status)
    {
        if (blockIdx.x != 0u || threadIdx.x != 0u)
            return;
        beginStatus(status, DeviceMoEOverlayEpochOperation::ReserveCandidate);

        const std::uint64_t candidate_epoch = *candidate_epoch_ptr;
        const std::uint64_t selector =
            atomicLoad64(&control->published_selector);
        const std::uint32_t published = selectorBank(selector);
        const std::uint64_t generation = selectorGeneration(selector);
        if (generation == 0u || published >= kBankCount)
        {
            finishStatus(
                status,
                DeviceMoEOverlayEpochOperation::ReserveCandidate,
                DeviceMoEOverlayEpochStatusCode::InvalidControl,
                candidate_epoch,
                selector,
                kInvalidBank,
                rawState(DeviceMoEOverlayEpochBankState::Empty));
            return;
        }

        const std::uint64_t published_epoch =
            atomicLoad64(&control->bank_epochs[published]);
        if (candidate_epoch == 0u || candidate_epoch <= published_epoch)
        {
            finishStatus(
                status,
                DeviceMoEOverlayEpochOperation::ReserveCandidate,
                DeviceMoEOverlayEpochStatusCode::InvalidArgument,
                candidate_epoch,
                selector,
                kInvalidBank,
                rawState(DeviceMoEOverlayEpochBankState::Empty));
            return;
        }

        const std::uint32_t candidate = 1u - published;
        std::uint32_t observed_state =
            atomicLoad32(&control->bank_states[candidate]);
        /* A prior publication may have left this physical bank Retiring while
         * an old inference ticket drained. Reclaim it opportunistically once
         * both halves of the grace period are empty; this keeps a captured
         * maintenance replay device-resident instead of requiring the host to
         * discover and name the predecessor epoch. */
        if (observed_state ==
                rawState(DeviceMoEOverlayEpochBankState::Retiring) &&
            atomicLoad64(&control->acquisitions_in_flight) == 0u &&
            atomicLoad64(&control->bank_readers[candidate]) == 0u)
        {
            const std::uint32_t retired = atomicCAS(
                reinterpret_cast<unsigned int *>(
                    &control->bank_states[candidate]),
                rawState(DeviceMoEOverlayEpochBankState::Retiring),
                rawState(DeviceMoEOverlayEpochBankState::Empty));
            if (retired ==
                rawState(DeviceMoEOverlayEpochBankState::Retiring))
            {
                atomicStore64(&control->bank_epochs[candidate], 0u);
                observed_state =
                    rawState(DeviceMoEOverlayEpochBankState::Empty);
            }
            else
            {
                observed_state = retired;
            }
        }
        if (atomicLoad64(&control->bank_readers[candidate]) != 0u ||
            atomicLoad64(&control->bank_epochs[candidate]) != 0u ||
            observed_state !=
                rawState(DeviceMoEOverlayEpochBankState::Empty))
        {
            finishStatus(
                status,
                DeviceMoEOverlayEpochOperation::ReserveCandidate,
                DeviceMoEOverlayEpochStatusCode::Busy,
                candidate_epoch,
                selector,
                candidate,
                observed_state);
            return;
        }

        const std::uint32_t previous = atomicCAS(
            reinterpret_cast<unsigned int *>(
                &control->bank_states[candidate]),
            rawState(DeviceMoEOverlayEpochBankState::Empty),
            rawState(DeviceMoEOverlayEpochBankState::Candidate));
        if (previous != rawState(DeviceMoEOverlayEpochBankState::Empty))
        {
            finishStatus(
                status,
                DeviceMoEOverlayEpochOperation::ReserveCandidate,
                DeviceMoEOverlayEpochStatusCode::Busy,
                candidate_epoch,
                selector,
                candidate,
                previous);
            return;
        }

        atomicStore64(&control->bank_epochs[candidate], candidate_epoch);
        finishStatus(
            status,
            DeviceMoEOverlayEpochOperation::ReserveCandidate,
            DeviceMoEOverlayEpochStatusCode::Success,
            candidate_epoch,
            selector,
            candidate,
            rawState(DeviceMoEOverlayEpochBankState::Candidate));
    }

    /** @brief Publish that all candidate descriptors and weight events are ready. */
    __global__ void markCandidateReadyKernel(
        DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch_ptr,
        DeviceMoEOverlayEpochStatus *status)
    {
        if (blockIdx.x != 0u || threadIdx.x != 0u)
            return;
        beginStatus(status, DeviceMoEOverlayEpochOperation::MarkCandidateReady);

        const std::uint64_t candidate_epoch = *candidate_epoch_ptr;
        const std::uint32_t bank = findEpochBank(control, candidate_epoch);
        const std::uint64_t selector =
            atomicLoad64(&control->published_selector);
        if (bank >= kBankCount)
        {
            finishStatus(
                status,
                DeviceMoEOverlayEpochOperation::MarkCandidateReady,
                DeviceMoEOverlayEpochStatusCode::NotReady,
                candidate_epoch,
                selector,
                kInvalidBank,
                rawState(DeviceMoEOverlayEpochBankState::Empty));
            return;
        }

        const std::uint32_t previous = atomicCAS(
            reinterpret_cast<unsigned int *>(&control->bank_states[bank]),
            rawState(DeviceMoEOverlayEpochBankState::Candidate),
            rawState(DeviceMoEOverlayEpochBankState::Ready));
        const bool ready =
            previous == rawState(DeviceMoEOverlayEpochBankState::Candidate);
        finishStatus(
            status,
            DeviceMoEOverlayEpochOperation::MarkCandidateReady,
            ready
                ? DeviceMoEOverlayEpochStatusCode::Success
                : DeviceMoEOverlayEpochStatusCode::NotReady,
            candidate_epoch,
            selector,
            bank,
            ready ? rawState(DeviceMoEOverlayEpochBankState::Ready) : previous);
    }

    /** @brief Switch admission to a Ready bank without waiting for old readers. */
    __global__ void publishCandidateKernel(
        DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch_ptr,
        DeviceMoEOverlayEpochStatus *status)
    {
        if (blockIdx.x != 0u || threadIdx.x != 0u)
            return;
        beginStatus(status, DeviceMoEOverlayEpochOperation::PublishCandidate);

        const std::uint64_t candidate_epoch = *candidate_epoch_ptr;
        const std::uint32_t candidate =
            findEpochBank(control, candidate_epoch);
        const std::uint64_t old_selector =
            atomicLoad64(&control->published_selector);
        const std::uint32_t previous = selectorBank(old_selector);
        const std::uint64_t old_generation =
            selectorGeneration(old_selector);
        if (candidate >= kBankCount || previous >= kBankCount ||
            candidate == previous || old_generation == 0u)
        {
            finishStatus(
                status,
                DeviceMoEOverlayEpochOperation::PublishCandidate,
                DeviceMoEOverlayEpochStatusCode::InvalidControl,
                candidate_epoch,
                old_selector,
                candidate,
                candidate < kBankCount
                    ? atomicLoad32(&control->bank_states[candidate])
                    : rawState(DeviceMoEOverlayEpochBankState::Empty));
            return;
        }
        if (old_generation == (~0ull >> 1u))
        {
            finishStatus(
                status,
                DeviceMoEOverlayEpochOperation::PublishCandidate,
                DeviceMoEOverlayEpochStatusCode::GenerationOverflow,
                candidate_epoch,
                old_selector,
                candidate,
                atomicLoad32(&control->bank_states[candidate]));
            return;
        }

        const std::uint32_t candidate_state =
            atomicLoad32(&control->bank_states[candidate]);
        const std::uint32_t previous_state =
            atomicLoad32(&control->bank_states[previous]);
        const std::uint64_t previous_epoch =
            atomicLoad64(&control->bank_epochs[previous]);
        if (candidate_state != rawState(DeviceMoEOverlayEpochBankState::Ready) ||
            previous_state !=
                rawState(DeviceMoEOverlayEpochBankState::Published) ||
            previous_epoch == 0u || candidate_epoch <= previous_epoch)
        {
            finishStatus(
                status,
                DeviceMoEOverlayEpochOperation::PublishCandidate,
                DeviceMoEOverlayEpochStatusCode::NotReady,
                candidate_epoch,
                old_selector,
                candidate,
                candidate_state);
            return;
        }

        const std::uint32_t claimed = atomicCAS(
            reinterpret_cast<unsigned int *>(
                &control->bank_states[candidate]),
            rawState(DeviceMoEOverlayEpochBankState::Ready),
            rawState(DeviceMoEOverlayEpochBankState::Published));
        if (claimed != rawState(DeviceMoEOverlayEpochBankState::Ready))
        {
            finishStatus(
                status,
                DeviceMoEOverlayEpochOperation::PublishCandidate,
                DeviceMoEOverlayEpochStatusCode::NotReady,
                candidate_epoch,
                old_selector,
                candidate,
                claimed);
            return;
        }

        /* Descriptor uploads happen-before this selector release publication. */
        __threadfence();
        const std::uint64_t selector =
            makeSelector(old_generation + 1u, candidate);
        atomicStore64(&control->published_selector, selector);
        atomicStore32(
            &control->bank_states[previous],
            rawState(DeviceMoEOverlayEpochBankState::Retiring));
        finishStatus(
            status,
            DeviceMoEOverlayEpochOperation::PublishCandidate,
            DeviceMoEOverlayEpochStatusCode::Success,
            candidate_epoch,
            selector,
            candidate,
            rawState(DeviceMoEOverlayEpochBankState::Published));
    }

    /** @brief Return an unpublished bank to Empty after a failed preparation. */
    __global__ void abortCandidateKernel(
        DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch_ptr,
        DeviceMoEOverlayEpochStatus *status)
    {
        if (blockIdx.x != 0u || threadIdx.x != 0u)
            return;
        beginStatus(status, DeviceMoEOverlayEpochOperation::AbortCandidate);

        const std::uint64_t candidate_epoch = *candidate_epoch_ptr;
        const std::uint32_t bank = findEpochBank(control, candidate_epoch);
        const std::uint64_t selector =
            atomicLoad64(&control->published_selector);
        if (bank >= kBankCount ||
            atomicLoad64(&control->bank_readers[bank]) != 0u)
        {
            finishStatus(
                status,
                DeviceMoEOverlayEpochOperation::AbortCandidate,
                DeviceMoEOverlayEpochStatusCode::NotReady,
                candidate_epoch,
                selector,
                bank,
                bank < kBankCount
                    ? atomicLoad32(&control->bank_states[bank])
                    : rawState(DeviceMoEOverlayEpochBankState::Empty));
            return;
        }

        auto *state = reinterpret_cast<unsigned int *>(
            &control->bank_states[bank]);
        std::uint32_t observed = atomicAdd(state, 0u);
        while (observed == rawState(DeviceMoEOverlayEpochBankState::Candidate) ||
               observed == rawState(DeviceMoEOverlayEpochBankState::Ready))
        {
            const std::uint32_t previous = atomicCAS(
                state,
                observed,
                rawState(DeviceMoEOverlayEpochBankState::Empty));
            if (previous == observed)
            {
                atomicStore64(&control->bank_epochs[bank], 0u);
                finishStatus(
                    status,
                    DeviceMoEOverlayEpochOperation::AbortCandidate,
                    DeviceMoEOverlayEpochStatusCode::Success,
                    candidate_epoch,
                    selector,
                    bank,
                    rawState(DeviceMoEOverlayEpochBankState::Empty));
                return;
            }
            observed = previous;
        }

        finishStatus(
            status,
            DeviceMoEOverlayEpochOperation::AbortCandidate,
            DeviceMoEOverlayEpochStatusCode::NotReady,
            candidate_epoch,
            selector,
            bank,
            observed);
    }

    /** @brief Reclaim an old bank only after both halves of its grace period. */
    __global__ void retireEpochKernel(
        DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *retiring_epoch_ptr,
        DeviceMoEOverlayEpochStatus *status)
    {
        if (blockIdx.x != 0u || threadIdx.x != 0u)
            return;
        beginStatus(status, DeviceMoEOverlayEpochOperation::Retire);

        const std::uint64_t retiring_epoch = *retiring_epoch_ptr;
        const std::uint32_t bank = findEpochBank(control, retiring_epoch);
        const std::uint64_t selector =
            atomicLoad64(&control->published_selector);
        if (bank >= kBankCount ||
            atomicLoad32(&control->bank_states[bank]) !=
                rawState(DeviceMoEOverlayEpochBankState::Retiring))
        {
            finishStatus(
                status,
                DeviceMoEOverlayEpochOperation::Retire,
                DeviceMoEOverlayEpochStatusCode::NotReady,
                retiring_epoch,
                selector,
                bank,
                bank < kBankCount
                    ? atomicLoad32(&control->bank_states[bank])
                    : rawState(DeviceMoEOverlayEpochBankState::Empty));
            return;
        }

        if (atomicLoad64(&control->acquisitions_in_flight) != 0u ||
            atomicLoad64(&control->bank_readers[bank]) != 0u)
        {
            finishStatus(
                status,
                DeviceMoEOverlayEpochOperation::Retire,
                DeviceMoEOverlayEpochStatusCode::Busy,
                retiring_epoch,
                selector,
                bank,
                rawState(DeviceMoEOverlayEpochBankState::Retiring));
            return;
        }

        const std::uint32_t previous = atomicCAS(
            reinterpret_cast<unsigned int *>(&control->bank_states[bank]),
            rawState(DeviceMoEOverlayEpochBankState::Retiring),
            rawState(DeviceMoEOverlayEpochBankState::Empty));
        if (previous != rawState(DeviceMoEOverlayEpochBankState::Retiring))
        {
            finishStatus(
                status,
                DeviceMoEOverlayEpochOperation::Retire,
                DeviceMoEOverlayEpochStatusCode::Busy,
                retiring_epoch,
                selector,
                bank,
                previous);
            return;
        }

        atomicStore64(&control->bank_epochs[bank], 0u);
        finishStatus(
            status,
            DeviceMoEOverlayEpochOperation::Retire,
            DeviceMoEOverlayEpochStatusCode::Success,
            retiring_epoch,
            selector,
            bank,
            rawState(DeviceMoEOverlayEpochBankState::Empty));
    }

    /** @brief Select the exact CUDA device and reject default-stream launches. */
    bool prepareLaunch(int device_ordinal, void *stream, const char *operation)
    {
        if (!stream)
        {
            std::fprintf(stderr, "%s requires a non-null CUDA stream\n", operation);
            return false;
        }
        const cudaError_t error = cudaSetDevice(device_ordinal);
        if (error != cudaSuccess)
        {
            std::fprintf(
                stderr,
                "%s could not select CUDA device %d: %s\n",
                operation,
                device_ordinal,
                cudaGetErrorString(error));
            return false;
        }
        return true;
    }

    /** @return Whether CUDA accepted the preceding asynchronous launch. */
    bool finishLaunch(const char *operation)
    {
        const cudaError_t error = cudaPeekAtLastError();
        if (error == cudaSuccess)
            return true;
        std::fprintf(
            stderr,
            "%s launch failed: %s\n",
            operation,
            cudaGetErrorString(error));
        return false;
    }
} // namespace

extern "C"
{
    bool cudaMoEOverlayEpochAcquire(
        DeviceMoEOverlayEpochControl *control,
        DeviceMoEOverlayEpochTicket *ticket,
        DeviceMoEOverlayEpochStatus *status,
        const std::uint64_t *external_admission_epoch,
        DeviceMoEOverlayEpochAdmissionBarrierBinding admission_barrier,
        MoEOverlayPeerPlacementEpochBinding peer_placement_epoch,
        int device_ordinal,
        void *stream)
    {
        if (!control || !ticket || !status ||
            !prepareLaunch(device_ordinal, stream, "cudaMoEOverlayEpochAcquire"))
            return false;
        acquireEpochKernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
            control,
            ticket,
            status,
            external_admission_epoch,
            admission_barrier,
            peer_placement_epoch);
        return finishLaunch("cudaMoEOverlayEpochAcquire");
    }

    bool cudaMoEOverlayEpochRelease(
        DeviceMoEOverlayEpochControl *control,
        DeviceMoEOverlayEpochTicket *ticket,
        DeviceMoEOverlayEpochStatus *status,
        int device_ordinal,
        void *stream)
    {
        if (!control || !ticket || !status ||
            !prepareLaunch(device_ordinal, stream, "cudaMoEOverlayEpochRelease"))
            return false;
        releaseEpochKernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
            control, ticket, status);
        return finishLaunch("cudaMoEOverlayEpochRelease");
    }

    bool cudaMoEOverlayEpochReserveCandidate(
        DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch,
        DeviceMoEOverlayEpochStatus *status,
        int device_ordinal,
        void *stream)
    {
        if (!control || !candidate_epoch || !status ||
            !prepareLaunch(
                device_ordinal, stream, "cudaMoEOverlayEpochReserveCandidate"))
            return false;
        reserveCandidateKernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
            control, candidate_epoch, status);
        return finishLaunch("cudaMoEOverlayEpochReserveCandidate");
    }

    bool cudaMoEOverlayEpochMarkCandidateReady(
        DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch,
        DeviceMoEOverlayEpochStatus *status,
        int device_ordinal,
        void *stream)
    {
        if (!control || !candidate_epoch || !status ||
            !prepareLaunch(
                device_ordinal, stream, "cudaMoEOverlayEpochMarkCandidateReady"))
            return false;
        markCandidateReadyKernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
            control, candidate_epoch, status);
        return finishLaunch("cudaMoEOverlayEpochMarkCandidateReady");
    }

    bool cudaMoEOverlayEpochPublishCandidate(
        DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch,
        DeviceMoEOverlayEpochStatus *status,
        int device_ordinal,
        void *stream)
    {
        if (!control || !candidate_epoch || !status ||
            !prepareLaunch(
                device_ordinal, stream, "cudaMoEOverlayEpochPublishCandidate"))
            return false;
        publishCandidateKernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
            control, candidate_epoch, status);
        return finishLaunch("cudaMoEOverlayEpochPublishCandidate");
    }

    bool cudaMoEOverlayEpochAbortCandidate(
        DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch,
        DeviceMoEOverlayEpochStatus *status,
        int device_ordinal,
        void *stream)
    {
        if (!control || !candidate_epoch || !status ||
            !prepareLaunch(
                device_ordinal, stream, "cudaMoEOverlayEpochAbortCandidate"))
            return false;
        abortCandidateKernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
            control, candidate_epoch, status);
        return finishLaunch("cudaMoEOverlayEpochAbortCandidate");
    }

    bool cudaMoEOverlayEpochRetire(
        DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *retiring_epoch,
        DeviceMoEOverlayEpochStatus *status,
        int device_ordinal,
        void *stream)
    {
        if (!control || !retiring_epoch || !status ||
            !prepareLaunch(device_ordinal, stream, "cudaMoEOverlayEpochRetire"))
            return false;
        retireEpochKernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
            control, retiring_epoch, status);
        return finishLaunch("cudaMoEOverlayEpochRetire");
    }
}
