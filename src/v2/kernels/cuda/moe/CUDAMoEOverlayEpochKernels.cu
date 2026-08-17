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

    /** @brief Install a request reader in the exact selector it observes. */
    __global__ void acquireEpochKernel(
        DeviceMoEOverlayEpochControl *control,
        DeviceMoEOverlayEpochTicket *ticket,
        DeviceMoEOverlayEpochStatus *status)
    {
        if (blockIdx.x != 0u || threadIdx.x != 0u)
            return;
        beginStatus(status, DeviceMoEOverlayEpochOperation::Acquire);

        /* A live ticket here means the preceding request failed to release. */
        if (ticket->epoch != 0u || ticket->selector != 0u)
        {
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
        const std::uint32_t bank = selectorBank(selector);
        const std::uint64_t generation = selectorGeneration(selector);
        if (generation == 0u || bank >= kBankCount)
        {
            atomicAdd(
                reinterpret_cast<unsigned long long *>(
                    &control->acquisitions_in_flight),
                ~0ull);
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

        const std::uint32_t state =
            atomicLoad32(&control->bank_states[bank]);
        const std::uint64_t epoch =
            atomicLoad64(&control->bank_epochs[bank]);
        if (epoch == 0u ||
            (state != rawState(DeviceMoEOverlayEpochBankState::Published) &&
             state != rawState(DeviceMoEOverlayEpochBankState::Retiring)))
        {
            atomicAdd(
                reinterpret_cast<unsigned long long *>(
                    &control->acquisitions_in_flight),
                ~0ull);
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
        ticket->selector = selector;
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
        int device_ordinal,
        void *stream)
    {
        if (!control || !ticket || !status ||
            !prepareLaunch(device_ordinal, stream, "cudaMoEOverlayEpochAcquire"))
            return false;
        acquireEpochKernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
            control, ticket, status);
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
