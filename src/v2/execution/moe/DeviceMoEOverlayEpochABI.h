/**
 * @file DeviceMoEOverlayEpochABI.h
 * @brief Device-friendly RCU ABI for captured ExpertOverlay placement epochs.
 *
 * A heterogeneous forward pass may leave a captured GPU segment, execute
 * sparse work on other participants, and later re-enter a captured continuation
 * segment.  The placement selected before the first boundary must remain
 * addressable for that entire interval even when background migration publishes
 * a newer epoch.  This header contains only fixed-width, trivially copyable
 * records so CPU, CUDA, and ROCm implementations can share the exact layout.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace llaminar2
{
    struct MoEOverlayDeviceControllerInferenceEpochRecord;

    /** Number of simultaneously retained ExpertOverlay execution banks. */
    inline constexpr std::uint32_t kDeviceMoEOverlayEpochBankCount = 2u;

    /** Sentinel used when an operation result does not name a physical bank. */
    inline constexpr std::uint32_t kDeviceMoEOverlayInvalidBank = 0xffffffffu;

    /** Lifecycle of one physical descriptor/weight bank. */
    enum class DeviceMoEOverlayEpochBankState : std::uint32_t
    {
        Empty = 0u,     ///< Reusable only after the prior epoch's grace period.
        Candidate = 1u, ///< Background preparation owns an unpublished bank.
        Ready = 2u,     ///< All descriptor uploads and weight events are ready.
        Published = 3u, ///< New device tickets select this bank.
        Retiring = 4u,  ///< Existing tickets may use it; new tickets select peer.
    };

    /** Operation that last published an ExpertOverlay epoch status record. */
    enum class DeviceMoEOverlayEpochOperation : std::uint32_t
    {
        None = 0u,
        Acquire = 1u,
        Release = 2u,
        ReserveCandidate = 3u,
        MarkCandidateReady = 4u,
        PublishCandidate = 5u,
        AbortCandidate = 6u,
        Retire = 7u,
    };

    /** Device-visible result of one epoch operation. */
    enum class DeviceMoEOverlayEpochStatusCode : std::uint32_t
    {
        Idle = 0u,
        Success = 1u,
        InvalidArgument = 2u,
        InvalidControl = 3u,
        InvalidTicket = 4u,
        Busy = 5u,
        NotReady = 6u,
        ReaderUnderflow = 7u,
        GenerationOverflow = 8u,
    };

    /**
     * @brief Pack a monotonic publication generation and selected bank atomically.
     * @param generation Positive publication generation.
     * @param bank Bank index in `[0, 1]`.
     * @return One selector whose low bit is the bank and upper bits are generation.
     */
    [[nodiscard]] constexpr std::uint64_t deviceMoEOverlayEpochSelector(
        std::uint64_t generation,
        std::uint32_t bank) noexcept
    {
        return (generation << 1u) |
               static_cast<std::uint64_t>(bank & 1u);
    }

    /** @return Bank index encoded in a publication selector. */
    [[nodiscard]] constexpr std::uint32_t deviceMoEOverlayEpochSelectorBank(
        std::uint64_t selector) noexcept
    {
        return static_cast<std::uint32_t>(selector & 1u);
    }

    /** @return Monotonic generation encoded in a publication selector. */
    [[nodiscard]] constexpr std::uint64_t
    deviceMoEOverlayEpochSelectorGeneration(std::uint64_t selector) noexcept
    {
        return selector >> 1u;
    }

    /**
     * @brief Device-owned publication and grace-period state for one model role.
     *
     * `published_selector` is the sole admission linearization point.  An
     * acquisition increments `acquisitions_in_flight` before reading it, then
     * installs a reader in the selected bank before dropping that guard.
     * Maintenance may reuse a retiring bank only when both the guard and that
     * bank's reader count are zero.  This closes the delayed-increment race
     * without a host mirror, stream synchronization, or inference-side lock.
     *
     * The record is cache-line aligned for the CPU implementation and naturally
     * aligned for 64-bit CUDA/HIP atomics.  GPU code must mutate these fields
     * through the backend's atomic protocol, never through an ordinary store.
     */
    struct alignas(64) DeviceMoEOverlayEpochControl
    {
        std::uint64_t bank_epochs[kDeviceMoEOverlayEpochBankCount] = {};
        std::uint64_t bank_readers[kDeviceMoEOverlayEpochBankCount] = {};
        std::uint64_t acquisitions_in_flight = 0u;
        std::uint32_t bank_states[kDeviceMoEOverlayEpochBankCount] = {};
        std::uint64_t published_selector = 0u;
        std::uint64_t reserved = 0u;
    };

    /**
     * @brief Request-lifetime immutable placement selected by captured admission.
     *
     * The selector retains both the physical bank and its ABA-resistant
     * generation.  `epoch` is copied to pinned heterogeneous dispatch tickets so
     * CPU and remote participants acquire the matching host residency bank.
     */
    struct DeviceMoEOverlayEpochTicket
    {
        std::uint64_t epoch = 0u;
        std::uint64_t selector = 0u;

        /** @return Whether epoch, generation, and bank encoding are usable. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            return epoch != 0u &&
                   deviceMoEOverlayEpochSelectorGeneration(selector) != 0u &&
                   deviceMoEOverlayEpochSelectorBank(selector) <
                       kDeviceMoEOverlayEpochBankCount;
        }

        /** @return Exact physical bank retained by this ticket. */
        [[nodiscard]] constexpr std::uint32_t bank() const noexcept
        {
            return deviceMoEOverlayEpochSelectorBank(selector);
        }

        /** @return Publication generation observed by captured admission. */
        [[nodiscard]] constexpr std::uint64_t generation() const noexcept
        {
            return deviceMoEOverlayEpochSelectorGeneration(selector);
        }
    };

    /**
     * @brief Optional node-local transaction barrier bound to one epoch acquire.
     *
     * The record address is a process-local CUDA/HIP alias of the same physical
     * mapped pages used by every continuation participant. `participant_id` is
     * the immutable global id whose disjoint arrival lane this device may write.
     * An empty binding preserves ordinary local or peer-selected admission.
     */
    struct DeviceMoEOverlayEpochAdmissionBarrierBinding
    {
        MoEOverlayDeviceControllerInferenceEpochRecord *record = nullptr;
        std::uint32_t participant_id = 0xffffffffu;
        std::uint32_t reserved = 0u;

        /** @return Whether stable address and participant identity are present. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            return record != nullptr && participant_id != 0xffffffffu;
        }

        /** @return Whether no transaction-wide synchronization was requested. */
        [[nodiscard]] constexpr bool empty() const noexcept
        {
            return record == nullptr && participant_id == 0xffffffffu;
        }
    };

    /**
     * @brief Device-resident completion and diagnostic for one epoch operation.
     *
     * Kernel launch success only proves that work entered the explicit stream.
     * This record reports the semantic result after that work executes.  It is
     * persistent graph state: inference and maintenance code inspect it through
     * an ordered device consumer or at an explicit asynchronous observation
     * boundary, never by synchronizing the launch stream in the hot path.
     */
    struct alignas(16) DeviceMoEOverlayEpochStatus
    {
        std::uint64_t epoch = 0u;    ///< Epoch named or observed by the operation.
        std::uint64_t selector = 0u; ///< Selector observed or published.
        std::uint32_t operation = 0u; ///< `DeviceMoEOverlayEpochOperation` value.
        std::uint32_t code = 0u;      ///< `DeviceMoEOverlayEpochStatusCode` value.
        std::uint32_t bank = kDeviceMoEOverlayInvalidBank; ///< Named bank or invalid sentinel.
        std::uint32_t observed_state = 0u; ///< Bank lifecycle seen on failure/success.

        /** @return Whether the completed operation succeeded semantically. */
        [[nodiscard]] constexpr bool succeeded() const noexcept
        {
            return code == static_cast<std::uint32_t>(
                               DeviceMoEOverlayEpochStatusCode::Success);
        }

        /** @return Typed operation written by the producer. */
        [[nodiscard]] constexpr DeviceMoEOverlayEpochOperation typedOperation() const noexcept
        {
            return static_cast<DeviceMoEOverlayEpochOperation>(operation);
        }

        /** @return Typed result written by the producer. */
        [[nodiscard]] constexpr DeviceMoEOverlayEpochStatusCode typedCode() const noexcept
        {
            return static_cast<DeviceMoEOverlayEpochStatusCode>(code);
        }
    };

    static_assert(std::is_trivially_copyable_v<DeviceMoEOverlayEpochControl>);
    static_assert(std::is_trivially_copyable_v<DeviceMoEOverlayEpochTicket>);
    static_assert(std::is_trivially_copyable_v<DeviceMoEOverlayEpochStatus>);
    static_assert(std::is_trivially_copyable_v<
                  DeviceMoEOverlayEpochAdmissionBarrierBinding>);
    static_assert(sizeof(DeviceMoEOverlayEpochControl) == 64u);
    static_assert(alignof(DeviceMoEOverlayEpochControl) == 64u);
    static_assert(sizeof(DeviceMoEOverlayEpochTicket) == 16u);
    static_assert(sizeof(DeviceMoEOverlayEpochStatus) == 32u);
    static_assert(alignof(DeviceMoEOverlayEpochStatus) == 16u);
} // namespace llaminar2
