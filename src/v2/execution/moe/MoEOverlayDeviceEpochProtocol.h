/**
 * @file MoEOverlayDeviceEpochProtocol.h
 * @brief CPU implementation and reference oracle for device epoch RCU.
 *
 * The class operates directly on one caller-owned
 * `DeviceMoEOverlayEpochControl`.  It is the first-class CPU backend protocol
 * and the executable specification for CUDA/HIP acquire, release, publication,
 * and retirement kernels.  It must never be used as a host shadow of a
 * GPU-owned control block; GPU production code operates on the device record
 * through stream-ordered kernels and polls only explicit completion events.
 */

#pragma once

#include "DeviceMoEOverlayEpochABI.h"

#include <cstdint>
#include <optional>

namespace llaminar2
{
    /** Identity returned after atomically publishing one ready candidate. */
    struct MoEOverlayDeviceEpochPublication
    {
        std::uint64_t previous_epoch = 0u;
        std::uint64_t published_epoch = 0u;
        std::uint32_t previous_bank = 0u;
        std::uint32_t published_bank = 0u;
        std::uint64_t generation = 0u;

        /** @return Whether the publication names two distinct valid epochs. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            return previous_epoch != 0u && published_epoch > previous_epoch &&
                   previous_bank < kDeviceMoEOverlayEpochBankCount &&
                   published_bank < kDeviceMoEOverlayEpochBankCount &&
                   previous_bank != published_bank && generation != 0u;
        }
    };

    /**
     * @brief Lock-free two-bank placement protocol used by CPU inference.
     *
     * Inference calls only @ref tryAcquirePublished and @ref release.  Those
     * methods execute a bounded number of atomic operations and never wait for
     * maintenance.  Candidate preparation, readiness, publication, abort, and
     * retirement are maintenance operations and must be serialized by their
     * owning residency service.
     */
    class MoEOverlayDeviceEpochProtocol final
    {
    public:
        /**
         * @brief Initialize a pristine control block with one published epoch.
         * @param control Caller-owned zero-initialized CPU control block.
         * @param initial_epoch Positive initial residency epoch.
         * @param initial_bank Physical descriptor bank already populated for
         *        @p initial_epoch. This lets a runtime table publish its first
         *        prepared bank before the RCU selector is exposed.
         * @throws std::invalid_argument for epoch zero.
         * @throws std::logic_error when the block was already initialized.
         */
        static void initialize(
            DeviceMoEOverlayEpochControl &control,
            std::uint64_t initial_epoch,
            std::uint32_t initial_bank = 0u);

        /** @brief Bind operations to the authoritative CPU control block. */
        explicit MoEOverlayDeviceEpochProtocol(
            DeviceMoEOverlayEpochControl &control) noexcept;

        /**
         * @brief Acquire the epoch selected at one atomic publication point.
         * @return A reader-holding ticket, or empty only for corrupt/uninitialized
         *         control state.
         *
         * The global acquisition guard is installed before the selector load.
         * Retirement therefore cannot recycle the selected bank between that
         * load and the per-bank reader increment.
         */
        [[nodiscard]] std::optional<DeviceMoEOverlayEpochTicket>
        tryAcquirePublished() noexcept;

        /**
         * @brief Acquire one exact globally admitted epoch from either live bank.
         * @param admission_epoch Positive epoch published by the topology-wide
         *        controller, or zero to select the participant-local publication.
         * @return A reader-holding ticket, or empty when the admitted epoch is
         *         absent/corrupt rather than Published or Retiring.
         *
         * Multi-participant publication flips local selectors independently.
         * During that bounded fan-out some participants have the new bank
         * Published while others still expose the old one. The topology-wide
         * admission word remains old until every flip completes, so this
         * method deliberately permits acquisition of that old Retiring bank.
         * The acquisition guard spans exact-epoch lookup and reader install,
         * preventing retirement from reclaiming it in between.
         */
        [[nodiscard]] std::optional<DeviceMoEOverlayEpochTicket>
        tryAcquireAdmitted(std::uint64_t admission_epoch) noexcept;

        /**
         * @brief Acquire an authenticated peer-selected prepared epoch exactly.
         * @param exact_epoch Positive epoch carried by a validated activation
         *        descriptor after global inactive-bank preparation completed.
         * @return A reader-holding ticket, or empty when the epoch is absent,
         *         corrupt, or not Ready, Published, or Retiring.
         *
         * Heterogeneous selector publication is a parallel fan-out. A peer may
         * therefore send E+1 after its selector flips while this participant's
         * E+1 bank is still Ready. Global preparation consensus guarantees that
         * the bank and runtime selectors are complete before any peer can name
         * it. This exact admission pins that prepared bank without changing the
         * participant-local publication floor; ordinary admission must continue
         * to use @ref tryAcquireAdmitted and may never consume Ready state.
         */
        [[nodiscard]] std::optional<DeviceMoEOverlayEpochTicket>
        tryAcquirePreparedExact(std::uint64_t exact_epoch) noexcept;

        /**
         * @brief Release one reader installed by @ref tryAcquirePublished.
         * @param ticket Exact immutable ticket being released.
         * @return False for stale, malformed, or already-released identity.
         */
        [[nodiscard]] bool release(
            const DeviceMoEOverlayEpochTicket &ticket) noexcept;

        /**
         * @brief Reserve the non-published empty bank for a newer epoch.
         * @param candidate_epoch Strictly newer candidate epoch.
         * @return Reserved bank, or empty while the other bank is still retained.
         * @throws std::invalid_argument for a non-monotonic epoch.
         * @throws std::logic_error for an uninitialized/corrupt publication.
         */
        [[nodiscard]] std::optional<std::uint32_t> reserveCandidate(
            std::uint64_t candidate_epoch);

        /**
         * @brief Publish completion of all candidate descriptor/weight events.
         * @param candidate_epoch Exact reserved epoch.
         * @throws std::logic_error when the named bank is absent or not staging.
         */
        void markCandidateReady(std::uint64_t candidate_epoch);

        /**
         * @brief Switch new ticket admission to one ready candidate.
         * @param candidate_epoch Exact ready epoch.
         * @return Previous/new bank and generation identity.
         * @throws std::logic_error for an invalid lifecycle or generation overflow.
         *
         * Existing readers remain on the previous bank.  The method performs no
         * wait and does not require their count to be zero.
         */
        [[nodiscard]] MoEOverlayDeviceEpochPublication publishReadyCandidate(
            std::uint64_t candidate_epoch);

        /**
         * @brief Abort one unpublished candidate and return its bank to Empty.
         * @param candidate_epoch Exact Candidate or Ready epoch.
         * @return False when the epoch does not name an abortable bank.
         */
        [[nodiscard]] bool abortCandidate(
            std::uint64_t candidate_epoch) noexcept;

        /**
         * @brief Reclaim a retiring bank after its complete device grace period.
         * @param retiring_epoch Exact old epoch.
         * @return True when reclaimed; false while an acquisition or reader lives.
         * @throws std::logic_error when the epoch is absent or not Retiring.
         */
        [[nodiscard]] bool tryRetire(std::uint64_t retiring_epoch);

        /** @return Current published epoch without acquiring an inference reader. */
        [[nodiscard]] std::uint64_t publishedEpoch() const noexcept;

        /** @return Current published bank without acquiring an inference reader. */
        [[nodiscard]] std::uint32_t publishedBank() const noexcept;

        /** @return Current monotonic publication generation. */
        [[nodiscard]] std::uint64_t publicationGeneration() const noexcept;

        /** @return Atomically observed lifecycle for @p bank. */
        [[nodiscard]] DeviceMoEOverlayEpochBankState bankState(
            std::uint32_t bank) const noexcept;

        /** @return Atomically observed epoch in @p bank, or zero when invalid. */
        [[nodiscard]] std::uint64_t bankEpoch(
            std::uint32_t bank) const noexcept;

        /** @return Exact installed inference reader count for @p bank. */
        [[nodiscard]] std::uint64_t bankReaderCount(
            std::uint32_t bank) const noexcept;

        /** @return Acquisitions between selector read and bank-reader install. */
        [[nodiscard]] std::uint64_t acquisitionsInFlight() const noexcept;

    private:
        /** Typed admission semantics for one shared acquire implementation. */
        enum class AdmissionPolicy : std::uint8_t
        {
            PublishedFloor,
            ExactPreparedPeer,
        };

        /**
         * @brief Install one reader under the selected admission semantics.
         * @param epoch Zero for the local publication floor, otherwise exact.
         * @param policy Whether a globally prepared Ready bank is admissible.
         * @return One reader-holding ticket or empty for invalid state.
         */
        [[nodiscard]] std::optional<DeviceMoEOverlayEpochTicket> tryAcquire(
            std::uint64_t epoch,
            AdmissionPolicy policy) noexcept;

        /** @return Bank holding @p epoch, irrespective of its lifecycle. */
        [[nodiscard]] std::optional<std::uint32_t> bankForEpoch(
            std::uint64_t epoch) const noexcept;

        DeviceMoEOverlayEpochControl *control_ = nullptr;
    };
} // namespace llaminar2
