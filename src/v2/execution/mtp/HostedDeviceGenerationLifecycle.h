/**
 * @file HostedDeviceGenerationLifecycle.h
 * @brief Typed host-visible cursors for authenticated device-generation tickets.
 *
 * HIP and explicitly heterogeneous generation cannot place every MTP branch in
 * one native conditional graph.  The device controller therefore publishes a
 * fixed-size authenticated ticket and the host submits the retained branch it
 * names.  These cursors model only that submission protocol; model, sampler,
 * verifier, KV, and response state remain device-owned.
 *
 * Keeping scheduler and fragment progress in explicit state machines makes a
 * stale ticket, duplicated fragment, overlapping advance, or premature
 * terminal retirement unrepresentable without a rejected transition.  The
 * types contain no backend handles and are therefore exhaustively testable in
 * the device-free unit suite.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace llaminar2
{
    /** Lifecycle of one hosted device-generation request. */
    enum class HostedDeviceGenerationCursorState : std::uint8_t
    {
        Inactive = 0, ///< No hosted scheduler owns the current request.
        SchedulerReady, ///< Scheduler initialized; no ticket copy is in flight.
        AwaitingTicket, ///< One immutable ticket D2H observation is pending.
        TicketObserved, ///< One authenticated ticket awaits branch submission.
        Submitting, ///< The selected retained branch is being submitted.
        TerminalSubmitted, ///< The device terminal publication was submitted.
    };

    /**
     * @brief Transaction cursor for one authenticated hosted MTP request.
     *
     * Exactly one transition owns each ticket observation and retained branch.
     * Transaction numbers must be contiguous, start at one, and match the
     * ticket currently being submitted.  `reset()` is reserved for model/request
     * teardown after the caller has joined device work; ordinary successful
     * completion must cross `retireCompleted()`.
     */
    class HostedDeviceGenerationCursor
    {
    public:
        /** @return Current explicit protocol state. */
        [[nodiscard]] HostedDeviceGenerationCursorState state() const noexcept
        {
            return state_;
        }

        /** @return Whether no hosted scheduler owns the request. */
        [[nodiscard]] bool inactive() const noexcept
        {
            return state_ == HostedDeviceGenerationCursorState::Inactive;
        }

        /**
         * @return Whether ticket observation may enter or resume.
         *
         * `Inactive` admits the one-time scheduler setup. `AwaitingTicket`
         * resumes after the already-submitted D2H event becomes visible.
         */
        [[nodiscard]] bool mayObserveTicket() const noexcept
        {
            return inactive() ||
                   state_ ==
                       HostedDeviceGenerationCursorState::AwaitingTicket;
        }

        /** @return Whether the one-time hosted scheduler setup has run. */
        [[nodiscard]] bool schedulerStarted() const noexcept
        {
            return !inactive();
        }

        /** @return Whether a new ticket observation may be submitted. */
        [[nodiscard]] bool maySubmitTicketObservation() const noexcept
        {
            return state_ ==
                   HostedDeviceGenerationCursorState::SchedulerReady;
        }

        /** @return Whether one ticket observation is currently pending. */
        [[nodiscard]] bool ticketObservationPending() const noexcept
        {
            return state_ ==
                   HostedDeviceGenerationCursorState::AwaitingTicket;
        }

        /** @return Whether the authenticated ticket may begin its branch. */
        [[nodiscard]] bool mayBeginAdvance(
            std::int32_t transaction_count) const noexcept
        {
            return state_ ==
                       HostedDeviceGenerationCursorState::TicketObserved &&
                   transaction_count == last_transaction_count_;
        }

        /** @return Whether a branch currently owns fragment submission. */
        [[nodiscard]] bool submitting() const noexcept
        {
            return state_ ==
                   HostedDeviceGenerationCursorState::Submitting;
        }

        /** @return Whether the terminal device publication was submitted. */
        [[nodiscard]] bool terminalSubmitted() const noexcept
        {
            return state_ ==
                   HostedDeviceGenerationCursorState::TerminalSubmitted;
        }

        /** @return Last authenticated contiguous transaction number. */
        [[nodiscard]] std::int32_t lastTransactionCount() const noexcept
        {
            return last_transaction_count_;
        }

        /**
         * @brief Enter the scheduler-owned lifecycle exactly once.
         * @return `true` only for `Inactive -> SchedulerReady`.
         */
        [[nodiscard]] bool startScheduler() noexcept
        {
            if (!inactive())
                return false;
            state_ = HostedDeviceGenerationCursorState::SchedulerReady;
            return true;
        }

        /**
         * @brief Publish ownership of one asynchronous ticket observation.
         * @return `true` only for `SchedulerReady -> AwaitingTicket`.
         */
        [[nodiscard]] bool markTicketObservationSubmitted() noexcept
        {
            if (!maySubmitTicketObservation())
                return false;
            state_ = HostedDeviceGenerationCursorState::AwaitingTicket;
            return true;
        }

        /**
         * @brief Authenticate the next contiguous device ticket.
         * @param transaction_count Ticket transaction ordinal.
         * @return `true` only for the exact next positive ordinal.
         */
        [[nodiscard]] bool acceptTicket(
            std::int32_t transaction_count) noexcept
        {
            if (!ticketObservationPending() ||
                last_transaction_count_ ==
                    std::numeric_limits<std::int32_t>::max() ||
                transaction_count != last_transaction_count_ + 1)
            {
                return false;
            }
            last_transaction_count_ = transaction_count;
            state_ = HostedDeviceGenerationCursorState::TicketObserved;
            return true;
        }

        /**
         * @brief Transfer the observed ticket into branch submission.
         * @param transaction_count Exact authenticated ticket ordinal.
         * @return `true` only for `TicketObserved -> Submitting`.
         */
        [[nodiscard]] bool beginAdvance(
            std::int32_t transaction_count) noexcept
        {
            if (!mayBeginAdvance(transaction_count))
                return false;
            state_ = HostedDeviceGenerationCursorState::Submitting;
            return true;
        }

        /**
         * @brief Complete one fully submitted retained branch.
         * @param transaction_count Exact authenticated ticket ordinal.
         * @param terminal Whether this branch published terminal output.
         * @return `true` only for a matching in-flight branch.
         */
        [[nodiscard]] bool finishAdvance(
            std::int32_t transaction_count,
            bool terminal) noexcept
        {
            if (!submitting() ||
                transaction_count != last_transaction_count_)
            {
                return false;
            }
            state_ = terminal
                         ? HostedDeviceGenerationCursorState::
                               TerminalSubmitted
                         : HostedDeviceGenerationCursorState::SchedulerReady;
            return true;
        }

        /**
         * @brief Retire a successfully completed hosted request.
         *
         * Native conditional execution never starts this hosted cursor, so an
         * inactive cursor is also a valid terminal bridge input.
         *
         * @return Whether retirement was legal; success resets the cursor.
         */
        [[nodiscard]] bool retireCompleted() noexcept
        {
            if (!inactive() && !terminalSubmitted())
                return false;
            reset();
            return true;
        }

        /**
         * @brief Clear the cursor after an externally ordered teardown/reset.
         *
         * Callers must first join all device work. This method deliberately
         * performs no synchronization and carries no recovery semantics.
         */
        void reset() noexcept
        {
            state_ = HostedDeviceGenerationCursorState::Inactive;
            last_transaction_count_ = 0;
        }

    private:
        HostedDeviceGenerationCursorState state_ =
            HostedDeviceGenerationCursorState::Inactive;
        std::int32_t last_transaction_count_ = 0;
    };

    /** Lifecycle of the fragment cursor for one authenticated branch. */
    enum class HostedDeviceGenerationAdvanceState : std::uint8_t
    {
        Idle = 0,
        Submitting,
    };

    /**
     * @brief Strict ordinal cursor for retained hosted branch fragments.
     *
     * Selection computes the exact number of fragments once. Thereafter every
     * successful submission must present the next ordinal, and finish is legal
     * only after the complete selected count has been recorded.
     */
    class HostedDeviceGenerationAdvanceCursor
    {
    public:
        /** @return Current branch-submission state. */
        [[nodiscard]] HostedDeviceGenerationAdvanceState state() const noexcept
        {
            return state_;
        }

        /** @return Whether no branch currently owns this cursor. */
        [[nodiscard]] bool idle() const noexcept
        {
            return state_ == HostedDeviceGenerationAdvanceState::Idle;
        }

        /** @return Immutable selected-fragment count for this branch. */
        [[nodiscard]] std::size_t fragmentCount() const noexcept
        {
            return fragment_count_;
        }

        /** @return Next exact fragment ordinal accepted by the cursor. */
        [[nodiscard]] std::size_t nextFragment() const noexcept
        {
            return next_fragment_;
        }

        /**
         * @brief Admit one non-overlapping retained branch.
         * @param fragment_count Number of fragments selected by the ticket.
         * @return `true` only for `Idle -> Submitting`.
         */
        [[nodiscard]] bool begin(std::size_t fragment_count) noexcept
        {
            if (!idle())
                return false;
            state_ = HostedDeviceGenerationAdvanceState::Submitting;
            fragment_count_ = fragment_count;
            next_fragment_ = 0;
            return true;
        }

        /** @return Whether @p fragment_index is the next legal ordinal. */
        [[nodiscard]] bool maySubmit(
            std::size_t fragment_index) const noexcept
        {
            return state_ == HostedDeviceGenerationAdvanceState::Submitting &&
                   fragment_index == next_fragment_ &&
                   fragment_index < fragment_count_;
        }

        /**
         * @brief Record one successfully submitted exact fragment.
         * @param fragment_index Exact ordinal just submitted.
         * @return Whether the transition was accepted.
         */
        [[nodiscard]] bool recordSubmission(
            std::size_t fragment_index) noexcept
        {
            if (!maySubmit(fragment_index))
                return false;
            ++next_fragment_;
            return true;
        }

        /** @return Whether every selected fragment has been submitted. */
        [[nodiscard]] bool mayFinish() const noexcept
        {
            return state_ == HostedDeviceGenerationAdvanceState::Submitting &&
                   next_fragment_ == fragment_count_;
        }

        /**
         * @brief Retire one completely submitted branch.
         * @return `true` only when all selected fragments were recorded.
         */
        [[nodiscard]] bool finish() noexcept
        {
            if (!mayFinish())
                return false;
            reset();
            return true;
        }

        /** @brief Clear the branch cursor during graph/request teardown. */
        void reset() noexcept
        {
            state_ = HostedDeviceGenerationAdvanceState::Idle;
            fragment_count_ = 0;
            next_fragment_ = 0;
        }

    private:
        HostedDeviceGenerationAdvanceState state_ =
            HostedDeviceGenerationAdvanceState::Idle;
        std::size_t fragment_count_ = 0;
        std::size_t next_fragment_ = 0;
    };

} // namespace llaminar2
