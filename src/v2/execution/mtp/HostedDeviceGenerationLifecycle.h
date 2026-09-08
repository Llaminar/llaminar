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

#include "execution/local_execution/graph/DeviceExecutionTimeline.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <utility>

namespace llaminar2
{
    /**
     * @brief Semantic writer that publishes one generation-controller frontier.
     *
     * Admission creates the first controller version.  Every later publication
     * closes a transaction that previously borrowed that version.  Keeping the
     * distinction typed prevents a failed verifier from pretending that an
     * uncommitted controller is ready for another transaction.
     */
    enum class DeviceGenerationStatePublicationKind : std::uint8_t
    {
        Admission = 0, ///< Request admission initialized controller storage.
        CommittedTransaction, ///< One verifier transaction committed its state.
        Terminal, ///< The complete generation policy published terminal state.
    };

    /**
     * @brief Exclusive ownership phase of the reusable generation controller.
     */
    enum class DeviceGenerationStateHandoffPhase : std::uint8_t
    {
        Inactive = 0, ///< No admitted request owns the controller rows.
        Published, ///< A recorded event exposes one immutable controller version.
        Borrowed, ///< One exact stream owns the next controller transition.
    };

    /**
     * @brief Device-free state machine for one event-backed controller handoff.
     *
     * The class owns only lifecycle metadata.  Backend event submission remains
     * in the orchestrator and is supplied as a callback to each transition.  A
     * transition commits only after that callback succeeds, so host metadata can
     * never outrun the CUDA/HIP event edge it describes.
     *
     * A borrowed frontier is deliberately retained.  If graph construction or
     * verification fails before the normal committed-state publication, request
     * reset can record the same lifecycle event after the borrower stream and
     * join it without synchronizing either stream or inventing valid controller
     * contents.
     */
    class DeviceGenerationStateHandoff final
    {
    public:
        /** @return Current exclusive ownership phase. */
        [[nodiscard]] DeviceGenerationStateHandoffPhase phase() const noexcept
        {
            return phase_;
        }

        /** @return Whether no request owns this handoff. */
        [[nodiscard]] bool inactive() const noexcept
        {
            return phase_ == DeviceGenerationStateHandoffPhase::Inactive;
        }

        /** @return Whether a recorded controller version awaits one consumer. */
        [[nodiscard]] bool published() const noexcept
        {
            return phase_ == DeviceGenerationStateHandoffPhase::Published;
        }

        /** @return Whether one stream owns an in-progress controller transition. */
        [[nodiscard]] bool borrowed() const noexcept
        {
            return phase_ == DeviceGenerationStateHandoffPhase::Borrowed;
        }

        /** @return Request rows authenticated by the current handoff. */
        [[nodiscard]] int requestCount() const noexcept
        {
            return request_count_;
        }

        /** @return Exact producer or borrower stream at the current frontier. */
        [[nodiscard]] void *frontierStream() const noexcept
        {
            return frontier_stream_;
        }

        /** @return Semantic owner of a borrowed frontier, or `Count` otherwise. */
        [[nodiscard]] DeviceTimelineRole borrowerRole() const noexcept
        {
            return borrower_role_;
        }

        /**
         * @brief Publish admission, committed, or terminal controller state.
         *
         * Admission is legal only from `Inactive`; committed and terminal state
         * are legal only from `Borrowed`.  @p submit_event must record the real
         * backend event after every producer-stream write.  The state transition
         * occurs only after that operation succeeds.
         *
         * @tparam SubmitEvent Callable with no arguments returning `bool`.
         * @param kind Exact semantic publication being closed.
         * @param producer_stream Stream containing the completed writes.
         * @param request_count Number of controller rows in the transaction.
         * @param submit_event Backend publication operation.
         * @return Whether both the backend edge and typed transition succeeded.
         */
        template <typename SubmitEvent>
        [[nodiscard]] bool publish(
            DeviceGenerationStatePublicationKind kind,
            void *producer_stream,
            int request_count,
            SubmitEvent &&submit_event)
        {
            const bool admission =
                kind == DeviceGenerationStatePublicationKind::Admission;
            const bool phase_valid = admission ? inactive() : borrowed();
            const bool request_valid =
                request_count > 0 &&
                (admission || request_count == request_count_);
            if (!phase_valid || !request_valid || !producer_stream ||
                !std::invoke(std::forward<SubmitEvent>(submit_event)))
            {
                return false;
            }

            phase_ = DeviceGenerationStateHandoffPhase::Published;
            frontier_stream_ = producer_stream;
            request_count_ = request_count;
            borrower_role_ = DeviceTimelineRole::Count;
            last_publication_ = kind;
            return true;
        }

        /**
         * @brief Transfer one published controller version to an exact stream.
         *
         * @p submit_wait must enqueue the event dependency from the current
         * producer to @p consumer_stream.  The previous producer identity is
         * passed to the callback so same-stream elision remains a backend/event
         * concern rather than an implicit state-machine shortcut.
         *
         * @tparam SubmitWait Callable `(void *producer, void *consumer) -> bool`.
         * @param consumer_stream Exact stream that will read or mutate state.
         * @param consumer_role Semantic owner of that stream.
         * @param expected_request_count Required controller-row count.
         * @param submit_wait Backend event-wait operation.
         * @return Whether the wait and `Published -> Borrowed` transition succeeded.
         */
        template <typename SubmitWait>
        [[nodiscard]] bool borrow(
            void *consumer_stream,
            DeviceTimelineRole consumer_role,
            int expected_request_count,
            SubmitWait &&submit_wait)
        {
            if (!published() || !frontier_stream_ || !consumer_stream ||
                consumer_role == DeviceTimelineRole::Count ||
                expected_request_count <= 0 ||
                request_count_ != expected_request_count ||
                !std::invoke(
                    std::forward<SubmitWait>(submit_wait),
                    frontier_stream_,
                    consumer_stream))
            {
                return false;
            }

            phase_ = DeviceGenerationStateHandoffPhase::Borrowed;
            frontier_stream_ = consumer_stream;
            borrower_role_ = consumer_role;
            return true;
        }

        /**
         * @brief Join and retire either live frontier during ordered reset.
         *
         * A published frontier already has a completion event.  A borrowed
         * frontier needs that event re-recorded after the borrower stream before
         * reset waits on it.  @p submit_retirement receives the exact phase,
         * frontier stream, borrower role, and reset stream so the orchestrator
         * can perform the appropriate nonblocking event DAG.
         *
         * @tparam SubmitRetirement Callable accepting the four values above.
         * @param reset_stream Exact stream that will mutate request state.
         * @param expected_request_count Required controller-row count.
         * @param submit_retirement Backend event publication/wait operation.
         * @return Whether the frontier was joined and the lifecycle retired.
         */
        template <typename SubmitRetirement>
        [[nodiscard]] bool retireForReset(
            void *reset_stream,
            int expected_request_count,
            SubmitRetirement &&submit_retirement)
        {
            if (inactive() || !frontier_stream_ || !reset_stream ||
                expected_request_count <= 0 ||
                request_count_ != expected_request_count ||
                !std::invoke(
                    std::forward<SubmitRetirement>(submit_retirement),
                    phase_,
                    frontier_stream_,
                    borrower_role_,
                    reset_stream))
            {
                return false;
            }
            clear();
            return true;
        }

        /**
         * @brief Retire a host-result borrow after its exact event completed.
         *
         * This transition performs no backend operation: the terminal bridge
         * has already waited for the event that follows its D2H copies.
         */
        [[nodiscard]] bool retireAfterHostCompletion(
            int expected_request_count) noexcept
        {
            if (!borrowed() ||
                borrower_role_ != DeviceTimelineRole::HostResultBridge ||
                expected_request_count <= 0 ||
                request_count_ != expected_request_count)
            {
                return false;
            }
            clear();
            return true;
        }

        /** @return Last semantic publication, for diagnostics only. */
        [[nodiscard]] DeviceGenerationStatePublicationKind
        lastPublication() const noexcept
        {
            return last_publication_;
        }

    private:
        /** @brief Return to the sole quiescent state after a proven join. */
        void clear() noexcept
        {
            phase_ = DeviceGenerationStateHandoffPhase::Inactive;
            frontier_stream_ = nullptr;
            request_count_ = 0;
            borrower_role_ = DeviceTimelineRole::Count;
            last_publication_ =
                DeviceGenerationStatePublicationKind::Admission;
        }

        DeviceGenerationStateHandoffPhase phase_ =
            DeviceGenerationStateHandoffPhase::Inactive;
        void *frontier_stream_ = nullptr;
        int request_count_ = 0;
        DeviceTimelineRole borrower_role_ = DeviceTimelineRole::Count;
        DeviceGenerationStatePublicationKind last_publication_ =
            DeviceGenerationStatePublicationKind::Admission;
    };

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
