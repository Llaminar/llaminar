/**
 * @file MoEOverlayEpochLeaseLifecycle.h
 * @brief Typed host-submission lifecycle for one device-owned overlay epoch reader.
 *
 * ExpertOverlay residency tickets are device-owned.  The host nevertheless has
 * to enqueue the retained acquire/release graphs and publish the exact events
 * which connect their streams.  Inference and the background controller may
 * submit those recipes from different host workers, so the enqueue sequence is
 * one small critical section even though all resulting GPU work remains fully
 * asynchronous.
 *
 * This class publishes semantic ownership states and the exact producer stream
 * required by a terminal release receipt.  A caller holds a @ref Submission
 * while it validates pointers, enqueues kernels, and records the terminal
 * event; the new state and receipt become visible together only after that
 * complete recipe succeeds.  Other callers therefore cannot observe temporary
 * states such as "acquire pending" or "release submitting", nor can they see a
 * `ReleasePublished` state whose producer stream has already been retired by a
 * concurrent acquire.
 */

#pragma once

#include "execution/mtp/MTPSidecarCaptureLayout.h"

#include <atomic>
#include <cstdint>
#include <mutex>

namespace llaminar2
{
    /** @brief Sole placement owner for one semantic MTP sidecar role. */
    enum class MoEOverlaySidecarEpochOwnership : std::uint8_t
    {
        NoExpertAccess, ///< KV append uses fixed projections, never routed experts.
        ExternalReader, ///< A standalone full sidecar acquires its own reader.
        GraphSequence, ///< The coordinator's admitted sequence owns the reader.
    };

    /**
     * @brief Select residency ownership independently of maintenance activity.
     * @param role Semantic role of the retained sidecar graph.
     * @param has_coordinator Whether an overlay graph-sequence authority is bound.
     * @return The only authority allowed to acquire placement for this sidecar.
     *
     * KV-only work must not pin an ambient reader: placement may advance between
     * shifted-KV completion and admission of the next main transaction. That
     * transaction acquires its own admitted epoch after joining the KV event.
     */
    [[nodiscard]] constexpr MoEOverlaySidecarEpochOwnership
    moeOverlaySidecarEpochOwnership(
        MTPSidecarCaptureRole role, bool has_coordinator)
    {
        switch (role)
        {
        case MTPSidecarCaptureRole::KVOnly:
            return MoEOverlaySidecarEpochOwnership::NoExpertAccess;
        case MTPSidecarCaptureRole::Full:
        case MTPSidecarCaptureRole::Chained:
            return has_coordinator
                       ? MoEOverlaySidecarEpochOwnership::GraphSequence
                       : MoEOverlaySidecarEpochOwnership::ExternalReader;
        }
        throw std::logic_error("Unknown MTP sidecar residency role");
    }

    /**
     * @brief Complete semantic owner of one device-resident overlay reader.
     *
     * These values describe who may use the ticket, not which host API call is
     * halfway through execution.  That distinction is what keeps background
     * maintenance from exposing partial acquire/release publication to the next
     * inference graph.
     */
    enum class MoEOverlayEpochLeaseState : std::uint8_t
    {
        Idle = 0,            ///< No reader or terminal receipt is live.
        ExternalReader,      ///< A non-sequence external transaction owns it.
        ExternalSequence,    ///< One complete direct MTP sequence owns it.
        HostedParent,        ///< A hosted MTP parent owns it.
        CapturedMainForward, ///< A complete captured main graph is in flight.
        ExternalForward,     ///< One external auxiliary forward owns it.
        HostedChildForward,  ///< A child temporarily borrows its hosted parent.
        ReleasePublished,    ///< The immutable release event is consumable.
    };

    /**
     * @brief Legal action for a background observer at an inference boundary.
     *
     * Background telemetry and maintenance are consumers of completed
     * inference publication; they are never owners of an inference reader.
     * Keeping this decision beside the lease states makes it impossible for a
     * caller to reinterpret a live reader as abandoned work and clear its
     * device ticket before the admitted forward consumes it.
     */
    enum class MoEOverlayEpochObservationAction : std::uint8_t
    {
        SubmitAtIdle = 0,       ///< No inference lease must be joined.
        WaitForPublishedRelease, ///< Consume the immutable terminal event.
        DeferToInferenceOwner,   ///< Retry after the live owner publishes release.
    };

    /**
     * @brief Exact action for an authenticated placement-maintenance writer.
     *
     * A due device-controller ticket is stronger than passive observation: its
     * maintenance graph will write a successor placement bank.  The writer may
     * close an admitted ambient reader, or it may join a release already
     * published by a self-contained forward.  Those cases are mutually
     * exclusive and must be selected while holding the lifecycle submission
     * authority; treating both as "release now" double-closes the latter.
     */
    enum class MoEOverlayEpochPlacementMaintenanceAction : std::uint8_t
    {
        CloseExternalReader = 0, ///< Publish the sole release for a live reader.
        JoinPublishedRelease,    ///< Wait on the existing immutable release.
        RejectMissingBoundary,   ///< No committed inference boundary exists.
        RejectLiveInferenceOwner, ///< A non-reader inference owner is still live.
    };

    /**
     * @brief Owner of the inference-to-maintenance edge for one graph launch.
     *
     * Native conditional maintenance joins the committed inference timeline in
     * its graph launch dependency.  A backend without conditional graphs first
     * authenticates a fixed-size dispatch ticket, then claims that same edge
     * before submitting the retained maintenance graph.  A scoped enum keeps
     * this ownership out of an ambiguous `boundary_already_published` boolean.
     */
    enum class MoEOverlayEpochMaintenanceBoundarySource : std::uint8_t
    {
        GraphLaunchDependency = 0, ///< The launch hook constructs the edge.
        AuthenticatedDispatchTicket, ///< Ticket submission already claimed it.
    };

    /**
     * @brief Return a stable diagnostic name for a maintenance edge owner.
     * @param source Typed owner selected by backend scheduling policy.
     * @return Static lowercase identifier suitable for PerfStats tags.
     */
    [[nodiscard]] constexpr const char *
    moeOverlayEpochMaintenanceBoundarySourceName(
        MoEOverlayEpochMaintenanceBoundarySource source) noexcept
    {
        switch (source)
        {
        case MoEOverlayEpochMaintenanceBoundarySource::GraphLaunchDependency:
            return "graph_launch_dependency";
        case MoEOverlayEpochMaintenanceBoundarySource::
            AuthenticatedDispatchTicket:
            return "authenticated_dispatch_ticket";
        }
        return "invalid";
    }

    /**
     * @brief Complete residency-epoch envelope owned by one forward submission.
     *
     * This policy is selected before the live-state prelude and is deliberately
     * independent of whether maintenance happened to run.  Maintenance only
     * publishes an event which the forward consumes; it never acquires a reader
     * for a future graph.  Consequently the coordinator-selected placement floor
     * and the device ticket are bound by the same exact graph submission.
     */
    enum class MoEOverlayForwardEpochSubmissionPolicy : std::uint8_t
    {
        Unbound = 0, ///< The participant has no device residency-epoch arena.
        RetainedPerForwardTransaction, ///< Retained Acquire, graph, retained Release.
        CapturedMainTransaction, ///< Acquire and Release are roots/terminal in main graph.
    };

    /**
     * @brief Select the sole residency envelope for one GPU forward graph.
     * @param has_epoch_binding Whether this participant owns a device epoch slot.
     * @param main_inference Whether this graph is the ordinary main forward.
     * @return One total, typed submission policy.
     *
     * Every ordinary GPU main graph is already one complete captured production
     * transaction, independent of whether placement authority is host- or
     * device-owned. It therefore embeds both epoch boundaries in that same
     * graph. Auxiliary and MTP child graphs deliberately retain the external
     * envelope owned by their enclosing sequence or hosted parent.
     */
    [[nodiscard]] constexpr MoEOverlayForwardEpochSubmissionPolicy
    moeOverlayForwardEpochSubmissionPolicy(
        bool has_epoch_binding,
        bool main_inference) noexcept
    {
        using Policy = MoEOverlayForwardEpochSubmissionPolicy;
        if (!has_epoch_binding)
            return Policy::Unbound;
        if (main_inference)
            return Policy::CapturedMainTransaction;
        return Policy::RetainedPerForwardTransaction;
    }

    /**
     * @brief Classify one complete lease state for background observation.
     * @param state Complete semantic lease state published by inference.
     * @return The only legal non-blocking action for a background boundary.
     *
     * This function is intentionally total over @ref MoEOverlayEpochLeaseState.
     * In particular, @ref MoEOverlayEpochLeaseState::ExternalReader is live
     * request admission, not an incomplete release recipe.  Only inference may
     * promote or release it.
     */
    [[nodiscard]] constexpr MoEOverlayEpochObservationAction
    moeOverlayEpochObservationAction(
        MoEOverlayEpochLeaseState state) noexcept
    {
        using Action = MoEOverlayEpochObservationAction;
        using State = MoEOverlayEpochLeaseState;
        switch (state)
        {
        case State::Idle:
            return Action::SubmitAtIdle;
        case State::ReleasePublished:
            return Action::WaitForPublishedRelease;
        case State::ExternalReader:
        case State::ExternalSequence:
        case State::HostedParent:
        case State::CapturedMainForward:
        case State::ExternalForward:
        case State::HostedChildForward:
            return Action::DeferToInferenceOwner;
        }
        return Action::DeferToInferenceOwner;
    }

    /**
     * @brief Classify one lease for a placement-writing maintenance claim.
     * @param state Complete semantic lease state observed under submission lock.
     * @return The only legal action for the authenticated writer.
     *
     * Unlike passive observation, an `ExternalReader` may be closed here: the
     * authenticated due-ticket path has already joined all inference producers
     * and is about to write placement.  `ReleasePublished` is not idempotently
     * released; its existing event is consumed.  Every other live owner is an
     * ordering violation, while `Idle` proves that no committed boundary exists.
     */
    [[nodiscard]] constexpr MoEOverlayEpochPlacementMaintenanceAction
    moeOverlayEpochPlacementMaintenanceAction(
        MoEOverlayEpochLeaseState state) noexcept
    {
        using Action = MoEOverlayEpochPlacementMaintenanceAction;
        using State = MoEOverlayEpochLeaseState;
        switch (state)
        {
        case State::ExternalReader:
            return Action::CloseExternalReader;
        case State::ReleasePublished:
            return Action::JoinPublishedRelease;
        case State::Idle:
            return Action::RejectMissingBoundary;
        case State::ExternalSequence:
        case State::HostedParent:
        case State::CapturedMainForward:
        case State::ExternalForward:
        case State::HostedChildForward:
            return Action::RejectLiveInferenceOwner;
        }
        return Action::RejectLiveInferenceOwner;
    }

    /**
     * @brief Serializes only the host enqueue/publication recipe for an epoch.
     *
     * The mutex never covers device completion, transport progress, histogram
     * analysis, or migration.  A normal critical section consists only of
     * stream-event waits, one retained graph launch, and one event record.  The
     * state itself remains atomic so diagnostics and maintenance preflight can
     * take a non-owning snapshot without delaying inference.
     */
    class MoEOverlayEpochLeaseLifecycle final
    {
    public:
        /**
         * @brief Exclusive owner of one indivisible host submission recipe.
         *
         * The state is not changed on construction.  Call @ref commit only after
         * every stream operation and terminal event record succeeds.  Destruction
         * without a commit is an automatic rollback because no partial state was
         * ever published.
         */
        class Submission final
        {
        public:
            /** Submission authority is unique and cannot be duplicated. */
            Submission(const Submission &) = delete;
            /** Submission authority is unique and cannot be reassigned. */
            Submission &operator=(const Submission &) = delete;
            /**
             * Moving is forbidden because a moved-from raw owner pointer could
             * otherwise outlive the transferred mutex lock and publish state
             * without host submission authority.
             */
            Submission(Submission &&) = delete;
            /** @copydoc Submission::Submission(Submission&&) */
            Submission &operator=(Submission &&) = delete;
            ~Submission() = default;

            /** @return Stable state observed after acquiring submission authority. */
            [[nodiscard]] MoEOverlayEpochLeaseState state() const noexcept
            {
                return initial_state_;
            }

            /**
             * @brief Return the exact producer stream of a published release.
             * @return Non-null only while this submission owns a complete
             *         @ref MoEOverlayEpochLeaseState::ReleasePublished receipt.
             *
             * The stream is deliberately accessible only through a live
             * submission.  This couples the semantic state and its ordering
             * payload under one lock, preventing a maintenance observer from
             * racing the next inference acquire as that acquire consumes and
             * retires the receipt.
             */
            [[nodiscard]] void *publishedReleaseProducerStream() const noexcept
            {
                const bool observed_release =
                    initial_state_ ==
                    MoEOverlayEpochLeaseState::ReleasePublished;
                const bool published_release =
                    owner_ && committed_ &&
                    owner_->state_.load(std::memory_order_relaxed) ==
                        MoEOverlayEpochLeaseState::ReleasePublished;
                return owner_ && (observed_release || published_release)
                           ? owner_->release_producer_stream_
                           : nullptr;
            }

            /**
             * @brief Publish one legal semantic transition after enqueue succeeds.
             * @param desired Fully constructed semantic state to expose.
             * @return true exactly once and only for a legal transition.
             *
             * A false return leaves the original state untouched.  This makes
             * backend launch failure recoverable without a second rollback store
             * and prevents a future caller from seeing contradictory metadata.
             */
            [[nodiscard]] bool commit(
                MoEOverlayEpochLeaseState desired) noexcept
            {
                /* A release state without its stream receipt is incomplete. */
                if (desired == MoEOverlayEpochLeaseState::ReleasePublished)
                    return false;
                return commitImpl(desired, nullptr);
            }

            /**
             * @brief Atomically publish a terminal release and its producer.
             * @param producer_stream Exact non-null stream on which the
             *        persistent released-event was recorded.
             * @return true exactly once and only for a legal transition into
             *         @ref MoEOverlayEpochLeaseState::ReleasePublished.
             *
             * Call this only after the release graph and event record have both
             * been enqueued.  The receipt remains immutable and multi-consumer
             * until a later submission transitions to another semantic owner.
             */
            [[nodiscard]] bool publishRelease(
                void *producer_stream) noexcept
            {
                if (!producer_stream)
                    return false;
                return commitImpl(
                    MoEOverlayEpochLeaseState::ReleasePublished,
                    producer_stream);
            }

        private:
            friend class MoEOverlayEpochLeaseLifecycle;

            /** Acquire the one host-side enqueue authority and snapshot state. */
            explicit Submission(MoEOverlayEpochLeaseLifecycle &owner) noexcept
                : owner_(&owner), lock_(owner.submission_mutex_),
                  initial_state_(
                      owner.state_.load(std::memory_order_acquire))
            {
            }

            /** @return Whether @p from -> @p to is a complete legal recipe. */
            static bool legalTransition(
                MoEOverlayEpochLeaseState from,
                MoEOverlayEpochLeaseState to) noexcept
            {
                using State = MoEOverlayEpochLeaseState;
                switch (from)
                {
                case State::Idle:
                case State::ReleasePublished:
                    return to == State::ExternalReader ||
                           to == State::ExternalSequence ||
                           to == State::HostedParent ||
                           to == State::CapturedMainForward ||
                           to == State::ExternalForward ||
                           (from == State::ReleasePublished &&
                            to == State::Idle);
                case State::ExternalReader:
                    return to == State::ExternalForward ||
                           to == State::ReleasePublished;
                case State::ExternalSequence:
                    return to == State::ExternalForward ||
                           to == State::ReleasePublished;
                case State::HostedParent:
                    return to == State::HostedChildForward ||
                           to == State::ReleasePublished;
                case State::CapturedMainForward:
                case State::ExternalForward:
                    return to == State::ReleasePublished;
                case State::HostedChildForward:
                    return to == State::HostedParent;
                }
                return false;
            }

            /**
             * @brief Publish one complete state/payload pair under the lock.
             * @param desired Semantic state visible after this recipe.
             * @param release_producer_stream Non-null exactly for a published
             *        release receipt.
             * @return true when the legal pair was atomically installed.
             */
            [[nodiscard]] bool commitImpl(
                MoEOverlayEpochLeaseState desired,
                void *release_producer_stream) noexcept
            {
                const bool publishes_release =
                    desired == MoEOverlayEpochLeaseState::ReleasePublished;
                if (!owner_ || committed_ ||
                    publishes_release !=
                        (release_producer_stream != nullptr) ||
                    !legalTransition(initial_state_, desired) ||
                    owner_->state_.load(std::memory_order_relaxed) !=
                        initial_state_)
                {
                    return false;
                }

                /*
                 * Store the payload before the release-state publication and
                 * clear it before publishing a non-release successor.  No
                 * caller may read this field without holding the same lock.
                 */
                owner_->release_producer_stream_ =
                    release_producer_stream;
                owner_->state_.store(desired, std::memory_order_release);
                committed_ = true;
                return true;
            }

            MoEOverlayEpochLeaseLifecycle *owner_ = nullptr; ///< State owner.
            std::unique_lock<std::mutex> lock_; ///< Host enqueue critical section.
            MoEOverlayEpochLeaseState initial_state_ =
                MoEOverlayEpochLeaseState::Idle; ///< Immutable starting state.
            bool committed_ = false; ///< Guards double publication.
        };

        MoEOverlayEpochLeaseLifecycle() = default;
        MoEOverlayEpochLeaseLifecycle(
            const MoEOverlayEpochLeaseLifecycle &) = delete;
        MoEOverlayEpochLeaseLifecycle &operator=(
            const MoEOverlayEpochLeaseLifecycle &) = delete;

        /** @return A non-owning diagnostic snapshot of the published state. */
        [[nodiscard]] MoEOverlayEpochLeaseState load() const noexcept
        {
            return state_.load(std::memory_order_acquire);
        }

        /**
         * @brief Begin one indivisible acquire/forward/release submission recipe.
         * @return RAII authority holding only the host metadata/enqueue lock.
         */
        [[nodiscard]] Submission beginSubmission() noexcept
        {
            return Submission(*this);
        }

    private:
        std::mutex submission_mutex_; ///< Protects partial host enqueue recipes.
        /** Exact stream paired with `ReleasePublished`; lock-protected. */
        void *release_producer_stream_ = nullptr;
        std::atomic<MoEOverlayEpochLeaseState> state_{
            MoEOverlayEpochLeaseState::Idle}; ///< Last complete publication.
    };
} // namespace llaminar2
