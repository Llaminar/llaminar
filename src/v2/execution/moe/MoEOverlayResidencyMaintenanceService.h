/**
 * @file MoEOverlayResidencyMaintenanceService.h
 * @brief Background driver for transactional ExpertOverlay tier migration.
 *
 * Inference threads only publish routing evidence and acquire immutable
 * residency tickets. This service owns the host maintenance thread that
 * rotates completed histogram windows, retries shadow-capacity backpressure,
 * polls exact transfer/publication events, and retires old banks. No method on
 * the inference path waits for this worker.
 */

#pragma once

#include "MoEOverlayEconomyCertificationController.h"
#include "MoEOptimizationStatus.h"
#include "MoEOverlayDeviceServiceTelemetryPublisher.h"
#include "MoEOverlayResidencyProposalPublisher.h"
#include "MoEOverlayResidencyAuthority.h"
#include "../InferenceMeasurementReadiness.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace llaminar2
{
    class IMPIContext;

    /** @brief Observable lifecycle of the ExpertOverlay maintenance worker. */
    enum class MoEOverlayMaintenanceState
    {
        Prepared,   ///< Dependencies validated; no worker or protocol progress.
        Starting,   ///< Worker created but not yet inside its poll loop.
        CertifyingEconomy, ///< Real service/movement evidence is still sealing.
        AwaitingDemandActivation, ///< Certificate sealed; next request opens demand.
        Waiting,    ///< No full histogram window or migration wave is ready.
        DrainingEvidence, ///< Device histogram banks are being copied/reset.
        ReconcilingDemand, ///< A completed decision is checking queued demand.
        PlanningProposal, ///< Frozen evidence is becoming one bounded plan.
        PublishingProposal, ///< Coordinator is sending one canonical plan.
        ReceivingProposal, ///< Peer is awaiting one canonical root plan.
        Deferred,   ///< Frozen transaction awaits destination shadow capacity.
        Staging,    ///< Preparation and transfer events remain in flight.
        Preparing, ///< Inactive participant banks are being completed.
        Publishing, ///< Ready device selectors are being published globally.
        Draining,   ///< Shutdown rejected new work and is reaping resources.
        Failed,     ///< Fatal protocol or transport error stopped proposals.
        Stopped,    ///< Every wave, abort, and retirement has quiesced.
    };

    /**
     * @brief Exact coordination boundary used to stop one maintenance worker.
     *
     * Setup rollback may destroy only process-local, not-yet-live composition.
     * Terminal inference shutdown must instead keep every topology follower
     * runnable until the arbitrary proposal coordinator has drained its last
     * admitted generation. Keeping these operations distinct prevents a local
     * destructor or setup failure from accidentally entering an MPI barrier.
     */
    enum class MoEOverlayMaintenanceDrainScope : std::uint8_t
    {
        ProcessLocalComposition, ///< Prepared/local-only setup ownership.
        DistributedTopology,     ///< Terminal all-rank serving shutdown.
    };

    /**
     * @brief Project the worker lifecycle into the public optimization state.
     * @param state Exact maintenance-worker state.
     * @return Typed background activity without consulting telemetry.
     */
    [[nodiscard]] constexpr MoEOptimizationActivityState
    moeOptimizationActivityState(
        MoEOverlayMaintenanceState state) noexcept
    {
        switch (state)
        {
        case MoEOverlayMaintenanceState::Prepared:
        case MoEOverlayMaintenanceState::Starting:
        case MoEOverlayMaintenanceState::ReconcilingDemand:
        case MoEOverlayMaintenanceState::DrainingEvidence:
        case MoEOverlayMaintenanceState::AwaitingDemandActivation:
            return MoEOptimizationActivityState::ReconcilingDemand;
        case MoEOverlayMaintenanceState::CertifyingEconomy:
            return MoEOptimizationActivityState::LearningEconomy;
        case MoEOverlayMaintenanceState::Waiting:
            return MoEOptimizationActivityState::CollectingDemand;
        case MoEOverlayMaintenanceState::PlanningProposal:
            return MoEOptimizationActivityState::PlanningMovement;
        case MoEOverlayMaintenanceState::PublishingProposal:
            return MoEOptimizationActivityState::ExchangingProposal;
        case MoEOverlayMaintenanceState::ReceivingProposal:
            return MoEOptimizationActivityState::AwaitingAuthorityProposal;
        case MoEOverlayMaintenanceState::Deferred:
        case MoEOverlayMaintenanceState::Staging:
        case MoEOverlayMaintenanceState::Preparing:
            return MoEOptimizationActivityState::MovingWeights;
        case MoEOverlayMaintenanceState::Publishing:
            return MoEOptimizationActivityState::PublishingResidency;
        case MoEOverlayMaintenanceState::Draining:
        case MoEOverlayMaintenanceState::Stopped:
            return MoEOptimizationActivityState::Draining;
        case MoEOverlayMaintenanceState::Failed:
            return MoEOptimizationActivityState::Failed;
        }
        return MoEOptimizationActivityState::Failed;
    }

    /**
     * @brief Worker-owned lifecycle of the one immutable retained transaction.
     *
     * A distributed coordinator creates the transaction before publishing its
     * canonical plan. `PublishingProposal` covers both coordinator publication
     * and a peer's post-adoption acknowledgement, making premature physical
     * staging unrepresentable. Only the terminal acknowledgement transitions it
     * to `ReadyToStage`. Deferred capacity keeps that exact state and
     * transaction, while `Active` transfers ownership to the authority wave.
     */
    enum class MoEOverlayRetainedTransactionState : std::uint8_t
    {
        Empty,
        PublishingProposal,
        ReadyToStage,
        Active,
    };

    /**
     * @brief Typed ownership decision for the finite GPU service publisher.
     *
     * Readiness exchange is a temporary collective substate, not the end of
     * service observation. Keeping pause distinct from stop prevents one
     * unsuccessful all-rank readiness round from irreversibly destroying the
     * only GPU evidence source before production inference can exercise it.
     */
    enum class MoEOverlayDeviceServicePublicationAction : std::uint8_t
    {
        Pause,         ///< Preserve the publisher without launching a snapshot.
        PollAndImport, ///< Advance one boundary/publication/import edge.
        Stop,          ///< No later certification state may need new evidence.
    };

    /**
     * @brief Map certification lifecycle to exact publisher ownership.
     * @param state Current economy-certification state.
     * @return Whether maintenance must pause, poll, or permanently stop.
     */
    [[nodiscard]] constexpr MoEOverlayDeviceServicePublicationAction
    moeOverlayDeviceServicePublicationAction(
        MoEOverlayEconomyCertificationState state) noexcept
    {
        switch (state)
        {
        case MoEOverlayEconomyCertificationState::AwaitingServiceEvidence:
            return MoEOverlayDeviceServicePublicationAction::PollAndImport;
        case MoEOverlayEconomyCertificationState::CalibratingMovement:
        case MoEOverlayEconomyCertificationState::ExchangingServiceReadiness:
            return MoEOverlayDeviceServicePublicationAction::Pause;
        case MoEOverlayEconomyCertificationState::ExchangingServiceEvidence:
        case MoEOverlayEconomyCertificationState::RebasingRoutingEvidence:
        case MoEOverlayEconomyCertificationState::Complete:
        case MoEOverlayEconomyCertificationState::Failed:
        case MoEOverlayEconomyCertificationState::Stopped:
            return MoEOverlayDeviceServicePublicationAction::Stop;
        }
        return MoEOverlayDeviceServicePublicationAction::Stop;
    }

    /** @brief Race-safe counters for one maintenance-service lifetime. */
    struct MoEOverlayResidencyMaintenanceStats
    {
        uint64_t worker_starts = 0;       ///< Background worker entries.
        uint64_t poll_iterations = 0;     ///< Non-blocking authority polls.
        uint64_t notifications = 0;       ///< Explicit early-wake notifications.
        uint64_t economy_certification_polls = 0; ///< Setup evidence progress.
        uint64_t economy_certifications = 0; ///< Immutable installs observed.
        uint64_t device_service_snapshots_imported = 0; ///< GPU cumulative views accepted.
        uint64_t proposals = 0;           ///< Frozen histogram/static proposals.
        uint64_t deferred_attempts = 0;   ///< Backpressured stage attempts.
        uint64_t waves_started = 0;       ///< Async migration waves begun.
        uint64_t committed_waves = 0;     ///< Candidate epochs published.
        uint64_t dynamic_no_movement = 0; ///< Dynamic windows needing no move.
        uint64_t static_no_movement = 0;  ///< Explicit static immobility checks.
        uint64_t proposals_published = 0; ///< Coordinator send completions.
        uint64_t proposals_received = 0; ///< Authenticated peer plans.
        uint64_t proposal_receives_rearmed = 0; ///< Next-generation peer Irecvs.
        uint64_t fatal_failures = 0;      ///< Terminal service failures.
    };

    /**
     * @brief Sole background scheduler for one overlay residency authority.
     *
     * The service retains both authority and transport so their streams,
     * events, source pins, and inactive slots outlive every in-flight wave. A
     * deferred transaction is retained verbatim: later routing evidence lands
     * in the next RCU histogram bank and cannot mutate or replace the candidate
     * being retried.
     *
     * Destruction requests shutdown and drains by event polling. It never calls
     * a device/stream synchronize operation, but it deliberately waits for the
     * maintenance thread to release every owned asynchronous resource. Runner
     * teardown must therefore stop ticket admission before destroying this
     * service.
     */
    class MoEOverlayResidencyMaintenanceService final
    {
    public:
        /** @brief Construction dependencies and idle polling economy policy. */
        struct Config
        {
            /** Single publication authority shared with inference dispatch. */
            std::shared_ptr<MoEOverlayResidencyAuthority> authority;
            /** Persistent event-driven physical migration transport. */
            std::shared_ptr<IMoEOverlayResidencyTransport> transport;
            /**
             * Required for an initially uncertified dynamic local authority.
             * The maintenance worker polls it to completion before it rotates
             * or publishes any histogram window.
             */
            std::shared_ptr<MoEOverlayEconomyCertificationController>
                economy_certification;
            /**
             * Optional finite GPU observation graphs for host authority.
             *
             * CPU service measurements already land in the registry directly.
             * This owner is present only when the same host authority also has
             * local CUDA/ROCm participants whose counters require an explicit
             * mapped publication boundary.
             */
            std::shared_ptr<MoEOverlayDeviceServiceTelemetryPublisher>
                device_service_telemetry_publisher;
            /**
             * Optional authoritative canonical-proposal publication lane.
             *
             * When present, this service is in distributed mode. Only its
             * coordinator consults the process-local histogram and runs
             * policy. Peers only validate and materialize its published plan.
             */
            std::shared_ptr<IMoEOverlayResidencyProposalPublisher>
                proposal_publisher;
            /**
             * Optional topology barrier authority for a real distributed
             * proposal lane.
             *
             * Production supplies the same rank set used to construct the
             * publisher. Device-free in-process publishers leave this empty
             * and retain process-local drain semantics.
             */
            std::shared_ptr<IMPIContext> distributed_context;
            /**
             * Maximum delay before the worker polls without a notification.
             * This is a responsiveness policy, not a correctness timeout.
             */
            std::chrono::microseconds idle_poll_interval{
                std::chrono::milliseconds(2)};
            /** PerfStats device/topology label for maintenance evidence. */
            std::string perf_device;
        };

        /**
         * @brief Validate dependencies and retain a dormant prepared service.
         *
         * Construction never launches protocol progress. The owner must call
         * `start()` only after every distributed participant has committed the
         * complete composition phase.
         *
         * @throws std::invalid_argument for null dependencies or bad cadence.
         */
        explicit MoEOverlayResidencyMaintenanceService(Config config);

        /** @brief Request shutdown and drain all event-owned resources. */
        ~MoEOverlayResidencyMaintenanceService();

        MoEOverlayResidencyMaintenanceService(
            const MoEOverlayResidencyMaintenanceService &) = delete;
        MoEOverlayResidencyMaintenanceService &operator=(
            const MoEOverlayResidencyMaintenanceService &) = delete;

        /**
         * @brief Perform the sole `Prepared -> Starting` worker transition.
         *
         * This method is intentionally explicit so a rank cannot enter
         * background MPI/device progress while a peer is still composing its
         * transport, publication bank, or captured serving graph. It may be
         * called exactly once; starting a running, failed, or stopped service
         * is a lifecycle error.
         *
         * @throws std::logic_error unless the service is exactly Prepared.
         * @throws std::system_error if the host worker cannot be created.
         */
        void start();

        /**
         * @brief Wake the worker after routing evidence or capacity changes.
         *
         * Notification is optional because the worker also polls at the
         * configured cadence. The method never waits and is safe on inference
         * and destination-slot release paths.
         */
        void notifyMaintenanceProgress() noexcept;

        /**
         * @brief Stop accepting unpublished proposals and drain owned work.
         * @param scope Exact process-local or topology-terminal boundary.
         *
         * A distributed histogram generation that crossed publication remains
         * irrevocable and is completed through migration/publication before
         * shutdown. `DistributedTopology` drains the arbitrary coordinator
         * first while follower workers remain live, then drains followers
         * before any rank releases its publisher or starts prepared-context
         * restoration. `ProcessLocalComposition` rejects a live production
         * distributed worker because no one rank can prove peer quiescence.
         * This method is idempotent. It joins only the maintenance host thread;
         * GPU and network completion is observed through non-blocking polls.
         */
        void stopAndDrain(MoEOverlayMaintenanceDrainScope scope);

        /** @return Current worker lifecycle state. */
        [[nodiscard]] MoEOverlayMaintenanceState state() const noexcept;

        /** @return Whether no fatal protocol or transport failure was observed. */
        [[nodiscard]] bool healthy() const noexcept;

        /** @return Stable copy of the first fatal diagnostic, or empty string. */
        [[nodiscard]] std::string failureMessage() const;

        /** @return Race-safe copy of service counters. */
        [[nodiscard]] MoEOverlayResidencyMaintenanceStats stats() const noexcept;

        /**
         * @brief Report whether measured host-authority economics are installed.
         *
         * This method only snapshots atomics and the immutable certification
         * owner.  It does not wake or poll the worker.
         */
        [[nodiscard]] InferenceMeasurementReadiness
        measurementReadiness() const;

        /**
         * @brief Observe Dynamic economy and publication from their real owners.
         *
         * The snapshot is passive and never advances maintenance. In
         * particular, it does not derive lifecycle state from PerfStats.
         */
        [[nodiscard]] MoEOptimizationStatus optimizationStatus() const;

    private:
        /** @brief Stop and join this process-local worker with the mutex held. */
        void stopLocalAndDrainLocked();

        /** @brief Worker entry that catches all failures and owns state progress. */
        void run(std::stop_token stop_token) noexcept;

        /** @brief Perform one non-blocking maintenance iteration. */
        void pollOnce(bool allow_new_proposal);

        /**
         * @brief Publish a demand boundary reconciled through one generation.
         *
         * The worker calls this only after `progressHistogramWindow()` reports
         * no complete window. A concurrent notification advances
         * `notifications_` without changing this generation, so public status
         * remains non-quiescent until a later poll observes that notification.
         *
         * @param observed_progress_generation Notification generation captured
         *        immediately before the authoritative histogram poll.
         */
        void publishReconciledWaitingState(
            std::uint64_t observed_progress_generation) noexcept;

        /** @brief Start or retry the exact retained transaction. */
        void tryBeginRetainedTransaction();

        /** @brief Progress coordinator publication or peer reception once. */
        void progressDistributedProposal();

        /** @brief Validate and retain one exact coordinator-published proposal. */
        void retainDistributedProposal(
            std::shared_ptr<const MoEOverlayDistributedResidencyProposal>
                proposal);

        /** @brief Interpret progress for the authority-owned active wave. */
        void handleActiveWaveResult(
            const MoEOverlayResidencyApplyResult &result);

        /** @brief Record the first terminal failure and disable proposals. */
        void fail(std::string message) noexcept;

        /** @brief Export one low-frequency service counter to PerfStats. */
        void recordPerfCounter(const char *name, double value = 1.0) const;

        Config config_;
        std::jthread worker_;

        std::atomic<MoEOverlayMaintenanceState> state_{
            MoEOverlayMaintenanceState::Prepared};
        std::atomic<bool> healthy_{true};
        /** Serializes first-failure publication before `healthy_` becomes false. */
        std::atomic<bool> failure_recorded_{false};
        std::atomic<bool> shutdown_requested_{false};
        /** Serializes the one start edge against idempotent stop/drain. */
        std::mutex lifecycle_mutex_;

        std::mutex wake_mutex_;
        std::condition_variable wake_cv_;
        bool wake_requested_ = false;

        mutable std::mutex failure_mutex_;
        std::string failure_message_;

        /* These transaction fields are owned exclusively by `worker_`. */
        MoEOverlayResidencyTransaction retained_transaction_;
        MoEOverlayRetainedTransactionState retained_transaction_state_ =
            MoEOverlayRetainedTransactionState::Empty;
        bool static_check_complete_ = false;
        /** Coordinator-owned proposal retained until its MPI sends complete. */
        std::shared_ptr<const MoEOverlayDistributedResidencyProposal>
            publishing_proposal_;
        /** True while coordinator sends or the peer's next Irecv is active. */
        bool proposal_exchange_active_ = false;
        /** Worker-local edge detector for the one immutable certificate. */
        bool economy_certification_observed_ = false;
        std::atomic<uint64_t> worker_starts_{0};
        std::atomic<uint64_t> poll_iterations_{0};
        /** Published by inference/event producers before each wake. */
        std::atomic<uint64_t> notifications_{0};
        /** Latest notification generation proved empty by the policy worker. */
        std::atomic<uint64_t> reconciled_progress_generation_{0};
        std::atomic<uint64_t> economy_certification_polls_{0};
        std::atomic<uint64_t> economy_certifications_{0};
        std::atomic<uint64_t> device_service_snapshots_imported_{0};
        std::atomic<uint64_t> proposals_{0};
        std::atomic<uint64_t> deferred_attempts_{0};
        std::atomic<uint64_t> waves_started_{0};
        std::atomic<uint64_t> committed_waves_{0};
        std::atomic<uint64_t> dynamic_no_movement_{0};
        std::atomic<uint64_t> static_no_movement_{0};
        std::atomic<uint64_t> proposals_published_{0};
        std::atomic<uint64_t> proposals_received_{0};
        std::atomic<uint64_t> proposal_receives_rearmed_{0};
        std::atomic<uint64_t> fatal_failures_{0};
    };

} // namespace llaminar2
