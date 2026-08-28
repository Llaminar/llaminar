/**
 * @file MoEOverlayResidencyMaintenanceService.cpp
 * @brief Event-polled background scheduling for ExpertOverlay residency waves.
 */

#include "MoEOverlayResidencyMaintenanceService.h"

#include "interfaces/IMPIContext.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"

#include <exception>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /** @brief Return whether an apply status is a terminal protocol failure. */
        bool isFailureStatus(MoEOverlayResidencyApplyStatus status) noexcept
        {
            return status == MoEOverlayResidencyApplyStatus::Busy ||
                   status == MoEOverlayResidencyApplyStatus::Stale ||
                   status == MoEOverlayResidencyApplyStatus::StageFailed ||
                   status ==
                       MoEOverlayResidencyApplyStatus::PreparationFailed ||
                   status ==
                       MoEOverlayResidencyApplyStatus::PublicationFailed ||
                   status ==
                       MoEOverlayResidencyApplyStatus::RetirementFailed;
        }
    } // namespace

    MoEOverlayResidencyMaintenanceService::
        MoEOverlayResidencyMaintenanceService(Config config)
        : config_(std::move(config))
    {
        if (!config_.authority)
        {
            throw std::invalid_argument(
                "ExpertOverlay maintenance requires a residency authority");
        }
        if (!config_.transport)
        {
            throw std::invalid_argument(
                "ExpertOverlay maintenance requires a migration transport");
        }
        if (config_.proposal_publisher &&
            !config_.authority->migrationEnabled())
        {
            throw std::invalid_argument(
                "Distributed ExpertOverlay histogram publication requires Dynamic maintenance");
        }
        const bool dynamic =
            config_.authority->migrationEnabled();
        const bool certified =
            config_.authority->hasEconomyCertification();
        if (dynamic && !certified && !config_.economy_certification)
        {
            throw std::invalid_argument(
                "Dynamic ExpertOverlay maintenance requires measured economy certification before proposals");
        }
        if (config_.economy_certification &&
            (!dynamic || certified))
        {
            throw std::invalid_argument(
                "ExpertOverlay economy certification requires one uncertified dynamic authority");
        }
        if (config_.device_service_telemetry_publisher &&
            !config_.economy_certification)
        {
            throw std::invalid_argument(
                "GPU service telemetry publication requires an active host economy certifier");
        }
        if (config_.idle_poll_interval <= std::chrono::microseconds::zero())
        {
            throw std::invalid_argument(
                "ExpertOverlay maintenance poll interval must be positive");
        }

        /* Peer publishers guarantee their first exact-size receive is armed. */
        proposal_exchange_active_ =
            config_.proposal_publisher &&
            !config_.proposal_publisher->isCoordinator();

    }

    MoEOverlayResidencyMaintenanceService::
        ~MoEOverlayResidencyMaintenanceService()
    {
        try
        {
            stopAndDrain(
                MoEOverlayMaintenanceDrainScope::ProcessLocalComposition);
        }
        catch (...)
        {
            /* A join failure leaves asynchronous resource ownership ambiguous. */
            std::terminate();
        }
    }

    void MoEOverlayResidencyMaintenanceService::start()
    {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
        if (state_.load(std::memory_order_acquire) !=
                MoEOverlayMaintenanceState::Prepared ||
            worker_.joinable() ||
            shutdown_requested_.load(std::memory_order_acquire))
        {
            throw std::logic_error(
                "ExpertOverlay maintenance may start exactly once from Prepared");
        }

        state_.store(
            MoEOverlayMaintenanceState::Starting,
            std::memory_order_release);
        try
        {
            /*
             * Capture `this` only after every dependency, synchronization
             * object, and distributed composition phase is complete.
             * stopAndDrain() joins this worker before member destruction.
             */
            worker_ = std::jthread(
                [this](std::stop_token stop_token)
                { run(stop_token); });
        }
        catch (const std::exception &error)
        {
            fail(
                std::string("ExpertOverlay maintenance worker creation failed: ") +
                error.what());
            throw;
        }
        catch (...)
        {
            fail(
                "ExpertOverlay maintenance worker creation raised a non-standard exception");
            throw;
        }
    }

    void MoEOverlayResidencyMaintenanceService::
        notifyMaintenanceProgress() noexcept
    {
        /* Publish the generation before the wake. Status readers can now
         * reject a stale Waiting state even if the worker has not run yet. */
        notifications_.fetch_add(1, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lock(wake_mutex_);
            wake_requested_ = true;
        }
        wake_cv_.notify_one();
    }

    void MoEOverlayResidencyMaintenanceService::stopAndDrain(
        MoEOverlayMaintenanceDrainScope scope)
    {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
        if (state_.load(std::memory_order_acquire) ==
            MoEOverlayMaintenanceState::Stopped)
        {
            return;
        }

        const bool distributed =
            config_.proposal_publisher && config_.distributed_context &&
            config_.distributed_context->world_size() > 1;
        if (scope ==
            MoEOverlayMaintenanceDrainScope::ProcessLocalComposition)
        {
            if (distributed && worker_.joinable())
            {
                throw std::logic_error(
                    "A live distributed ExpertOverlay maintenance worker requires a topology drain");
            }
            stopLocalAndDrainLocked();
            return;
        }

        if (!distributed)
        {
            stopLocalAndDrainLocked();
            return;
        }

        /*
         * Closing inference admission is process-local, whereas proposal
         * publication is topology-wide. Keep every follower worker alive
         * until the arbitrary coordinator has retired the last generation it
         * could have admitted before shutdown. This is the same leader-first
         * drain shape used by the device-resident controller service.
         *
         * Barrier 1: every rank has closed inference and still owns a live
         * worker. Barrier 2: the coordinator can no longer publish. Barrier 3:
         * followers have drained/cancelled only passive receives, so prepared-
         * context restoration may begin symmetrically on every rank.
         */
        const bool coordinator =
            config_.proposal_publisher->isCoordinator();
        config_.distributed_context->barrier();
        if (coordinator)
            stopLocalAndDrainLocked();
        config_.distributed_context->barrier();
        if (!coordinator)
            stopLocalAndDrainLocked();
        config_.distributed_context->barrier();
        recordPerfCounter("maintenance_distributed_topology_drains");
    }

    void MoEOverlayResidencyMaintenanceService::
        stopLocalAndDrainLocked()
    {
        if (!worker_.joinable())
        {
            const auto current = state_.load(std::memory_order_acquire);
            if (current == MoEOverlayMaintenanceState::Stopped)
                return;

            /*
             * A prepared service owns no active wave. Close subordinate
             * observation controllers without manufacturing a worker solely
             * for teardown, then make the terminal explicit.
             */
            shutdown_requested_.store(true, std::memory_order_release);
            if (config_.economy_certification)
                config_.economy_certification->requestStop();
            if (config_.device_service_telemetry_publisher)
                config_.device_service_telemetry_publisher->requestStop();
            state_.store(
                MoEOverlayMaintenanceState::Stopped,
                std::memory_order_release);
            return;
        }
        if (worker_.get_id() == std::this_thread::get_id())
        {
            throw std::logic_error(
                "ExpertOverlay maintenance worker cannot join itself");
        }

        /*
         * The explicit flag closes the small interval in which the stop token
         * could be requested while a histogram proposal is being constructed.
         * The worker rechecks it before enqueueing physical work.
         */
        shutdown_requested_.store(true, std::memory_order_release);
        worker_.request_stop();
        notifyMaintenanceProgress();
        worker_.join();
    }

    MoEOverlayMaintenanceState
    MoEOverlayResidencyMaintenanceService::state() const noexcept
    {
        return state_.load(std::memory_order_acquire);
    }

    bool MoEOverlayResidencyMaintenanceService::healthy() const noexcept
    {
        return healthy_.load(std::memory_order_acquire);
    }

    std::string
    MoEOverlayResidencyMaintenanceService::failureMessage() const
    {
        std::lock_guard<std::mutex> lock(failure_mutex_);
        return failure_message_;
    }

    MoEOverlayResidencyMaintenanceStats
    MoEOverlayResidencyMaintenanceService::stats() const noexcept
    {
        return {
            .worker_starts = worker_starts_.load(std::memory_order_relaxed),
            .poll_iterations =
                poll_iterations_.load(std::memory_order_relaxed),
            .notifications = notifications_.load(std::memory_order_relaxed),
            .economy_certification_polls =
                economy_certification_polls_.load(
                    std::memory_order_relaxed),
            .economy_certifications =
                economy_certifications_.load(std::memory_order_relaxed),
            .device_service_snapshots_imported =
                device_service_snapshots_imported_.load(
                    std::memory_order_relaxed),
            .proposals = proposals_.load(std::memory_order_relaxed),
            .deferred_attempts =
                deferred_attempts_.load(std::memory_order_relaxed),
            .waves_started = waves_started_.load(std::memory_order_relaxed),
            .committed_waves =
                committed_waves_.load(std::memory_order_relaxed),
            .dynamic_no_movement =
                dynamic_no_movement_.load(std::memory_order_relaxed),
            .static_no_movement =
                static_no_movement_.load(std::memory_order_relaxed),
            .proposals_published =
                proposals_published_.load(
                    std::memory_order_relaxed),
            .proposals_received =
                proposals_received_.load(
                    std::memory_order_relaxed),
            .proposal_receives_rearmed =
                proposal_receives_rearmed_.load(
                    std::memory_order_relaxed),
            .fatal_failures = fatal_failures_.load(std::memory_order_relaxed),
        };
    }

    InferenceMeasurementReadiness
    MoEOverlayResidencyMaintenanceService::measurementReadiness() const
    {
        if (!healthy())
        {
            return {
                .state = InferenceMeasurementReadinessState::Failed,
                .owner = "expert_overlay_host_maintenance",
                .phase = "failed",
                .diagnostic = failureMessage().empty()
                                  ? "ExpertOverlay host maintenance failed"
                                  : failureMessage(),
            };
        }
        if (!config_.economy_certification)
            return {};
        return config_.economy_certification->measurementReadiness();
    }

    MoEOptimizationStatus
    MoEOverlayResidencyMaintenanceService::optimizationStatus() const
    {
        MoEOptimizationStatus status{
            .authority = MoEOptimizationAuthority::Host,
            .state = MoEOptimizationLifecycleState::MovementDisabled,
            .activity = MoEOptimizationActivityState::Dormant,
        };
        if (config_.authority)
        {
            const auto authority_stats = config_.authority->stats();
            status.published_movement_waves = authority_stats.committed_waves;
            status.completed_movement = {
                .transactions = authority_stats.committed_waves,
                .commands = authority_stats.committed_migrations,
                .physical_bytes = config_.transport
                                      ? config_.transport
                                            ->completedPlacementPayloadBytes()
                                      : 0u,
                .promotions = authority_stats.promotions,
                .demotions = authority_stats.demotions,
                .same_priority_moves = authority_stats.same_priority_moves,
            };
        }

        /* Only the coordinator (or a process-local authority) reconciles
         * demand. Distributed followers own a passive preposted mailbox and
         * therefore retain the normalized zero/zero generation pair. Load the
         * reconciled edge first so a concurrent producer can only make the
         * resulting snapshot conservatively non-quiescent. */
        const bool owns_demand_reconciliation =
            !config_.proposal_publisher ||
            config_.proposal_publisher->isCoordinator();
        if (owns_demand_reconciliation)
        {
            status.reconciled_progress_generation =
                reconciled_progress_generation_.load(
                    std::memory_order_acquire);
            status.published_progress_generation =
                notifications_.load(std::memory_order_acquire);
            /* The acquire above makes routed-row writes preceding the matching
             * inference notification visible before we snapshot the active RCU
             * bank. Distributed followers do not own this admission fact. */
            status.demand_window =
                config_.authority->optimizationDemandWindow();
        }

        if (!healthy())
        {
            status.state = MoEOptimizationLifecycleState::Failed;
            status.activity = MoEOptimizationActivityState::Failed;
            status.diagnostic = failureMessage().empty()
                                    ? "ExpertOverlay host maintenance failed"
                                    : failureMessage();
            return status;
        }
        if (!config_.authority ||
            config_.authority->maintenanceMode() !=
                MoERebalanceRuntimeMode::Dynamic)
        {
            return status;
        }

        /*
         * The residency authority owns the installed certificate. Observe it
         * before the helper controller so the brief release-publication edge
         * between installation and controller completion cannot make policy
         * appear less ready than it really is.
         */
        if (config_.authority->hasEconomyCertification())
        {
            status.state = MoEOptimizationLifecycleState::Active;
            status.activity = moeOptimizationActivityState(state());
            return status;
        }
        if (!config_.economy_certification)
        {
            status.state = MoEOptimizationLifecycleState::Failed;
            status.activity = MoEOptimizationActivityState::Failed;
            status.diagnostic =
                "Dynamic ExpertOverlay host authority has no economy certification owner";
            return status;
        }

        switch (config_.economy_certification->state())
        {
        case MoEOverlayEconomyCertificationState::Failed:
        case MoEOverlayEconomyCertificationState::Stopped:
            status.state = MoEOptimizationLifecycleState::Failed;
            status.activity = MoEOptimizationActivityState::Failed;
            status.diagnostic =
                config_.economy_certification->failureMessage();
            if (status.diagnostic.empty())
            {
                status.diagnostic =
                    "Dynamic ExpertOverlay economy certification stopped before activation";
            }
            break;
        case MoEOverlayEconomyCertificationState::Complete:
            status.state = MoEOptimizationLifecycleState::Failed;
            status.diagnostic =
                "Completed ExpertOverlay economy certification was not installed in its host authority";
            break;
        case MoEOverlayEconomyCertificationState::CalibratingMovement:
        case MoEOverlayEconomyCertificationState::AwaitingServiceEvidence:
        case MoEOverlayEconomyCertificationState::ExchangingServiceReadiness:
        case MoEOverlayEconomyCertificationState::ExchangingServiceEvidence:
        case MoEOverlayEconomyCertificationState::RebasingRoutingEvidence:
            status.state = MoEOptimizationLifecycleState::LearningEconomy;
            status.activity = MoEOptimizationActivityState::LearningEconomy;
            break;
        }
        return status;
    }

    void MoEOverlayResidencyMaintenanceService::
        publishReconciledWaitingState(
            std::uint64_t observed_progress_generation) noexcept
    {
        /* The generation is published before Waiting. A status reader that
         * observes Waiting therefore also observes this reconciliation. Any
         * producer racing either store increments `notifications_`, making the
         * two public generations unequal until the next authoritative poll. */
        reconciled_progress_generation_.store(
            observed_progress_generation,
            std::memory_order_release);
        state_.store(
            MoEOverlayMaintenanceState::Waiting,
            std::memory_order_release);
    }

    void MoEOverlayResidencyMaintenanceService::run(
        std::stop_token stop_token) noexcept
    {
        worker_starts_.fetch_add(1, std::memory_order_relaxed);
        recordPerfCounter("maintenance_worker_starts");
        /* Enter through reconciliation. `Waiting` is reserved for a poll that
         * actually observed no complete histogram window; publishing it here
         * would expose a false between-wave boundary before the first poll. */
        state_.store(
            MoEOverlayMaintenanceState::ReconcilingDemand,
            std::memory_order_release);

        while (true)
        {
            const bool stopping =
                stop_token.stop_requested() ||
                shutdown_requested_.load(std::memory_order_acquire);
            if (stopping)
            {
                state_.store(
                    MoEOverlayMaintenanceState::Draining,
                    std::memory_order_release);

                /*
                 * A process-local deferred proposal owns no externally visible
                 * state and may be discarded. A distributed proposal cannot:
                 * once its frozen histogram publication began, a peer may
                 * already have derived or staged the same generation. The
                 * coordinator must finish that exact generation before the
                 * runner releases remote worker loops.
                 */
                if (!config_.proposal_publisher)
                {
                    retained_transaction_ = {};
                    retained_transaction_state_ =
                        MoEOverlayRetainedTransactionState::Empty;
                }
                if (config_.economy_certification)
                    config_.economy_certification->requestStop();
                if (config_.device_service_telemetry_publisher)
                {
                    config_.device_service_telemetry_publisher->requestStop();
                }
            }

            try
            {
                pollOnce(!stopping && healthy());
            }
            catch (const std::exception &error)
            {
                fail(error.what());
            }
            catch (...)
            {
                fail(
                    "ExpertOverlay maintenance raised a non-standard exception");
            }

            const bool distributed_coordinator_drained =
                !config_.proposal_publisher ||
                !config_.proposal_publisher->isCoordinator() ||
                (!proposal_exchange_active_ && !publishing_proposal_);
            if (stopping &&
                retained_transaction_state_ ==
                    MoEOverlayRetainedTransactionState::Empty &&
                distributed_coordinator_drained &&
                !config_.authority->hasActiveBackgroundWave() &&
                config_.authority->pendingAbortCount() == 0 &&
                config_.authority->pendingRetirementCount() == 0 &&
                (!config_.economy_certification ||
                 config_.economy_certification->state() ==
                     MoEOverlayEconomyCertificationState::Complete ||
                 config_.economy_certification->state() ==
                     MoEOverlayEconomyCertificationState::Failed ||
                 config_.economy_certification->state() ==
                     MoEOverlayEconomyCertificationState::Stopped))
            {
                state_.store(
                    MoEOverlayMaintenanceState::Stopped,
                    std::memory_order_release);
                return;
            }

            std::unique_lock<std::mutex> wake_lock(wake_mutex_);
            wake_cv_.wait_for(
                wake_lock,
                config_.idle_poll_interval,
                [this, &stop_token]
                {
                    return wake_requested_ || stop_token.stop_requested() ||
                           shutdown_requested_.load(std::memory_order_acquire);
                });
            wake_requested_ = false;
        }
    }

    void MoEOverlayResidencyMaintenanceService::pollOnce(
        bool allow_new_proposal)
    {
        poll_iterations_.fetch_add(1, std::memory_order_relaxed);

        if (config_.economy_certification &&
            !economy_certification_observed_)
        {
            if (!allow_new_proposal)
                config_.economy_certification->requestStop();
            const auto pre_poll_certification_state =
                config_.economy_certification->state();
            const auto publication_action =
                moeOverlayDeviceServicePublicationAction(
                    pre_poll_certification_state);
            if (config_.device_service_telemetry_publisher &&
                publication_action ==
                    MoEOverlayDeviceServicePublicationAction::PollAndImport)
            {
                std::vector<MoEOverlayDeviceServiceTelemetrySnapshot>
                    snapshots;
                std::string publication_error;
                if (!config_.device_service_telemetry_publisher->poll(
                        &snapshots, &publication_error))
                {
                    fail(
                        publication_error.empty()
                            ? config_.device_service_telemetry_publisher
                                  ->failureMessage()
                            : std::move(publication_error));
                    return;
                }
                for (auto &snapshot : snapshots)
                {
                    if (!snapshot.valid())
                    {
                        fail(
                            "Host economy maintenance received an invalid GPU service snapshot");
                        return;
                    }
                    std::string import_error;
                    if (!config_.economy_certification
                             ->importDeviceServiceMeasurements(
                                 snapshot.participant_id,
                                 snapshot.rows,
                                 &import_error))
                    {
                        fail(
                            import_error.empty()
                                ? "Host economy certification rejected a GPU service snapshot"
                                : std::move(import_error));
                        return;
                    }
                    device_service_snapshots_imported_.fetch_add(
                        1u, std::memory_order_relaxed);
                    recordPerfCounter(
                        "maintenance_device_service_snapshots_imported");
                }
            }
            else if (config_.device_service_telemetry_publisher &&
                     publication_action ==
                         MoEOverlayDeviceServicePublicationAction::Stop)
            {
                /* A Ready round retained its immutable registry snapshot
                 * before entering service exchange. No later lifecycle state
                 * can return to evidence collection, so the publisher may now
                 * drain permanently. Readiness exchange itself merely pauses:
                 * an AwaitingEvidence result must be able to resume polling. */
                config_.device_service_telemetry_publisher->requestStop();
            }
            economy_certification_polls_.fetch_add(
                1, std::memory_order_relaxed);
            config_.economy_certification->poll();
            if (!config_.economy_certification->healthy())
            {
                fail(config_.economy_certification->failureMessage());
                return;
            }
            const auto certification_state =
                config_.economy_certification->state();
            if (certification_state ==
                MoEOverlayEconomyCertificationState::Complete)
            {
                if (!config_.authority->hasEconomyCertification())
                {
                    fail(
                        "ExpertOverlay certification completed without installing authority economics");
                    return;
                }
                economy_certification_observed_ = true;
                economy_certifications_.fetch_add(
                    1, std::memory_order_relaxed);
                recordPerfCounter("maintenance_economy_certifications");
            }
            else
            {
                if (allow_new_proposal)
                {
                    state_.store(
                        MoEOverlayMaintenanceState::CertifyingEconomy,
                        std::memory_order_release);
                }
                return;
            }
        }

        if (config_.authority->hasEconomyCertification() &&
            !config_.authority->optimizationDemandActive())
        {
            /*
             * Installation may occur during an admitted request, but movement
             * demand begins only at the next public prefill boundary. Keep the
             * worker passive while CPU and device histogram writers remain in
             * their shared quarantine; no proposal may consume that bank.
             */
            state_.store(
                MoEOverlayMaintenanceState::AwaitingDemandActivation,
                std::memory_order_release);
            return;
        }

        if (retained_transaction_state_ ==
            MoEOverlayRetainedTransactionState::Active)
        {
            handleActiveWaveResult(config_.authority->advanceBackground());
            return;
        }

        /* Reap lease-safe retirements and completed asynchronous aborts. */
        const auto idle_result = config_.authority->advanceBackground();
        if (idle_result.status != MoEOverlayResidencyApplyStatus::Idle)
        {
            if (isFailureStatus(idle_result.status))
            {
                fail(
                    idle_result.error.empty()
                        ? "ExpertOverlay authority failed while reaping background resources"
                        : idle_result.error);
                return;
            }
            fail(
                "ExpertOverlay authority reported active work not owned by its maintenance service");
            return;
        }

        if (retained_transaction_state_ ==
            MoEOverlayRetainedTransactionState::ReadyToStage)
        {
            tryBeginRetainedTransaction();
            return;
        }

        if (!allow_new_proposal)
        {
            /*
             * Only the histogram coordinator can own an irrevocable send at
             * this point. Peers stopping after the coordinated root has
             * drained hold at most a passive preposted receive, which their
             * publisher cancels during runner teardown.
             */
            if (config_.proposal_publisher &&
                config_.proposal_publisher->isCoordinator() &&
                proposal_exchange_active_)
            {
                progressDistributedProposal();
            }
            return;
        }

        if (config_.proposal_publisher)
        {
            progressDistributedProposal();
            return;
        }

        const bool dynamic = config_.authority->migrationEnabled();
        std::shared_ptr<const DecodeExpertHistogramWindow> local_window;
        if (dynamic)
        {
            const std::uint64_t observed_progress_generation =
                notifications_.load(std::memory_order_acquire);
            const auto window_result =
                config_.authority->progressHistogramWindow();
            if (window_result.progress ==
                MoEOverlayHistogramWindowProgress::Waiting)
            {
                publishReconciledWaitingState(
                    observed_progress_generation);
                return;
            }
            if (window_result.progress ==
                MoEOverlayHistogramWindowProgress::Pending)
            {
                state_.store(
                    MoEOverlayMaintenanceState::DrainingEvidence,
                    std::memory_order_release);
                recordPerfCounter("maintenance_histogram_drain_polls");
                return;
            }
            local_window = window_result.window;
            if (!local_window || !local_window->valid())
            {
                fail(
                    "ExpertOverlay maintenance prepared an invalid local histogram window");
                return;
            }
        }
        if (!dynamic && static_check_complete_)
        {
            state_.store(
                MoEOverlayMaintenanceState::Waiting,
                std::memory_order_release);
            return;
        }

        state_.store(
            MoEOverlayMaintenanceState::PlanningProposal,
            std::memory_order_release);
        retained_transaction_ = dynamic
                                    ? config_.authority
                                          ->proposeFromFrozenHistogramWindow(
                                              std::move(local_window))
                                    : config_.authority->proposeFromHistogram();
        retained_transaction_state_ =
            MoEOverlayRetainedTransactionState::ReadyToStage;
        proposals_.fetch_add(1, std::memory_order_relaxed);
        recordPerfCounter("maintenance_proposals");

        /* Do not launch new process-local work after a concurrent shutdown. */
        if (shutdown_requested_.load(std::memory_order_acquire))
        {
            retained_transaction_ = {};
            retained_transaction_state_ =
                MoEOverlayRetainedTransactionState::Empty;
            return;
        }
        tryBeginRetainedTransaction();
    }

    void MoEOverlayResidencyMaintenanceService::progressDistributedProposal()
    {
        auto &publisher = *config_.proposal_publisher;
        if (publisher.isCoordinator() && !proposal_exchange_active_)
        {
            const std::uint64_t observed_progress_generation =
                notifications_.load(std::memory_order_acquire);
            const auto window_result =
                config_.authority->progressHistogramWindow();
            if (window_result.progress ==
                MoEOverlayHistogramWindowProgress::Waiting)
            {
                publishReconciledWaitingState(
                    observed_progress_generation);
                return;
            }
            if (window_result.progress ==
                MoEOverlayHistogramWindowProgress::Pending)
            {
                state_.store(
                    MoEOverlayMaintenanceState::DrainingEvidence,
                    std::memory_order_release);
                recordPerfCounter("maintenance_histogram_drain_polls");
                return;
            }

            if (!window_result.window || !window_result.window->valid())
            {
                fail(
                    "ExpertOverlay coordinator froze an invalid histogram window");
                return;
            }

            /*
             * The continuation root is the sole distributed policy authority.
             * It performs smoothing, placement, skew, wave bounding, and
             * economy once, then publishes the selected executable candidate.
             * Peers never repeat this decision from rank-local measurements.
             */
            state_.store(
                MoEOverlayMaintenanceState::PlanningProposal,
                std::memory_order_release);
            auto transaction =
                config_.authority->proposeFromFrozenHistogramWindow(
                    std::move(window_result.window));
            auto proposal =
                std::make_shared<MoEOverlayDistributedResidencyProposal>(
                    makeMoEOverlayDistributedResidencyProposal(
                        config_.authority
                            ->exportAuthoritativeResidencyPlan(transaction),
                        transaction));
            std::string error;
            if (!publisher.beginPublish(*proposal, &error))
            {
                fail(
                    error.empty()
                        ? "ExpertOverlay coordinator failed to begin proposal publication"
                        : std::move(error));
                return;
            }

            retained_transaction_ = std::move(transaction);
            retained_transaction_state_ =
                MoEOverlayRetainedTransactionState::PublishingProposal;
            publishing_proposal_ = std::move(proposal);
            proposals_.fetch_add(1, std::memory_order_relaxed);
            proposal_exchange_active_ = true;
            state_.store(
                MoEOverlayMaintenanceState::PublishingProposal,
                std::memory_order_release);
            recordPerfCounter("maintenance_distributed_proposals");
            recordPerfCounter("maintenance_proposal_publications_started");
            LOG_INFO(
                "[ExpertOverlay][Residency] Authored distributed proposal"
                << " device=" << config_.perf_device
                << " histogram_generation="
                << retained_transaction_.histogram_generation
                << " expected_epoch="
                << retained_transaction_.expected_epoch
                << " migrations="
                << retained_transaction_.migrations.size()
                << " cycles="
                << retained_transaction_.migration_cycles.size()
                << " service_profile="
                << (retained_transaction_.economy.enabled
                        ? retained_transaction_.economy
                              .service_profile_identity
                        : std::string{"disabled"})
                << " projected_net_benefit_ns="
                << retained_transaction_.economy
                       .projected_net_benefit_ns);
            return;
        }

        if (!proposal_exchange_active_)
        {
            fail(
                "ExpertOverlay peer has no armed authoritative proposal receive");
            return;
        }

        std::shared_ptr<const MoEOverlayDistributedResidencyProposal>
            received_proposal;
        std::string error;
        const auto progress = publisher.poll(&received_proposal, &error);
        if (progress == MoEOverlayResidencyWaveProgress::Pending)
        {
            state_.store(
                publisher.isCoordinator()
                    ? MoEOverlayMaintenanceState::PublishingProposal
                    : MoEOverlayMaintenanceState::ReceivingProposal,
                std::memory_order_release);
            return;
        }
        if (progress != MoEOverlayResidencyWaveProgress::Ready)
        {
            fail(
                error.empty()
                    ? "ExpertOverlay authoritative proposal publication failed"
                    : std::move(error));
            return;
        }

        if (publisher.isCoordinator())
        {
            proposal_exchange_active_ = false;
            proposals_published_.fetch_add(
                1,
                std::memory_order_relaxed);
            recordPerfCounter("maintenance_proposals_published");
            auto proposal = std::move(publishing_proposal_);
            if (!proposal || received_proposal ||
                retained_transaction_state_ !=
                    MoEOverlayRetainedTransactionState::
                        PublishingProposal)
            {
                fail(
                    "ExpertOverlay coordinator publication returned invalid proposal ownership");
                return;
            }

            /*
             * Peer acknowledgement is the irrevocability edge. Every rank now
             * owns the same authenticated plan, so this exact retained root
             * transaction must reach a terminal distributed result.
             */
            retained_transaction_state_ =
                MoEOverlayRetainedTransactionState::ReadyToStage;
            tryBeginRetainedTransaction();
            return;
        }

        if (received_proposal)
        {
            retainDistributedProposal(std::move(received_proposal));
            return;
        }

        /*
         * The second Ready edge is the completion of an acknowledgement that
         * could only be posted after semantic adoption and fingerprint proof.
         * Rearm the fixed receive before staging so generation N+1 has a mailbox,
         * but do not let bytes alone make the transaction executable.
         */
        if (retained_transaction_state_ !=
            MoEOverlayRetainedTransactionState::PublishingProposal)
        {
            fail(
                "ExpertOverlay peer proposal acknowledgement completed without an adopted transaction");
            return;
        }
        if (!publisher.armReceive(&error))
        {
            fail(
                error.empty()
                    ? "ExpertOverlay peer failed to arm its next proposal receive"
                    : std::move(error));
            return;
        }
        proposal_exchange_active_ = true;
        proposal_receives_rearmed_.fetch_add(1, std::memory_order_relaxed);
        recordPerfCounter("maintenance_proposal_receives_rearmed");
        retained_transaction_state_ =
            MoEOverlayRetainedTransactionState::ReadyToStage;
        tryBeginRetainedTransaction();
    }

    void MoEOverlayResidencyMaintenanceService::retainDistributedProposal(
        std::shared_ptr<const MoEOverlayDistributedResidencyProposal>
            proposal)
    {
        const std::uint64_t generation =
            proposal && proposal->plan.histogram_window
                ? proposal->plan.histogram_window->generation
                : 0u;
        const auto reject = [&](std::string diagnostic)
        {
            retained_transaction_ = {};
            config_.proposal_publisher->rejectReceivedProposal(
                generation, diagnostic);
            /* Device-free adversarial publishers return so unit tests can
             * observe the exact failure. Production MPI rejection is fatal. */
            fail(std::move(diagnostic));
        };

        if (!proposal || !proposal->valid() ||
            retained_transaction_state_ !=
                MoEOverlayRetainedTransactionState::Empty)
        {
            reject(
                "ExpertOverlay maintenance cannot retain the published proposal in its current state");
            return;
        }

        try
        {
            retained_transaction_ =
                config_.authority->adoptAuthoritativeResidencyPlan(
                    proposal->plan);
        }
        catch (const std::exception &error)
        {
            reject(std::string{
                       "ExpertOverlay peer rejected authoritative plan: "} +
                   error.what());
            return;
        }
        catch (...)
        {
            reject(
                "ExpertOverlay peer rejected authoritative plan with a non-standard exception");
            return;
        }
        const auto reconstructed_fingerprint =
            fingerprintMoEOverlayResidencyExecutionPlan(
                retained_transaction_);
        if (reconstructed_fingerprint !=
            proposal->execution_fingerprint)
        {
            reject(
                "ExpertOverlay peer topology reconstructed a different execution plan from the root proposal");
            return;
        }
        std::string acknowledgement_error;
        if (!config_.proposal_publisher->acceptReceivedProposal(
                generation, &acknowledgement_error))
        {
            reject(
                acknowledgement_error.empty()
                    ? "ExpertOverlay peer failed to acknowledge its adopted authoritative plan"
                    : std::move(acknowledgement_error));
            return;
        }
        retained_transaction_state_ =
            MoEOverlayRetainedTransactionState::PublishingProposal;
        proposals_.fetch_add(1, std::memory_order_relaxed);
        proposals_received_.fetch_add(1, std::memory_order_relaxed);
        recordPerfCounter("maintenance_distributed_proposals");
        recordPerfCounter("maintenance_proposals_received");
        LOG_INFO(
            "[ExpertOverlay][Residency] Adopted distributed proposal"
            << " device=" << config_.perf_device
            << " histogram_generation="
            << retained_transaction_.histogram_generation
            << " expected_epoch="
            << retained_transaction_.expected_epoch
            << " migrations="
            << retained_transaction_.migrations.size()
            << " cycles="
            << retained_transaction_.migration_cycles.size()
            << " root_policy_fingerprint_low="
            << proposal->policy_fingerprint.low
            << " root_policy_fingerprint_high="
            << proposal->policy_fingerprint.high);

        /* The transaction remains non-executable until the async semantic
         * acknowledgement itself reaches its terminal Ready edge. */
    }

    void MoEOverlayResidencyMaintenanceService::
        tryBeginRetainedTransaction()
    {
        if (retained_transaction_state_ !=
            MoEOverlayRetainedTransactionState::ReadyToStage)
        {
            fail(
                "ExpertOverlay maintenance attempted to stage a missing transaction");
            return;
        }

        const bool dynamic = config_.authority->migrationEnabled();
        const auto result = config_.authority->beginApply(
            retained_transaction_,
            *config_.transport);
        switch (result.status)
        {
        case MoEOverlayResidencyApplyStatus::Started:
            retained_transaction_state_ =
                MoEOverlayRetainedTransactionState::Active;
            waves_started_.fetch_add(1, std::memory_order_relaxed);
            recordPerfCounter("maintenance_waves_started");
            state_.store(
                MoEOverlayMaintenanceState::Staging,
                std::memory_order_release);
            return;

        case MoEOverlayResidencyApplyStatus::Deferred:
            deferred_attempts_.fetch_add(1, std::memory_order_relaxed);
            recordPerfCounter("maintenance_deferred_attempts");
            state_.store(
                MoEOverlayMaintenanceState::Deferred,
                std::memory_order_release);
            return;

        case MoEOverlayResidencyApplyStatus::DynamicNoMovement:
            if (!dynamic)
            {
                fail(
                    "Static ExpertOverlay maintenance returned a dynamic no-movement result");
                return;
            }
            dynamic_no_movement_.fetch_add(1, std::memory_order_relaxed);
            recordPerfCounter("maintenance_dynamic_no_movement");
            retained_transaction_ = {};
            retained_transaction_state_ =
                MoEOverlayRetainedTransactionState::Empty;
            /* A second completed window may already be queued. Reconcile it
             * on the next poll before advertising an idle demand boundary. */
            state_.store(
                MoEOverlayMaintenanceState::ReconcilingDemand,
                std::memory_order_release);
            return;

        case MoEOverlayResidencyApplyStatus::StaticNoMovement:
            if (dynamic)
            {
                fail(
                    "Dynamic ExpertOverlay maintenance returned a static no-movement result");
                return;
            }
            static_check_complete_ = true;
            static_no_movement_.fetch_add(1, std::memory_order_relaxed);
            recordPerfCounter("maintenance_static_no_movement");
            retained_transaction_ = {};
            retained_transaction_state_ =
                MoEOverlayRetainedTransactionState::Empty;
            state_.store(
                MoEOverlayMaintenanceState::Waiting,
                std::memory_order_release);
            return;

        default:
            fail(
                result.error.empty()
                    ? "ExpertOverlay authority rejected a maintenance transaction"
                    : result.error);
            return;
        }
    }

    void MoEOverlayResidencyMaintenanceService::handleActiveWaveResult(
        const MoEOverlayResidencyApplyResult &result)
    {
        switch (result.status)
        {
        case MoEOverlayResidencyApplyStatus::Staging:
            state_.store(
                MoEOverlayMaintenanceState::Staging,
                std::memory_order_release);
            return;
        case MoEOverlayResidencyApplyStatus::Preparing:
            state_.store(
                MoEOverlayMaintenanceState::Preparing,
                std::memory_order_release);
            return;
        case MoEOverlayResidencyApplyStatus::Publishing:
            state_.store(
                MoEOverlayMaintenanceState::Publishing,
                std::memory_order_release);
            return;
        case MoEOverlayResidencyApplyStatus::Published:
            retained_transaction_ = {};
            retained_transaction_state_ =
                MoEOverlayRetainedTransactionState::Empty;
            committed_waves_.fetch_add(1, std::memory_order_relaxed);
            recordPerfCounter("maintenance_waves_committed");
            /* Publication completes one wave, not the decision that no later
             * frozen window exists. The following poll owns that observation. */
            state_.store(
                MoEOverlayMaintenanceState::ReconcilingDemand,
                std::memory_order_release);
            return;
        case MoEOverlayResidencyApplyStatus::Deferred:
            retained_transaction_state_ =
                MoEOverlayRetainedTransactionState::ReadyToStage;
            deferred_attempts_.fetch_add(1, std::memory_order_relaxed);
            recordPerfCounter("maintenance_deferred_attempts");
            state_.store(
                MoEOverlayMaintenanceState::Deferred,
                std::memory_order_release);
            return;
        default:
            retained_transaction_state_ =
                MoEOverlayRetainedTransactionState::Empty;
            fail(
                result.error.empty()
                    ? "ExpertOverlay active migration returned an invalid state"
                    : result.error);
            return;
        }
    }

    void MoEOverlayResidencyMaintenanceService::fail(
        std::string message) noexcept
    {
        bool expected = false;
        if (!failure_recorded_.compare_exchange_strong(
                expected,
                true,
                std::memory_order_acq_rel,
                std::memory_order_acquire))
        {
            return;
        }

        if (message.empty())
            message = "Unknown ExpertOverlay maintenance failure";

        /*
         * A fatal result is not backpressure. Once no physical wave is active,
         * the immutable proposal is only retry intent; keeping it live would
         * resubmit the same fatal operation on every worker poll. Clear that
         * intent before publishing unhealthy so observers can never race a
         * second submission. An active wave remains owned by the authority and
         * is still polled through its asynchronous abort/retirement edges.
         */
        if (retained_transaction_state_ !=
            MoEOverlayRetainedTransactionState::Active)
        {
            retained_transaction_ = {};
            retained_transaction_state_ =
                MoEOverlayRetainedTransactionState::Empty;
        }
        {
            std::lock_guard<std::mutex> lock(failure_mutex_);
            failure_message_ = message;
        }
        fatal_failures_.fetch_add(1, std::memory_order_relaxed);
        state_.store(
            MoEOverlayMaintenanceState::Failed,
            std::memory_order_release);

        /*
         * Publish health last. An observer that acquires `false` can therefore
         * immediately read the Failed state, counter, and complete diagnostic.
         */
        healthy_.store(false, std::memory_order_release);
        try
        {
            recordPerfCounter("maintenance_fatal_failures");
            LOG_ERROR("[MoEOverlayResidencyMaintenanceService] " << message);
        }
        catch (...)
        {
            /* Failure reporting must not terminate the resource-draining worker. */
        }
    }

    void MoEOverlayResidencyMaintenanceService::recordPerfCounter(
        const char *name,
        double value) const
    {
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            name,
            value,
            "maintenance",
            config_.perf_device,
            {{"policy", toString(config_.authority->residencyPolicy())}});
    }

} // namespace llaminar2
