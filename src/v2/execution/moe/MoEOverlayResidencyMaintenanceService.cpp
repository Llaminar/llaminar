/**
 * @file MoEOverlayResidencyMaintenanceService.cpp
 * @brief Event-polled background scheduling for ExpertOverlay residency waves.
 */

#include "MoEOverlayResidencyMaintenanceService.h"

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
        if (config_.histogram_publisher &&
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
        histogram_exchange_active_ =
            config_.histogram_publisher &&
            !config_.histogram_publisher->isCoordinator();

    }

    MoEOverlayResidencyMaintenanceService::
        ~MoEOverlayResidencyMaintenanceService()
    {
        try
        {
            stopAndDrain();
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
        notifications_.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(wake_mutex_);
            wake_requested_ = true;
        }
        wake_cv_.notify_one();
    }

    void MoEOverlayResidencyMaintenanceService::stopAndDrain()
    {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
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
            .histogram_windows_published =
                histogram_windows_published_.load(
                    std::memory_order_relaxed),
            .histogram_windows_received =
                histogram_windows_received_.load(
                    std::memory_order_relaxed),
            .histogram_receives_rearmed =
                histogram_receives_rearmed_.load(
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

    void MoEOverlayResidencyMaintenanceService::run(
        std::stop_token stop_token) noexcept
    {
        worker_starts_.fetch_add(1, std::memory_order_relaxed);
        recordPerfCounter("maintenance_worker_starts");
        state_.store(
            MoEOverlayMaintenanceState::Waiting,
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
                if (!config_.histogram_publisher)
                {
                    retained_transaction_ = {};
                    has_retained_transaction_ = false;
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
                !config_.histogram_publisher ||
                !config_.histogram_publisher->isCoordinator() ||
                (!histogram_exchange_active_ &&
                 !publishing_histogram_window_);
            if (stopping && !active_wave_ &&
                !has_retained_transaction_ &&
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

        if (active_wave_)
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

        if (has_retained_transaction_)
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
            if (config_.histogram_publisher &&
                config_.histogram_publisher->isCoordinator() &&
                histogram_exchange_active_)
            {
                progressDistributedProposal();
            }
            return;
        }

        if (config_.histogram_publisher)
        {
            progressDistributedProposal();
            return;
        }

        const bool dynamic = config_.authority->migrationEnabled();
        std::shared_ptr<const DecodeExpertHistogramWindow> local_window;
        if (dynamic)
        {
            const auto window_result =
                config_.authority->progressHistogramWindow();
            if (window_result.progress ==
                MoEOverlayHistogramWindowProgress::Waiting)
            {
                state_.store(
                    MoEOverlayMaintenanceState::Waiting,
                    std::memory_order_release);
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

        retained_transaction_ = dynamic
                                    ? config_.authority
                                          ->proposeFromFrozenHistogramWindow(
                                              std::move(local_window))
                                    : config_.authority->proposeFromHistogram();
        has_retained_transaction_ = true;
        proposals_.fetch_add(1, std::memory_order_relaxed);
        recordPerfCounter("maintenance_proposals");

        /* Do not launch new process-local work after a concurrent shutdown. */
        if (shutdown_requested_.load(std::memory_order_acquire))
        {
            retained_transaction_ = {};
            has_retained_transaction_ = false;
            return;
        }
        tryBeginRetainedTransaction();
    }

    void MoEOverlayResidencyMaintenanceService::progressDistributedProposal()
    {
        auto &publisher = *config_.histogram_publisher;
        if (publisher.isCoordinator() && !histogram_exchange_active_)
        {
            const auto window_result =
                config_.authority->progressHistogramWindow();
            if (window_result.progress ==
                MoEOverlayHistogramWindowProgress::Waiting)
            {
                state_.store(
                    MoEOverlayMaintenanceState::Waiting,
                    std::memory_order_release);
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

            publishing_histogram_window_ = window_result.window;
            if (!publishing_histogram_window_ ||
                !publishing_histogram_window_->valid())
            {
                fail(
                    "ExpertOverlay coordinator froze an invalid histogram window");
                return;
            }
            std::string error;
            if (!publisher.beginPublish(
                    *publishing_histogram_window_, &error))
            {
                fail(
                    error.empty()
                        ? "ExpertOverlay coordinator failed to begin histogram publication"
                        : std::move(error));
                return;
            }
            histogram_exchange_active_ = true;
            state_.store(
                MoEOverlayMaintenanceState::PublishingEvidence,
                std::memory_order_release);
            recordPerfCounter("maintenance_histogram_publications_started");
            return;
        }

        if (!histogram_exchange_active_)
        {
            fail(
                "ExpertOverlay peer has no armed authoritative histogram receive");
            return;
        }

        std::shared_ptr<const DecodeExpertHistogramWindow> received_window;
        std::string error;
        const auto progress = publisher.poll(&received_window, &error);
        if (progress == MoEOverlayResidencyWaveProgress::Pending)
        {
            state_.store(
                publisher.isCoordinator()
                    ? MoEOverlayMaintenanceState::PublishingEvidence
                    : MoEOverlayMaintenanceState::ReceivingEvidence,
                std::memory_order_release);
            return;
        }
        if (progress != MoEOverlayResidencyWaveProgress::Ready)
        {
            fail(
                error.empty()
                    ? "ExpertOverlay authoritative histogram publication failed"
                    : std::move(error));
            return;
        }

        if (publisher.isCoordinator())
        {
            histogram_exchange_active_ = false;
            histogram_windows_published_.fetch_add(
                1,
                std::memory_order_relaxed);
            recordPerfCounter("maintenance_histogram_windows_published");
            auto window = std::move(publishing_histogram_window_);
            if (!window || received_window)
            {
                fail(
                    "ExpertOverlay coordinator publication returned invalid window ownership");
                return;
            }
            retainDistributedProposal(std::move(window));
            return;
        }

        if (!received_window || !received_window->valid())
        {
            fail(
                "ExpertOverlay peer received no valid authoritative histogram window");
            return;
        }

        /*
         * Decoding copied the packet into immutable transaction-owned storage,
         * so the fixed MPI wire buffer can immediately receive generation N+1.
         * This keeps the coordinator independent of peer maintenance latency.
         */
        if (!publisher.armReceive(&error))
        {
            fail(
                error.empty()
                    ? "ExpertOverlay peer failed to arm its next histogram receive"
                    : std::move(error));
            return;
        }
        histogram_exchange_active_ = true;
        histogram_windows_received_.fetch_add(1, std::memory_order_relaxed);
        histogram_receives_rearmed_.fetch_add(1, std::memory_order_relaxed);
        recordPerfCounter("maintenance_histogram_windows_received");
        recordPerfCounter("maintenance_histogram_receives_rearmed");
        retainDistributedProposal(std::move(received_window));
    }

    void MoEOverlayResidencyMaintenanceService::retainDistributedProposal(
        std::shared_ptr<const DecodeExpertHistogramWindow> window)
    {
        if (!window || !window->valid() || has_retained_transaction_ ||
            active_wave_)
        {
            fail(
                "ExpertOverlay maintenance cannot retain the published histogram in its current state");
            return;
        }

        retained_transaction_ =
            config_.authority->proposeFromFrozenHistogramWindow(
                std::move(window));
        has_retained_transaction_ = true;
        proposals_.fetch_add(1, std::memory_order_relaxed);
        recordPerfCounter("maintenance_distributed_proposals");
        LOG_INFO(
            "[ExpertOverlay][Residency] Derived distributed proposal"
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

        /*
         * Histogram acknowledgement is the distributed irrevocability edge.
         * Even if shutdown raced this local callback, every rank must now
         * derive and finish the same transaction before resources are freed.
         */
        tryBeginRetainedTransaction();
    }

    void MoEOverlayResidencyMaintenanceService::
        tryBeginRetainedTransaction()
    {
        if (!has_retained_transaction_)
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
            active_wave_ = true;
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
            has_retained_transaction_ = false;
            state_.store(
                MoEOverlayMaintenanceState::Waiting,
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
            has_retained_transaction_ = false;
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
            active_wave_ = false;
            retained_transaction_ = {};
            has_retained_transaction_ = false;
            committed_waves_.fetch_add(1, std::memory_order_relaxed);
            recordPerfCounter("maintenance_waves_committed");
            state_.store(
                MoEOverlayMaintenanceState::Waiting,
                std::memory_order_release);
            return;
        case MoEOverlayResidencyApplyStatus::Deferred:
            active_wave_ = false;
            deferred_attempts_.fetch_add(1, std::memory_order_relaxed);
            recordPerfCounter("maintenance_deferred_attempts");
            state_.store(
                MoEOverlayMaintenanceState::Deferred,
                std::memory_order_release);
            return;
        default:
            active_wave_ = false;
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
        if (!active_wave_)
        {
            retained_transaction_ = {};
            has_retained_transaction_ = false;
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
