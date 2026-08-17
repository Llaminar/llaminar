/**
 * @file MoEOverlayEconomyCalibrationController.cpp
 * @brief Event-polled live-inference and real-migration calibration protocol.
 */

#include "MoEOverlayEconomyCalibrationController.h"

#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <array>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /** @brief Stable PerfStats spelling for one production inference phase. */
        const char *calibrationSourceName(
            ExpertHistogramSource source) noexcept
        {
            switch (source)
            {
            case ExpertHistogramSource::DecodeToken:
                return "decode";
            case ExpertHistogramSource::PrefillChunk:
                return "prefill";
            case ExpertHistogramSource::GroupedVerifier:
                return "grouped_verifier";
            case ExpertHistogramSource::SyntheticTest:
                return "synthetic_test";
            }
            return "unknown";
        }

        /**
         * @brief Publish one identity-preserving probe lifecycle event.
         *
         * This is diagnostic state, not a scheduling command. Production
         * traffic remains authoritative: the probe is satisfied only when a
         * matching real prefill, decode, or grouped-verifier interval claims
         * it. Keeping the complete identity on both arm and sample events lets
         * operators distinguish an outstanding request from one the lock-free
         * probe already consumed without inspecting controller-private state.
         *
         * @param event_name Stable PerfStats lifecycle counter name.
         * @param perf_device Participant/device label owning calibration.
         * @param request Exact immutable probe request identity.
         */
        void recordProbeEvent(
            const char *event_name,
            const std::string &perf_device,
            const MoEOverlayInterferenceProbeRequest &request)
        {
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                event_name,
                1.0,
                "maintenance",
                perf_device,
                {{"calibration_sequence",
                  std::to_string(request.calibration_sequence)},
                 {"layer", std::to_string(request.coordinate.layer)},
                 {"mode",
                  request.mode ==
                          MoEOverlayInterferenceProbeMode::Baseline
                      ? "baseline"
                      : "concurrent_movement"},
                 {"source", calibrationSourceName(request.source)},
                 {"source_participant",
                  std::to_string(
                      request.coordinate.source_participant)},
                 {"destination_participant",
                  std::to_string(
                      request.coordinate.destination_participant)}});
        }

        /** @brief Publish one newly armed live-inference calibration probe. */
        void recordProbeArm(
            const std::string &perf_device,
            const MoEOverlayInterferenceProbeRequest &request)
        {
            recordProbeEvent(
                "economy_calibration_probe_arms",
                perf_device,
                request);
        }

        /** @brief Publish one production interval consumed by calibration. */
        void recordProbeSample(
            const std::string &perf_device,
            const MoEOverlayInterferenceProbeRequest &request)
        {
            recordProbeEvent(
                "economy_calibration_probe_samples",
                perf_device,
                request);
        }

        /**
         * @brief Publish one reason-coded, retryable overlap rejection.
         * @param perf_device Participant/device label owning calibration.
         * @param request Exact concurrent probe whose attempt was rejected.
         * @param reason Stable causal spelling for dashboards and tests.
         *
         * A rejection is expected under unlucky scheduling and is not a fatal
         * error. Publishing it is nevertheless essential: repeated retries
         * without an accepted pair otherwise look identical to useful
         * calibration progress from outside the maintenance thread.
         */
        void recordCalibrationRejection(
            const std::string &perf_device,
            const MoEOverlayInterferenceProbeRequest &request,
            const char *reason)
        {
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "economy_calibration_attempt_rejections",
                1.0,
                "maintenance",
                perf_device,
                {{"calibration_sequence",
                  std::to_string(request.calibration_sequence)},
                 {"layer", std::to_string(request.coordinate.layer)},
                 {"reason", reason},
                 {"source", calibrationSourceName(request.source)},
                 {"source_participant",
                  std::to_string(
                      request.coordinate.source_participant)},
                 {"destination_participant",
                  std::to_string(
                      request.coordinate.destination_participant)}});
        }
    } // namespace

    MoEOverlayEconomyCalibrationController::
        MoEOverlayEconomyCalibrationController(Config config)
        : config_(std::move(config))
    {
        if (!config_.planner || !config_.ledger || !config_.journal ||
            !config_.probe || !config_.transport || !config_.probe->idle())
        {
            throw std::invalid_argument(
                "ExpertOverlay economy calibration requires complete idle production dependencies");
        }
        if (config_.evidence_exchange &&
            (!config_.evidence_exchange->idle() ||
             config_.evidence_exchange->worldSize() < 2 ||
             config_.evidence_exchange->worldRank() < 0 ||
             config_.evidence_exchange->worldRank() >=
                 config_.evidence_exchange->worldSize()))
        {
            throw std::invalid_argument(
                "Distributed ExpertOverlay calibration requires one idle valid evidence lane");
        }
        if (config_.perf_device.empty())
            config_.perf_device = "expert_overlay";

        constexpr std::array<ExpertHistogramSource, 3> sources{
            ExpertHistogramSource::DecodeToken,
            ExpertHistogramSource::PrefillChunk,
            ExpertHistogramSource::GroupedVerifier,
        };
        const auto &required_sources = config_.ledger->requiredSources();
        for (std::size_t phase = 0; phase < sources.size(); ++phase)
        {
            if (required_sources[phase])
                phases_.push_back(sources[phase]);
        }
        if (phases_.empty())
        {
            throw std::invalid_argument(
                "ExpertOverlay calibration requires a runtime-reachable phase");
        }

        const auto &coordinates = config_.planner->requiredCoordinates();
        if (coordinates.empty() ||
            coordinates.size() != config_.ledger->coordinateCount())
        {
            throw std::invalid_argument(
                "ExpertOverlay calibration planner and ledger coordinates disagree");
        }
        for (const auto &coordinate : coordinates)
        {
            if (coordinate.source_participant >=
                coordinate.destination_participant)
            {
                continue;
            }
            const MoEOverlayMigrationMeasurementCoordinate reverse{
                .source_participant = coordinate.destination_participant,
                .destination_participant = coordinate.source_participant,
                .layer = coordinate.layer,
            };
            if (!std::binary_search(
                    coordinates.begin(), coordinates.end(), reverse))
            {
                throw std::invalid_argument(
                    "ExpertOverlay calibration is missing a reverse directed coordinate");
            }
            jobs_.push_back({
                .first_participant = coordinate.source_participant,
                .second_participant = coordinate.destination_participant,
                .layer = coordinate.layer,
            });
        }
        if (jobs_.empty())
        {
            throw std::invalid_argument(
                "ExpertOverlay calibration requires at least one unordered participant pair");
        }

        const std::uint64_t observations =
            config_.ledger->requiredObservationsPerCoordinate();
        const std::uint64_t jobs = static_cast<std::uint64_t>(jobs_.size());
        const std::uint64_t phases =
            static_cast<std::uint64_t>(phases_.size());
        const std::uint64_t maximum =
            std::numeric_limits<std::uint64_t>::max();
        if (observations == 0 || jobs > maximum / observations ||
            phases > maximum / (jobs * observations))
        {
            throw std::overflow_error(
                "ExpertOverlay calibration plan has an invalid accepted-pair cardinality");
        }
        const std::uint64_t expected_pairs =
            jobs * phases * observations;
        /*
         * Publish the exact finite corpus before the worker starts polling.
         * Tests and operators can derive progress bounds from production policy
         * instead of embedding topology-specific timeout/forward guesses.
         */
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "economy_calibration_expected_pairs",
            static_cast<double>(expected_pairs),
            "model_setup",
            config_.perf_device,
            {{"unordered_jobs", std::to_string(jobs)},
             {"runtime_phases", std::to_string(phases)},
             {"observations_per_coordinate",
              std::to_string(observations)}});
        local_rows_.reserve(2);
    }

    ExpertHistogramSource
    MoEOverlayEconomyCalibrationController::currentSource() const noexcept
    {
        return phases_[phase_index_];
    }

    MoEOverlayMigrationMeasurementCoordinate
    MoEOverlayEconomyCalibrationController::currentCoordinate() const noexcept
    {
        const auto &job = jobs_[job_index_];
        return {
            .source_participant = job.first_participant,
            .destination_participant = job.second_participant,
            .layer = job.layer,
        };
    }

    void MoEOverlayEconomyCalibrationController::poll() noexcept
    {
        polls_.fetch_add(1, std::memory_order_relaxed);
        if (!healthy() ||
            state() == MoEOverlayEconomyCalibrationState::Complete ||
            state() == MoEOverlayEconomyCalibrationState::Stopped)
            return;
        try
        {
            if (stop_requested_.load(std::memory_order_acquire))
            {
                progressStop();
                return;
            }
            switch (state())
            {
            case MoEOverlayEconomyCalibrationState::ArmBaseline:
                armBaseline();
                break;
            case MoEOverlayEconomyCalibrationState::AwaitBaseline:
                awaitBaseline();
                break;
            case MoEOverlayEconomyCalibrationState::StartWave:
                startWave();
                break;
            case MoEOverlayEconomyCalibrationState::AwaitConcurrentInference:
                awaitConcurrentInference();
                break;
            case MoEOverlayEconomyCalibrationState::AwaitConcurrentWave:
                awaitConcurrentWave();
                break;
            case MoEOverlayEconomyCalibrationState::AbortCleanup:
                pollAbortCleanup();
                break;
            case MoEOverlayEconomyCalibrationState::AwaitLateConcurrentSample:
                awaitLateConcurrentSample();
                break;
            case MoEOverlayEconomyCalibrationState::AwaitEvidenceExchange:
                pollEvidenceExchange();
                break;
            case MoEOverlayEconomyCalibrationState::Complete:
            case MoEOverlayEconomyCalibrationState::Failed:
            case MoEOverlayEconomyCalibrationState::Stopped:
                break;
            }
        }
        catch (const std::exception &error)
        {
            if (active_wave_)
                beginAbort(error.what());
            else
                deferFailureUntilProbeQuiescent(error.what());
        }
        catch (...)
        {
            if (active_wave_)
            {
                beginAbort(
                    "ExpertOverlay calibration raised a non-standard exception");
            }
            else
            {
                deferFailureUntilProbeQuiescent(
                    "ExpertOverlay calibration raised a non-standard exception");
            }
        }
    }

    void MoEOverlayEconomyCalibrationController::requestStop() noexcept
    {
        stop_requested_.store(true, std::memory_order_release);
    }

    void MoEOverlayEconomyCalibrationController::armBaseline()
    {
        if (!config_.probe->idle() || active_wave_ || baseline_sample_ ||
            concurrent_sample_)
        {
            throw std::logic_error(
                "ExpertOverlay calibration baseline arm found live attempt state");
        }
        ++calibration_sequence_;
        if (calibration_sequence_ == 0)
            throw std::overflow_error("ExpertOverlay calibration sequence overflowed");
        const MoEOverlayInterferenceProbeRequest request{
                .coordinate = currentCoordinate(),
                .source = currentSource(),
                .mode = MoEOverlayInterferenceProbeMode::Baseline,
                .calibration_sequence = calibration_sequence_,
            };
        if (!config_.probe->arm(request))
        {
            throw std::logic_error(
                "ExpertOverlay calibration could not arm its baseline probe");
        }
        recordProbeArm(config_.perf_device, request);
        baseline_arms_.fetch_add(1, std::memory_order_relaxed);
        state_.store(
            MoEOverlayEconomyCalibrationState::AwaitBaseline,
            std::memory_order_release);
    }

    void MoEOverlayEconomyCalibrationController::awaitBaseline()
    {
        MoEOverlayInterferenceProbeSample sample;
        if (!config_.probe->consume(&sample))
            return;
        if (!sample.valid() ||
            sample.request.mode != MoEOverlayInterferenceProbeMode::Baseline ||
            sample.request.coordinate != currentCoordinate() ||
            sample.request.source != currentSource() ||
            sample.request.calibration_sequence != calibration_sequence_)
        {
            throw std::logic_error(
                "ExpertOverlay calibration consumed a mismatched baseline sample");
        }
        recordProbeSample(config_.perf_device, sample.request);
        baseline_sample_ = sample;
        baseline_samples_.fetch_add(1, std::memory_order_relaxed);
        state_.store(
            MoEOverlayEconomyCalibrationState::StartWave,
            std::memory_order_release);
    }

    void MoEOverlayEconomyCalibrationController::startWave()
    {
        if (!baseline_sample_ || active_wave_ || !config_.probe->idle())
        {
            throw std::logic_error(
                "ExpertOverlay calibration cannot start a wave without one idle paired baseline");
        }
        const auto &job = jobs_[job_index_];
        const auto transaction = config_.planner->buildPairSwap(
            job.first_participant,
            job.second_participant,
            job.layer,
            calibration_sequence_);
        wave_start_attempts_.fetch_add(1, std::memory_order_relaxed);
        auto started = config_.transport->beginStage(transaction);
        if (!started.valid())
        {
            throw std::logic_error(
                "ExpertOverlay calibration transport returned invalid start ownership");
        }
        if (started.status == MoEOverlayResidencyStageStartStatus::Deferred)
        {
            waves_deferred_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (started.status == MoEOverlayResidencyStageStartStatus::Failed)
        {
            active_wave_ = std::move(started.cleanup_wave);
            const std::string message = started.error.empty()
                                            ? "ExpertOverlay calibration transport failed to start"
                                            : std::move(started.error);
            if (active_wave_)
                beginAbort(message);
            else
                fail(message);
            return;
        }

        active_wave_ = std::move(started.wave);
        waves_started_.fetch_add(1, std::memory_order_relaxed);
        const auto concurrent_request = currentConcurrentRequest();
        if (!config_.probe->arm(concurrent_request))
        {
            beginAbort(
                "ExpertOverlay calibration could not arm its concurrent probe");
            return;
        }
        recordProbeArm(config_.perf_device, concurrent_request);
        concurrent_arms_.fetch_add(1, std::memory_order_relaxed);
        state_.store(
            MoEOverlayEconomyCalibrationState::AwaitConcurrentInference,
            std::memory_order_release);
    }

    MoEOverlayInterferenceProbeRequest
    MoEOverlayEconomyCalibrationController::currentConcurrentRequest() const
    {
        if (!baseline_sample_)
        {
            throw std::logic_error(
                "ExpertOverlay calibration has no baseline workload for its concurrent request");
        }
        return {
            .coordinate = currentCoordinate(),
            .source = currentSource(),
            .require_exact_workload = true,
            .required_workload = baseline_sample_->workload,
            .mode = MoEOverlayInterferenceProbeMode::ConcurrentMovement,
            .calibration_sequence = calibration_sequence_,
        };
    }

    void MoEOverlayEconomyCalibrationController::awaitConcurrentInference()
    {
        if (!active_wave_ || !baseline_sample_ || stage_dispatch_started_)
        {
            throw std::logic_error(
                "ExpertOverlay calibration launch handshake found incomplete wave ownership");
        }

        switch (config_.probe->progress(currentConcurrentRequest()))
        {
        case MoEOverlayInterferenceProbeProgress::Armed:
            concurrent_launch_wait_polls_.fetch_add(
                1, std::memory_order_relaxed);
            return;
        case MoEOverlayInterferenceProbeProgress::Running:
            /*
             * Only this edge releases reservation consensus and queued transfer
             * operations.  The inference caller never polls transport work and
             * the maintenance worker never waits for the caller.
             */
            stage_dispatch_started_ = true;
            concurrent_launches_during_inference_.fetch_add(
                1, std::memory_order_relaxed);
            state_.store(
                MoEOverlayEconomyCalibrationState::AwaitConcurrentWave,
                std::memory_order_release);
            awaitConcurrentWave();
            return;
        case MoEOverlayInterferenceProbeProgress::Completed:
            /* The exact workload ended before maintenance could release bytes. */
            (void)tryConsumeConcurrentSample();
            concurrent_launch_misses_.fetch_add(1, std::memory_order_relaxed);
            partial_overlap_rejections_.fetch_add(
                1, std::memory_order_relaxed);
            recordCalibrationRejection(
                config_.perf_device,
                currentConcurrentRequest(),
                "inference_completed_before_dispatch");
            beginAbort();
            return;
        case MoEOverlayInterferenceProbeProgress::Missing:
            throw std::logic_error(
                "ExpertOverlay calibration lost its armed concurrent inference request");
        }
        throw std::logic_error(
            "ExpertOverlay calibration observed an unknown probe lifecycle");
    }

    bool MoEOverlayEconomyCalibrationController::
        tryConsumeConcurrentSample()
    {
        if (concurrent_sample_)
            return true;
        MoEOverlayInterferenceProbeSample sample;
        if (!config_.probe->consume(&sample))
            return false;
        if (!sample.valid() ||
            sample.request.mode !=
                MoEOverlayInterferenceProbeMode::ConcurrentMovement ||
            sample.request.coordinate != currentCoordinate() ||
            sample.request.source != currentSource() ||
            sample.request.calibration_sequence != calibration_sequence_ ||
            !baseline_sample_ ||
            sample.workload != baseline_sample_->workload)
        {
            workload_pair_rejections_.fetch_add(1, std::memory_order_relaxed);
            throw std::logic_error(
                "ExpertOverlay calibration consumed a mismatched concurrent sample");
        }
        recordProbeSample(config_.perf_device, sample.request);
        concurrent_sample_ = sample;
        concurrent_samples_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    void MoEOverlayEconomyCalibrationController::awaitConcurrentWave()
    {
        if (!stage_dispatch_started_)
        {
            throw std::logic_error(
                "ExpertOverlay calibration polled staging before inference-authorized dispatch");
        }
        (void)tryConsumeConcurrentSample();
        std::string error;
        const auto progress = active_wave_->pollStage(&error);
        if (progress == MoEOverlayResidencyWaveProgress::Pending)
            return;
        if (progress == MoEOverlayResidencyWaveProgress::Deferred)
        {
            beginAbort();
            return;
        }
        if (progress == MoEOverlayResidencyWaveProgress::Failed)
        {
            beginAbort(
                error.empty()
                    ? "ExpertOverlay calibration staging failed"
                    : std::move(error));
            return;
        }

        stage_completed_ = true;
        stage_interval_ = active_wave_->completedStageInterval();
        if (!stage_interval_ || !stage_interval_->valid())
        {
            beginAbort(
                "ExpertOverlay calibration wave omitted its exact staging interval");
            return;
        }
        /*
         * If no inference call claimed the request before physical readiness,
         * cancel it now. A running call cannot be cancelled; its eventual end
         * necessarily falls outside the frozen stage interval and is rejected.
         */
        if (!concurrent_sample_)
            (void)config_.probe->cancelArmed();
        beginAbort();
    }

    void MoEOverlayEconomyCalibrationController::beginAbort(
        std::string failure_after_cleanup)
    {
        if (!active_wave_)
        {
            if (!failure_after_cleanup.empty())
                deferFailureUntilProbeQuiescent(
                    std::move(failure_after_cleanup));
            else
                retryAttempt();
            return;
        }
        if (!failure_after_cleanup.empty())
            failure_after_cleanup_ = std::move(failure_after_cleanup);
        active_wave_->abortStaged();
        waves_aborted_.fetch_add(1, std::memory_order_relaxed);
        state_.store(
            MoEOverlayEconomyCalibrationState::AbortCleanup,
            std::memory_order_release);
    }

    void MoEOverlayEconomyCalibrationController::pollAbortCleanup()
    {
        (void)tryConsumeConcurrentSample();
        std::string error;
        const auto progress = active_wave_->pollAbort(&error);
        if (progress == MoEOverlayResidencyWaveProgress::Pending)
            return;
        if (progress != MoEOverlayResidencyWaveProgress::Ready)
        {
            fail(
                error.empty()
                    ? "ExpertOverlay calibration abort cleanup failed"
                    : std::move(error));
            return;
        }
        active_wave_.reset();
        if (!failure_after_cleanup_.empty())
        {
            if (config_.probe->discardAvailable())
            {
                auto message = std::move(failure_after_cleanup_);
                failure_after_cleanup_.clear();
                fail(std::move(message));
            }
            else
            {
                state_.store(
                    MoEOverlayEconomyCalibrationState::
                        AwaitLateConcurrentSample,
                    std::memory_order_release);
            }
            return;
        }
        if (stop_requested_.load(std::memory_order_acquire))
        {
            progressStop();
            return;
        }
        if (!stage_completed_)
        {
            retryAttempt();
            return;
        }

        std::string journal_error;
        if (!config_.journal->consume(&local_rows_, &journal_error))
        {
            fail(
                journal_error.empty()
                    ? "ExpertOverlay calibration stage produced no measurement journal row"
                    : std::move(journal_error));
            return;
        }
        if (tryConsumeConcurrentSample())
        {
            finalizeAttempt();
            return;
        }
        if (config_.probe->idle())
        {
            if (config_.evidence_exchange)
            {
                finalizeAttempt();
                return;
            }
            partial_overlap_rejections_.fetch_add(
                1, std::memory_order_relaxed);
            recordCalibrationRejection(
                config_.perf_device,
                currentConcurrentRequest(),
                "sample_missing_after_stage");
            retryAttempt();
            return;
        }
        state_.store(
            MoEOverlayEconomyCalibrationState::AwaitLateConcurrentSample,
            std::memory_order_release);
    }

    void MoEOverlayEconomyCalibrationController::
        awaitLateConcurrentSample()
    {
        if (!failure_after_cleanup_.empty())
        {
            if (!config_.probe->discardAvailable())
                return;
            auto message = std::move(failure_after_cleanup_);
            failure_after_cleanup_.clear();
            fail(std::move(message));
            return;
        }
        if (stop_requested_.load(std::memory_order_acquire))
        {
            progressStop();
            return;
        }
        if (!tryConsumeConcurrentSample())
            return;
        finalizeAttempt();
    }

    void MoEOverlayEconomyCalibrationController::finalizeAttempt()
    {
        if (!baseline_sample_ || !stage_interval_ || local_rows_.empty())
        {
            throw std::logic_error(
                "ExpertOverlay calibration attempted to finalize incomplete evidence");
        }
        const bool exact_overlap =
            concurrent_sample_ &&
            concurrent_sample_->whollyContains(
                stage_interval_->begin_steady_nanoseconds,
                stage_interval_->end_steady_nanoseconds);
        MoEOverlayCalibrationAttemptEvidence evidence{
            .calibration_sequence = calibration_sequence_,
            .coordinate = currentCoordinate(),
            .source = currentSource(),
            .workload = baseline_sample_->workload,
            .baseline_nanoseconds = baseline_sample_->durationNanoseconds(),
            .concurrent_nanoseconds = concurrent_sample_
                                          ? concurrent_sample_
                                                ->durationNanoseconds()
                                          : 0,
            .exact_overlap = exact_overlap,
            .local_measurements = local_rows_,
        };
        if (!evidence.valid())
        {
            throw std::logic_error(
                "ExpertOverlay calibration constructed invalid attempt evidence");
        }

        if (config_.evidence_exchange)
        {
            std::string exchange_error;
            if (!config_.evidence_exchange->beginAttempt(
                    evidence, &exchange_error))
            {
                throw std::logic_error(
                    exchange_error.empty()
                        ? "ExpertOverlay calibration failed to begin its all-rank evidence exchange"
                        : std::move(exchange_error));
            }
            pending_attempt_evidence_ = std::move(evidence);
            state_.store(
                MoEOverlayEconomyCalibrationState::AwaitEvidenceExchange,
                std::memory_order_release);
            return;
        }

        if (!exact_overlap)
        {
            partial_overlap_rejections_.fetch_add(
                1, std::memory_order_relaxed);
            recordCalibrationRejection(
                config_.perf_device,
                currentConcurrentRequest(),
                "local_interval_not_contained");
            retryAttempt();
            return;
        }
        recordAcceptedAttempt(
            MoEOverlayMigrationMeasurementMerger::merge({local_rows_}),
            evidence.baseline_nanoseconds,
            evidence.concurrent_nanoseconds);
    }

    void MoEOverlayEconomyCalibrationController::pollEvidenceExchange()
    {
        if (!config_.evidence_exchange || !pending_attempt_evidence_)
        {
            throw std::logic_error(
                "ExpertOverlay calibration has no active evidence exchange");
        }
        MoEOverlayCalibrationAttemptResult result;
        std::string error;
        const auto progress = config_.evidence_exchange->pollAttempt(
            &result, &error);
        if (progress == MoEOverlayResidencyWaveProgress::Pending)
            return;
        if (progress != MoEOverlayResidencyWaveProgress::Ready)
        {
            throw std::runtime_error(
                error.empty()
                    ? "ExpertOverlay calibration evidence exchange failed"
                    : std::move(error));
        }
        if (!result.valid())
        {
            throw std::logic_error(
                "ExpertOverlay calibration exchange returned an invalid result");
        }
        pending_attempt_evidence_.reset();
        if (!result.accepted)
        {
            partial_overlap_rejections_.fetch_add(
                1, std::memory_order_relaxed);
            recordCalibrationRejection(
                config_.perf_device,
                currentConcurrentRequest(),
                "distributed_overlap_vote_rejected");
            retryAttempt();
            return;
        }
        recordAcceptedAttempt(
            result.measurements,
            result.baseline_nanoseconds,
            result.concurrent_nanoseconds);
    }

    void MoEOverlayEconomyCalibrationController::recordAcceptedAttempt(
        const std::vector<MoEOverlayCompletedMigrationMeasurement> &rows,
        std::uint64_t baseline_nanoseconds,
        std::uint64_t concurrent_nanoseconds)
    {
        std::string ledger_error;
        if (!config_.ledger->recordCompletedWave(rows, &ledger_error))
        {
            throw std::logic_error(
                ledger_error.empty()
                    ? "ExpertOverlay calibration ledger rejected migration evidence"
                    : std::move(ledger_error));
        }
        const auto &job = jobs_[job_index_];
        for (const auto coordinate : {
                 MoEOverlayMigrationMeasurementCoordinate{
                     .source_participant = job.first_participant,
                     .destination_participant = job.second_participant,
                     .layer = job.layer,
                 },
                 MoEOverlayMigrationMeasurementCoordinate{
                     .source_participant = job.second_participant,
                     .destination_participant = job.first_participant,
                     .layer = job.layer,
                 }})
        {
            if (!config_.ledger->recordInterferenceSample(
                    coordinate,
                    currentSource(),
                    baseline_nanoseconds,
                    concurrent_nanoseconds,
                    &ledger_error))
            {
                throw std::logic_error(
                    ledger_error.empty()
                        ? "ExpertOverlay calibration ledger rejected interference evidence"
                        : std::move(ledger_error));
            }
        }
        exact_overlap_samples_.fetch_add(1, std::memory_order_relaxed);
        accepted_pairs_.fetch_add(1, std::memory_order_relaxed);
        const auto tags = PerfStatsCollector::Tags{
            {"calibration_sequence",
             std::to_string(calibration_sequence_)},
            {"layer", std::to_string(job.layer)},
            {"source", calibrationSourceName(currentSource())},
            {"source_participant",
             std::to_string(job.first_participant)},
            {"destination_participant",
             std::to_string(job.second_participant)},
            {"observation", std::to_string(sample_index_)},
        };
        const auto add_timing = [&]
            (const char *name, std::uint64_t nanoseconds)
        {
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                name,
                static_cast<double>(nanoseconds),
                "maintenance",
                config_.perf_device,
                tags);
        };
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "economy_calibration_pairs_accepted",
            1.0,
            "maintenance",
            config_.perf_device,
            tags);

        /*
         * Publish the exact accepted pair rather than only its final robust
         * median.  The maximum merged row is the distributed pair-swap
         * critical path; inference delta remains separate so operators can
         * distinguish movement latency from physical resource contention.
         */
        const auto slowest_row = std::max_element(
            rows.begin(),
            rows.end(),
            [](const auto &left, const auto &right)
            {
                return left.wave_wall_nanoseconds <
                       right.wave_wall_nanoseconds;
            });
        add_timing(
            "economy_calibration_wave_wall_ns",
            slowest_row->wave_wall_nanoseconds);
        add_timing(
            "economy_calibration_inference_baseline_ns",
            baseline_nanoseconds);
        add_timing(
            "economy_calibration_inference_concurrent_ns",
            concurrent_nanoseconds);
        add_timing(
            "economy_calibration_inference_interference_ns",
            concurrent_nanoseconds > baseline_nanoseconds
                ? concurrent_nanoseconds - baseline_nanoseconds
                : 0u);
        advanceAcceptedPair();
    }

    void MoEOverlayEconomyCalibrationController::retryAttempt()
    {
        baseline_sample_.reset();
        concurrent_sample_.reset();
        stage_interval_.reset();
        local_rows_.clear();
        pending_attempt_evidence_.reset();
        stage_completed_ = false;
        stage_dispatch_started_ = false;
        failure_after_cleanup_.clear();
        state_.store(
            MoEOverlayEconomyCalibrationState::ArmBaseline,
            std::memory_order_release);
    }

    void MoEOverlayEconomyCalibrationController::advanceAcceptedPair()
    {
        baseline_sample_.reset();
        concurrent_sample_.reset();
        stage_interval_.reset();
        local_rows_.clear();
        pending_attempt_evidence_.reset();
        stage_completed_ = false;
        stage_dispatch_started_ = false;

        ++sample_index_;
        if (sample_index_ >=
            config_.ledger->requiredObservationsPerCoordinate())
        {
            sample_index_ = 0;
            ++phase_index_;
        }
        if (phase_index_ >= phases_.size())
        {
            phase_index_ = 0;
            ++job_index_;
        }
        if (job_index_ >= jobs_.size())
        {
            if (!config_.ledger->ready())
            {
                throw std::logic_error(
                    "ExpertOverlay calibration exhausted jobs before its ledger became ready");
            }
            sealed_ = config_.ledger->seal();
            state_.store(
                MoEOverlayEconomyCalibrationState::Complete,
                std::memory_order_release);
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "economy_calibration_complete",
                1.0,
                "maintenance",
                config_.perf_device,
                {{"blocking", "false"}, {"movement", "real"}});
            return;
        }
        state_.store(
            MoEOverlayEconomyCalibrationState::ArmBaseline,
            std::memory_order_release);
    }

    void MoEOverlayEconomyCalibrationController::fail(
        std::string message) noexcept
    {
        if (!healthy_.exchange(false, std::memory_order_acq_rel))
            return;
        if (message.empty())
            message = "Unknown ExpertOverlay economy calibration failure";
        {
            std::lock_guard<std::mutex> lock(failure_mutex_);
            failure_message_ = std::move(message);
        }
        fatal_failures_.fetch_add(1, std::memory_order_relaxed);
        state_.store(
            MoEOverlayEconomyCalibrationState::Failed,
            std::memory_order_release);
    }

    void MoEOverlayEconomyCalibrationController::
        deferFailureUntilProbeQuiescent(std::string message) noexcept
    {
        if (message.empty())
            message = "Unknown ExpertOverlay economy calibration failure";
        if (!failure_after_cleanup_.empty())
        {
            /* The first causal diagnostic remains authoritative through cleanup. */
            if (config_.probe->discardAvailable())
            {
                auto retained = std::move(failure_after_cleanup_);
                failure_after_cleanup_.clear();
                fail(std::move(retained));
                return;
            }
            state_.store(
                MoEOverlayEconomyCalibrationState::AwaitLateConcurrentSample,
                std::memory_order_release);
            return;
        }
        if (config_.probe->discardAvailable())
        {
            fail(std::move(message));
            return;
        }
        failure_after_cleanup_ = std::move(message);
        state_.store(
            MoEOverlayEconomyCalibrationState::AwaitLateConcurrentSample,
            std::memory_order_release);
    }

    void MoEOverlayEconomyCalibrationController::progressStop()
    {
        if (config_.evidence_exchange &&
            !config_.evidence_exchange->idle())
        {
            MoEOverlayCalibrationAttemptResult discarded;
            std::string error;
            const auto progress = config_.evidence_exchange->pollAttempt(
                &discarded, &error);
            if (progress == MoEOverlayResidencyWaveProgress::Pending)
                return;
            if (progress != MoEOverlayResidencyWaveProgress::Ready)
            {
                fail(
                    error.empty()
                        ? "ExpertOverlay calibration could not drain its evidence exchange"
                        : std::move(error));
                return;
            }
            pending_attempt_evidence_.reset();
        }
        if (active_wave_)
        {
            if (state() != MoEOverlayEconomyCalibrationState::AbortCleanup)
            {
                (void)config_.probe->discardAvailable();
                active_wave_->abortStaged();
                waves_aborted_.fetch_add(1, std::memory_order_relaxed);
                state_.store(
                    MoEOverlayEconomyCalibrationState::AbortCleanup,
                    std::memory_order_release);
                return;
            }
            pollAbortCleanup();
            return;
        }

        if (!config_.probe->discardAvailable())
        {
            state_.store(
                MoEOverlayEconomyCalibrationState::
                    AwaitLateConcurrentSample,
                std::memory_order_release);
            return;
        }
        baseline_sample_.reset();
        concurrent_sample_.reset();
        stage_interval_.reset();
        local_rows_.clear();
        stage_completed_ = false;
        stage_dispatch_started_ = false;
        state_.store(
            MoEOverlayEconomyCalibrationState::Stopped,
            std::memory_order_release);
    }

    MoEOverlayEconomyCalibrationState
    MoEOverlayEconomyCalibrationController::state() const noexcept
    {
        return state_.load(std::memory_order_acquire);
    }

    bool MoEOverlayEconomyCalibrationController::healthy() const noexcept
    {
        return healthy_.load(std::memory_order_acquire);
    }

    std::string
    MoEOverlayEconomyCalibrationController::failureMessage() const
    {
        std::lock_guard<std::mutex> lock(failure_mutex_);
        return failure_message_;
    }

    MoEOverlayEconomyCalibrationStats
    MoEOverlayEconomyCalibrationController::stats() const noexcept
    {
        return {
            .polls = polls_.load(std::memory_order_relaxed),
            .baseline_arms = baseline_arms_.load(std::memory_order_relaxed),
            .baseline_samples = baseline_samples_.load(std::memory_order_relaxed),
            .wave_start_attempts = wave_start_attempts_.load(std::memory_order_relaxed),
            .waves_started = waves_started_.load(std::memory_order_relaxed),
            .waves_deferred = waves_deferred_.load(std::memory_order_relaxed),
            .concurrent_arms = concurrent_arms_.load(std::memory_order_relaxed),
            .concurrent_launch_wait_polls =
                concurrent_launch_wait_polls_.load(std::memory_order_relaxed),
            .concurrent_launches_during_inference =
                concurrent_launches_during_inference_.load(
                    std::memory_order_relaxed),
            .concurrent_launch_misses =
                concurrent_launch_misses_.load(std::memory_order_relaxed),
            .concurrent_samples = concurrent_samples_.load(std::memory_order_relaxed),
            .exact_overlap_samples = exact_overlap_samples_.load(std::memory_order_relaxed),
            .partial_overlap_rejections = partial_overlap_rejections_.load(std::memory_order_relaxed),
            .workload_pair_rejections = workload_pair_rejections_.load(std::memory_order_relaxed),
            .waves_aborted = waves_aborted_.load(std::memory_order_relaxed),
            .accepted_pairs = accepted_pairs_.load(std::memory_order_relaxed),
            .fatal_failures = fatal_failures_.load(std::memory_order_relaxed),
        };
    }

    const MoEOverlaySealedMigrationMeasurements *
    MoEOverlayEconomyCalibrationController::sealedMeasurements()
        const noexcept
    {
        /* Complete is release-published only after `sealed_` is initialized. */
        if (state_.load(std::memory_order_acquire) !=
            MoEOverlayEconomyCalibrationState::Complete)
        {
            return nullptr;
        }
        return sealed_ ? &*sealed_ : nullptr;
    }
} // namespace llaminar2
