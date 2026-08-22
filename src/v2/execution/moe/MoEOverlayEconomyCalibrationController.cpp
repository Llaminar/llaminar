/**
 * @file MoEOverlayEconomyCalibrationController.cpp
 * @brief Bounded production-lane transfer profiling without synthetic inference.
 */

#include "MoEOverlayEconomyCalibrationController.h"

#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /** @brief Stable counter helper for rare setup/profile lifecycle edges. */
        void recordProfileCounter(
            const std::string &device,
            const char *name,
            double value,
            PerfStatsCollector::Tags tags = {})
        {
            tags.emplace("synthetic_inference", "false");
            tags.emplace("publish_residency", "false");
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                name,
                value,
                "model_setup",
                device,
                std::move(tags));
        }
    } // namespace

    MoEOverlayEconomyCalibrationController::
        MoEOverlayEconomyCalibrationController(Config config)
        : config_(std::move(config)),
          profiling_started_at_(std::chrono::steady_clock::now())
    {
        if (!config_.planner || !config_.ledger || !config_.journal ||
            !config_.transport)
        {
            throw std::invalid_argument(
                "ExpertOverlay transfer profiling requires complete production dependencies");
        }
        if (config_.evidence_exchange &&
            (!config_.evidence_exchange->idle() ||
             config_.evidence_exchange->worldSize() < 2 ||
             config_.evidence_exchange->worldRank() < 0 ||
             config_.evidence_exchange->worldRank() >=
                 config_.evidence_exchange->worldSize()))
        {
            throw std::invalid_argument(
                "Distributed ExpertOverlay transfer profiling requires one idle valid evidence lane");
        }
        if (config_.perf_device.empty())
            config_.perf_device = "expert_overlay";

        const auto &coordinates = config_.planner->requiredCoordinates();
        if (coordinates.empty() ||
            coordinates.size() != config_.ledger->coordinateCount())
        {
            throw std::invalid_argument(
                "ExpertOverlay transfer profiler and ledger coordinates disagree");
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
                    "ExpertOverlay transfer profile is missing a reverse directed coordinate");
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
                "ExpertOverlay transfer profile requires a reciprocal endpoint pair");
        }

        const std::uint64_t observations =
            config_.ledger->requiredObservationsPerCoordinate();
        if (observations == 0 ||
            jobs_.size() >
                std::numeric_limits<std::uint64_t>::max() / observations)
        {
            throw std::overflow_error(
                "ExpertOverlay transfer profile has invalid finite geometry");
        }
        expected_profile_waves_ =
            static_cast<std::uint64_t>(jobs_.size()) * observations;
        local_rows_.reserve(2);

        recordProfileCounter(
            config_.perf_device,
            "economy_transport_profile_expected_waves",
            static_cast<double>(expected_profile_waves_),
            {{"reciprocal_jobs", std::to_string(jobs_.size())},
             {"observations_per_job", std::to_string(observations)}});
        LOG_INFO(
            "[ExpertOverlay][Economy] Starting bounded transport profile jobs="
            << jobs_.size() << " observations_per_job=" << observations
            << " total_waves=" << expected_profile_waves_
            << " synthetic_inference=false");
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
        const auto observed = state();
        if (observed == MoEOverlayEconomyCalibrationState::Complete ||
            observed == MoEOverlayEconomyCalibrationState::Failed ||
            observed == MoEOverlayEconomyCalibrationState::Stopped)
        {
            return;
        }
        try
        {
            if (stop_requested_.load(std::memory_order_acquire))
            {
                progressStop();
                return;
            }
            switch (observed)
            {
            case MoEOverlayEconomyCalibrationState::StartWave:
                startWave();
                break;
            case MoEOverlayEconomyCalibrationState::AwaitConcurrentWave:
                pollWave();
                break;
            case MoEOverlayEconomyCalibrationState::AbortCleanup:
                pollAbortCleanup();
                break;
            case MoEOverlayEconomyCalibrationState::AwaitEvidenceExchange:
                pollEvidenceExchange();
                break;
            case MoEOverlayEconomyCalibrationState::Complete:
            case MoEOverlayEconomyCalibrationState::Failed:
            case MoEOverlayEconomyCalibrationState::Stopped:
                break;
            case MoEOverlayEconomyCalibrationState::ArmBaseline:
            case MoEOverlayEconomyCalibrationState::AwaitBaseline:
            case MoEOverlayEconomyCalibrationState::AwaitBaselineReadiness:
            case MoEOverlayEconomyCalibrationState::AwaitConcurrentInference:
            case MoEOverlayEconomyCalibrationState::AwaitLateConcurrentSample:
                throw std::logic_error(
                    "Obsolete inference-pair state entered the transfer profiler");
            }
        }
        catch (const std::exception &error)
        {
            if (active_wave_)
                beginAbort(error.what());
            else
                fail(error.what());
        }
        catch (...)
        {
            if (active_wave_)
            {
                beginAbort(
                    "ExpertOverlay transfer profiler raised a non-standard exception");
            }
            else
            {
                fail(
                    "ExpertOverlay transfer profiler raised a non-standard exception");
            }
        }
    }

    void MoEOverlayEconomyCalibrationController::startWave()
    {
        if (active_wave_ || evidence_exchange_active_ ||
            config_.journal->hasUnreadWave())
        {
            throw std::logic_error(
                "ExpertOverlay transfer profiler found stale attempt ownership");
        }
        if (active_profile_sequence_ == 0)
        {
            if (next_profile_sequence_ ==
                std::numeric_limits<std::uint64_t>::max())
            {
                throw std::overflow_error(
                    "ExpertOverlay transfer-profile sequence overflowed");
            }
            active_profile_sequence_ = ++next_profile_sequence_;
        }

        const auto &job = jobs_[job_index_];
        const auto transaction = config_.planner->buildPairSwap(
            job.first_participant,
            job.second_participant,
            job.layer,
            active_profile_sequence_);
        wave_start_attempts_.fetch_add(1, std::memory_order_relaxed);
        auto started = config_.transport->beginStage(transaction);
        if (!started.valid())
        {
            throw std::logic_error(
                "ExpertOverlay transfer profiler received invalid transport ownership");
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
                                            ? "ExpertOverlay transfer-profile wave failed to start"
                                            : std::move(started.error);
            if (active_wave_)
                beginAbort(message);
            else
                fail(message);
            return;
        }

        active_wave_ = std::move(started.wave);
        stage_completed_ = false;
        waves_started_.fetch_add(1, std::memory_order_relaxed);
        if (sample_index_ == 0)
        {
            LOG_DEBUG(
                "[ExpertOverlay][Economy] Profiling physical pair="
                << job.first_participant << "<->" << job.second_participant
                << " representative_layer=" << job.layer);
        }
        state_.store(
            MoEOverlayEconomyCalibrationState::AwaitConcurrentWave,
            std::memory_order_release);
    }

    void MoEOverlayEconomyCalibrationController::pollWave()
    {
        if (!active_wave_)
        {
            throw std::logic_error(
                "ExpertOverlay transfer profiler lost its active wave");
        }
        std::string error;
        const auto progress = active_wave_->pollStage(&error);
        if (progress == MoEOverlayResidencyWaveProgress::Pending)
            return;
        if (progress == MoEOverlayResidencyWaveProgress::Failed)
        {
            beginAbort(
                error.empty()
                    ? "ExpertOverlay transfer-profile staging failed"
                    : std::move(error));
            return;
        }
        if (progress == MoEOverlayResidencyWaveProgress::Deferred)
        {
            beginAbort();
            return;
        }
        stage_completed_ = true;
        waves_completed_.fetch_add(1, std::memory_order_relaxed);
        beginAbort();
    }

    void MoEOverlayEconomyCalibrationController::beginAbort(
        std::string failure_after_cleanup)
    {
        if (!active_wave_)
        {
            if (!failure_after_cleanup.empty())
                fail(std::move(failure_after_cleanup));
            return;
        }
        if (!failure_after_cleanup.empty() &&
            failure_after_cleanup_.empty())
        {
            failure_after_cleanup_ = std::move(failure_after_cleanup);
        }
        active_wave_->abortStaged();
        waves_aborted_.fetch_add(1, std::memory_order_relaxed);
        state_.store(
            MoEOverlayEconomyCalibrationState::AbortCleanup,
            std::memory_order_release);
    }

    void MoEOverlayEconomyCalibrationController::pollAbortCleanup()
    {
        if (!active_wave_)
        {
            throw std::logic_error(
                "ExpertOverlay transfer profiler lost abort ownership");
        }
        std::string error;
        const auto progress = active_wave_->pollAbort(&error);
        if (progress == MoEOverlayResidencyWaveProgress::Pending)
            return;
        if (progress != MoEOverlayResidencyWaveProgress::Ready)
        {
            fail(
                error.empty()
                    ? "ExpertOverlay transfer-profile abort cleanup failed"
                    : std::move(error));
            return;
        }
        active_wave_.reset();

        if (!failure_after_cleanup_.empty())
        {
            auto message = std::move(failure_after_cleanup_);
            failure_after_cleanup_.clear();
            fail(std::move(message));
            return;
        }
        if (!stage_completed_)
        {
            active_profile_sequence_ = 0;
            state_.store(
                MoEOverlayEconomyCalibrationState::StartWave,
                std::memory_order_release);
            return;
        }

        std::string journal_error;
        if (!config_.journal->consume(&local_rows_, &journal_error))
        {
            fail(
                journal_error.empty()
                    ? "ExpertOverlay transfer profile produced no timing rows"
                    : std::move(journal_error));
            return;
        }
        if (stop_requested_.load(std::memory_order_acquire))
        {
            local_rows_.clear();
            state_.store(
                MoEOverlayEconomyCalibrationState::Stopped,
                std::memory_order_release);
            return;
        }
        if (!config_.evidence_exchange)
        {
            recordCompletedWave(local_rows_);
            return;
        }

        const MoEOverlayMigrationProfileEvidence evidence{
            .profile_sequence = active_profile_sequence_,
            .coordinate = currentCoordinate(),
            .local_measurements = local_rows_,
        };
        std::string exchange_error;
        if (!config_.evidence_exchange->beginMigrationProfile(
                evidence, &exchange_error))
        {
            fail(
                exchange_error.empty()
                    ? "ExpertOverlay could not begin migration-profile evidence exchange"
                    : std::move(exchange_error));
            return;
        }
        evidence_exchange_active_ = true;
        evidence_exchanges_started_.fetch_add(1, std::memory_order_relaxed);
        state_.store(
            MoEOverlayEconomyCalibrationState::AwaitEvidenceExchange,
            std::memory_order_release);
    }

    void MoEOverlayEconomyCalibrationController::pollEvidenceExchange()
    {
        if (!config_.evidence_exchange || !evidence_exchange_active_)
        {
            throw std::logic_error(
                "ExpertOverlay transfer profiler lost evidence exchange ownership");
        }
        MoEOverlayMigrationProfileResult result;
        std::string error;
        const auto progress = config_.evidence_exchange->pollMigrationProfile(
            &result, &error);
        if (progress == MoEOverlayResidencyWaveProgress::Pending)
            return;
        evidence_exchange_active_ = false;
        if (progress != MoEOverlayResidencyWaveProgress::Ready ||
            !result.valid())
        {
            fail(
                error.empty()
                    ? "ExpertOverlay migration-profile evidence exchange failed"
                    : std::move(error));
            return;
        }
        evidence_exchanges_completed_.fetch_add(1, std::memory_order_relaxed);
        recordCompletedWave(result.measurements);
    }

    void MoEOverlayEconomyCalibrationController::recordCompletedWave(
        const std::vector<MoEOverlayCompletedMigrationMeasurement> &rows)
    {
        std::string ledger_error;
        if (!config_.ledger->recordCompletedWave(rows, &ledger_error))
        {
            throw std::logic_error(
                ledger_error.empty()
                    ? "ExpertOverlay transfer-profile ledger rejected timing rows"
                    : std::move(ledger_error));
        }
        accepted_pairs_.fetch_add(1, std::memory_order_relaxed);
        recordProfileCounter(
            config_.perf_device,
            "economy_transport_profile_waves_completed",
            1.0,
            {{"source_participant",
              std::to_string(currentCoordinate().source_participant)},
             {"destination_participant",
              std::to_string(currentCoordinate().destination_participant)},
             {"layer", std::to_string(currentCoordinate().layer)}});
        advanceProfileWave();
    }

    void MoEOverlayEconomyCalibrationController::advanceProfileWave()
    {
        local_rows_.clear();
        stage_completed_ = false;
        active_profile_sequence_ = 0;
        ++sample_index_;
        if (sample_index_ >=
            config_.ledger->requiredObservationsPerCoordinate())
        {
            sample_index_ = 0;
            ++job_index_;
        }
        if (job_index_ < jobs_.size())
        {
            state_.store(
                MoEOverlayEconomyCalibrationState::StartWave,
                std::memory_order_release);
            return;
        }
        if (!config_.ledger->ready())
        {
            throw std::logic_error(
                "ExpertOverlay transfer profiler exhausted its finite jobs before the ledger became ready");
        }
        sealed_ = config_.ledger->seal();
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - profiling_started_at_)
                                 .count();
        state_.store(
            MoEOverlayEconomyCalibrationState::Complete,
            std::memory_order_release);
        recordProfileCounter(
            config_.perf_device,
            "economy_transport_profile_complete",
            1.0,
            {{"elapsed_nanoseconds", std::to_string(std::max<int64_t>(1, elapsed))},
             {"waves", std::to_string(expected_profile_waves_)}});
        LOG_INFO(
            "[ExpertOverlay][Economy] Transport profile complete waves="
            << expected_profile_waves_ << " elapsed_ms="
            << (std::max<int64_t>(1, elapsed) / 1000000.0)
            << " synthetic_inference=false");
    }

    void MoEOverlayEconomyCalibrationController::progressStop()
    {
        if (active_wave_)
        {
            if (state() != MoEOverlayEconomyCalibrationState::AbortCleanup)
                beginAbort();
            pollAbortCleanup();
            return;
        }
        if (evidence_exchange_active_)
        {
            MoEOverlayMigrationProfileResult ignored;
            std::string error;
            const auto progress =
                config_.evidence_exchange->pollMigrationProfile(
                    &ignored, &error);
            if (progress == MoEOverlayResidencyWaveProgress::Pending)
                return;
            evidence_exchange_active_ = false;
            if (progress != MoEOverlayResidencyWaveProgress::Ready)
            {
                fail(
                    error.empty()
                        ? "ExpertOverlay transfer-profile exchange failed while stopping"
                        : std::move(error));
                return;
            }
        }
        local_rows_.clear();
        state_.store(
            MoEOverlayEconomyCalibrationState::Stopped,
            std::memory_order_release);
    }

    void MoEOverlayEconomyCalibrationController::requestStop() noexcept
    {
        stop_requested_.store(true, std::memory_order_release);
    }

    void MoEOverlayEconomyCalibrationController::fail(
        std::string message) noexcept
    {
        if (!healthy_.exchange(false, std::memory_order_acq_rel))
            return;
        if (message.empty())
            message = "Unknown ExpertOverlay transfer-profile failure";
        {
            std::lock_guard<std::mutex> lock(failure_mutex_);
            failure_message_ = std::move(message);
        }
        fatal_failures_.fetch_add(1, std::memory_order_relaxed);
        state_.store(
            MoEOverlayEconomyCalibrationState::Failed,
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

    std::string MoEOverlayEconomyCalibrationController::failureMessage() const
    {
        std::lock_guard<std::mutex> lock(failure_mutex_);
        return failure_message_;
    }

    MoEOverlayEconomyCalibrationStats
    MoEOverlayEconomyCalibrationController::stats() const noexcept
    {
        return {
            .polls = polls_.load(std::memory_order_relaxed),
            .wave_start_attempts =
                wave_start_attempts_.load(std::memory_order_relaxed),
            .waves_started = waves_started_.load(std::memory_order_relaxed),
            .waves_deferred = waves_deferred_.load(std::memory_order_relaxed),
            .waves_completed = waves_completed_.load(std::memory_order_relaxed),
            .waves_aborted = waves_aborted_.load(std::memory_order_relaxed),
            .evidence_exchanges_started =
                evidence_exchanges_started_.load(std::memory_order_relaxed),
            .evidence_exchanges_completed =
                evidence_exchanges_completed_.load(std::memory_order_relaxed),
            .accepted_pairs = accepted_pairs_.load(std::memory_order_relaxed),
            .fatal_failures = fatal_failures_.load(std::memory_order_relaxed),
        };
    }

    const MoEOverlaySealedMigrationMeasurements *
    MoEOverlayEconomyCalibrationController::sealedMeasurements() const noexcept
    {
        return sealed_ ? &*sealed_ : nullptr;
    }
} // namespace llaminar2
