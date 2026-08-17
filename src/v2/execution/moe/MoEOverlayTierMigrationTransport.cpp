/**
 * @file MoEOverlayTierMigrationTransport.cpp
 * @brief Non-blocking arbitrary-tier residency wave composition.
 *
 * The composite owns no device stream itself. Exact endpoint operations own
 * streams/events and are polled independently. This layer only enforces the
 * transaction barrier: all projections Ready, then inactive-bank commit Ready,
 * then the outer residency authority may publish the candidate epoch.
 */

#include "MoEOverlayTierMigrationTransport.h"

#include "../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    bool MoEOverlayCompletedMigrationMeasurement::valid() const noexcept
    {
        if (expected_epoch == 0 || candidate_epoch != expected_epoch + 1 ||
            source_participant < 0 || destination_participant < 0 ||
            source_participant == destination_participant || layer < 0 ||
            expert < 0 || wave_wall_nanoseconds == 0)
        {
            return false;
        }
        return std::all_of(
            projections.begin(),
            projections.end(),
            [](const auto &projection)
            {
                return !projection.has_value() || projection->valid();
            });
    }

    namespace
    {
        /** @brief Store a diagnostic only when the caller requested one. */
        void assignTierTransportError(
            std::string *error,
            const std::string &message)
        {
            if (error)
                *error = message;
        }
    } // namespace

    bool MoEOverlayTierPreparedWave::valid(
        std::size_t expected_transfer_count) const noexcept
    {
        const bool all_transfers_owned = std::all_of(
            transfers.begin(),
            transfers.end(),
            [](const auto &transfer)
            {
                return transfer != nullptr;
            });
        if (status == MoEOverlayResidencyStageStartStatus::Started)
        {
            return inactive_bank != nullptr &&
                   transfers.size() == expected_transfer_count &&
                   all_transfers_owned;
        }
        if (status == MoEOverlayResidencyStageStartStatus::Deferred)
            return transfers.empty() && inactive_bank == nullptr;

        /* A failed factory may own a partially enqueued, fenced prefix. */
        return all_transfers_owned;
    }

    /** @brief Atomic counters shared with waves that may outlive the facade. */
    struct MoEOverlayTierMigrationTransportSharedStats
    {
        std::atomic<std::uint64_t> waves_started{0};
        std::atomic<std::uint64_t> waves_deferred{0};
        std::atomic<std::uint64_t> waves_failed_to_prepare{0};
        std::atomic<std::uint64_t> transfer_operations_started{0};
        std::atomic<std::uint64_t> transfer_operations_completed{0};
        std::atomic<std::uint64_t> stage_pending_polls{0};
        std::atomic<std::uint64_t> commits_started{0};
        std::atomic<std::uint64_t> commits_completed{0};
        std::atomic<std::uint64_t> commit_pending_polls{0};
        std::atomic<std::uint64_t> waves_aborted{0};
        std::atomic<std::uint64_t> abort_pending_polls{0};
        std::atomic<std::uint64_t> abort_cleanups_completed{0};
        std::atomic<std::uint64_t> old_banks_retired{0};
    };

    namespace
    {
        /**
         * @brief Owned composite work implementing the authority's wave ABI.
         *
         * Ready flags prevent repeatedly polling completed operations. All
         * remaining operations are still visited in a maintenance pass even
         * when an earlier one is pending, preserving cross-lane concurrency.
         */
        class CompositeResidencyWave final : public IMoEOverlayResidencyWave
        {
        public:
            /**
             * @brief Take ownership of every transfer and inactive bank.
             * @param transfers Already-enqueued projection operations.
             * @param inactive_bank Reserved candidate runtime bank.
             * @param stats Shared lifetime-safe transport counters.
             * @param perf_device Stable evidence device/topology label.
             * @param migration_count Expert movements in the transaction.
             */
            CompositeResidencyWave(
                std::vector<std::unique_ptr<IMoEOverlayTierTransferOperation>>
                    transfers,
                std::unique_ptr<IMoEOverlayInactiveBankTransaction>
                    inactive_bank,
                std::shared_ptr<MoEOverlayTierMigrationTransportSharedStats>
                    stats,
                std::string perf_device,
                std::size_t migration_count,
                std::vector<MoEOverlayCompletedMigrationMeasurement>
                    measurements = {},
                std::shared_ptr<IMoEOverlayMigrationMeasurementSink>
                    measurement_sink = nullptr,
                bool require_complete_local_measurements = false,
                bool commit_allowed = true)
                : transfers_(std::move(transfers)),
                  transfer_ready_(transfers_.size(), false),
                  transfer_abort_ready_(transfers_.size(), false),
                  inactive_bank_(std::move(inactive_bank)),
                  stats_(std::move(stats)),
                  perf_device_(std::move(perf_device)),
                  migration_count_(migration_count),
                  measurements_(std::move(measurements)),
                  measurement_sink_(std::move(measurement_sink)),
                  require_complete_local_measurements_(
                      require_complete_local_measurements),
                  commit_allowed_(commit_allowed)
            {
                if ((!measurements_.empty() ||
                     require_complete_local_measurements_) &&
                    !measurement_sink_)
                {
                    throw std::invalid_argument(
                        "Measured ExpertOverlay wave requires a measurement sink");
                }
                if (!measurements_.empty() &&
                    measurements_.size() != migration_count_)
                {
                    throw std::invalid_argument(
                        "Measured ExpertOverlay wave has inconsistent migration geometry");
                }
            }

            /** @brief Poll every unfinished transfer once without waiting. */
            MoEOverlayResidencyWaveProgress pollStage(
                std::string *error) noexcept override
            {
                if (aborted_)
                {
                    assignTierTransportError(
                        error,
                        failure_.empty()
                            ? "Composite residency wave was aborted"
                            : failure_);
                    return MoEOverlayResidencyWaveProgress::Failed;
                }
                if (stage_ready_)
                    return MoEOverlayResidencyWaveProgress::Ready;

                if (!stage_dispatch_started_)
                {
                    /*
                     * Construction reserves and pins the complete wave, but its
                     * queued operations release no bytes until this first poll.
                     * Starting the interval here lets calibration safely retain
                     * prepared ownership while awaiting an inference ticket.
                     */
                    wave_started_at_ = std::chrono::steady_clock::now();
                    stage_dispatch_started_ = true;
                }

                bool any_pending = false;
                for (std::size_t index = 0;
                     index < transfers_.size();
                     ++index)
                {
                    if (transfer_ready_[index])
                        continue;
                    std::string operation_error;
                    const auto progress =
                        transfers_[index]->poll(&operation_error);
                    if (progress == MoEOverlayResidencyWaveProgress::Failed)
                    {
                        failure_ = operation_error.empty()
                                       ? "A tier transfer operation failed"
                                       : operation_error;
                        assignTierTransportError(error, failure_);
                        abortStaged();
                        return MoEOverlayResidencyWaveProgress::Failed;
                    }
                    if (progress == MoEOverlayResidencyWaveProgress::Ready)
                    {
                        if (!measurements_.empty())
                        {
                            const std::size_t migration_index = index / 3u;
                            const std::size_t projection_index = index % 3u;
                            if (migration_index >= measurements_.size())
                            {
                                failure_ =
                                    "Measured ExpertOverlay operation order exceeds its transaction geometry";
                                assignTierTransportError(error, failure_);
                                abortStaged();
                                return MoEOverlayResidencyWaveProgress::Failed;
                            }
                            auto measurement =
                                transfers_[index]->completedMeasurement();
                            if (require_complete_local_measurements_ &&
                                (!measurement || !measurement->valid()))
                            {
                                failure_ =
                                    "A required local ExpertOverlay projection omitted exact timing evidence";
                                assignTierTransportError(error, failure_);
                                abortStaged();
                                return MoEOverlayResidencyWaveProgress::Failed;
                            }
                            measurements_[migration_index]
                                .projections[projection_index] =
                                std::move(measurement);
                        }
                        transfer_ready_[index] = true;
                        stats_->transfer_operations_completed.fetch_add(
                            1,
                            std::memory_order_relaxed);
                    }
                    else
                    {
                        any_pending = true;
                    }
                }

                stage_ready_ = std::all_of(
                    transfer_ready_.begin(),
                    transfer_ready_.end(),
                    [](bool ready) { return ready; });
                if (!stage_ready_ || any_pending)
                {
                    stats_->stage_pending_polls.fetch_add(
                        1,
                        std::memory_order_relaxed);
                    return MoEOverlayResidencyWaveProgress::Pending;
                }

                const auto wave_finished_at =
                    std::chrono::steady_clock::now();
                const auto begin_count =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        wave_started_at_.time_since_epoch()).count();
                const auto end_count =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        wave_finished_at.time_since_epoch()).count();
                const std::uint64_t begin_nanoseconds =
                    static_cast<std::uint64_t>(
                        std::max<std::int64_t>(1, begin_count));
                const std::uint64_t end_nanoseconds =
                    static_cast<std::uint64_t>(
                        std::max<std::int64_t>(
                            static_cast<std::int64_t>(begin_nanoseconds + 1u),
                            end_count));
                stage_interval_ = MoEOverlayResidencyWaveInterval{
                    .begin_steady_nanoseconds = begin_nanoseconds,
                    .end_steady_nanoseconds = end_nanoseconds,
                };

                if (!measurements_.empty() && !measurement_published_)
                {
                    const auto elapsed =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            wave_finished_at - wave_started_at_)
                            .count();
                    const std::uint64_t wave_nanoseconds =
                        static_cast<std::uint64_t>(
                            std::max<std::int64_t>(1, elapsed));
                    for (auto &measurement : measurements_)
                    {
                        measurement.wave_wall_nanoseconds = wave_nanoseconds;
                        if (!measurement.valid())
                        {
                            failure_ =
                                "ExpertOverlay completed-wave measurement is internally inconsistent";
                            assignTierTransportError(error, failure_);
                            abortStaged();
                            return MoEOverlayResidencyWaveProgress::Failed;
                        }
                    }
                    std::string measurement_error;
                    if (!measurement_sink_->recordCompletedWave(
                            measurements_, &measurement_error))
                    {
                        failure_ = measurement_error.empty()
                                       ? "ExpertOverlay measurement sink rejected a completed wave"
                                       : std::move(measurement_error);
                        assignTierTransportError(error, failure_);
                        abortStaged();
                        return MoEOverlayResidencyWaveProgress::Failed;
                    }
                    measurement_published_ = true;
                }

                PerfStatsCollector::addCounter(
                    "moe_overlay_residency",
                    "composite_transfer_operations_ready",
                    static_cast<double>(transfers_.size()),
                    "maintenance",
                    perf_device_,
                    {{"migrations", std::to_string(migration_count_)}});
                return MoEOverlayResidencyWaveProgress::Ready;
            }

            /** @brief Return the exact local prepare-to-ready interval. */
            [[nodiscard]] std::optional<MoEOverlayResidencyWaveInterval>
            completedStageInterval() const noexcept override
            {
                if (!stage_ready_ || !stage_interval_.valid())
                    return std::nullopt;
                return stage_interval_;
            }

            /** @brief Begin bank commit only after all transfer operations are ready. */
            bool beginCommit(std::string *error) noexcept override
            {
                if (!commit_allowed_ || aborted_ || !stage_ready_ ||
                    commit_started_)
                {
                    assignTierTransportError(
                        error,
                        !commit_allowed_
                            ? "Economy calibration waves cannot commit an inactive residency bank"
                        : aborted_
                            ? "Cannot commit an aborted residency wave"
                            : (!stage_ready_
                                   ? "Cannot commit before every transfer is ready"
                                   : "Residency wave commit was already started"));
                    return false;
                }
                if (!inactive_bank_->beginCommit(error))
                    return false;
                commit_started_ = true;
                stats_->commits_started.fetch_add(1, std::memory_order_relaxed);
                return true;
            }

            /** @brief Poll the one candidate-bank completion authority. */
            MoEOverlayResidencyWaveProgress pollCommit(
                std::string *error) noexcept override
            {
                if (aborted_ || !commit_started_)
                {
                    assignTierTransportError(
                        error,
                        aborted_
                            ? "Cannot poll commit for an aborted residency wave"
                            : "Residency wave commit was not started");
                    return MoEOverlayResidencyWaveProgress::Failed;
                }
                if (commit_ready_)
                    return MoEOverlayResidencyWaveProgress::Ready;

                const auto progress = inactive_bank_->pollCommit(error);
                if (progress == MoEOverlayResidencyWaveProgress::Pending)
                {
                    stats_->commit_pending_polls.fetch_add(
                        1,
                        std::memory_order_relaxed);
                    return progress;
                }
                if (progress == MoEOverlayResidencyWaveProgress::Failed)
                {
                    failure_ = error && !error->empty()
                                   ? *error
                                   : "Inactive residency bank commit failed";
                    abortStaged();
                    return progress;
                }

                commit_ready_ = true;
                stats_->commits_completed.fetch_add(
                    1,
                    std::memory_order_relaxed);
                PerfStatsCollector::addCounter(
                    "moe_overlay_residency",
                    "inactive_banks_ready",
                    1.0,
                    "maintenance",
                    perf_device_,
                    {{"migrations", std::to_string(migration_count_)}});
                return progress;
            }

            /** @brief Abort each transfer and the inactive-bank reservation once. */
            void abortStaged() noexcept override
            {
                if (aborted_ || retired_)
                    return;
                aborted_ = true;
                for (auto &transfer : transfers_)
                    transfer->abort();
                if (inactive_bank_)
                    inactive_bank_->abort();
                stats_->waves_aborted.fetch_add(1, std::memory_order_relaxed);
            }

            /** @brief Poll every abort edge and retain all resources until ready. */
            MoEOverlayResidencyWaveProgress pollAbort(
                std::string *error) noexcept override
            {
                if (!aborted_)
                {
                    assignTierTransportError(
                        error,
                        "Cannot poll abort cleanup before abortStaged()");
                    return MoEOverlayResidencyWaveProgress::Failed;
                }
                if (abort_cleanup_ready_)
                    return MoEOverlayResidencyWaveProgress::Ready;

                bool any_pending = false;
                for (std::size_t index = 0;
                     index < transfers_.size();
                     ++index)
                {
                    if (transfer_abort_ready_[index])
                        continue;
                    std::string operation_error;
                    const auto progress =
                        transfers_[index]->pollAbort(&operation_error);
                    if (progress == MoEOverlayResidencyWaveProgress::Failed)
                    {
                        assignTierTransportError(
                            error,
                            operation_error.empty()
                                ? "A tier transfer abort edge failed"
                                : operation_error);
                        return progress;
                    }
                    if (progress == MoEOverlayResidencyWaveProgress::Ready)
                        transfer_abort_ready_[index] = true;
                    else
                        any_pending = true;
                }

                if (!inactive_bank_)
                {
                    inactive_bank_abort_ready_ = true;
                }
                else if (!inactive_bank_abort_ready_)
                {
                    std::string bank_error;
                    const auto progress =
                        inactive_bank_->pollAbort(&bank_error);
                    if (progress == MoEOverlayResidencyWaveProgress::Failed)
                    {
                        assignTierTransportError(
                            error,
                            bank_error.empty()
                                ? "Inactive-bank abort edge failed"
                                : bank_error);
                        return progress;
                    }
                    if (progress == MoEOverlayResidencyWaveProgress::Ready)
                        inactive_bank_abort_ready_ = true;
                    else
                        any_pending = true;
                }

                abort_cleanup_ready_ =
                    inactive_bank_abort_ready_ &&
                    std::all_of(
                        transfer_abort_ready_.begin(),
                        transfer_abort_ready_.end(),
                        [](bool ready) { return ready; });
                if (!abort_cleanup_ready_ || any_pending)
                {
                    stats_->abort_pending_polls.fetch_add(
                        1,
                        std::memory_order_relaxed);
                    return MoEOverlayResidencyWaveProgress::Pending;
                }

                stats_->abort_cleanups_completed.fetch_add(
                    1,
                    std::memory_order_relaxed);
                PerfStatsCollector::addCounter(
                    "moe_overlay_residency",
                    "composite_abort_cleanup_ready",
                    1.0,
                    "maintenance",
                    perf_device_,
                    {{"migrations", std::to_string(migration_count_)}});
                return MoEOverlayResidencyWaveProgress::Ready;
            }

            /** @brief Delegate old-bank retirement after the authority drains leases. */
            void retirePrevious() noexcept override
            {
                if (retired_ || aborted_)
                    return;
                inactive_bank_->retirePrevious();
                retired_ = true;
                stats_->old_banks_retired.fetch_add(
                    1,
                    std::memory_order_relaxed);
            }

        private:
            std::vector<std::unique_ptr<IMoEOverlayTierTransferOperation>>
                transfers_;
            std::vector<bool> transfer_ready_;
            std::vector<bool> transfer_abort_ready_;
            std::unique_ptr<IMoEOverlayInactiveBankTransaction> inactive_bank_;
            std::shared_ptr<MoEOverlayTierMigrationTransportSharedStats> stats_;
            std::string perf_device_;
            std::size_t migration_count_ = 0;
            std::vector<MoEOverlayCompletedMigrationMeasurement>
                measurements_;
            std::shared_ptr<IMoEOverlayMigrationMeasurementSink>
                measurement_sink_;
            bool require_complete_local_measurements_ = false;
            std::chrono::steady_clock::time_point wave_started_at_{};
            MoEOverlayResidencyWaveInterval stage_interval_{};
            bool measurement_published_ = false;
            bool commit_allowed_ = true;
            bool stage_dispatch_started_ = false;
            bool stage_ready_ = false;
            bool commit_started_ = false;
            bool commit_ready_ = false;
            bool aborted_ = false;
            bool inactive_bank_abort_ready_ = false;
            bool abort_cleanup_ready_ = false;
            bool retired_ = false;
            std::string failure_;
        };
    } // namespace

    MoEOverlayTierMigrationTransport::MoEOverlayTierMigrationTransport(
        Config config)
        : config_(std::move(config)),
          stats_(
              std::make_shared<
                  MoEOverlayTierMigrationTransportSharedStats>())
    {
        if (!config_.factory)
            throw std::invalid_argument(
                "Tier migration transport requires an endpoint wave factory");
        if (config_.projections_per_expert == 0)
            throw std::invalid_argument(
                "Tier migration transport requires positive projection cardinality");
        if (config_.projections_per_expert != 3 && config_.measurement_sink)
        {
            throw std::invalid_argument(
                "ExpertOverlay measurement sink requires the gate/up/down projection ABI");
        }
        if (config_.require_complete_local_measurements &&
            !config_.measurement_sink)
        {
            throw std::invalid_argument(
                "Required ExpertOverlay measurements need an installed sink");
        }
        if (config_.perf_device.empty())
            config_.perf_device = "expert_overlay";
    }

    MoEOverlayResidencyStageStart
    MoEOverlayTierMigrationTransport::beginStage(
        const MoEOverlayResidencyTransaction &transaction)
    {
        if (!transaction.valid() || transaction.empty())
        {
            stats_->waves_failed_to_prepare.fetch_add(
                1,
                std::memory_order_relaxed);
            return {
                .status = MoEOverlayResidencyStageStartStatus::Failed,
                .error = "Tier migration transport requires a valid non-empty transaction",
            };
        }

        const std::size_t expected_transfer_count =
            transaction.migrations.size() * config_.projections_per_expert;
        MoEOverlayTierPreparedWave prepared =
            config_.factory->prepare(transaction);
        if (!prepared.valid(expected_transfer_count))
        {
            stats_->waves_failed_to_prepare.fetch_add(
                1,
                std::memory_order_relaxed);
            prepared.transfers.erase(
                std::remove_if(
                    prepared.transfers.begin(),
                    prepared.transfers.end(),
                    [](const auto &transfer) { return transfer == nullptr; }),
                prepared.transfers.end());
            std::unique_ptr<IMoEOverlayResidencyWave> cleanup;
            if (!prepared.transfers.empty() || prepared.inactive_bank)
            {
                cleanup = std::make_unique<CompositeResidencyWave>(
                    std::move(prepared.transfers),
                    std::move(prepared.inactive_bank),
                    stats_,
                    config_.perf_device,
                    transaction.migrations.size());
            }
            return {
                .status = MoEOverlayResidencyStageStartStatus::Failed,
                .cleanup_wave = std::move(cleanup),
                .error = prepared.error.empty()
                             ? "Endpoint factory returned an invalid prepared-wave shape"
                             : std::move(prepared.error),
            };
        }

        if (prepared.status ==
            MoEOverlayResidencyStageStartStatus::Deferred)
        {
            stats_->waves_deferred.fetch_add(1, std::memory_order_relaxed);
            return {
                .status = prepared.status,
                .error = std::move(prepared.error),
            };
        }
        if (prepared.status == MoEOverlayResidencyStageStartStatus::Failed)
        {
            stats_->waves_failed_to_prepare.fetch_add(
                1,
                std::memory_order_relaxed);
            std::unique_ptr<IMoEOverlayResidencyWave> cleanup;
            if (!prepared.transfers.empty() || prepared.inactive_bank)
            {
                cleanup = std::make_unique<CompositeResidencyWave>(
                    std::move(prepared.transfers),
                    std::move(prepared.inactive_bank),
                    stats_,
                    config_.perf_device,
                    transaction.migrations.size());
            }
            return {
                .status = prepared.status,
                .cleanup_wave = std::move(cleanup),
                .error = prepared.error.empty()
                             ? "Endpoint factory failed to prepare a residency wave"
                             : std::move(prepared.error),
            };
        }

        stats_->waves_started.fetch_add(1, std::memory_order_relaxed);
        stats_->transfer_operations_started.fetch_add(
            expected_transfer_count,
            std::memory_order_relaxed);
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "composite_background_waves_started",
            1.0,
            "maintenance",
            config_.perf_device,
            {{"migrations", std::to_string(transaction.migrations.size())},
             {"transfer_operations", std::to_string(expected_transfer_count)}});

        const bool collect_measurements =
            config_.measurement_sink &&
            (config_.measurement_scope ==
                 MoEOverlayMigrationMeasurementScope::AllWaves ||
             transaction.purpose ==
                 MoEOverlayResidencyTransactionPurpose::EconomyCalibration);
        std::vector<MoEOverlayCompletedMigrationMeasurement> measurements;
        if (collect_measurements)
        {
            measurements.reserve(transaction.migrations.size());
            for (const auto &migration : transaction.migrations)
            {
                measurements.push_back({
                    .expected_epoch = transaction.expected_epoch,
                    .candidate_epoch = transaction.candidate->epoch,
                    .source_participant =
                        migration.source.owner_participant,
                    .destination_participant =
                        migration.destination.owner_participant,
                    .layer = migration.layer_idx,
                    .expert = migration.expert_id,
                });
            }
        }

        return {
            .status = MoEOverlayResidencyStageStartStatus::Started,
            .wave = std::make_unique<CompositeResidencyWave>(
                std::move(prepared.transfers),
                std::move(prepared.inactive_bank),
                stats_,
                config_.perf_device,
                transaction.migrations.size(),
                std::move(measurements),
                collect_measurements ? config_.measurement_sink : nullptr,
                collect_measurements &&
                    config_.require_complete_local_measurements,
                transaction.purpose ==
                    MoEOverlayResidencyTransactionPurpose::PlacementChange),
        };
    }

    MoEOverlayTierMigrationTransportStats
    MoEOverlayTierMigrationTransport::stats() const noexcept
    {
        return {
            .waves_started =
                stats_->waves_started.load(std::memory_order_relaxed),
            .waves_deferred =
                stats_->waves_deferred.load(std::memory_order_relaxed),
            .waves_failed_to_prepare =
                stats_->waves_failed_to_prepare.load(std::memory_order_relaxed),
            .transfer_operations_started =
                stats_->transfer_operations_started.load(
                    std::memory_order_relaxed),
            .transfer_operations_completed =
                stats_->transfer_operations_completed.load(
                    std::memory_order_relaxed),
            .stage_pending_polls =
                stats_->stage_pending_polls.load(std::memory_order_relaxed),
            .commits_started =
                stats_->commits_started.load(std::memory_order_relaxed),
            .commits_completed =
                stats_->commits_completed.load(std::memory_order_relaxed),
            .commit_pending_polls =
                stats_->commit_pending_polls.load(std::memory_order_relaxed),
            .waves_aborted =
                stats_->waves_aborted.load(std::memory_order_relaxed),
            .abort_pending_polls =
                stats_->abort_pending_polls.load(std::memory_order_relaxed),
            .abort_cleanups_completed =
                stats_->abort_cleanups_completed.load(
                    std::memory_order_relaxed),
            .old_banks_retired =
                stats_->old_banks_retired.load(std::memory_order_relaxed),
            .inference_stream_waits = 0,
            .blocking_synchronizations = 0,
        };
    }
} // namespace llaminar2
