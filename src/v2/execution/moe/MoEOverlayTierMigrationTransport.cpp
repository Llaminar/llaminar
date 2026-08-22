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
             * @brief Single lifecycle for transfer, bank, and retirement barriers.
             *
             * Per-operation readiness remains a vector because transfers
             * genuinely progress independently.  The wave itself has exactly
             * one phase, eliminating contradictory flag combinations at the
             * publication and abort boundaries.
             */
            enum class Phase
            {
                Reserved,
                Staging,
                Staged,
                Preparing,
                Prepared,
                Publishing,
                Published,
                RetirementFencing,
                RetirementReady,
                Aborting,
                Aborted,
                Retired,
            };

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
                std::shared_ptr<IMoEOverlayTransferProgressAuthority>
                    transfer_progress_authority,
                std::vector<MoEOverlayCompletedMigrationMeasurement>
                    measurements = {},
                std::shared_ptr<IMoEOverlayMigrationMeasurementSink>
                    measurement_sink = nullptr,
                bool require_complete_local_measurements = false,
                bool publication_allowed = true)
                : transfers_(std::move(transfers)),
                  transfer_ready_(transfers_.size(), false),
                  transfer_abort_ready_(transfers_.size(), false),
                  inactive_bank_(std::move(inactive_bank)),
                  stats_(std::move(stats)),
                  perf_device_(std::move(perf_device)),
                  migration_count_(migration_count),
                  transfer_progress_authority_(
                      std::move(transfer_progress_authority)),
                  measurements_(std::move(measurements)),
                  measurement_sink_(std::move(measurement_sink)),
                  require_complete_local_measurements_(
                      require_complete_local_measurements),
                  publication_allowed_(publication_allowed)
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
                if (phase_ == Phase::Aborting || phase_ == Phase::Aborted)
                {
                    assignTierTransportError(
                        error,
                        failure_.empty()
                            ? "Composite residency wave was aborted"
                            : failure_);
                    return MoEOverlayResidencyWaveProgress::Failed;
                }
                if (phase_ == Phase::Staged)
                    return MoEOverlayResidencyWaveProgress::Ready;
                if (phase_ != Phase::Reserved && phase_ != Phase::Staging)
                {
                    assignTierTransportError(
                        error,
                        "Composite residency stage was polled outside its staging lifecycle");
                    return MoEOverlayResidencyWaveProgress::Failed;
                }

                if (phase_ == Phase::Reserved)
                {
                    /*
                     * Construction reserves and pins the complete wave, but its
                     * queued operations release no bytes until this first poll.
                     * Starting the interval here lets calibration safely retain
                     * prepared ownership while awaiting an inference ticket.
                     */
                    wave_started_at_ = std::chrono::steady_clock::now();
                    phase_ = Phase::Staging;
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

                /*
                 * A lane may publish its first or next chunk while the
                 * inference graph's captured command-claim branch is already
                 * past graph entry. Kick the fabric-owned, device-deduplicated
                 * retained epochs after visiting every lane. This waits only
                 * for each worker to enqueue its independent graph; CUDA/ROCm
                 * transfer streams then progress concurrently with inference
                 * and with one another.
                 */
                if (!submitTransferProgress(error))
                {
                    failure_ = error && !error->empty()
                                   ? *error
                                   : "ExpertOverlay transfer-progress submission failed";
                    abortStaged();
                    return MoEOverlayResidencyWaveProgress::Failed;
                }

                const bool all_transfers_ready = std::all_of(
                    transfer_ready_.begin(),
                    transfer_ready_.end(),
                    [](bool ready) { return ready; });
                if (!all_transfers_ready || any_pending)
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

                if (!measurements_.empty())
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
                }

                phase_ = Phase::Staged;
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
                if (!stage_interval_.valid())
                    return std::nullopt;
                return stage_interval_;
            }

            /** @brief Begin inactive-bank preparation after every transfer is ready. */
            bool beginPrepare(std::string *error) noexcept override
            {
                if (!publication_allowed_ || phase_ != Phase::Staged)
                {
                    assignTierTransportError(
                        error,
                        !publication_allowed_
                            ? "Economy calibration waves cannot prepare a publishable residency bank"
                            : "Residency wave can prepare only after staging completes");
                    return false;
                }
                if (!inactive_bank_->beginPrepare(error))
                    return false;
                phase_ = Phase::Preparing;
                stats_->commits_started.fetch_add(1, std::memory_order_relaxed);
                return true;
            }

            /** @brief Poll the one candidate-bank preparation authority. */
            MoEOverlayResidencyWaveProgress pollPrepare(
                std::string *error) noexcept override
            {
                if (phase_ == Phase::Prepared)
                    return MoEOverlayResidencyWaveProgress::Ready;
                if (phase_ != Phase::Preparing)
                {
                    assignTierTransportError(
                        error,
                        "Residency wave preparation was not in flight");
                    return MoEOverlayResidencyWaveProgress::Failed;
                }

                const auto progress = inactive_bank_->pollPrepare(error);
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
                                   : "Inactive residency bank preparation failed";
                    abortStaged();
                    return progress;
                }

                phase_ = Phase::Prepared;
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

            /** @brief Begin the irreversible selector fan-out after preparation. */
            bool beginPublication(std::string *error) noexcept override
            {
                if (phase_ != Phase::Prepared)
                {
                    assignTierTransportError(
                        error,
                        "Residency publication requires one prepared wave");
                    return false;
                }
                /* Enter the irreversible phase before delegating the first
                 * selector submission. A partial fan-out is never abortable. */
                phase_ = Phase::Publishing;
                if (!inactive_bank_->beginPublication(error))
                    return false;
                return true;
            }

            /** @brief Poll inference-visible selector publication without waiting. */
            MoEOverlayResidencyWaveProgress pollPublication(
                std::string *error) noexcept override
            {
                if (phase_ == Phase::Published)
                    return MoEOverlayResidencyWaveProgress::Ready;
                if (phase_ != Phase::Publishing)
                {
                    assignTierTransportError(
                        error,
                        "Residency publication was not in flight");
                    return MoEOverlayResidencyWaveProgress::Failed;
                }

                const auto progress = inactive_bank_->pollPublication(error);
                if (progress == MoEOverlayResidencyWaveProgress::Ready)
                {
                    phase_ = Phase::Published;
                    PerfStatsCollector::addCounter(
                        "moe_overlay_residency",
                        "runtime_banks_published",
                        1.0,
                        "maintenance",
                        perf_device_,
                        {{"migrations", std::to_string(migration_count_)}});
                }
                return progress;
            }

            /** @brief Abort each transfer and the inactive-bank reservation once. */
            void abortStaged() noexcept override
            {
                if (phase_ == Phase::Aborting || phase_ == Phase::Aborted ||
                    phase_ == Phase::Retired)
                    return;
                if (phase_ == Phase::Publishing ||
                    phase_ == Phase::Published ||
                    phase_ == Phase::RetirementFencing ||
                    phase_ == Phase::RetirementReady)
                {
                    std::terminate();
                }
                phase_ = Phase::Aborting;
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
                if (phase_ == Phase::Aborted)
                    return MoEOverlayResidencyWaveProgress::Ready;
                if (phase_ != Phase::Aborting)
                {
                    assignTierTransportError(
                        error,
                        "Cannot poll abort cleanup before abortStaged()");
                    return MoEOverlayResidencyWaveProgress::Failed;
                }

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

                /* Abort is a drain, not cancellation of already-published GPU
                 * commands. Keep the same exact progress authority alive until
                 * every lane releases its completion fence. */
                if (!submitTransferProgress(error))
                    return MoEOverlayResidencyWaveProgress::Failed;

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

                const bool abort_cleanup_ready =
                    inactive_bank_abort_ready_ &&
                    std::all_of(
                        transfer_abort_ready_.begin(),
                        transfer_abort_ready_.end(),
                        [](bool ready) { return ready; });
                if (!abort_cleanup_ready || any_pending)
                {
                    stats_->abort_pending_polls.fetch_add(
                        1,
                        std::memory_order_relaxed);
                    return MoEOverlayResidencyWaveProgress::Pending;
                }

                stats_->abort_cleanups_completed.fetch_add(
                    1,
                    std::memory_order_relaxed);
                phase_ = Phase::Aborted;
                PerfStatsCollector::addCounter(
                    "moe_overlay_residency",
                    "composite_abort_cleanup_ready",
                    1.0,
                    "maintenance",
                    perf_device_,
                    {{"migrations", std::to_string(migration_count_)}});
                return MoEOverlayResidencyWaveProgress::Ready;
            }

            /** @brief Poll device reader drainage before cross-rank retirement. */
            MoEOverlayResidencyWaveProgress pollRetirementFence(
                std::string *error) noexcept override
            {
                if (phase_ == Phase::RetirementReady)
                    return MoEOverlayResidencyWaveProgress::Ready;
                if (phase_ != Phase::Published &&
                    phase_ != Phase::RetirementFencing)
                {
                    assignTierTransportError(
                        error,
                        "Cannot retire a residency wave before publication completes");
                    return MoEOverlayResidencyWaveProgress::Failed;
                }
                phase_ = Phase::RetirementFencing;
                const auto progress =
                    inactive_bank_->pollRetirementFence(error);
                if (progress == MoEOverlayResidencyWaveProgress::Ready)
                    phase_ = Phase::RetirementReady;
                return progress;
            }

            /** @brief Delegate old-bank retirement after the authority drains leases. */
            void retirePrevious() noexcept override
            {
                if (phase_ == Phase::Retired)
                    return;
                if (phase_ != Phase::RetirementReady)
                    std::terminate();
                inactive_bank_->retirePrevious();
                phase_ = Phase::Retired;
                stats_->old_banks_retired.fetch_add(
                    1,
                    std::memory_order_relaxed);
            }

        private:
            /** @brief Enqueue one device-deduplicated maintenance progress pass. */
            [[nodiscard]] bool submitTransferProgress(
                std::string *error) noexcept
            {
                if (!transfer_progress_authority_)
                    return true;
                return transfer_progress_authority_
                    ->submitOutstandingTransferProgress(error);
            }

            std::vector<std::unique_ptr<IMoEOverlayTierTransferOperation>>
                transfers_;
            std::vector<bool> transfer_ready_;
            std::vector<bool> transfer_abort_ready_;
            std::unique_ptr<IMoEOverlayInactiveBankTransaction> inactive_bank_;
            std::shared_ptr<MoEOverlayTierMigrationTransportSharedStats> stats_;
            std::string perf_device_;
            std::size_t migration_count_ = 0;
            std::shared_ptr<IMoEOverlayTransferProgressAuthority>
                transfer_progress_authority_;
            std::vector<MoEOverlayCompletedMigrationMeasurement>
                measurements_;
            std::shared_ptr<IMoEOverlayMigrationMeasurementSink>
                measurement_sink_;
            bool require_complete_local_measurements_ = false;
            std::chrono::steady_clock::time_point wave_started_at_{};
            MoEOverlayResidencyWaveInterval stage_interval_{};
            bool publication_allowed_ = true;
            bool inactive_bank_abort_ready_ = false;
            Phase phase_ = Phase::Reserved;
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
                    transaction.migrations.size(),
                    config_.transfer_progress_authority);
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
                    transaction.migrations.size(),
                    config_.transfer_progress_authority);
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
                config_.transfer_progress_authority,
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
