/**
 * @file Test__MoEOverlayTierMigrationTransport.cpp
 * @brief Device-free adversarial tests for composite arbitrary-tier waves.
 *
 * These tests lock down the background protocol independently of CUDA, ROCm,
 * MPI, and prepared-weight implementations: every projection progresses, commit
 * starts only after all projections are ready, publication remains atomic, old
 * banks wait for ticket retirement, malformed reservations abort, and failures
 * never expose the candidate epoch.
 */

#include "execution/moe/MoEOverlayTierMigrationTransport.h"
#include "execution/moe/MoEOverlayEconomyCalibrationPlanner.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        /** @brief Shared observations retained after wave objects are destroyed. */
        struct ProtocolObservations
        {
            std::size_t prepared_migrations = 0;
            std::size_t transfer_polls = 0;
            std::size_t transfer_aborts = 0;
            std::size_t transfer_abort_polls = 0;
            std::size_t transfers_destroyed = 0;
            std::size_t bank_begin_commits = 0;
            std::size_t bank_commit_polls = 0;
            std::size_t bank_aborts = 0;
            std::size_t bank_abort_polls = 0;
            std::size_t banks_destroyed = 0;
            std::size_t bank_retires = 0;
            std::size_t measurement_sink_calls = 0;
            std::size_t bank_commits_seen_by_measurement_sink = 0;
        };

        /** @brief Device-free operation that can be pending once or fail. */
        class ScriptedTransfer final : public IMoEOverlayTierTransferOperation
        {
        public:
            /** @brief Bind shared observations and the requested terminal script. */
            ScriptedTransfer(
                std::shared_ptr<ProtocolObservations> observations,
                bool fail,
                bool abort_pending_once = false,
                std::optional<ExpertTierProjectionTransferMeasurement>
                    measurement = std::nullopt)
                : observations_(std::move(observations)),
                  fail_(fail),
                  abort_pending_once_(abort_pending_once),
                  measurement_(std::move(measurement))
            {
            }

            /** @brief Record destruction so tests can reject premature recycling. */
            ~ScriptedTransfer() override
            {
                ++observations_->transfers_destroyed;
            }

            /** @brief Return Pending once, then the configured terminal state. */
            MoEOverlayResidencyWaveProgress poll(
                std::string *error) noexcept override
            {
                ++observations_->transfer_polls;
                if (!pending_observed_)
                {
                    pending_observed_ = true;
                    return MoEOverlayResidencyWaveProgress::Pending;
                }
                if (fail_)
                {
                    if (error)
                        *error = "scripted projection failure";
                    return MoEOverlayResidencyWaveProgress::Failed;
                }
                ready_ = true;
                return MoEOverlayResidencyWaveProgress::Ready;
            }

            /** @brief Record exactly one abort request. */
            void abort() noexcept override
            {
                if (!aborted_)
                {
                    aborted_ = true;
                    ++observations_->transfer_aborts;
                }
            }

            /** @brief Optionally expose one pending cleanup event after abort. */
            MoEOverlayResidencyWaveProgress pollAbort(
                std::string *error) noexcept override
            {
                ++observations_->transfer_abort_polls;
                if (!aborted_)
                {
                    if (error)
                        *error = "transfer cleanup polled before abort";
                    return MoEOverlayResidencyWaveProgress::Failed;
                }
                if (abort_pending_once_ && !abort_pending_observed_)
                {
                    abort_pending_observed_ = true;
                    return MoEOverlayResidencyWaveProgress::Pending;
                }
                return MoEOverlayResidencyWaveProgress::Ready;
            }

            /** @brief Return scripted evidence only after normal readiness. */
            [[nodiscard]] std::optional<
                ExpertTierProjectionTransferMeasurement>
            completedMeasurement() const noexcept override
            {
                return ready_ ? measurement_ : std::nullopt;
            }

        private:
            std::shared_ptr<ProtocolObservations> observations_;
            bool fail_ = false;
            bool abort_pending_once_ = false;
            bool pending_observed_ = false;
            bool aborted_ = false;
            bool abort_pending_observed_ = false;
            bool ready_ = false;
            std::optional<ExpertTierProjectionTransferMeasurement>
                measurement_;
        };

        /** @brief Device-free sink proving evidence ordering and rejection. */
        class RecordingMeasurementSink final
            : public IMoEOverlayMigrationMeasurementSink
        {
        public:
            /** @brief Bind protocol observations and choose acceptance policy. */
            RecordingMeasurementSink(
                std::shared_ptr<ProtocolObservations> observations,
                bool accept = true)
                : observations_(std::move(observations)), accept_(accept)
            {
            }

            /** @brief Copy pointer-free evidence without waiting. */
            bool recordCompletedWave(
                const std::vector<MoEOverlayCompletedMigrationMeasurement> &
                    measurements,
                std::string *error) noexcept override
            {
                ++observations_->measurement_sink_calls;
                observations_->bank_commits_seen_by_measurement_sink =
                    observations_->bank_begin_commits;
                recorded = measurements;
                if (!accept_)
                {
                    if (error)
                        *error = "scripted measurement rejection";
                    return false;
                }
                return true;
            }

            std::vector<MoEOverlayCompletedMigrationMeasurement> recorded;

        private:
            std::shared_ptr<ProtocolObservations> observations_;
            bool accept_ = true;
        };

        /** @brief Device-free inactive bank with one pending commit poll. */
        class ScriptedInactiveBank final
            : public IMoEOverlayInactiveBankTransaction
        {
        public:
            /** @brief Bind observations retained beyond this bank's lifetime. */
            explicit ScriptedInactiveBank(
                std::shared_ptr<ProtocolObservations> observations)
                : observations_(std::move(observations))
            {
            }

            /** @brief Record destruction so tests can reject premature recycling. */
            ~ScriptedInactiveBank() override
            {
                ++observations_->banks_destroyed;
            }

            /** @brief Record the unique inactive-bank build request. */
            bool beginCommit(std::string *) noexcept override
            {
                ++observations_->bank_begin_commits;
                begun_ = true;
                return true;
            }

            /** @brief Return Pending once, then Ready after beginCommit(). */
            MoEOverlayResidencyWaveProgress pollCommit(
                std::string *error) noexcept override
            {
                ++observations_->bank_commit_polls;
                if (!begun_)
                {
                    if (error)
                        *error = "commit polled before begin";
                    return MoEOverlayResidencyWaveProgress::Failed;
                }
                if (!pending_observed_)
                {
                    pending_observed_ = true;
                    return MoEOverlayResidencyWaveProgress::Pending;
                }
                return MoEOverlayResidencyWaveProgress::Ready;
            }

            /** @brief Record one unpublished-bank abort. */
            void abort() noexcept override
            {
                if (!aborted_)
                {
                    aborted_ = true;
                    ++observations_->bank_aborts;
                }
            }

            /** @brief Device-free bank aborts are immediately safe to reclaim. */
            MoEOverlayResidencyWaveProgress pollAbort(
                std::string *error) noexcept override
            {
                ++observations_->bank_abort_polls;
                if (!aborted_)
                {
                    if (error)
                        *error = "bank cleanup polled before abort";
                    return MoEOverlayResidencyWaveProgress::Failed;
                }
                return MoEOverlayResidencyWaveProgress::Ready;
            }

            /** @brief Record one lease-safe previous-bank retirement. */
            void retirePrevious() noexcept override
            {
                if (!retired_)
                {
                    retired_ = true;
                    ++observations_->bank_retires;
                }
            }

        private:
            std::shared_ptr<ProtocolObservations> observations_;
            bool begun_ = false;
            bool pending_observed_ = false;
            bool aborted_ = false;
            bool retired_ = false;
        };

        /** @brief Factory producing exact, malformed, deferred, or failing waves. */
        class ScriptedWaveFactory final
            : public IMoEOverlayTierPreparedWaveFactory
        {
        public:
            enum class Mode
            {
                Exact,
                Malformed,
                Deferred,
                TransferFailure,
                TransferFailureWithPendingAbort,
                PartialPreparationFailureWithPendingAbort,
            };

            /** @brief Select a stable preparation script. */
            explicit ScriptedWaveFactory(
                Mode mode,
                bool include_measurements = false,
                std::optional<std::size_t> omitted_measurement = std::nullopt)
                : mode_(mode),
                  include_measurements_(include_measurements),
                  omitted_measurement_(omitted_measurement),
                  observations(std::make_shared<ProtocolObservations>())
            {
            }

            /** @brief Build three operations per migration unless scripted otherwise. */
            MoEOverlayTierPreparedWave prepare(
                const MoEOverlayResidencyTransaction &transaction) override
            {
                observations->prepared_migrations =
                    transaction.migrations.size();
                if (mode_ == Mode::Deferred)
                {
                    return {
                        .status =
                            MoEOverlayResidencyStageStartStatus::Deferred,
                        .error = "scripted shadow backpressure",
                    };
                }

                const std::size_t exact_count =
                    transaction.migrations.size() * 3;
                const std::size_t operation_count =
                    mode_ == Mode::Malformed
                        ? exact_count - 1
                        : (mode_ ==
                                   Mode::PartialPreparationFailureWithPendingAbort
                               ? 5
                               : exact_count);
                MoEOverlayTierPreparedWave wave;
                wave.status =
                    mode_ == Mode::PartialPreparationFailureWithPendingAbort
                        ? MoEOverlayResidencyStageStartStatus::Failed
                        : MoEOverlayResidencyStageStartStatus::Started;
                if (mode_ == Mode::PartialPreparationFailureWithPendingAbort)
                    wave.error = "scripted fifth-operation enqueue failure";
                wave.inactive_bank =
                    std::make_unique<ScriptedInactiveBank>(observations);
                for (std::size_t index = 0;
                     index < operation_count;
                     ++index)
                {
                    wave.transfers.push_back(
                        std::make_unique<ScriptedTransfer>(
                            observations,
                            (mode_ == Mode::TransferFailure ||
                             mode_ == Mode::TransferFailureWithPendingAbort) &&
                                index == 4,
                            (mode_ == Mode::TransferFailureWithPendingAbort ||
                             mode_ ==
                                 Mode::PartialPreparationFailureWithPendingAbort) &&
                                index == 4,
                            include_measurements_ &&
                                    (!omitted_measurement_ ||
                                     *omitted_measurement_ != index)
                                ? std::optional{
                                      ExpertTierProjectionTransferMeasurement{
                                          .sequence = index + 1,
                                          .bytes = 1024 + index,
                                          .wall_nanoseconds = 10'000 + index,
                                          .device_nanoseconds = 5'000 + index,
                                          .host_nanoseconds = index,
                                      }}
                                : std::nullopt));
                }
                return wave;
            }

            std::shared_ptr<ProtocolObservations> observations;

        private:
            Mode mode_;
            bool include_measurements_ = false;
            std::optional<std::size_t> omitted_measurement_;
        };

        /** @brief Construct one single-participant routed domain. */
        RoutedExpertDomain makeDomain(
            std::string name,
            GlobalDeviceAddress participant,
            int world_rank,
            CollectiveBackendType backend)
        {
            RoutedExpertDomain result;
            result.name = std::move(name);
            result.scope = ExecutionDomainScope::SINGLE;
            result.backend = backend;
            result.participants = {participant};
            result.world_ranks = {world_rank};
            result.owner_rank = world_rank;
            result.routed_compute_policy =
                RoutedExpertComputePolicy::Apportioned;
            return result;
        }

        /** @brief Construct one fixed-capacity tier. */
        RoutedExpertTier makeTier(
            std::string name,
            std::string domain,
            int priority,
            int capacity,
            bool fallback = false)
        {
            return {
                .name = std::move(name),
                .domain = std::move(domain),
                .priority = priority,
                .max_experts_per_layer = capacity,
                .fallback = fallback,
            };
        }

        /** @brief Construct CUDA-hot/ROCm-warm/CPU-cold one-slot tiers. */
        MoERoutedExpertPlacementPlan rotationPlan()
        {
            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan.continuation_domain = "cuda_hot";
            plan.shared_expert_domain = "cuda_hot";
            plan.residency_policy =
                RoutedExpertResidencyPolicy::RoutedTierRebalanced;
            plan.owner_order = RoutedExpertOwnerOrder::Ordinal;
            plan.domains = {
                makeDomain(
                    "cuda_hot",
                    GlobalDeviceAddress::cuda(0, 0),
                    0,
                    CollectiveBackendType::NCCL),
                makeDomain(
                    "rocm_warm",
                    GlobalDeviceAddress::rocm(1, 0),
                    1,
                    CollectiveBackendType::RCCL),
                makeDomain(
                    "cpu_cold",
                    GlobalDeviceAddress::cpu(2),
                    2,
                    CollectiveBackendType::MPI),
            };
            plan.routed_tiers = {
                makeTier("hot", "cuda_hot", 0, 1),
                makeTier("warm", "rocm_warm", 1, 1),
                makeTier("cold", "cpu_cold", 2, 0, true),
            };
            return plan;
        }

        /** @brief Construct metadata for one layer and three routed experts. */
        MoERoutedExpertModelMetadata rotationMetadata()
        {
            return {
                .num_layers = 1,
                .num_experts = 3,
                .d_model = 256,
                .routed_intermediate_size = 192,
                .routed_quant_type = "Q4_0",
            };
        }

        /** @brief Freeze evidence that rotates cold->hot->warm->cold. */
        std::unique_ptr<DecodeExpertHistogram> rotationHistogram()
        {
            DecodeExpertHistogramConfig config;
            config.num_layers = 1;
            config.num_experts = 3;
            config.top_k = 1;
            config.window_size = 3;
            config.sockets = {
                DeviceId::cuda(0),
                DeviceId::rocm(0),
                DeviceId::cpu(),
            };
            config.ownership = MoELayeredExpertOwnership::uniform(
                1,
                3,
                {0, 1, 2});
            auto histogram = std::make_unique<DecodeExpertHistogram>(config);
            const std::uint64_t counts[] = {80, 70, 100};
            histogram->mergeLayerCounts(0, counts, 3, false);
            return histogram;
        }

        /** @brief Bundle authority plus its histogram lifetime for one test. */
        struct AuthorityFixture
        {
            std::unique_ptr<DecodeExpertHistogram> histogram;
            std::unique_ptr<MoEOverlayResidencyAuthority> authority;
        };

        /** @brief Construct a live epoch-1 authority with a three-edge proposal. */
        AuthorityFixture makeAuthority()
        {
            AuthorityFixture fixture;
            fixture.histogram = rotationHistogram();
            fixture.authority =
                std::make_unique<MoEOverlayResidencyAuthority>(
                    MoEOverlayResidencyAuthority::Config{
                        .initial_plan = rotationPlan(),
                        .model_metadata = rotationMetadata(),
                        .maintenance_mode =
                            MoERebalanceRuntimeMode::Dynamic,
                        .histogram = fixture.histogram.get(),
                        .perf_device = "heterogeneous-test",
                    });
            return fixture;
        }
    } // namespace

    TEST(
        MoEOverlayTierMigrationTransport,
        AllNineProjectionsFinishBeforeCommitAndOldBankWaitsForTicket)
    {
        auto fixture = makeAuthority();
        auto old_ticket = fixture.authority->tryAcquireTicketSnapshot();
        ASSERT_TRUE(old_ticket.has_value());
        ASSERT_EQ((*old_ticket)->epoch, 1u);

        const auto transaction = fixture.authority->proposeFromHistogram();
        ASSERT_TRUE(transaction.valid());
        ASSERT_EQ(transaction.migrations.size(), 3u);
        ASSERT_EQ(transaction.migration_cycles.size(), 1u);

        ScriptedWaveFactory factory(ScriptedWaveFactory::Mode::Exact);
        MoEOverlayTierMigrationTransport transport({
            .factory = &factory,
            .projections_per_expert = 3,
            .perf_device = "heterogeneous-test",
        });
        ASSERT_EQ(
            fixture.authority->beginApply(transaction, transport).status,
            MoEOverlayResidencyApplyStatus::Started);

        EXPECT_EQ(
            fixture.authority->advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Staging);
        EXPECT_EQ(factory.observations->bank_begin_commits, 0u);
        EXPECT_EQ(fixture.authority->snapshot()->epoch, 1u);

        EXPECT_EQ(
            fixture.authority->advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Committing);
        EXPECT_EQ(factory.observations->bank_begin_commits, 1u);
        EXPECT_EQ(fixture.authority->snapshot()->epoch, 1u);

        EXPECT_EQ(
            fixture.authority->advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Committing);
        EXPECT_EQ(fixture.authority->snapshot()->epoch, 1u);

        EXPECT_EQ(
            fixture.authority->advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Committed);
        EXPECT_EQ(fixture.authority->snapshot()->epoch, 2u);
        EXPECT_EQ(factory.observations->bank_retires, 0u);
        EXPECT_EQ(fixture.authority->pendingRetirementCount(), 1u);

        const auto stats = transport.stats();
        EXPECT_EQ(stats.waves_started, 1u);
        EXPECT_EQ(stats.transfer_operations_started, 9u);
        EXPECT_EQ(stats.transfer_operations_completed, 9u);
        EXPECT_EQ(stats.commits_started, 1u);
        EXPECT_EQ(stats.commits_completed, 1u);
        EXPECT_EQ(stats.inference_stream_waits, 0u);
        EXPECT_EQ(stats.blocking_synchronizations, 0u);

        old_ticket.reset();
        EXPECT_EQ(
            fixture.authority->advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Idle);
        EXPECT_EQ(factory.observations->bank_retires, 1u);
        EXPECT_EQ(fixture.authority->pendingRetirementCount(), 0u);
    }

    TEST(
        MoEOverlayTierMigrationTransport,
        MalformedProjectionCardinalityAbortsBeforePublication)
    {
        auto fixture = makeAuthority();
        const auto transaction = fixture.authority->proposeFromHistogram();
        ScriptedWaveFactory factory(ScriptedWaveFactory::Mode::Malformed);
        MoEOverlayTierMigrationTransport transport({
            .factory = &factory,
            .projections_per_expert = 3,
        });

        const auto result = fixture.authority->beginApply(
            transaction,
            transport);
        EXPECT_EQ(result.status, MoEOverlayResidencyApplyStatus::StageFailed);
        EXPECT_EQ(fixture.authority->snapshot()->epoch, 1u);
        EXPECT_EQ(factory.observations->transfer_aborts, 8u);
        EXPECT_EQ(factory.observations->bank_aborts, 1u);
        EXPECT_EQ(transport.stats().waves_failed_to_prepare, 1u);
    }

    TEST(
        MoEOverlayTierMigrationTransport,
        CompleteMeasurementsAreRecordedBeforeInactiveBankCommit)
    {
        auto fixture = makeAuthority();
        const auto transaction = fixture.authority->proposeFromHistogram();
        ASSERT_TRUE(transaction.valid());
        ScriptedWaveFactory factory(
            ScriptedWaveFactory::Mode::Exact,
            true);
        auto sink = std::make_shared<RecordingMeasurementSink>(
            factory.observations);
        MoEOverlayTierMigrationTransport transport({
            .factory = &factory,
            .projections_per_expert = 3,
            .measurement_sink = sink,
            .require_complete_local_measurements = true,
        });

        ASSERT_EQ(
            fixture.authority->beginApply(transaction, transport).status,
            MoEOverlayResidencyApplyStatus::Started);
        EXPECT_EQ(
            fixture.authority->advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Staging);
        EXPECT_EQ(factory.observations->measurement_sink_calls, 0u);

        EXPECT_EQ(
            fixture.authority->advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Committing);
        ASSERT_EQ(factory.observations->measurement_sink_calls, 1u);
        EXPECT_EQ(
            factory.observations->bank_commits_seen_by_measurement_sink,
            0u);
        ASSERT_EQ(sink->recorded.size(), transaction.migrations.size());
        for (std::size_t migration = 0;
             migration < sink->recorded.size();
             ++migration)
        {
            const auto &measurement = sink->recorded[migration];
            EXPECT_TRUE(measurement.valid());
            EXPECT_EQ(measurement.expected_epoch, transaction.expected_epoch);
            EXPECT_EQ(measurement.candidate_epoch, transaction.candidate->epoch);
            EXPECT_EQ(
                measurement.source_participant,
                transaction.migrations[migration].source.owner_participant);
            EXPECT_EQ(
                measurement.destination_participant,
                transaction.migrations[migration]
                    .destination.owner_participant);
            for (std::size_t projection = 0; projection < 3; ++projection)
            {
                ASSERT_TRUE(measurement.projections[projection].has_value());
                EXPECT_EQ(
                    measurement.projections[projection]->sequence,
                    migration * 3 + projection + 1);
            }
        }
    }

    TEST(
        MoEOverlayTierMigrationTransport,
        RequiredMeasurementGapAbortsBeforeCommitOrPublication)
    {
        auto fixture = makeAuthority();
        const auto transaction = fixture.authority->proposeFromHistogram();
        ScriptedWaveFactory factory(
            ScriptedWaveFactory::Mode::Exact,
            true,
            4u);
        auto sink = std::make_shared<RecordingMeasurementSink>(
            factory.observations);
        MoEOverlayTierMigrationTransport transport({
            .factory = &factory,
            .projections_per_expert = 3,
            .measurement_sink = sink,
            .require_complete_local_measurements = true,
        });

        ASSERT_EQ(
            fixture.authority->beginApply(transaction, transport).status,
            MoEOverlayResidencyApplyStatus::Started);
        ASSERT_EQ(
            fixture.authority->advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Staging);
        const auto failed = fixture.authority->advanceBackground();
        EXPECT_EQ(failed.status, MoEOverlayResidencyApplyStatus::StageFailed);
        EXPECT_NE(failed.error.find("omitted exact timing evidence"),
                  std::string::npos);
        EXPECT_EQ(factory.observations->measurement_sink_calls, 0u);
        EXPECT_EQ(factory.observations->bank_begin_commits, 0u);
        EXPECT_EQ(fixture.authority->snapshot()->epoch, 1u);
    }

    TEST(
        MoEOverlayTierMigrationTransport,
        MeasurementSinkRejectionAbortsBeforeCommitOrPublication)
    {
        auto fixture = makeAuthority();
        const auto transaction = fixture.authority->proposeFromHistogram();
        ScriptedWaveFactory factory(
            ScriptedWaveFactory::Mode::Exact,
            true);
        auto sink = std::make_shared<RecordingMeasurementSink>(
            factory.observations,
            false);
        MoEOverlayTierMigrationTransport transport({
            .factory = &factory,
            .projections_per_expert = 3,
            .measurement_sink = sink,
            .require_complete_local_measurements = true,
        });

        ASSERT_EQ(
            fixture.authority->beginApply(transaction, transport).status,
            MoEOverlayResidencyApplyStatus::Started);
        ASSERT_EQ(
            fixture.authority->advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Staging);
        const auto failed = fixture.authority->advanceBackground();
        EXPECT_EQ(failed.status, MoEOverlayResidencyApplyStatus::StageFailed);
        EXPECT_EQ(failed.error, "scripted measurement rejection");
        EXPECT_EQ(factory.observations->measurement_sink_calls, 1u);
        EXPECT_EQ(factory.observations->bank_begin_commits, 0u);
        EXPECT_EQ(fixture.authority->snapshot()->epoch, 1u);
    }

    TEST(
        MoEOverlayTierMigrationTransport,
        ProjectionFailureAbortsWholeWaveWithoutPublishing)
    {
        auto fixture = makeAuthority();
        const auto transaction = fixture.authority->proposeFromHistogram();
        ScriptedWaveFactory factory(
            ScriptedWaveFactory::Mode::TransferFailure);
        MoEOverlayTierMigrationTransport transport({
            .factory = &factory,
            .projections_per_expert = 3,
        });
        ASSERT_EQ(
            fixture.authority->beginApply(transaction, transport).status,
            MoEOverlayResidencyApplyStatus::Started);
        EXPECT_EQ(
            fixture.authority->advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Staging);

        const auto failed = fixture.authority->advanceBackground();
        EXPECT_EQ(failed.status, MoEOverlayResidencyApplyStatus::StageFailed);
        EXPECT_EQ(failed.error, "scripted projection failure");
        EXPECT_EQ(fixture.authority->snapshot()->epoch, 1u);
        EXPECT_EQ(factory.observations->transfer_aborts, 9u);
        EXPECT_EQ(factory.observations->bank_aborts, 1u);
        EXPECT_EQ(transport.stats().waves_aborted, 1u);
        EXPECT_EQ(fixture.authority->pendingAbortCount(), 0u);
        EXPECT_EQ(fixture.authority->stats().aborted_waves_reaped, 1u);
    }

    TEST(
        MoEOverlayTierMigrationTransport,
        FailedWaveRetainsEveryResourceUntilAbortEventsQuiesce)
    {
        auto fixture = makeAuthority();
        const auto transaction = fixture.authority->proposeFromHistogram();
        ScriptedWaveFactory factory(
            ScriptedWaveFactory::Mode::TransferFailureWithPendingAbort);
        MoEOverlayTierMigrationTransport transport({
            .factory = &factory,
            .projections_per_expert = 3,
        });
        ASSERT_EQ(
            fixture.authority->beginApply(transaction, transport).status,
            MoEOverlayResidencyApplyStatus::Started);
        ASSERT_EQ(
            fixture.authority->advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Staging);

        const auto failed = fixture.authority->advanceBackground();
        ASSERT_EQ(failed.status, MoEOverlayResidencyApplyStatus::StageFailed);
        EXPECT_EQ(fixture.authority->snapshot()->epoch, 1u);
        EXPECT_EQ(fixture.authority->pendingAbortCount(), 1u);
        EXPECT_EQ(factory.observations->transfer_aborts, 9u);
        EXPECT_EQ(factory.observations->bank_aborts, 1u);
        EXPECT_EQ(factory.observations->transfers_destroyed, 0u);
        EXPECT_EQ(factory.observations->banks_destroyed, 0u);
        EXPECT_EQ(transport.stats().abort_pending_polls, 1u);

        /* The maintenance pass observes the final event before destruction. */
        EXPECT_EQ(
            fixture.authority->advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Idle);
        EXPECT_EQ(fixture.authority->pendingAbortCount(), 0u);
        EXPECT_EQ(factory.observations->transfers_destroyed, 9u);
        EXPECT_EQ(factory.observations->banks_destroyed, 1u);
        EXPECT_EQ(transport.stats().abort_cleanups_completed, 1u);
        EXPECT_EQ(fixture.authority->stats().aborted_waves_reaped, 1u);
    }

    TEST(
        MoEOverlayTierMigrationTransport,
        PartialPreparationFailureRetainsAlreadyEnqueuedPrefixUntilQuiescent)
    {
        auto fixture = makeAuthority();
        const auto transaction = fixture.authority->proposeFromHistogram();
        ScriptedWaveFactory factory(
            ScriptedWaveFactory::Mode::PartialPreparationFailureWithPendingAbort);
        MoEOverlayTierMigrationTransport transport({
            .factory = &factory,
            .projections_per_expert = 3,
        });

        const auto failed = fixture.authority->beginApply(
            transaction,
            transport);
        ASSERT_EQ(failed.status, MoEOverlayResidencyApplyStatus::StageFailed);
        EXPECT_EQ(failed.error, "scripted fifth-operation enqueue failure");
        EXPECT_EQ(fixture.authority->snapshot()->epoch, 1u);
        EXPECT_EQ(fixture.authority->pendingAbortCount(), 1u);
        EXPECT_EQ(factory.observations->transfer_aborts, 5u);
        EXPECT_EQ(factory.observations->bank_aborts, 1u);
        EXPECT_EQ(factory.observations->transfers_destroyed, 0u);
        EXPECT_EQ(factory.observations->banks_destroyed, 0u);

        EXPECT_EQ(
            fixture.authority->advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Idle);
        EXPECT_EQ(fixture.authority->pendingAbortCount(), 0u);
        EXPECT_EQ(factory.observations->transfers_destroyed, 5u);
        EXPECT_EQ(factory.observations->banks_destroyed, 1u);
        EXPECT_EQ(transport.stats().waves_failed_to_prepare, 1u);
        EXPECT_EQ(transport.stats().abort_cleanups_completed, 1u);
    }

    TEST(
        MoEOverlayTierMigrationTransport,
        ShadowBackpressureDefersWithoutCreatingPhysicalOperations)
    {
        auto fixture = makeAuthority();
        const auto transaction = fixture.authority->proposeFromHistogram();
        ScriptedWaveFactory factory(ScriptedWaveFactory::Mode::Deferred);
        MoEOverlayTierMigrationTransport transport({
            .factory = &factory,
            .projections_per_expert = 3,
        });

        const auto result = fixture.authority->beginApply(
            transaction,
            transport);
        EXPECT_EQ(result.status, MoEOverlayResidencyApplyStatus::Deferred);
        EXPECT_EQ(fixture.authority->snapshot()->epoch, 1u);
        EXPECT_EQ(factory.observations->transfer_polls, 0u);
        EXPECT_EQ(transport.stats().waves_deferred, 1u);
    }

    TEST(
        MoEOverlayTierMigrationTransport,
        CalibrationPlannerCoversEveryDirectedEdgeAndCommitIsImpossible)
    {
        auto fixture = makeAuthority();
        const auto live = fixture.authority->snapshot();
        ASSERT_TRUE(live && live->valid());
        MoEOverlayEconomyCalibrationPlanner planner({
            .live_snapshot = live,
            .complete_expert_bytes_per_layer = {3'072},
        });
        ASSERT_EQ(planner.requiredCoordinates().size(), 6u);
        EXPECT_EQ(
            planner.requiredCoordinates().front(),
            (MoEOverlayMigrationMeasurementCoordinate{
                .source_participant = 0,
                .destination_participant = 1,
                .layer = 0,
            }));
        EXPECT_EQ(
            planner.requiredCoordinates().back(),
            (MoEOverlayMigrationMeasurementCoordinate{
                .source_participant = 2,
                .destination_participant = 1,
                .layer = 0,
            }));

        const auto calibration = planner.buildPairSwap(0, 2, 0, 17);
        ASSERT_TRUE(calibration.valid());
        EXPECT_EQ(
            calibration.purpose,
            MoEOverlayResidencyTransactionPurpose::EconomyCalibration);
        EXPECT_EQ(calibration.calibration_sequence, 17u);
        ASSERT_EQ(calibration.migrations.size(), 2u);
        EXPECT_EQ(
            calibration.migrations[0].direction,
            MoEOverlayTierMigrationDirection::Demotion);
        EXPECT_EQ(
            calibration.migrations[1].direction,
            MoEOverlayTierMigrationDirection::Promotion);
        EXPECT_EQ(
            calibration.previous->layered_ownership,
            calibration.candidate->layered_ownership);

        ScriptedWaveFactory factory(ScriptedWaveFactory::Mode::Exact);
        MoEOverlayTierMigrationTransport transport({
            .factory = &factory,
            .projections_per_expert = 3,
        });

        /* The publication authority rejects calibration before transport use. */
        const auto rejected =
            fixture.authority->beginApply(calibration, transport);
        EXPECT_EQ(rejected.status, MoEOverlayResidencyApplyStatus::Stale);
        EXPECT_NE(rejected.error.find("cannot enter residency publication"),
                  std::string::npos);

        auto start = transport.beginStage(calibration);
        ASSERT_EQ(
            start.status,
            MoEOverlayResidencyStageStartStatus::Started)
            << start.error;
        ASSERT_NE(start.wave, nullptr);
        std::string error;
        EXPECT_EQ(
            start.wave->pollStage(&error),
            MoEOverlayResidencyWaveProgress::Pending)
            << error;
        EXPECT_EQ(
            start.wave->pollStage(&error),
            MoEOverlayResidencyWaveProgress::Ready)
            << error;
        const auto interval = start.wave->completedStageInterval();
        ASSERT_TRUE(interval.has_value());
        EXPECT_TRUE(interval->valid());
        EXPECT_FALSE(start.wave->beginCommit(&error));
        EXPECT_NE(error.find("calibration waves cannot commit"),
                  std::string::npos);
        start.wave->abortStaged();
        EXPECT_EQ(
            start.wave->pollAbort(&error),
            MoEOverlayResidencyWaveProgress::Ready)
            << error;
    }
} // namespace llaminar2::test
