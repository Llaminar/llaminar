/**
 * @file Test__MoEOverlayEconomyCalibrationController.cpp
 * @brief Device-free proof of non-blocking live-path economy calibration.
 */

#include "execution/moe/MoEOverlayEconomyCalibrationController.h"
#include "execution/moe/MoEOverlayEconomyCertificationController.h"
#include "execution/moe/MoEOverlayEconomyEvidenceExchange.h"
#include "execution/moe/MoEOverlayTierMigrationTransport.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        /** @brief Shared proof that calibration aborted and never committed. */
        struct PhysicalObservations
        {
            std::uint64_t waves_prepared = 0;
            std::uint64_t projection_polls = 0;
            std::uint64_t projection_aborts = 0;
            std::uint64_t bank_aborts = 0;
            std::uint64_t bank_commits = 0;
        };

        /** @brief Immediately-ready measured operation with explicit cleanup. */
        class MeasuredOperation final
            : public IMoEOverlayTierTransferOperation
        {
        public:
            /** @brief Retain fixed evidence and shared lifecycle observations. */
            MeasuredOperation(
                std::shared_ptr<PhysicalObservations> observations,
                ExpertTierProjectionTransferMeasurement measurement)
                : observations_(std::move(observations)),
                  measurement_(measurement)
            {
            }

            /** @brief Publish readiness without waiting or sleeping. */
            MoEOverlayResidencyWaveProgress poll(
                std::string *) noexcept override
            {
                ++observations_->projection_polls;
                ready_ = true;
                return MoEOverlayResidencyWaveProgress::Ready;
            }

            /** @brief Mark this unpublished projection for recycling. */
            void abort() noexcept override
            {
                if (!aborted_)
                {
                    aborted_ = true;
                    ++observations_->projection_aborts;
                }
            }

            /** @brief Device-free cleanup is immediately complete. */
            MoEOverlayResidencyWaveProgress pollAbort(
                std::string *) noexcept override
            {
                return aborted_ ? MoEOverlayResidencyWaveProgress::Ready
                                : MoEOverlayResidencyWaveProgress::Failed;
            }

            /** @brief Return exact evidence only after readiness publication. */
            std::optional<ExpertTierProjectionTransferMeasurement>
            completedMeasurement() const noexcept override
            {
                return ready_ ? std::optional{measurement_} : std::nullopt;
            }

        private:
            std::shared_ptr<PhysicalObservations> observations_;
            ExpertTierProjectionTransferMeasurement measurement_;
            bool ready_ = false;
            bool aborted_ = false;
        };

        /** @brief Inactive-bank reservation that rejects any commit attempt. */
        class CalibrationBank final
            : public IMoEOverlayInactiveBankTransaction
        {
        public:
            /** @brief Retain lifecycle observations beyond wave destruction. */
            explicit CalibrationBank(
                std::shared_ptr<PhysicalObservations> observations)
                : observations_(std::move(observations))
            {
            }

            /** @brief Record the forbidden operation if controller regresses. */
            bool beginCommit(std::string *) noexcept override
            {
                ++observations_->bank_commits;
                return false;
            }

            /** @brief Commit polling is unreachable for calibration waves. */
            MoEOverlayResidencyWaveProgress pollCommit(
                std::string *) noexcept override
            {
                return MoEOverlayResidencyWaveProgress::Failed;
            }

            /** @brief Release the unpublished destination reservation. */
            void abort() noexcept override
            {
                if (!aborted_)
                {
                    aborted_ = true;
                    ++observations_->bank_aborts;
                }
            }

            /** @brief Device-free cleanup is immediately complete. */
            MoEOverlayResidencyWaveProgress pollAbort(
                std::string *) noexcept override
            {
                return aborted_ ? MoEOverlayResidencyWaveProgress::Ready
                                : MoEOverlayResidencyWaveProgress::Failed;
            }

            /** @brief Calibration never owns a published previous bank. */
            void retirePrevious() noexcept override {}

        private:
            std::shared_ptr<PhysicalObservations> observations_;
            bool aborted_ = false;
        };

        /** @brief Produce six complete measured operations for each pair swap. */
        class CalibrationFactory final
            : public IMoEOverlayTierPreparedWaveFactory
        {
        public:
            CalibrationFactory()
                : observations(std::make_shared<PhysicalObservations>())
            {
            }

            /** @brief Allocate device-free operation objects on maintenance. */
            MoEOverlayTierPreparedWave prepare(
                const MoEOverlayResidencyTransaction &transaction) override
            {
                ++observations->waves_prepared;
                MoEOverlayTierPreparedWave result;
                result.status = MoEOverlayResidencyStageStartStatus::Started;
                result.inactive_bank =
                    std::make_unique<CalibrationBank>(observations);
                const std::size_t operation_count =
                    transaction.migrations.size() * 3u;
                for (std::size_t index = 0; index < operation_count; ++index)
                {
                    const std::size_t projection = index % 3u;
                    result.transfers.push_back(
                        std::make_unique<MeasuredOperation>(
                            observations,
                            ExpertTierProjectionTransferMeasurement{
                                .sequence = observations->waves_prepared,
                                .bytes = 1'024u + projection,
                                .wall_nanoseconds = 100u + projection,
                                .device_nanoseconds = 80u + projection,
                            }));
                }
                return result;
            }

            std::shared_ptr<PhysicalObservations> observations;
        };

        /** @brief Construct one rank-local routed domain for planner geometry. */
        RoutedExpertDomain domain(
            std::string name,
            GlobalDeviceAddress address,
            CollectiveBackendType backend)
        {
            RoutedExpertDomain result;
            result.name = std::move(name);
            result.scope = ExecutionDomainScope::SINGLE;
            result.backend = backend;
            result.participants = {address};
            result.world_ranks = {0};
            result.owner_rank = 0;
            result.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
            return result;
        }

        /** @brief Build two opaque integer-priority tiers with one expert each. */
        std::shared_ptr<const MoEOverlayResidencySnapshot> snapshot(
            int layer_count = 1)
        {
            auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
            plan->enabled = true;
            plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan->continuation_domain = "tier_a_domain";
            plan->shared_expert_domain = "tier_a_domain";
            plan->residency_policy =
                RoutedExpertResidencyPolicy::RoutedTierRebalanced;
            plan->owner_order = RoutedExpertOwnerOrder::Ordinal;
            plan->domains = {
                domain(
                    "tier_a_domain",
                    GlobalDeviceAddress::cuda(0, 0),
                    CollectiveBackendType::NCCL),
                domain(
                    "tier_b_domain",
                    GlobalDeviceAddress::cpu(0),
                    CollectiveBackendType::MPI),
            };
            plan->routed_tiers = {
                {.name = "tier_a",
                 .domain = "tier_a_domain",
                 .priority = -4,
                 .max_experts_per_layer = 1},
                {.name = "tier_b",
                 .domain = "tier_b_domain",
                 .priority = 9,
                 .max_experts_per_layer = 1,
                 .fallback = true},
            };
            for (int layer = 0; layer < layer_count; ++layer)
            {
                plan->placements.push_back({
                    .layer = layer,
                    .routed_expert_tier = {0, 1},
                });
            }

            auto result = std::make_shared<MoEOverlayResidencySnapshot>();
            result->epoch = 1;
            result->placement_plan = plan;
            result->owner_map = MoEExpertOwnerMap::build(*plan);
            result->layered_ownership =
                result->owner_map.layeredOwnership(layer_count, 2);
            if (!result->valid())
                throw std::logic_error("test snapshot is invalid");
            return result;
        }

        /** @brief Build two participants whose rank ownership is explicit. */
        MoEExpertOwnerMap distributedOwnerMap(int layer_count = 2)
        {
            auto first = domain(
                "first_domain",
                GlobalDeviceAddress::cuda(0, 0),
                CollectiveBackendType::NCCL);
            first.world_ranks = {0};
            first.owner_rank = 0;
            auto second = domain(
                "second_domain",
                GlobalDeviceAddress::rocm(1, 0),
                CollectiveBackendType::RCCL);
            second.world_ranks = {1};
            second.owner_rank = 1;

            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan.continuation_domain = first.name;
            plan.shared_expert_domain = first.name;
            plan.residency_policy =
                RoutedExpertResidencyPolicy::RoutedTierRebalanced;
            plan.owner_order = RoutedExpertOwnerOrder::Ordinal;
            plan.domains = {std::move(first), std::move(second)};
            plan.routed_tiers = {
                {.name = "first",
                 .domain = "first_domain",
                 .priority = -11,
                 .max_experts_per_layer = 1},
                {.name = "second",
                 .domain = "second_domain",
                 .priority = 23,
                 .max_experts_per_layer = 1,
                 .fallback = true},
            };
            for (int layer = 0; layer < layer_count; ++layer)
            {
                plan.placements.push_back({
                    .layer = layer,
                    .routed_expert_tier = {0, 1},
                });
            }
            return MoEExpertOwnerMap::build(plan);
        }

        /** @brief Build one valid partial physical row for rank-wise merge. */
        MoEOverlayCompletedMigrationMeasurement partialMigration(
            int source,
            int destination,
            int expert,
            std::initializer_list<std::size_t> present_projections,
            std::uint64_t rank_bias)
        {
            MoEOverlayCompletedMigrationMeasurement row{
                .expected_epoch = 7,
                .candidate_epoch = 8,
                .source_participant = source,
                .destination_participant = destination,
                .layer = 0,
                .expert = expert,
                .wave_wall_nanoseconds = 500 + rank_bias,
            };
            for (const std::size_t projection : present_projections)
            {
                row.projections.at(projection) = {
                    .sequence = 10 + rank_bias + projection,
                    .bytes = 4'096 + projection,
                    .wall_nanoseconds = 100 + rank_bias + projection,
                    .device_nanoseconds = 80 + rank_bias + projection,
                };
            }
            if (!row.valid())
                throw std::logic_error("test partial migration is invalid");
            return row;
        }

        /** @brief Build sampled prepared-expert service totals for one cell. */
        MoEOverlayParticipantLayerServiceTotals serviceRow(
            int participant,
            int layer,
            std::uint64_t bias)
        {
            MoEOverlayParticipantLayerServiceTotals row{
                .participant_id = participant,
                .layer = layer,
            };
            for (std::size_t phase = 0;
                 phase < kExpertHistogramProductionSourceCount;
                 ++phase)
            {
                row.total_nanoseconds[phase] = bias + phase + 10;
                row.activation_count[phase] = phase + 1;
                row.sample_count[phase] = 3;
            }
            if (!row.valid())
                throw std::logic_error("test service row is invalid");
            return row;
        }

        /** @brief Build one synchronized rank-local calibration outcome. */
        MoEOverlayCalibrationAttemptEvidence attemptEvidence(
            int rank,
            bool exact_overlap = true)
        {
            const auto source = ExpertHistogramSource::GroupedVerifier;
            const MoEOverlayInferenceWorkloadIdentity exact_workload{
                .source = source,
                .real_rows = 3,
                .execution_rows = 3,
                .transaction_count = 1,
                .speculative_depth = 2,
                .schedule_fingerprint = 902,
            };
            return {
                .calibration_sequence = 19,
                .coordinate = {
                    .source_participant = 0,
                    .destination_participant = 1,
                    .layer = 0,
                },
                .source = source,
                .workload = exact_workload,
                .baseline_nanoseconds = rank == 0 ? 100u : 200u,
                .concurrent_nanoseconds = exact_overlap
                                              ? (rank == 0 ? 140u : 260u)
                                              : 0u,
                .exact_overlap = exact_overlap,
                .local_measurements = {
                    partialMigration(
                        0,
                        1,
                        0,
                        rank == 0
                            ? std::initializer_list<std::size_t>{0}
                            : std::initializer_list<std::size_t>{1, 2},
                        static_cast<std::uint64_t>(rank)),
                    partialMigration(
                        1,
                        0,
                        1,
                        rank == 0
                            ? std::initializer_list<std::size_t>{0}
                            : std::initializer_list<std::size_t>{1, 2},
                        static_cast<std::uint64_t>(rank)),
                },
            };
        }

        /** @brief Construct one projection contract for catalog tests. */
        MoEOverlayProjectionWeightManifest projectionManifest(
            ExpertTierWeightProjection role,
            int N,
            int K,
            const NativeVnniFormatInfo &format)
        {
            return {
                .projection = role,
                .N = N,
                .K = K,
                .format = ExpertWeightFormat::nativeVnni({
                    .codebook_id = format.codebook_id,
                    .is_superblock = format.is_superblock,
                    .present = true,
                }),
            };
        }

        /** @brief Construct one complete layer with an exact geometry signature. */
        MoEOverlayLayerWeightManifest layerManifest(
            int layer,
            int width,
            const NativeVnniFormatInfo &format)
        {
            return {
                .layer_idx = layer,
                .projections = {{
                    projectionManifest(
                        ExpertTierWeightProjection::Gate,
                        width,
                        64,
                        format),
                    projectionManifest(
                        ExpertTierWeightProjection::Up,
                        width,
                        64,
                        format),
                    projectionManifest(
                        ExpertTierWeightProjection::Down,
                        64,
                        width,
                        format),
                }},
            };
        }

        /** @brief Complete robust row suitable for catalog expansion. */
        MoEOverlayParticipantLayerMigrationMeasurement migrationRow(
            int source,
            int destination,
            int layer)
        {
            MoEOverlayParticipantLayerMigrationMeasurement row;
            row.source_participant = source;
            row.destination_participant = destination;
            row.layer = layer;
            for (std::size_t projection = 0;
                 projection < row.projections.size();
                 ++projection)
            {
                row.projections[projection] = {
                    .sequence = 1u + projection,
                    .bytes = 1'024u + projection,
                    .wall_nanoseconds = 100u + projection,
                    .device_nanoseconds = 80u + projection,
                };
            }
            row.wave_wall_nanoseconds = 200;
            row.wave_sample_count = 3;
            row.inference_interference_nanoseconds = 7;
            row.interference_sample_count = 3;
            return row;
        }

        /** @brief Exact workload emitted for each possible production phase. */
        MoEOverlayInferenceWorkloadIdentity workload(
            ExpertHistogramSource source)
        {
            const int depth =
                source == ExpertHistogramSource::GroupedVerifier ? 2 : 0;
            const int rows = source == ExpertHistogramSource::PrefillChunk
                                 ? 16
                                 : depth + 1;
            return {
                .source = source,
                .real_rows = rows,
                .execution_rows = rows,
                .transaction_count = 1,
                .speculative_depth = depth,
                .schedule_fingerprint =
                    900u + static_cast<std::uint64_t>(source),
            };
        }

        /**
         * @brief Let every phase claim while maintenance progresses concurrently.
         * @param probe Lock-free live-inference timing authority.
         * @param poll_maintenance Bounded maintenance poll performed while the
         *        winning inference ticket remains live.
         */
        template <typename PollMaintenance>
        void runOneInferenceInvocationPerPhase(
            MoEOverlayInferenceInterferenceProbe &probe,
            PollMaintenance poll_maintenance)
        {
            for (const auto source : {
                     ExpertHistogramSource::DecodeToken,
                     ExpertHistogramSource::PrefillChunk,
                     ExpertHistogramSource::GroupedVerifier})
            {
                const auto ticket = probe.beginSample(workload(source));
                if (!ticket.valid())
                    continue;
                /*
                 * Two state-machine edges are enough for the immediately-ready
                 * fake wave: authenticate Running, then reap abort ownership.
                 * Real maintenance uses the same non-blocking polling contract.
                 */
                poll_maintenance();
                poll_maintenance();
                EXPECT_TRUE(probe.finishSample(ticket));
            }
        }

        /** @brief Complete inference phases without progressing maintenance. */
        void runOneInferenceInvocationPerPhase(
            MoEOverlayInferenceInterferenceProbe &probe)
        {
            runOneInferenceInvocationPerPhase(probe, [] {});
        }

        /** @brief Complete controller ownership bundle used by each test. */
        struct CalibrationFixture
        {
            std::shared_ptr<MoEOverlayEconomyCalibrationPlanner> planner;
            std::shared_ptr<MoEOverlayMigrationMeasurementLedger> ledger;
            std::shared_ptr<MoEOverlayMigrationMeasurementJournal> journal;
            std::shared_ptr<MoEOverlayInferenceInterferenceProbe> probe;
            CalibrationFactory factory;
            std::shared_ptr<MoEOverlayTierMigrationTransport> transport;
            std::unique_ptr<MoEOverlayEconomyCalibrationController> controller;

            /** @brief Wire the real composite transport to a bounded journal. */
            explicit CalibrationFixture(bool grouped_verifier_enabled = true)
            {
                planner = std::make_shared<MoEOverlayEconomyCalibrationPlanner>(
                    MoEOverlayEconomyCalibrationPlanner::Config{
                        .live_snapshot = snapshot(),
                        .complete_expert_bytes_per_layer = {3'072},
                    });
                ledger = std::make_shared<MoEOverlayMigrationMeasurementLedger>(
                    MoEOverlayMigrationMeasurementLedger::Config{
                        .required_coordinates = planner->requiredCoordinates(),
                        .warmup_samples_per_coordinate = 0,
                        .measured_samples_per_coordinate = 3,
                        .required_sources = {
                            true,
                            true,
                            grouped_verifier_enabled,
                        },
                        .measurement_identity = "device-free-calibration",
                    });
                journal = std::make_shared<
                    MoEOverlayMigrationMeasurementJournal>(
                    MoEOverlayMigrationMeasurementJournal::Config{
                        .maximum_migrations_per_wave = 2,
                    });
                probe = std::make_shared<
                    MoEOverlayInferenceInterferenceProbe>();
                transport = std::make_shared<
                    MoEOverlayTierMigrationTransport>(
                    MoEOverlayTierMigrationTransport::Config{
                        .factory = &factory,
                        .projections_per_expert = 3,
                        .measurement_sink = journal,
                        .require_complete_local_measurements = true,
                        .perf_device = "device-free-calibration",
                    });
                controller = std::make_unique<
                    MoEOverlayEconomyCalibrationController>(
                    MoEOverlayEconomyCalibrationController::Config{
                        .planner = planner,
                        .ledger = ledger,
                        .journal = journal,
                        .probe = probe,
                        .transport = transport,
                        .perf_device = "device-free-calibration",
                    });
            }
        };
    } // namespace

    TEST(
        MoEOverlayEconomyEvidenceMerger,
        ConservativelyJoinsPartialRanksAndUsesMaximumInterferenceWorkload)
    {
        const auto merged =
            MoEOverlayEconomyEvidenceMerger::mergeAttempt({
                attemptEvidence(0),
                attemptEvidence(1),
            });
        ASSERT_TRUE(merged.valid());
        ASSERT_TRUE(merged.accepted);
        ASSERT_EQ(merged.measurements.size(), 2u);
        EXPECT_EQ(merged.baseline_nanoseconds, 200u);
        EXPECT_EQ(merged.concurrent_nanoseconds, 260u);
        for (const auto &migration : merged.measurements)
        {
            EXPECT_TRUE(std::all_of(
                migration.projections.begin(),
                migration.projections.end(),
                [](const auto &projection)
                { return projection.has_value(); }));
            EXPECT_EQ(migration.wave_wall_nanoseconds, 501u);
        }

        auto mismatched_workload = attemptEvidence(1);
        ++mismatched_workload.workload.execution_rows;
        EXPECT_THROW(
            (void)MoEOverlayEconomyEvidenceMerger::mergeAttempt({
                attemptEvidence(0),
                mismatched_workload,
            }),
            std::invalid_argument);
    }

    TEST(
        MoEOverlayEconomyEvidenceMerger,
        RejectsTheAttemptOnEveryRankWhenOneOverlapIsNotExact)
    {
        const auto merged =
            MoEOverlayEconomyEvidenceMerger::mergeAttempt({
                attemptEvidence(0),
                attemptEvidence(1, false),
            });
        EXPECT_TRUE(merged.valid());
        EXPECT_FALSE(merged.accepted);
        EXPECT_TRUE(merged.measurements.empty());
        EXPECT_EQ(merged.baseline_nanoseconds, 0u);
        EXPECT_EQ(merged.concurrent_nanoseconds, 0u);
    }

    TEST(
        MoEOverlayEconomyEvidenceMerger,
        RequiresEveryServiceCellFromItsExactOwningWorldRank)
    {
        const auto owner_map = distributedOwnerMap();
        const std::vector<std::vector<
            MoEOverlayParticipantLayerServiceTotals>> rank_rows{
            {
                serviceRow(0, 0, 100),
                serviceRow(0, 1, 110),
            },
            {
                serviceRow(1, 0, 200),
                serviceRow(1, 1, 210),
            },
        };
        const auto merged = MoEOverlayEconomyEvidenceMerger::mergeService(
            rank_rows, owner_map, 2);
        ASSERT_EQ(merged.size(), 4u);
        EXPECT_EQ(merged[0].participant_id, 0);
        EXPECT_EQ(merged[0].layer, 0);
        EXPECT_EQ(merged[1].participant_id, 0);
        EXPECT_EQ(merged[1].layer, 1);
        EXPECT_EQ(merged[2].participant_id, 1);
        EXPECT_EQ(merged[2].layer, 0);
        EXPECT_EQ(merged[3].participant_id, 1);
        EXPECT_EQ(merged[3].layer, 1);

        auto wrong_rank = rank_rows;
        wrong_rank[0].push_back(wrong_rank[1].back());
        wrong_rank[1].pop_back();
        EXPECT_THROW(
            (void)MoEOverlayEconomyEvidenceMerger::mergeService(
                wrong_rank, owner_map, 2),
            std::invalid_argument);

        auto missing = rank_rows;
        missing[1].pop_back();
        EXPECT_THROW(
            (void)MoEOverlayEconomyEvidenceMerger::mergeService(
                missing, owner_map, 2),
            std::invalid_argument);

        auto mtp_disabled = rank_rows;
        for (auto &per_rank : mtp_disabled)
        {
            for (auto &row : per_rank)
            {
                row.total_nanoseconds[2] = 0;
                row.activation_count[2] = 0;
                row.sample_count[2] = 0;
            }
        }
        EXPECT_EQ(
            MoEOverlayEconomyEvidenceMerger::mergeService(
                mtp_disabled,
                owner_map,
                2,
                {true, true, false})
                .size(),
            4u);
        EXPECT_THROW(
            (void)MoEOverlayEconomyEvidenceMerger::mergeService(
                rank_rows,
                owner_map,
                2,
                {true, true, false}),
            std::invalid_argument);
    }

    TEST(
        MoEOverlayEconomyCalibrationController,
        RealPairSwapOverlapsEveryExactProductionWorkloadAndNeverCommits)
    {
        CalibrationFixture fixture;
        for (int iteration = 0;
             iteration < 1'000 &&
             fixture.controller->state() !=
                 MoEOverlayEconomyCalibrationState::Complete;
             ++iteration)
        {
            fixture.controller->poll();
            runOneInferenceInvocationPerPhase(
                *fixture.probe,
                [&fixture] { fixture.controller->poll(); });
        }

        ASSERT_TRUE(fixture.controller->healthy())
            << fixture.controller->failureMessage();
        ASSERT_EQ(
            fixture.controller->state(),
            MoEOverlayEconomyCalibrationState::Complete);
        const auto *sealed = fixture.controller->sealedMeasurements();
        ASSERT_NE(sealed, nullptr);
        ASSERT_TRUE(sealed->valid());
        ASSERT_EQ(sealed->rows.size(), 2u);

        const auto stats = fixture.controller->stats();
        EXPECT_EQ(stats.accepted_pairs, 9u);
        EXPECT_EQ(stats.exact_overlap_samples, 9u);
        EXPECT_EQ(stats.partial_overlap_rejections, 0u);
        EXPECT_EQ(stats.waves_started, 9u);
        EXPECT_EQ(stats.concurrent_launches_during_inference, 9u);
        EXPECT_EQ(stats.concurrent_launch_misses, 0u);
        EXPECT_EQ(stats.waves_aborted, 9u);
        EXPECT_EQ(fixture.factory.observations->waves_prepared, 9u);
        EXPECT_EQ(fixture.factory.observations->projection_aborts, 54u);
        EXPECT_EQ(fixture.factory.observations->bank_aborts, 9u);
        EXPECT_EQ(fixture.factory.observations->bank_commits, 0u);
        EXPECT_TRUE(fixture.probe->idle());
    }

    TEST(
        MoEOverlayEconomyCalibrationController,
        MTPDisabledCalibrationRequiresOnlyDecodeAndPrefillEvidence)
    {
        CalibrationFixture fixture(false);
        EXPECT_EQ(
            fixture.controller->requiredSources(),
            (ExpertHistogramProductionSourceMask{true, true, false}));
        for (int iteration = 0;
             iteration < 1'000 &&
             fixture.controller->state() !=
                 MoEOverlayEconomyCalibrationState::Complete;
             ++iteration)
        {
            fixture.controller->poll();
            /* The grouped scope cannot claim an unarmed ticket and is harmless. */
            runOneInferenceInvocationPerPhase(
                *fixture.probe,
                [&fixture] { fixture.controller->poll(); });
        }

        ASSERT_TRUE(fixture.controller->healthy())
            << fixture.controller->failureMessage();
        ASSERT_EQ(
            fixture.controller->state(),
            MoEOverlayEconomyCalibrationState::Complete);
        const auto stats = fixture.controller->stats();
        EXPECT_EQ(stats.accepted_pairs, 6u);
        EXPECT_EQ(stats.exact_overlap_samples, 6u);
        EXPECT_EQ(stats.waves_started, 6u);
        EXPECT_EQ(stats.concurrent_launches_during_inference, 6u);
        EXPECT_EQ(stats.waves_aborted, 6u);
        ASSERT_NE(fixture.controller->sealedMeasurements(), nullptr);
        EXPECT_TRUE(fixture.controller->sealedMeasurements()->valid());
    }

    TEST(
        MoEOverlayEconomyCalibrationController,
        PreparedWaveCannotDispatchBeforeInferenceAndMissedWindowIsRetried)
    {
        CalibrationFixture fixture;

        fixture.controller->poll(); // Arm baseline.
        runOneInferenceInvocationPerPhase(*fixture.probe);
        fixture.controller->poll(); // Consume baseline.
        fixture.controller->poll(); // Prepare wave and arm concurrent sample.
        fixture.controller->poll(); // Armed: no physical operation may start.
        EXPECT_EQ(fixture.factory.observations->projection_polls, 0u);
        runOneInferenceInvocationPerPhase(*fixture.probe); // Miss launch window.
        fixture.controller->poll(); // Observe Completed and abort unlaunched wave.
        fixture.controller->poll(); // Reap abort cleanup and retry.
        EXPECT_EQ(
            fixture.controller->stats().partial_overlap_rejections,
            1u);
        EXPECT_EQ(fixture.controller->stats().concurrent_launch_misses, 1u);
        EXPECT_EQ(fixture.factory.observations->projection_polls, 0u);

        for (int iteration = 0;
             iteration < 1'000 &&
             fixture.controller->state() !=
                 MoEOverlayEconomyCalibrationState::Complete;
             ++iteration)
        {
            fixture.controller->poll();
            runOneInferenceInvocationPerPhase(
                *fixture.probe,
                [&fixture] { fixture.controller->poll(); });
        }
        ASSERT_TRUE(fixture.controller->healthy())
            << fixture.controller->failureMessage();
        EXPECT_EQ(
            fixture.controller->state(),
            MoEOverlayEconomyCalibrationState::Complete);
        EXPECT_GE(fixture.controller->stats().waves_started, 10u);
        EXPECT_EQ(fixture.factory.observations->bank_commits, 0u);
    }

    TEST(
        MoEOverlayEconomyCalibrationController,
        StopNeverInvalidatesAnInferenceOwnedTimingTicket)
    {
        CalibrationFixture fixture;
        fixture.controller->poll(); // Arm the baseline request.
        const auto ticket = fixture.probe->beginSample(
            workload(ExpertHistogramSource::DecodeToken));
        ASSERT_TRUE(ticket.valid());

        fixture.controller->requestStop();
        fixture.controller->poll();
        EXPECT_NE(
            fixture.controller->state(),
            MoEOverlayEconomyCalibrationState::Stopped);
        EXPECT_FALSE(fixture.probe->idle());

        ASSERT_TRUE(fixture.probe->finishSample(ticket));
        for (int iteration = 0;
             iteration < 10 &&
             fixture.controller->state() !=
                 MoEOverlayEconomyCalibrationState::Stopped;
             ++iteration)
        {
            fixture.controller->poll();
        }
        EXPECT_TRUE(fixture.controller->healthy())
            << fixture.controller->failureMessage();
        EXPECT_EQ(
            fixture.controller->state(),
            MoEOverlayEconomyCalibrationState::Stopped);
        EXPECT_TRUE(fixture.probe->idle());
        EXPECT_EQ(fixture.probe->stats().completed_discards, 1u);
        EXPECT_EQ(fixture.factory.observations->bank_commits, 0u);
    }

    TEST(
        MoEOverlayEconomyCalibrationController,
        StopAbortsAndReapsAnAlreadyStartedPhysicalWave)
    {
        CalibrationFixture fixture;
        fixture.controller->poll(); // Arm baseline.
        runOneInferenceInvocationPerPhase(*fixture.probe);
        fixture.controller->poll(); // Consume baseline.
        fixture.controller->poll(); // Prepare physical wave and arm overlap.
        ASSERT_EQ(fixture.factory.observations->waves_prepared, 1u);
        EXPECT_EQ(fixture.factory.observations->projection_polls, 0u);
        const auto concurrent_ticket = fixture.probe->beginSample(
            workload(ExpertHistogramSource::DecodeToken));
        ASSERT_TRUE(concurrent_ticket.valid());
        fixture.controller->poll(); // Authenticated Running releases the wave.
        EXPECT_EQ(fixture.factory.observations->projection_polls, 6u);
        ASSERT_TRUE(fixture.probe->finishSample(concurrent_ticket));

        fixture.controller->requestStop();
        for (int iteration = 0;
             iteration < 10 &&
             fixture.controller->state() !=
                 MoEOverlayEconomyCalibrationState::Stopped;
             ++iteration)
        {
            fixture.controller->poll();
        }
        EXPECT_TRUE(fixture.controller->healthy())
            << fixture.controller->failureMessage();
        EXPECT_EQ(
            fixture.controller->state(),
            MoEOverlayEconomyCalibrationState::Stopped);
        EXPECT_TRUE(fixture.probe->idle());
        EXPECT_EQ(fixture.factory.observations->projection_aborts, 6u);
        EXPECT_EQ(fixture.factory.observations->bank_aborts, 1u);
        EXPECT_EQ(fixture.factory.observations->bank_commits, 0u);
    }

    TEST(
        MoEOverlayEconomyCalibrationController,
        ExactLayerCatalogMeasuresRepresentativesAndExpandsTheFullMatrix)
    {
        const std::vector<MoEOverlayLayerWeightManifest> manifest{
            layerManifest(0, 96, native_vnni_formats::Q4_0),
            layerManifest(1, 96, native_vnni_formats::Q4_0),
            layerManifest(2, 128, native_vnni_formats::Q4_0),
        };
        MoEOverlayEconomyCalibrationLayerCatalog catalog(manifest);
        ASSERT_FALSE(catalog.identity().empty());
        ASSERT_EQ(catalog.groups().size(), 2u);
        EXPECT_EQ(catalog.groups()[0].representative_layer, 0);
        EXPECT_EQ(catalog.groups()[0].member_layers, (std::vector<int>{0, 1}));
        EXPECT_EQ(catalog.groups()[1].representative_layer, 2);
        EXPECT_EQ(catalog.groups()[1].member_layers, (std::vector<int>{2}));
        EXPECT_EQ(catalog.representativeLayers(), (std::vector<int>{0, 2}));
        ASSERT_EQ(catalog.completeExpertBytesPerLayer().size(), 3u);
        EXPECT_EQ(
            catalog.completeExpertBytesPerLayer()[0],
            catalog.completeExpertBytesPerLayer()[1]);

        MoEOverlayEconomyCalibrationPlanner planner({
            .live_snapshot = snapshot(3),
            .complete_expert_bytes_per_layer =
                catalog.completeExpertBytesPerLayer(),
            .calibration_layers = catalog.representativeLayers(),
        });
        ASSERT_EQ(planner.requiredCoordinates().size(), 4u);
        EXPECT_THROW(
            (void)planner.buildPairSwap(0, 1, 1, 1),
            std::invalid_argument);
        EXPECT_TRUE(planner.buildPairSwap(0, 1, 2, 1).valid());

        MoEOverlaySealedMigrationMeasurements representatives{
            .identity = "representative-corpus",
            .rows = {
                migrationRow(0, 1, 0),
                migrationRow(0, 1, 2),
                migrationRow(1, 0, 0),
                migrationRow(1, 0, 2),
            },
        };
        ASSERT_TRUE(representatives.valid());
        const auto expanded = catalog.expand(representatives);
        ASSERT_TRUE(expanded.valid());
        EXPECT_EQ(expanded.rows.size(), 6u);
        EXPECT_NE(expanded.identity, representatives.identity);
        for (const auto &[source, destination] :
             std::array<std::pair<int, int>, 2>{{{0, 1}, {1, 0}}})
        {
            for (int layer = 0; layer < 3; ++layer)
            {
                EXPECT_EQ(
                    std::count_if(
                        expanded.rows.begin(),
                        expanded.rows.end(),
                        [source, destination, layer](const auto &row)
                        {
                            return row.source_participant == source &&
                                   row.destination_participant == destination &&
                                   row.layer == layer;
                        }),
                    1);
            }
        }
    }

    TEST(
        MoEOverlayEconomyCalibrationController,
        LocalCertificationInstallsCompleteProfilesBeforeFirstProposal)
    {
        const auto initial = snapshot();
        DecodeExpertHistogramConfig histogram_config;
        histogram_config.num_layers = 1;
        histogram_config.num_experts = 2;
        histogram_config.top_k = 1;
        histogram_config.window_size = 1;
        histogram_config.token_boundary_layer_idx = 0;
        histogram_config.sockets = {
            DeviceId::cuda(0),
            DeviceId::cpu(),
        };
        histogram_config.ownership = initial->layered_ownership;
        auto histogram = std::make_shared<DecodeExpertHistogram>(
            std::move(histogram_config));
        auto authority = std::make_shared<MoEOverlayResidencyAuthority>(
            MoEOverlayResidencyAuthority::Config{
                .initial_plan = *initial->placement_plan,
                .model_metadata = {
                    .num_layers = 1,
                    .num_experts = 2,
                    .d_model = 64,
                    .routed_intermediate_size = 96,
                    .routed_quant_type = "Q4_0",
                },
                .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
                .histogram = histogram.get(),
                .perf_device = "local-certification-test",
            });
        ASSERT_FALSE(authority->hasEconomyCertification());

        const std::vector<MoEOverlayLayerWeightManifest> manifest{
            layerManifest(0, 96, native_vnni_formats::Q4_0),
        };
        auto catalog = std::make_shared<
            MoEOverlayEconomyCalibrationLayerCatalog>(manifest);
        auto planner = std::make_shared<
            MoEOverlayEconomyCalibrationPlanner>(
            MoEOverlayEconomyCalibrationPlanner::Config{
                .live_snapshot = authority->snapshot(),
                .complete_expert_bytes_per_layer =
                    catalog->completeExpertBytesPerLayer(),
                .calibration_layers = catalog->representativeLayers(),
            });
        auto ledger = std::make_shared<
            MoEOverlayMigrationMeasurementLedger>(
            MoEOverlayMigrationMeasurementLedger::Config{
                .required_coordinates = planner->requiredCoordinates(),
                .warmup_samples_per_coordinate = 0,
                .measured_samples_per_coordinate = 3,
                .measurement_identity = catalog->identity(),
            });
        auto journal = std::make_shared<
            MoEOverlayMigrationMeasurementJournal>(
            MoEOverlayMigrationMeasurementJournal::Config{
                .maximum_migrations_per_wave = 2,
            });
        auto probe =
            std::make_shared<MoEOverlayInferenceInterferenceProbe>();
        CalibrationFactory factory;
        auto transport = std::make_shared<MoEOverlayTierMigrationTransport>(
            MoEOverlayTierMigrationTransport::Config{
                .factory = &factory,
                .projections_per_expert = 3,
                .measurement_sink = journal,
                .measurement_scope =
                    MoEOverlayMigrationMeasurementScope::
                        EconomyCalibrationOnly,
                .require_complete_local_measurements = true,
                .perf_device = "local-certification-test",
            });
        auto calibration = std::make_shared<
            MoEOverlayEconomyCalibrationController>(
            MoEOverlayEconomyCalibrationController::Config{
                .planner = planner,
                .ledger = ledger,
                .journal = journal,
                .probe = probe,
                .transport = transport,
                .perf_device = "local-certification-test",
            });

        auto registry = std::make_shared<
            MoEOverlayParticipantResidencyRegistry>(
            MoEOverlayParticipantResidencyRegistry::Config{
                .owner_map = authority->snapshot()->owner_map,
                .local_participant_ids = {0, 1},
                .num_layers = 1,
                .num_experts = 2,
                .initial_epoch = authority->snapshot()->epoch,
                .collect_economy_service_measurements = true,
            });
        for (int participant = 0; participant < 2; ++participant)
        {
            const auto endpoint = registry->endpoint(participant);
            ASSERT_NE(endpoint, nullptr);
            for (const auto source : {
                     ExpertHistogramSource::DecodeToken,
                     ExpertHistogramSource::PrefillChunk,
                     ExpertHistogramSource::GroupedVerifier})
            {
                ASSERT_EQ(
                    endpoint->recordServiceMeasurement(
                        0,
                        source,
                        participant == 0 ? 10u : 100u,
                        1),
                    MoEOverlayServiceMeasurementRecordStatus::Recorded);
            }
        }

        MoEOverlayEconomyCertificationController certification({
            .calibration = calibration,
            .registry = registry,
            .layer_catalog = catalog,
            .authority = authority,
            .model_metadata = {
                .num_layers = 1,
                .num_experts = 2,
                .d_model = 64,
                .routed_intermediate_size = 96,
                .routed_quant_type = "Q4_0",
            },
            .economy_policy = {
                .historical_window_weight = 0,
                .current_window_weight = 1,
                .payoff_horizon_tokens = 8,
                .minimum_residency_generations = 0,
            },
            .perf_device = "local-certification-test",
        });
        for (int iteration = 0;
             iteration < 1'000 &&
             certification.state() !=
                 MoEOverlayEconomyCertificationState::Complete;
             ++iteration)
        {
            certification.poll();
            runOneInferenceInvocationPerPhase(
                *probe,
                [&certification] { certification.poll(); });
        }
        ASSERT_TRUE(certification.healthy())
            << certification.failureMessage();
        ASSERT_EQ(
            certification.state(),
            MoEOverlayEconomyCertificationState::Complete);
        EXPECT_TRUE(authority->hasEconomyCertification());
        EXPECT_EQ(authority->stats().checks, 0u);
        EXPECT_EQ(certification.stats().certifications_installed, 1u);
        EXPECT_EQ(factory.observations->bank_commits, 0u);

        const std::uint64_t counts[]{1, 100};
        histogram->mergeLayerCounts(0, counts, 2, false);
        histogram->recordTokenBoundary(0, 1);
        const auto transaction = authority->proposeFromHistogram();
        EXPECT_TRUE(transaction.valid());
        EXPECT_TRUE(transaction.economy.enabled);
        EXPECT_FALSE(transaction.economy.service_profile_identity.empty());
        EXPECT_FALSE(transaction.economy.migration_profile_identity.empty());
    }
} // namespace llaminar2::test
