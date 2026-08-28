/**
 * @file Test__MoEOverlayEconomyCalibrationController.cpp
 * @brief Device-free proof of bounded real-transfer economy profiling.
 *
 * The startup profiler must exercise the production transfer transaction,
 * retain robust physical timings, and abort every staged residency candidate.
 * It must never ask the caller to run synthetic inference or publish a bank.
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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        /** @brief Shared proof that profile waves abort and never publish. */
        struct PhysicalObservations
        {
            std::uint64_t waves_prepared = 0;
            std::uint64_t projection_polls = 0;
            std::uint64_t projection_aborts = 0;
            std::uint64_t bank_aborts = 0;
            std::uint64_t bank_preparations = 0;
        };

        /** @brief Immediately-ready measured transfer with explicit cleanup. */
        class MeasuredOperation final
            : public IMoEOverlayTierTransferOperation
        {
        public:
            /** @brief Retain deterministic evidence and lifecycle counters. */
            MeasuredOperation(
                std::shared_ptr<PhysicalObservations> observations,
                ExpertTierProjectionTransferMeasurement measurement)
                : observations_(std::move(observations)),
                  measurement_(measurement)
            {
            }

            /** @brief Publish completion without sleeping or blocking. */
            MoEOverlayResidencyWaveProgress poll(
                std::string *) noexcept override
            {
                ++observations_->projection_polls;
                ready_ = true;
                return MoEOverlayResidencyWaveProgress::Ready;
            }

            /** @brief Mark the unpublished transfer for recycling. */
            void abort() noexcept override
            {
                if (!aborted_)
                {
                    aborted_ = true;
                    ++observations_->projection_aborts;
                }
            }

            /** @brief Device-free recycling completes immediately. */
            MoEOverlayResidencyWaveProgress pollAbort(
                std::string *) noexcept override
            {
                return aborted_ ? MoEOverlayResidencyWaveProgress::Ready
                                : MoEOverlayResidencyWaveProgress::Failed;
            }

            /** @brief Expose timing only after the transfer terminal. */
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

        /** @brief Inactive-bank reservation that rejects publication phases. */
        class ProfileBank final
            : public IMoEOverlayInactiveBankTransaction
        {
        public:
            /** @brief Retain counters beyond transaction destruction. */
            explicit ProfileBank(
                std::shared_ptr<PhysicalObservations> observations)
                : observations_(std::move(observations))
            {
            }

            /** @brief A profile regression must be visible before failing. */
            bool beginPrepare(std::string *) noexcept override
            {
                ++observations_->bank_preparations;
                return false;
            }

            /** @brief Profile transactions never enter preparation polling. */
            MoEOverlayResidencyWaveProgress pollPrepare(
                std::string *) noexcept override
            {
                return MoEOverlayResidencyWaveProgress::Failed;
            }

            /** @brief Profiling is transfer-only and cannot publish banks. */
            bool beginPublication(std::string *) noexcept override
            {
                return false;
            }

            /** @brief Profiling never enters publication polling. */
            MoEOverlayResidencyWaveProgress pollPublication(
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

            /** @brief Device-free cleanup completes immediately. */
            MoEOverlayResidencyWaveProgress pollAbort(
                std::string *) noexcept override
            {
                return aborted_ ? MoEOverlayResidencyWaveProgress::Ready
                                : MoEOverlayResidencyWaveProgress::Failed;
            }

            /** @brief No previous bank can exist for an unpublished profile. */
            void retirePrevious() noexcept override {}

        private:
            std::shared_ptr<PhysicalObservations> observations_;
            bool aborted_ = false;
        };

        /** @brief Produce complete measured projections for every pair swap. */
        class ProfileWaveFactory final
            : public IMoEOverlayTierPreparedWaveFactory
        {
        public:
            ProfileWaveFactory()
                : observations(std::make_shared<PhysicalObservations>())
            {
            }

            /** @brief Materialize deterministic production-lane operations. */
            MoEOverlayTierPreparedWave prepare(
                const MoEOverlayResidencyTransaction &transaction) override
            {
                ++observations->waves_prepared;
                MoEOverlayTierPreparedWave result;
                result.status = MoEOverlayResidencyStageStartStatus::Started;
                result.inactive_bank =
                    std::make_unique<ProfileBank>(observations);
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

        /** @brief Construct one rank-local routed domain. */
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
            result.routed_compute_policy =
                RoutedExpertComputePolicy::Apportioned;
            return result;
        }

        /** @brief Build two integer-priority tiers with one expert each. */
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

        /** @brief Build explicit two-rank ownership for service merge tests. */
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
                throw std::logic_error("partial migration is invalid");
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
                throw std::logic_error("service row is invalid");
            return row;
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

        /** @brief Construct one complete layer geometry signature. */
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

        /** @brief Complete robust physical row suitable for catalog expansion. */
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
            return row;
        }

        /** @brief Complete production-owned profiler fixture. */
        struct ProfileFixture
        {
            std::shared_ptr<MoEOverlayEconomyCalibrationPlanner> planner;
            std::shared_ptr<MoEOverlayMigrationMeasurementLedger> ledger;
            std::shared_ptr<MoEOverlayMigrationMeasurementJournal> journal;
            ProfileWaveFactory factory;
            std::shared_ptr<MoEOverlayTierMigrationTransport> transport;
            std::shared_ptr<MoEOverlayEconomyCalibrationController> controller;

            /** @brief Wire the real composite transport to a bounded journal. */
            explicit ProfileFixture(int layer_count = 1)
            {
                planner = std::make_shared<MoEOverlayEconomyCalibrationPlanner>(
                    MoEOverlayEconomyCalibrationPlanner::Config{
                        .live_snapshot = snapshot(layer_count),
                        .complete_expert_bytes_per_layer =
                            std::vector<std::size_t>(layer_count, 3'072u),
                    });
                ledger = std::make_shared<MoEOverlayMigrationMeasurementLedger>(
                    MoEOverlayMigrationMeasurementLedger::Config{
                        .required_coordinates = planner->requiredCoordinates(),
                        .warmup_samples_per_coordinate = 0,
                        .measured_samples_per_coordinate = 3,
                        .measurement_identity = "device-free-profile",
                    });
                journal = std::make_shared<
                    MoEOverlayMigrationMeasurementJournal>(
                    MoEOverlayMigrationMeasurementJournal::Config{
                        .maximum_migrations_per_wave = 2,
                    });
                transport = std::make_shared<
                    MoEOverlayTierMigrationTransport>(
                    MoEOverlayTierMigrationTransport::Config{
                        .factory = &factory,
                        .projections_per_expert = 3,
                        .measurement_sink = journal,
                        .measurement_scope =
                            MoEOverlayMigrationMeasurementScope::
                                EconomyCalibrationOnly,
                        .require_complete_local_measurements = true,
                        .perf_device = "device-free-profile",
                    });
                controller = std::make_shared<
                    MoEOverlayEconomyCalibrationController>(
                    MoEOverlayEconomyCalibrationController::Config{
                        .planner = planner,
                        .ledger = ledger,
                        .journal = journal,
                        .transport = transport,
                        .perf_device = "device-free-profile",
                    });
            }
        };

        /** @brief Poll a bounded device-free controller to a terminal state. */
        void pollToTerminal(MoEOverlayEconomyCalibrationController &controller)
        {
            for (int iteration = 0;
                 iteration < 1'000 &&
                 controller.state() !=
                     MoEOverlayEconomyCalibrationState::Complete &&
                 controller.state() !=
                     MoEOverlayEconomyCalibrationState::Failed &&
                 controller.state() !=
                     MoEOverlayEconomyCalibrationState::Stopped;
                 ++iteration)
            {
                controller.poll();
            }
        }
    } // namespace

    TEST(
        MoEOverlayEconomyCalibrationController,
        BoundedRealTransfersSealRobustEvidenceAndNeverPublishResidency)
    {
        ProfileFixture fixture;
        EXPECT_EQ(fixture.controller->expectedAcceptedPairs(), 3u);
        EXPECT_FALSE(fixture.controller->awaitsInferenceSample());
        EXPECT_EQ(fixture.controller->awaitedInferenceGeneration(), 0u);
        EXPECT_EQ(
            fixture.controller->awaitedInferenceSource(),
            ExpertHistogramSource::SyntheticTest);

        pollToTerminal(*fixture.controller);
        ASSERT_TRUE(fixture.controller->healthy())
            << fixture.controller->failureMessage();
        ASSERT_EQ(
            fixture.controller->state(),
            MoEOverlayEconomyCalibrationState::Complete);

        const auto *sealed = fixture.controller->sealedMeasurements();
        ASSERT_NE(sealed, nullptr);
        ASSERT_TRUE(sealed->valid());
        ASSERT_EQ(sealed->rows.size(), 2u);
        for (const auto &row : sealed->rows)
        {
            EXPECT_EQ(row.wave_sample_count, 3u);
            EXPECT_EQ(row.inference_interference_nanoseconds, 0u);
            EXPECT_EQ(row.interference_sample_count, 0u);
        }

        const auto stats = fixture.controller->stats();
        EXPECT_EQ(stats.waves_started, 3u);
        EXPECT_EQ(stats.waves_completed, 3u);
        EXPECT_EQ(stats.waves_aborted, 3u);
        EXPECT_EQ(stats.accepted_pairs, 3u);
        EXPECT_EQ(fixture.factory.observations->waves_prepared, 3u);
        EXPECT_EQ(fixture.factory.observations->projection_aborts, 18u);
        EXPECT_EQ(fixture.factory.observations->bank_aborts, 3u);
        EXPECT_EQ(fixture.factory.observations->bank_preparations, 0u);
    }

    TEST(
        MoEOverlayEconomyCalibrationController,
        StopAsynchronouslyAbortsAndReapsAnActivePhysicalWave)
    {
        ProfileFixture fixture;
        fixture.controller->poll();
        ASSERT_EQ(
            fixture.controller->state(),
            MoEOverlayEconomyCalibrationState::AwaitConcurrentWave);

        fixture.controller->requestStop();
        pollToTerminal(*fixture.controller);
        ASSERT_TRUE(fixture.controller->healthy())
            << fixture.controller->failureMessage();
        EXPECT_EQ(
            fixture.controller->state(),
            MoEOverlayEconomyCalibrationState::Stopped);
        EXPECT_EQ(fixture.factory.observations->waves_prepared, 1u);
        EXPECT_EQ(fixture.factory.observations->projection_aborts, 6u);
        EXPECT_EQ(fixture.factory.observations->bank_aborts, 1u);
        EXPECT_EQ(fixture.factory.observations->bank_preparations, 0u);
        EXPECT_EQ(fixture.controller->stats().accepted_pairs, 0u);
    }

    TEST(
        MoEOverlayEconomyEvidenceMerger,
        MigrationProfileCompletesComplementaryRanksAndRejectsIdentityDrift)
    {
        const MoEOverlayMigrationMeasurementCoordinate coordinate{
            .source_participant = 0,
            .destination_participant = 1,
            .layer = 0,
        };
        const MoEOverlayMigrationProfileEvidence rank_zero{
            .profile_sequence = 19,
            .coordinate = coordinate,
            .local_measurements = {
                partialMigration(0, 1, 0, {0}, 0),
                partialMigration(1, 0, 1, {0}, 0),
            },
        };
        const MoEOverlayMigrationProfileEvidence rank_one{
            .profile_sequence = 19,
            .coordinate = coordinate,
            .local_measurements = {
                partialMigration(0, 1, 0, {1, 2}, 1),
                partialMigration(1, 0, 1, {1, 2}, 1),
            },
        };
        ASSERT_TRUE(rank_zero.valid());
        ASSERT_TRUE(rank_one.valid());

        const auto merged =
            MoEOverlayEconomyEvidenceMerger::mergeMigrationProfile(
                {rank_zero, rank_one});
        ASSERT_TRUE(merged.valid());
        ASSERT_EQ(merged.measurements.size(), 2u);
        for (const auto &migration : merged.measurements)
        {
            EXPECT_TRUE(std::all_of(
                migration.projections.begin(),
                migration.projections.end(),
                [](const auto &projection)
                { return projection.has_value(); }));
            EXPECT_EQ(migration.wave_wall_nanoseconds, 501u);
        }

        auto divergent = rank_one;
        ++divergent.profile_sequence;
        EXPECT_THROW(
            (void)MoEOverlayEconomyEvidenceMerger::mergeMigrationProfile(
                {rank_zero, divergent}),
            std::invalid_argument);
    }

    TEST(
        MoEOverlayEconomyEvidenceMerger,
        RequiresEveryServiceCellFromItsExactOwningWorldRank)
    {
        const auto owner_map = distributedOwnerMap();
        const std::vector<std::vector<
            MoEOverlayParticipantLayerServiceTotals>> rank_rows{
            {serviceRow(0, 0, 100), serviceRow(0, 1, 110)},
            {serviceRow(1, 0, 200), serviceRow(1, 1, 210)},
        };
        const auto merged = MoEOverlayEconomyEvidenceMerger::mergeService(
            rank_rows, owner_map, 2);
        ASSERT_EQ(merged.size(), 4u);

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
    }

    TEST(
        MoEOverlayEconomyCalibrationController,
        ExactLayerCatalogProfilesRepresentativesAndExpandsTheFullMatrix)
    {
        const std::vector<MoEOverlayLayerWeightManifest> manifest{
            layerManifest(0, 96, native_vnni_formats::Q4_0),
            layerManifest(1, 96, native_vnni_formats::Q4_0),
            layerManifest(2, 128, native_vnni_formats::Q4_0),
        };
        MoEOverlayEconomyCalibrationLayerCatalog catalog(manifest);
        ASSERT_EQ(catalog.groups().size(), 2u);
        EXPECT_EQ(catalog.representativeLayers(), (std::vector<int>{0, 2}));

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
    }

    TEST(
        MoEOverlayEconomyCertificationController,
        ReadinessBlocksOnlyForTopologyProfileThenNaturalTrafficCertifies)
    {
        const auto initial = snapshot();
        DecodeExpertHistogramConfig histogram_config;
        histogram_config.num_layers = 1;
        histogram_config.num_experts = 2;
        histogram_config.top_k = 1;
        histogram_config.window_size = 1;
        histogram_config.token_boundary_layer_idx = 0;
        histogram_config.sockets = {DeviceId::cuda(0), DeviceId::cpu()};
        histogram_config.ownership = initial->layered_ownership;
        auto histogram = std::make_shared<DecodeExpertHistogram>(
            std::move(histogram_config));
        int histogram_rebase_polls = 0;
        histogram->registerRuntimeHistogramDrain(
            [&]()
            {
                ++histogram_rebase_polls;
                return histogram_rebase_polls == 1
                           ? RuntimeExpertHistogramDrainResult::pending()
                           : RuntimeExpertHistogramDrainResult::ready();
            });
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
                .perf_device = "certification-readiness-test",
            });

        const std::vector<MoEOverlayLayerWeightManifest> manifest{
            layerManifest(0, 96, native_vnni_formats::Q4_0),
        };
        auto catalog = std::make_shared<
            MoEOverlayEconomyCalibrationLayerCatalog>(manifest);
        auto planner = std::make_shared<MoEOverlayEconomyCalibrationPlanner>(
            MoEOverlayEconomyCalibrationPlanner::Config{
                .live_snapshot = authority->snapshot(),
                .complete_expert_bytes_per_layer =
                    catalog->completeExpertBytesPerLayer(),
                .calibration_layers = catalog->representativeLayers(),
            });
        auto ledger = std::make_shared<MoEOverlayMigrationMeasurementLedger>(
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
        ProfileWaveFactory factory;
        auto transport = std::make_shared<MoEOverlayTierMigrationTransport>(
            MoEOverlayTierMigrationTransport::Config{
                .factory = &factory,
                .projections_per_expert = 3,
                .measurement_sink = journal,
                .measurement_scope =
                    MoEOverlayMigrationMeasurementScope::EconomyCalibrationOnly,
                .require_complete_local_measurements = true,
                .perf_device = "certification-readiness-test",
            });
        auto profile = std::make_shared<
            MoEOverlayEconomyCalibrationController>(
            MoEOverlayEconomyCalibrationController::Config{
                .planner = planner,
                .ledger = ledger,
                .journal = journal,
                .transport = transport,
                .perf_device = "certification-readiness-test",
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
        MoEOverlayEconomyCertificationController certification({
            .calibration = profile,
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
            .perf_device = "certification-readiness-test",
        });

        const auto initial_readiness = certification.measurementReadiness();
        EXPECT_FALSE(initial_readiness.ready());
        EXPECT_FALSE(initial_readiness.inference_requested);
        EXPECT_EQ(
            initial_readiness.requested_workload,
            InferenceMeasurementWorkloadKind::None);
        EXPECT_EQ(initial_readiness.required_work_units, 3u);

        for (int iteration = 0;
             iteration < 1'000 &&
             certification.state() ==
                 MoEOverlayEconomyCertificationState::CalibratingMovement;
             ++iteration)
        {
            certification.poll();
        }
        ASSERT_TRUE(certification.healthy())
            << certification.failureMessage();
        ASSERT_EQ(
            certification.state(),
            MoEOverlayEconomyCertificationState::AwaitingServiceEvidence);
        const auto natural_traffic_readiness =
            certification.measurementReadiness();
        EXPECT_TRUE(natural_traffic_readiness.ready());
        EXPECT_FALSE(natural_traffic_readiness.inference_requested);
        EXPECT_FALSE(authority->hasEconomyCertification());

        for (int participant = 0; participant < 2; ++participant)
        {
            const auto endpoint = registry->endpoint(participant);
            ASSERT_NE(endpoint, nullptr);
            for (const auto source : {
                     ExpertHistogramSource::DecodeToken,
                     ExpertHistogramSource::PrefillChunk,
                     ExpertHistogramSource::GroupedVerifier})
            {
                EXPECT_EQ(
                    endpoint->recordServiceMeasurement(
                        0,
                        source,
                        participant == 0 ? 10u : 100u,
                        1),
                    MoEOverlayServiceMeasurementRecordStatus::Recorded);
            }
        }
        certification.poll();
        ASSERT_EQ(
            certification.state(),
            MoEOverlayEconomyCertificationState::
                RebasingRoutingEvidence);
        EXPECT_FALSE(authority->hasEconomyCertification());

        certification.poll();
        EXPECT_EQ(
            certification.state(),
            MoEOverlayEconomyCertificationState::
                RebasingRoutingEvidence);
        EXPECT_FALSE(authority->hasEconomyCertification());
        EXPECT_EQ(histogram_rebase_polls, 1);

        certification.poll();
        ASSERT_TRUE(certification.healthy())
            << certification.failureMessage();
        EXPECT_EQ(
            certification.state(),
            MoEOverlayEconomyCertificationState::Complete);
        EXPECT_TRUE(authority->hasEconomyCertification());
        EXPECT_EQ(histogram_rebase_polls, 2);
        EXPECT_EQ(certification.stats().routing_evidence_rebases, 1u);
        EXPECT_EQ(certification.stats().certifications_installed, 1u);
        EXPECT_EQ(factory.observations->bank_preparations, 0u);
    }
} // namespace llaminar2::test
