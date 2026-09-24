/**
 * @file Test__MoEOverlayEconomyCalibrationController.cpp
 * @brief Device-free proof of bounded real-transfer economy profiling.
 *
 * The startup profiler must exercise the production transfer transaction,
 * retain robust physical timings, and abort every staged residency candidate.
 * It must never ask the caller to run synthetic inference or publish a bank.
 * Service-coverage regressions also keep cold exact-format classes explicit:
 * preparation must not borrow a price from another format or participant.
 */

#include "execution/moe/MoEOverlayEconomyCalibrationController.h"
#include "execution/moe/MoEOverlayEconomyCertificationController.h"
#include "execution/moe/MoEOverlayEconomyEvidenceExchange.h"
#include "execution/moe/MoEOverlayTierMigrationTransport.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
        /** @brief Enable only economy PerfStats and restore process state. */
        class ScopedEconomyPerfStats final
        {
        public:
            /** @brief Save the environment and enable the focused domain. */
            ScopedEconomyPerfStats()
                : old_summary_(environmentValue(
                      "LLAMINAR_PERF_STATS_SUMMARY")),
                  old_filter_(environmentValue(
                      "LLAMINAR_PERF_STATS_FILTER"))
            {
                setenv("LLAMINAR_PERF_STATS_SUMMARY", "1", 1);
                setenv(
                    "LLAMINAR_PERF_STATS_FILTER",
                    "moe_overlay_residency",
                    1);
                mutableDebugEnv().reload();
                PerfStatsCollector::reset();
            }

            /** @brief Clear evidence and restore the caller's environment. */
            ~ScopedEconomyPerfStats()
            {
                PerfStatsCollector::reset();
                restore(
                    "LLAMINAR_PERF_STATS_SUMMARY", old_summary_);
                restore("LLAMINAR_PERF_STATS_FILTER", old_filter_);
                mutableDebugEnv().reload();
            }

            ScopedEconomyPerfStats(const ScopedEconomyPerfStats &) = delete;
            ScopedEconomyPerfStats &operator=(
                const ScopedEconomyPerfStats &) = delete;

        private:
            /** @return Current environment value, preserving unset state. */
            static std::optional<std::string> environmentValue(
                const char *name)
            {
                const char *value = std::getenv(name);
                return value ? std::optional<std::string>(value)
                             : std::nullopt;
            }

            /** @brief Restore one exact saved environment value. */
            static void restore(
                const char *name,
                const std::optional<std::string> &value)
            {
                if (value)
                    setenv(name, value->c_str(), 1);
                else
                    unsetenv(name);
            }

            std::optional<std::string> old_summary_;
            std::optional<std::string> old_filter_;
        };

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

        /** @brief Build one MTP-only sidecar row with impossible phases empty. */
        MoEOverlayParticipantLayerServiceTotals mtpSidecarServiceRow(
            int participant,
            int layer,
            std::uint64_t bias)
        {
            auto row = serviceRow(participant, layer, bias);
            for (const std::size_t phase : {std::size_t{0}, std::size_t{1}})
            {
                row.total_nanoseconds[phase] = 0;
                row.activation_count[phase] = 0;
                row.sample_count[phase] = 0;
            }
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
            explicit ProfileFixture(
                int layer_count = 1,
                std::shared_ptr<
                    const MoEOverlaySealedMigrationMeasurements>
                    presealed_measurements = nullptr)
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
                        .presealed_measurements =
                            std::move(presealed_measurements),
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
        MoEOverlayEconomyCalibrationController,
        ExactSealedProfileCompletesWithoutLaunchingPhysicalWaves)
    {
        ScopedEconomyPerfStats perf_stats;
        ProfileFixture source;
        pollToTerminal(*source.controller);
        ASSERT_EQ(
            source.controller->state(),
            MoEOverlayEconomyCalibrationState::Complete);
        const auto *sealed = source.controller->sealedMeasurements();
        ASSERT_NE(sealed, nullptr);
        auto retained = std::make_shared<
            const MoEOverlaySealedMigrationMeasurements>(*sealed);

        PerfStatsCollector::reset();
        ProfileFixture reused(/*layer_count=*/1, retained);
        EXPECT_EQ(
            reused.controller->state(),
            MoEOverlayEconomyCalibrationState::Complete);
        EXPECT_TRUE(reused.controller->healthy());
        EXPECT_EQ(reused.controller->sealedMeasurements()->identity,
                  retained->identity);
        const auto stats = reused.controller->stats();
        EXPECT_EQ(stats.waves_started, 0u);
        EXPECT_EQ(stats.waves_completed, 0u);
        EXPECT_EQ(stats.accepted_pairs,
                  reused.controller->expectedAcceptedPairs());
        EXPECT_EQ(stats.reused_sealed_profiles, 1u);
        EXPECT_EQ(reused.factory.observations->waves_prepared, 0u);

        const auto evidence = PerfStatsCollector::snapshot(
            {"moe_overlay_residency"});
        const auto complete = std::find_if(
            evidence.begin(), evidence.end(),
            [](const PerfStatRecord &record)
            {
                return record.name ==
                       "economy_transport_profile_complete";
            });
        ASSERT_NE(complete, evidence.end());
        EXPECT_EQ(complete->value, 1.0);
        EXPECT_EQ(complete->count, 1u);
        EXPECT_EQ(complete->tags.at("origin"), "reused");
        EXPECT_EQ(
            complete->tags.at("measurement_identity"),
            retained->identity);
        EXPECT_EQ(complete->tags.at("coordinate_count"), "2");
        EXPECT_EQ(complete->tags.at("waves"), "3");
        EXPECT_EQ(complete->tags.at("elapsed_nanoseconds"), "0");
        EXPECT_EQ(complete->tags.at("synthetic_inference"), "false");
        EXPECT_EQ(complete->tags.at("publish_residency"), "false");
    }

    TEST(
        MoEOverlayEconomyCalibrationController,
        RejectsSealedProfileFromAnotherParticipantLayerTopology)
    {
        ProfileFixture source;
        pollToTerminal(*source.controller);
        const auto *sealed = source.controller->sealedMeasurements();
        ASSERT_NE(sealed, nullptr);
        auto incompatible = *sealed;
        ASSERT_FALSE(incompatible.rows.empty());
        ++incompatible.rows.front().layer;
        ASSERT_TRUE(incompatible.valid());

        EXPECT_THROW(
            (void)ProfileFixture(
                /*layer_count=*/1,
                std::make_shared<
                    const MoEOverlaySealedMigrationMeasurements>(
                    std::move(incompatible))),
            std::invalid_argument);
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
            rank_rows,
            owner_map,
            ExpertHistogramProductionTopology::uniform(
                2, kAllExpertHistogramProductionSources));
        ASSERT_EQ(merged.size(), 4u);

        auto wrong_rank = rank_rows;
        wrong_rank[0].push_back(wrong_rank[1].back());
        wrong_rank[1].pop_back();
        EXPECT_THROW(
            (void)MoEOverlayEconomyEvidenceMerger::mergeService(
                wrong_rank,
                owner_map,
                ExpertHistogramProductionTopology::uniform(
                    2, kAllExpertHistogramProductionSources)),
            std::invalid_argument);

        auto missing = rank_rows;
        missing[1].pop_back();
        EXPECT_THROW(
            (void)MoEOverlayEconomyEvidenceMerger::mergeService(
                missing,
                owner_map,
                ExpertHistogramProductionTopology::uniform(
                    2, kAllExpertHistogramProductionSources)),
            std::invalid_argument);
    }

    TEST(
        MoEOverlayEconomyEvidenceMerger,
        UsesReachabilityRatherThanEconomyMaskAndRejectsImpossibleSidecarEvidence)
    {
        const auto owner_map = distributedOwnerMap();
        const auto topology =
            ExpertHistogramProductionTopology::forRetainedExecution(
                /*retained_layer_count=*/3,
                /*main_inference_layer_count=*/2,
                ExpertHistogramServingRegime::PositiveDepthMTP);
        const std::vector<std::vector<
            MoEOverlayParticipantLayerServiceTotals>> rank_rows{
            {
                serviceRow(0, 0, 100),
                serviceRow(0, 1, 110),
                mtpSidecarServiceRow(0, 2, 120),
            },
            {
                serviceRow(1, 0, 200),
                serviceRow(1, 1, 210),
                mtpSidecarServiceRow(1, 2, 220),
            },
        };

        const auto merged = MoEOverlayEconomyEvidenceMerger::mergeService(
            rank_rows, owner_map, topology);
        ASSERT_EQ(merged.size(), 6u);
        EXPECT_EQ(merged[0].sample_count[0], 3u)
            << "Serial catch-up decode is reachable even when it is not economy-priced";

        auto impossible = rank_rows;
        auto &sidecar_decode = impossible[1][2];
        sidecar_decode.total_nanoseconds[0] = 999;
        sidecar_decode.activation_count[0] = 1;
        sidecar_decode.sample_count[0] = 1;
        EXPECT_THROW(
            (void)MoEOverlayEconomyEvidenceMerger::mergeService(
                impossible, owner_map, topology),
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
        EXPECT_EQ(
            catalog.serviceTelemetryLayers(),
            (std::vector<int>{0, 1, 2}));
        EXPECT_EQ(catalog.layerCount(), 3u);
        EXPECT_TRUE(catalog.isRepresentativeLayer(0));
        EXPECT_FALSE(catalog.isRepresentativeLayer(1));
        EXPECT_TRUE(catalog.isRepresentativeLayer(2));
        EXPECT_FALSE(catalog.isRepresentativeLayer(-1));
        EXPECT_FALSE(catalog.isRepresentativeLayer(3));

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
        MoEOverlayEconomyCalibrationController,
        ServiceTelemetryUsesDeterministicSquareRootLayerStrata)
    {
        std::vector<MoEOverlayLayerWeightManifest> manifest;
        manifest.reserve(17u);
        for (int layer = 0; layer < 16; ++layer)
        {
            manifest.push_back(
                layerManifest(layer, 96, native_vnni_formats::Q4_0));
        }
        manifest.push_back(
            layerManifest(16, 128, native_vnni_formats::Q4_0));

        const MoEOverlayEconomyCalibrationLayerCatalog catalog(manifest);
        ASSERT_EQ(catalog.groups().size(), 2u);
        EXPECT_EQ(catalog.representativeLayers(), (std::vector<int>{0, 16}));
        EXPECT_EQ(
            catalog.groups()[0].service_telemetry_layers,
            (std::vector<int>{2, 6, 10, 14}));
        EXPECT_EQ(
            catalog.groups()[1].service_telemetry_layers,
            (std::vector<int>{16}));
        EXPECT_EQ(
            catalog.serviceTelemetryLayers(),
            (std::vector<int>{2, 6, 10, 14, 16}));
        for (int layer = 0; layer < 17; ++layer)
        {
            const bool expected =
                layer == 2 || layer == 6 || layer == 10 || layer == 14 ||
                layer == 16;
            EXPECT_EQ(catalog.isServiceTelemetryLayer(layer), expected);
        }
        EXPECT_FALSE(catalog.isServiceTelemetryLayer(-1));
        EXPECT_FALSE(catalog.isServiceTelemetryLayer(17));
    }

    /**
     * @brief Idle singleton-format experts produce bounded, participant-local gaps.
     *
     * This reproduces the three-tier model's readiness failure without loading
     * it: common layers execute, while one participant never routes to the
     * singleton class. More observations of the common class cannot close it.
     */
    TEST(MoEOverlayEconomyCalibrationLayerCatalog,
         ColdSingletonFormatEnumeratesEveryMissingPhaseWithoutBorrowing)
    {
        std::vector<MoEOverlayLayerWeightManifest> manifest{
            layerManifest(0, 256, native_vnni_formats::Q4_K),
            layerManifest(1, 256, native_vnni_formats::Q5_K),
            layerManifest(2, 256, native_vnni_formats::Q4_K),
        };
        // Match the failure's mixed triplets: Q4_K/Q4_K/Q5_K in common
        // layers, Q5_K/Q5_K/Q6_K in the otherwise identical singleton.
        for (auto &layer : manifest)
        {
            const auto &down = layer.layer_idx == 1
                                   ? native_vnni_formats::Q6_K
                                   : native_vnni_formats::Q5_K;
            layer.projections[2] = projectionManifest(
                ExpertTierWeightProjection::Down, 64, 256, down);
        }
        const MoEOverlayEconomyCalibrationLayerCatalog catalog(manifest);
        const auto topology = ExpertHistogramProductionTopology::uniform(
            3, kAllExpertHistogramProductionSources);
        std::vector<MoEOverlayParticipantLayerServiceTotals> rows{
            serviceRow(0, 0, 10), serviceRow(0, 1, 10), serviceRow(0, 2, 10),
            serviceRow(3, 0, 100), {.participant_id = 3, .layer = 1},
            serviceRow(3, 2, 100),
        };
        const auto before = rows;
        const auto gaps = catalog.serviceEvidenceGaps(rows, {0, 3}, topology);
        ASSERT_EQ(gaps.size(), 3u);
        for (std::size_t phase = 0; phase < gaps.size(); ++phase)
        {
            EXPECT_EQ(gaps[phase].participant_id, 3);
            EXPECT_EQ(gaps[phase].representative_layer, 1);
            EXPECT_EQ(expertHistogramProductionSourceIndex(gaps[phase].source), phase);
            EXPECT_EQ(gaps[phase].eligible_layers, (std::vector<int>{1}));
        }
        for (std::size_t index = 0; index < rows.size(); ++index)
        {
            EXPECT_EQ(rows[index].total_nanoseconds, before[index].total_nanoseconds);
            EXPECT_EQ(rows[index].activation_count, before[index].activation_count);
            EXPECT_EQ(rows[index].sample_count, before[index].sample_count);
        }
        rows[3] = serviceRow(3, 0, 100'000);
        EXPECT_EQ(catalog.serviceEvidenceGaps(rows, {0, 3}, topology).size(), 3u);
        // Only actual evidence of the missing coordinate completes coverage.
        rows[4] = serviceRow(3, 1, 100);
        EXPECT_TRUE(catalog.serviceEvidenceGaps(rows, {0, 3}, topology).empty());
    }

    /** @brief A dormant class can be priced without changing any live counter. */
    TEST(MoEOverlayEconomyCalibrationLayerCatalog,
         PreparedSamplesCompleteColdClassesWithoutBecomingRuntimeCounters)
    {
        const MoEOverlayEconomyCalibrationLayerCatalog catalog({
            layerManifest(0, 256, native_vnni_formats::Q4_K),
            layerManifest(1, 256, native_vnni_formats::Q5_K),
            layerManifest(2, 256, native_vnni_formats::Q4_K),
        });
        const auto topology = ExpertHistogramProductionTopology::uniform(
            3, kAllExpertHistogramProductionSources);
        const std::vector<MoEOverlayParticipantLayerServiceTotals> live{
            serviceRow(0, 0, 10), serviceRow(0, 1, 20), serviceRow(0, 2, 30),
            {.participant_id = 3, .layer = 0}, {.participant_id = 3, .layer = 1},
            serviceRow(3, 2, 40),
        };
        const std::vector<MoEOverlayParticipantLayerServiceTotals> prepared{
            serviceRow(0, 0, 1000), serviceRow(0, 1, 2000), serviceRow(0, 2, 3000),
            serviceRow(3, 0, 4000), serviceRow(3, 1, 5000), serviceRow(3, 2, 6000),
        };
        const auto combined = catalog.withPreparedServiceEvidence(
            live, prepared, {0, 3}, topology);
        EXPECT_TRUE(catalog.serviceEvidenceGaps(combined, {0, 3}, topology).empty());
        for (std::size_t row = 0; row < combined.size(); ++row)
        {
            // Only the wholly unobserved class uses setup evidence. Live
            // participant 3/layer 2 already owns the common class, so a probe
            // at its equivalent but idle layer 0 must not skew that price.
            const auto &expected = row == 4 ? prepared[row] : live[row];
            EXPECT_EQ(combined[row].total_nanoseconds, expected.total_nanoseconds);
            EXPECT_EQ(combined[row].activation_count, expected.activation_count);
            EXPECT_EQ(combined[row].sample_count, expected.sample_count);
        }
        EXPECT_EQ(live[4].sample_count, (std::array<std::uint64_t, 3>{}));
        EXPECT_EQ(catalog.serviceEvidenceGaps(live, {0, 3}, topology).size(), 3u);
        // A subsequent natural observation takes precedence without adding
        // setup counts to the device's cumulative producer values.
        auto later_live = live;
        later_live[4] = serviceRow(3, 1, 7);
        const auto later = catalog.withPreparedServiceEvidence(
            later_live, prepared, {0, 3}, topology);
        EXPECT_EQ(later[4].total_nanoseconds, later_live[4].total_nanoseconds);
        EXPECT_EQ(later[4].sample_count, later_live[4].sample_count);
    }

    /** @brief An empty or corrupt setup price cannot masquerade as coverage. */
    TEST(MoEOverlayEconomyCalibrationLayerCatalog,
         PreparedEvidenceCannotHideMissingOrMalformedCoordinates)
    {
        const MoEOverlayEconomyCalibrationLayerCatalog catalog({
            layerManifest(0, 256, native_vnni_formats::Q4_K),
            layerManifest(1, 256, native_vnni_formats::Q5_K),
        });
        const auto topology = ExpertHistogramProductionTopology::uniform(
            2, kAllExpertHistogramProductionSources);
        const std::vector<MoEOverlayParticipantLayerServiceTotals> live{
            serviceRow(3, 0, 10), {.participant_id = 3, .layer = 1},
        };
        auto probes = live;
        const auto incomplete = catalog.withPreparedServiceEvidence(live, probes, {3}, topology);
        EXPECT_EQ(catalog.serviceEvidenceGaps(incomplete, {3}, topology).size(), 3u);
        probes[1] = serviceRow(3, 1, 20);
        probes[1].overflowed[0] = true;
        EXPECT_THROW((void)catalog.withPreparedServiceEvidence(live, probes, {3}, topology),
                     std::invalid_argument);
        // Even wholly covered live evidence must validate the unused probe
        // bank: a phase/owner mismatch is not silently discarded as irrelevant.
        const std::vector complete{serviceRow(3, 0, 10), serviceRow(3, 1, 10)};
        probes = complete;
        probes[0].participant_id = 4;
        EXPECT_THROW((void)catalog.withPreparedServiceEvidence(complete, probes, {3}, topology),
                     std::invalid_argument);
    }

    /** @brief Equivalent layers share prices only within the priced phase mask. */
    TEST(MoEOverlayEconomyCalibrationLayerCatalog,
         MissingPlanSelectsPricedMembersNotTheTransferRepresentative)
    {
        const MoEOverlayEconomyCalibrationLayerCatalog catalog({
            layerManifest(0, 256, native_vnni_formats::Q4_K),
            layerManifest(1, 256, native_vnni_formats::Q4_K),
        });
        const ExpertHistogramProductionTopology topology(
            {{true, true, true}, {true, true, true}},
            {{false, false, false}, {false, true, true}});
        std::vector<MoEOverlayParticipantLayerServiceTotals> rows{
            serviceRow(2, 0, 10), {.participant_id = 2, .layer = 1},
        };
        const auto gaps = catalog.serviceEvidenceGaps(rows, {2}, topology);
        ASSERT_EQ(gaps.size(), 2u);
        EXPECT_EQ(gaps[0].source, ExpertHistogramSource::PrefillChunk);
        EXPECT_EQ(gaps[1].source, ExpertHistogramSource::GroupedVerifier);
        for (const auto &gap : gaps)
        {
            EXPECT_EQ(gap.representative_layer, 0);
            EXPECT_EQ(gap.eligible_layers, (std::vector<int>{1}));
        }
        EXPECT_TRUE(catalog.serviceEvidenceGaps({}, {}, topology).empty());
    }

    /** @brief A cold early coordinate must not hide malformed later evidence. */
    TEST(MoEOverlayEconomyCalibrationLayerCatalog,
         MissingPlanRejectsMalformedEvidenceBeforeReportingGaps)
    {
        const MoEOverlayEconomyCalibrationLayerCatalog catalog({
            layerManifest(0, 256, native_vnni_formats::Q4_K),
        });
        const auto topology = ExpertHistogramProductionTopology::uniform(
            1, kAllExpertHistogramProductionSources);
        std::vector<MoEOverlayParticipantLayerServiceTotals> rows{
            {.participant_id = 0, .layer = 0}, serviceRow(3, 0, 10),
        };
        EXPECT_THROW((void)catalog.serviceEvidenceGaps(rows, {3, 0}, topology),
                     std::invalid_argument);
        EXPECT_THROW((void)catalog.serviceEvidenceGaps(rows, {0, 0}, topology),
                     std::invalid_argument);
        EXPECT_THROW((void)catalog.serviceEvidenceGaps(rows, {0}, topology),
                     std::invalid_argument);
        EXPECT_THROW((void)catalog.serviceEvidenceGaps(rows, {-1, 3}, topology),
                     std::invalid_argument);
        rows[1].layer = 1;
        EXPECT_THROW((void)catalog.serviceEvidenceGaps(rows, {0, 3}, topology),
                     std::invalid_argument);
        rows[1] = serviceRow(3, 0, 10);
        const auto serial = ExpertHistogramProductionTopology::uniform(
            1, {true, true, false});
        EXPECT_THROW((void)catalog.serviceEvidenceGaps(rows, {0, 3}, serial),
                     std::invalid_argument);
        rows[1].overflowed[0] = true;
        EXPECT_THROW((void)catalog.serviceEvidenceGaps(rows, {0, 3}, topology),
                     std::invalid_argument);
    }

    /** @brief Every codebook and floating weight type retains a distinct price. */
    TEST(MoEOverlayEconomyCalibrationLayerCatalog,
         MissingPlanKeepsAllSourceCodebooksAndFloatingFormatsDistinct)
    {
        std::vector<MoEOverlayLayerWeightManifest> manifest;
        for (const auto &source : native_vnni_formats::kAllSourceFormats)
        {
            manifest.push_back(layerManifest(
                static_cast<int>(manifest.size()), 256, *source.metadata));
        }
        for (const auto type : {TensorType::FP16, TensorType::BF16, TensorType::FP32})
        {
            auto layer = layerManifest(
                static_cast<int>(manifest.size()), 256, native_vnni_formats::Q4_K);
            for (auto &projection : layer.projections)
                projection.format = ExpertWeightFormat::floating(type);
            manifest.push_back(std::move(layer));
        }
        const MoEOverlayEconomyCalibrationLayerCatalog catalog(manifest);
        ASSERT_EQ(catalog.groups().size(), manifest.size());
        const auto topology = ExpertHistogramProductionTopology::uniform(
            static_cast<int>(manifest.size()), kAllExpertHistogramProductionSources);
        std::vector<MoEOverlayParticipantLayerServiceTotals> rows;
        for (std::size_t layer = 0; layer < manifest.size(); ++layer)
            rows.push_back({.participant_id = 7, .layer = static_cast<int>(layer)});
        for (std::size_t layer = 0; layer < manifest.size(); ++layer)
        {
            const auto gaps = catalog.serviceEvidenceGaps(rows, {7}, topology);
            ASSERT_EQ(gaps.size(), (manifest.size() - layer) * 3u);
            EXPECT_EQ(gaps.front().representative_layer, static_cast<int>(layer));
            rows[layer] = serviceRow(7, static_cast<int>(layer), 100);
        }
        EXPECT_TRUE(catalog.serviceEvidenceGaps(rows, {7}, topology).empty());
    }

    /** @brief Independent measurement producers feeding the same certificate. */
    enum class ServiceEvidenceOrigin { NaturalTraffic, PreparedKernels };

    /** @brief Both evidence origins retain one certification/rebase state machine. */
    static void verifyCertification(ServiceEvidenceOrigin origin)
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
                .required_sources = {false, true, true},
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
        std::vector<MoEOverlayParticipantLayerServiceTotals> prepared;
        if (origin == ServiceEvidenceOrigin::PreparedKernels)
        {
            for (int participant = 0; participant < 2; ++participant)
            {
                auto row = serviceRow(participant, 0, participant == 0 ? 10 : 100);
                row.total_nanoseconds[0] = 0;
                row.activation_count[0] = 0;
                row.sample_count[0] = 0;
                prepared.push_back(row);
            }
            // Device snapshots remain the sole cumulative writer for the GPU
            // endpoint even when startup measurements complete its prices.
            std::string error;
            ASSERT_TRUE(registry->importDeviceServiceMeasurements(
                0, {{.participant_id = 0, .layer = 0}}, &error)) << error;
        }
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
            .production_topology =
                ExpertHistogramProductionTopology::forRetainedExecution(
                    /*retained_layer_count=*/1,
                    /*main_inference_layer_count=*/1,
                    ExpertHistogramServingRegime::PositiveDepthMTP),
            .prepared_service_measurements = std::move(prepared),
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
            if (origin == ServiceEvidenceOrigin::PreparedKernels) continue;
            const auto endpoint = registry->endpoint(participant);
            ASSERT_NE(endpoint, nullptr);
            for (const auto source : {
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
        EXPECT_EQ(authority->snapshot()->epoch, initial->epoch);
        if (origin == ServiceEvidenceOrigin::PreparedKernels)
        {
            std::vector<MoEOverlayParticipantLayerServiceTotals> live;
            ASSERT_TRUE(registry->trySnapshotServiceMeasurements(&live));
            ASSERT_EQ(live.size(), 2u);
            for (const auto &row : live)
            {
                EXPECT_EQ(row.total_nanoseconds, (std::array<std::uint64_t, 3>{}));
                EXPECT_EQ(row.activation_count, (std::array<std::uint64_t, 3>{}));
                EXPECT_EQ(row.sample_count, (std::array<std::uint64_t, 3>{}));
            }
        }
    }

    TEST(MoEOverlayEconomyCertificationController,
         FixedMTPTopologyCertifiesWithoutImpossibleDecodeEvidence)
    {
        verifyCertification(ServiceEvidenceOrigin::NaturalTraffic);
    }

    TEST(MoEOverlayEconomyCertificationController,
         PreparedEvidenceCertifiesWithoutRoutingOrChangingDeviceCounters)
    {
        verifyCertification(ServiceEvidenceOrigin::PreparedKernels);
    }
} // namespace llaminar2::test
