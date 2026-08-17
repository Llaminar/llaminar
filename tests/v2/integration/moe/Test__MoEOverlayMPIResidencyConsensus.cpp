/**
 * @file Test__MoEOverlayMPIResidencyConsensus.cpp
 * @brief Real-MPI integration proof for asynchronous residency votes.
 *
 * Two background-capable MPI ranks exchange staging and inactive-commit votes
 * over the production private communicator.  The tests also inject a remote
 * physical failure and a divergent transaction identity to prove that neither
 * can become a publishable ExpertOverlay epoch.
 */

#include "execution/moe/MoEOverlayMPIResidencyConsensus.h"
#include "execution/moe/MoEOverlayMPIEconomyEvidenceExchange.h"
#include "execution/moe/MoEOverlayMPIHistogramPublisher.h"
#include "execution/moe/MoEOverlayMPIRemoteProjectionTransport.h"
#include "execution/moe/MoEOverlayDistributedResidencyTransport.h"
#include "execution/moe/MoEOverlayEconomyCalibrationController.h"
#include "execution/moe/MoEOverlayEconomyCertificationController.h"
#include "execution/moe/MoEOverlayParticipantMigration.h"
#include "execution/moe/MoEOverlayPhysicalResidencyFabric.h"
#include "execution/moe/MoEOverlayResidencyMaintenanceService.h"
#include "execution/moe/MoEOverlayTierMigrationTransport.h"
#include "execution/moe/CpuExpertSlotPool.h"
#include "utils/MPIContext.h"

#include <gtest/gtest.h>
#include <mpi.h>

#include <chrono>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        /** @brief Return one structurally valid synthetic transaction identity. */
        MoEOverlayDistributedResidencyWaveIdentity identity(
            std::uint64_t fingerprint_low = 0x1122334455667788ull)
        {
            return {
                .expected_epoch = 9,
                .candidate_epoch = 10,
                .histogram_generation = 17,
                .migration_count = 2,
                .cycle_count = 1,
                .transaction_fingerprint = {
                    .low = fingerprint_low,
                    .high = 0x8877665544332211ull,
                },
            };
        }

        /** @brief Bind the current world communicator without rank assumptions. */
        std::shared_ptr<MPIContext> worldContext()
        {
            int rank = -1;
            int world_size = 0;
            EXPECT_EQ(MPI_Comm_rank(MPI_COMM_WORLD, &rank), MPI_SUCCESS);
            EXPECT_EQ(MPI_Comm_size(MPI_COMM_WORLD, &world_size), MPI_SUCCESS);
            return std::make_shared<MPIContext>(
                rank,
                world_size,
                MPI_COMM_WORLD);
        }

        /**
         * @brief Progress one exchange to completion without a blocking MPI wait.
         *
         * MPI_Test is the progress engine on ordinary Open MPI installations,
         * so this loop deliberately yields only after each progress call.  The
         * production lane enforces the canonical 30-second fatal deadline.
         */
        std::vector<MoEOverlayDistributedResidencyVote> finishExchange(
            MoEOverlayMPIResidencyConsensus &lane)
        {
            std::vector<MoEOverlayDistributedResidencyVote> votes;
            for (;;)
            {
                std::string error;
                const auto progress = lane.poll(&votes, &error);
                if (progress == MoEOverlayResidencyWaveProgress::Ready)
                    return votes;
                EXPECT_EQ(
                    progress,
                    MoEOverlayResidencyWaveProgress::Pending)
                    << error;
                if (progress == MoEOverlayResidencyWaveProgress::Failed)
                    return {};
                std::this_thread::yield();
            }
        }

        /** @brief Progress one coordinator publication without an MPI wait. */
        std::shared_ptr<const DecodeExpertHistogramWindow>
        finishHistogramPublication(
            MoEOverlayMPIHistogramPublisher &publisher)
        {
            std::shared_ptr<const DecodeExpertHistogramWindow> window;
            for (;;)
            {
                std::string error;
                const auto progress = publisher.poll(&window, &error);
                if (progress == MoEOverlayResidencyWaveProgress::Ready)
                    return window;
                EXPECT_EQ(
                    progress,
                    MoEOverlayResidencyWaveProgress::Pending)
                    << error;
                if (progress == MoEOverlayResidencyWaveProgress::Failed)
                    return {};
                std::this_thread::yield();
            }
        }

        /** @brief Poll one economy-attempt all-gather without an MPI wait. */
        MoEOverlayCalibrationAttemptResult finishEconomyAttempt(
            MoEOverlayMPIEconomyEvidenceExchange &exchange)
        {
            MoEOverlayCalibrationAttemptResult result;
            for (;;)
            {
                std::string error;
                const auto progress = exchange.pollAttempt(&result, &error);
                if (progress == MoEOverlayResidencyWaveProgress::Ready)
                    return result;
                EXPECT_EQ(
                    progress,
                    MoEOverlayResidencyWaveProgress::Pending)
                    << error;
                if (progress == MoEOverlayResidencyWaveProgress::Failed)
                    return {};
                std::this_thread::yield();
            }
        }

        /** @brief Poll one service-readiness reduction without an MPI wait. */
        bool finishEconomyServiceReadiness(
            MoEOverlayMPIEconomyEvidenceExchange &exchange)
        {
            bool all_ranks_ready = false;
            for (;;)
            {
                std::string error;
                const auto progress = exchange.pollServiceReadiness(
                    &all_ranks_ready, &error);
                if (progress == MoEOverlayResidencyWaveProgress::Ready)
                    return all_ranks_ready;
                EXPECT_EQ(
                    progress,
                    MoEOverlayResidencyWaveProgress::Pending)
                    << error;
                if (progress == MoEOverlayResidencyWaveProgress::Failed)
                    return false;
                std::this_thread::yield();
            }
        }

        /** @brief Poll one economy-service all-gather without an MPI wait. */
        std::vector<MoEOverlayParticipantLayerServiceTotals>
        finishEconomyService(
            MoEOverlayMPIEconomyEvidenceExchange &exchange)
        {
            std::vector<MoEOverlayParticipantLayerServiceTotals> result;
            for (;;)
            {
                std::string error;
                const auto progress = exchange.pollService(&result, &error);
                if (progress == MoEOverlayResidencyWaveProgress::Ready)
                    return result;
                EXPECT_EQ(
                    progress,
                    MoEOverlayResidencyWaveProgress::Pending)
                    << error;
                if (progress == MoEOverlayResidencyWaveProgress::Failed)
                    return {};
                std::this_thread::yield();
            }
        }

        /** @brief Skip coherently unless CTest supplied the exact two-rank cell. */
        bool requireTwoRanks(const MPIContext &context)
        {
            return context.world_size() == 2;
        }

        /** @brief Build one exact rank-owned routed-expert domain. */
        RoutedExpertDomain domain(
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

        /** @brief Build one thermally ordered routed-expert tier. */
        RoutedExpertTier tier(
            std::string name,
            std::string domain_name,
            int priority,
            int capacity,
            bool cold_remainder = false)
        {
            RoutedExpertTier result;
            result.name = std::move(name);
            result.domain = std::move(domain_name);
            result.priority = priority;
            result.max_experts_per_layer = capacity;
            result.fallback = cold_remainder;
            return result;
        }

        /** @brief Own a deterministic real authority transaction on each rank. */
        struct RealTransactionFixture
        {
            std::unique_ptr<DecodeExpertHistogram> histogram;
            std::unique_ptr<MoEOverlayResidencyAuthority> authority;
            MoEOverlayResidencyTransaction transaction;
        };

        /** @brief Produce identical two-tier movement from independent ranks. */
        RealTransactionFixture realTransaction(int hot_world_rank = 0)
        {
            const int cold_world_rank = hot_world_rank == 0 ? 1 : 0;
            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan.continuation_domain = "gpu_hot";
            plan.shared_expert_domain = "gpu_hot";
            plan.residency_policy =
                RoutedExpertResidencyPolicy::RoutedTierRebalanced;
            plan.owner_order = RoutedExpertOwnerOrder::Ordinal;
            plan.domains = {
                domain(
                    "gpu_hot",
                    GlobalDeviceAddress::cuda(0, 0, "node-a"),
                    hot_world_rank,
                    CollectiveBackendType::NCCL),
                domain(
                    "cpu_cold",
                    GlobalDeviceAddress::cpu(1, "node-a"),
                    cold_world_rank,
                    CollectiveBackendType::MPI),
            };
            plan.routed_tiers = {
                tier("hot", "gpu_hot", 0, 2),
                tier("cold", "cpu_cold", 1, 0, true),
            };

            MoERoutedExpertModelMetadata metadata;
            metadata.num_layers = 1;
            metadata.num_experts = 4;
            metadata.d_model = 16;
            metadata.routed_intermediate_size = 8;
            metadata.routed_quant_type = "F32";

            DecodeExpertHistogramConfig histogram_config;
            histogram_config.num_layers = 1;
            histogram_config.num_experts = 4;
            histogram_config.top_k = 2;
            histogram_config.window_size = 4;
            histogram_config.token_boundary_layer_idx = 0;
            histogram_config.sockets = {
                DeviceId::cuda(0),
                DeviceId::cpu(),
            };
            histogram_config.ownership =
                MoELayeredExpertOwnership::uniform(1, 2, {0, 0, 1, 1});

            RealTransactionFixture fixture;
            fixture.histogram =
                std::make_unique<DecodeExpertHistogram>(histogram_config);
            const std::vector<std::uint64_t> counts{1, 2, 100, 90};
            fixture.histogram->mergeLayerCounts(
                0,
                counts.data(),
                static_cast<int>(counts.size()),
                false);
            fixture.histogram->recordTokenBoundary(0, 4);
            fixture.authority =
                std::make_unique<MoEOverlayResidencyAuthority>(
                    MoEOverlayResidencyAuthority::Config{
                        .initial_plan = std::move(plan),
                        .model_metadata = metadata,
                        .maintenance_mode =
                            MoERebalanceRuntimeMode::Dynamic,
                        .histogram = fixture.histogram.get(),
                        .phase_service_profile = []
                        {
                            auto profile = std::make_shared<
                                MoERoutedTierServiceProfile>();
                            profile->identity =
                                "mpi-maintenance-service-profile-v1";
                            profile->costs = {
                                {.tier_index = 0,
                                 .layer = 0,
                                 .nanoseconds_per_activation = {10, 20, 30}},
                                {.tier_index = 1,
                                 .layer = 0,
                                 .nanoseconds_per_activation = {100, 200, 300}},
                            };
                            return profile;
                        }(),
                        .migration_cost_profile = []
                        {
                            auto profile = std::make_shared<
                                MoEOverlayMigrationCostProfile>();
                            profile->identity =
                                "mpi-maintenance-migration-profile-v1";
                            profile->costs = {
                                {.source_participant = 0,
                                 .destination_participant = 1,
                                 .layer = 0,
                                 .transfer_and_repack_ns = 1},
                                {.source_participant = 1,
                                 .destination_participant = 0,
                                 .layer = 0,
                                 .transfer_and_repack_ns = 1},
                            };
                            return profile;
                        }(),
                        .migration_economy_policy =
                            MoEOverlayMigrationEconomyPolicy{
                                .historical_window_weight = 0,
                                .current_window_weight = 1,
                                .payoff_horizon_tokens = 4,
                                .minimum_residency_generations = 0,
                            },
                        .perf_device = "real_mpi_transport_test",
                    });
            fixture.transaction = fixture.authority->proposeFromHistogram();
            if (!fixture.transaction.valid() || fixture.transaction.empty())
            {
                throw std::logic_error(
                    "Real MPI fixture did not produce a residency wave");
            }
            return fixture;
        }

        /** @brief Immediate CPU event wave used beneath the production wrapper. */
        class ImmediateLocalResidencyTransport final
            : public IMoEOverlayResidencyTransport
        {
        public:
            /** @brief Device-free local bank with independently failable commit. */
            class Wave final : public IMoEOverlayResidencyWave
            {
            public:
                /** @brief Bind the local scripted result owner. */
                explicit Wave(ImmediateLocalResidencyTransport *owner)
                    : owner_(owner)
                {
                }

                /** @brief Local physical staging is already complete. */
                MoEOverlayResidencyWaveProgress pollStage(
                    std::string *) noexcept override
                {
                    return MoEOverlayResidencyWaveProgress::Ready;
                }

                /** @brief Return the rank-local scripted bank enqueue result. */
                bool beginCommit(std::string *error) noexcept override
                {
                    if (!owner_->commit_begin_ok && error)
                        *error = "injected real-MPI local commit failure";
                    return owner_->commit_begin_ok;
                }

                /** @brief Successful enqueues are immediately observable. */
                MoEOverlayResidencyWaveProgress pollCommit(
                    std::string *) noexcept override
                {
                    return MoEOverlayResidencyWaveProgress::Ready;
                }

                /** @brief Record global abort ownership. */
                void abortStaged() noexcept override { ++owner_->aborts; }

                /** @brief Record old-bank retirement after publication. */
                void retirePrevious() noexcept override
                {
                    ++owner_->retirements;
                }

            private:
                ImmediateLocalResidencyTransport *owner_ = nullptr;
            };

            /** @brief Return the current local start script without waiting. */
            MoEOverlayResidencyStageStart beginStage(
                const MoEOverlayResidencyTransaction &transaction) override
            {
                ++begin_calls;
                fingerprints.push_back(
                    fingerprintMoEOverlayResidencyTransaction(transaction));
                if (defer_start)
                {
                    return {
                        .status =
                            MoEOverlayResidencyStageStartStatus::Deferred,
                        .error = "injected real-MPI local backpressure",
                    };
                }
                return {
                    .status = MoEOverlayResidencyStageStartStatus::Started,
                    .wave = std::make_unique<Wave>(this),
                };
            }

            bool defer_start = false;
            bool commit_begin_ok = true;
            int begin_calls = 0;
            int aborts = 0;
            int retirements = 0;
            std::vector<MoEOverlayResidencyTransactionFingerprint> fingerprints;
        };

        /** @brief Progress one authority wave without a blocking MPI wait. */
        MoEOverlayResidencyApplyResult finishAuthorityWave(
            MoEOverlayResidencyAuthority &authority)
        {
            for (;;)
            {
                const auto result = authority.advanceBackground();
                if (result.status != MoEOverlayResidencyApplyStatus::Staging &&
                    result.status !=
                        MoEOverlayResidencyApplyStatus::Committing)
                {
                    return result;
                }
                std::this_thread::yield();
            }
        }

        /** @brief Poll the non-blocking all-rank retirement fence to quiescence. */
        bool drainAuthorityRetirements(
            MoEOverlayResidencyAuthority &authority,
            std::string *error = nullptr)
        {
            const auto deadline =
                std::chrono::steady_clock::now() +
                std::chrono::seconds(5);
            while (authority.pendingRetirementCount() != 0u &&
                   std::chrono::steady_clock::now() < deadline)
            {
                const auto progress = authority.advanceBackground();
                if (progress.status ==
                    MoEOverlayResidencyApplyStatus::CommitFailed)
                {
                    if (error)
                        *error = progress.error;
                    return false;
                }
                std::this_thread::yield();
            }
            if (authority.pendingRetirementCount() != 0u)
            {
                if (error)
                    *error = "Distributed old-epoch retirement did not drain within five seconds";
                return false;
            }
            if (error)
                error->clear();
            return true;
        }

        /** @brief Wait for asynchronous service state under a short test bound. */
        template <typename Predicate>
        bool waitForService(Predicate &&predicate)
        {
            const auto deadline =
                std::chrono::steady_clock::now() +
                std::chrono::seconds(5);
            while (!predicate())
            {
                if (std::chrono::steady_clock::now() >= deadline)
                    return false;
                std::this_thread::yield();
            }
            return true;
        }

        /** @brief Build deterministic final CPU projection bytes for MPI traffic. */
        cpu::native_vnni::CPUNativeVNNIPackedWeights remoteCpuProjection(
            std::uint8_t seed)
        {
            using namespace cpu::native_vnni;
            CPUNativeVNNIPackedWeights packed;
            packed.N = 32;
            packed.K = 64;
            packed.N_padded = 64;
            packed.blocks_per_row = 2;
            packed.codebook_id = native_vnni_formats::Q4_0.codebook_id;
            packed.payload_bytes = 16;
            packed.encoding = CPUNativeVNNIEncoding::NibbleLUT;
            packed.is_asymmetric = false;
            packed.is_superblock = false;
            packed.data_stride = static_cast<int>(
                preparedDataStride(packed.encoding));
            packed.interleaved_block_stride =
                preparedInterleavedBlockStride(
                    packed.encoding, packed.is_asymmetric);
            const std::size_t bytes =
                static_cast<std::size_t>(packed.N_padded / 64) *
                static_cast<std::size_t>(packed.blocks_per_row) *
                static_cast<std::size_t>(
                    packed.interleaved_block_stride);
            packed.native_interleaved.resize_uninitialized(bytes);
            for (std::size_t index = 0; index < bytes; ++index)
            {
                packed.native_interleaved[index] =
                    static_cast<std::uint8_t>(
                        seed + static_cast<std::uint8_t>(index * 29u));
            }
            return packed;
        }

        /** @brief Name one direction of a bidirectional two-rank data wave. */
        MoEOverlayRemoteProjectionIdentity remoteProjectionIdentity(
            int source_rank,
            int destination_rank,
            std::uint64_t migration_index,
            ExpertTierWeightProjection projection)
        {
            return {
                .expected_epoch = 1,
                .candidate_epoch = 2,
                .transaction_fingerprint = {
                    .low = 0x6758493021abcdefull,
                    .high = 0xfedcba9876543210ull,
                },
                .migration_index = migration_index,
                .layer_idx = 0,
                .expert_id = static_cast<int>(migration_index),
                .projection = projection,
                .source_participant = source_rank,
                .destination_participant = destination_rank,
                .source_world_rank = source_rank,
                .destination_world_rank = destination_rank,
                .source_device = DeviceId::cpu(),
                .destination_device = DeviceId::cpu(),
            };
        }

        /** @brief Present one CPU packed allocation as a four-region source. */
        std::array<std::span<const std::uint8_t>, 4> cpuSourceRegions(
            const cpu::native_vnni::CPUNativeVNNIPackedWeights &packed)
        {
            return {
                std::span<const std::uint8_t>(
                    packed.native_interleaved.data(),
                    packed.native_interleaved.size()),
                std::span<const std::uint8_t>{},
                std::span<const std::uint8_t>{},
                std::span<const std::uint8_t>{},
            };
        }

        /** @brief Present one final CPU allocation as four writable regions. */
        std::array<std::span<std::uint8_t>, 4> cpuDestinationRegions(
            std::vector<std::uint8_t> &destination)
        {
            return {
                std::span<std::uint8_t>(
                    destination.data(), destination.size()),
                std::span<std::uint8_t>{},
                std::span<std::uint8_t>{},
                std::span<std::uint8_t>{},
            };
        }

        /** @brief Construct exactly the endpoint owned by this MPI rank. */
        MoEOverlayMPIRemoteProjectionBinding localHostBinding(
            int world_rank,
            std::size_t lane_index,
            const MoEOverlayRemoteProjectionIdentity &identity,
            const MoEOverlayRemoteProjectionManifest &manifest,
            const cpu::native_vnni::CPUNativeVNNIPackedWeights &source,
            std::vector<std::uint8_t> &destination,
            const std::shared_ptr<void> &source_lifetime,
            const std::shared_ptr<void> &destination_lifetime)
        {
            MoEOverlayMPIRemoteProjectionBinding binding{
                .lane_index = lane_index,
                .identity = identity,
            };
            if (world_rank == identity.source_world_rank)
            {
                binding.source = std::make_shared<
                    MoEOverlayHostRemoteProjectionSource>(
                    manifest,
                    cpuSourceRegions(source),
                    source_lifetime);
            }
            else if (world_rank == identity.destination_world_rank)
            {
                binding.destination = std::make_shared<
                    MoEOverlayHostRemoteProjectionDestination>(
                    identity,
                    cpuDestinationRegions(destination),
                    destination_lifetime);
            }
            return binding;
        }

        /** @brief Build two CPU tiers with hot ownership deliberately on rank one. */
        MoERoutedExpertPlacementPlan distributedCpuOverlayPlan()
        {
            constexpr int hot_world_rank = 1;
            constexpr int cold_world_rank = 0;
            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan.continuation_domain = "cpu_hot";
            plan.shared_expert_domain = "cpu_hot";
            plan.residency_policy =
                RoutedExpertResidencyPolicy::RoutedTierRebalanced;
            plan.owner_order = RoutedExpertOwnerOrder::Ordinal;
            plan.domains = {
                domain(
                    "cpu_hot",
                    GlobalDeviceAddress::cpu(),
                    hot_world_rank,
                    CollectiveBackendType::MPI),
                domain(
                    "cpu_cold",
                    GlobalDeviceAddress::cpu(),
                    cold_world_rank,
                    CollectiveBackendType::MPI),
            };
            plan.routed_tiers = {
                tier("hot", "cpu_hot", 0, 1),
                tier("cold", "cpu_cold", 1, 0, true),
            };
            plan.placements = {{
                .layer = 0,
                .routed_expert_tier = {0, 1},
            }};
            return plan;
        }

        /** @brief Q4_0 shapes used by every real distributed CPU expert. */
        std::vector<CpuExpertSlotPool::ProjectionSpec>
        distributedCpuProjectionSpecs()
        {
            const NativeVnniSourceIdentity source_identity{
                .codebook_id = native_vnni_formats::Q4_0.codebook_id,
                .is_superblock =
                    native_vnni_formats::Q4_0.is_superblock,
                .present = true,
            };
            return {
                {
                    .projection = ExpertTierWeightProjection::Gate,
                    .N = 32,
                    .K = 64,
                    .format =
                        ExpertWeightFormat::nativeVnni(source_identity),
                },
                {
                    .projection = ExpertTierWeightProjection::Up,
                    .N = 32,
                    .K = 64,
                    .format =
                        ExpertWeightFormat::nativeVnni(source_identity),
                },
                {
                    .projection = ExpertTierWeightProjection::Down,
                    .N = 64,
                    .K = 32,
                    .format =
                        ExpertWeightFormat::nativeVnni(source_identity),
                },
            };
        }

        /** @brief Fill one real prepared CPU expert with deterministic final bytes. */
        MoEOverlayPreparedExpertTriplet distributedCpuSourceExpert(
            int participant_id,
            int expert_id,
            std::uint8_t seed)
        {
            auto pool = CpuExpertSlotPool::create({
                .participant_id = participant_id,
                .layer_idx = 0,
                .capacity = 1,
                .projections = distributedCpuProjectionSpecs(),
                .memory_placement =
                    CpuExpertSlotPool::MemoryPlacement::aggregateDomain(),
                .perf_device = "mpi-distributed-physical-cpu",
            });
            auto lease = pool->acquire(expert_id, /*residency_epoch=*/1);
            if (!lease || lease->projections.size() != 3)
                throw std::runtime_error(
                    "Distributed CPU physical test could not acquire source expert");

            MoEOverlayPreparedExpertTriplet triplet;
            for (const auto &projection : lease->projections)
            {
                for (std::size_t byte = 0;
                     byte < projection.destination_bytes.size();
                     ++byte)
                {
                    projection.destination_bytes[byte] =
                        static_cast<std::uint8_t>(
                            seed + static_cast<std::uint8_t>(byte * 13u));
                }
                switch (projection.projection)
                {
                case ExpertTierWeightProjection::Gate:
                    triplet.gate = projection.engine;
                    break;
                case ExpertTierWeightProjection::Up:
                    triplet.up = projection.engine;
                    break;
                case ExpertTierWeightProjection::Down:
                    triplet.down = projection.engine;
                    break;
                }
            }
            if (!triplet.complete())
                throw std::runtime_error(
                    "Distributed CPU physical source triplet is incomplete");
            return triplet;
        }

        /** @brief Publish the one initial layer owned by this process. */
        std::shared_ptr<MoEOverlayParticipantResidencyRegistry>
        distributedCpuRegistry(
            const std::shared_ptr<const MoEOverlayResidencySnapshot> &snapshot,
            int participant_id,
            int expert_id,
            const MoEOverlayPreparedExpertTriplet &expert,
            bool collect_economy_service_measurements = false)
        {
            auto registry = std::make_shared<
                MoEOverlayParticipantResidencyRegistry>(
                MoEOverlayParticipantResidencyRegistry::Config{
                    .owner_map = snapshot->owner_map,
                    .local_participant_ids = {participant_id},
                    .num_layers = 1,
                    .num_experts = 2,
                    .initial_epoch = snapshot->epoch,
                    .retained_epoch_capacity = 2,
                    .collect_economy_service_measurements =
                        collect_economy_service_measurements,
                });
            const auto mask = snapshot->owner_map.expertMaskForParticipant(
                0, participant_id, 2);
            std::vector<MoEOverlayPreparedExpertTriplet> engines(2);
            engines[static_cast<std::size_t>(expert_id)] = expert;
            std::string error;
            if (!registry->registerInitialLayer(
                    participant_id, 0, mask, engines, &error) ||
                !registry->allInitialBanksReady())
            {
                throw std::runtime_error(
                    error.empty()
                        ? "Distributed CPU registry did not publish its initial bank"
                        : std::move(error));
            }
            return registry;
        }

        /** @brief Exact Q4_0 projection manifest used by the CPU fixture. */
        std::vector<MoEOverlayLayerWeightManifest>
        distributedCpuLayerManifest()
        {
            const NativeVnniSourceIdentity source_identity{
                .codebook_id = native_vnni_formats::Q4_0.codebook_id,
                .is_superblock =
                    native_vnni_formats::Q4_0.is_superblock,
                .present = true,
            };
            return {{
                .layer_idx = 0,
                .projections = {{
                    {
                        .projection = ExpertTierWeightProjection::Gate,
                        .N = 32,
                        .K = 64,
                        .format =
                            ExpertWeightFormat::nativeVnni(source_identity),
                    },
                    {
                        .projection = ExpertTierWeightProjection::Up,
                        .N = 32,
                        .K = 64,
                        .format =
                            ExpertWeightFormat::nativeVnni(source_identity),
                    },
                    {
                        .projection = ExpertTierWeightProjection::Down,
                        .N = 64,
                        .K = 32,
                        .format =
                            ExpertWeightFormat::nativeVnni(source_identity),
                    },
                }},
            }};
        }

        /**
         * @brief Offer one exact live workload and keep concurrent work alive.
         *
         * A real MPI transfer needs several nonblocking progress polls. Keep the
         * winning inference ticket open until calibration has completed the
         * physical stage, matching the production invariant that the whole wave
         * interval must lie inside the inference interval.
         */
        void runEconomyInferencePhases(
            MoEOverlayInferenceInterferenceProbe &probe,
            MoEOverlayEconomyCalibrationController &calibration,
            MoEOverlayEconomyCertificationController &certification)
        {
            for (const auto source : {
                     ExpertHistogramSource::DecodeToken,
                     ExpertHistogramSource::PrefillChunk,
                     ExpertHistogramSource::GroupedVerifier})
            {
                const int depth =
                    source == ExpertHistogramSource::GroupedVerifier ? 3 : 0;
                const int rows =
                    source == ExpertHistogramSource::PrefillChunk
                        ? 16
                        : depth + 1;
                const auto ticket = probe.beginSample({
                    .source = source,
                    .real_rows = rows,
                    .execution_rows = rows,
                    .transaction_count = 1,
                    .speculative_depth = depth,
                    .schedule_fingerprint =
                        0xabc000u +
                        static_cast<std::uint64_t>(source),
                });
                if (!ticket.valid())
                    continue;

                if (calibration.state() ==
                    MoEOverlayEconomyCalibrationState::
                        AwaitConcurrentInference)
                {
                    const auto deadline =
                        std::chrono::steady_clock::now() +
                        std::chrono::seconds(2);
                    while ((calibration.state() ==
                                MoEOverlayEconomyCalibrationState::
                                    AwaitConcurrentInference ||
                            calibration.state() ==
                                MoEOverlayEconomyCalibrationState::
                                    AwaitConcurrentWave) &&
                           calibration.healthy() &&
                           std::chrono::steady_clock::now() < deadline)
                    {
                        certification.poll();
                        std::this_thread::yield();
                    }
                }
                EXPECT_TRUE(probe.finishSample(ticket));
            }
        }

        /** @brief Verify every final byte against the source's fill function. */
        bool distributedCpuBytesMatchSeed(
            const MoEOverlayPreparedExpertTriplet &triplet,
            std::uint8_t seed)
        {
            const std::array<std::shared_ptr<ITensorGemm>, 3> engines{
                triplet.gate,
                triplet.up,
                triplet.down,
            };
            for (const auto &engine : engines)
            {
                const auto *packed = engine
                                         ? engine
                                               ->exportCPUNativeVNNIPackedWeights()
                                         : nullptr;
                if (!packed)
                    return false;
                for (std::size_t byte = 0;
                     byte < packed->native_interleaved.size();
                     ++byte)
                {
                    const auto expected = static_cast<std::uint8_t>(
                        seed + static_cast<std::uint8_t>(byte * 13u));
                    if (packed->native_interleaved[byte] != expected)
                        return false;
                }
            }
            return true;
        }

        /** @brief One valid partial migration observation for an MPI rank. */
        MoEOverlayCompletedMigrationMeasurement economyMigration(
            int source,
            int destination,
            int expert,
            int world_rank)
        {
            MoEOverlayCompletedMigrationMeasurement row{
                .expected_epoch = 11,
                .candidate_epoch = 12,
                .source_participant = source,
                .destination_participant = destination,
                .layer = 0,
                .expert = expert,
                .wave_wall_nanoseconds = static_cast<std::uint64_t>(
                    900 + world_rank),
            };
            const std::size_t first = world_rank == 0 ? 0u : 1u;
            const std::size_t count = world_rank == 0 ? 1u : 2u;
            for (std::size_t offset = 0; offset < count; ++offset)
            {
                const std::size_t projection = first + offset;
                row.projections[projection] = {
                    .sequence = static_cast<std::uint64_t>(
                        30 + world_rank * 3) + projection,
                    .bytes = 8'192u + projection,
                    .wall_nanoseconds = static_cast<std::uint64_t>(
                        300 + world_rank * 10) + projection,
                    .device_nanoseconds = static_cast<std::uint64_t>(
                        250 + world_rank * 10) + projection,
                };
            }
            if (!row.valid())
                throw std::logic_error(
                    "MPI economy test built invalid migration evidence");
            return row;
        }

        /** @brief One exact production workload and local migration pair. */
        MoEOverlayCalibrationAttemptEvidence economyAttempt(
            int world_rank,
            bool exact_overlap)
        {
            return {
                .calibration_sequence = 41,
                .coordinate = {
                    .source_participant = 0,
                    .destination_participant = 1,
                    .layer = 0,
                },
                .source = ExpertHistogramSource::GroupedVerifier,
                .workload = {
                    .source = ExpertHistogramSource::GroupedVerifier,
                    .real_rows = 4,
                    .execution_rows = 4,
                    .transaction_count = 1,
                    .speculative_depth = 3,
                    .schedule_fingerprint = 0x99887766u,
                },
                .baseline_nanoseconds = static_cast<std::uint64_t>(
                    world_rank == 0 ? 100 : 200),
                .concurrent_nanoseconds = exact_overlap
                                              ? static_cast<std::uint64_t>(
                                                    world_rank == 0 ? 125 : 260)
                                              : 0u,
                .exact_overlap = exact_overlap,
                .local_measurements = {
                    economyMigration(0, 1, 0, world_rank),
                    economyMigration(1, 0, 1, world_rank),
                },
            };
        }

        /** @brief Sampled local prepared-expert service row. */
        MoEOverlayParticipantLayerServiceTotals economyServiceRow(
            int participant,
            int world_rank)
        {
            MoEOverlayParticipantLayerServiceTotals row{
                .participant_id = participant,
                .layer = 0,
            };
            for (std::size_t phase = 0;
                 phase < kExpertHistogramProductionSourceCount;
                 ++phase)
            {
                row.total_nanoseconds[phase] =
                    static_cast<std::uint64_t>(
                        1'000 + world_rank * 100) + phase;
                row.activation_count[phase] = phase + 1;
                row.sample_count[phase] = 3;
            }
            if (!row.valid())
                throw std::logic_error(
                    "MPI economy test built invalid service evidence");
            return row;
        }
    } // namespace

    TEST(
        Test__MoEOverlayMPIResidencyConsensus,
        EconomyEvidenceLaneMergesAttemptsAndOwnedServiceWithoutBlocking)
    {
        auto context = worldContext();
        if (!requireTwoRanks(*context))
            GTEST_SKIP() << "Economy evidence exchange requires two ranks";

        const auto owner_map =
            MoEExpertOwnerMap::build(distributedCpuOverlayPlan());
        MoEOverlayMPIEconomyEvidenceExchange exchange({
            .mpi_context = context,
            .owner_map = owner_map,
            .num_layers = 1,
            .perf_device = "mpi_economy_evidence_cpu_test",
        });

        std::string error;
        ASSERT_TRUE(exchange.beginAttempt(
            economyAttempt(context->rank(), true), &error))
            << error;
        const auto accepted = finishEconomyAttempt(exchange);
        ASSERT_TRUE(accepted.valid());
        ASSERT_TRUE(accepted.accepted);
        ASSERT_EQ(accepted.measurements.size(), 2u);
        EXPECT_EQ(accepted.baseline_nanoseconds, 200u);
        EXPECT_EQ(accepted.concurrent_nanoseconds, 260u);
        for (const auto &migration : accepted.measurements)
        {
            EXPECT_EQ(migration.wave_wall_nanoseconds, 901u);
            EXPECT_TRUE(std::all_of(
                migration.projections.begin(),
                migration.projections.end(),
                [](const auto &projection)
                { return projection.has_value(); }));
        }
        EXPECT_TRUE(exchange.idle());

        ASSERT_TRUE(exchange.beginAttempt(
            economyAttempt(
                context->rank(),
                context->rank() != 1),
            &error))
            << error;
        const auto rejected = finishEconomyAttempt(exchange);
        EXPECT_TRUE(rejected.valid());
        EXPECT_FALSE(rejected.accepted);
        EXPECT_TRUE(rejected.measurements.empty());
        EXPECT_TRUE(exchange.idle());

        int local_participant = -1;
        for (const auto &participant : owner_map.participants())
        {
            if (participant.world_rank_known &&
                participant.world_rank == context->rank())
            {
                local_participant = participant.participant_id;
            }
        }
        ASSERT_GE(local_participant, 0);
        ASSERT_TRUE(exchange.beginServiceReadiness(
            context->rank() == 0, &error))
            << error;
        EXPECT_FALSE(finishEconomyServiceReadiness(exchange))
            << "One ready rank must not enter the service all-gather alone";
        EXPECT_TRUE(exchange.idle());

        ASSERT_TRUE(exchange.beginServiceReadiness(true, &error))
            << error;
        EXPECT_TRUE(finishEconomyServiceReadiness(exchange));
        EXPECT_TRUE(exchange.idle());

        ASSERT_TRUE(exchange.beginService(
            {economyServiceRow(local_participant, context->rank())},
            &error))
            << error;
        const auto service = finishEconomyService(exchange);
        ASSERT_EQ(service.size(), 2u);
        EXPECT_EQ(service[0].participant_id, 0);
        EXPECT_EQ(service[1].participant_id, 1);
        EXPECT_TRUE(exchange.idle());

        const auto stats = exchange.stats();
        EXPECT_EQ(stats.attempt_exchanges_started, 2u);
        EXPECT_EQ(stats.attempt_exchanges_completed, 2u);
        EXPECT_EQ(stats.service_readiness_exchanges_started, 2u);
        EXPECT_EQ(stats.service_readiness_exchanges_completed, 2u);
        EXPECT_EQ(stats.service_readiness_incomplete, 1u);
        EXPECT_EQ(stats.service_exchanges_started, 1u);
        EXPECT_EQ(stats.service_exchanges_completed, 1u);
        EXPECT_GT(stats.progress_polls, 0u);
        EXPECT_EQ(stats.rejected_packets, 0u);
        EXPECT_EQ(stats.concurrent_start_rejections, 0u);
        EXPECT_EQ(stats.mpi_failures, 0u);
    }

    TEST(
        Test__MoEOverlayMPIResidencyConsensus,
        BidirectionalRemoteCpuProjectionLanesMoveExactFinalBytesWithoutBlocking)
    {
        auto context = worldContext();
        if (!requireTwoRanks(*context))
            GTEST_SKIP() << "Remote projection data plane requires two ranks";

        constexpr std::size_t staging_bytes = 97;
        MoEOverlayMPIRemoteProjectionTransport transport({
            .mpi_context = context,
            .lane_budget = {
                .maximum_participants_per_cycle = 2,
                .maximum_concurrent_cycles = 1,
                .projections_per_expert = 3,
            },
            .staging_capacity_bytes = staging_bytes,
            .perf_device = "mpi_remote_projection_cpu_test",
        });

        auto rank_zero_source = remoteCpuProjection(17);
        auto rank_one_source = remoteCpuProjection(193);
        const auto zero_to_one = remoteProjectionIdentity(
            0, 1, 0, ExpertTierWeightProjection::Gate);
        const auto one_to_zero = remoteProjectionIdentity(
            1, 0, 1, ExpertTierWeightProjection::Up);
        const auto zero_manifest = makeMoEOverlayRemoteCpuProjectionManifest(
            zero_to_one, rank_zero_source,
            static_cast<std::uint32_t>(staging_bytes));
        const auto one_manifest = makeMoEOverlayRemoteCpuProjectionManifest(
            one_to_zero, rank_one_source,
            static_cast<std::uint32_t>(staging_bytes));

        std::vector<std::uint8_t> received_from_peer(
            context->rank() == 0
                ? rank_one_source.native_interleaved.size()
                : rank_zero_source.native_interleaved.size(),
            0xa5u);
        const auto source_lifetime = std::make_shared<int>(41);
        const auto destination_lifetime = std::make_shared<int>(73);

        std::shared_ptr<IMoEOverlayRemoteProjectionSourceEndpoint> first_source;
        std::shared_ptr<IMoEOverlayRemoteProjectionDestinationEndpoint>
            first_destination;
        std::shared_ptr<IMoEOverlayRemoteProjectionSourceEndpoint> second_source;
        std::shared_ptr<IMoEOverlayRemoteProjectionDestinationEndpoint>
            second_destination;
        if (context->rank() == 0)
        {
            first_source = std::make_shared<
                MoEOverlayHostRemoteProjectionSource>(
                zero_manifest,
                cpuSourceRegions(rank_zero_source),
                source_lifetime);
            second_destination = std::make_shared<
                MoEOverlayHostRemoteProjectionDestination>(
                one_to_zero,
                cpuDestinationRegions(received_from_peer),
                destination_lifetime);
        }
        else
        {
            first_destination = std::make_shared<
                MoEOverlayHostRemoteProjectionDestination>(
                zero_to_one,
                cpuDestinationRegions(received_from_peer),
                destination_lifetime);
            second_source = std::make_shared<
                MoEOverlayHostRemoteProjectionSource>(
                one_manifest,
                cpuSourceRegions(rank_one_source),
                source_lifetime);
        }

        auto first = transport.reserve(
            /*lane_index=*/0,
            zero_to_one,
            first_source,
            first_destination);
        ASSERT_EQ(first.status, MoEOverlayResidencyStageStartStatus::Started)
            << first.error;
        ASSERT_NE(first.operation, nullptr);
        auto second = transport.reserve(
            /*lane_index=*/4,
            one_to_zero,
            second_source,
            second_destination);
        ASSERT_EQ(second.status, MoEOverlayResidencyStageStartStatus::Started)
            << second.error;
        ASSERT_NE(second.operation, nullptr);

        /* Reservation owns the lane but has not posted one MPI operation yet. */
        std::shared_ptr<IMoEOverlayRemoteProjectionSourceEndpoint> duplicate_source;
        std::shared_ptr<IMoEOverlayRemoteProjectionDestinationEndpoint>
            duplicate_destination;
        if (context->rank() == 0)
        {
            duplicate_source = std::make_shared<
                MoEOverlayHostRemoteProjectionSource>(
                zero_manifest,
                cpuSourceRegions(rank_zero_source),
                source_lifetime);
        }
        else
        {
            duplicate_destination = std::make_shared<
                MoEOverlayHostRemoteProjectionDestination>(
                zero_to_one,
                cpuDestinationRegions(received_from_peer),
                destination_lifetime);
        }
        auto duplicate = transport.reserve(
            0, zero_to_one, duplicate_source, duplicate_destination);
        EXPECT_EQ(
            duplicate.status,
            MoEOverlayResidencyStageStartStatus::Deferred);
        EXPECT_EQ(duplicate.operation, nullptr);

        /*
         * A complete wave containing one free lane and retained lane 4 must
         * reserve neither. Acquiring lane 2 immediately afterward proves the
         * failed batch did not leave a partial process-local reservation.
         */
        const auto atomic_free_identity = remoteProjectionIdentity(
            0, 1, 2, ExpertTierWeightProjection::Down);
        const auto atomic_free_manifest =
            makeMoEOverlayRemoteCpuProjectionManifest(
                atomic_free_identity,
                rank_zero_source,
                static_cast<std::uint32_t>(staging_bytes));
        std::vector<std::uint8_t> atomic_destination(
            rank_zero_source.native_interleaved.size(), 0x3cu);
        std::vector<MoEOverlayMPIRemoteProjectionBinding> blocked_bindings;
        blocked_bindings.push_back(localHostBinding(
            context->rank(),
            2,
            atomic_free_identity,
            atomic_free_manifest,
            rank_zero_source,
            atomic_destination,
            source_lifetime,
            destination_lifetime));
        blocked_bindings.push_back(localHostBinding(
            context->rank(),
            4,
            one_to_zero,
            one_manifest,
            rank_one_source,
            received_from_peer,
            source_lifetime,
            destination_lifetime));
        auto blocked_wave = transport.reserveWave(
            std::move(blocked_bindings));
        EXPECT_EQ(
            blocked_wave.status,
            MoEOverlayResidencyStageStartStatus::Deferred);
        EXPECT_TRUE(blocked_wave.operations.empty());

        auto free_binding = localHostBinding(
            context->rank(),
            2,
            atomic_free_identity,
            atomic_free_manifest,
            rank_zero_source,
            atomic_destination,
            source_lifetime,
            destination_lifetime);
        auto free_after_failed_batch = transport.reserve(
            free_binding.lane_index,
            free_binding.identity,
            std::move(free_binding.source),
            std::move(free_binding.destination));
        ASSERT_EQ(
            free_after_failed_batch.status,
            MoEOverlayResidencyStageStartStatus::Started)
            << free_after_failed_batch.error;
        ASSERT_NE(free_after_failed_batch.operation, nullptr);
        free_after_failed_batch.operation->abort();
        std::string abort_error;
        EXPECT_EQ(
            free_after_failed_batch.operation->pollAbort(&abort_error),
            MoEOverlayResidencyWaveProgress::Ready)
            << abort_error;

        bool first_ready = false;
        bool second_ready = false;
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while ((!first_ready || !second_ready) &&
               std::chrono::steady_clock::now() < deadline)
        {
            std::string error;
            if (!first_ready)
            {
                const auto progress = first.operation->poll(&error);
                ASSERT_NE(progress, MoEOverlayResidencyWaveProgress::Failed)
                    << error;
                first_ready =
                    progress == MoEOverlayResidencyWaveProgress::Ready;
            }
            if (!second_ready)
            {
                const auto progress = second.operation->poll(&error);
                ASSERT_NE(progress, MoEOverlayResidencyWaveProgress::Failed)
                    << error;
                second_ready =
                    progress == MoEOverlayResidencyWaveProgress::Ready;
            }
            std::this_thread::yield();
        }
        ASSERT_TRUE(first_ready);
        ASSERT_TRUE(second_ready);

        const auto first_measurement =
            first.operation->completedMeasurement();
        const auto second_measurement =
            second.operation->completedMeasurement();
        ASSERT_TRUE(first_measurement.has_value());
        ASSERT_TRUE(second_measurement.has_value());
        EXPECT_TRUE(first_measurement->valid());
        EXPECT_TRUE(second_measurement->valid());
        EXPECT_EQ(
            first_measurement->bytes,
            rank_zero_source.native_interleaved.size());
        EXPECT_EQ(
            second_measurement->bytes,
            rank_one_source.native_interleaved.size());
        EXPECT_GT(first_measurement->transport_nanoseconds, 0u);
        EXPECT_GT(second_measurement->transport_nanoseconds, 0u);
        EXPECT_EQ(first_measurement->device_nanoseconds, 0u);
        EXPECT_EQ(second_measurement->device_nanoseconds, 0u);
        // The MPI lane now reports the bounded endpoint/protocol work instead
        // of hiding it inside a monolithic transport wall measurement.  This
        // CPU-to-CPU path must therefore contain real host work but no device
        // work.  Each component remains bounded by the observed operation wall;
        // components are not added because request progress and endpoint work
        // may overlap on asynchronous implementations.
        EXPECT_GT(first_measurement->host_nanoseconds, 0u);
        EXPECT_GT(second_measurement->host_nanoseconds, 0u);
        EXPECT_LE(
            first_measurement->host_nanoseconds,
            first_measurement->wall_nanoseconds);
        EXPECT_LE(
            second_measurement->host_nanoseconds,
            second_measurement->wall_nanoseconds);
        EXPECT_LE(
            first_measurement->transport_nanoseconds,
            first_measurement->wall_nanoseconds);
        EXPECT_LE(
            second_measurement->transport_nanoseconds,
            second_measurement->wall_nanoseconds);

        const auto &expected = context->rank() == 0
                                   ? rank_one_source.native_interleaved
                                   : rank_zero_source.native_interleaved;
        EXPECT_TRUE(std::equal(
            received_from_peer.begin(),
            received_from_peer.end(),
            expected.begin(),
            expected.end()));

        const auto stats = transport.stats();
        EXPECT_EQ(stats.lanes_materialized, 6u);
        EXPECT_EQ(stats.wave_reservations_started, 3u);
        EXPECT_EQ(stats.wave_reservations_deferred, 2u);
        EXPECT_EQ(stats.wave_reservations_failed, 0u);
        EXPECT_EQ(stats.reservations_started, 3u);
        EXPECT_EQ(stats.reservations_deferred, 2u);
        EXPECT_EQ(
            stats.source_operations,
            context->rank() == 0 ? 2u : 1u);
        EXPECT_EQ(
            stats.destination_operations,
            context->rank() == 0 ? 1u : 2u);
        EXPECT_EQ(stats.operations_completed, 2u);
        EXPECT_EQ(stats.operations_aborted, 1u);
        EXPECT_GT(stats.chunks_sent, 1u);
        EXPECT_GT(stats.chunks_received, 1u);
        EXPECT_EQ(stats.bytes_sent, expected.size());
        EXPECT_EQ(stats.bytes_received, received_from_peer.size());
        EXPECT_EQ(stats.mpi_failures, 0u);
        EXPECT_EQ(stats.protocol_failures, 0u);
        EXPECT_EQ(stats.inference_stream_waits, 0u);
        EXPECT_EQ(stats.blocking_synchronizations, 0u);
    }

    TEST(
        Test__MoEOverlayMPIResidencyConsensus,
        DistributedPhysicalCpuFabricPublishesByteExactRCUBanks)
    {
        auto context = worldContext();
        if (!requireTwoRanks(*context))
            GTEST_SKIP() << "Distributed physical fabric requires two ranks";

        const auto plan = distributedCpuOverlayPlan();
        const auto owner_map = MoEExpertOwnerMap::build(plan);
        DecodeExpertHistogramConfig histogram_config;
        histogram_config.num_layers = 1;
        histogram_config.num_experts = 2;
        histogram_config.top_k = 1;
        histogram_config.window_size = 2;
        histogram_config.sockets = {
            DeviceId::cpu(),
            DeviceId::cpu(),
        };
        histogram_config.ownership = owner_map.layeredOwnership(1, 2);
        auto histogram = std::make_shared<DecodeExpertHistogram>(
            histogram_config);
        const std::uint64_t counts[]{1, 100};
        histogram->mergeLayerCounts(0, counts, 2, false);

        auto authority = std::make_shared<MoEOverlayResidencyAuthority>(
            MoEOverlayResidencyAuthority::Config{
                .initial_plan = plan,
                .model_metadata = {
                    .num_layers = 1,
                    .num_experts = 2,
                    .d_model = 64,
                    .routed_intermediate_size = 32,
                    .routed_quant_type = "Q4_0",
                },
                .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
                .histogram = histogram.get(),
                .perf_device = "mpi-distributed-physical-cpu",
            });
        const auto initial_snapshot = authority->snapshot();
        ASSERT_NE(initial_snapshot, nullptr);

        std::vector<int> local_participants;
        for (const auto &participant :
             initial_snapshot->owner_map.participants())
        {
            if (participant.world_rank_known &&
                participant.world_rank == context->rank())
            {
                local_participants.push_back(participant.participant_id);
            }
        }
        ASSERT_EQ(local_participants.size(), 1u);
        const int local_participant = local_participants.front();
        const auto initially_owned =
            initial_snapshot->owner_map.expertsForParticipant(
                0, local_participant);
        ASSERT_EQ(initially_owned.size(), 1u);
        const int local_expert = initially_owned.front();
        const auto seed_for_expert = [](int expert_id)
        {
            return static_cast<std::uint8_t>(
                expert_id == 0 ? 11u : 177u);
        };
        const auto source_expert = distributedCpuSourceExpert(
            local_participant,
            local_expert,
            seed_for_expert(local_expert));
        auto registry = distributedCpuRegistry(
            initial_snapshot,
            local_participant,
            local_expert,
            source_expert);

        constexpr std::size_t staging_bytes = 64;
        auto remote_projection_transport = std::make_shared<
            MoEOverlayMPIRemoteProjectionTransport>(
            MoEOverlayMPIRemoteProjectionTransport::Config{
                .mpi_context = context,
                .lane_budget = {
                    .maximum_participants_per_cycle = 2,
                    .maximum_concurrent_cycles = 1,
                    .projections_per_expert = 3,
                },
                .staging_capacity_bytes = staging_bytes,
                .perf_device = "mpi-distributed-physical-cpu",
            });
        auto fabric = MoEOverlayPhysicalResidencyFabric::create({
            .registry = registry,
            .initial_snapshot = initial_snapshot,
            .remote_projection_transport = remote_projection_transport,
            .shadow_slots_per_endpoint_layer = 1,
            .staging_capacity_bytes = staging_bytes,
            .perf_device = "mpi-distributed-physical-cpu",
        });
        MoEOverlayParticipantPreparedWaveFactory factory({
            .registry = registry,
            .transfer_provider = fabric,
            .perf_device = "mpi-distributed-physical-cpu",
        });
        MoEOverlayTierMigrationTransport local_transport({
            .factory = &factory,
            .projections_per_expert = 3,
            .perf_device = "mpi-distributed-physical-cpu",
        });
        auto consensus = std::make_shared<
            MoEOverlayMPIResidencyConsensus>(
            MoEOverlayMPIResidencyConsensus::Config{
                .mpi_context = context,
                .perf_device = "mpi-distributed-physical-cpu",
            });
        MoEOverlayDistributedResidencyTransport distributed_transport({
            .local_transport = &local_transport,
            .consensus = consensus,
            .perf_device = "mpi-distributed-physical-cpu",
        });

        const auto transaction = authority->proposeFromHistogram();
        ASSERT_TRUE(transaction.valid());
        ASSERT_EQ(transaction.migrations.size(), 2u);
        EXPECT_TRUE(std::all_of(
            transaction.migrations.begin(),
            transaction.migrations.end(),
            [](const auto &migration)
            {
                return migration.crossesWorldRank() &&
                       !migration.crossesBackend();
            }));

        auto progress = authority->beginApply(
            transaction, distributed_transport);
        ASSERT_EQ(progress.status, MoEOverlayResidencyApplyStatus::Started)
            << progress.error;
        bool committed = false;
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < deadline)
        {
            progress = authority->advanceBackground();
            ASSERT_TRUE(progress.ok()) << progress.error;
            if (progress.status == MoEOverlayResidencyApplyStatus::Committed)
            {
                committed = true;
                break;
            }
            std::this_thread::yield();
        }
        ASSERT_TRUE(committed)
            << "Distributed physical fabric did not publish within five seconds";

        const auto candidate_bank = registry->endpoint(local_participant)
                                        ->acquire(
                                            transaction.candidate->epoch);
        ASSERT_NE(candidate_bank, nullptr);
        EXPECT_EQ(
            candidate_bank->layers[0].resident_mask,
            transaction.candidate->owner_map.expertMaskForParticipant(
                0, local_participant, 2));

        const auto incoming = std::find_if(
            transaction.migrations.begin(),
            transaction.migrations.end(),
            [local_participant](const auto &migration)
            {
                return migration.destination.owner_participant ==
                       local_participant;
            });
        ASSERT_NE(incoming, transaction.migrations.end());
        const auto &arrived = candidate_bank->layers[0]
                                  .experts[static_cast<std::size_t>(
                                      incoming->expert_id)];
        ASSERT_TRUE(arrived.complete());
        EXPECT_TRUE(distributedCpuBytesMatchSeed(
            arrived, seed_for_expert(incoming->expert_id)))
            << "Published remote CPU projection differs from exact source bytes";

        std::string retirement_error;
        ASSERT_TRUE(drainAuthorityRetirements(
            *authority, &retirement_error))
            << retirement_error;

        EXPECT_EQ(
            registry->endpoint(local_participant)
                ->acquire(initial_snapshot->epoch),
            nullptr)
            << "The old local bank must retire after global publication and ticket drain";

        const auto fabric_stats = fabric->stats();
        EXPECT_EQ(fabric_stats.waves_prepared, 1u);
        EXPECT_EQ(fabric_stats.projection_operations_prepared, 6u);
        EXPECT_EQ(fabric_stats.cpu_copy_operations, 0u);
        EXPECT_EQ(fabric_stats.remote_cpu_operations, 6u);
        EXPECT_EQ(fabric_stats.inference_stream_waits, 0u);
        EXPECT_EQ(fabric_stats.blocking_synchronizations, 0u);

        const auto projection_stats = remote_projection_transport->stats();
        EXPECT_EQ(projection_stats.wave_reservations_started, 1u);
        EXPECT_EQ(projection_stats.reservations_started, 6u);
        EXPECT_EQ(projection_stats.source_operations, 3u);
        EXPECT_EQ(projection_stats.destination_operations, 3u);
        EXPECT_EQ(projection_stats.operations_completed, 6u);
        EXPECT_GT(projection_stats.bytes_sent, 0u);
        EXPECT_EQ(
            projection_stats.bytes_sent,
            projection_stats.bytes_received);
        EXPECT_EQ(projection_stats.mpi_failures, 0u);
        EXPECT_EQ(projection_stats.protocol_failures, 0u);
        EXPECT_EQ(projection_stats.inference_stream_waits, 0u);
        EXPECT_EQ(projection_stats.blocking_synchronizations, 0u);

        const auto distributed_stats = distributed_transport.stats();
        EXPECT_EQ(distributed_stats.waves_started, 1u);
        EXPECT_EQ(distributed_stats.reservation_consensus_ready, 1u);
        EXPECT_EQ(distributed_stats.stage_consensus_ready, 1u);
        EXPECT_EQ(distributed_stats.commit_consensus_ready, 1u);
        EXPECT_EQ(distributed_stats.waves_published, 1u);
        EXPECT_EQ(distributed_stats.inference_thread_waits, 0u);
        EXPECT_EQ(distributed_stats.blocking_synchronizations, 0u);

        const auto authority_stats = authority->stats();
        EXPECT_EQ(authority_stats.committed_waves, 1u);
        EXPECT_EQ(authority_stats.committed_migrations, 2u);
        EXPECT_EQ(authority_stats.cross_rank_migrations, 2u);
        EXPECT_EQ(authority->pendingRetirementCount(), 0u);
    }

    TEST(
        Test__MoEOverlayMPIResidencyConsensus,
        DistributedEconomyCertificationUsesRealAbortOnlyMovementBeforeProposals)
    {
        auto context = worldContext();
        if (!requireTwoRanks(*context))
            GTEST_SKIP() << "Distributed certification requires two ranks";

        const auto plan = distributedCpuOverlayPlan();
        const auto owner_map = MoEExpertOwnerMap::build(plan);
        DecodeExpertHistogramConfig histogram_config;
        histogram_config.num_layers = 1;
        histogram_config.num_experts = 2;
        histogram_config.top_k = 1;
        histogram_config.window_size = 2;
        histogram_config.token_boundary_layer_idx = 0;
        histogram_config.sockets = {DeviceId::cpu(), DeviceId::cpu()};
        histogram_config.ownership = owner_map.layeredOwnership(1, 2);
        auto histogram = std::make_shared<DecodeExpertHistogram>(
            histogram_config);
        auto authority = std::make_shared<MoEOverlayResidencyAuthority>(
            MoEOverlayResidencyAuthority::Config{
                .initial_plan = plan,
                .model_metadata = {
                    .num_layers = 1,
                    .num_experts = 2,
                    .d_model = 64,
                    .routed_intermediate_size = 32,
                    .routed_quant_type = "Q4_0",
                },
                .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
                .histogram = histogram.get(),
                .perf_device = "mpi-distributed-economy-cpu",
            });
        ASSERT_FALSE(authority->hasEconomyCertification());
        EXPECT_EQ(authority->stats().checks, 0u);
        const auto initial_snapshot = authority->snapshot();
        ASSERT_NE(initial_snapshot, nullptr);

        int local_participant = -1;
        for (const auto &participant : owner_map.participants())
        {
            if (participant.world_rank_known &&
                participant.world_rank == context->rank())
            {
                local_participant = participant.participant_id;
            }
        }
        ASSERT_GE(local_participant, 0);
        const auto local_experts = owner_map.expertsForParticipant(
            0, local_participant);
        ASSERT_EQ(local_experts.size(), 1u);
        const int local_expert = local_experts.front();
        const auto source_expert = distributedCpuSourceExpert(
            local_participant,
            local_expert,
            static_cast<std::uint8_t>(
                local_expert == 0 ? 29u : 181u));
        auto registry = distributedCpuRegistry(
            initial_snapshot,
            local_participant,
            local_expert,
            source_expert,
            /*collect_economy_service_measurements=*/true);
        const auto endpoint = registry->endpoint(local_participant);
        ASSERT_NE(endpoint, nullptr);
        const auto recordLocalServiceEvidence = [&endpoint, local_participant]
        {
            for (const auto source : {
                     ExpertHistogramSource::DecodeToken,
                     ExpertHistogramSource::PrefillChunk,
                     ExpertHistogramSource::GroupedVerifier})
            {
                ASSERT_EQ(
                    endpoint->recordServiceMeasurement(
                        0,
                        source,
                        local_participant == 0 ? 10u : 1'000u,
                        1),
                    MoEOverlayServiceMeasurementRecordStatus::Recorded);
            }
        };
        /*
         * Rank zero deliberately starts without service evidence. The first
         * all-rank readiness vote must complete false instead of allowing rank
         * one to enter the larger service all-gather alone.
         */
        bool local_service_recorded = context->rank() != 0;
        if (local_service_recorded)
            recordLocalServiceEvidence();

        constexpr std::size_t staging_bytes = 64;
        auto remote_projection_transport = std::make_shared<
            MoEOverlayMPIRemoteProjectionTransport>(
            MoEOverlayMPIRemoteProjectionTransport::Config{
                .mpi_context = context,
                .lane_budget = {
                    .maximum_participants_per_cycle = 2,
                    .maximum_concurrent_cycles = 1,
                    .projections_per_expert = 3,
                },
                .staging_capacity_bytes = staging_bytes,
                .perf_device = "mpi-distributed-economy-cpu",
            });
        auto consensus =
            std::make_shared<MoEOverlayMPIResidencyConsensus>(
                MoEOverlayMPIResidencyConsensus::Config{
                    .mpi_context = context,
                    .perf_device = "mpi-distributed-economy-cpu",
                });
        auto evidence_exchange = std::make_shared<
            MoEOverlayMPIEconomyEvidenceExchange>(
            MoEOverlayMPIEconomyEvidenceExchange::Config{
                .mpi_context = context,
                .owner_map = owner_map,
                .num_layers = 1,
                .perf_device = "mpi-distributed-economy-cpu",
            });
        auto fabric = MoEOverlayPhysicalResidencyFabric::create({
            .registry = registry,
            .initial_snapshot = initial_snapshot,
            .remote_projection_transport = remote_projection_transport,
            .shadow_slots_per_endpoint_layer = 1,
            .staging_capacity_bytes = staging_bytes,
            .collect_economy_measurements = true,
            .perf_device = "mpi-distributed-economy-cpu",
        });
        MoEOverlayParticipantPreparedWaveFactory factory({
            .registry = registry,
            .transfer_provider = fabric,
            .perf_device = "mpi-distributed-economy-cpu",
        });
        auto journal = std::make_shared<
            MoEOverlayMigrationMeasurementJournal>(
            MoEOverlayMigrationMeasurementJournal::Config{
                .maximum_migrations_per_wave = 2,
            });
        auto local_transport =
            std::make_shared<MoEOverlayTierMigrationTransport>(
                MoEOverlayTierMigrationTransport::Config{
                    .factory = &factory,
                    .projections_per_expert = 3,
                    .measurement_sink = journal,
                    .measurement_scope =
                        MoEOverlayMigrationMeasurementScope::
                            EconomyCalibrationOnly,
                    .require_complete_local_measurements = false,
                    .perf_device = "mpi-distributed-economy-cpu",
                });
        auto distributed_transport = std::make_shared<
            MoEOverlayDistributedResidencyTransport>(
            MoEOverlayDistributedResidencyTransport::Config{
                .local_transport = local_transport.get(),
                .consensus = consensus,
                .perf_device = "mpi-distributed-economy-cpu",
            });

        auto catalog = std::make_shared<
            MoEOverlayEconomyCalibrationLayerCatalog>(
            distributedCpuLayerManifest());
        auto planner =
            std::make_shared<MoEOverlayEconomyCalibrationPlanner>(
                MoEOverlayEconomyCalibrationPlanner::Config{
                    .live_snapshot = initial_snapshot,
                    .complete_expert_bytes_per_layer =
                        catalog->completeExpertBytesPerLayer(),
                    .calibration_layers =
                        catalog->representativeLayers(),
                });
        auto ledger = std::make_shared<
            MoEOverlayMigrationMeasurementLedger>(
            MoEOverlayMigrationMeasurementLedger::Config{
                .required_coordinates = planner->requiredCoordinates(),
                .warmup_samples_per_coordinate = 0,
                .measured_samples_per_coordinate =
                    MoEOverlayEconomyProfileComposer::
                        kMinimumMigrationSamples,
                .measurement_identity = catalog->identity(),
            });
        auto probe =
            std::make_shared<MoEOverlayInferenceInterferenceProbe>();
        auto calibration = std::make_shared<
            MoEOverlayEconomyCalibrationController>(
            MoEOverlayEconomyCalibrationController::Config{
                .planner = planner,
                .ledger = ledger,
                .journal = journal,
                .probe = probe,
                .transport = distributed_transport,
                .evidence_exchange = evidence_exchange,
                .perf_device = "mpi-distributed-economy-cpu",
            });
        MoEOverlayEconomyCertificationController certification({
            .calibration = calibration,
            .registry = registry,
            .layer_catalog = catalog,
            .authority = authority,
            .model_metadata = {
                .num_layers = 1,
                .num_experts = 2,
                .d_model = 64,
                .routed_intermediate_size = 32,
                .routed_quant_type = "Q4_0",
            },
            .economy_policy = {
                .historical_window_weight = 0,
                .current_window_weight = 1,
                .payoff_horizon_tokens = 1'000'000,
                .minimum_residency_generations = 0,
            },
            .evidence_exchange = evidence_exchange,
            .perf_device = "mpi-distributed-economy-cpu",
        });

        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (certification.state() !=
                   MoEOverlayEconomyCertificationState::Complete &&
               certification.healthy() &&
               std::chrono::steady_clock::now() < deadline)
        {
            certification.poll();
            if (!local_service_recorded &&
                evidence_exchange->stats()
                        .service_readiness_incomplete != 0)
            {
                recordLocalServiceEvidence();
                local_service_recorded = true;
            }
            runEconomyInferencePhases(
                *probe, *calibration, certification);
            std::this_thread::yield();
        }
        ASSERT_TRUE(certification.healthy())
            << certification.failureMessage();
        ASSERT_EQ(
            certification.state(),
            MoEOverlayEconomyCertificationState::Complete);
        EXPECT_TRUE(authority->hasEconomyCertification());
        EXPECT_EQ(authority->stats().checks, 0u)
            << "Certification must precede the first histogram proposal";
        EXPECT_EQ(authority->snapshot()->epoch, initial_snapshot->epoch)
            << "Calibration waves are never publishable";
        EXPECT_TRUE(probe->idle());
        EXPECT_TRUE(evidence_exchange->idle());

        const auto calibration_stats = calibration->stats();
        EXPECT_EQ(calibration_stats.accepted_pairs, 9u);
        EXPECT_EQ(calibration_stats.exact_overlap_samples, 9u);
        EXPECT_EQ(calibration_stats.partial_overlap_rejections, 0u);
        EXPECT_EQ(calibration_stats.waves_started, 9u);
        EXPECT_EQ(calibration_stats.waves_aborted, 9u);
        EXPECT_EQ(fabric->stats().waves_prepared, 9u);
        EXPECT_EQ(local_transport->stats().commits_started, 0u);
        EXPECT_EQ(local_transport->stats().waves_aborted, 9u);
        EXPECT_EQ(
            distributed_transport->stats().waves_published,
            0u);
        EXPECT_EQ(
            distributed_transport->stats().blocking_synchronizations,
            0u);
        EXPECT_EQ(
            remote_projection_transport->stats().mpi_failures,
            0u);
        EXPECT_EQ(evidence_exchange->stats().mpi_failures, 0u);
        EXPECT_GE(
            evidence_exchange->stats()
                .service_readiness_exchanges_started,
            2u);
        EXPECT_GE(
            evidence_exchange->stats().service_readiness_incomplete,
            1u);
        EXPECT_EQ(
            certification.stats().certifications_installed,
            1u);
    }

    TEST(
        Test__MoEOverlayMPIResidencyConsensus,
        ArbitraryCoordinatorPublishesAuthenticatedFrozenWindow)
    {
        auto context = worldContext();
        if (!requireTwoRanks(*context))
            GTEST_SKIP() << "Histogram publication requires exactly two ranks";

        /* Rank one is deliberate: hardware/routing ownership is not rank zero. */
        constexpr int coordinator_world_rank = 1;
        MoEOverlayMPIHistogramPublisher publisher({
            .mpi_context = context,
            .coordinator_world_rank = coordinator_world_rank,
            .num_layers = 2,
            .num_experts = 3,
            .perf_device = "mpi_cpu_test",
        });

        DecodeExpertHistogramWindow source{
            .generation = 12,
            .token_count = 256,
            .source_token_counts = {256, 0, 0},
            .num_layers = 2,
            .num_experts = 3,
            .expert_counts = {9, 8, 7, 60, 50, 40},
            .source_expert_counts = {
                9, 8, 7, 60, 50, 40,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
            },
        };
        if (publisher.isCoordinator())
        {
            std::string error;
            ASSERT_TRUE(publisher.beginPublish(source, &error)) << error;
        }

        const auto received = finishHistogramPublication(publisher);
        if (publisher.isCoordinator())
        {
            EXPECT_EQ(received, nullptr);
            EXPECT_EQ(
                publisher.state(),
                MoEOverlayMPIHistogramPublisherState::Idle);
            const auto stats = publisher.stats();
            EXPECT_EQ(stats.publications_started, 1u);
            EXPECT_EQ(stats.publications_completed, 1u);
            EXPECT_EQ(
                stats.bytes_sent,
                moeOverlayDistributedHistogramWireBytes(2, 3));
        }
        else
        {
            ASSERT_NE(received, nullptr);
            EXPECT_EQ(received->generation, source.generation);
            EXPECT_EQ(received->token_count, source.token_count);
            EXPECT_EQ(received->expert_counts, source.expert_counts);
            const auto stats = publisher.stats();
            EXPECT_EQ(stats.receives_armed, 1u);
            EXPECT_EQ(stats.windows_received, 1u);
            EXPECT_EQ(
                stats.bytes_received,
                moeOverlayDistributedHistogramWireBytes(2, 3));
        }
        EXPECT_EQ(publisher.stats().blocking_inference_waits, 0u);
    }

    TEST(
        Test__MoEOverlayMPIResidencyConsensus,
        UnanimousStageAndCommitVotesCompleteOnPrivateLane)
    {
        auto context = worldContext();
        if (!requireTwoRanks(*context))
            GTEST_SKIP() << "Consensus integration requires exactly two ranks";

        MoEOverlayMPIResidencyConsensus lane({
            .mpi_context = context,
            .perf_device = "mpi_cpu_test",
        });
        MoEOverlayDistributedResidencyProtocol protocol({
            .identity = identity(),
            .local_world_rank = context->rank(),
            .world_size = context->world_size(),
        });

        std::string error;
        const auto reservation_vote = protocol.makeLocalVote(
            MoEOverlayDistributedResidencyVoteDecision::Ready);
        ASSERT_TRUE(lane.begin(reservation_vote, &error)) << error;
        const auto reservation_votes = finishExchange(lane);
        ASSERT_EQ(reservation_votes.size(), 2u);
        ASSERT_TRUE(protocol.acceptConsensus(reservation_votes, &error))
            << error;
        EXPECT_EQ(
            protocol.state(),
            MoEOverlayDistributedResidencyProtocolState::
                AwaitingLocalStage);

        const auto stage_vote = protocol.makeLocalVote(
            MoEOverlayDistributedResidencyVoteDecision::Ready);
        ASSERT_TRUE(lane.begin(stage_vote, &error)) << error;
        const auto stage_votes = finishExchange(lane);
        ASSERT_EQ(stage_votes.size(), 2u);
        ASSERT_TRUE(protocol.acceptConsensus(stage_votes, &error)) << error;
        EXPECT_EQ(
            protocol.state(),
            MoEOverlayDistributedResidencyProtocolState::
                AwaitingLocalCommit);

        const auto commit_vote = protocol.makeLocalVote(
            MoEOverlayDistributedResidencyVoteDecision::Ready);
        ASSERT_TRUE(lane.begin(commit_vote, &error)) << error;
        const auto commit_votes = finishExchange(lane);
        ASSERT_EQ(commit_votes.size(), 2u);
        ASSERT_TRUE(protocol.acceptConsensus(commit_votes, &error)) << error;
        EXPECT_EQ(
            protocol.state(),
            MoEOverlayDistributedResidencyProtocolState::ReadyToPublish);
        protocol.markPublished();

        const auto stats = lane.stats();
        EXPECT_EQ(stats.exchanges_started, 3u);
        EXPECT_EQ(stats.exchanges_completed, 3u);
        EXPECT_GT(stats.progress_polls, 0u);
        EXPECT_EQ(stats.mpi_failures, 0u);
        EXPECT_TRUE(lane.idle());
    }

    TEST(
        Test__MoEOverlayMPIResidencyConsensus,
        RemoteStageFailureAbortsEveryRankBeforeCommit)
    {
        auto context = worldContext();
        if (!requireTwoRanks(*context))
            GTEST_SKIP() << "Consensus integration requires exactly two ranks";

        MoEOverlayMPIResidencyConsensus lane({
            .mpi_context = context,
            .perf_device = "mpi_cpu_test",
        });
        MoEOverlayDistributedResidencyProtocol protocol({
            .identity = identity(),
            .local_world_rank = context->rank(),
            .world_size = context->world_size(),
        });

        std::string error;
        const auto reservation_vote = protocol.makeLocalVote(
            MoEOverlayDistributedResidencyVoteDecision::Ready);
        ASSERT_TRUE(lane.begin(reservation_vote, &error)) << error;
        const auto reservation_votes = finishExchange(lane);
        ASSERT_EQ(reservation_votes.size(), 2u);
        ASSERT_TRUE(protocol.acceptConsensus(reservation_votes, &error))
            << error;

        const bool injected_failure = context->rank() == 1;
        const auto vote = protocol.makeLocalVote(
            injected_failure
                ? MoEOverlayDistributedResidencyVoteDecision::Failed
                : MoEOverlayDistributedResidencyVoteDecision::Ready,
            injected_failure ? 61 : 0,
            injected_failure
                ? "injected rank-one physical transfer failure"
                : std::string{});
        ASSERT_TRUE(lane.begin(vote, &error)) << error;
        const auto votes = finishExchange(lane);
        ASSERT_EQ(votes.size(), 2u);
        EXPECT_FALSE(protocol.acceptConsensus(votes, &error));
        EXPECT_NE(error.find("world rank 1"), std::string::npos);
        ASSERT_TRUE(protocol.failure().has_value());
        EXPECT_EQ(protocol.failure()->world_rank, 1);
        EXPECT_EQ(protocol.failure()->error_code, 61);
        EXPECT_THROW(protocol.markPublished(), std::logic_error);
        EXPECT_TRUE(lane.idle());
    }

    TEST(
        Test__MoEOverlayMPIResidencyConsensus,
        DivergentRankIdentityIsVisibleToEveryParticipant)
    {
        auto context = worldContext();
        if (!requireTwoRanks(*context))
            GTEST_SKIP() << "Consensus integration requires exactly two ranks";

        const auto local_identity = identity(
            context->rank() == 0
                ? 0x1122334455667788ull
                : 0x1122334455667799ull);
        MoEOverlayMPIResidencyConsensus lane({
            .mpi_context = context,
            .perf_device = "mpi_cpu_test",
        });
        MoEOverlayDistributedResidencyProtocol protocol({
            .identity = local_identity,
            .local_world_rank = context->rank(),
            .world_size = context->world_size(),
        });

        const auto vote = protocol.makeLocalVote(
            MoEOverlayDistributedResidencyVoteDecision::Ready);
        std::string error;
        ASSERT_TRUE(lane.begin(vote, &error)) << error;
        const auto votes = finishExchange(lane);
        ASSERT_EQ(votes.size(), 2u);
        EXPECT_FALSE(protocol.acceptConsensus(votes, &error));
        EXPECT_NE(error.find("disagree"), std::string::npos);
        EXPECT_EQ(
            protocol.state(),
            MoEOverlayDistributedResidencyProtocolState::Failed);
        EXPECT_TRUE(lane.idle());
    }

    TEST(
        Test__MoEOverlayMPIResidencyConsensus,
        DistributedTransportDefersAllRanksThenRetriesExactWave)
    {
        auto context = worldContext();
        if (!requireTwoRanks(*context))
            GTEST_SKIP() << "Distributed transport requires exactly two ranks";

        auto fixture = realTransaction();
        ImmediateLocalResidencyTransport local;
        local.defer_start = context->rank() == 1;
        auto lane = std::make_shared<MoEOverlayMPIResidencyConsensus>(
            MoEOverlayMPIResidencyConsensus::Config{
                .mpi_context = context,
                .perf_device = "real_mpi_transport_test",
            });
        MoEOverlayDistributedResidencyTransport transport({
            .local_transport = &local,
            .consensus = lane,
            .perf_device = "real_mpi_transport_test",
        });

        auto result = fixture.authority->beginApply(
            fixture.transaction,
            transport);
        ASSERT_EQ(result.status, MoEOverlayResidencyApplyStatus::Started)
            << result.error;
        result = finishAuthorityWave(*fixture.authority);
        ASSERT_EQ(result.status, MoEOverlayResidencyApplyStatus::Deferred)
            << result.error;
        EXPECT_EQ(fixture.authority->snapshot()->epoch, 1u);
        EXPECT_TRUE(lane->idle());

        local.defer_start = false;
        result = fixture.authority->beginApply(
            fixture.transaction,
            transport);
        ASSERT_EQ(result.status, MoEOverlayResidencyApplyStatus::Started)
            << result.error;
        result = finishAuthorityWave(*fixture.authority);
        ASSERT_EQ(result.status, MoEOverlayResidencyApplyStatus::Committed)
            << result.error;
        EXPECT_EQ(fixture.authority->snapshot()->epoch, 2u);
        ASSERT_EQ(local.fingerprints.size(), 2u);
        EXPECT_EQ(local.fingerprints[0], local.fingerprints[1]);

        /*
         * Publication may immediately start the non-blocking retirement vote,
         * so the shared lane is not required to be idle at the return edge of
         * Committed. Poll the maintenance authority until both ranks have
         * drained that exact fence; inference remains absent from this loop.
         */
        std::string retirement_error;
        ASSERT_TRUE(drainAuthorityRetirements(
            *fixture.authority, &retirement_error))
            << retirement_error;
        EXPECT_TRUE(lane->idle());
        EXPECT_EQ(local.retirements, 1);
        const auto stats = transport.stats();
        EXPECT_EQ(stats.reservation_consensus_deferred, 1u);
        EXPECT_EQ(stats.reservation_consensus_ready, 1u);
        EXPECT_EQ(stats.stage_consensus_ready, 1u);
        EXPECT_EQ(stats.commit_consensus_ready, 1u);
        EXPECT_EQ(stats.waves_published, 1u);
        EXPECT_EQ(stats.inference_thread_waits, 0u);
        EXPECT_EQ(stats.blocking_synchronizations, 0u);
    }

    TEST(
        Test__MoEOverlayMPIResidencyConsensus,
        DistributedCommitStartFailureReachesEveryRankWithoutHang)
    {
        auto context = worldContext();
        if (!requireTwoRanks(*context))
            GTEST_SKIP() << "Distributed transport requires exactly two ranks";

        auto fixture = realTransaction();
        ImmediateLocalResidencyTransport local;
        local.commit_begin_ok = context->rank() != 1;
        auto lane = std::make_shared<MoEOverlayMPIResidencyConsensus>(
            MoEOverlayMPIResidencyConsensus::Config{
                .mpi_context = context,
                .perf_device = "real_mpi_transport_test",
            });
        MoEOverlayDistributedResidencyTransport transport({
            .local_transport = &local,
            .consensus = lane,
            .perf_device = "real_mpi_transport_test",
        });

        auto result = fixture.authority->beginApply(
            fixture.transaction,
            transport);
        ASSERT_EQ(result.status, MoEOverlayResidencyApplyStatus::Started)
            << result.error;
        result = finishAuthorityWave(*fixture.authority);
        EXPECT_EQ(result.status, MoEOverlayResidencyApplyStatus::CommitFailed)
            << result.error;
        EXPECT_NE(result.error.find("world rank 1"), std::string::npos)
            << result.error;
        EXPECT_NE(result.error.find("2103"), std::string::npos)
            << result.error;
        EXPECT_EQ(fixture.authority->snapshot()->epoch, 1u);
        EXPECT_TRUE(lane->idle());
        EXPECT_EQ(transport.stats().commit_consensus_failed, 1u);
        EXPECT_EQ(local.aborts, 1);

        /* Reap the synchronously ready CPU abort before lane destruction. */
        EXPECT_EQ(
            fixture.authority->advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Idle);
        EXPECT_EQ(fixture.authority->pendingAbortCount(), 0u);
    }

    TEST(
        Test__MoEOverlayMPIResidencyConsensus,
        ArbitraryCoordinatorDrivesCompleteDistributedMaintenanceService)
    {
        auto context = worldContext();
        if (!requireTwoRanks(*context))
            GTEST_SKIP() << "Distributed maintenance requires exactly two ranks";

        /* The logical continuation/router root deliberately lives on rank one. */
        constexpr int coordinator_world_rank = 1;
        auto fixture = realTransaction(coordinator_world_rank);
        if (context->rank() == coordinator_world_rank)
        {
            const std::vector<std::uint64_t> counts{1, 2, 100, 90};
            fixture.histogram->mergeLayerCounts(
                0,
                counts.data(),
                static_cast<int>(counts.size()),
                false);
            fixture.histogram->recordTokenBoundary(0, 4);
        }
        else
        {
            ASSERT_FALSE(fixture.histogram->windowFull())
                << "The peer must have no substitute routing evidence";
        }

        ImmediateLocalResidencyTransport local;
        auto consensus =
            std::make_shared<MoEOverlayMPIResidencyConsensus>(
                MoEOverlayMPIResidencyConsensus::Config{
                    .mpi_context = context,
                    .perf_device = "real_mpi_maintenance_test",
                });
        auto local_transport =
            std::make_shared<MoEOverlayDistributedResidencyTransport>(
                MoEOverlayDistributedResidencyTransport::Config{
                    .local_transport = &local,
                    .consensus = consensus,
                    .perf_device = "real_mpi_maintenance_test",
                });
        auto publisher =
            std::make_shared<MoEOverlayMPIHistogramPublisher>(
                MoEOverlayMPIHistogramPublisher::Config{
                    .mpi_context = context,
                    .coordinator_world_rank = coordinator_world_rank,
                    .num_layers = 1,
                    .num_experts = 4,
                    .perf_device = "real_mpi_maintenance_test",
                });
        auto authority = std::shared_ptr<MoEOverlayResidencyAuthority>(
            std::move(fixture.authority));
        MoEOverlayResidencyMaintenanceService service({
            .authority = authority,
            .transport = local_transport,
            .histogram_publisher = publisher,
            .idle_poll_interval = std::chrono::microseconds(100),
            .perf_device = "real_mpi_maintenance_test",
        });

        ASSERT_TRUE(waitForService(
            [&]
            {
                return !service.healthy() ||
                       (authority->snapshot()->epoch == 2u &&
                        service.stats().committed_waves == 1u);
            })) << service.failureMessage();
        ASSERT_TRUE(service.healthy()) << service.failureMessage();
        EXPECT_EQ(authority->snapshot()->epoch, 2u);
        EXPECT_EQ(service.stats().proposals, 1u);
        EXPECT_EQ(service.stats().committed_waves, 1u);
        EXPECT_EQ(local_transport->stats().stage_consensus_ready, 1u);
        EXPECT_EQ(local_transport->stats().commit_consensus_ready, 1u);
        EXPECT_EQ(local_transport->stats().waves_published, 1u);

        if (context->rank() == coordinator_world_rank)
        {
            EXPECT_EQ(service.stats().histogram_windows_published, 1u);
            EXPECT_EQ(publisher->stats().publications_completed, 1u);
        }
        else
        {
            EXPECT_EQ(service.stats().histogram_windows_received, 1u);
            EXPECT_EQ(service.stats().histogram_receives_rearmed, 1u);
            EXPECT_EQ(publisher->stats().windows_received, 1u);
            EXPECT_EQ(publisher->stats().receives_armed, 2u);
        }

        service.stopAndDrain();
        publisher->stopAndDrain();
        EXPECT_TRUE(consensus->idle());
        ASSERT_EQ(local.fingerprints.size(), 1u);
        EXPECT_EQ(local.retirements, 1);
    }
} // namespace llaminar2::test
