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
#include "execution/moe/MoEOverlayMPIResidencyProposalPublisher.h"
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
#include <atomic>
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
                .execution_fingerprint = {
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
         * @brief Private test-only communicator for out-of-band coordination.
         *
         * The maintenance service deliberately blocks in collectives on its
         * production context during terminal drain. A distinct communicator
         * lets the test release a deliberately gated proposal without racing
         * two collective sequences on the same communicator.
         */
        class ScopedControlCommunicator final
        {
        public:
            /** @brief Duplicate the supplied live communicator collectively. */
            explicit ScopedControlCommunicator(MPI_Comm source)
            {
                if (MPI_Comm_dup(source, &communicator_) != MPI_SUCCESS ||
                    communicator_ == MPI_COMM_NULL)
                {
                    throw std::runtime_error(
                        "Could not duplicate the MPI lifecycle-test communicator");
                }
            }

            /** @brief Release the private communicator after every worker joins. */
            ~ScopedControlCommunicator()
            {
                if (communicator_ != MPI_COMM_NULL &&
                    MPI_Comm_free(&communicator_) != MPI_SUCCESS)
                {
                    std::terminate();
                }
            }

            ScopedControlCommunicator(const ScopedControlCommunicator &) =
                delete;
            ScopedControlCommunicator &operator=(
                const ScopedControlCommunicator &) = delete;

            /** @return The private communicator owned by this scope. */
            [[nodiscard]] MPI_Comm get() const noexcept
            {
                return communicator_;
            }

        private:
            MPI_Comm communicator_ = MPI_COMM_NULL;
        };

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

        /** @brief Progress one canonical proposal publication without an MPI wait. */
        std::shared_ptr<const MoEOverlayDistributedResidencyProposal>
        finishProposalPublication(
            MoEOverlayMPIResidencyProposalPublisher &publisher)
        {
            std::shared_ptr<const MoEOverlayDistributedResidencyProposal>
                accepted_proposal;
            for (;;)
            {
                std::shared_ptr<const MoEOverlayDistributedResidencyProposal>
                    proposal;
                std::string error;
                const auto progress = publisher.poll(&proposal, &error);
                if (progress == MoEOverlayResidencyWaveProgress::Ready)
                {
                    if (!publisher.isCoordinator() && proposal)
                    {
                        accepted_proposal = std::move(proposal);
                        const bool accepted =
                            publisher.acceptReceivedProposal(
                                accepted_proposal->plan.histogram_window
                                    ->generation,
                                &error);
                        EXPECT_TRUE(accepted) << error;
                        if (!accepted)
                            return {};
                        continue;
                    }
                    return accepted_proposal;
                }
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

        /** @brief Poll one calibration-readiness all-gather without waiting. */
        MoEOverlayResidencyWaveProgress finishCalibrationReadiness(
            MoEOverlayMPIEconomyEvidenceExchange &exchange)
        {
            for (;;)
            {
                std::string error;
                const auto progress =
                    exchange.pollCalibrationReadiness(&error);
                if (progress == MoEOverlayResidencyWaveProgress::Ready ||
                    progress == MoEOverlayResidencyWaveProgress::Deferred)
                {
                    EXPECT_TRUE(error.empty());
                    return progress;
                }
                EXPECT_EQ(
                    progress,
                    MoEOverlayResidencyWaveProgress::Pending)
                    << error;
                if (progress == MoEOverlayResidencyWaveProgress::Failed)
                    return progress;
                std::this_thread::yield();
            }
        }

        /** @brief Poll one typed service-readiness round without an MPI wait. */
        MoEOverlayServiceEvidenceReadiness finishEconomyServiceReadiness(
            MoEOverlayMPIEconomyEvidenceExchange &exchange)
        {
            auto global_readiness =
                MoEOverlayServiceEvidenceReadiness::AwaitingEvidence;
            for (;;)
            {
                std::string error;
                const auto progress = exchange.pollServiceReadiness(
                    &global_readiness, &error);
                if (progress == MoEOverlayResidencyWaveProgress::Ready)
                    return global_readiness;
                EXPECT_EQ(
                    progress,
                    MoEOverlayResidencyWaveProgress::Pending)
                    << error;
                if (progress == MoEOverlayResidencyWaveProgress::Failed)
                {
                    return MoEOverlayServiceEvidenceReadiness::Stopping;
                }
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
                            /*
                             * Live transaction admission prices the exact
                             * participant critical path, not only its tier
                             * aggregate. This two-participant fixture has one
                             * layer, so totality requires one measured row per
                             * physical owner. Keep the participant rows equal
                             * to their single-member tier rows: the test is
                             * isolating distributed transaction lifecycle, not
                             * manufacturing an independent skew objective.
                             */
                            profile->participant_costs = {
                                {.participant_id = 0,
                                 .layer = 0,
                                 .nanoseconds_per_activation = {10, 20, 30}},
                                {.participant_id = 1,
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
            /** @brief Device-free local bank with independently failable preparation. */
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
                bool beginPrepare(std::string *error) noexcept override
                {
                    if (!owner_->prepare_begin_ok && error)
                        *error = "injected real-MPI local preparation failure";
                    return owner_->prepare_begin_ok;
                }

                /** @brief Successful enqueues are immediately observable. */
                MoEOverlayResidencyWaveProgress pollPrepare(
                    std::string *) noexcept override
                {
                    return MoEOverlayResidencyWaveProgress::Ready;
                }

                /** @brief Publish the already-prepared device-free selector. */
                bool beginPublication(std::string *) noexcept override
                {
                    return true;
                }

                /** @brief Device-free publication is immediately observable. */
                MoEOverlayResidencyWaveProgress pollPublication(
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
            bool prepare_begin_ok = true;
            int begin_calls = 0;
            int aborts = 0;
            int retirements = 0;
            std::vector<MoEOverlayResidencyTransactionFingerprint> fingerprints;
        };

        /**
         * @brief Hold real MPI proposal polling at an admitted generation.
         *
         * `beginPublish()` and the peer's constructor-time receive are real;
         * only maintenance polling is withheld. This deterministically puts
         * shutdown between proposal admission and acknowledgement without
         * replacing any production transport or lifecycle implementation.
         */
        class PollGatedProposalPublisher final
            : public IMoEOverlayResidencyProposalPublisher
        {
        public:
            /** @brief Retain the real MPI publisher whose polls will be gated. */
            explicit PollGatedProposalPublisher(
                std::shared_ptr<MoEOverlayMPIResidencyProposalPublisher>
                    publisher)
                : publisher_(std::move(publisher))
            {
                if (!publisher_)
                {
                    throw std::invalid_argument(
                        "A poll gate requires a real proposal publisher");
                }
            }

            /** @brief Delegate publication so MPI owns the real packet sends. */
            bool beginPublish(
                const MoEOverlayDistributedResidencyProposal &proposal,
                std::string *error) override
            {
                return publisher_->beginPublish(proposal, error);
            }

            /** @brief Suppress observation until the test releases progress. */
            MoEOverlayResidencyWaveProgress poll(
                std::shared_ptr<const MoEOverlayDistributedResidencyProposal>
                    *received_proposal,
                std::string *error) override
            {
                if (!polling_enabled_.load(std::memory_order_acquire))
                {
                    if (received_proposal)
                        received_proposal->reset();
                    if (error)
                        error->clear();
                    return MoEOverlayResidencyWaveProgress::Pending;
                }
                return publisher_->poll(received_proposal, error);
            }

            /** @brief Delegate post-adoption semantic acceptance. */
            bool acceptReceivedProposal(
                std::uint64_t histogram_generation,
                std::string *error) override
            {
                return publisher_->acceptReceivedProposal(
                    histogram_generation, error);
            }

            /** @brief Delegate fatal semantic rejection to the real MPI lane. */
            void rejectReceivedProposal(
                std::uint64_t histogram_generation,
                std::string diagnostic) override
            {
                publisher_->rejectReceivedProposal(
                    histogram_generation, std::move(diagnostic));
            }

            /** @brief Delegate the peer's next persistent receive arm. */
            bool armReceive(std::string *error) override
            {
                return publisher_->armReceive(error);
            }

            /** @brief Drain the real MPI requests after maintenance quiesces. */
            void stopAndDrain() override
            {
                publisher_->stopAndDrain();
            }

            /** @return Whether this rank is the arbitrary coordinator. */
            [[nodiscard]] bool isCoordinator() const noexcept override
            {
                return publisher_->isCoordinator();
            }

            /** @brief Release the poll edge after terminal drain is active. */
            void enablePolling() noexcept
            {
                polling_enabled_.store(true, std::memory_order_release);
            }

        private:
            std::shared_ptr<MoEOverlayMPIResidencyProposalPublisher>
                publisher_;
            std::atomic<bool> polling_enabled_{false};
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
                        MoEOverlayResidencyApplyStatus::Preparing &&
                    result.status !=
                        MoEOverlayResidencyApplyStatus::Publishing)
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
                    MoEOverlayResidencyApplyStatus::RetirementFailed)
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
                .execution_fingerprint = {
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

        /** @brief One valid pre-staging identity, optionally rank-divergent. */
        MoEOverlayCalibrationReadiness economyReadiness(
            int world_rank,
            bool divergent)
        {
            return {
                .kind =
                    MoEOverlayCalibrationReadinessKind::BaselineDeviceComplete,
                .calibration_sequence = 40,
                .coordinate = {
                    .source_participant = 0,
                    .destination_participant = 1,
                    .layer = 0,
                },
                .source = ExpertHistogramSource::PrefillChunk,
                .workload = {
                    .source = ExpertHistogramSource::PrefillChunk,
                    .real_rows = divergent && world_rank == 1 ? 31 : 32,
                    .execution_rows = 32,
                    .transaction_count = 1,
                    .speculative_depth = 0,
                    .schedule_fingerprint = 0x1234u,
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
        ASSERT_TRUE(exchange.beginCalibrationReadiness(
            economyReadiness(context->rank(), false), &error))
            << error;
        EXPECT_EQ(
            finishCalibrationReadiness(exchange),
            MoEOverlayResidencyWaveProgress::Ready);
        EXPECT_TRUE(exchange.idle());

        ASSERT_TRUE(exchange.beginCalibrationReadiness(
            economyReadiness(context->rank(), true), &error))
            << error;
        EXPECT_EQ(
            finishCalibrationReadiness(exchange),
            MoEOverlayResidencyWaveProgress::Deferred);
        EXPECT_TRUE(exchange.idle());

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
            context->rank() == 0
                ? MoEOverlayServiceEvidenceReadiness::Ready
                : MoEOverlayServiceEvidenceReadiness::AwaitingEvidence,
            &error))
            << error;
        EXPECT_EQ(
            finishEconomyServiceReadiness(exchange),
            MoEOverlayServiceEvidenceReadiness::AwaitingEvidence)
            << "One ready rank must not enter the service all-gather alone";
        EXPECT_TRUE(exchange.idle());

        ASSERT_TRUE(exchange.beginServiceReadiness(
            context->rank() == 0
                ? MoEOverlayServiceEvidenceReadiness::Ready
                : MoEOverlayServiceEvidenceReadiness::Stopping,
            &error))
            << error;
        EXPECT_EQ(
            finishEconomyServiceReadiness(exchange),
            MoEOverlayServiceEvidenceReadiness::Stopping)
            << "One stopping rank must close the shared readiness lifecycle";
        EXPECT_TRUE(exchange.idle());

        ASSERT_TRUE(exchange.beginServiceReadiness(
            MoEOverlayServiceEvidenceReadiness::Ready, &error))
            << error;
        EXPECT_EQ(
            finishEconomyServiceReadiness(exchange),
            MoEOverlayServiceEvidenceReadiness::Ready);
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
        EXPECT_EQ(stats.calibration_readiness_exchanges_started, 2u);
        EXPECT_EQ(stats.calibration_readiness_exchanges_completed, 1u);
        EXPECT_EQ(stats.calibration_readiness_retries, 1u);
        EXPECT_EQ(stats.service_readiness_exchanges_started, 3u);
        EXPECT_EQ(stats.service_readiness_exchanges_completed, 3u);
        EXPECT_EQ(stats.service_readiness_incomplete, 1u);
        EXPECT_EQ(stats.service_readiness_stops, 1u);
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
        PreparedContextRestorationPurposeMovesExactRemoteBytes)
    {
        auto context = worldContext();
        if (!requireTwoRanks(*context))
            GTEST_SKIP() << "Remote restoration data plane requires two ranks";

        constexpr std::size_t staging_bytes = 97;
        MoEOverlayMPIRemoteProjectionTransport transport({
            .mpi_context = context,
            .lane_budget = {
                .maximum_participants_per_cycle = 1,
                .maximum_concurrent_cycles = 1,
                .projections_per_expert = 1,
            },
            .staging_capacity_bytes = staging_bytes,
            .perf_device = "mpi_remote_projection_restoration_test",
        });

        auto source_projection = remoteCpuProjection(117);
        const auto projection_identity = remoteProjectionIdentity(
            /*source_rank=*/0,
            /*destination_rank=*/1,
            /*migration_index=*/0,
            ExpertTierWeightProjection::Gate);
        const auto manifest = makeMoEOverlayRemoteCpuProjectionManifest(
            projection_identity,
            source_projection,
            static_cast<std::uint32_t>(staging_bytes));
        std::vector<std::uint8_t> destination_bytes(
            source_projection.native_interleaved.size(), 0xa5u);
        const auto source_lifetime = std::make_shared<int>(19);
        const auto destination_lifetime = std::make_shared<int>(23);

        auto binding = localHostBinding(
            context->rank(),
            /*lane_index=*/0,
            projection_identity,
            manifest,
            source_projection,
            destination_bytes,
            source_lifetime,
            destination_lifetime);
        binding.purpose = MoEOverlayResidencyTransactionPurpose::
            PreparedContextRestoration;
        auto wave = transport.reserveWave({std::move(binding)});
        ASSERT_EQ(wave.status, MoEOverlayResidencyStageStartStatus::Started)
            << wave.error;
        ASSERT_EQ(wave.operations.size(), 1u);
        ASSERT_NE(wave.operations.front(), nullptr);

        bool ready = false;
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!ready && std::chrono::steady_clock::now() < deadline)
        {
            std::string error;
            const auto progress = wave.operations.front()->poll(&error);
            ASSERT_NE(progress, MoEOverlayResidencyWaveProgress::Failed)
                << error;
            ready = progress == MoEOverlayResidencyWaveProgress::Ready;
            std::this_thread::yield();
        }
        ASSERT_TRUE(ready);

        if (context->rank() == 1)
        {
            ASSERT_EQ(
                destination_bytes.size(),
                source_projection.native_interleaved.size());
            EXPECT_TRUE(std::equal(
                destination_bytes.begin(),
                destination_bytes.end(),
                source_projection.native_interleaved.begin(),
                source_projection.native_interleaved.end()));
        }
        const auto stats = transport.stats();
        EXPECT_EQ(stats.wave_reservations_started, 1u);
        EXPECT_EQ(stats.wave_reservations_failed, 0u);
        EXPECT_EQ(stats.operations_completed, 1u);
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
        /* Every rank rotates the same pre-certification generation even though
         * only the coordinator below retains policy/economy evidence. */
        ASSERT_EQ(
            authority->progressEconomyEvidenceRebase(),
            MoEOverlayHistogramRebaseProgress::Complete);
        if (context->rank() == 1)
        {
            auto service =
                std::make_shared<MoERoutedTierServiceProfile>();
            service->identity = "root-service-profile";
            service->costs = {
                {.tier_index = 0,
                 .layer = 0,
                 .nanoseconds_per_activation = {1, 1, 1}},
                {.tier_index = 1,
                 .layer = 0,
                 .nanoseconds_per_activation = {100, 100, 100}},
            };
            service->participant_costs = {
                {.participant_id = 0,
                 .layer = 0,
                 .nanoseconds_per_activation = {1, 1, 1}},
                {.participant_id = 1,
                 .layer = 0,
                 .nanoseconds_per_activation = {100, 100, 100}},
            };
            auto migration =
                std::make_shared<MoEOverlayMigrationCostProfile>();
            migration->identity = "root-migration-profile";
            migration->costs = {
                {.source_participant = 0,
                 .destination_participant = 1,
                 .layer = 0,
                 .transfer_and_repack_ns = 1},
                {.source_participant = 1,
                 .destination_participant = 0,
                 .layer = 0,
                 .transfer_and_repack_ns = 1},
            };
            authority->installEconomyCertification(
                std::move(service),
                std::move(migration),
                {
                    .historical_window_weight = 0,
                    .current_window_weight = 1,
                    .payoff_horizon_tokens = 1'000'000,
                    .minimum_net_benefit_ns = 0,
                    .minimum_residency_generations = 0,
                });
            ASSERT_EQ(
                authority->activateOptimizationDemandAtRequestBoundary(),
                MoEOverlayDemandActivationResult::Activated);
        }

        /* This is the first policy-bearing production window. Seed it only
         * after the root has discarded its pre-certification generation so
         * root and follower exercise the same executable proposal. */
        const std::uint64_t counts[]{1, 100};
        histogram->mergeLayerCounts(0, counts, 2, false);
        histogram->recordTokenBoundary(0, 2);
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

        /*
         * Rank one is the sole policy authority.  Publish its dense executable
         * plan over the same authenticated control lane used by production;
         * the follower must reconstruct the transaction instead of rerunning
         * policy without the coordinator's economy evidence.
         */
        constexpr int coordinator_world_rank = 1;
        MoEOverlayMPIResidencyProposalPublisher proposal_publisher({
            .mpi_context = context,
            .coordinator_world_rank = coordinator_world_rank,
            .num_layers = 1,
            .num_experts = 2,
            .perf_device = "mpi-distributed-physical-cpu",
        });
        MoEOverlayResidencyTransaction transaction;
        if (proposal_publisher.isCoordinator())
        {
            transaction = authority->proposeFromHistogram();
            ASSERT_TRUE(transaction.valid());
            const auto proposal =
                makeMoEOverlayDistributedResidencyProposal(
                    authority->exportAuthoritativeResidencyPlan(transaction),
                    transaction);
            std::string proposal_error;
            ASSERT_TRUE(proposal_publisher.beginPublish(
                proposal, &proposal_error)) << proposal_error;
        }

        bool proposal_complete = false;
        const auto proposal_deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!proposal_complete &&
               std::chrono::steady_clock::now() < proposal_deadline)
        {
            std::shared_ptr<const MoEOverlayDistributedResidencyProposal>
                received;
            std::string proposal_error;
            const auto proposal_progress = proposal_publisher.poll(
                &received, &proposal_error);
            ASSERT_NE(
                proposal_progress,
                MoEOverlayResidencyWaveProgress::Failed) << proposal_error;
            if (!proposal_publisher.isCoordinator() && received)
            {
                transaction = authority->adoptAuthoritativeResidencyPlan(
                    received->plan);
                ASSERT_TRUE(transaction.valid());
                ASSERT_EQ(
                    fingerprintMoEOverlayResidencyExecutionPlan(transaction),
                    received->execution_fingerprint);
                ASSERT_TRUE(proposal_publisher.acceptReceivedProposal(
                    received->plan.histogram_window->generation,
                    &proposal_error)) << proposal_error;
                continue;
            }
            proposal_complete =
                proposal_progress == MoEOverlayResidencyWaveProgress::Ready;
            std::this_thread::yield();
        }
        ASSERT_TRUE(proposal_complete)
            << "Authoritative physical-fabric proposal did not complete within five seconds";
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

        /*
         * Model the production coordinator/follower split exactly. Rank one
         * owns policy measurements while rank zero adopts only the executable
         * plan. Before executable and policy fingerprints became distinct
         * types, each rank embedded a different digest in its remote manifest
         * and the destination rejected the source before any payload moved.
         */
        if (proposal_publisher.isCoordinator())
        {
            ASSERT_TRUE(transaction.economy.enabled);
            auto follower_view = transaction;
            follower_view.economy = {};
            ASSERT_TRUE(follower_view.valid());
            EXPECT_EQ(
                fingerprintMoEOverlayResidencyExecutionPlan(transaction),
                fingerprintMoEOverlayResidencyExecutionPlan(follower_view));
            EXPECT_NE(
                fingerprintMoEOverlayResidencyTransaction(transaction),
                fingerprintMoEOverlayResidencyTransaction(follower_view));
        }
        else
        {
            EXPECT_FALSE(transaction.economy.enabled);
        }

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
            if (progress.status == MoEOverlayResidencyApplyStatus::Published)
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
        EXPECT_EQ(distributed_stats.preparation_consensus_ready, 1u);
        EXPECT_EQ(distributed_stats.publication_consensus_ready, 1u);
        EXPECT_EQ(distributed_stats.waves_published, 1u);
        EXPECT_EQ(distributed_stats.inference_thread_waits, 0u);
        EXPECT_EQ(distributed_stats.blocking_synchronizations, 0u);

        const auto authority_stats = authority->stats();
        EXPECT_EQ(authority_stats.committed_waves, 1u);
        EXPECT_EQ(authority_stats.committed_migrations, 2u);
        EXPECT_EQ(authority_stats.cross_rank_migrations, 2u);
        EXPECT_EQ(authority->pendingRetirementCount(), 0u);
    }

    /**
     * @brief Exercise distributed certification through completion or stop.
     * @param stop_while_peer_waits When true, stop after one shared incomplete
     *        round while rank zero still lacks local evidence.
     */
    void assertDistributedEconomyCertification(
        bool stop_while_peer_waits)
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
         * Rank zero deliberately starts without service evidence. Both ranks
         * must complete one typed AwaitingEvidence round before either can
         * enter a later Ready round or the larger service all-gather.
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
        auto calibration = std::make_shared<
            MoEOverlayEconomyCalibrationController>(
            MoEOverlayEconomyCalibrationController::Config{
                .planner = planner,
                .ledger = ledger,
                .journal = journal,
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
            .service_readiness_retry_interval =
                stop_while_peer_waits
                    ? std::chrono::milliseconds(1'000)
                    : std::chrono::milliseconds(0),
            .perf_device = "mpi-distributed-economy-cpu",
        });

        const auto terminal_state = [&]
        {
            return certification.state() ==
                   (stop_while_peer_waits
                        ? MoEOverlayEconomyCertificationState::Stopped
                        : MoEOverlayEconomyCertificationState::Complete);
        };
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(10);
        if (stop_while_peer_waits)
        {
            while ((certification.state() !=
                        MoEOverlayEconomyCertificationState::
                            AwaitingServiceEvidence ||
                    evidence_exchange->stats()
                            .service_readiness_incomplete == 0u) &&
                   certification.healthy() &&
                   std::chrono::steady_clock::now() < deadline)
            {
                certification.poll();
                std::this_thread::yield();
            }
            ASSERT_TRUE(certification.healthy())
                << certification.failureMessage();
            ASSERT_EQ(
                certification.state(),
                MoEOverlayEconomyCertificationState::
                    AwaitingServiceEvidence);

            /*
             * Synchronize only the test harness on MPI_COMM_WORLD.  The
             * production economy exchange owns its duplicated communicator.
             * Both controllers observe stop before either opens the next
             * retry round, then the typed Stopping reduction closes the same
             * lifecycle on every rank.
             */
            int local_edge_ready = 1;
            int every_edge_ready = 0;
            ASSERT_EQ(
                MPI_Allreduce(
                    &local_edge_ready,
                    &every_edge_ready,
                    1,
                    MPI_INT,
                    MPI_LAND,
                    MPI_COMM_WORLD),
                MPI_SUCCESS);
            ASSERT_EQ(every_edge_ready, 1);
            certification.requestStop();
            while (!terminal_state() && certification.healthy() &&
                   std::chrono::steady_clock::now() < deadline)
            {
                certification.poll();
                std::this_thread::yield();
            }
        }
        else
        {
            while (!terminal_state() && certification.healthy() &&
                   std::chrono::steady_clock::now() < deadline)
            {
                certification.poll();
                if (!local_service_recorded &&
                    evidence_exchange->stats()
                            .service_readiness_incomplete != 0u)
                {
                    recordLocalServiceEvidence();
                    local_service_recorded = true;
                }
                std::this_thread::yield();
            }
        }
        ASSERT_TRUE(certification.healthy())
            << certification.failureMessage();
        ASSERT_EQ(
            certification.state(),
            stop_while_peer_waits
                ? MoEOverlayEconomyCertificationState::Stopped
                : MoEOverlayEconomyCertificationState::Complete);
        EXPECT_EQ(
            authority->hasEconomyCertification(),
            !stop_while_peer_waits);
        EXPECT_EQ(authority->stats().checks, 0u)
            << "Certification must precede the first histogram proposal";
        EXPECT_EQ(authority->snapshot()->epoch, initial_snapshot->epoch)
            << "Calibration waves are never publishable";
        EXPECT_TRUE(evidence_exchange->idle());

        const auto calibration_stats = calibration->stats();
        EXPECT_EQ(calibration_stats.accepted_pairs, 3u);
        EXPECT_EQ(calibration_stats.exact_overlap_samples, 0u);
        EXPECT_EQ(calibration_stats.partial_overlap_rejections, 0u);
        EXPECT_EQ(calibration_stats.waves_started, 3u);
        EXPECT_EQ(calibration_stats.waves_aborted, 3u);
        EXPECT_EQ(fabric->stats().waves_prepared, 3u);
        EXPECT_EQ(local_transport->stats().commits_started, 0u);
        EXPECT_EQ(local_transport->stats().waves_aborted, 3u);
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
        EXPECT_EQ(
            evidence_exchange->stats()
                .service_readiness_exchanges_started,
            2u);
        EXPECT_EQ(
            evidence_exchange->stats().service_readiness_incomplete,
            1u);
        EXPECT_EQ(
            evidence_exchange->stats().service_readiness_stops,
            stop_while_peer_waits ? 1u : 0u);
        EXPECT_EQ(
            certification.stats().certifications_installed,
            stop_while_peer_waits ? 0u : 1u);
    }

    TEST(
        Test__MoEOverlayMPIResidencyConsensus,
        DistributedEconomyCertificationUsesRealAbortOnlyMovementBeforeProposals)
    {
        assertDistributedEconomyCertification(
            /*stop_while_peer_waits=*/false);
    }

    TEST(
        Test__MoEOverlayMPIResidencyConsensus,
        DistributedEconomyCertificationPropagatesTypedStopAfterIncompleteRound)
    {
        assertDistributedEconomyCertification(
            /*stop_while_peer_waits=*/true);
    }

    TEST(
        Test__MoEOverlayMPIResidencyConsensus,
        ArbitraryCoordinatorPublishesAuthenticatedCanonicalProposal)
    {
        auto context = worldContext();
        if (!requireTwoRanks(*context))
            GTEST_SKIP() << "Proposal publication requires exactly two ranks";

        /* Rank one is deliberate: hardware/routing ownership is not rank zero. */
        constexpr int coordinator_world_rank = 1;
        MoEOverlayMPIResidencyProposalPublisher publisher({
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
        MoEOverlayDistributedResidencyProposal source_proposal{
            .plan = {
                .expected_epoch = 1,
                .num_layers = 2,
                .num_experts = 3,
                .histogram_window =
                    std::make_shared<DecodeExpertHistogramWindow>(source),
                .entries = std::vector<
                    MoEOverlayAuthoritativeResidencyEntry>(
                    6,
                    MoEOverlayAuthoritativeResidencyEntry{
                        .candidate_tier_idx = 0,
                        .candidate_owner_participant = 0,
                    }),
            },
            .execution_fingerprint = {
                .low = 0x1020304050607080ull,
                .high = 0x8070605040302010ull,
            },
            .policy_fingerprint = {
                .low = 0x1122334455667788ull,
                .high = 0x8877665544332211ull,
            },
        };
        ASSERT_TRUE(source_proposal.valid());
        if (publisher.isCoordinator())
        {
            std::string error;
            ASSERT_TRUE(publisher.beginPublish(source_proposal, &error))
                << error;
        }

        const auto received = finishProposalPublication(publisher);
        if (publisher.isCoordinator())
        {
            EXPECT_EQ(received, nullptr);
            EXPECT_EQ(
                publisher.state(),
                MoEOverlayMPIResidencyProposalPublisherState::Idle);
            const auto stats = publisher.stats();
            EXPECT_EQ(stats.publications_started, 1u);
            EXPECT_EQ(stats.publications_completed, 1u);
            EXPECT_EQ(
                stats.bytes_sent,
                moeOverlayDistributedResidencyProposalWireBytes(2, 3));
        }
        else
        {
            ASSERT_NE(received, nullptr);
            EXPECT_EQ(
                received->plan.histogram_window->generation,
                source.generation);
            EXPECT_EQ(
                received->plan.histogram_window->token_count,
                source.token_count);
            EXPECT_EQ(
                received->plan.histogram_window->expert_counts,
                source.expert_counts);
            EXPECT_EQ(
                received->execution_fingerprint,
                source_proposal.execution_fingerprint);
            const auto stats = publisher.stats();
            EXPECT_EQ(stats.receives_armed, 1u);
            EXPECT_EQ(stats.windows_received, 1u);
            EXPECT_EQ(
                stats.bytes_received,
                moeOverlayDistributedResidencyProposalWireBytes(2, 3));
        }
        EXPECT_EQ(publisher.stats().blocking_inference_waits, 0u);
    }

    TEST(
        Test__MoEOverlayMPIResidencyConsensus,
        RepeatedProposalPublisherLifecyclesDrainBeforeCommunicatorReuse)
    {
        auto context = worldContext();
        if (!requireTwoRanks(*context))
            GTEST_SKIP() << "Proposal lifecycle reuse requires exactly two ranks";

        /*
         * Production parity retains one MPI process while repeatedly replacing
         * model runners.  Every runner owns a fresh proposal communicator, and
         * the follower normally tears it down with the next passive receive
         * already armed.  Exercise enough alternating coordinator lifetimes to
         * force MPI context-id reuse and prove that no cancelled receive,
         * acknowledgement, or packet can escape into a later runner.
         */
        constexpr std::uint64_t kLifecycleCount = 64u;
        for (std::uint64_t lifecycle = 0; lifecycle < kLifecycleCount;
             ++lifecycle)
        {
            const int coordinator_world_rank =
                static_cast<int>(lifecycle % 2u);
            MoEOverlayMPIResidencyProposalPublisher publisher({
                .mpi_context = context,
                .coordinator_world_rank = coordinator_world_rank,
                .num_layers = 2,
                .num_experts = 3,
                .perf_device = "mpi_proposal_reuse_test",
            });

            const std::uint64_t generation = lifecycle + 1u;
            MoEOverlayDistributedResidencyProposal proposal{
                .plan = {
                    .expected_epoch = generation,
                    .num_layers = 2,
                    .num_experts = 3,
                    .histogram_window =
                        std::make_shared<DecodeExpertHistogramWindow>(
                            DecodeExpertHistogramWindow{
                                .generation = generation,
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
                            }),
                    .entries = std::vector<
                        MoEOverlayAuthoritativeResidencyEntry>(
                        6,
                        MoEOverlayAuthoritativeResidencyEntry{
                            .candidate_tier_idx = 0,
                            .candidate_owner_participant = 0,
                        }),
                },
                .execution_fingerprint = {
                    .low = 0x1020304050607080ull ^ generation,
                    .high = 0x8070605040302010ull,
                },
                .policy_fingerprint = {
                    .low = 0x1122334455667788ull ^ generation,
                    .high = 0x8877665544332211ull,
                },
            };
            ASSERT_TRUE(proposal.valid());
            if (publisher.isCoordinator())
            {
                std::string error;
                ASSERT_TRUE(publisher.beginPublish(proposal, &error))
                    << "lifecycle=" << lifecycle << " " << error;
            }

            const auto received = finishProposalPublication(publisher);
            if (publisher.isCoordinator())
            {
                EXPECT_EQ(received, nullptr) << "lifecycle=" << lifecycle;
                EXPECT_EQ(
                    publisher.stats().publications_completed,
                    1u) << "lifecycle=" << lifecycle;
            }
            else
            {
                ASSERT_NE(received, nullptr) << "lifecycle=" << lifecycle;
                EXPECT_EQ(
                    received->plan.histogram_window->generation,
                    generation) << "lifecycle=" << lifecycle;

                /* Match the maintenance service's steady state: it rearms the
                 * immutable mailbox before the current runner is retired. */
                std::string error;
                ASSERT_TRUE(publisher.armReceive(&error))
                    << "lifecycle=" << lifecycle << " " << error;
            }

            publisher.stopAndDrain();
            EXPECT_EQ(
                publisher.state(),
                MoEOverlayMPIResidencyProposalPublisherState::Stopped)
                << "lifecycle=" << lifecycle;

            /* Keep collective communicator creation/free order symmetric even
             * if one rank completes local acknowledgement progress first. */
            context->barrier();
        }
    }

    TEST(
        Test__MoEOverlayMPIResidencyConsensus,
        RepeatedDistributedMaintenanceLifecyclesDrainBeforeContextReuse)
    {
        auto context = worldContext();
        if (!requireTwoRanks(*context))
        {
            GTEST_SKIP()
                << "Distributed maintenance reuse requires exactly two ranks";
        }

        /*
         * A production parity process retains MPI_COMM_WORLD while replacing
         * complete model runners.  This boundary is deliberately broader than
         * the publisher-only regression above: every lifetime creates a
         * maintenance worker, policy authority, proposal lane, distributed
         * transaction lane, and their private communicators.  Alternating the
         * arbitrary coordinator and exceeding the campaign's observed
         * pre-failure lifetime count catches leaked worker progress or MPI
         * requests that a fresh publisher in isolation cannot expose.
         */
        constexpr std::uint64_t kLifecycleCount = 64u;
        for (std::uint64_t lifecycle = 0; lifecycle < kLifecycleCount;
             ++lifecycle)
        {
            const int coordinator_world_rank =
                static_cast<int>(lifecycle % 2u);
            {
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
                        << "lifecycle=" << lifecycle;
                }

                ImmediateLocalResidencyTransport local;
                auto consensus =
                    std::make_shared<MoEOverlayMPIResidencyConsensus>(
                        MoEOverlayMPIResidencyConsensus::Config{
                            .mpi_context = context,
                            .perf_device =
                                "repeated_mpi_maintenance_test",
                        });
                auto transport =
                    std::make_shared<
                        MoEOverlayDistributedResidencyTransport>(
                        MoEOverlayDistributedResidencyTransport::Config{
                            .local_transport = &local,
                            .consensus = consensus,
                            .perf_device =
                                "repeated_mpi_maintenance_test",
                        });
                auto publisher =
                    std::make_shared<
                        MoEOverlayMPIResidencyProposalPublisher>(
                        MoEOverlayMPIResidencyProposalPublisher::Config{
                            .mpi_context = context,
                            .coordinator_world_rank =
                                coordinator_world_rank,
                            .num_layers = 1,
                            .num_experts = 4,
                            .perf_device =
                                "repeated_mpi_maintenance_test",
                        });
                auto authority =
                    std::shared_ptr<MoEOverlayResidencyAuthority>(
                        std::move(fixture.authority));
                MoEOverlayResidencyMaintenanceService service({
                    .authority = authority,
                    .transport = transport,
                    .proposal_publisher = publisher,
                    .distributed_context = context,
                    .idle_poll_interval = std::chrono::microseconds(100),
                    .perf_device = "repeated_mpi_maintenance_test",
                });

                context->barrier();
                service.start();
                ASSERT_TRUE(waitForService(
                    [&]
                    {
                        return !service.healthy() ||
                               (authority->snapshot()->epoch == 2u &&
                                service.stats().committed_waves == 1u);
                    })) << "lifecycle=" << lifecycle << " "
                        << service.failureMessage();
                ASSERT_TRUE(service.healthy())
                    << "lifecycle=" << lifecycle << " "
                    << service.failureMessage();

                service.stopAndDrain(
                    MoEOverlayMaintenanceDrainScope::DistributedTopology);
                EXPECT_EQ(
                    service.state(),
                    MoEOverlayMaintenanceState::Stopped)
                    << "lifecycle=" << lifecycle;
                EXPECT_EQ(service.stats().committed_waves, 1u)
                    << "lifecycle=" << lifecycle;
                EXPECT_EQ(local.retirements, 1)
                    << "lifecycle=" << lifecycle;
                EXPECT_TRUE(consensus->idle())
                    << "lifecycle=" << lifecycle;

                /* The stopped worker no longer polls this lane.  Explicitly
                 * cancel the follower's prearmed next-generation receive (or
                 * reap the coordinator's idle lane) before any shared owner is
                 * released, matching runner teardown order. */
                publisher->stopAndDrain();
                EXPECT_EQ(
                    publisher->state(),
                    MoEOverlayMPIResidencyProposalPublisherState::Stopped)
                    << "lifecycle=" << lifecycle;
            }

            /* Complete destruction before collective context allocation for
             * the next runner, and keep that order identical on both ranks. */
            context->barrier();
        }
    }

    TEST(
        Test__MoEOverlayMPIResidencyConsensus,
        UnanimousStagePrepareAndPublicationVotesCompleteOnPrivateLane)
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
                AwaitingLocalPrepare);

        const auto prepare_vote = protocol.makeLocalVote(
            MoEOverlayDistributedResidencyVoteDecision::Ready);
        ASSERT_TRUE(lane.begin(prepare_vote, &error)) << error;
        const auto prepare_votes = finishExchange(lane);
        ASSERT_EQ(prepare_votes.size(), 2u);
        ASSERT_TRUE(protocol.acceptConsensus(prepare_votes, &error)) << error;
        EXPECT_EQ(
            protocol.state(),
            MoEOverlayDistributedResidencyProtocolState::
                AwaitingLocalPublication);

        const auto publication_vote = protocol.makeLocalVote(
            MoEOverlayDistributedResidencyVoteDecision::Ready);
        ASSERT_TRUE(lane.begin(publication_vote, &error)) << error;
        const auto publication_votes = finishExchange(lane);
        ASSERT_EQ(publication_votes.size(), 2u);
        ASSERT_TRUE(protocol.acceptConsensus(publication_votes, &error))
            << error;
        EXPECT_EQ(
            protocol.state(),
            MoEOverlayDistributedResidencyProtocolState::
                ReadyForAuthorityPublication);
        protocol.markAuthorityPublished();

        const auto stats = lane.stats();
        EXPECT_EQ(stats.exchanges_started, 4u);
        EXPECT_EQ(stats.exchanges_completed, 4u);
        EXPECT_GT(stats.progress_polls, 0u);
        EXPECT_EQ(stats.mpi_failures, 0u);
        EXPECT_TRUE(lane.idle());
    }

    TEST(
        Test__MoEOverlayMPIResidencyConsensus,
        RemoteStageFailureAbortsEveryRankBeforePreparation)
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
        EXPECT_THROW(protocol.markAuthorityPublished(), std::logic_error);
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
        ASSERT_EQ(result.status, MoEOverlayResidencyApplyStatus::Published)
            << result.error;
        EXPECT_EQ(fixture.authority->snapshot()->epoch, 2u);
        ASSERT_EQ(local.fingerprints.size(), 2u);
        EXPECT_EQ(local.fingerprints[0], local.fingerprints[1]);

        /*
         * Publication may immediately start the non-blocking retirement vote,
         * so the shared lane is not required to be idle at the return edge of
         * Published. Poll the maintenance authority until both ranks have
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
        EXPECT_EQ(stats.preparation_consensus_ready, 1u);
        EXPECT_EQ(stats.publication_consensus_ready, 1u);
        EXPECT_EQ(stats.waves_published, 1u);
        EXPECT_EQ(stats.inference_thread_waits, 0u);
        EXPECT_EQ(stats.blocking_synchronizations, 0u);
    }

    TEST(
        Test__MoEOverlayMPIResidencyConsensus,
        DistributedPrepareStartFailureReachesEveryRankWithoutHang)
    {
        auto context = worldContext();
        if (!requireTwoRanks(*context))
            GTEST_SKIP() << "Distributed transport requires exactly two ranks";

        auto fixture = realTransaction();
        ImmediateLocalResidencyTransport local;
        local.prepare_begin_ok = context->rank() != 1;
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
        EXPECT_EQ(
            result.status,
            MoEOverlayResidencyApplyStatus::PreparationFailed)
            << result.error;
        EXPECT_NE(result.error.find("world rank 1"), std::string::npos)
            << result.error;
        EXPECT_NE(result.error.find("2103"), std::string::npos)
            << result.error;
        EXPECT_EQ(fixture.authority->snapshot()->epoch, 1u);
        EXPECT_TRUE(lane->idle());
        EXPECT_EQ(transport.stats().preparation_consensus_failed, 1u);
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
        ScopedControlCommunicator control(context->communicator());

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
            std::make_shared<MoEOverlayMPIResidencyProposalPublisher>(
                MoEOverlayMPIResidencyProposalPublisher::Config{
                    .mpi_context = context,
                    .coordinator_world_rank = coordinator_world_rank,
                    .num_layers = 1,
                    .num_experts = 4,
                    .perf_device = "real_mpi_maintenance_test",
                });
        auto poll_gate =
            std::make_shared<PollGatedProposalPublisher>(publisher);
        auto authority = std::shared_ptr<MoEOverlayResidencyAuthority>(
            std::move(fixture.authority));
        MoEOverlayResidencyMaintenanceService service({
            .authority = authority,
            .transport = local_transport,
            .proposal_publisher = poll_gate,
            .distributed_context = context,
            .idle_poll_interval = std::chrono::microseconds(100),
            .perf_device = "real_mpi_maintenance_test",
        });
        ASSERT_EQ(service.state(), MoEOverlayMaintenanceState::Prepared);
        context->barrier();
        service.start();

        /* Hold the real packet between MPI admission and peer acknowledgement. */
        int publication_admitted = 0;
        if (context->rank() == coordinator_world_rank)
        {
            publication_admitted = waitForService(
                [&]
                {
                    return !service.healthy() ||
                           publisher->stats().publications_started == 1u;
                })
                ? 1
                : 0;
        }
        ASSERT_EQ(
            MPI_Bcast(
                &publication_admitted,
                1,
                MPI_INT,
                coordinator_world_rank,
                control.get()),
            MPI_SUCCESS);
        ASSERT_EQ(publication_admitted, 1) << service.failureMessage();
        ASSERT_TRUE(service.healthy()) << service.failureMessage();
        EXPECT_EQ(
            publisher->state(),
            context->rank() == coordinator_world_rank
                ? MoEOverlayMPIResidencyProposalPublisherState::Publishing
                : MoEOverlayMPIResidencyProposalPublisherState::Receiving);

        /* A live distributed worker cannot be torn down by one rank alone. */
        EXPECT_THROW(
            service.stopAndDrain(
                MoEOverlayMaintenanceDrainScope::ProcessLocalComposition),
            std::logic_error);

        std::atomic<bool> drain_returned{false};
        std::jthread drain_thread(
            [&]
            {
                service.stopAndDrain(
                    MoEOverlayMaintenanceDrainScope::DistributedTopology);
                drain_returned.store(true, std::memory_order_release);
            });

        /* The coordinator must enter drain while the follower remains live. */
        int coordinator_is_draining = 0;
        if (context->rank() == coordinator_world_rank)
        {
            coordinator_is_draining = waitForService(
                [&]
                {
                    return service.state() ==
                           MoEOverlayMaintenanceState::Draining;
                })
                ? 1
                : 0;
        }
        EXPECT_EQ(
            MPI_Bcast(
                &coordinator_is_draining,
                1,
                MPI_INT,
                coordinator_world_rank,
                control.get()),
            MPI_SUCCESS);
        EXPECT_EQ(coordinator_is_draining, 1);
        EXPECT_FALSE(drain_returned.load(std::memory_order_acquire));

        poll_gate->enablePolling();
        service.notifyMaintenanceProgress();
        drain_thread.join();

        ASSERT_TRUE(service.healthy()) << service.failureMessage();
        EXPECT_TRUE(drain_returned.load(std::memory_order_acquire));
        EXPECT_EQ(service.state(), MoEOverlayMaintenanceState::Stopped);
        EXPECT_EQ(authority->snapshot()->epoch, 2u);
        EXPECT_EQ(service.stats().proposals, 1u);
        EXPECT_EQ(service.stats().committed_waves, 1u);
        EXPECT_EQ(local_transport->stats().stage_consensus_ready, 1u);
        EXPECT_EQ(local_transport->stats().preparation_consensus_ready, 1u);
        EXPECT_EQ(local_transport->stats().publication_consensus_ready, 1u);
        EXPECT_EQ(local_transport->stats().waves_published, 1u);

        if (context->rank() == coordinator_world_rank)
        {
            EXPECT_EQ(service.stats().proposals_published, 1u);
            EXPECT_EQ(publisher->stats().publications_completed, 1u);
        }
        else
        {
            EXPECT_EQ(service.stats().proposals_received, 1u);
            EXPECT_EQ(service.stats().proposal_receives_rearmed, 1u);
            EXPECT_EQ(publisher->stats().windows_received, 1u);
            EXPECT_EQ(publisher->stats().receives_armed, 2u);
        }

        poll_gate->stopAndDrain();
        EXPECT_TRUE(consensus->idle());
        ASSERT_EQ(local.fingerprints.size(), 1u);
        EXPECT_EQ(local.retirements, 1);
    }
} // namespace llaminar2::test
