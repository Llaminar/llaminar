/**
 * @file Test__MoEOverlayMPITransactionPublication.cpp
 * @brief Real-MPI proof of compact, immutable ExpertOverlay demand publication.
 *
 * Exercise the production proposal mailbox with a nonzero coordinator, changing
 * batch sizes and a retained old generation. Physical claims must cover the
 * exact live payload, and MPI byte counters must describe used bytes rather
 * than the capacity of a persistent receive. This is a model-free functional
 * regression in the existing residency-consensus preflight executable.
 * Terminal tests combine real rank-local model-retention obligations before
 * controller workers act, including a sole nonzero-rank reuse consumer.
 */

#include "execution/moe/MoEOverlayMPIResidencyProposalPublisher.h"
#include "execution/moe/MoEOverlayHostDemandMemoryPlan.h"
#include "execution/moe/MoEOverlayDeviceControllerGraphService.h"
#include "execution/runner/ModelContextRetirement.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "utils/MPIContext.h"

#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <thread>

namespace llaminar2::test
{
    namespace
    {
        /** @return A rank-local PMA certificate for the exact concurrent test BOM. */
        std::shared_ptr<PhysicalMemoryAuthority> transactionMemory(int rank, std::size_t bytes)
        {
            PhysicalMemoryBOMBuilder bom({.world_rank = rank, .device = DeviceId::cpu(),
                .total_bytes = bytes, .admission_available_bytes = bytes});
            bom.add(PhysicalMemoryOwner::ExecutionWorkspace, bytes);
            PhysicalMemoryPlanBuilder plan;
            plan.add(bom.build());
            return std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(plan.build()), rank);
        }

        /**
         * @brief Progress and semantically accept a decoded packet without blocking waits.
         * @param publisher The production model-lifetime publication lane.
         * @return The follower's immutable proposal, or null on its coordinator.
         * @throws std::runtime_error On a broken protocol or missed test deadline.
         */
        std::shared_ptr<const MoEOverlayDistributedResidencyProposal> finishPublication(
            MoEOverlayMPIResidencyProposalPublisher &publisher)
        {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            std::shared_ptr<const MoEOverlayDistributedResidencyProposal> retained;
            for (;;)
            {
                std::shared_ptr<const MoEOverlayDistributedResidencyProposal> received;
                std::string error;
                const auto progress = publisher.poll(&received, &error);
                if (progress == MoEOverlayResidencyWaveProgress::Failed)
                    throw std::runtime_error(error);
                if (progress == MoEOverlayResidencyWaveProgress::Ready)
                {
                    if (!received) return retained;
                    retained = std::move(received);
                    if (!publisher.acceptReceivedProposal(retained->plan.histogram_window->generation, &error))
                        throw std::runtime_error(error);
                }
                if (std::chrono::steady_clock::now() >= deadline)
                    throw std::runtime_error("Transaction publication missed its bounded progress deadline");
                std::this_thread::yield();
            }
        }
    }

    /** Real rank agreement must retain peer obligations without inventing any. */
    TEST(Test__MoEOverlayMPITransactionPublication, TerminalDrainJoinsModelRetentionAcrossRanks)
    {
        int rank = -1;
        int size = 0;
        ASSERT_EQ(MPI_Comm_rank(MPI_COMM_WORLD, &rank), MPI_SUCCESS);
        ASSERT_EQ(MPI_Comm_size(MPI_COMM_WORLD, &size), MPI_SUCCESS);
        ASSERT_EQ(size, 2);
        MPIContext context(rank, size, MPI_COMM_WORLD);
        using Intent = MoEOverlayDeviceControllerDrainIntent;
        // Repeatedly alternate no consumer, either sole consumer, and both.
        // A previous retained model must not make the next discarded one move.
        for (int iteration = 0; iteration < 20; ++iteration)
        {
            for (int retained_ranks = 0; retained_ranks < 4; ++retained_ranks)
            {
                std::shared_ptr<ModelContextReuseAuthority> authority;
                if ((retained_ranks & (1 << rank)) != 0)
                    authority = std::make_shared<ModelContextReuseAuthority>();
                const auto local = modelContextOverlayDrainIntent(
                    authority, MoERebalanceRuntimeMode::Dynamic);
                EXPECT_EQ(agreeMoEOverlayDeviceControllerDrainIntent(context, local),
                          retained_ranks == 0 ? Intent::ReleaseResources : Intent::RestorePreparedContext);
            }
        }
        // Even malformed local intent must enter the same collective first,
        // so every rank rejects it rather than stranding its peer in MPI.
        EXPECT_THROW((void)agreeMoEOverlayDeviceControllerDrainIntent(context,
            rank == 1 ? static_cast<Intent>(255) : Intent::ReleaseResources), std::logic_error);
        EXPECT_EQ(agreeMoEOverlayDeviceControllerDrainIntent(context, Intent::ReleaseResources),
                  Intent::ReleaseResources);
    }

    TEST(Test__MoEOverlayMPITransactionPublication, CompactBatchesSurviveMailboxReuseAndReleaseExactClaims)
    {
        int rank = -1;
        int size = 0;
        ASSERT_EQ(MPI_Comm_rank(MPI_COMM_WORLD, &rank), MPI_SUCCESS);
        ASSERT_EQ(MPI_Comm_size(MPI_COMM_WORLD, &size), MPI_SUCCESS);
        ASSERT_EQ(size, 2);
        auto context = std::make_shared<MPIContext>(rank, size, MPI_COMM_WORLD);
        constexpr int layers = 2;
        constexpr int experts = 3;
        constexpr int coordinator = 1;
        const MoEOverlayHostDemandMemoryPlan demand_memory({.num_layers = layers, .num_experts = experts,
            .top_k = 2, .initial_window_rows = 4, .maximum_window_rows = 16, .maximum_invocation_rows = 4,
            .publication = MoEOverlayDemandPublicationScope::Distributed});
        const auto capacity = demand_memory.capacity();
        const auto mailbox_bytes = demand_memory.mailboxBytes();
        // Production admission owns this equation. The current source, decoded
        // copy and an old retained observation stress all three snapshot slots.
        auto memory = transactionMemory(rank, demand_memory.allocationBytes());
        const auto claimed = [&] {
            return memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ExecutionWorkspace,
                                         PhysicalMemoryMaterializationKind::NewAllocation);
        };
        std::shared_ptr<const DecodeExpertTransactionWindow> oldest;
        {
            ExpertHistogramTransactionConfig admission{capacity, memory};
            DecodeExpertHistogramConfig config;
            config.num_layers = layers;
            config.num_experts = experts;
            config.top_k = 2;
            config.window_size = capacity.target_rows;
            config.sockets = {DeviceId::cpu(), DeviceId(DeviceType::CPU, 1)};
            config.ownership = MoELayeredExpertOwnership::uniform(layers, 2, {0, 0, 1});
            config.transaction_demand = admission;
            DecodeExpertHistogram histogram(config);
            const auto bank_bytes = claimed();
            MoEOverlayMPIResidencyProposalPublisher publisher({
                .mpi_context = context, .coordinator_world_rank = coordinator,
                .num_layers = layers, .num_experts = experts, .perf_device = "mpi_transaction_test",
                .transaction_demand = admission,
            });
            EXPECT_EQ(claimed(), bank_bytes + mailbox_bytes);
            std::uint64_t total_sent = 0;
            for (int generation = 0; generation < 20; ++generation)
            {
                SCOPED_TRACE(generation);
                // Alternate larger and smaller packets: the smaller receive
                // must never decode stale trailing bytes from its predecessor.
                const int prefill_rows = generation % 2 == 0 ? 3 : 1;
                const int ids[]{0, 1, 1, 2, 0, 2, -1, -1};
                const float weights[]{0.5f, 0.5f};
                std::array<std::uint64_t, experts> scratch{};
                for (int layer = 0; layer < layers; ++layer)
                {
                    ASSERT_TRUE(histogram.mergeRoutedExpertRows(ids, {
                        .source = ExpertHistogramSource::PrefillChunk, .layer_idx = layer,
                        .real_token_count = prefill_rows, .bucket_token_count = 4,
                        .top_k = 2, .route_stride = 2, .count_window_tokens = true,
                    }, scratch));
                    histogram.record(layer, ids, weights, 2);
                }
                auto source = std::make_shared<const DecodeExpertHistogramWindow>(histogram.freezeAndRotateWindow());
                // The very first real RCU generation is zero, not an absent
                // identity; it must be acknowledged as faithfully as later ones.
                ASSERT_EQ(source->generation, static_cast<std::uint64_t>(generation));
                MoEOverlayDistributedResidencyProposal proposal{
                    .plan = {.expected_epoch = static_cast<std::uint64_t>(generation + 1),
                        .num_layers = layers, .num_experts = experts, .histogram_window = source,
                        .entries = std::vector<MoEOverlayAuthoritativeResidencyEntry>(layers * experts,
                            {.candidate_tier_idx = 0, .candidate_owner_participant = 0})},
                    .execution_fingerprint = {.low = 1, .high = 2},
                    .policy_fingerprint = {.low = 3, .high = 4},
                };
                ASSERT_TRUE(proposal.valid());
                const auto packet_bytes = moeOverlayDistributedResidencyProposalWireBytes(proposal);
                ASSERT_LT(packet_bytes, mailbox_bytes);
                total_sent += packet_bytes;
                if (publisher.isCoordinator())
                {
                    // Dropping evidence cannot silently select marginal-only
                    // publication on an already admitted transaction lane.
                    auto counts_only = std::make_shared<DecodeExpertHistogramWindow>(*source);
                    counts_only->transaction_demand.reset();
                    auto invalid = proposal;
                    invalid.plan.histogram_window = std::move(counts_only);
                    EXPECT_FALSE(publisher.beginPublish(invalid));
                    std::string error;
                    ASSERT_TRUE(publisher.beginPublish(proposal, &error)) << error;
                }
                const auto received = finishPublication(publisher);
                if (!publisher.isCoordinator()) ASSERT_NE(received, nullptr);
                const auto &window = publisher.isCoordinator() ? source : received->plan.histogram_window;
                ASSERT_TRUE(window->valid());
                EXPECT_EQ(fingerprintDecodeExpertHistogramWindow(*window), fingerprintDecodeExpertHistogramWindow(*source));
                ASSERT_NE(window->transaction_demand, nullptr);
                for (int layer = 0; layer < layers; ++layer)
                {
                    const auto batches = window->transaction_demand->layerTransactions(layer);
                    ASSERT_EQ(batches.size(), 2u);
                    EXPECT_EQ(batches[0].logical_rows, prefill_rows);
                    EXPECT_EQ(batches[0].phase, ExpertHistogramSource::PrefillChunk);
                    EXPECT_EQ(batches[1].phase, ExpertHistogramSource::DecodeToken);
                }
                if (!oldest) oldest = window->transaction_demand;
                EXPECT_EQ(oldest->routes(0, 0).logical_rows, 3u);
                EXPECT_EQ(oldest->routes(0, 0).expert_ids[5], 2);
                if (!publisher.isCoordinator() && generation != 19)
                    ASSERT_TRUE(publisher.armReceive());
            }
            const auto stats = publisher.stats();
            EXPECT_EQ(stats.bytes_sent, rank == coordinator ? total_sent : 0u);
            EXPECT_EQ(stats.bytes_received, rank == coordinator ? 0u : total_sent);
            EXPECT_EQ(stats.publications_completed, rank == coordinator ? 20u : 0u);
            EXPECT_EQ(stats.acknowledgements_completed, rank == coordinator ? 0u : 20u);
            EXPECT_EQ(stats.blocking_inference_waits, 0u);
        }
        ASSERT_NE(oldest, nullptr);
        EXPECT_EQ(claimed(), oldest->allocationBytes());
        oldest.reset();
        EXPECT_EQ(claimed(), 0u);
    }
} // namespace llaminar2::test
