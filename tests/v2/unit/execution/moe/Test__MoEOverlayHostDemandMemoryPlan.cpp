/**
 * @file Test__MoEOverlayHostDemandMemoryPlan.cpp
 * @brief Device-free proof that host evidence geometry and allocation admission agree.
 *
 * Exercise non-default windows/batches, overflow rejection, compact MPI sizing,
 * and the live PMA claim lifetime. No weights, backend runtime or synthetic
 * inference are needed to expose an under-admitted evidence bank.
 */
#include "execution/moe/MoEOverlayHostDemandMemoryPlan.h"
#include "execution/moe/MoEOverlayDistributedResidencyProtocol.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "../../../utils/ObservedExpertDemandFixture.h"

#include <gtest/gtest.h>
#include <array>
#include <limits>

namespace llaminar2::test
{
    TEST(MoEOverlayHostDemandMemoryPlan, ComposesPayloadOwnersWithoutDuplicatingTheirLayout)
    {
        for (const int top_k : {1, 2, 8, 16})
            for (const int maximum_rows : {1, 16, 48, 512, 8192})
                for (const auto publication : {MoEOverlayDemandPublicationScope::ProcessLocal,
                                              MoEOverlayDemandPublicationScope::Distributed})
                {
                    const MoEOverlayHostDemandMemoryPlan plan({.num_layers = 41, .num_experts = 256,
                        .top_k = top_k, .initial_window_rows = 17, .maximum_window_rows = 73,
                        .maximum_invocation_rows = maximum_rows, .publication = publication});
                    EXPECT_EQ(plan.capacity().target_rows, 73u);
                    EXPECT_EQ(plan.capacity().max_transaction_rows, maximum_rows);
                    EXPECT_EQ(plan.capacity().top_k, top_k);
                    EXPECT_EQ(plan.mutableBankBytes(), 2u * 41u * plan.capacity().allocationBytes());
                    EXPECT_EQ(plan.observationBytes(), 3u * DecodeExpertTransactionWindow::maximumAllocationBytes(
                        plan.capacity(), 41, 256));
                    EXPECT_EQ(plan.mailboxBytes(), publication == MoEOverlayDemandPublicationScope::ProcessLocal ? 0u :
                        moeOverlayDistributedResidencyProposalWireBytes(41, 256, &plan.capacity()));
                    EXPECT_EQ(plan.allocationBytes(), plan.mutableBankBytes() + plan.observationBytes() + plan.mailboxBytes());
                }
    }

    TEST(MoEOverlayHostDemandMemoryPlan, RejectsInvalidGeometryAndOversizedMpiMailboxBeforeAllocation)
    {
        const MoEOverlayHostDemandGeometry valid{.num_layers = 2, .num_experts = 4, .top_k = 2,
            .initial_window_rows = 3, .maximum_window_rows = 7, .maximum_invocation_rows = 4};
        for (const auto member : {&MoEOverlayHostDemandGeometry::num_layers, &MoEOverlayHostDemandGeometry::num_experts,
             &MoEOverlayHostDemandGeometry::top_k, &MoEOverlayHostDemandGeometry::initial_window_rows,
             &MoEOverlayHostDemandGeometry::maximum_invocation_rows})
        {
            auto bad = valid;
            bad.*member = 0;
            EXPECT_THROW((void)MoEOverlayHostDemandMemoryPlan(bad), std::invalid_argument);
        }
        auto bad = valid;
        bad.maximum_window_rows = 2;
        EXPECT_THROW((void)MoEOverlayHostDemandMemoryPlan(bad), std::invalid_argument);
        bad = valid;
        bad.top_k = 5;
        EXPECT_THROW((void)MoEOverlayHostDemandMemoryPlan(bad), std::invalid_argument);
        bad = valid;
        bad.publication = static_cast<MoEOverlayDemandPublicationScope>(99);
        EXPECT_THROW((void)MoEOverlayHostDemandMemoryPlan(bad), std::invalid_argument);
        bad = valid;
        bad.maximum_window_rows = std::numeric_limits<int>::max();
        bad.publication = MoEOverlayDemandPublicationScope::Distributed;
        EXPECT_THROW((void)MoEOverlayHostDemandMemoryPlan(bad), std::overflow_error);
        auto fixed = valid;
        fixed.maximum_window_rows = 0;
        EXPECT_EQ(MoEOverlayHostDemandMemoryPlan(fixed).capacity().target_rows, 3u);
    }

    TEST(MoEOverlayHostDemandMemoryPlan, ExactAdmissionOwnsBanksAndOverlappingFrozenGenerations)
    {
        const MoEOverlayHostDemandMemoryPlan plan({.num_layers = 1, .num_experts = 4, .top_k = 2,
            .initial_window_rows = 3, .maximum_window_rows = 7, .maximum_invocation_rows = 4});
        PhysicalMemoryBOMBuilder bom({.world_rank = 0, .device = DeviceId::cpu(),
            .total_bytes = plan.allocationBytes(), .admission_available_bytes = plan.allocationBytes()});
        bom.add(PhysicalMemoryOwner::ExecutionWorkspace, plan.allocationBytes());
        PhysicalMemoryPlanBuilder physical_plan;
        physical_plan.add(bom.build());
        auto memory = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(physical_plan.build()), 0);
        const auto claimed = [&] { return memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ExecutionWorkspace,
            PhysicalMemoryMaterializationKind::NewAllocation); };
        {
            DecodeExpertHistogramConfig config;
            config.num_layers = 1;
            config.num_experts = 4;
            config.top_k = 2;
            config.window_size = 3;
            config.sockets = {DeviceId::cpu()};
            config.ownership = MoELayeredExpertOwnership::uniform(1, 1, {0, 0, 0, 0});
            // Retained setup may copy the immutable plan, but it must bind the
            // very same geometry and live ledger when rebuilding a runner.
            const auto retained = plan;
            config.transaction_demand = retained.bind(config, memory);
            EXPECT_EQ(config.transaction_demand->memory.get(), memory.get());
            EXPECT_THROW((void)retained.bind(config, nullptr), std::invalid_argument);
            for (const auto member : {&DecodeExpertHistogramConfig::num_layers,
                 &DecodeExpertHistogramConfig::num_experts, &DecodeExpertHistogramConfig::top_k,
                 &DecodeExpertHistogramConfig::window_size})
            {
                auto stale = config;
                ++(stale.*member);
                EXPECT_THROW((void)retained.bind(stale, memory), std::invalid_argument);
            }
            DecodeExpertHistogram histogram(config);
            EXPECT_EQ(claimed(), plan.mutableBankBytes());
            std::array<DecodeExpertHistogramWindow, 3> observations;
            for (auto &window : observations)
            {
                // A whole four-row call crosses the three-row sample target.
                // Storage must retain it as one invocation, never truncate it.
                recordObservedExpertBatch(histogram, 0, ExpertHistogramSource::PrefillChunk,
                    std::array<int, 8>{0, 1, 1, 2, 2, 3, 3, 0});
                window = histogram.freezeAndRotateWindow();
                ASSERT_NE(window.transaction_demand, nullptr);
                EXPECT_EQ(window.transaction_demand->layerTransactions(0).size(), 1u);
                EXPECT_EQ(window.token_count, 4u);
            }
            EXPECT_LE(claimed(), plan.allocationBytes());
            EXPECT_THROW(histogram.setWindowSize(8), std::invalid_argument);
        }
        EXPECT_EQ(claimed(), 0u);
    }
}
