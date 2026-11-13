/**
 * @file Test__MoEGraphComputePolicy.cpp
 * @brief Replay scalar graph policy projection without native context admission.
 *
 * Participant counts describe pure declarations; they do not qualify physical
 * socket deployments. The original CPU source policy and membership survive
 * projection onto the independently executed accelerator graph lane.
 */
#include "execution/moe/MoEGraphOwnerProjection.h"
#include <gtest/gtest.h>
using namespace llaminar2;

/** @test CPU fragment policy remains independent while unsupported GPU mixtures reject. */
TEST(MoEGraphComputePolicy, CPUFragmentsPreserveGPUScalarPolicy)
{
    for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
    for (int count = 1; count <= 8; ++count)
    {
        SCOPED_TRACE(device.toString() + " CPU participants=" + std::to_string(count));
        MoERoutedExpertPlacementPlan plan;
        RoutedExpertDomain gpu;
        gpu.name = "gpu";
        gpu.participants = {GlobalDeviceAddress::fromLocalDeviceId(device)};
        RoutedExpertDomain cpu;
        cpu.name = "cpu";
        cpu.scope = ExecutionDomainScope::NODE_LOCAL;
        cpu.routed_compute_policy = RoutedExpertComputePolicy::TensorSharded;
        for (int i = 0; i < count; ++i)
        {
            cpu.participants.push_back(GlobalDeviceAddress::cpu(i));
            cpu.world_ranks.push_back(i);
        }
        plan.domains = {gpu, cpu};
        plan.routed_tiers.resize(2);
        plan.routed_tiers[0].domain = "gpu";
        plan.routed_tiers[1].domain = "cpu";
        std::string error;
        ASSERT_EQ(routedOverlayComputePolicy(plan, &error), RoutedExpertComputePolicy::Apportioned);
        EXPECT_EQ(plan.domains[1].routed_compute_policy, RoutedExpertComputePolicy::TensorSharded);
        EXPECT_EQ(plan.domains[1].participants, cpu.participants);
        EXPECT_EQ(plan.domains[1].world_ranks, cpu.world_ranks);

        plan.domains[0].routed_compute_policy = RoutedExpertComputePolicy::GateUpOwnedDownColumns;
        EXPECT_EQ(routedOverlayComputePolicy(plan, &error), RoutedExpertComputePolicy::GateUpOwnedDownColumns);
        auto cpu_only = plan;
        cpu_only.routed_tiers.erase(cpu_only.routed_tiers.begin());
        EXPECT_EQ(routedOverlayComputePolicy(cpu_only, &error), RoutedExpertComputePolicy::TensorSharded);

        // A referenced second GPU domain retains its own scalar policy and
        // cannot use the CPU fragment exception to hide a conflicting lowering.
        plan.domains[1].participants = {GlobalDeviceAddress::fromLocalDeviceId(device)};
        plan.domains[1].world_ranks.clear();
        plan.domains[1].scope = ExecutionDomainScope::RANK_LOCAL;
        EXPECT_FALSE(routedOverlayComputePolicy(plan, &error));
        EXPECT_NE(error.find("mixed routed expert compute policies"), std::string::npos);
    }
}
