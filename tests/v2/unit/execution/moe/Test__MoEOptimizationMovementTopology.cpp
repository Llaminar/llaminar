/**
 * @file Test__MoEOptimizationMovementTopology.cpp
 * @brief Device-free tests of capacity-resolved movement evidence geometry.
 *
 * Real expert membership, not requested caps or observed moves, establishes
 * exchange opportunities. No weights, devices, MPI state or memory ledger are
 * created; backend addresses are inert topology declarations.
 */
#include "execution/moe/MoEOptimizationMovementTopology.h"
#include "execution/moe/MoERoutedExpertPlacementPlan.h"
#include <gtest/gtest.h>

using namespace llaminar2;

namespace
{
    /** @return A resolved two-tier plan with arbitrary signed priorities. */
    MoERoutedExpertPlacementPlan topology(GlobalDeviceAddress first, GlobalDeviceAddress second)
    {
        MoERoutedExpertPlacementPlan plan;
        plan.enabled = true;
        plan.domains = {{.name = "one", .participants = {first}}, {.name = "two", .participants = {second}}};
        plan.routed_tiers = {{.name = "alpha", .domain = "one", .priority = -7},
                             {.name = "beta", .domain = "two", .priority = 53}};
        plan.placements = {{.layer = 0, .routed_expert_tier = {0, 1}}};
        plan.authority_execution = resolveMoEOverlayAuthorityExecutionKind(plan);
        return plan;
    }
}

/** All-GPU ownership is device resident regardless of vendor or tier order. */
TEST(MoEOptimizationMovementTopology, BackendAndTierOrderDoNotInventHostAuthority)
{
    for (const auto first : {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(0)})
        for (const auto second : {GlobalDeviceAddress::cpu(1), GlobalDeviceAddress::cuda(1), GlobalDeviceAddress::rocm(1)})
        {
            auto plan = topology(first, second);
            const auto result = describeMoEOptimizationMovementTopology(&plan);
            EXPECT_EQ(result.authority, first.isGPU() && second.isGPU()
                ? MoEOptimizationAuthority::Device : MoEOptimizationAuthority::Host);
            EXPECT_EQ(result.axes, MoEOptimizationMovementAxes::TierResidency);
            EXPECT_TRUE(result.valid());
            std::swap(plan.routed_tiers[0].priority, plan.routed_tiers[1].priority);
            EXPECT_EQ(describeMoEOptimizationMovementTopology(&plan), result);
        }
}

/** Only concrete, same-layer occupied quotas permit tier exchange. */
TEST(MoEOptimizationMovementTopology, RequestedCapacityAndSeparateLayersAreNotMembership)
{
    auto plan = topology(GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cpu(1));
    plan.routed_tiers[1].max_experts_per_layer = 100;
    plan.placements[0].routed_expert_tier = {0, 0};
    EXPECT_EQ(availableMoEOptimizationMovementAxes(plan), MoEOptimizationMovementAxes::None);
    plan.placements.push_back({.layer = 1, .routed_expert_tier = {1, 1}});
    EXPECT_EQ(availableMoEOptimizationMovementAxes(plan), MoEOptimizationMovementAxes::None);
    plan.placements[1].routed_expert_tier = {0, 1};
    EXPECT_EQ(availableMoEOptimizationMovementAxes(plan), MoEOptimizationMovementAxes::TierResidency);
    plan.routed_tiers[1].priority = plan.routed_tiers[0].priority;
    EXPECT_EQ(availableMoEOptimizationMovementAxes(plan), MoEOptimizationMovementAxes::None);
}

/** Independent whole-expert ownership, not TP participant count, permits skew correction. */
TEST(MoEOptimizationMovementTopology, SingleTierAndTwoAxesRespectComputeOwnership)
{
    for (const auto endpoints : std::vector<std::vector<GlobalDeviceAddress>>{
             {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)},
             {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)},
             {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)}})
    {
        auto plan = topology(endpoints[0], endpoints[1]);
        plan.domains[0].participants = endpoints;
        plan.placements[0].routed_expert_tier = {0, 0};
        EXPECT_EQ(availableMoEOptimizationMovementAxes(plan), MoEOptimizationMovementAxes::ParticipantPlacement);
        plan.placements[0].routed_expert_tier.push_back(1);
        EXPECT_EQ(availableMoEOptimizationMovementAxes(plan), MoEOptimizationMovementAxes::Both);
        for (const auto policy : {RoutedExpertComputePolicy::TensorSharded, RoutedExpertComputePolicy::Replicated})
        {
            plan.domains[0].routed_compute_policy = policy;
            EXPECT_EQ(availableMoEOptimizationMovementAxes(plan), MoEOptimizationMovementAxes::TierResidency);
        }
        plan.domains[0].routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        plan.placements[0].routed_expert_tier = {0, 1};
        EXPECT_EQ(availableMoEOptimizationMovementAxes(plan), MoEOptimizationMovementAxes::TierResidency);
    }
}

/** Unsupported and stale states fail; absence of an overlay is explicit. */
TEST(MoEOptimizationMovementTopology, RejectsUnresolvedAndMalformedPlans)
{
    EXPECT_EQ(describeMoEOptimizationMovementTopology(nullptr).authority, MoEOptimizationAuthority::None);
    MoERoutedExpertPlacementPlan disabled;
    EXPECT_TRUE(describeMoEOptimizationMovementTopology(&disabled).valid());
    const auto valid = topology(GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(1));
    for (const auto authority : {MoEOverlayAuthorityExecutionKind::Unresolved, MoEOverlayAuthorityExecutionKind::HostResident})
    {
        auto plan = valid;
        plan.authority_execution = authority;
        EXPECT_THROW((void)describeMoEOptimizationMovementTopology(&plan), std::invalid_argument);
    }
    for (int mutation = 0; mutation < 6; ++mutation)
    {
        auto plan = valid;
        switch (mutation)
        {
        case 0: plan.placements.clear(); break;
        case 1: plan.placements[0].routed_expert_tier = {-1}; break;
        case 2: plan.placements[0].routed_expert_tier = {2}; break;
        case 3: plan.domains[0].participants.clear(); break;
        case 4: plan.placements.push_back(plan.placements[0]); break;
        case 5: plan.routed_tiers[0].domain = "missing"; break;
        }
        EXPECT_THROW((void)describeMoEOptimizationMovementTopology(&plan), std::invalid_argument);
    }
}
