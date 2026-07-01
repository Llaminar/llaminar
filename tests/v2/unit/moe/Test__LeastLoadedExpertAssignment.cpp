/**
 * @file Test__LeastLoadedExpertAssignment.cpp
 * @brief Unit tests for the backend-neutral Least-Loaded EP assignment helper.
 */

#include <gtest/gtest.h>

#include "execution/moe/LeastLoadedExpertAssignment.h"
#include "execution/moe/RoutedExpertAssignmentPolicyShared.h"

#include <array>
#include <cstdint>
#include <vector>

using namespace llaminar2::least_loaded_ep;
namespace routed_expert_assignment = llaminar2::routed_expert_assignment;

namespace
{
    struct PlannerFixture
    {
        std::vector<uint32_t> sorted;
        std::vector<uint64_t> pending;
        std::vector<uint64_t> assigned;
        std::vector<LeastLoadedExpertAssignmentSpan> spans;
        std::vector<LeastLoadedExpertWeightTransfer> transfers;
        LeastLoadedExpertAssignmentStatus status{};

        explicit PlannerFixture(uint32_t expert_count,
                                uint32_t participant_count,
                                uint32_t span_capacity = 32,
                                uint32_t transfer_capacity = 16)
            : sorted(expert_count),
              pending(participant_count),
              assigned(participant_count),
              spans(span_capacity),
              transfers(transfer_capacity)
        {
        }

        bool plan(const std::vector<uint64_t> &loads,
                  const std::vector<uint32_t> &owners,
                  LeastLoadedExpertAssignmentConfig config)
        {
            LeastLoadedExpertAssignmentWorkspace workspace{
                sorted.data(),
                pending.data(),
                assigned.data()};
            return planLeastLoadedExpertAssignment(
                loads.data(),
                owners.data(),
                config,
                workspace,
                spans.data(),
                static_cast<uint32_t>(spans.size()),
                transfers.data(),
                static_cast<uint32_t>(transfers.size()),
                &status);
        }

        bool planWithResidency(const std::vector<uint64_t> &loads,
                               const std::vector<uint32_t> &owners,
                               const std::vector<uint32_t> &resident_masks,
                               LeastLoadedExpertAssignmentConfig config)
        {
            LeastLoadedExpertAssignmentWorkspace workspace{
                sorted.data(),
                pending.data(),
                assigned.data()};
            return planLeastLoadedExpertAssignment(
                loads.data(),
                owners.data(),
                config,
                workspace,
                spans.data(),
                static_cast<uint32_t>(spans.size()),
                transfers.data(),
                static_cast<uint32_t>(transfers.size()),
                &status,
                resident_masks.data());
        }

        bool planPolicy(const std::vector<uint64_t> &loads,
                        const std::vector<uint32_t> &owners,
                        routed_expert_assignment::PolicyConfig config)
        {
            LeastLoadedExpertAssignmentWorkspace workspace{
                sorted.data(),
                pending.data(),
                assigned.data()};
            return routed_expert_assignment::planRoutedExpertAssignment(
                loads.data(),
                owners.data(),
                config,
                workspace,
                spans.data(),
                static_cast<uint32_t>(spans.size()),
                transfers.data(),
                static_cast<uint32_t>(transfers.size()),
                &status);
        }

        bool planTransfersOnly(const std::vector<uint64_t> &loads,
                               const std::vector<uint32_t> &owners,
                               LeastLoadedExpertAssignmentConfig config)
        {
            LeastLoadedExpertAssignmentWorkspace workspace{
                sorted.data(),
                pending.data(),
                assigned.data()};
            return planLeastLoadedExpertWeightTransfers(
                loads.data(),
                owners.data(),
                config,
                workspace,
                transfers.data(),
                static_cast<uint32_t>(transfers.size()),
                &status);
        }
    };

    LeastLoadedExpertAssignmentConfig configFor(uint32_t expert_count,
                                                uint32_t participant_count)
    {
        LeastLoadedExpertAssignmentConfig config;
        config.expert_count = expert_count;
        config.participant_count = participant_count;
        config.lambda_numerator = 13;
        config.lambda_denominator = 10;
        config.alpha_numerator = 1;
        config.alpha_denominator = 1;
        config.min_chunk_tokens = 0;
        return config;
    }
} // namespace

TEST(Test__LeastLoadedExpertAssignment, BalancedRoutingSelectsStandardEP)
{
    std::vector<uint64_t> loads{10, 10, 10, 10};
    std::vector<uint32_t> owners{0, 0, 1, 1};
    auto config = configFor(4, 2);

    PlannerFixture fixture(config.expert_count, config.participant_count);
    ASSERT_TRUE(fixture.plan(loads, owners, config));

    EXPECT_EQ(fixture.status.total_load, 40u);
    EXPECT_EQ(fixture.status.max_expert_load, 10u);
    EXPECT_EQ(fixture.status.standard_load_min, 20u);
    EXPECT_EQ(fixture.status.standard_load_max, 20u);
    EXPECT_EQ(fixture.status.standard_load_spread, 0u);
    EXPECT_EQ(fixture.status.assigned_load_min, 20u);
    EXPECT_EQ(fixture.status.assigned_load_max, 20u);
    EXPECT_EQ(fixture.status.assigned_load_spread, 0u);
    EXPECT_EQ(fixture.status.assigned_load_spread_improvement, 0u);
    EXPECT_EQ(fixture.status.skipped_balanced, 1u);
    EXPECT_EQ(fixture.status.standard_ep_selected, 1u);
    EXPECT_EQ(fixture.status.span_count, 0u);
    EXPECT_EQ(fixture.status.weight_transfer_count, 0u);
}

TEST(Test__LeastLoadedExpertAssignment, PolicyDispatcherTreatsStaticOwnerAsFirstClassPolicy)
{
    std::vector<uint64_t> loads{80, 20, 0, 5};
    std::vector<uint32_t> owners{1, 0, 1, 0};
    routed_expert_assignment::PolicyConfig policy;
    policy.algorithm = routed_expert_assignment::Algorithm::StaticOwner;
    policy.least_loaded = configFor(4, 2);

    PlannerFixture fixture(policy.least_loaded.expert_count, policy.least_loaded.participant_count);
    ASSERT_TRUE(fixture.planPolicy(loads, owners, policy));

    EXPECT_EQ(fixture.status.standard_ep_selected, 1u);
    EXPECT_EQ(fixture.status.total_load, 105u);
    EXPECT_EQ(fixture.status.capacity_per_participant, 53u);
    EXPECT_EQ(fixture.status.standard_load_min, 25u);
    EXPECT_EQ(fixture.status.standard_load_max, 80u);
    EXPECT_EQ(fixture.status.standard_load_spread, 55u);
    EXPECT_EQ(fixture.status.assigned_load_min, 25u);
    EXPECT_EQ(fixture.status.assigned_load_max, 80u);
    EXPECT_EQ(fixture.status.assigned_load_spread, 55u);
    EXPECT_EQ(fixture.status.assigned_load_spread_improvement, 0u);
    EXPECT_EQ(fixture.status.span_count, 3u);
    EXPECT_EQ(fixture.status.native_rows, 105u);
    EXPECT_EQ(fixture.status.spilled_rows, 0u);
    EXPECT_EQ(fixture.status.weight_transfer_count, 0u);
    EXPECT_EQ(fixture.assigned[0], 25u);
    EXPECT_EQ(fixture.assigned[1], 80u);

    EXPECT_EQ(fixture.spans[0].expert, 0u);
    EXPECT_EQ(fixture.spans[0].destination_participant, 1u);
    EXPECT_EQ(fixture.spans[0].route_row_end, 80u);
    EXPECT_EQ(fixture.spans[1].expert, 1u);
    EXPECT_EQ(fixture.spans[1].destination_participant, 0u);
    EXPECT_EQ(fixture.spans[2].expert, 3u);
    EXPECT_EQ(fixture.spans[2].destination_participant, 0u);
}

TEST(Test__LeastLoadedExpertAssignment, PolicyDispatcherRoutesLeastLoadedEPToSharedAlgorithm)
{
    std::vector<uint64_t> loads{80, 20, 0, 0};
    std::vector<uint32_t> owners{0, 0, 1, 1};
    auto config = configFor(4, 2);

    PlannerFixture direct(config.expert_count, config.participant_count);
    ASSERT_TRUE(direct.plan(loads, owners, config));

    routed_expert_assignment::PolicyConfig policy;
    policy.algorithm = routed_expert_assignment::Algorithm::LeastLoadedEP;
    policy.least_loaded = config;
    PlannerFixture dispatched(config.expert_count, config.participant_count);
    ASSERT_TRUE(dispatched.planPolicy(loads, owners, policy));

    EXPECT_EQ(dispatched.status.span_count, direct.status.span_count);
    EXPECT_EQ(dispatched.status.native_rows, direct.status.native_rows);
    EXPECT_EQ(dispatched.status.spilled_rows, direct.status.spilled_rows);
    EXPECT_EQ(dispatched.status.weight_transfer_count, direct.status.weight_transfer_count);
    ASSERT_EQ(dispatched.status.span_count, 3u);
    for (uint32_t index = 0; index < dispatched.status.span_count; ++index)
    {
        EXPECT_EQ(dispatched.spans[index].expert, direct.spans[index].expert);
        EXPECT_EQ(dispatched.spans[index].owner_participant, direct.spans[index].owner_participant);
        EXPECT_EQ(dispatched.spans[index].destination_participant,
                  direct.spans[index].destination_participant);
        EXPECT_EQ(dispatched.spans[index].route_row_begin, direct.spans[index].route_row_begin);
        EXPECT_EQ(dispatched.spans[index].route_row_end, direct.spans[index].route_row_end);
        EXPECT_EQ(dispatched.spans[index].needs_foreign_weight,
                  direct.spans[index].needs_foreign_weight);
    }
}

TEST(Test__LeastLoadedExpertAssignment, TransferOnlyPlannerMatchesFullPlannerForeignWeights)
{
    std::vector<uint64_t> loads{80, 20, 0, 0};
    std::vector<uint32_t> owners{0, 0, 1, 1};
    auto config = configFor(4, 2);

    PlannerFixture full(config.expert_count, config.participant_count);
    ASSERT_TRUE(full.plan(loads, owners, config));
    ASSERT_EQ(full.status.weight_transfer_count, 1u);

    PlannerFixture transfer_only(config.expert_count, config.participant_count);
    ASSERT_TRUE(transfer_only.planTransfersOnly(loads, owners, config));

    EXPECT_EQ(transfer_only.status.total_load, full.status.total_load);
    EXPECT_EQ(transfer_only.status.standard_load_spread, full.status.standard_load_spread);
    EXPECT_EQ(transfer_only.status.assigned_load_spread, full.status.assigned_load_spread);
    EXPECT_EQ(transfer_only.status.assigned_load_spread_improvement,
              full.status.assigned_load_spread_improvement);
    EXPECT_EQ(transfer_only.status.span_count, full.status.span_count);
    EXPECT_EQ(transfer_only.status.native_rows, full.status.native_rows);
    EXPECT_EQ(transfer_only.status.spilled_rows, full.status.spilled_rows);
    ASSERT_EQ(transfer_only.status.weight_transfer_count, full.status.weight_transfer_count);
    EXPECT_EQ(transfer_only.transfers[0].expert, full.transfers[0].expert);
    EXPECT_EQ(transfer_only.transfers[0].source_participant,
              full.transfers[0].source_participant);
    EXPECT_EQ(transfer_only.transfers[0].destination_participant,
              full.transfers[0].destination_participant);
}

TEST(Test__LeastLoadedExpertAssignment, EmitsNativeAssignmentsWhenBalancedSkipDisabled)
{
    std::vector<uint64_t> loads{10, 10, 10, 10};
    std::vector<uint32_t> owners{0, 0, 1, 1};
    auto config = configFor(4, 2);
    config.enable_balanced_skip = false;

    PlannerFixture fixture(config.expert_count, config.participant_count);
    ASSERT_TRUE(fixture.plan(loads, owners, config));

    EXPECT_EQ(fixture.status.capacity_per_participant, 20u);
    EXPECT_EQ(fixture.status.span_count, 4u);
    EXPECT_EQ(fixture.status.native_rows, 40u);
    EXPECT_EQ(fixture.status.spilled_rows, 0u);
    EXPECT_EQ(fixture.assigned[0], 20u);
    EXPECT_EQ(fixture.assigned[1], 20u);
}

TEST(Test__LeastLoadedExpertAssignment, PartiallyAssignsNativeThenSpillsExcess)
{
    std::vector<uint64_t> loads{80, 20, 0, 0};
    std::vector<uint32_t> owners{0, 0, 1, 1};
    auto config = configFor(4, 2);

    PlannerFixture fixture(config.expert_count, config.participant_count);
    ASSERT_TRUE(fixture.plan(loads, owners, config));

    ASSERT_EQ(fixture.status.span_count, 3u);
    EXPECT_EQ(fixture.status.capacity_per_participant, 50u);
    EXPECT_EQ(fixture.status.standard_load_min, 0u);
    EXPECT_EQ(fixture.status.standard_load_max, 100u);
    EXPECT_EQ(fixture.status.standard_load_spread, 100u);
    EXPECT_EQ(fixture.status.assigned_load_min, 50u);
    EXPECT_EQ(fixture.status.assigned_load_max, 50u);
    EXPECT_EQ(fixture.status.assigned_load_spread, 0u);
    EXPECT_EQ(fixture.status.assigned_load_spread_improvement, 100u);
    EXPECT_EQ(fixture.status.required_spread_improvement, 0u);
    EXPECT_EQ(fixture.status.native_rows, 50u);
    EXPECT_EQ(fixture.status.spilled_rows, 50u);
    EXPECT_EQ(fixture.assigned[0], 50u);
    EXPECT_EQ(fixture.assigned[1], 50u);

    EXPECT_EQ(fixture.spans[0].expert, 0u);
    EXPECT_EQ(fixture.spans[0].destination_participant, 0u);
    EXPECT_EQ(fixture.spans[0].route_row_begin, 0u);
    EXPECT_EQ(fixture.spans[0].route_row_end, 30u);
    EXPECT_EQ(fixture.spans[0].needs_foreign_weight, 0u);

    EXPECT_EQ(fixture.spans[1].expert, 0u);
    EXPECT_EQ(fixture.spans[1].destination_participant, 1u);
    EXPECT_EQ(fixture.spans[1].route_row_begin, 30u);
    EXPECT_EQ(fixture.spans[1].route_row_end, 80u);
    EXPECT_EQ(fixture.spans[1].needs_foreign_weight, 1u);

    EXPECT_EQ(fixture.status.weight_transfer_count, 1u);
    EXPECT_EQ(fixture.transfers[0].expert, 0u);
    EXPECT_EQ(fixture.transfers[0].source_participant, 0u);
    EXPECT_EQ(fixture.transfers[0].destination_participant, 1u);
}

TEST(Test__LeastLoadedExpertAssignment, ResidentReplicaDestinationDoesNotRequireWeightTransfer)
{
    std::vector<uint64_t> loads{80, 20, 0, 0};
    std::vector<uint32_t> owners{0, 0, 1, 1};
    std::vector<uint32_t> resident_masks{0b11u, 0b01u, 0b10u, 0b10u};
    auto config = configFor(4, 2);

    PlannerFixture fixture(config.expert_count, config.participant_count);
    ASSERT_TRUE(fixture.planWithResidency(loads, owners, resident_masks, config));

    ASSERT_EQ(fixture.status.span_count, 3u);
    EXPECT_EQ(fixture.status.native_rows, 100u);
    EXPECT_EQ(fixture.status.spilled_rows, 0u);
    EXPECT_EQ(fixture.status.weight_transfer_count, 0u);
    EXPECT_EQ(fixture.spans[1].expert, 0u);
    EXPECT_EQ(fixture.spans[1].destination_participant, 1u);
    EXPECT_EQ(fixture.spans[1].needs_foreign_weight, 0u);
}

TEST(Test__LeastLoadedExpertAssignment, SpreadImprovementGateFallsBackWhenTransferCostDominates)
{
    std::vector<uint64_t> loads{80, 20, 0, 0};
    std::vector<uint32_t> owners{0, 0, 1, 1};
    auto config = configFor(4, 2);
    config.min_spread_improvement = 101;

    PlannerFixture fixture(config.expert_count, config.participant_count);
    ASSERT_TRUE(fixture.plan(loads, owners, config));

    EXPECT_EQ(fixture.status.standard_load_spread, 100u);
    EXPECT_EQ(fixture.status.assigned_load_spread, 100u);
    EXPECT_EQ(fixture.status.assigned_load_spread_improvement, 0u);
    EXPECT_EQ(fixture.status.required_spread_improvement, 101u);
    EXPECT_EQ(fixture.status.skipped_insufficient_spread_improvement, 1u);
    EXPECT_EQ(fixture.status.standard_ep_selected, 1u);
    EXPECT_EQ(fixture.status.span_count, 0u);
    EXPECT_EQ(fixture.status.weight_transfer_count, 0u);
    EXPECT_EQ(fixture.status.native_rows, 0u);
    EXPECT_EQ(fixture.status.spilled_rows, 0u);
}

TEST(Test__LeastLoadedExpertAssignment, SpreadImprovementGateCanPriceEachForeignTransfer)
{
    std::vector<uint64_t> loads{80, 20, 0, 0};
    std::vector<uint32_t> owners{0, 0, 1, 1};
    auto config = configFor(4, 2);
    config.min_spread_improvement_per_transfer = 64;

    PlannerFixture accepted(config.expert_count, config.participant_count);
    ASSERT_TRUE(accepted.plan(loads, owners, config));
    EXPECT_EQ(accepted.status.assigned_load_spread_improvement, 100u);
    EXPECT_EQ(accepted.status.required_spread_improvement, 64u);
    EXPECT_EQ(accepted.status.skipped_insufficient_spread_improvement, 0u);
    EXPECT_EQ(accepted.status.weight_transfer_count, 1u);

    config.min_spread_improvement_per_transfer = 128;
    PlannerFixture rejected(config.expert_count, config.participant_count);
    ASSERT_TRUE(rejected.plan(loads, owners, config));
    EXPECT_EQ(rejected.status.required_spread_improvement, 128u);
    EXPECT_EQ(rejected.status.skipped_insufficient_spread_improvement, 1u);
    EXPECT_EQ(rejected.status.standard_ep_selected, 1u);
    EXPECT_EQ(rejected.status.weight_transfer_count, 0u);
}

TEST(Test__LeastLoadedExpertAssignment, ForeignRowsGatePricesUsefulRowsPerTransfer)
{
    std::vector<uint64_t> loads{80, 20, 0, 0};
    std::vector<uint32_t> owners{0, 0, 1, 1};
    auto config = configFor(4, 2);
    config.min_foreign_rows_per_transfer = 50;

    PlannerFixture accepted(config.expert_count, config.participant_count);
    ASSERT_TRUE(accepted.plan(loads, owners, config));
    EXPECT_EQ(accepted.status.spilled_rows, 50u);
    EXPECT_EQ(accepted.status.required_foreign_rows, 50u);
    EXPECT_EQ(accepted.status.skipped_insufficient_foreign_rows, 0u);
    EXPECT_EQ(accepted.status.weight_transfer_count, 1u);

    config.min_foreign_rows_per_transfer = 51;
    PlannerFixture rejected(config.expert_count, config.participant_count);
    ASSERT_TRUE(rejected.plan(loads, owners, config));
    EXPECT_EQ(rejected.status.required_foreign_rows, 51u);
    EXPECT_EQ(rejected.status.skipped_insufficient_foreign_rows, 1u);
    EXPECT_EQ(rejected.status.standard_ep_selected, 1u);
    EXPECT_EQ(rejected.status.spilled_rows, 0u);
    EXPECT_EQ(rejected.status.weight_transfer_count, 0u);
}

TEST(Test__LeastLoadedExpertAssignment, RelativeSpreadImprovementGatePricesTotalLoad)
{
    std::vector<uint64_t> loads{70, 30, 30, 30};
    std::vector<uint32_t> owners{0, 0, 1, 1};
    auto config = configFor(4, 2);
    config.min_spread_improvement_divisor = 4;

    PlannerFixture accepted(config.expert_count, config.participant_count);
    ASSERT_TRUE(accepted.plan(loads, owners, config));
    EXPECT_EQ(accepted.status.total_load, 160u);
    EXPECT_EQ(accepted.status.standard_load_spread, 40u);
    EXPECT_EQ(accepted.status.assigned_load_spread_improvement, 40u);
    EXPECT_EQ(accepted.status.required_spread_improvement, 40u);
    EXPECT_EQ(accepted.status.skipped_insufficient_spread_improvement, 0u);
    EXPECT_EQ(accepted.status.weight_transfer_count, 1u);

    config.min_spread_improvement_divisor = 3;
    PlannerFixture rejected(config.expert_count, config.participant_count);
    ASSERT_TRUE(rejected.plan(loads, owners, config));
    EXPECT_EQ(rejected.status.total_load, 160u);
    EXPECT_EQ(rejected.status.required_spread_improvement, 53u);
    EXPECT_EQ(rejected.status.assigned_load_spread_improvement, 0u);
    EXPECT_EQ(rejected.status.skipped_insufficient_spread_improvement, 1u);
    EXPECT_EQ(rejected.status.standard_ep_selected, 1u);
    EXPECT_EQ(rejected.status.weight_transfer_count, 0u);
}

TEST(Test__LeastLoadedExpertAssignment, TransferOnlyPlannerHonorsRelativeSpreadImprovementGate)
{
    std::vector<uint64_t> loads{70, 30, 30, 30};
    std::vector<uint32_t> owners{0, 0, 1, 1};
    auto config = configFor(4, 2);
    config.min_spread_improvement_divisor = 4;

    PlannerFixture accepted(config.expert_count, config.participant_count);
    ASSERT_TRUE(accepted.planTransfersOnly(loads, owners, config));
    EXPECT_EQ(accepted.status.assigned_load_spread_improvement, 40u);
    EXPECT_EQ(accepted.status.required_spread_improvement, 40u);
    EXPECT_EQ(accepted.status.weight_transfer_count, 1u);

    config.min_spread_improvement_divisor = 3;
    PlannerFixture rejected(config.expert_count, config.participant_count);
    ASSERT_TRUE(rejected.planTransfersOnly(loads, owners, config));
    EXPECT_EQ(rejected.status.required_spread_improvement, 53u);
    EXPECT_EQ(rejected.status.assigned_load_spread_improvement, 0u);
    EXPECT_EQ(rejected.status.skipped_insufficient_spread_improvement, 1u);
    EXPECT_EQ(rejected.status.weight_transfer_count, 0u);
}

TEST(Test__LeastLoadedExpertAssignment, TransferOnlyPlannerHonorsForeignRowsGate)
{
    std::vector<uint64_t> loads{80, 20, 0, 0};
    std::vector<uint32_t> owners{0, 0, 1, 1};
    auto config = configFor(4, 2);
    config.min_foreign_rows_per_transfer = 50;

    PlannerFixture accepted(config.expert_count, config.participant_count);
    ASSERT_TRUE(accepted.planTransfersOnly(loads, owners, config));
    EXPECT_EQ(accepted.status.spilled_rows, 50u);
    EXPECT_EQ(accepted.status.required_foreign_rows, 50u);
    EXPECT_EQ(accepted.status.weight_transfer_count, 1u);

    config.min_foreign_rows_per_transfer = 51;
    PlannerFixture rejected(config.expert_count, config.participant_count);
    ASSERT_TRUE(rejected.planTransfersOnly(loads, owners, config));
    EXPECT_EQ(rejected.status.required_foreign_rows, 51u);
    EXPECT_EQ(rejected.status.skipped_insufficient_foreign_rows, 1u);
    EXPECT_EQ(rejected.status.weight_transfer_count, 0u);
}

TEST(Test__LeastLoadedExpertAssignment, FullySpillsExpertWhenNativeIsReservedForPendingLocalWork)
{
    std::vector<uint64_t> loads{30, 30, 60, 0};
    std::vector<uint32_t> owners{0, 0, 0, 1};
    auto config = configFor(4, 2);

    PlannerFixture fixture(config.expert_count, config.participant_count);
    ASSERT_TRUE(fixture.plan(loads, owners, config));

    ASSERT_GE(fixture.status.span_count, 3u);
    EXPECT_EQ(fixture.status.capacity_per_participant, 60u);
    EXPECT_EQ(fixture.spans[0].expert, 2u);
    EXPECT_EQ(fixture.spans[0].owner_participant, 0u);
    EXPECT_EQ(fixture.spans[0].destination_participant, 1u);
    EXPECT_EQ(fixture.spans[0].route_row_begin, 0u);
    EXPECT_EQ(fixture.spans[0].route_row_end, 60u);
    EXPECT_EQ(fixture.spans[0].needs_foreign_weight, 1u);
    EXPECT_EQ(fixture.assigned[0], 60u);
    EXPECT_EQ(fixture.assigned[1], 60u);
}

TEST(Test__LeastLoadedExpertAssignment, MinChunkFloorForcesRemainingRowsWhenAllCapacityIsTooSmall)
{
    std::vector<uint64_t> loads{140, 95, 65};
    std::vector<uint32_t> owners{0, 1, 2};
    auto config = configFor(3, 3);
    config.min_chunk_tokens = 64;

    PlannerFixture fixture(config.expert_count, config.participant_count);
    ASSERT_TRUE(fixture.plan(loads, owners, config));

    ASSERT_GE(fixture.status.span_count, 2u);
    EXPECT_EQ(fixture.status.capacity_per_participant, 100u);
    EXPECT_EQ(fixture.spans[0].expert, 0u);
    EXPECT_EQ(fixture.spans[0].destination_participant, 0u);
    EXPECT_EQ(fixture.spans[0].route_row_end, 100u);
    EXPECT_EQ(fixture.spans[1].expert, 0u);
    EXPECT_EQ(fixture.spans[1].destination_participant, 2u);
    EXPECT_EQ(fixture.spans[1].route_row_begin, 100u);
    EXPECT_EQ(fixture.spans[1].route_row_end, 140u);
    EXPECT_EQ(fixture.spans[1].forced, 1u);
    EXPECT_GT(fixture.status.min_chunk_skips, 0u);
    EXPECT_GT(fixture.status.forced_spills, 0u);
}

TEST(Test__LeastLoadedExpertAssignment, UsesExplicitOwnerMapInsteadOfContiguousExpertIds)
{
    std::vector<uint64_t> loads{60, 0, 0, 60};
    std::vector<uint32_t> owners{1, 0, 1, 0};
    auto config = configFor(4, 2);
    config.enable_balanced_skip = false;

    PlannerFixture fixture(config.expert_count, config.participant_count);
    ASSERT_TRUE(fixture.plan(loads, owners, config));

    ASSERT_EQ(fixture.status.span_count, 2u);
    EXPECT_EQ(fixture.status.spilled_rows, 0u);
    EXPECT_EQ(fixture.status.weight_transfer_count, 0u);

    EXPECT_EQ(fixture.spans[0].expert, 0u);
    EXPECT_EQ(fixture.spans[0].owner_participant, 1u);
    EXPECT_EQ(fixture.spans[0].destination_participant, 1u);
    EXPECT_EQ(fixture.spans[1].expert, 3u);
    EXPECT_EQ(fixture.spans[1].owner_participant, 0u);
    EXPECT_EQ(fixture.spans[1].destination_participant, 0u);
}

TEST(Test__LeastLoadedExpertAssignment, ReportsOverflowWhenSpanCapacityIsTooSmall)
{
    std::vector<uint64_t> loads{80, 20, 0, 0};
    std::vector<uint32_t> owners{0, 0, 1, 1};
    auto config = configFor(4, 2);

    PlannerFixture fixture(config.expert_count, config.participant_count, 1, 1);
    EXPECT_FALSE(fixture.plan(loads, owners, config));
    EXPECT_EQ(fixture.status.overflow, 1u);
    EXPECT_EQ(fixture.status.span_count, 1u);
}

TEST(Test__LeastLoadedExpertAssignment, RejectsInvalidOwnerMap)
{
    std::vector<uint64_t> loads{10, 20};
    std::vector<uint32_t> owners{0, 2};
    auto config = configFor(2, 2);

    PlannerFixture fixture(config.expert_count, config.participant_count);
    EXPECT_FALSE(fixture.plan(loads, owners, config));
    EXPECT_EQ(fixture.status.invalid_config, 1u);
}

TEST(Test__LeastLoadedExpertAssignment, RejectsParticipantCountAboveCurrentScratchLimit)
{
    std::vector<uint64_t> loads{10};
    std::vector<uint32_t> owners{0};
    auto config = configFor(1, 65);

    PlannerFixture fixture(config.expert_count, config.participant_count);
    EXPECT_FALSE(fixture.plan(loads, owners, config));
    EXPECT_EQ(fixture.status.invalid_config, 1u);
}

TEST(Test__LeastLoadedExpertAssignment, ResidentMaskHelpersNormalizeAndSelectLeastLoaded)
{
    EXPECT_EQ(participantMaskLimit(0), 0u);
    EXPECT_EQ(participantMaskLimit(3), 0b111u);

    EXPECT_EQ(normalizeResidentParticipantMask(0b1010u, 0, 2, 3), 0b010u)
        << "masks must be trimmed to participant_count";
    EXPECT_EQ(normalizeResidentParticipantMask(0u, 1, 0, 3), 0b010u)
        << "missing resident masks fall back to the owner when valid";
    EXPECT_EQ(normalizeResidentParticipantMask(0u, -1, 2, 3), 0b100u)
        << "missing owner falls back to the local participant";

    const std::array<uint64_t, 4> participant_loads{9, 2, 7, 2};
    EXPECT_EQ(selectLeastLoadedResidentParticipant(0b1101u, participant_loads.data(), 4, 0), 3u);
    EXPECT_EQ(selectLeastLoadedResidentParticipant(0u, participant_loads.data(), 4, 2), 2u);
}

TEST(Test__LeastLoadedExpertAssignment, ResidentAssignmentSelectsHighestLoadUnassignedExpert)
{
    std::array<int32_t, 5> expert_loads{0, 7, 7, 2, 9};
    std::array<int32_t, 5> assigned{-1, -1, -1, -1, -1};

    EXPECT_EQ(selectHighestLoadUnassignedExpert(expert_loads.data(), assigned.data(), expert_loads.size()), 4);
    assigned[4] = 1;
    EXPECT_EQ(selectHighestLoadUnassignedExpert(expert_loads.data(), assigned.data(), expert_loads.size()), 1)
        << "ties use lower expert id for deterministic CUDA/ROCm behavior";
    assigned[1] = 0;
    assigned[2] = 1;
    assigned[3] = 0;
    EXPECT_EQ(selectHighestLoadUnassignedExpert(expert_loads.data(), assigned.data(), expert_loads.size()), -1);
}

TEST(Test__LeastLoadedExpertAssignment, ResidentSplitCountsMatchPerRowLeastLoadedAssignment)
{
    struct Case
    {
        uint64_t rows;
        uint32_t resident_mask;
        uint32_t fallback;
        std::array<uint64_t, 4> initial_loads;
    };

    const std::array<Case, 4> cases{{
        {11, 0b111u, 0, {0, 0, 0, 0}},
        {10, 0b111u, 0, {5, 0, 5, 0}},
        {6, 0b0101u, 0, {3, 100, 0, 0}},
        {5, 0u, 2, {9, 4, 7, 0}},
    }};

    for (const auto &test_case : cases)
    {
        std::array<uint64_t, 4> row_loads = test_case.initial_loads;
        std::array<uint32_t, 4> row_counts{};
        uint32_t row_mask = test_case.resident_mask & participantMaskLimit(row_loads.size());
        if (row_mask == 0u)
            row_mask = 1u << test_case.fallback;

        for (uint64_t row = 0; row < test_case.rows; ++row)
        {
            const uint32_t participant =
                selectLeastLoadedResidentParticipant(
                    row_mask,
                    row_loads.data(),
                    row_loads.size(),
                    test_case.fallback);
            ++row_loads[participant];
            ++row_counts[participant];
        }

        std::array<uint64_t, 4> chunk_loads = test_case.initial_loads;
        std::array<uint32_t, 4> chunk_counts{};
        assignLeastLoadedResidentSplitCounts(
            test_case.rows,
            test_case.resident_mask,
            chunk_loads.data(),
            chunk_loads.size(),
            test_case.fallback,
            chunk_counts.data());

        EXPECT_EQ(chunk_counts, row_counts);
        EXPECT_EQ(chunk_loads, row_loads);
    }
}
