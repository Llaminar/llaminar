/**
 * @file Test__MoEOverlayDevicePlacementPolicy.cpp
 * @brief Fast adversarial tests for two-axis device overlay placement policy.
 */

#include "execution/moe/DeviceMoERebalancePolicyShared.h"
#include "execution/moe/MoEOverlayDeviceControllerKernels.h"
#include "execution/moe/MoEOverlayDevicePlacementPolicy.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        /** Attach exact synthetic measured costs matching the device ABI. */
        void attachMeasuredEconomy(
            MoEOverlayDevicePlacementPolicyInput *input)
        {
            if (!input)
                throw std::invalid_argument("test economy input is null");
            std::vector<std::int32_t> priorities;
            for (const auto &participant : input->participants)
                priorities.push_back(participant.tier_priority);
            std::sort(priorities.begin(), priorities.end());
            priorities.erase(
                std::unique(priorities.begin(), priorities.end()),
                priorities.end());
            for (auto &participant : input->participants)
            {
                participant.tier_index = static_cast<std::int32_t>(
                    std::lower_bound(
                        priorities.begin(),
                        priorities.end(),
                        participant.tier_priority) -
                    priorities.begin());
            }

            MoEOverlayDevicePlacementEconomyInput economy;
            economy.tier_count = static_cast<std::uint32_t>(priorities.size());
            economy.routed_experts_per_token = 1u;
            economy.transaction_generation = 1u;
            const std::size_t plane_words =
                static_cast<std::size_t>(input->num_layers) *
                input->num_experts;
            economy.phase_expert_demand.assign(
                kMoEOverlayDeviceControllerDemandPhaseCount * plane_words,
                0u);
            for (std::uint32_t layer = 0u;
                 layer < input->num_layers;
                 ++layer)
            {
                for (std::uint32_t expert = 0u;
                     expert < input->num_experts;
                     ++expert)
                {
                    std::uint64_t demand = 0u;
                    for (std::uint32_t participant = 0u;
                         participant < input->participants.size();
                         ++participant)
                    {
                        demand += moe_rebalance_policy::
                            collectedStateActivationCount(
                                input->collected_state[
                                    (static_cast<std::size_t>(participant) *
                                         input->num_layers +
                                     layer) *
                                        input->num_experts +
                                    expert]);
                    }
                    economy.phase_expert_demand[
                        (static_cast<std::size_t>(
                             kMoEOverlayDeviceControllerEconomyPrefillPhase) *
                             input->num_layers +
                         layer) *
                            input->num_experts +
                        expert] = demand;
                }
            }
            economy.service_costs.assign(
                static_cast<std::size_t>(economy.tier_count) *
                    input->num_layers *
                    kMoEOverlayDeviceControllerEconomyServicePhaseCount,
                0u);
            for (std::uint32_t tier = 0u; tier < economy.tier_count; ++tier)
            {
                for (std::uint32_t layer = 0u;
                     layer < input->num_layers;
                     ++layer)
                {
                    for (std::uint32_t phase = 0u;
                         phase <
                             kMoEOverlayDeviceControllerEconomyServicePhaseCount;
                         ++phase)
                    {
                        economy.service_costs[
                            (static_cast<std::size_t>(tier) *
                                 input->num_layers +
                             layer) *
                                kMoEOverlayDeviceControllerEconomyServicePhaseCount +
                            phase] = 10u + tier * 90u;
                    }
                }
            }
            economy.migration_costs.assign(
                input->participants.size() * input->participants.size() *
                    input->num_layers,
                {});
            for (std::uint32_t source = 0u;
                 source < input->participants.size();
                 ++source)
            {
                for (std::uint32_t destination = 0u;
                     destination < input->participants.size();
                     ++destination)
                {
                    if (source == destination)
                        continue;
                    for (std::uint32_t layer = 0u;
                         layer < input->num_layers;
                         ++layer)
                    {
                        economy.migration_costs[
                            (static_cast<std::size_t>(source) *
                                 input->participants.size() +
                             destination) *
                                input->num_layers +
                            layer] = {
                            .transfer_and_repack_ns = 1u,
                            .inference_interference_ns = 0u,
                        };
                    }
                }
            }
            economy.last_moved_generation.assign(
                plane_words,
                kMoEOverlayDeviceControllerNeverMovedGeneration);
            economy.payoff_horizon_tokens = 65'536u;
            input->economy = std::move(economy);
        }

        /** Build one layer whose histogram counts live on the durable owner. */
        MoEOverlayDevicePlacementPolicyInput input(
            std::vector<std::int32_t> priorities,
            const std::vector<std::int32_t> &owners,
            const std::vector<std::uint64_t> &counts,
            std::uint32_t maximum_cycles = 16u)
        {
            if (owners.size() != counts.size())
                throw std::invalid_argument("test owner/count geometry differs");
            MoEOverlayDevicePlacementPolicyInput result;
            result.num_layers = 1u;
            result.num_experts = static_cast<std::uint32_t>(owners.size());
            for (std::uint32_t participant = 0u;
                 participant < priorities.size();
                 ++participant)
            {
                result.participants.push_back({
                    .participant_id = participant,
                    .tier_priority = priorities[participant],
                });
            }
            result.collected_state.assign(
                priorities.size() * owners.size(), 0u);
            for (std::uint32_t expert = 0u;
                 expert < owners.size();
                 ++expert)
            {
                const auto owner = static_cast<std::uint32_t>(owners[expert]);
                result.collected_state[
                    static_cast<std::size_t>(owner) * owners.size() + expert] =
                    moe_rebalance_policy::packCollectedState(
                        counts[expert],
                        /*active_transfer_slots=*/0u,
                        /*physically_resident=*/true,
                        /*transfer_backed=*/false,
                        /*authoritative_owner=*/true);
            }
            result.payload_bytes_per_layer = {4096u};
            result.base_epoch = 7u;
            result.minimum_window_activations = 1u;
            result.maximum_cycles_per_wave = maximum_cycles;
            result.command_capacity =
                static_cast<std::uint32_t>(owners.size());
            attachMeasuredEconomy(&result);
            return result;
        }

        /** Count candidate owners for one participant after the wave. */
        std::size_t ownerCount(
            const MoEOverlayDevicePlacementPolicyPlan &plan,
            std::int32_t participant)
        {
            return static_cast<std::size_t>(std::count(
                plan.candidate_owner.begin(),
                plan.candidate_owner.end(),
                participant));
        }

        /** Assert the fixed logical movement ABI for every emitted command. */
        void expectCanonicalCommands(
            const MoEOverlayDevicePlacementPolicyPlan &plan)
        {
            for (std::uint32_t index = 0u;
                 index < plan.commands.size();
                 ++index)
            {
                const auto &command = plan.commands[index];
                EXPECT_EQ(command.magic,
                          kMoEOverlayDeviceMovementCommandMagic);
                EXPECT_EQ(command.version,
                          kMoEOverlayDeviceMovementCommandVersion);
                EXPECT_EQ(command.ordinal, index);
                EXPECT_EQ(command.payload_slot, index);
                EXPECT_EQ(command.payload_bytes, 4096u);
                EXPECT_EQ(command.source_epoch, 7u);
                EXPECT_EQ(command.candidate_epoch, 8u);
                EXPECT_NE(command.source_participant,
                          command.destination_participant);
                const auto axis = static_cast<MoEOverlayDeviceMovementAxis>(
                    command.flags);
                EXPECT_TRUE(
                    axis == MoEOverlayDeviceMovementAxis::TierResidency ||
                    axis ==
                        MoEOverlayDeviceMovementAxis::ParticipantPlacement ||
                    axis == MoEOverlayDeviceMovementAxis::Combined)
                    << "device policy must author every movement objective";
                if (index == 0u)
                    continue;
                const auto &previous = plan.commands[index - 1u];
                EXPECT_TRUE(
                    previous.destination_participant <
                        command.destination_participant ||
                    (previous.destination_participant ==
                         command.destination_participant &&
                     (previous.layer < command.layer ||
                      (previous.layer == command.layer &&
                       previous.expert < command.expert))));
            }
        }
    } // namespace

    TEST(Test__MoEOverlayDevicePlacementPolicy,
         AdversarialTwoPriorityLayoutPromotesHotExpertsAndDemotesColdExperts)
    {
        const auto policy_input = input(
            /*priorities=*/{0, 0, 17, 17},
            /*owners=*/{2, 2, 3, 3, 0, 0, 1, 1},
            /*counts=*/{800, 700, 600, 500, 40, 30, 20, 10});

        const auto plan =
            MoEOverlayDevicePlacementPolicyReference::planDynamic(
                policy_input);
        ASSERT_TRUE(plan.hasMovement());
        EXPECT_TRUE(plan.evidence.improves());
        EXPECT_EQ(plan.evidence.promotions, 4u);
        EXPECT_EQ(plan.evidence.demotions, 4u);
        EXPECT_EQ(plan.evidence.same_priority_moves, 0u);
        EXPECT_LT(plan.evidence.priority_cost_after,
                  plan.evidence.priority_cost_before);
        for (std::int32_t participant = 0; participant < 4; ++participant)
            EXPECT_EQ(ownerCount(plan, participant), 2u);
        expectCanonicalCommands(plan);
    }

    /**
     * @brief A bounded device wave must not starve one independent Dynamic axis.
     *
     * This reproduces the CUDA2/ROCm4 depth-one production failure.  The
     * topology exposes profitable tier exchanges and a profitable same-tier
     * correction, but pure net-benefit ordering spends both available cycle
     * slots on tier residency.  The device authority must reserve the second
     * slot for participant placement while retaining every ordinary economy,
     * capacity, and hysteresis gate.
     */
    TEST(Test__MoEOverlayDevicePlacementPolicy,
         TwoCycleWaveAdvancesTierAndParticipantObjectives)
    {
        auto policy_input = input(
            /*priorities=*/{0, 0, 17, 17, 17, 17},
            /*owners=*/{
                2, 3, 1, 0, 0, 1,
                5, 4, 5, 4, 3, 2,
            },
            /*counts=*/{
                1200, 1100, 1000, 900,
                850, 825, 800, 700,
                600, 500, 100, 50,
            },
            /*maximum_cycles=*/2u);
        policy_input.dynamic_maximum_cycles_per_layer = 2u;
        policy_input.dynamic_imbalance_threshold_per_mille = 1000u;
        policy_input.dynamic_minimum_improvement_per_mille = 0u;

        const auto plan =
            MoEOverlayDevicePlacementPolicyReference::planDynamic(
                policy_input);
        ASSERT_TRUE(plan.hasMovement());
        ASSERT_EQ(plan.evidence.accepted_cycles, 2u);

        bool advances_tier = false;
        bool advances_participant = false;
        for (const auto &command : plan.commands)
        {
            const auto axis = static_cast<MoEOverlayDeviceMovementAxis>(
                command.flags);
            advances_tier = advances_tier ||
                axis == MoEOverlayDeviceMovementAxis::TierResidency ||
                axis == MoEOverlayDeviceMovementAxis::Combined;
            advances_participant = advances_participant ||
                axis == MoEOverlayDeviceMovementAxis::ParticipantPlacement ||
                axis == MoEOverlayDeviceMovementAxis::Combined;
        }
        EXPECT_TRUE(advances_tier);
        EXPECT_TRUE(advances_participant)
            << "bounded net-benefit ordering starved within-tier skew correction";
    }

    /**
     * @brief Axis reservation searches later layers before spending its slot.
     *
     * Layer zero exposes two profitable tier exchanges but no participant
     * correction. Layer one is already tier-optimal and exposes one profitable
     * same-tier correction. A two-cycle wave must not consume both slots in
     * layer zero merely because the participant objective lives later in the
     * device-owned layer cursor.
     */
    TEST(Test__MoEOverlayDevicePlacementPolicy,
         TwoCycleWaveSearchesLaterLayersBeforeEconomyFallback)
    {
        constexpr std::array<std::int32_t, 12> layer_zero_owners = {
            2, 3, 1, 0, 0, 1,
            4, 5, 5, 4, 3, 2,
        };
        constexpr std::array<std::int32_t, 12> layer_one_owners = {
            0, 1, 1, 0, 2, 3,
            5, 4, 5, 4, 3, 2,
        };
        constexpr std::array<std::uint64_t, 12> counts = {
            1200, 1100, 1000, 900,
            850, 825, 800, 700,
            600, 500, 100, 50,
        };
        auto policy_input = input(
            /*priorities=*/{0, 0, 17, 17, 17, 17},
            std::vector<std::int32_t>(
                layer_zero_owners.begin(), layer_zero_owners.end()),
            std::vector<std::uint64_t>(counts.begin(), counts.end()),
            /*maximum_cycles=*/2u);
        policy_input.num_layers = 2u;
        policy_input.collected_state.assign(
            policy_input.participants.size() * policy_input.num_layers *
                policy_input.num_experts,
            0u);
        for (std::uint32_t layer = 0u;
             layer < policy_input.num_layers;
             ++layer)
        {
            const auto &owners = layer == 0u
                                     ? layer_zero_owners
                                     : layer_one_owners;
            for (std::uint32_t expert = 0u;
                 expert < policy_input.num_experts;
                 ++expert)
            {
                const auto owner = static_cast<std::uint32_t>(owners[expert]);
                policy_input.collected_state[
                    (static_cast<std::size_t>(owner) *
                         policy_input.num_layers +
                     layer) *
                        policy_input.num_experts +
                    expert] = moe_rebalance_policy::packCollectedState(
                    counts[expert],
                    /*active_transfer_slots=*/0u,
                    /*physically_resident=*/true,
                    /*transfer_backed=*/false,
                    /*authoritative_owner=*/true);
            }
        }
        policy_input.payload_bytes_per_layer = {4096u, 4096u};
        policy_input.dynamic_maximum_cycles_per_layer = 2u;
        policy_input.dynamic_imbalance_threshold_per_mille = 1000u;
        policy_input.dynamic_minimum_improvement_per_mille = 0u;
        attachMeasuredEconomy(&policy_input);

        const auto plan =
            MoEOverlayDevicePlacementPolicyReference::planDynamic(
                policy_input);
        ASSERT_EQ(plan.evidence.accepted_cycles, 2u);
        EXPECT_EQ(plan.evidence.changed_layers, 2u);

        bool advances_tier = false;
        bool advances_participant = false;
        for (const auto &command : plan.commands)
        {
            const auto axis = static_cast<MoEOverlayDeviceMovementAxis>(
                command.flags);
            advances_tier = advances_tier ||
                axis == MoEOverlayDeviceMovementAxis::TierResidency ||
                axis == MoEOverlayDeviceMovementAxis::Combined;
            advances_participant = advances_participant ||
                axis == MoEOverlayDeviceMovementAxis::ParticipantPlacement ||
                axis == MoEOverlayDeviceMovementAxis::Combined;
        }
        EXPECT_TRUE(advances_tier);
        EXPECT_TRUE(advances_participant);
    }

    TEST(Test__MoEOverlayDevicePlacementPolicy,
         OnePriorityLayoutRebalancesSkewWithoutTierMovement)
    {
        const auto policy_input = input(
            /*priorities=*/{4, 4, 4, 4},
            /*owners=*/{0, 0, 1, 1, 2, 2, 3, 3},
            /*counts=*/{1000, 900, 800, 700, 10, 9, 8, 7});

        const auto plan =
            MoEOverlayDevicePlacementPolicyReference::planDynamic(
                policy_input);
        ASSERT_TRUE(plan.hasMovement());
        EXPECT_TRUE(plan.evidence.improves());
        EXPECT_EQ(plan.evidence.promotions, 0u);
        EXPECT_EQ(plan.evidence.demotions, 0u);
        EXPECT_GT(plan.evidence.same_priority_moves, 0u);
        EXPECT_EQ(plan.evidence.priority_cost_after,
                  plan.evidence.priority_cost_before);
        EXPECT_LT(plan.evidence.same_priority_makespan_after,
                  plan.evidence.same_priority_makespan_before);
        EXPECT_TRUE(std::all_of(
            plan.commands.begin(),
            plan.commands.end(),
            [](const auto &command)
            {
                return static_cast<MoEOverlayDeviceMovementAxis>(
                           command.flags) ==
                       MoEOverlayDeviceMovementAxis::ParticipantPlacement;
            }));
        for (std::int32_t participant = 0; participant < 4; ++participant)
            EXPECT_EQ(ownerCount(plan, participant), 2u);
        expectCanonicalCommands(plan);
    }

    TEST(Test__MoEOverlayDevicePlacementPolicy,
         ArbitraryThreePriorityTopologyPublishesOneClosedParallelCycle)
    {
        const auto policy_input = input(
            /*priorities=*/{-9, 2, 41},
            /*owners=*/{1, 2, 0},
            /*counts=*/{300, 200, 100},
            /*maximum_cycles=*/1u);

        const auto plan =
            MoEOverlayDevicePlacementPolicyReference::planDynamic(
                policy_input);
        ASSERT_TRUE(plan.hasMovement());
        ASSERT_EQ(plan.commands.size(), 3u);
        EXPECT_EQ(plan.evidence.accepted_cycles, 1u);
        EXPECT_EQ(plan.evidence.promotions, 2u);
        EXPECT_EQ(plan.evidence.demotions, 1u);
        EXPECT_EQ(plan.evidence.same_priority_moves, 0u);
        for (std::int32_t participant = 0; participant < 3; ++participant)
            EXPECT_EQ(ownerCount(plan, participant), 1u);
        expectCanonicalCommands(plan);
    }

    TEST(Test__MoEOverlayDevicePlacementPolicy,
         TunableCycleBoundPublishesOnlyCompleteCapacityPreservingCycles)
    {
        const auto policy_input = input(
            /*priorities=*/{0, 0, 17, 17},
            /*owners=*/{2, 2, 3, 3, 0, 0, 1, 1},
            /*counts=*/{800, 700, 600, 500, 40, 30, 20, 10},
            /*maximum_cycles=*/1u);

        const auto plan =
            MoEOverlayDevicePlacementPolicyReference::planDynamic(
                policy_input);
        ASSERT_TRUE(plan.hasMovement());
        EXPECT_EQ(plan.evidence.accepted_cycles, 1u);
        ASSERT_EQ(plan.commands.size(), 2u);
        for (std::int32_t participant = 0; participant < 4; ++participant)
            EXPECT_EQ(ownerCount(plan, participant), 2u);
    }

    TEST(Test__MoEOverlayDevicePlacementPolicy,
         AlreadyOptimalLayoutProducesStaticEquivalentZeroMovement)
    {
        const auto policy_input = input(
            /*priorities=*/{0, 0, 17, 17},
            /*owners=*/{0, 1, 1, 0, 2, 3, 3, 2},
            /*counts=*/{800, 700, 600, 500, 40, 30, 20, 10});

        const auto plan =
            MoEOverlayDevicePlacementPolicyReference::planDynamic(
                policy_input);
        EXPECT_FALSE(plan.hasMovement());
        EXPECT_TRUE(plan.commands.empty());
        EXPECT_EQ(plan.evidence.promotions, 0u);
        EXPECT_EQ(plan.evidence.demotions, 0u);
        EXPECT_EQ(plan.evidence.same_priority_moves, 0u);
    }

    TEST(Test__MoEOverlayDevicePlacementPolicy,
         SamePriorityMovementRequiresConfiguredSkewAndImprovement)
    {
        auto policy_input = input(
            /*priorities=*/{5, 5},
            /*owners=*/{0, 0, 1, 1},
            /*counts=*/{110, 100, 90, 80});
        policy_input.dynamic_minimum_improvement_per_mille = 0u;
        policy_input.dynamic_imbalance_threshold_per_mille = 1300u;

        const auto below_skew_floor =
            MoEOverlayDevicePlacementPolicyReference::planDynamic(
                policy_input);
        EXPECT_FALSE(below_skew_floor.hasMovement());
        EXPECT_GT(below_skew_floor.evidence.rejected_cycles, 0u);

        policy_input.dynamic_imbalance_threshold_per_mille = 1200u;
        policy_input.dynamic_minimum_improvement_per_mille = 100u;
        const auto below_improvement_floor =
            MoEOverlayDevicePlacementPolicyReference::planDynamic(
                policy_input);
        EXPECT_FALSE(below_improvement_floor.hasMovement());

        policy_input.dynamic_minimum_improvement_per_mille = 90u;
        const auto economical =
            MoEOverlayDevicePlacementPolicyReference::planDynamic(
                policy_input);
        EXPECT_TRUE(economical.hasMovement());
        EXPECT_GT(economical.evidence.same_priority_moves, 0u);
    }

    TEST(Test__MoEOverlayDevicePlacementPolicy,
         CommandBudgetNeverPublishesAPartialCapacityCycle)
    {
        auto policy_input = input(
            /*priorities=*/{0, 0, 17, 17},
            /*owners=*/{2, 2, 3, 3, 0, 0, 1, 1},
            /*counts=*/{800, 700, 600, 500, 40, 30, 20, 10});
        policy_input.dynamic_maximum_commands_per_wave = 1u;

        const auto plan =
            MoEOverlayDevicePlacementPolicyReference::planDynamic(
                policy_input);
        EXPECT_FALSE(plan.hasMovement());
        EXPECT_TRUE(plan.commands.empty());
        EXPECT_EQ(plan.evidence.accepted_cycles, 0u);
    }

    TEST(Test__MoEOverlayDevicePlacementPolicy,
         MeasuredTransferAndInterferenceCostCanRejectAProxyImprovement)
    {
        auto policy_input = input(
            /*priorities=*/{0, 17},
            /*owners=*/{1, 0},
            /*counts=*/{1000, 10});
        ASSERT_TRUE(policy_input.economy.has_value());
        for (auto &cost : policy_input.economy->migration_costs)
        {
            cost.transfer_and_repack_ns = 1'000'000'000'000u;
            cost.inference_interference_ns = 1'000'000'000'000u;
        }

        const auto plan =
            MoEOverlayDevicePlacementPolicyReference::planDynamic(
                policy_input);
        EXPECT_FALSE(plan.hasMovement());
        EXPECT_EQ(plan.evidence.accepted_cycles, 0u);
        EXPECT_GT(plan.evidence.payoff_rejected_cycles, 0u);
        EXPECT_EQ(plan.evidence.residency_rejected_cycles, 0u);
        EXPECT_EQ(plan.evidence.projected_net_benefit_ns, 0u);
    }

    TEST(Test__MoEOverlayDevicePlacementPolicy,
         DecodeGainCannotCrossSubsidizeAPrefillRegression)
    {
        auto policy_input = input(
            /*priorities=*/{0, 17},
            /*owners=*/{1, 0},
            /*counts=*/{1000, 10});
        ASSERT_TRUE(policy_input.economy.has_value());
        auto &demand = policy_input.economy->phase_expert_demand;
        const auto phase_offset = [&](std::uint32_t phase)
        {
            return static_cast<std::size_t>(phase) *
                   policy_input.num_layers * policy_input.num_experts;
        };

        /*
         * The desired swap produces a large decode win: expert zero leaves the
         * slow tier. Prefill deliberately routes more work to expert one, so
         * the same swap makes prefill slower. The old scalar sum admitted this
         * because the decode gain was larger than the prefill loss.
         */
        demand[phase_offset(
                   kMoEOverlayDeviceControllerEconomyDecodePhase) +
               0u] = 1000u;
        demand[phase_offset(
                   kMoEOverlayDeviceControllerEconomyDecodePhase) +
               1u] = 10u;
        demand[phase_offset(
                   kMoEOverlayDeviceControllerEconomyPrefillPhase) +
               0u] = 10u;
        demand[phase_offset(
                   kMoEOverlayDeviceControllerEconomyPrefillPhase) +
               1u] = 20u;

        const auto rejected =
            MoEOverlayDevicePlacementPolicyReference::planDynamic(
                policy_input);
        EXPECT_FALSE(rejected.hasMovement());
        EXPECT_EQ(rejected.evidence.accepted_cycles, 0u);
        EXPECT_GT(rejected.evidence.payoff_rejected_cycles, 0u);

        /* Both phases favor the same move, which must remain economical. */
        demand[phase_offset(
                   kMoEOverlayDeviceControllerEconomyPrefillPhase) +
               0u] = 100u;
        demand[phase_offset(
                   kMoEOverlayDeviceControllerEconomyPrefillPhase) +
               1u] = 1u;
        const auto accepted =
            MoEOverlayDevicePlacementPolicyReference::planDynamic(
                policy_input);
        EXPECT_TRUE(accepted.hasMovement());
        EXPECT_GT(accepted.evidence.accepted_cycles, 0u);
        EXPECT_EQ(accepted.evidence.payoff_rejected_cycles, 0u);
    }

    TEST(Test__MoEOverlayDevicePlacementPolicy,
         CommittedMovementHysteresisRejectsThenAgesIntoEligibility)
    {
        auto policy_input = input(
            /*priorities=*/{0, 17},
            /*owners=*/{1, 0},
            /*counts=*/{1000, 10});
        ASSERT_TRUE(policy_input.economy.has_value());
        policy_input.economy->minimum_residency_generations = 2u;
        std::fill(
            policy_input.economy->last_moved_generation.begin(),
            policy_input.economy->last_moved_generation.end(),
            1u);
        policy_input.economy->transaction_generation = 1u;

        const auto immediate =
            MoEOverlayDevicePlacementPolicyReference::planDynamic(
                policy_input);
        EXPECT_FALSE(immediate.hasMovement());
        EXPECT_GT(immediate.evidence.residency_rejected_cycles, 0u);

        policy_input.economy->transaction_generation = 3u;
        const auto aged =
            MoEOverlayDevicePlacementPolicyReference::planDynamic(
                policy_input);
        ASSERT_TRUE(aged.hasMovement());
        EXPECT_TRUE(aged.evidence.improves());
        EXPECT_EQ(aged.evidence.residency_rejected_cycles, 0u);
        EXPECT_GT(aged.evidence.projected_service_gain_ns, 0u);
        EXPECT_GT(aged.evidence.projected_net_benefit_ns, 0u);
    }

    TEST(Test__MoEOverlayDevicePlacementPolicy,
         BoundedWavesAdvanceADeviceOwnedCursorAcrossEveryLayer)
    {
        auto policy_input = input(
            /*priorities=*/{0, 0, 17, 17},
            /*owners=*/{2, 2, 3, 3, 0, 0, 1, 1},
            /*counts=*/{800, 700, 600, 500, 40, 30, 20, 10},
            /*maximum_cycles=*/1u);
        constexpr std::uint32_t kLayers = 4u;
        const auto one_layer_state = policy_input.collected_state;
        const std::size_t experts = policy_input.num_experts;
        policy_input.num_layers = kLayers;
        policy_input.payload_bytes_per_layer.assign(kLayers, 4096u);
        policy_input.collected_state.assign(
            policy_input.participants.size() * kLayers * experts, 0u);
        for (std::size_t participant = 0u;
             participant < policy_input.participants.size();
             ++participant)
        {
            for (std::uint32_t layer = 0u; layer < kLayers; ++layer)
            {
                for (std::size_t expert = 0u; expert < experts; ++expert)
                {
                    policy_input.collected_state[
                        (participant * kLayers + layer) * experts + expert] =
                        one_layer_state[participant * experts + expert];
                }
            }
        }
        attachMeasuredEconomy(&policy_input);

        for (std::uint32_t expected_layer : {2u, 3u, 0u, 1u})
        {
            policy_input.layer_start_cursor = expected_layer;
            const auto plan =
                MoEOverlayDevicePlacementPolicyReference::planDynamic(
                    policy_input);
            ASSERT_TRUE(plan.hasMovement());
            ASSERT_FALSE(plan.commands.empty());
            EXPECT_TRUE(std::all_of(
                plan.commands.begin(),
                plan.commands.end(),
                [expected_layer](const auto &command)
                {
                    return command.layer == expected_layer;
                }));
            EXPECT_EQ(plan.evidence.layer_scan_start, expected_layer);
            EXPECT_EQ(
                plan.evidence.layer_scan_next,
                (expected_layer + 1u) % kLayers);
        }
    }

    TEST(Test__MoEOverlayDevicePlacementPolicy,
         DuplicateOrMissingDurableOwnerIsRejected)
    {
        auto policy_input = input(
            /*priorities=*/{0, 1},
            /*owners=*/{0, 1},
            /*counts=*/{100, 10});
        policy_input.collected_state[2u] =
            moe_rebalance_policy::packCollectedState(
                0u, 0u, true, false, true);

        EXPECT_THROW(
            (void)MoEOverlayDevicePlacementPolicyReference::planDynamic(
                policy_input),
            std::invalid_argument);
    }

    TEST(Test__MoEOverlayDevicePlacementPolicy,
         PhaseHistoryCombinesPrefillAndDecodeWithoutOverwriteOrWrap)
    {
        constexpr std::uint32_t kLayers = 2u;
        constexpr std::uint32_t kExperts = 3u;
        constexpr std::uint64_t kPlaneWords = kLayers * kExperts;
        std::vector<std::uint64_t> history(
            kMoEOverlayDeviceControllerDemandPhaseCount * kPlaneWords, 0u);

        const auto prefill_phase = moeOverlayDeviceDemandPhaseIndex(
            MoEOverlayDeviceDemandPhase::Prefill);
        const auto decode_phase = moeOverlayDeviceDemandPhaseIndex(
            MoEOverlayDeviceDemandPhase::Decode);
        ASSERT_EQ(prefill_phase, 0u);
        ASSERT_EQ(decode_phase, 1u);
        EXPECT_EQ(
            moeOverlayDeviceDemandPhaseIndex(
                MoEOverlayDeviceDemandPhase::Invalid),
            kMoEOverlayDeviceControllerDemandPhaseCount);

        const auto update = [&](MoEOverlayDeviceDemandPhase phase,
                                std::uint32_t layer,
                                std::uint32_t expert,
                                std::uint64_t delta)
        {
            const auto selected = moeOverlayDeviceDemandPhaseIndex(phase);
            EXPECT_LT(
                selected, kMoEOverlayDeviceControllerDemandPhaseCount);
            const auto other = 1u - selected;
            const auto selected_offset =
                moeOverlayDeviceDemandHistoryOffset(
                    selected, layer, expert, kLayers, kExperts);
            const auto other_offset = moeOverlayDeviceDemandHistoryOffset(
                other, layer, expert, kLayers, kExperts);
            const auto result = moeOverlayAccumulateDeviceDemandHistory(
                history[selected_offset], history[other_offset], delta);
            history[selected_offset] = result.updated_phase_count;
            return result;
        };

        const auto prefill = update(
            MoEOverlayDeviceDemandPhase::Prefill, 1u, 2u, 700u);
        EXPECT_EQ(prefill.updated_phase_count, 700u);
        EXPECT_EQ(prefill.combined_count, 700u);
        EXPECT_EQ(history[5u], 700u);
        EXPECT_EQ(history[kPlaneWords + 5u], 0u);

        const auto decode = update(
            MoEOverlayDeviceDemandPhase::Decode, 1u, 2u, 11u);
        EXPECT_EQ(decode.updated_phase_count, 11u);
        EXPECT_EQ(decode.combined_count, 711u);
        EXPECT_EQ(history[5u], 700u)
            << "decode must retain the accumulated prefill plane";
        EXPECT_EQ(history[kPlaneWords + 5u], 11u);

        const auto next_prefill = update(
            MoEOverlayDeviceDemandPhase::Prefill, 1u, 2u, 13u);
        EXPECT_EQ(next_prefill.updated_phase_count, 713u);
        EXPECT_EQ(next_prefill.combined_count, 724u);
        EXPECT_EQ(history[kPlaneWords + 5u], 11u)
            << "prefill must retain the accumulated decode plane";

        constexpr std::uint64_t kMaximum = ~std::uint64_t{0};
        const auto saturated = moeOverlayAccumulateDeviceDemandHistory(
            kMaximum - 2u, 9u, 3u);
        EXPECT_EQ(saturated.updated_phase_count, kMaximum);
        EXPECT_EQ(saturated.combined_count, kMaximum);
    }
} // namespace llaminar2::test
