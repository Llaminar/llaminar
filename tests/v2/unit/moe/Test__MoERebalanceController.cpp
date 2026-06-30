/**
 * @file Test__MoERebalanceController.cpp
 * @brief Unit tests for MoERebalanceController
 */

#include <gtest/gtest.h>
#include "execution/moe/DeviceMoERebalanceController.h"
#include "execution/moe/DeviceMoERebalancePolicyShared.h"
#include "execution/moe/MoERebalanceController.h"
#include <algorithm>
#include <array>
#include <iterator>
#include <utility>
#include <vector>

using namespace llaminar2;

// ── Helpers ───────────────────────────────────────────

static MoERebalanceController::Config makeConfig(
    MoERebalanceMode mode,
    int num_experts = 8,
    int num_sockets = 2,
    int num_layers = 2,
    int top_k = 2,
    int window_size = 16)
{
    MoERebalanceController::Config cfg;
    cfg.mode = mode;
    cfg.num_layers = num_layers;
    cfg.num_experts = num_experts;
    cfg.top_k = top_k;
    cfg.window_size = window_size;

    for (int s = 0; s < num_sockets; ++s)
        cfg.sockets.push_back(DeviceId(DeviceType::CPU, s));

    // Round-robin expert-to-socket
    cfg.initial_expert_to_socket.resize(num_experts);
    for (int e = 0; e < num_experts; ++e)
        cfg.initial_expert_to_socket[e] = e % num_sockets;

    cfg.rebalance_config.imbalance_threshold = 1.3f;
    cfg.rebalance_config.max_swaps_per_layer = 4;
    cfg.rebalance_config.max_total_swaps = 16;
    cfg.rebalance_config.min_improvement_ratio = 0.05f;
    cfg.rebalance_config.layer_cooldown_generations = 0; // no cooldown for tests
    cfg.rebalance_config.min_window_activations = 1;

    return cfg;
}

/// Fill the histogram window with balanced routing across all experts
static void fillWindowBalanced(DecodeExpertHistogram &hist, int window_size,
                               int num_layers, int num_experts, int top_k)
{
    for (int t = 0; t < window_size; ++t)
    {
        for (int l = 0; l < num_layers; ++l)
        {
            std::vector<int> indices(top_k);
            std::vector<float> weights(top_k);
            for (int k = 0; k < top_k; ++k)
            {
                indices[k] = (t * top_k + k) % num_experts;
                weights[k] = 1.0f / static_cast<float>(top_k);
            }
            hist.record(l, indices.data(), weights.data(), top_k);
        }
    }
}

/// Fill the histogram window with heavily skewed routing (all to experts 0,1)
static void fillWindowSkewed(DecodeExpertHistogram &hist, int window_size,
                             int num_layers, int top_k)
{
    for (int t = 0; t < window_size; ++t)
    {
        for (int l = 0; l < num_layers; ++l)
        {
            // Always route to experts 0 and 1 (both on socket 0 in round-robin)
            std::vector<int> indices(top_k);
            std::vector<float> weights(top_k);
            for (int k = 0; k < top_k; ++k)
            {
                indices[k] = k; // experts 0, 1
                weights[k] = 1.0f / static_cast<float>(top_k);
            }
            hist.record(l, indices.data(), weights.data(), top_k);
        }
    }
}

static void recordExpertHits(DecodeExpertHistogram &hist, int layer,
                             const std::vector<std::pair<int, int>> &expert_counts);

static DeviceMoELayerRuntime makeDeviceRuntimeLayerForLoadStats()
{
    DeviceMoELayerRuntime runtime{};
    runtime.active_bank = 0;
    runtime.active_epoch = 1;
    runtime.expert_count = 4;
    runtime.top_k = 2;
    runtime.participant_id = 0;
    runtime.participant_count = 2;

    auto &bank = runtime.banks[0];
    bank.epoch = runtime.active_epoch;
    bank.expert_count = runtime.expert_count;
    for (uint32_t expert = 0; expert < runtime.expert_count; ++expert)
    {
        const uint32_t owner = expert % runtime.participant_count;
        auto &desc = bank.experts[expert];
        desc.logical_expert_id = static_cast<int32_t>(expert);
        desc.owner_participant = static_cast<int32_t>(owner);
        desc.local_slot = static_cast<int32_t>(expert);
        desc.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Valid) |
                     toMoEExpertFlags(DeviceMoEExpertFlags::Resident);
        bank.resident_participant_mask[expert] = 1u << owner;
        bank.local_compute_mask[expert] = owner == runtime.participant_id ? 1u : 0u;
        bank.replica_role[expert] = owner == runtime.participant_id
                                        ? static_cast<uint8_t>(DeviceMoEReplicaRole::Primary)
                                        : static_cast<uint8_t>(DeviceMoEReplicaRole::None);
    }
    return runtime;
}

// ── Tests ─────────────────────────────────────────────

TEST(Test__MoERebalanceController, DeviceSideProjectedLoadSpreadTracksReplicaBenefit)
{
    uint64_t single_owner_load[3] = {};
    uint64_t replicated_load[3] = {};

    for (uint32_t participant = 0; participant < 3; ++participant)
    {
        single_owner_load[participant] =
            moe_rebalance_policy::projectedParticipantLoadForExpert(
                12,
                moe_rebalance_policy::participantBit(0),
                3,
                participant);
        replicated_load[participant] =
            moe_rebalance_policy::projectedParticipantLoadForExpert(
                12,
                moe_rebalance_policy::participantBit(0) |
                    moe_rebalance_policy::participantBit(1),
                3,
                participant);
    }

    uint64_t single_total = 0;
    uint64_t single_min = 0;
    uint64_t single_max = 0;
    uint64_t replicated_total = 0;
    uint64_t replicated_min = 0;
    uint64_t replicated_max = 0;
    moe_rebalance_policy::finalizeLoadSpread(
        single_owner_load, 3, single_total, single_min, single_max);
    moe_rebalance_policy::finalizeLoadSpread(
        replicated_load, 3, replicated_total, replicated_min, replicated_max);

    EXPECT_EQ(single_total, 12u);
    EXPECT_EQ(replicated_total, 12u);
    EXPECT_EQ(single_max - single_min, 12u);
    EXPECT_EQ(replicated_max - replicated_min, 6u);
    EXPECT_LT(replicated_max - replicated_min, single_max - single_min);
}

TEST(Test__MoERebalanceController, DeviceSideLeastLoadedReplicaPaybackUsesDestinationLoad)
{
    uint64_t current_load[3] = {90, 10, 10};
    const uint32_t owner_mask = moe_rebalance_policy::participantBit(0);
    const uint32_t add_participant_one =
        owner_mask | moe_rebalance_policy::participantBit(1);

    const auto uniform_delta =
        moe_rebalance_policy::evaluateAddingResidentLoadSpread(
            current_load,
            /*expert_count=*/30,
            owner_mask,
            add_participant_one,
            /*participant_count=*/3,
            /*min_improvement=*/20,
            /*min_improvement_divisor=*/0);
    const auto llep_delta =
        moe_rebalance_policy::evaluateAddingResidentLeastLoadedSpread(
            current_load,
            /*expert_count=*/30,
            owner_mask,
            add_participant_one,
            /*participant_count=*/3,
            /*min_improvement=*/20,
            /*min_improvement_divisor=*/0);

    EXPECT_EQ(uniform_delta.improvement, 15u);
    EXPECT_FALSE(uniform_delta.meets_floor)
        << "Uniform resident splitting underestimates what the least-loaded router can do.";
    EXPECT_EQ(llep_delta.improvement, 30u);
    EXPECT_TRUE(llep_delta.meets_floor);

    uint64_t scratch[3] = {};
    ASSERT_TRUE(moe_rebalance_policy::addingResidentImprovesLeastLoadedSpread(
        current_load,
        /*expert_count=*/30,
        owner_mask,
        add_participant_one,
        /*participant_count=*/3,
        scratch,
        /*min_improvement=*/20,
        /*min_improvement_divisor=*/0));
    EXPECT_EQ(scratch[0], 60u);
    EXPECT_EQ(scratch[1], 40u);
    EXPECT_EQ(scratch[2], 10u);
}

TEST(Test__MoERebalanceController, DeviceSideLeastLoadedDestinationRejectsEmptyExperts)
{
    uint64_t current_load[3] = {90, 10, 10};
    const uint32_t owner_mask = moe_rebalance_policy::participantBit(0);

    const auto empty_choice =
        moe_rebalance_policy::bestLeastLoadedMissingResidentDestination(
            current_load,
            /*expert_count=*/0,
            owner_mask,
            /*participant_count=*/3);
    EXPECT_FALSE(empty_choice.valid);

    const auto choice =
        moe_rebalance_policy::bestLeastLoadedMissingResidentDestination(
            current_load,
            /*expert_count=*/30,
            owner_mask,
            /*participant_count=*/3,
            nullptr,
            0,
            /*min_improvement=*/20,
            /*min_improvement_divisor=*/0);
    ASSERT_TRUE(choice.valid);
    EXPECT_EQ(choice.destination_participant, 1u);
    EXPECT_EQ(choice.delta.improvement, 30u);
}

TEST(Test__MoERebalanceController, SharedDynamicPolicyUsesWindowFloorAndLeastLoadedTarget)
{
    uint64_t current_load[3] = {90, 10, 30};
    const uint32_t owner_mask = moe_rebalance_policy::participantBit(0);

    const auto tiny =
        moe_rebalance_policy::bestDynamicMissingResidentDestination(
            current_load,
            /*expert_count=*/8,
            owner_mask,
            /*participant_count=*/3,
            /*preferred_source_participant=*/0,
            nullptr,
            0,
            /*window_size_tokens=*/256);
    EXPECT_FALSE(tiny.valid)
        << "Dynamic keeps the shared max(2, window/16) admission floor.";

    const auto choice =
        moe_rebalance_policy::bestDynamicMissingResidentDestination(
            current_load,
            /*expert_count=*/30,
            owner_mask,
            /*participant_count=*/3,
            /*preferred_source_participant=*/0,
            nullptr,
            0,
            /*window_size_tokens=*/256);
    ASSERT_TRUE(choice.valid);
    EXPECT_EQ(choice.source_participant, 0u);
    EXPECT_EQ(choice.destination_participant, 1u);
    EXPECT_EQ(choice.projected_shift, 30u);
    EXPECT_EQ(choice.delta.improvement, 30u);

    uint64_t scratch[3] = {};
    ASSERT_TRUE(moe_rebalance_policy::addingResidentImprovesDynamicSpread(
        current_load,
        /*expert_count=*/30,
        owner_mask,
        owner_mask | moe_rebalance_policy::participantBit(choice.destination_participant),
        /*participant_count=*/3,
        choice.source_participant,
        choice.destination_participant,
        scratch,
        /*window_size_tokens=*/256));
    EXPECT_EQ(scratch[0], 60u);
    EXPECT_EQ(scratch[1], 40u);
    EXPECT_EQ(scratch[2], 30u);
}

TEST(Test__MoERebalanceController, DynamicPolicyDefaultsAreSharedAcrossCpuAndDevice)
{
    SocketRebalanceConfig cpu_config;
    DeviceMoERebalanceConfig device_config;

    EXPECT_FLOAT_EQ(cpu_config.imbalance_threshold,
                    moe_rebalance_policy::kDefaultDynamicImbalanceThresholdRatio);
    EXPECT_FLOAT_EQ(cpu_config.min_improvement_ratio,
                    moe_rebalance_policy::kDefaultDynamicMinImprovementRatio);
    EXPECT_EQ(cpu_config.max_swaps_per_layer,
              static_cast<int>(moe_rebalance_policy::kDefaultDynamicMaxSwapsPerLayer));
    EXPECT_EQ(cpu_config.max_total_swaps,
              static_cast<int>(moe_rebalance_policy::kDefaultDynamicMaxPlanEntriesPerWave));
    EXPECT_EQ(cpu_config.min_window_activations,
              moe_rebalance_policy::kDefaultDynamicMinWindowActivations);

    EXPECT_EQ(device_config.dynamic_imbalance_threshold_per_mille,
              moe_rebalance_policy::kDefaultDynamicImbalanceThresholdPerMille);
    EXPECT_EQ(device_config.dynamic_min_improvement_per_mille,
              moe_rebalance_policy::kDefaultDynamicMinImprovementPerMille);
    EXPECT_EQ(device_config.dynamic_max_swaps_per_layer,
              moe_rebalance_policy::kDefaultDynamicMaxSwapsPerLayer);
    EXPECT_EQ(device_config.dynamic_max_plan_entries_per_wave,
              moe_rebalance_policy::kDefaultDynamicMaxPlanEntriesPerWave);
    EXPECT_EQ(device_config.dynamic_min_window_activations,
              moe_rebalance_policy::kDefaultDynamicMinWindowActivations);
}

TEST(Test__MoERebalanceController, DeviceSidePayloadBucketsRoundToPowerOfTwoCapacity)
{
    EXPECT_EQ(deviceMoERebalancePayloadBucketSlots(0, 8), 0u);
    EXPECT_EQ(deviceMoERebalancePayloadBucketSlots(1, 8), 1u);
    EXPECT_EQ(deviceMoERebalancePayloadBucketSlots(2, 8), 2u);
    EXPECT_EQ(deviceMoERebalancePayloadBucketSlots(3, 8), 4u);
    EXPECT_EQ(deviceMoERebalancePayloadBucketSlots(4, 8), 4u);
    EXPECT_EQ(deviceMoERebalancePayloadBucketSlots(5, 8), 8u);
    EXPECT_EQ(deviceMoERebalancePayloadBucketSlots(9, 8), 8u);

    EXPECT_EQ(deviceMoERebalancePayloadBucketIndex(0), 0u);
    EXPECT_EQ(deviceMoERebalancePayloadBucketIndex(1), 0u);
    EXPECT_EQ(deviceMoERebalancePayloadBucketIndex(2), 1u);
    EXPECT_EQ(deviceMoERebalancePayloadBucketIndex(4), 2u);
    EXPECT_EQ(deviceMoERebalancePayloadBucketIndex(8), 3u);

    EXPECT_EQ(moe_rebalance_policy::payloadBucketSlots(3, 8),
              deviceMoERebalancePayloadBucketSlots(3, 8))
        << "CUDA, ROCm, and CPU mirrors must share one bucket scheduler policy.";
}

TEST(Test__MoERebalanceController, DeviceSideTransferWaveValueGateUsesPayloadSlots)
{
    EXPECT_TRUE(moe_rebalance_policy::transferWaveMeetsSpreadImprovementFloor(
        0, 0, 256))
        << "Resident-only waves have no payload slot transfer cost.";
    EXPECT_TRUE(moe_rebalance_policy::transferWaveMeetsSpreadImprovementFloor(
        1, 1, 0))
        << "A zero configured floor disables the wave-level value gate.";
    EXPECT_FALSE(moe_rebalance_policy::transferWaveMeetsSpreadImprovementFloor(
        255, 1, 256));
    EXPECT_TRUE(moe_rebalance_policy::transferWaveMeetsSpreadImprovementFloor(
        256, 1, 256));
    EXPECT_FALSE(moe_rebalance_policy::transferWaveMeetsSpreadImprovementFloor(
        511, 2, 256));
    EXPECT_TRUE(moe_rebalance_policy::transferWaveMeetsSpreadImprovementFloor(
        512, 2, 256));
}

TEST(Test__MoERebalanceController, DeviceSideTransferWavePostLoadSpreadCeilingUsesPermille)
{
    EXPECT_TRUE(moe_rebalance_policy::transferWaveMeetsPostLoadSpreadCeiling(
        999, 10000, 0, 100))
        << "Resident-only waves have no transfer cost to amortize.";
    EXPECT_TRUE(moe_rebalance_policy::transferWaveMeetsPostLoadSpreadCeiling(
        1001, 10000, 1, 0))
        << "A zero configured ceiling disables the residual-spread gate.";
    EXPECT_TRUE(moe_rebalance_policy::transferWaveMeetsPostLoadSpreadCeiling(
        1000, 10000, 1, 100));
    EXPECT_FALSE(moe_rebalance_policy::transferWaveMeetsPostLoadSpreadCeiling(
        1001, 10000, 1, 100));
    EXPECT_TRUE(moe_rebalance_policy::transferWaveMeetsPostLoadSpreadCeiling(
        0, 0, 1, 100))
        << "No projected load means there is no residual imbalance to reject.";
}

TEST(Test__MoERebalanceController, DeviceSideTransferWaveRequiresAggregateSpreadImprovement)
{
    EXPECT_TRUE(moe_rebalance_policy::transferWaveImprovesAggregateLoadSpread(
        200, 300, 10000, 10000, 0))
        << "Resident-only waves do not pay payload transfer cost.";
    EXPECT_TRUE(moe_rebalance_policy::transferWaveImprovesAggregateLoadSpread(
        200, 199, 10000, 10000, 1));
    EXPECT_FALSE(moe_rebalance_policy::transferWaveImprovesAggregateLoadSpread(
        200, 200, 10000, 10000, 1))
        << "A paid transfer wave must do more than tie the aggregate projected spread.";
    EXPECT_FALSE(moe_rebalance_policy::transferWaveImprovesAggregateLoadSpread(
        200, 201, 10000, 10000, 1))
        << "A paid transfer wave must never worsen aggregate projected spread.";
    EXPECT_FALSE(moe_rebalance_policy::transferWaveImprovesAggregateLoadSpread(
        200, 199, 10000, 9999, 1))
        << "Projected transfer waves should preserve total routed work.";
    EXPECT_FALSE(moe_rebalance_policy::transferWaveImprovesAggregateLoadSpread(
        0, 0, 0, 0, 1))
        << "No projected load cannot justify a payload transfer.";
}

TEST(Test__MoERebalanceController, DeviceSideTransferWaveRejectsWorseParticipantSpreadWithoutRouterPayback)
{
    EXPECT_TRUE(moe_rebalance_policy::transferWaveParticipantSpreadIsAcceptable(
        200, 300, 10000, 10000, 0, false))
        << "Resident-only waves do not pay payload transfer cost.";
    EXPECT_TRUE(moe_rebalance_policy::transferWaveParticipantSpreadIsAcceptable(
        200, 199, 10000, 10000, 1, false));
    EXPECT_FALSE(moe_rebalance_policy::transferWaveParticipantSpreadIsAcceptable(
        200, 200, 10000, 10000, 1, false))
        << "Paid waves should not merely tie projected participant spread.";
    EXPECT_FALSE(moe_rebalance_policy::transferWaveParticipantSpreadIsAcceptable(
        200, 201, 10000, 10000, 1, false))
        << "Paid waves should not worsen projected participant spread without measured router payback.";
    EXPECT_TRUE(moe_rebalance_policy::transferWaveParticipantSpreadIsAcceptable(
        200, 201, 10000, 10000, 1, true))
        << "Measured cache-aware routing payback can justify accepting the next paid wave.";
    EXPECT_FALSE(moe_rebalance_policy::transferWaveParticipantSpreadIsAcceptable(
        200, 199, 10000, 9999, 1, true))
        << "Router payback must not paper over inconsistent projected totals.";
    EXPECT_FALSE(moe_rebalance_policy::transferWaveParticipantSpreadIsAcceptable(
        0, 0, 0, 0, 1, true))
        << "No projected load cannot justify a payload transfer.";
}

TEST(Test__MoERebalanceController, DeviceSideTransferWavePrunePreservesResidentAssignmentCommands)
{
    std::vector<DeviceMoERebalancePlanEntry> entries(5);
    entries[0].op = static_cast<uint32_t>(DeviceMoERebalancePlanOp::ExpertPayloadArrival);
    entries[0].expert = 10;
    entries[1].op = static_cast<uint32_t>(DeviceMoERebalancePlanOp::ResidentExpertAssignment);
    entries[1].expert = 11;
    entries[2].op = static_cast<uint32_t>(DeviceMoERebalancePlanOp::ExpertPayloadArrival);
    entries[2].expert = 12;
    entries[3].op = static_cast<uint32_t>(DeviceMoERebalancePlanOp::ResidentExpertAssignment);
    entries[3].expert = 13;
    entries[4].op = 0u;
    entries[4].expert = 14;

    uint32_t resident_count = 0;
    const uint32_t compacted =
        moe_rebalance_policy::prunePayloadArrivalsPreservingResidentAssignments(
            entries.data(),
            static_cast<uint32_t>(entries.size()),
            &resident_count);

    EXPECT_EQ(compacted, 2u);
    EXPECT_EQ(resident_count, 2u);
    EXPECT_EQ(entries[0].op,
              static_cast<uint32_t>(DeviceMoERebalancePlanOp::ResidentExpertAssignment));
    EXPECT_EQ(entries[0].expert, 11u);
    EXPECT_EQ(entries[1].op,
              static_cast<uint32_t>(DeviceMoERebalancePlanOp::ResidentExpertAssignment));
    EXPECT_EQ(entries[1].expert, 13u);

    entries[0].op = static_cast<uint32_t>(DeviceMoERebalancePlanOp::ExpertPayloadArrival);
    entries[1].op = static_cast<uint32_t>(DeviceMoERebalancePlanOp::ExpertPayloadArrival);
    EXPECT_EQ(moe_rebalance_policy::prunePayloadArrivalsPreservingResidentAssignments(
                  entries.data(),
                  2u,
                  &resident_count),
              0u);
    EXPECT_EQ(resident_count, 0u);
}

TEST(Test__MoERebalanceController, SharedDynamicPolicyChoosesPairedOwnershipSwap)
{
    uint64_t participant_load[2] = {100u, 10u};
    uint64_t expert_counts[4] = {70u, 30u, 1u, 9u};
    int32_t expert_owner[4] = {0, 0, 1, 1};

    const auto choice = moe_rebalance_policy::bestDynamicOwnershipSwap(
        participant_load,
        expert_counts,
        expert_owner,
        /*num_experts=*/4,
        /*participant_count=*/2,
        /*imbalance_threshold_per_mille=*/1300,
        /*min_improvement_per_mille=*/50,
        /*min_window_activations=*/64);

    ASSERT_TRUE(choice.valid);
    EXPECT_EQ(choice.overloaded_participant, 0u);
    EXPECT_EQ(choice.underloaded_participant, 1u);
    EXPECT_EQ(choice.heavy_expert, 0u);
    EXPECT_EQ(choice.light_expert, 2u);
    EXPECT_EQ(choice.heavy_count, 70u);
    EXPECT_EQ(choice.light_count, 1u);
    EXPECT_EQ(choice.old_min_load, 10u);
    EXPECT_EQ(choice.old_max_load, 100u);
    EXPECT_EQ(choice.new_min_load, 31u);
    EXPECT_EQ(choice.new_max_load, 79u);

    ASSERT_TRUE(moe_rebalance_policy::applyDynamicOwnershipSwap(
        participant_load,
        expert_owner,
        choice));
    EXPECT_EQ(participant_load[0], 31u);
    EXPECT_EQ(participant_load[1], 79u);
    EXPECT_EQ(expert_owner[0], 1);
    EXPECT_EQ(expert_owner[2], 0);
}

TEST(Test__MoERebalanceController, HostApplyRefreshesMultiResidentVisibility)
{
    DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 0;
    config.participant_count = 2;
    config.root_participant = 0;
    config.window_size_tokens = 1;

    std::array<DeviceMoELayerRuntime, 1> runtime_layers{
        makeDeviceRuntimeLayerForLoadStats()};

    DeviceMoERebalancePlanEntry plan;
    plan.op = static_cast<uint32_t>(DeviceMoERebalancePlanOp::ResidentExpertAssignment);
    plan.layer = 0;
    plan.expert = 1;
    plan.source_participant = 1;
    plan.destination_participant = config.participant_id;
    plan.source_resident_mask = 0b10u;
    plan.destination_slot = kDeviceMoEInvalidSlot;
    plan.payload_slot = kDeviceMoEInvalidSlot;

    DeviceMoERebalanceApplyStatus status;
    ASSERT_TRUE(applyDeviceMoERebalanceArrivalsHost(
        runtime_layers.data(),
        &plan,
        1,
        nullptr,
        nullptr,
        0,
        config,
        &status));

    const auto &runtime = runtime_layers[0];
    const auto &bank = runtime.banks[runtime.active_bank];
    EXPECT_EQ(status.status_code,
              static_cast<uint32_t>(DeviceMoERebalanceApplyStatusCode::Ok));
    EXPECT_EQ(status.applied_arrivals, 1u);
    EXPECT_EQ(status.changed_layers, 1u);
    EXPECT_EQ(status.post_apply_multi_resident_experts, 1u);
    EXPECT_EQ(bank.resident_participant_mask[1], 0b11u);
    EXPECT_EQ(bank.reserved[0], 1u)
        << "host mirror must refresh the cheap hot-cache visibility gate.";
}

TEST(Test__MoERebalanceController, DeviceSideLoadSpreadStatusIsOptIn)
{
    DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 0;
    config.participant_count = 2;
    config.root_participant = 0;
    config.window_size_tokens = 1;
    config.max_hot_replicas_per_participant = 1;
    config.layer_window_count = 1;
    config.layer_wave_count = 1;

    std::vector<uint64_t> gathered_histograms(
        static_cast<size_t>(config.participant_count) *
            static_cast<size_t>(config.num_layers) *
            static_cast<size_t>(config.num_experts),
        0);
    gathered_histograms[0] = 2;
    gathered_histograms[config.num_experts] = 2;

    DeviceMoELayerRuntime default_runtime = makeDeviceRuntimeLayerForLoadStats();
    DeviceMoERebalanceStatus default_status;
    ASSERT_TRUE(applyDeviceMoERebalancePolicyHost(
        &default_runtime,
        gathered_histograms.data(),
        config,
        &default_status));
    EXPECT_EQ(default_status.pre_policy_load_total, 0u);
    EXPECT_EQ(default_status.post_policy_load_total, 0u);
    EXPECT_EQ(default_status.pre_policy_imbalance_numerator, 0u);
    EXPECT_EQ(default_status.post_policy_imbalance_numerator, 0u);

    config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::CollectLoadStats);
    DeviceMoELayerRuntime stats_runtime = makeDeviceRuntimeLayerForLoadStats();
    DeviceMoERebalanceStatus stats_status;
    ASSERT_TRUE(applyDeviceMoERebalancePolicyHost(
        &stats_runtime,
        gathered_histograms.data(),
        config,
        &stats_status));
    EXPECT_GT(stats_status.pre_policy_load_total, 0u);
    EXPECT_GT(stats_status.pre_policy_imbalance_numerator, 0u);
    EXPECT_GT(stats_status.post_policy_load_total, 0u);
}

TEST(Test__MoERebalanceController, Construction_OffMode)
{
    auto cfg = makeConfig(MoERebalanceMode::OFF);
    MoERebalanceController ctrl(cfg);

    EXPECT_EQ(ctrl.mode(), MoERebalanceMode::OFF);
    EXPECT_EQ(ctrl.histogram(), nullptr);
    EXPECT_FALSE(ctrl.shouldRebalance());
    EXPECT_EQ(ctrl.totalRebalances(), 0);
    EXPECT_EQ(ctrl.totalSwaps(), 0);
}

TEST(Test__MoERebalanceController, Construction_ObserveMode)
{
    auto cfg = makeConfig(MoERebalanceMode::OBSERVE);
    MoERebalanceController ctrl(cfg);

    EXPECT_EQ(ctrl.mode(), MoERebalanceMode::OBSERVE);
    EXPECT_NE(ctrl.histogram(), nullptr);
    EXPECT_FALSE(ctrl.shouldRebalance());
}

TEST(Test__MoERebalanceController, Construction_DynamicMode)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC);
    MoERebalanceController ctrl(cfg);

    EXPECT_EQ(ctrl.mode(), MoERebalanceMode::DYNAMIC);
    EXPECT_EQ(ctrl.requestedMode(), MoERebalanceMode::DYNAMIC);
    EXPECT_NE(ctrl.histogram(), nullptr);
    EXPECT_FALSE(ctrl.shouldRebalance());
    EXPECT_EQ(ctrl.rebalanceDecision().reason, MoERebalanceDecisionReason::WindowNotFull);
}

TEST(Test__MoERebalanceController, RebalanceDecisionReasonsAreStableStrings)
{
    EXPECT_STREQ(toString(MoERebalanceDecisionReason::ModeOff), "mode_off");
    EXPECT_STREQ(toString(MoERebalanceDecisionReason::DynamicDisabledForDomain), "dynamic_disabled_for_domain");
    EXPECT_STREQ(toString(MoERebalanceDecisionReason::SingleParticipantObserveOnly), "single_participant_observe_only");
    EXPECT_STREQ(toString(MoERebalanceDecisionReason::WindowNotFull), "window_not_full");
    EXPECT_STREQ(toString(MoERebalanceDecisionReason::Ready), "ready");
}

TEST(Test__MoERebalanceController, ObserveModeReportsDynamicDisabledReason)
{
    auto cfg = makeConfig(MoERebalanceMode::OBSERVE);
    MoERebalanceController ctrl(cfg);

    fillWindowBalanced(*ctrl.histogram(), 16, 2, 8, 2);

    const auto decision = ctrl.rebalanceDecision();
    EXPECT_FALSE(decision.ready);
    EXPECT_EQ(decision.reason, MoERebalanceDecisionReason::DynamicDisabledForDomain);
    EXPECT_FALSE(ctrl.shouldRebalance());
}

TEST(Test__MoERebalanceController, DynamicSingleParticipantDowngradesToObserveOnly)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, /*num_experts=*/8, /*num_sockets=*/1,
                          /*num_layers=*/2, /*top_k=*/2, /*window_size=*/16);
    for (int e = 0; e < 8; ++e)
        cfg.initial_expert_to_socket[e] = 0;

    MoERebalanceController ctrl(cfg);
    const auto initial = ctrl.currentPlacement();

    EXPECT_EQ(ctrl.requestedMode(), MoERebalanceMode::DYNAMIC);
    EXPECT_EQ(ctrl.mode(), MoERebalanceMode::OBSERVE);
    ASSERT_NE(ctrl.histogram(), nullptr);
    EXPECT_EQ(ctrl.participantCount(), 1);

    fillWindowSkewed(*ctrl.histogram(), 16, 2, 2);

    const auto decision = ctrl.rebalanceDecision();
    EXPECT_FALSE(decision.ready);
    EXPECT_EQ(decision.reason, MoERebalanceDecisionReason::SingleParticipantObserveOnly);
    EXPECT_FALSE(ctrl.shouldRebalance());

    EXPECT_TRUE(ctrl.rebalance().empty());
    ctrl.rebalanceLPT();
    const auto replicas = ctrl.proposeReplicas(/*max_replicas_per_socket=*/1);

    EXPECT_EQ(ctrl.currentPlacement(), initial);
    EXPECT_EQ(ctrl.placementEpoch(), 0u);
    EXPECT_EQ(ctrl.totalRebalances(), 0);
    EXPECT_EQ(replicas.num_replicated, 0);
}

TEST(Test__MoERebalanceController, ParticipantVocabularyAliasesLegacySocketState)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, /*num_experts=*/6, /*num_sockets=*/3,
                          /*num_layers=*/2, /*top_k=*/2, /*window_size=*/16);
    MoERebalanceController ctrl(cfg);

    EXPECT_EQ(ctrl.participantCount(), 3);
    EXPECT_EQ(ctrl.participantDevices(), cfg.sockets);
    EXPECT_EQ(ctrl.currentParticipantPlacement(), ctrl.currentPlacement());
    EXPECT_EQ(ctrl.computeExpertMasksForParticipant(2), ctrl.computeExpertMasks(2));

    fillWindowBalanced(*ctrl.histogram(), 16, 2, 6, 2);
    const auto legacy_replicas = ctrl.proposeReplicas(/*max_replicas_per_socket=*/1);

    auto cfg_alias = cfg;
    MoERebalanceController alias_ctrl(cfg_alias);
    fillWindowBalanced(*alias_ctrl.histogram(), 16, 2, 6, 2);
    const auto participant_replicas =
        alias_ctrl.proposeReplicasForParticipants(/*max_replicas_per_participant=*/1);

    EXPECT_TRUE(participant_replicas.sameReplicaPlacement(legacy_replicas));
}

TEST(Test__MoERebalanceController, NonCpuParticipantsSupportOwnershipAndReplicaRebalance)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, /*num_experts=*/8, /*num_sockets=*/2,
                          /*num_layers=*/1, /*top_k=*/2, /*window_size=*/16);
    cfg.domain_id = "cuda_ep";
    cfg.sockets = {DeviceId::cuda(0), DeviceId::cuda(1)};
    for (int e = 0; e < 8; ++e)
        cfg.initial_expert_to_socket[e] = (e < 6) ? 0 : 1;

    MoERebalanceController replica_ctrl(cfg);
    EXPECT_EQ(replica_ctrl.participantDevices(), cfg.sockets);

    fillWindowSkewed(*replica_ctrl.histogram(), 16, 1, 2);
    auto replicas = replica_ctrl.proposeReplicasForParticipants(/*max_replicas_per_participant=*/1);
    ASSERT_GT(replicas.num_replicated, 0);
    EXPECT_EQ(replicas.domain_id, "cuda_ep");

    auto replica_masks0 = replica_ctrl.computeExpertMasksForParticipant(0);
    auto replica_masks1 = replica_ctrl.computeExpertMasksForParticipant(1);
    ASSERT_EQ(replica_masks0.size(), 1u);
    ASSERT_EQ(replica_masks1.size(), 1u);
    ASSERT_EQ(replica_masks0[0].size(), 8u);
    ASSERT_EQ(replica_masks1[0].size(), 8u);
    EXPECT_TRUE(replica_masks0[0][0]);
    EXPECT_TRUE(replica_masks1[0][0]);

    MoERebalanceController swap_ctrl(cfg);
    fillWindowSkewed(*swap_ctrl.histogram(), 16, 1, 2);
    ASSERT_TRUE(swap_ctrl.shouldRebalance());
    const auto new_placement = swap_ctrl.rebalance();
    ASSERT_EQ(new_placement.size(), 8u);
    EXPECT_NE(new_placement, cfg.initial_expert_to_socket);
    EXPECT_EQ(swap_ctrl.placementEpoch(), 1u);

    auto masks0 = swap_ctrl.computeExpertMasksForParticipant(0);
    auto masks1 = swap_ctrl.computeExpertMasksForParticipant(1);
    ASSERT_EQ(masks0.size(), 1u);
    ASSERT_EQ(masks1.size(), 1u);
    EXPECT_EQ(masks0[0].size(), 8u);
    EXPECT_EQ(masks1[0].size(), 8u);
}

TEST(Test__MoERebalanceController, ReplicaSetsCarryDomainId)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, /*num_experts=*/8, /*num_sockets=*/2,
                          /*num_layers=*/2, /*top_k=*/2, /*window_size=*/16);
    cfg.domain_id = "expert_hot";
    MoERebalanceController ctrl(cfg);

    recordExpertHits(*ctrl.histogram(), 0, {{0, 16}});
    const auto replicas = ctrl.proposeReplicasForParticipants(/*max_replicas_per_participant=*/1);
    ASSERT_GT(replicas.num_replicated, 0);
    EXPECT_EQ(replicas.domain_id, "expert_hot");

    const auto arrivals = replicas.arrivalsSince(ExpertReplicaSet{});
    EXPECT_EQ(arrivals.domain_id, "expert_hot");

    auto same_domain = replicas;
    EXPECT_TRUE(replicas.sameReplicaPlacement(same_domain));

    auto other_domain = replicas;
    other_domain.domain_id = "expert_cold";
    EXPECT_FALSE(replicas.sameReplicaPlacement(other_domain));
}

TEST(Test__MoERebalanceController, ShouldRebalance_WindowNotFull)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    MoERebalanceController ctrl(cfg);

    // Record fewer tokens than window size
    int indices[] = {0, 1};
    float weights[] = {0.5f, 0.5f};
    ctrl.histogram()->record(0, indices, weights, 2);

    EXPECT_FALSE(ctrl.shouldRebalance());
}

TEST(Test__MoERebalanceController, ShouldRebalance_WindowFull)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    MoERebalanceController ctrl(cfg);

    fillWindowBalanced(*ctrl.histogram(), 16, 2, 8, 2);

    EXPECT_TRUE(ctrl.shouldRebalance());
}

TEST(Test__MoERebalanceController, ShouldRebalance_UsesConfiguredTokenBoundaryLayer)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 3, 2, 2);
    cfg.token_boundary_layer_idx = 1;
    MoERebalanceController ctrl(cfg);

    int indices[] = {0, 1};
    float weights[] = {0.5f, 0.5f};

    ctrl.histogram()->record(0, indices, weights, 2);
    ctrl.histogram()->record(1, indices, weights, 2);
    EXPECT_FALSE(ctrl.shouldRebalance());

    ctrl.histogram()->record(0, indices, weights, 2);
    ctrl.histogram()->record(1, indices, weights, 2);
    EXPECT_TRUE(ctrl.shouldRebalance())
        << "A non-routed tail layer must not be required to fill the decode window.";
}

TEST(Test__MoERebalanceController, Rebalance_NoImbalance)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    MoERebalanceController ctrl(cfg);

    // Fill with balanced routing
    fillWindowBalanced(*ctrl.histogram(), 16, 2, 8, 2);
    ASSERT_TRUE(ctrl.shouldRebalance());

    auto result = ctrl.rebalance();

    // With balanced routing, proposal should be empty (no beneficial swaps)
    EXPECT_TRUE(result.empty());
    // Window should be reset
    EXPECT_FALSE(ctrl.shouldRebalance());
}

TEST(Test__MoERebalanceController, Rebalance_WithImbalance)
{
    // Placement: experts 0-5 on socket 0, experts 6-7 on socket 1
    // Routing skewed to experts 0,1 (both on socket 0) → socket 0 is heavily overloaded
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    for (int e = 0; e < 8; ++e)
        cfg.initial_expert_to_socket[e] = (e < 6) ? 0 : 1;
    MoERebalanceController ctrl(cfg);

    // Skewed routing → heavy imbalance between sockets
    fillWindowSkewed(*ctrl.histogram(), 16, 2, 2);
    ASSERT_TRUE(ctrl.shouldRebalance());

    auto result = ctrl.rebalance();

    // With heavy imbalance, rebalancer should propose swaps and return new placement
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(static_cast<int>(result.size()), 8); // Full placement vector
    EXPECT_EQ(ctrl.placementEpoch(), 1u);

    const auto &before = ctrl.lastImbalanceBefore();
    const auto &after = ctrl.lastImbalanceAfter();
    ASSERT_TRUE(before.valid);
    ASSERT_TRUE(after.valid);
    EXPECT_GT(before.average_spread, after.average_spread);
    EXPECT_EQ(before.total_activations, after.total_activations);
}

TEST(Test__MoERebalanceController, Rebalance_UpdatesPlacement)
{
    // Experts 0-5 on socket 0, 6-7 on socket 1
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    for (int e = 0; e < 8; ++e)
        cfg.initial_expert_to_socket[e] = (e < 6) ? 0 : 1;
    MoERebalanceController ctrl(cfg);

    auto initial = ctrl.currentPlacement();

    fillWindowSkewed(*ctrl.histogram(), 16, 2, 2);
    auto result = ctrl.rebalance();

    if (!result.empty())
    {
        // Pair swaps move a heavy expert to socket 1 and a light expert to socket 0.
        // At least one expert's socket assignment should differ from initial.
        bool placement_changed = false;
        for (int e = 0; e < 8; ++e)
        {
            if (ctrl.currentPlacement()[e] != initial[e])
                placement_changed = true;
        }
        EXPECT_TRUE(placement_changed);
    }
}

TEST(Test__MoERebalanceController, Rebalance_ResetsWindow)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    MoERebalanceController ctrl(cfg);

    fillWindowBalanced(*ctrl.histogram(), 16, 2, 8, 2);
    ASSERT_TRUE(ctrl.shouldRebalance());

    ctrl.rebalance();

    // Window should be reset after rebalance
    EXPECT_FALSE(ctrl.shouldRebalance());
    EXPECT_FALSE(ctrl.histogram()->windowFull());
}

TEST(Test__MoERebalanceController, Rebalance_CountsTotal)
{
    // Experts 0-5 on socket 0, 6-7 on socket 1 for forced imbalance
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    for (int e = 0; e < 8; ++e)
        cfg.initial_expert_to_socket[e] = (e < 6) ? 0 : 1;
    MoERebalanceController ctrl(cfg);

    EXPECT_EQ(ctrl.totalRebalances(), 0);
    EXPECT_EQ(ctrl.totalSwaps(), 0);

    // First rebalance cycle
    fillWindowSkewed(*ctrl.histogram(), 16, 2, 2);
    auto result1 = ctrl.rebalance();

    if (!result1.empty())
    {
        EXPECT_EQ(ctrl.totalRebalances(), 1);
        EXPECT_GT(ctrl.totalSwaps(), 0);
        int swaps_after_first = ctrl.totalSwaps();

        // Second rebalance cycle
        fillWindowSkewed(*ctrl.histogram(), 16, 2, 2);
        auto result2 = ctrl.rebalance();

        if (!result2.empty())
        {
            EXPECT_EQ(ctrl.totalRebalances(), 2);
            EXPECT_GE(ctrl.totalSwaps(), swaps_after_first);
        }
    }
}

TEST(Test__MoERebalanceController, ObserveMode_NeverRebalances)
{
    // Experts 0-5 on socket 0, 6-7 on socket 1 for extreme imbalance
    auto cfg = makeConfig(MoERebalanceMode::OBSERVE, 8, 2, 2, 2, 16);
    for (int e = 0; e < 8; ++e)
        cfg.initial_expert_to_socket[e] = (e < 6) ? 0 : 1;
    MoERebalanceController ctrl(cfg);

    // Fill window
    fillWindowSkewed(*ctrl.histogram(), 16, 2, 2);

    // Window is full but mode is OBSERVE -> never triggers
    EXPECT_FALSE(ctrl.shouldRebalance());
    EXPECT_EQ(ctrl.totalRebalances(), 0);
}

TEST(Test__MoERebalanceController, LogHistogramSummary_NoThrow)
{
    // OFF mode
    {
        auto cfg = makeConfig(MoERebalanceMode::OFF);
        MoERebalanceController ctrl(cfg);
        EXPECT_NO_THROW(ctrl.logHistogramSummary());
    }

    // OBSERVE mode with some data
    {
        auto cfg = makeConfig(MoERebalanceMode::OBSERVE, 8, 2, 2, 2, 16);
        MoERebalanceController ctrl(cfg);
        int indices[] = {0, 1};
        float weights[] = {0.5f, 0.5f};
        ctrl.histogram()->record(0, indices, weights, 2);
        EXPECT_NO_THROW(ctrl.logHistogramSummary());
    }

    // DYNAMIC mode with full window
    {
        auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
        MoERebalanceController ctrl(cfg);
        fillWindowBalanced(*ctrl.histogram(), 16, 2, 8, 2);
        EXPECT_NO_THROW(ctrl.logHistogramSummary());
    }
}

// ── Helper for targeted expert routing ────────────────

/// Fill histogram with routing that always activates the given experts
static void fillWindowWithExperts(DecodeExpertHistogram &hist, int window_size,
                                  int num_layers, const std::vector<int> &active_experts)
{
    int top_k = static_cast<int>(active_experts.size());
    for (int t = 0; t < window_size; ++t)
    {
        for (int l = 0; l < num_layers; ++l)
        {
            std::vector<float> weights(top_k, 1.0f / top_k);
            hist.record(l, active_experts.data(), weights.data(), top_k);
        }
    }
}

static void recordExpertHits(DecodeExpertHistogram &hist, int layer,
                             const std::vector<std::pair<int, int>> &expert_counts)
{
    constexpr float weight = 1.0f;
    for (const auto &[expert, count] : expert_counts)
    {
        for (int i = 0; i < count; ++i)
            hist.record(layer, &expert, &weight, 1);
    }
}

static bool anyGpuSocketHas(const std::vector<std::vector<std::vector<bool>>> &masks,
                            int layer, int expert)
{
    return masks[0][layer][expert] || masks[1][layer][expert];
}

static bool anyCpuSocketHas(const std::vector<std::vector<std::vector<bool>>> &masks,
                            int layer, int expert)
{
    return masks[2][layer][expert] || masks[3][layer][expert];
}

TEST(Test__MoERebalanceController, PlacementEpochTracksBasePlacementAndReplicas)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, /*num_experts=*/8, /*num_sockets=*/2,
                          /*num_layers=*/2, /*top_k=*/2, /*window_size=*/16);
    for (int e = 0; e < 8; ++e)
        cfg.initial_expert_to_socket[e] = (e < 4) ? 0 : 1;

    MoERebalanceController ctrl(cfg);
    EXPECT_EQ(ctrl.placementEpoch(), 0u);

    fillWindowSkewed(*ctrl.histogram(), 16, 2, 2);
    const auto rebalanced = ctrl.rebalance();
    ASSERT_FALSE(rebalanced.empty());
    EXPECT_EQ(ctrl.placementEpoch(), 1u);

    const auto placement_after_rebalance = ctrl.currentPlacement();
    auto hot_expert = std::find(placement_after_rebalance.begin(), placement_after_rebalance.end(), 0);
    if (hot_expert == placement_after_rebalance.end())
        hot_expert = placement_after_rebalance.begin();
    ASSERT_NE(hot_expert, placement_after_rebalance.end());
    recordExpertHits(
        *ctrl.histogram(),
        0,
        {{static_cast<int>(std::distance(placement_after_rebalance.begin(), hot_expert)), 16}});
    const auto replicas = ctrl.proposeReplicas(/*max_replicas_per_socket=*/1);
    ASSERT_GT(replicas.num_replicated, 0);
    EXPECT_EQ(ctrl.placementEpoch(), 2u);

    const auto same_replicas = ctrl.proposeReplicas(/*max_replicas_per_socket=*/1);
    EXPECT_EQ(same_replicas.num_replicated, replicas.num_replicated);
    EXPECT_EQ(ctrl.placementEpoch(), 2u);
}

TEST(Test__MoERebalanceController, ReplicasExpandMasksButPreserveBasePlacement)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, /*num_experts=*/8, /*num_sockets=*/2,
                          /*num_layers=*/2, /*top_k=*/2, /*window_size=*/16);
    for (int e = 0; e < 8; ++e)
        cfg.initial_expert_to_socket[e] = (e < 4) ? 0 : 1;

    MoERebalanceController ctrl(cfg);
    const auto initial_placement = ctrl.currentPlacement();

    recordExpertHits(*ctrl.histogram(), 0, {{0, 20}});
    recordExpertHits(*ctrl.histogram(), 1, {{4, 20}});
    ASSERT_TRUE(ctrl.shouldRebalance());

    auto replicas = ctrl.proposeReplicas(/*max_replicas_per_socket=*/1);
    ASSERT_EQ(replicas.num_replicated, 2);
    EXPECT_TRUE(replicas.is_replicated[0]);
    EXPECT_TRUE(replicas.is_replicated[4]);
    EXPECT_TRUE(replicas.hasLayerReplicaPlacement());
    EXPECT_TRUE(replicas.hasReplicaOnParticipant(0, 0, 1));
    EXPECT_TRUE(replicas.hasReplicaOnParticipant(1, 4, 0));
    EXPECT_FALSE(replicas.hasReplicaOnParticipant(0, 4, 0))
        << "a hot expert id must not implicitly expand to every layer";
    EXPECT_FALSE(replicas.hasReplicaOnParticipant(1, 0, 1))
        << "a hot expert id must not implicitly expand to every layer";
    EXPECT_EQ(ctrl.currentPlacement(), initial_placement);

    auto socket0_masks = ctrl.computeExpertMasks(0);
    auto socket1_masks = ctrl.computeExpertMasks(1);
    EXPECT_EQ(ctrl.computeExpertMasksForParticipant(0), socket0_masks);
    EXPECT_EQ(ctrl.computeExpertMasksForParticipant(1), socket1_masks);
    ASSERT_EQ(static_cast<int>(socket0_masks.size()), 2);
    ASSERT_EQ(static_cast<int>(socket1_masks.size()), 2);

    EXPECT_TRUE(socket0_masks[0][0]);
    EXPECT_TRUE(socket1_masks[0][0]);
    EXPECT_FALSE(socket0_masks[0][4]);
    EXPECT_TRUE(socket1_masks[0][4]);
    EXPECT_TRUE(socket0_masks[1][0]);
    EXPECT_FALSE(socket1_masks[1][0]);
    EXPECT_TRUE(socket0_masks[1][4]);
    EXPECT_TRUE(socket1_masks[1][4]);

    for (int layer = 0; layer < 2; ++layer)
    {
        EXPECT_TRUE(socket0_masks[layer][1]);
        EXPECT_FALSE(socket1_masks[layer][1]);
        EXPECT_FALSE(socket0_masks[layer][5]);
        EXPECT_TRUE(socket1_masks[layer][5]);
    }

    ctrl.resetRebalanceWindow();
    EXPECT_FALSE(ctrl.shouldRebalance());
    EXPECT_FALSE(ctrl.histogram()->windowFull());
    EXPECT_EQ(ctrl.currentPlacement(), initial_placement);
}

TEST(Test__MoERebalanceController, ReplicaProposalSkipsHotRemoteExpertWhenTargetIsAlreadyHeavier)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, /*num_experts=*/4, /*num_sockets=*/2,
                          /*num_layers=*/1, /*top_k=*/1, /*window_size=*/16);
    MoERebalanceController ctrl(cfg);

    recordExpertHits(*ctrl.histogram(), 0, {{0, 100}, {1, 90}});

    auto replicas = ctrl.proposeReplicasForParticipants(/*max_replicas_per_participant=*/1);
    ASSERT_EQ(replicas.num_sockets, 2);
    ASSERT_EQ(replicas.num_replicated, 1);

    EXPECT_TRUE(replicas.hasReplicaOnParticipant(0, 0, 1))
        << "expert 0 is owned by the heavier participant and should be offloaded";
    EXPECT_FALSE(replicas.hasReplicaOnParticipant(0, 1, 0))
        << "expert 1 is hot, but copying it to the already-heavier participant would not reduce load";

    const auto &before = ctrl.lastImbalanceBefore();
    const auto &after = ctrl.lastImbalanceAfter();
    ASSERT_TRUE(before.valid);
    ASSERT_TRUE(after.valid);
    EXPECT_GT(before.average_spread, after.average_spread)
        << "hot replicas should publish projected load-spread relief";
    EXPECT_EQ(before.total_activations, after.total_activations);
}

TEST(Test__MoERebalanceController, ReplicaProposalSkipsLowBenefitCandidates)
{
    auto low_cfg = makeConfig(MoERebalanceMode::DYNAMIC, /*num_experts=*/4, /*num_sockets=*/2,
                              /*num_layers=*/1, /*top_k=*/1, /*window_size=*/16);
    MoERebalanceController low_ctrl(low_cfg);
    recordExpertHits(*low_ctrl.histogram(), 0, {{0, 2}});

    auto low_replicas = low_ctrl.proposeReplicasForParticipants(/*max_replicas_per_participant=*/1);
    EXPECT_EQ(low_replicas.num_replicated, 0)
        << "a one-assignment projected shift is below the replica admission threshold";

    auto high_cfg = makeConfig(MoERebalanceMode::DYNAMIC, /*num_experts=*/4, /*num_sockets=*/2,
                               /*num_layers=*/1, /*top_k=*/1, /*window_size=*/16);
    MoERebalanceController high_ctrl(high_cfg);
    recordExpertHits(*high_ctrl.histogram(), 0, {{0, 4}});

    auto high_replicas = high_ctrl.proposeReplicasForParticipants(/*max_replicas_per_participant=*/1);
    ASSERT_EQ(high_replicas.num_replicated, 1);
    EXPECT_TRUE(high_replicas.hasReplicaOnParticipant(0, 0, 1))
        << "a two-assignment projected shift is large enough to admit";
}

TEST(Test__MoERebalanceController, ReplicaAdmissionThresholdScalesWithWindowSize)
{
    auto low_cfg = makeConfig(MoERebalanceMode::DYNAMIC, /*num_experts=*/4, /*num_sockets=*/2,
                              /*num_layers=*/1, /*top_k=*/1, /*window_size=*/256);
    MoERebalanceController low_ctrl(low_cfg);
    recordExpertHits(*low_ctrl.histogram(), 0, {{0, 30}});

    auto low_replicas = low_ctrl.proposeReplicasForParticipants(/*max_replicas_per_participant=*/1);
    EXPECT_EQ(low_replicas.num_replicated, 0)
        << "a 15-assignment projected shift is below the 256-token admission threshold";

    auto high_cfg = makeConfig(MoERebalanceMode::DYNAMIC, /*num_experts=*/4, /*num_sockets=*/2,
                               /*num_layers=*/1, /*top_k=*/1, /*window_size=*/256);
    MoERebalanceController high_ctrl(high_cfg);
    recordExpertHits(*high_ctrl.histogram(), 0, {{0, 32}});

    auto high_replicas = high_ctrl.proposeReplicasForParticipants(/*max_replicas_per_participant=*/1);
    ASSERT_EQ(high_replicas.num_replicated, 1);
    EXPECT_TRUE(high_replicas.hasReplicaOnParticipant(0, 0, 1))
        << "a 16-assignment projected shift is large enough to admit";
}

TEST(Test__MoERebalanceController, ReplicaProposalIsLayerAndParticipantScopedForVariableDomain)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, /*num_experts=*/6, /*num_sockets=*/3,
                          /*num_layers=*/3, /*top_k=*/1, /*window_size=*/16);
    MoERebalanceController ctrl(cfg);

    recordExpertHits(*ctrl.histogram(), 0, {{1, 20}});
    recordExpertHits(*ctrl.histogram(), 2, {{2, 15}});

    auto replicas = ctrl.proposeReplicasForParticipants(/*max_replicas_per_participant=*/1);
    ASSERT_EQ(replicas.num_sockets, 3);
    ASSERT_EQ(replicas.num_replicated, 3);
    EXPECT_TRUE(replicas.hasLayerReplicaPlacement());

    EXPECT_TRUE(replicas.hasReplicaOnParticipant(0, 1, 0));
    EXPECT_FALSE(replicas.hasReplicaOnParticipant(0, 1, 1))
        << "participant 1 owns expert 1 and should not receive its own replica";
    EXPECT_TRUE(replicas.hasReplicaOnParticipant(0, 1, 2));

    EXPECT_FALSE(replicas.hasReplicaOnParticipant(2, 2, 0))
        << "participant 0 spent its one replica slot on the hotter layer-0 expert";
    EXPECT_TRUE(replicas.hasReplicaOnParticipant(2, 2, 1));
    EXPECT_FALSE(replicas.hasReplicaOnParticipant(2, 2, 2))
        << "participant 2 owns expert 2 and should not receive its own replica";

    for (int participant = 0; participant < 3; ++participant)
    {
        EXPECT_FALSE(replicas.hasReplicaOnParticipant(1, 1, participant));
        EXPECT_FALSE(replicas.hasReplicaOnParticipant(1, 2, participant));
    }

    auto masks0 = ctrl.computeExpertMasksForParticipant(0);
    auto masks1 = ctrl.computeExpertMasksForParticipant(1);
    auto masks2 = ctrl.computeExpertMasksForParticipant(2);
    ASSERT_EQ(masks0.size(), 3u);
    ASSERT_EQ(masks1.size(), 3u);
    ASSERT_EQ(masks2.size(), 3u);

    EXPECT_TRUE(masks0[0][1]);
    EXPECT_TRUE(masks1[0][1]);
    EXPECT_TRUE(masks2[0][1]);
    EXPECT_FALSE(masks0[1][1]);
    EXPECT_TRUE(masks1[1][1]);
    EXPECT_FALSE(masks2[1][1]);

    EXPECT_FALSE(masks0[2][2]);
    EXPECT_TRUE(masks1[2][2]);
    EXPECT_TRUE(masks2[2][2]);
}

TEST(Test__MoERebalanceController, ReplicaProposalKeepsStillWarmExistingReplica)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, /*num_experts=*/6, /*num_sockets=*/2,
                          /*num_layers=*/1, /*top_k=*/1, /*window_size=*/16);
    MoERebalanceController ctrl(cfg);

    recordExpertHits(*ctrl.histogram(), 0, {{1, 10}});
    auto first = ctrl.proposeReplicas(/*max_replicas_per_socket=*/1);
    ASSERT_EQ(first.num_replicated, 1);
    ASSERT_TRUE(first.is_replicated[1]);

    ctrl.resetRebalanceWindow();
    recordExpertHits(*ctrl.histogram(), 0, {{1, 8}, {3, 10}});
    auto second = ctrl.proposeReplicas(/*max_replicas_per_socket=*/1);
    EXPECT_EQ(second.num_replicated, 1);
    EXPECT_TRUE(second.is_replicated[1])
        << "existing replica remains within 50% of the hottest replacement";
    EXPECT_FALSE(second.is_replicated[3]);
    EXPECT_EQ(ctrl.placementEpoch(), 1u)
        << "keeping the same replica placement must not invalidate graph caches";
}

TEST(Test__MoERebalanceController, ReplicaProposalReplacesColdExistingReplica)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, /*num_experts=*/6, /*num_sockets=*/2,
                          /*num_layers=*/1, /*top_k=*/1, /*window_size=*/16);
    MoERebalanceController ctrl(cfg);

    recordExpertHits(*ctrl.histogram(), 0, {{1, 10}});
    auto first = ctrl.proposeReplicas(/*max_replicas_per_socket=*/1);
    ASSERT_EQ(first.num_replicated, 1);
    ASSERT_TRUE(first.is_replicated[1]);

    ctrl.resetRebalanceWindow();
    recordExpertHits(*ctrl.histogram(), 0, {{1, 4}, {3, 10}});
    auto second = ctrl.proposeReplicas(/*max_replicas_per_socket=*/1);
    EXPECT_EQ(second.num_replicated, 1);
    EXPECT_FALSE(second.is_replicated[1]);
    EXPECT_TRUE(second.is_replicated[3])
        << "cold existing replica should make room for a much hotter expert";
    EXPECT_EQ(ctrl.placementEpoch(), 2u);
}

TEST(Test__MoERebalanceController, ReplicaProposalPreservesExistingReplicasWithoutNewSignal)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, /*num_experts=*/6, /*num_sockets=*/2,
                          /*num_layers=*/1, /*top_k=*/1, /*window_size=*/16);
    MoERebalanceController ctrl(cfg);

    recordExpertHits(*ctrl.histogram(), 0, {{1, 10}});
    auto first = ctrl.proposeReplicas(/*max_replicas_per_socket=*/1);
    ASSERT_EQ(first.num_replicated, 1);
    ASSERT_TRUE(first.is_replicated[1]);
    const auto epoch_after_first = ctrl.placementEpoch();

    ctrl.resetRebalanceWindow();
    auto second = ctrl.proposeReplicas(/*max_replicas_per_socket=*/1);
    EXPECT_TRUE(second.sameReplicaPlacement(first))
        << "an empty post-reset window is no evidence, not a replica eviction signal";
    EXPECT_EQ(ctrl.currentReplicas().num_replicated, first.num_replicated);
    EXPECT_EQ(ctrl.placementEpoch(), epoch_after_first);
}

TEST(Test__MoERebalanceController, ReplicaPrefillMaskKeepsReplicatedExpertsOnOwnerOnly)
{
    ExpertReplicaSet replicas;
    replicas.is_replicated = {true, false, false, false, true, false, false, false};
    replicas.owner_socket = {0, 0, 0, 0, 1, 1, 1, 1};
    replicas.num_replicated = 2;
    replicas.num_sockets = 2;

    std::vector<bool> socket0_mask = {true, true, true, true, true, false, false, false};
    std::vector<bool> socket1_mask = {true, false, false, false, true, true, true, true};

    auto socket0_replicas = replicas;
    socket0_replicas.buildPrefillMask(0, socket0_mask);
    EXPECT_TRUE(socket0_replicas.prefill_mask[0]);
    EXPECT_FALSE(socket0_replicas.prefill_mask[4]);
    EXPECT_TRUE(socket0_replicas.prefill_mask[1]);
    EXPECT_FALSE(socket0_replicas.prefill_mask[5]);

    auto socket1_replicas = replicas;
    socket1_replicas.buildPrefillMask(1, socket1_mask);
    EXPECT_FALSE(socket1_replicas.prefill_mask[0]);
    EXPECT_TRUE(socket1_replicas.prefill_mask[4]);
    EXPECT_FALSE(socket1_replicas.prefill_mask[1]);
    EXPECT_TRUE(socket1_replicas.prefill_mask[5]);
}

TEST(Test__MoERebalanceController, ReplicaPlacementComparisonIgnoresPrefillMask)
{
    ExpertReplicaSet a;
    a.is_replicated = {true, false, true, false};
    a.owner_socket = {0, 0, 1, 1};
    a.num_replicated = 2;
    a.num_sockets = 2;
    a.prefill_mask = {true, true, false, false};

    ExpertReplicaSet b = a;
    b.prefill_mask = {false, false, true, true};
    EXPECT_TRUE(a.sameReplicaPlacement(b));

    b.is_replicated[1] = true;
    b.num_replicated = 3;
    EXPECT_FALSE(a.sameReplicaPlacement(b));
}

TEST(Test__MoERebalanceController, ReplicaArrivalsSinceReturnsOnlyNewResidentExperts)
{
    ExpertReplicaSet previous;
    previous.is_replicated = {true, false, true, false, false, false};
    previous.owner_socket = {0, 0, 1, 1, 0, 1};
    previous.num_replicated = 2;
    previous.num_sockets = 2;

    ExpertReplicaSet current;
    current.is_replicated = {true, true, true, false, true, false};
    current.owner_socket = {0, 0, 0, 1, 0, 1};
    current.num_replicated = 4;
    current.num_sockets = 2;

    auto arrivals = current.arrivalsSince(previous);
    EXPECT_EQ(arrivals.num_replicated, 3);
    EXPECT_FALSE(arrivals.is_replicated[0]); // unchanged owner/socket residency
    EXPECT_TRUE(arrivals.is_replicated[1]);  // new replica
    EXPECT_TRUE(arrivals.is_replicated[2]);  // owner changed, must resend
    EXPECT_FALSE(arrivals.is_replicated[3]);
    EXPECT_TRUE(arrivals.is_replicated[4]); // new replica
    EXPECT_FALSE(arrivals.is_replicated[5]);
    EXPECT_EQ(arrivals.owner_socket, current.owner_socket);
    EXPECT_EQ(arrivals.num_sockets, current.num_sockets);
}

TEST(Test__MoERebalanceController, ReplicaArrivalsSinceUsesLayerParticipantResidency)
{
    ExpertReplicaSet previous;
    previous.owner_socket = {0, 1, 2, 0};
    previous.num_sockets = 3;
    previous.is_replicated.assign(4, false);
    previous.replica_participants_by_layer.assign(
        3,
        std::vector<std::vector<bool>>(4, std::vector<bool>(3, false)));
    previous.setReplicaOnParticipant(0, 1, 0);
    previous.setReplicaOnParticipant(1, 2, 0);
    previous.rebuildAggregateReplicaFlags();
    ASSERT_EQ(previous.num_replicated, 2);

    ExpertReplicaSet current = previous;
    current.setReplicaOnParticipant(0, 1, 2);
    current.setReplicaOnParticipant(2, 3, 1);
    current.rebuildAggregateReplicaFlags();
    ASSERT_EQ(current.num_replicated, 4);

    auto arrivals = current.arrivalsSince(previous);
    EXPECT_EQ(arrivals.num_replicated, 2);
    EXPECT_FALSE(arrivals.hasReplicaOnParticipant(0, 1, 0));
    EXPECT_TRUE(arrivals.hasReplicaOnParticipant(0, 1, 2));
    EXPECT_FALSE(arrivals.hasReplicaOnParticipant(1, 2, 0));
    EXPECT_TRUE(arrivals.hasReplicaOnParticipant(2, 3, 1));

    EXPECT_FALSE(arrivals.hasReplicaOnParticipant(1, 1, 2))
        << "arrival residency should not expand to unrelated layers";
    EXPECT_EQ(arrivals.owner_socket, current.owner_socket);
    EXPECT_EQ(arrivals.num_sockets, current.num_sockets);
}

TEST(Test__MoERebalanceController, ReplicaDecodeDispatchAssignsEachRoutedExpertToExactlyOneSocket)
{
    ExpertReplicaSet replicas;
    replicas.is_replicated = {true, true, false, false};
    replicas.owner_socket = {0, 1, 0, 1};
    replicas.num_replicated = 2;
    replicas.num_sockets = 2;

    const int expert_indices[] = {0, 1, 2, 3};
    const float expert_weights[] = {0.4f, 0.3f, 0.2f, 0.1f};
    const std::vector<bool> socket0_mask = {true, true, true, false};
    const std::vector<bool> socket1_mask = {true, true, false, true};

    bool socket0_compute[4] = {};
    bool socket1_compute[4] = {};
    replicas.assignForToken(expert_indices, expert_weights, 4, 0, socket0_mask, socket0_compute);
    replicas.assignForToken(expert_indices, expert_weights, 4, 1, socket1_mask, socket1_compute);

    int socket0_count = 0;
    int socket1_count = 0;
    for (int k = 0; k < 4; ++k)
    {
        EXPECT_NE(socket0_compute[k], socket1_compute[k]) << "expert " << expert_indices[k]
                                                          << " must be computed by exactly one socket";
        socket0_count += socket0_compute[k] ? 1 : 0;
        socket1_count += socket1_compute[k] ? 1 : 0;
    }

    EXPECT_EQ(socket0_count, 2);
    EXPECT_EQ(socket1_count, 2);
    EXPECT_TRUE(socket0_compute[0]);
    EXPECT_TRUE(socket1_compute[1]);
    EXPECT_TRUE(socket0_compute[2]);
    EXPECT_TRUE(socket1_compute[3]);
}

TEST(Test__MoERebalanceController, GpuCacheMasks_AssignsHottestExpertsToGpuDomain)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, /*num_experts=*/8, /*num_sockets=*/4,
                          /*num_layers=*/1, /*top_k=*/1, /*window_size=*/32);
    cfg.sockets = {DeviceId::rocm(0), DeviceId::rocm(1), DeviceId::cpu(), DeviceId(DeviceType::CPU, 1)};
    MoERebalanceController ctrl(cfg);

    recordExpertHits(*ctrl.histogram(), 0, {{3, 12}, {5, 9}, {0, 6}, {7, 2}});

    auto masks = ctrl.computeGpuCacheExpertMasks(/*gpu_cache_experts_per_layer=*/2);
    ASSERT_EQ(static_cast<int>(masks.size()), 4);
    ASSERT_EQ(static_cast<int>(masks[0].size()), 1);
    ASSERT_EQ(static_cast<int>(masks[0][0].size()), 8);

    EXPECT_TRUE(anyGpuSocketHas(masks, 0, 3));
    EXPECT_TRUE(anyGpuSocketHas(masks, 0, 5));
    EXPECT_FALSE(anyCpuSocketHas(masks, 0, 3));
    EXPECT_FALSE(anyCpuSocketHas(masks, 0, 5));

    EXPECT_TRUE(anyCpuSocketHas(masks, 0, 0));
    EXPECT_TRUE(anyCpuSocketHas(masks, 0, 7));
    EXPECT_FALSE(anyGpuSocketHas(masks, 0, 0));
    EXPECT_FALSE(anyGpuSocketHas(masks, 0, 7));
}

TEST(Test__MoERebalanceController, GpuCacheMasks_SpreadsHotExpertsAcrossGpuSockets)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, /*num_experts=*/8, /*num_sockets=*/4,
                          /*num_layers=*/1, /*top_k=*/1, /*window_size=*/32);
    cfg.sockets = {DeviceId::rocm(0), DeviceId::rocm(1), DeviceId::cpu(), DeviceId(DeviceType::CPU, 1)};
    MoERebalanceController ctrl(cfg);

    recordExpertHits(*ctrl.histogram(), 0, {{0, 10}, {1, 10}, {2, 10}, {3, 10}});

    auto masks = ctrl.computeGpuCacheExpertMasks(/*gpu_cache_experts_per_layer=*/4);
    int gpu0_count = 0;
    int gpu1_count = 0;
    for (int e = 0; e < 4; ++e)
    {
        gpu0_count += masks[0][0][e] ? 1 : 0;
        gpu1_count += masks[1][0][e] ? 1 : 0;
    }

    EXPECT_EQ(gpu0_count, 2);
    EXPECT_EQ(gpu1_count, 2);
}

TEST(Test__MoERebalanceController, GpuCacheMasks_CanPlaceAllExpertsOnGpuDomain)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, /*num_experts=*/4, /*num_sockets=*/3,
                          /*num_layers=*/1, /*top_k=*/1, /*window_size=*/16);
    cfg.sockets = {DeviceId::rocm(0), DeviceId::cpu(), DeviceId(DeviceType::CPU, 1)};
    MoERebalanceController ctrl(cfg);

    recordExpertHits(*ctrl.histogram(), 0, {{0, 4}, {1, 3}, {2, 2}, {3, 1}});

    auto masks = ctrl.computeGpuCacheExpertMasks(/*gpu_cache_experts_per_layer=*/4);
    for (int e = 0; e < 4; ++e)
    {
        EXPECT_TRUE(masks[0][0][e]) << "expert " << e << " should be cached on GPU";
        EXPECT_FALSE(masks[1][0][e]);
        EXPECT_FALSE(masks[2][0][e]);
    }
}

TEST(Test__MoERebalanceController, GpuCacheMasks_FallsBackWithoutMixedDomains)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, /*num_experts=*/8, /*num_sockets=*/2,
                          /*num_layers=*/2, /*top_k=*/1, /*window_size=*/16);
    MoERebalanceController ctrl(cfg);

    recordExpertHits(*ctrl.histogram(), 0, {{0, 10}, {1, 8}});

    auto expected_s0 = ctrl.computeExpertMasks(0);
    auto expected_s1 = ctrl.computeExpertMasks(1);
    auto masks = ctrl.computeGpuCacheExpertMasks(/*gpu_cache_experts_per_layer=*/2);

    ASSERT_EQ(static_cast<int>(masks.size()), 2);
    EXPECT_EQ(masks[0], expected_s0);
    EXPECT_EQ(masks[1], expected_s1);
}

// ── rebalanceLPT() Tests ──────────────────────────────

TEST(Test__MoERebalanceController, RebalanceLPT_BalancedInput_NearPerfectBalance)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    MoERebalanceController ctrl(cfg);

    fillWindowBalanced(*ctrl.histogram(), 16, 2, 8, 2);
    ctrl.rebalanceLPT();

    // Verify near-perfect balance: compute load per socket
    // Re-fill histogram to measure post-LPT imbalance
    fillWindowBalanced(*ctrl.histogram(), 16, 2, 8, 2);

    const auto &placement = ctrl.currentPlacement();
    // Count experts per socket
    int count_s0 = 0, count_s1 = 0;
    for (int e = 0; e < 8; ++e)
    {
        if (placement[e] == 0)
            count_s0++;
        else
            count_s1++;
    }
    // With balanced input, LPT should assign 4 experts per socket
    EXPECT_EQ(count_s0, 4);
    EXPECT_EQ(count_s1, 4);
}

TEST(Test__MoERebalanceController, RebalanceLPT_SkewedInput_ImproveBalance)
{
    // Round-robin: experts 0,2,4,6→socket0, experts 1,3,5,7→socket1
    // Skewed routing to experts 0,1 → socket0 gets expert 0, socket1 gets expert 1
    // Both are equally hot, so initially balanced. Use contiguous partition instead.
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    // Contiguous: 0-3→socket0, 4-7→socket1
    for (int e = 0; e < 8; ++e)
        cfg.initial_expert_to_socket[e] = (e < 4) ? 0 : 1;
    MoERebalanceController ctrl(cfg);

    // Route exclusively to experts 0,1 (both on socket 0)
    fillWindowSkewed(*ctrl.histogram(), 16, 2, 2);

    ctrl.rebalanceLPT();

    // LPT should move one of {0,1} to socket 1 for better balance
    const auto &placement = ctrl.currentPlacement();
    int hot_on_s0 = 0;
    for (int e : {0, 1})
    {
        if (placement[e] == 0)
            hot_on_s0++;
    }
    // Expect at most 1 hot expert per socket (LPT splits them)
    EXPECT_LE(hot_on_s0, 1);
}

TEST(Test__MoERebalanceController, RebalanceLPT_UpdatesPlacement)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    // Contiguous: 0-3→socket0, 4-7→socket1
    for (int e = 0; e < 8; ++e)
        cfg.initial_expert_to_socket[e] = (e < 4) ? 0 : 1;
    MoERebalanceController ctrl(cfg);

    auto initial = ctrl.currentPlacement();

    // Skewed to experts 0,1 (both on socket 0) → LPT should change placement
    fillWindowSkewed(*ctrl.histogram(), 16, 2, 2);
    ctrl.rebalanceLPT();

    const auto &updated = ctrl.currentPlacement();
    ASSERT_EQ(static_cast<int>(updated.size()), 8);

    // At least one expert should have moved
    bool any_changed = false;
    for (int e = 0; e < 8; ++e)
    {
        if (updated[e] != initial[e])
            any_changed = true;
    }
    EXPECT_TRUE(any_changed);
}

TEST(Test__MoERebalanceController, RebalanceLPT_ResetsWindow)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    MoERebalanceController ctrl(cfg);

    fillWindowBalanced(*ctrl.histogram(), 16, 2, 8, 2);
    ASSERT_TRUE(ctrl.histogram()->windowFull());

    ctrl.rebalanceLPT();

    EXPECT_FALSE(ctrl.histogram()->windowFull());
}

TEST(Test__MoERebalanceController, RebalanceLPT_IncrementsRebalanceCount)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    MoERebalanceController ctrl(cfg);

    EXPECT_EQ(ctrl.totalRebalances(), 0);

    fillWindowBalanced(*ctrl.histogram(), 16, 2, 8, 2);
    ctrl.rebalanceLPT();
    EXPECT_EQ(ctrl.totalRebalances(), 1);

    // Second cycle
    fillWindowBalanced(*ctrl.histogram(), 16, 2, 8, 2);
    ctrl.rebalanceLPT();
    EXPECT_EQ(ctrl.totalRebalances(), 2);
}

TEST(Test__MoERebalanceController, RebalanceLPT_ContiguousPartition_SkewedRouting)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    // Contiguous: 0-3→socket0, 4-7→socket1
    for (int e = 0; e < 8; ++e)
        cfg.initial_expert_to_socket[e] = (e < 4) ? 0 : 1;
    MoERebalanceController ctrl(cfg);

    // Route only to experts 0-3 (all on socket 0)
    fillWindowWithExperts(*ctrl.histogram(), 16, 2, {0, 1, 2, 3});

    ctrl.rebalanceLPT();

    const auto &placement = ctrl.currentPlacement();
    // LPT should distribute experts 0-3 across both sockets
    int hot_on_s0 = 0, hot_on_s1 = 0;
    for (int e = 0; e < 4; ++e)
    {
        if (placement[e] == 0)
            hot_on_s0++;
        else
            hot_on_s1++;
    }
    // At least one of experts 0-3 should now be on socket 1
    EXPECT_GT(hot_on_s1, 0);
    // Should be roughly balanced: 2 hot experts per socket
    EXPECT_EQ(hot_on_s0, 2);
    EXPECT_EQ(hot_on_s1, 2);
}

TEST(Test__MoERebalanceController, RebalanceLPT_MultiSocket)
{
    // 4 sockets, 16 experts
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, /*num_experts=*/16, /*num_sockets=*/4,
                          /*num_layers=*/2, /*top_k=*/4, /*window_size=*/16);
    MoERebalanceController ctrl(cfg);

    // Route to experts 0-3 (all on socket 0 in round-robin with 4 sockets)
    fillWindowWithExperts(*ctrl.histogram(), 16, 2, {0, 1, 2, 3});

    ctrl.rebalanceLPT();

    const auto &placement = ctrl.currentPlacement();
    // LPT should spread experts 0-3 across all 4 sockets
    std::vector<int> socket_for_hot(4);
    for (int e = 0; e < 4; ++e)
        socket_for_hot[e] = placement[e];

    // With 4 equally-loaded hot experts and 4 sockets, each should go to a different socket
    std::sort(socket_for_hot.begin(), socket_for_hot.end());
    // All 4 hot experts on distinct sockets
    EXPECT_EQ(socket_for_hot[0], 0);
    EXPECT_EQ(socket_for_hot[1], 1);
    EXPECT_EQ(socket_for_hot[2], 2);
    EXPECT_EQ(socket_for_hot[3], 3);
}

TEST(Test__MoERebalanceController, RebalanceLPT_ZeroCountExperts)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    MoERebalanceController ctrl(cfg);

    // Only experts 0,1 are active — experts 2-7 have zero counts
    fillWindowWithExperts(*ctrl.histogram(), 16, 2, {0, 1});

    ctrl.rebalanceLPT();

    const auto &placement = ctrl.currentPlacement();

    // All 8 experts should still have valid placements (0 or 1)
    for (int e = 0; e < 8; ++e)
    {
        EXPECT_GE(placement[e], 0);
        EXPECT_LE(placement[e], 1);
    }

    // LPT balances by LOAD, not expert count.
    // The 2 hot experts should be split across sockets for load balance.
    int hot_on_s0 = 0, hot_on_s1 = 0;
    for (int e : {0, 1})
    {
        if (placement[e] == 0)
            hot_on_s0++;
        else
            hot_on_s1++;
    }
    EXPECT_EQ(hot_on_s0, 1);
    EXPECT_EQ(hot_on_s1, 1);

    // Zero-count experts all tie at load=0, so LPT tie-breaks to lowest-index socket.
    // They don't affect balance — just verify they're all placed somewhere valid.
    for (int e = 2; e < 8; ++e)
    {
        EXPECT_GE(placement[e], 0);
        EXPECT_LE(placement[e], 1);
    }
}

// ── computeExpertMasks() Tests ────────────────────────

TEST(Test__MoERebalanceController, ComputeExpertMasks_InitialPartition)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    MoERebalanceController ctrl(cfg);

    // Round-robin: experts 0,2,4,6→socket0; experts 1,3,5,7→socket1
    auto masks_s0 = ctrl.computeExpertMasks(0);
    auto masks_s1 = ctrl.computeExpertMasks(1);

    ASSERT_EQ(static_cast<int>(masks_s0.size()), 2);    // num_layers
    ASSERT_EQ(static_cast<int>(masks_s0[0].size()), 8); // num_experts

    // Check socket 0 mask matches round-robin
    for (int l = 0; l < 2; ++l)
    {
        EXPECT_TRUE(masks_s0[l][0]);  // expert 0 → socket 0
        EXPECT_FALSE(masks_s0[l][1]); // expert 1 → socket 1
        EXPECT_TRUE(masks_s0[l][2]);  // expert 2 → socket 0
        EXPECT_FALSE(masks_s0[l][3]); // expert 3 → socket 1
        EXPECT_TRUE(masks_s0[l][4]);  // expert 4 → socket 0
        EXPECT_FALSE(masks_s0[l][5]); // expert 5 → socket 1
        EXPECT_TRUE(masks_s0[l][6]);  // expert 6 → socket 0
        EXPECT_FALSE(masks_s0[l][7]); // expert 7 → socket 1
    }

    // Check socket 1 is the complement
    for (int l = 0; l < 2; ++l)
    {
        EXPECT_FALSE(masks_s1[l][0]);
        EXPECT_TRUE(masks_s1[l][1]);
        EXPECT_FALSE(masks_s1[l][2]);
        EXPECT_TRUE(masks_s1[l][3]);
        EXPECT_FALSE(masks_s1[l][4]);
        EXPECT_TRUE(masks_s1[l][5]);
        EXPECT_FALSE(masks_s1[l][6]);
        EXPECT_TRUE(masks_s1[l][7]);
    }
}

TEST(Test__MoERebalanceController, ComputeExpertMasks_AfterLPT)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    // Contiguous: 0-3→socket0, 4-7→socket1
    for (int e = 0; e < 8; ++e)
        cfg.initial_expert_to_socket[e] = (e < 4) ? 0 : 1;
    MoERebalanceController ctrl(cfg);

    // Skewed to experts 0,1 → LPT will move one to socket 1
    fillWindowSkewed(*ctrl.histogram(), 16, 2, 2);
    ctrl.rebalanceLPT();

    const auto &placement = ctrl.currentPlacement();
    auto masks_s0 = ctrl.computeExpertMasks(0);

    // Masks should reflect the updated LPT placement
    for (int l = 0; l < 2; ++l)
    {
        for (int e = 0; e < 8; ++e)
        {
            EXPECT_EQ(masks_s0[l][e], placement[e] == 0)
                << "Mismatch at layer=" << l << " expert=" << e;
        }
    }
}

TEST(Test__MoERebalanceController, ComputeExpertMasks_SocketComplementary)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    MoERebalanceController ctrl(cfg);

    fillWindowBalanced(*ctrl.histogram(), 16, 2, 8, 2);
    ctrl.rebalanceLPT();

    auto masks_s0 = ctrl.computeExpertMasks(0);
    auto masks_s1 = ctrl.computeExpertMasks(1);

    for (int l = 0; l < 2; ++l)
    {
        for (int e = 0; e < 8; ++e)
        {
            // Exactly one socket owns each expert (no overlap, no gap)
            EXPECT_NE(masks_s0[l][e], masks_s1[l][e])
                << "Overlap or gap at layer=" << l << " expert=" << e;
        }
    }
}

TEST(Test__MoERebalanceController, ComputeExpertMasks_AllLayersSameGlobalPartition)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, /*num_layers=*/4, 2, 16);
    MoERebalanceController ctrl(cfg);

    fillWindowBalanced(*ctrl.histogram(), 16, 4, 8, 2);
    ctrl.rebalanceLPT();

    auto masks = ctrl.computeExpertMasks(0);
    ASSERT_EQ(static_cast<int>(masks.size()), 4);

    // All layers should have identical masks (global LPT, not per-layer)
    for (int l = 1; l < 4; ++l)
    {
        EXPECT_EQ(masks[l], masks[0])
            << "Layer " << l << " mask differs from layer 0";
    }
}
