/**
 * @file Test__MoERebalanceController.cpp
 * @brief Unit tests for MoERebalanceController
 */

#include <gtest/gtest.h>
#include "execution/moe/DeviceMoERebalanceABI.h"
#include "execution/moe/DeviceMoERebalanceController.h"
#include "execution/moe/DeviceMoERebalancePolicyShared.h"
#include "execution/moe/MoERebalanceController.h"
#include <algorithm>
#include <array>
#include <initializer_list>
#include <iterator>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace llaminar2;

/** @brief Replace a controller fixture's complete ownership with one static row. */
static void setUniformOwnership(
    MoERebalanceController::Config &config,
    const std::vector<int> &owners)
{
    config.initial_ownership = MoELayeredExpertOwnership::uniform(
        config.num_layers,
        static_cast<int>(config.sockets.size()),
        owners);
}

/**
 * @brief Build a complete replica set from explicit layer/expert destinations.
 *
 * Every tuple is `{layer, expert, participant}`. Owner residency is implicit
 * in `base_ownership`; only non-owner copies belong in the replica matrix.
 */
static ExpertReplicaSet makeReplicaSet(
    MoELayeredExpertOwnership base_ownership,
    std::initializer_list<std::array<int, 3>> replicas,
    std::string domain_id = {})
{
    ExpertReplicaSet result;
    result.domain_id = std::move(domain_id);
    result.base_ownership = std::move(base_ownership);
    result.replica_participants_by_layer.assign(
        static_cast<size_t>(result.base_ownership.layerCount()),
        std::vector<std::vector<bool>>(
            static_cast<size_t>(result.base_ownership.expertCount()),
            std::vector<bool>(
                static_cast<size_t>(result.base_ownership.participantCount()),
                false)));
    for (const auto &replica : replicas)
    {
        result.setReplicaOnParticipant(
            replica[0], replica[1], replica[2]);
    }
    result.rebuildAggregateReplicaFlags();
    return result;
}

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

    // Repeat one round-robin static owner row across all routed layers.
    std::vector<int> owners(static_cast<size_t>(num_experts));
    for (int e = 0; e < num_experts; ++e)
        owners[static_cast<size_t>(e)] = e % num_sockets;
    setUniformOwnership(cfg, owners);

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

/// Fill the histogram window with routing concentrated on participant zero.
static void fillWindowSkewed(DecodeExpertHistogram &hist, int window_size,
                             int num_layers, int top_k)
{
    for (int t = 0; t < window_size; ++t)
    {
        for (int l = 0; l < num_layers; ++l)
        {
            // Under the default round-robin row, experts 0,2,4,... all belong
            // to participant zero. The old 0,1 fixture was accidentally
            // balanced and therefore failed to exercise Dynamic ownership.
            std::vector<int> indices(top_k);
            std::vector<float> weights(top_k);
            for (int k = 0; k < top_k; ++k)
            {
                indices[k] = k * 2;
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

/**
 * @brief Prove the hosted ticket admits exactly the three legal clock states.
 *
 * HIP's bound can observe a pending non-due edge, a pending due edge, or an
 * already-acknowledged idle controller.  The last state is expected when the
 * first generated token is sampled from prefill logits and therefore commits
 * no new model-state row.  A due-but-unadvanced state remains impossible.
 */
TEST(Test__MoERebalanceController,
     DispatchTicketAuthenticatesAcknowledgedIdleCadence)
{
    constexpr uint64_t session_epoch = 7u;
    constexpr uint64_t workspace_generation = 13u;
    DeviceMoERebalanceDispatchTicket ticket;
    ticket.magic = DeviceMoERebalanceDispatchTicket::kMagic;
    ticket.abi_version = DeviceMoERebalanceDispatchTicket::kABIVersion;
    ticket.session_epoch_low = static_cast<uint32_t>(session_epoch);
    ticket.workspace_generation_low =
        static_cast<uint32_t>(workspace_generation);
    ticket.participant_id = 0u;
    ticket.participant_count = 2u;
    ticket.healthy = 1u;
    ticket.controller_version = moe_rebalance_abi::kVersion;
    ticket.decode_rounds_committed = 63u;
    ticket.decode_rounds_until_maintenance = 1u;
    ticket.maintenance_due = 0u;
    ticket.decode_boundary_advanced = 0u;

    EXPECT_TRUE(ticket.matchesLifecycle(
        session_epoch,
        workspace_generation,
        /*expected_participant_id=*/0u,
        /*expected_participant_count=*/2u));
    EXPECT_EQ(
        ticket.dispatchAction(),
        DeviceMoERebalanceDispatchAction::None);

    ticket.decode_boundary_advanced = 1u;
    EXPECT_TRUE(ticket.matchesLifecycle(
        session_epoch,
        workspace_generation,
        0u,
        2u));
    EXPECT_EQ(
        ticket.dispatchAction(),
        DeviceMoERebalanceDispatchAction::Acknowledge);

    ticket.maintenance_due = 1u;
    ticket.decode_rounds_until_maintenance = 0u;
    EXPECT_TRUE(ticket.matchesLifecycle(
        session_epoch,
        workspace_generation,
        0u,
        2u));
    EXPECT_EQ(
        ticket.dispatchAction(),
        DeviceMoERebalanceDispatchAction::Maintain);

    ticket.decode_boundary_advanced = 0u;
    EXPECT_FALSE(ticket.matchesLifecycle(
        session_epoch,
        workspace_generation,
        0u,
        2u));
    EXPECT_EQ(
        ticket.dispatchAction(),
        DeviceMoERebalanceDispatchAction::Invalid);

    ticket.maintenance_due = 0u;
    EXPECT_FALSE(ticket.matchesLifecycle(
        session_epoch,
        workspace_generation,
        0u,
        2u));
    EXPECT_EQ(
        ticket.dispatchAction(),
        DeviceMoERebalanceDispatchAction::Invalid);
}

/**
 * @brief Lock the generation-stamped transfer-slot transaction into the ABI.
 *
 * Destination projection turns a logical root command into a participant-local
 * compare-and-replace transaction. The invalid prior identity is meaningful for
 * an empty slot, while generation zero is the first valid directory generation.
 * These defaults must therefore remain explicit and trivially transportable
 * through device command buffers.
 */
TEST(Test__MoERebalanceController,
     DevicePlanAbiCarriesAuthenticatedTransferSlotLease)
{
    static_assert(std::is_trivially_copyable_v<DeviceMoERebalancePlanEntry>);
    EXPECT_EQ(kDeviceMoERebalanceVersion, 11u);
    EXPECT_EQ(
        sizeof(DeviceMoERebalanceConfig),
        moe_rebalance_abi::kConfigBytes);

    const DeviceMoERebalancePlanEntry plan{};
    EXPECT_EQ(plan.destination_slot, kDeviceMoEInvalidSlot);
    EXPECT_EQ(plan.payload_slot, kDeviceMoEInvalidSlot);
    EXPECT_EQ(plan.destination_previous_layer, kDeviceMoEInvalidSlot);
    EXPECT_EQ(plan.destination_previous_expert, kDeviceMoEInvalidSlot);
    EXPECT_EQ(plan.destination_generation, 0u);
}

/**
 * @brief Prove active transfer-backed descriptors have unique physical slots.
 *
 * The maintenance controller exports this summary at request boundaries. The
 * regression deliberately gives two layers the same physical slot so a future
 * allocator change cannot silently restore the stale-runtime alias that made
 * prefill-published experts unsafe for later decode.
 */
TEST(Test__MoERebalanceController,
     TransferSlotClaimSummaryReportsCrossLayerAlias)
{
    DeviceMoERebalanceConfig config;
    config.num_layers = 2;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 0;
    config.participant_count = 2;

    std::array<DeviceMoELayerRuntime, 2> runtime_layers{
        makeDeviceRuntimeLayerForLoadStats(),
        makeDeviceRuntimeLayerForLoadStats()};

    constexpr uint32_t transfer_flags =
        toMoEExpertFlags(DeviceMoEExpertFlags::Valid) |
        toMoEExpertFlags(DeviceMoEExpertFlags::Resident) |
        toMoEExpertFlags(DeviceMoEExpertFlags::LocalCompute) |
        toMoEExpertFlags(DeviceMoEExpertFlags::TransferSlot);
    auto publish_transfer_claim =
        [&](uint32_t layer, uint32_t expert, int32_t slot)
    {
        auto &bank = runtime_layers[layer].banks[0];
        bank.experts[expert].local_slot = slot;
        bank.experts[expert].flags = transfer_flags;
        bank.local_compute_mask[expert] = 1u;
        bank.resident_participant_mask[expert] = 0b01u;
    };

    publish_transfer_claim(/*layer=*/0, /*expert=*/0, /*slot=*/3);
    publish_transfer_claim(/*layer=*/1, /*expert=*/2, /*slot=*/3);

    auto summary =
        deviceMoETransferSlotClaimSummary(runtime_layers.data(), config);
    EXPECT_EQ(summary.active_claims, 2u);
    EXPECT_EQ(summary.unique_claims, 1u);
    EXPECT_EQ(summary.duplicate_claims, 1u);
    EXPECT_EQ(summary.invalid_claims, 0u);
    EXPECT_EQ(summary.max_slot, 3u);
    EXPECT_EQ(summary.max_slot_layer, 0u);
    EXPECT_EQ(summary.max_slot_expert, 0u);
    EXPECT_EQ(summary.first_duplicate_slot, 3u);
    EXPECT_EQ(summary.first_duplicate_layer, 1u);
    EXPECT_EQ(summary.first_duplicate_expert, 2u);

    /*
     * Physical storage remains live even when the current grouped assignment
     * gives the expert no rows. Claim accounting must not use LocalCompute as a
     * proxy for transfer-slot occupancy.
     */
    auto &idle_claim =
        runtime_layers[1].banks[0].experts[2];
    idle_claim.flags &=
        ~toMoEExpertFlags(DeviceMoEExpertFlags::LocalCompute);
    runtime_layers[1].banks[0].local_compute_mask[2] = 0u;
    publish_transfer_claim(/*layer=*/1, /*expert=*/2, /*slot=*/4);
    idle_claim.flags &=
        ~toMoEExpertFlags(DeviceMoEExpertFlags::LocalCompute);
    runtime_layers[1].banks[0].local_compute_mask[2] = 0u;
    summary = deviceMoETransferSlotClaimSummary(runtime_layers.data(), config);
    EXPECT_EQ(summary.active_claims, 2u);
    EXPECT_EQ(summary.unique_claims, 2u);
    EXPECT_EQ(summary.duplicate_claims, 0u);
    EXPECT_EQ(summary.invalid_claims, 0u);
    EXPECT_EQ(summary.max_slot, 4u);
    EXPECT_EQ(summary.max_slot_layer, 1u);
    EXPECT_EQ(summary.max_slot_expert, 2u);

    publish_transfer_claim(/*layer=*/1, /*expert=*/2, /*slot=*/-1);
    summary = deviceMoETransferSlotClaimSummary(runtime_layers.data(), config);
    EXPECT_EQ(summary.active_claims, 2u);
    EXPECT_EQ(summary.unique_claims, 1u);
    EXPECT_EQ(summary.duplicate_claims, 0u);
    EXPECT_EQ(summary.invalid_claims, 1u);
    EXPECT_EQ(summary.first_invalid_slot, kDeviceMoEInvalidSlot);
    EXPECT_EQ(summary.first_invalid_layer, 1u);
    EXPECT_EQ(summary.first_invalid_expert, 2u);
    EXPECT_NE(
        summary.first_invalid_reasons &
            moe_rebalance_policy::TransferSlotClaimNegativeSlot,
        0u);
    EXPECT_EQ(summary.first_invalid_resident_mask, 0b01u);

    publish_transfer_claim(/*layer=*/1, /*expert=*/2, /*slot=*/4);
    config.active_transfer_slot_capacity = 4u;
    config.transfer_slot_directory_capacity = 4u;
    summary = deviceMoETransferSlotClaimSummary(
        runtime_layers.data(),
        config);
    EXPECT_EQ(summary.invalid_claims, 1u);
    EXPECT_EQ(summary.first_invalid_slot, 4u);
    EXPECT_NE(
        summary.first_invalid_reasons &
            moe_rebalance_policy::
                TransferSlotClaimExceedsDirectoryCapacity,
        0u);
}

/**
 * @brief Prove directory addressability is independent of active occupancy.
 *
 * A directory with one durable claim and one staging entry may publish its
 * next durable resident into slot 1 after retiring slot 0. The resulting one
 * active claim is valid even though its physical index is equal to the active
 * claim capacity. Slot 2 remains invalid because it is outside the directory.
 */
TEST(Test__MoERebalanceController,
     TransferSlotClaimSummaryAcceptsPromotedStagingOriginSlot)
{
    DeviceMoERebalanceConfig config;
    config.num_layers = 1;
    config.num_experts = 4;
    config.top_k = 2;
    config.participant_id = 0;
    config.participant_count = 2;
    config.window_size_tokens = 1;
    config.active_transfer_slot_capacity = 1;
    config.transfer_slot_directory_capacity = 2;
    ASSERT_TRUE(validateDeviceMoERebalanceConfig(config));

    auto runtime = makeDeviceRuntimeLayerForLoadStats();
    auto &bank = runtime.banks[runtime.active_bank];
    auto &descriptor = bank.experts[2];
    descriptor.local_slot = 1;
    descriptor.owner_participant = 1;
    descriptor.flags =
        toMoEExpertFlags(DeviceMoEExpertFlags::Valid) |
        toMoEExpertFlags(DeviceMoEExpertFlags::Resident) |
        toMoEExpertFlags(DeviceMoEExpertFlags::TransferSlot);
    bank.resident_participant_mask[2] = 0b11u;

    auto summary =
        deviceMoETransferSlotClaimSummary(&runtime, config);
    EXPECT_EQ(summary.active_claims, 1u);
    EXPECT_EQ(summary.unique_claims, 1u);
    EXPECT_EQ(summary.invalid_claims, 0u);
    EXPECT_EQ(summary.max_slot, 1u);

    descriptor.local_slot = 2;
    summary = deviceMoETransferSlotClaimSummary(&runtime, config);
    EXPECT_EQ(summary.active_claims, 1u);
    EXPECT_EQ(summary.unique_claims, 0u);
    EXPECT_EQ(summary.invalid_claims, 1u);
    EXPECT_NE(
        summary.first_invalid_reasons &
            moe_rebalance_policy::
                TransferSlotClaimExceedsDirectoryCapacity,
        0u);

    config.active_transfer_slot_capacity = 2;
    config.transfer_slot_directory_capacity = 1;
    EXPECT_FALSE(validateDeviceMoERebalanceConfig(config));
}

/**
 * @brief Prove payload retirement clears every pointer-bearing publication.
 */
TEST(Test__MoERebalanceController,
     LocalPayloadRetirementIsOneAtomicStateTransition)
{
    DeviceMoEExpertDescriptor descriptor;
    descriptor.gate.payload =
        reinterpret_cast<const uint8_t *>(0x1000u);
    descriptor.up.payload =
        reinterpret_cast<const uint8_t *>(0x2000u);
    descriptor.down.payload =
        reinterpret_cast<const uint8_t *>(0x3000u);
    descriptor.logical_expert_id = 17;
    descriptor.owner_participant = 1;
    descriptor.local_slot = 9;
    descriptor.flags =
        toMoEExpertFlags(DeviceMoEExpertFlags::Valid) |
        toMoEExpertFlags(DeviceMoEExpertFlags::Resident) |
        toMoEExpertFlags(DeviceMoEExpertFlags::Replicated) |
        toMoEExpertFlags(DeviceMoEExpertFlags::LocalCompute) |
        toMoEExpertFlags(DeviceMoEExpertFlags::TransferSlot) |
        toMoEExpertFlags(DeviceMoEExpertFlags::PreferredOwner);
    uint32_t resident_mask = 0b11u;
    constexpr uint32_t local_payload_flags =
        toMoEExpertFlags(DeviceMoEExpertFlags::Valid) |
        toMoEExpertFlags(DeviceMoEExpertFlags::Resident) |
        toMoEExpertFlags(DeviceMoEExpertFlags::Replicated) |
        toMoEExpertFlags(DeviceMoEExpertFlags::LocalCompute) |
        toMoEExpertFlags(DeviceMoEExpertFlags::TransferSlot);

    moe_rebalance_policy::retireLocalPayloadPublication(
        descriptor,
        resident_mask,
        /*local_participant_bit=*/0b01u,
        local_payload_flags);

    EXPECT_EQ(resident_mask, 0b10u);
    EXPECT_EQ(descriptor.gate.payload, nullptr);
    EXPECT_EQ(descriptor.up.payload, nullptr);
    EXPECT_EQ(descriptor.down.payload, nullptr);
    EXPECT_EQ(descriptor.local_slot, -1);
    EXPECT_EQ(descriptor.logical_expert_id, 17);
    EXPECT_EQ(descriptor.owner_participant, 1);
    EXPECT_EQ(
        descriptor.flags & local_payload_flags,
        0u);
    EXPECT_TRUE(hasMoEExpertFlag(
        descriptor.flags,
        DeviceMoEExpertFlags::PreferredOwner));
}

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

TEST(Test__MoERebalanceController, TransferSlotStorageLifetimeIsIndependentOfComputeEligibility)
{
    constexpr uint32_t local_participant = 1u;
    const uint32_t local_bit =
        moe_rebalance_policy::participantBit(local_participant);

    /*
     * This is the production failure shape: the local participant owns a
     * transfer-backed expert, but the current least-loaded assignment gives it
     * no local rows. Compute eligibility is intentionally not an input to the
     * policy; ownership alone keeps the allocation live and non-evictable.
     */
    const auto authoritative_without_local_work =
        moe_rebalance_policy::classifyTransferSlotOccupancy(
            /*transfer_backed=*/true,
            /*assigned_locally=*/false,
            /*resident_mask=*/local_bit,
            /*local_participant_bit=*/local_bit,
            /*owner_participant=*/static_cast<int32_t>(local_participant),
            local_participant);
    EXPECT_TRUE(authoritative_without_local_work.occupied);
    EXPECT_TRUE(authoritative_without_local_work.protected_from_reuse);

    /*
     * Fail closed when owner and residency metadata momentarily disagree. The
     * allocator must not destroy authoritative bytes; the runtime status gate
     * can then report and terminate the invalid publication.
     */
    const auto authoritative_with_stale_resident_mask =
        moe_rebalance_policy::classifyTransferSlotOccupancy(
            /*transfer_backed=*/true,
            /*assigned_locally=*/false,
            /*resident_mask=*/0u,
            /*local_participant_bit=*/local_bit,
            /*owner_participant=*/static_cast<int32_t>(local_participant),
            local_participant);
    EXPECT_TRUE(authoritative_with_stale_resident_mask.occupied);
    EXPECT_TRUE(authoritative_with_stale_resident_mask.protected_from_reuse);

    const auto cold_cached_replica =
        moe_rebalance_policy::classifyTransferSlotOccupancy(
            /*transfer_backed=*/true,
            /*assigned_locally=*/false,
            /*resident_mask=*/local_bit,
            /*local_participant_bit=*/local_bit,
            /*owner_participant=*/0,
            local_participant);
    EXPECT_TRUE(cold_cached_replica.occupied);
    EXPECT_FALSE(cold_cached_replica.protected_from_reuse);

    const auto assigned_cached_replica =
        moe_rebalance_policy::classifyTransferSlotOccupancy(
            /*transfer_backed=*/true,
            /*assigned_locally=*/true,
            /*resident_mask=*/local_bit,
            /*local_participant_bit=*/local_bit,
            /*owner_participant=*/0,
            local_participant);
    EXPECT_TRUE(assigned_cached_replica.occupied);
    EXPECT_TRUE(assigned_cached_replica.protected_from_reuse);

    const auto stale_remote_descriptor =
        moe_rebalance_policy::classifyTransferSlotOccupancy(
            /*transfer_backed=*/true,
            /*assigned_locally=*/false,
            /*resident_mask=*/moe_rebalance_policy::participantBit(0),
            /*local_participant_bit=*/local_bit,
            /*owner_participant=*/0,
            local_participant);
    EXPECT_FALSE(stale_remote_descriptor.occupied);
    EXPECT_FALSE(stale_remote_descriptor.protected_from_reuse);

    const auto native_weight_descriptor =
        moe_rebalance_policy::classifyTransferSlotOccupancy(
            /*transfer_backed=*/false,
            /*assigned_locally=*/true,
            /*resident_mask=*/local_bit,
            /*local_participant_bit=*/local_bit,
            /*owner_participant=*/static_cast<int32_t>(local_participant),
            local_participant);
    EXPECT_FALSE(native_weight_descriptor.occupied);
    EXPECT_FALSE(native_weight_descriptor.protected_from_reuse);
}

TEST(Test__MoERebalanceController, PrefillTransferSlotLifetimeFollowsForwardLayerOrdering)
{
    /*
     * Same-layer replicas remain live when the device-owned grouped
     * assignment sends rows to this participant. This is the only case where
     * assignment liveness protects a non-owner transfer slot.
     */
    EXPECT_TRUE(
        moe_rebalance_policy::prefillAssignmentReadsTransferSlotOccupant(
            /*prepared_layer=*/5u,
            /*occupant_layer=*/5u,
            /*same_layer_assigned_locally=*/true));
    EXPECT_FALSE(
        moe_rebalance_policy::prefillAssignmentReadsTransferSlotOccupant(
            /*prepared_layer=*/5u,
            /*occupant_layer=*/5u,
            /*same_layer_assigned_locally=*/false));

    /*
     * A completed earlier layer and a not-yet-executed later layer cannot be
     * current consumers at layer 5. The explicit compute-to-transfer event
     * orders earlier reads before replacement, while later layers republish
     * their own assignment and payload before executing.
     */
    EXPECT_FALSE(
        moe_rebalance_policy::prefillAssignmentReadsTransferSlotOccupant(
            /*prepared_layer=*/5u,
            /*occupant_layer=*/4u,
            /*same_layer_assigned_locally=*/true));
    EXPECT_FALSE(
        moe_rebalance_policy::prefillAssignmentReadsTransferSlotOccupant(
            /*prepared_layer=*/5u,
            /*occupant_layer=*/6u,
            /*same_layer_assigned_locally=*/true));
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
        moe_rebalance_policy::DynamicOwnershipEvidenceWindow{
            .routed_activations = 110,
            .minimum_routed_activations = 64,
        },
        /*imbalance_threshold_per_mille=*/1300,
        /*min_improvement_per_mille=*/50);

    ASSERT_TRUE(choice.valid);
    EXPECT_EQ(choice.overloaded_participant, 0u);
    EXPECT_EQ(choice.underloaded_participant, 1u);
    EXPECT_EQ(choice.heavy_expert, 1u);
    EXPECT_EQ(choice.light_expert, 2u);
    EXPECT_EQ(choice.heavy_count, 30u);
    EXPECT_EQ(choice.light_count, 1u);
    EXPECT_EQ(choice.old_min_load, 10u);
    EXPECT_EQ(choice.old_max_load, 100u);
    EXPECT_EQ(choice.new_min_load, 39u);
    EXPECT_EQ(choice.new_max_load, 71u);

    ASSERT_TRUE(moe_rebalance_policy::applyDynamicOwnershipSwap(
        participant_load,
        expert_owner,
        choice));
    EXPECT_EQ(participant_load[0], 71u);
    EXPECT_EQ(participant_load[1], 39u);
    EXPECT_EQ(expert_owner[1], 1);
    EXPECT_EQ(expert_owner[2], 0);
}

TEST(Test__MoERebalanceController, SharedDynamicPolicySearchesPastExtremaOvershoot)
{
    uint64_t participant_load[2] = {18u, 12u};
    uint64_t expert_counts[6] = {10u, 8u, 0u, 7u, 5u, 0u};
    int32_t expert_owner[6] = {0, 0, 0, 1, 1, 1};

    /*
     * The old hottest-for-coldest heuristic tried 10 <-> 0, producing
     * 8 versus 22, and then rejected the whole layer.  The exact selector must
     * find 10 <-> 7 and reach the optimal 15/15 ownership instead.
     */
    const auto choice = moe_rebalance_policy::bestDynamicOwnershipSwap(
        participant_load,
        expert_counts,
        expert_owner,
        /*num_experts=*/6,
        /*participant_count=*/2,
        moe_rebalance_policy::DynamicOwnershipEvidenceWindow{
            .routed_activations = 30,
            .minimum_routed_activations = 1,
        },
        /*imbalance_threshold_per_mille=*/1300,
        /*min_improvement_per_mille=*/50);

    ASSERT_TRUE(choice.valid);
    EXPECT_EQ(choice.heavy_expert, 0u);
    EXPECT_EQ(choice.light_expert, 3u);
    EXPECT_EQ(choice.new_min_load, 15u);
    EXPECT_EQ(choice.new_max_load, 15u);
    ASSERT_TRUE(moe_rebalance_policy::applyDynamicOwnershipSwap(
        participant_load,
        expert_owner,
        choice));
    EXPECT_EQ(participant_load[0], 15u);
    EXPECT_EQ(participant_load[1], 15u);
}

TEST(Test__MoERebalanceController, SharedDynamicPolicyRejectsLayerSwapThatWorsensWaveLoad)
{
    uint64_t aggregate_load[2] = {1924u, 2140u};
    moe_rebalance_policy::OwnershipSwapChoice choice{};
    choice.overloaded_participant = 0u;
    choice.underloaded_participant = 1u;
    choice.heavy_count = 30u;
    choice.light_count = 5u;
    choice.valid = true;

    const auto delta =
        moe_rebalance_policy::evaluateDynamicOwnershipSwapAgainstAggregateLoad(
            aggregate_load,
            2u,
            choice);
    EXPECT_TRUE(delta.total_preserved);
    EXPECT_EQ(delta.current_spread, 216u);
    EXPECT_EQ(delta.proposed_spread, 266u);
    EXPECT_FALSE(delta.improves)
        << "A locally overloaded layer must not push work toward the participant already overloaded across the wave.";
    EXPECT_FALSE(
        moe_rebalance_policy::applyDynamicOwnershipSwapToAggregateLoadIfImproved(
            aggregate_load,
            2u,
            choice));
    EXPECT_EQ(aggregate_load[0], 1924u);
    EXPECT_EQ(aggregate_load[1], 2140u);
}

TEST(Test__MoERebalanceController, SharedDynamicPolicyStopsBeforeAggregateOvershoot)
{
    uint64_t aggregate_load[2] = {100u, 0u};
    moe_rebalance_policy::OwnershipSwapChoice choice{};
    choice.overloaded_participant = 0u;
    choice.underloaded_participant = 1u;
    choice.heavy_count = 35u;
    choice.light_count = 5u;
    choice.valid = true;

    uint64_t improvement = 0u;
    ASSERT_TRUE(
        moe_rebalance_policy::applyDynamicOwnershipSwapToAggregateLoadIfImproved(
            aggregate_load,
            2u,
            choice,
            &improvement));
    EXPECT_EQ(aggregate_load[0], 70u);
    EXPECT_EQ(aggregate_load[1], 30u);
    EXPECT_EQ(improvement, 60u);

    ASSERT_TRUE(
        moe_rebalance_policy::applyDynamicOwnershipSwapToAggregateLoadIfImproved(
            aggregate_load,
            2u,
            choice,
            &improvement));
    EXPECT_EQ(aggregate_load[0], 40u);
    EXPECT_EQ(aggregate_load[1], 60u);
    EXPECT_EQ(improvement, 20u);

    EXPECT_FALSE(
        moe_rebalance_policy::applyDynamicOwnershipSwapToAggregateLoadIfImproved(
            aggregate_load,
            2u,
            choice,
            &improvement));
    EXPECT_EQ(aggregate_load[0], 40u);
    EXPECT_EQ(aggregate_load[1], 60u);
    EXPECT_EQ(improvement, 0u);
}

TEST(Test__MoERebalanceController, SharedDynamicPolicyChoosesCapacityReleasingSwap)
{
    uint64_t participant_load[2] = {100u, 10u};
    uint64_t expert_counts[4] = {70u, 30u, 1u, 9u};
    int32_t expert_owner[4] = {0, 0, 1, 1};
    uint32_t transfer_backed_mask[4] = {
        0u,
        moe_rebalance_policy::participantBit(0),
        0u,
        moe_rebalance_policy::participantBit(1),
    };
    uint32_t active_transfer_slots[2] = {1u, 1u};

    const auto choice = moe_rebalance_policy::bestDynamicOwnershipSwap(
        participant_load,
        expert_counts,
        expert_owner,
        /*num_experts=*/4,
        /*participant_count=*/2,
        moe_rebalance_policy::DynamicOwnershipEvidenceWindow{
            .routed_activations = 110,
            .minimum_routed_activations = 64,
        },
        /*imbalance_threshold_per_mille=*/1300,
        /*min_improvement_per_mille=*/50,
        transfer_backed_mask,
        active_transfer_slots,
        /*active_transfer_slot_capacity=*/1);

    ASSERT_TRUE(choice.valid);
    EXPECT_EQ(choice.heavy_expert, 1u)
        << "A full overloaded participant must send transfer-backed storage.";
    EXPECT_EQ(choice.light_expert, 3u)
        << "A full underloaded participant must also release its transfer slot.";

    ASSERT_TRUE(
        moe_rebalance_policy::applyDynamicOwnershipSwapTransferOccupancy(
            active_transfer_slots,
            transfer_backed_mask,
            choice));
    EXPECT_EQ(active_transfer_slots[0], 1u);
    EXPECT_EQ(active_transfer_slots[1], 1u);
    EXPECT_EQ(
        transfer_backed_mask[1],
        moe_rebalance_policy::participantBit(1));
    EXPECT_EQ(
        transfer_backed_mask[3],
        moe_rebalance_policy::participantBit(0));
}

TEST(Test__MoERebalanceController, CollectedStatePreservesCountAndPhysicalEvidence)
{
    const uint64_t packed = moe_rebalance_policy::packCollectedState(
        123456789u,
        39u,
        /*physically_resident=*/true,
        /*transfer_backed=*/true);

    EXPECT_EQ(
        moe_rebalance_policy::collectedStateActivationCount(packed),
        123456789u);
    EXPECT_EQ(
        moe_rebalance_policy::collectedStateActiveTransferSlots(packed),
        39u);
    EXPECT_TRUE(
        moe_rebalance_policy::collectedStatePhysicallyResident(packed));
    EXPECT_TRUE(
        moe_rebalance_policy::collectedStateTransferBacked(packed));
}

/**
 * @brief Prove that a static source cannot grow an already-full destination.
 *
 * Static model storage is not counted in the transfer-slot occupancy vector.
 * Moving its ownership therefore needs one new durable destination slot. The
 * planner must reject that move before publishing a transaction when the
 * destination has no remaining durable capacity.
 */
TEST(Test__MoERebalanceController,
     OwnershipTransferRejectsStaticSourceWhenDestinationCapacityIsFull)
{
    const uint32_t active_transfer_slots[2] = {0u, 1u};
    const uint32_t transfer_backed_masks[2] = {0u, 0u};

    EXPECT_FALSE(
        moe_rebalance_policy::ownershipTransferFitsPersistentCapacity(
            active_transfer_slots,
            transfer_backed_masks,
            /*expert=*/0u,
            /*source_participant=*/0u,
            /*destination_participant=*/1u,
            /*participant_count=*/2u,
            /*active_transfer_slot_capacity=*/1u));
}

/**
 * @brief Prove that moving transfer-backed ownership preserves bounded storage.
 *
 * A transfer-backed source releases one durable claim in the same immutable
 * ownership transaction that installs the destination claim. The planner's
 * scratch accounting must model both sides so later commands in the captured
 * wave observe the projected occupancy rather than stale pre-wave counts.
 */
TEST(Test__MoERebalanceController,
     OwnershipTransferMovesDurableOccupancyBetweenParticipants)
{
    uint32_t active_transfer_slots[2] = {1u, 0u};
    uint32_t transfer_backed_masks[1] = {
        moe_rebalance_policy::participantBit(0u)};

    ASSERT_TRUE(
        moe_rebalance_policy::ownershipTransferFitsPersistentCapacity(
            active_transfer_slots,
            transfer_backed_masks,
            /*expert=*/0u,
            /*source_participant=*/0u,
            /*destination_participant=*/1u,
            /*participant_count=*/2u,
            /*active_transfer_slot_capacity=*/1u));
    ASSERT_TRUE(moe_rebalance_policy::applyOwnershipTransferOccupancy(
        active_transfer_slots,
        transfer_backed_masks,
        /*expert=*/0u,
        /*source_participant=*/0u,
        /*destination_participant=*/1u));

    EXPECT_EQ(active_transfer_slots[0], 0u);
    EXPECT_EQ(active_transfer_slots[1], 1u);
    EXPECT_EQ(
        transfer_backed_masks[0],
        moe_rebalance_policy::participantBit(1u));
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
    EXPECT_EQ(bank.multi_resident_expert_count, 1u)
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
    EXPECT_EQ(ctrl.totalSwapPairs(), 0);
    EXPECT_EQ(ctrl.totalOwnershipChanges(), 0);
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
    MoERebalanceController ctrl(cfg);
    const auto initial = ctrl.currentOwnership();

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
    const auto replicas = ctrl.proposeReplicas(/*max_replicas_per_socket=*/1);

    EXPECT_EQ(ctrl.currentOwnership(), initial);
    EXPECT_EQ(ctrl.placementEpoch(), 0u);
    EXPECT_EQ(ctrl.totalRebalances(), 0);
    EXPECT_EQ(replicas.num_replicated, 0);
}

TEST(Test__MoERebalanceController, ParticipantVocabularyAliasesAreConsistent)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, /*num_experts=*/6, /*num_sockets=*/3,
                          /*num_layers=*/2, /*top_k=*/2, /*window_size=*/16);
    MoERebalanceController ctrl(cfg);

    EXPECT_EQ(ctrl.participantCount(), 3);
    EXPECT_EQ(ctrl.participantDevices(), cfg.sockets);
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
    std::vector<int> owners(8);
    for (int e = 0; e < 8; ++e)
        owners[static_cast<size_t>(e)] = (e < 6) ? 0 : 1;
    setUniformOwnership(cfg, owners);

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
    const auto changes = swap_ctrl.rebalance();
    ASSERT_FALSE(changes.empty());
    EXPECT_NE(swap_ctrl.currentOwnership(), cfg.initial_ownership);
    EXPECT_TRUE(
        swap_ctrl.currentOwnership().hasSameLayerCapacitiesAs(
            cfg.initial_ownership));
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
    std::vector<int> owners(8);
    for (int e = 0; e < 8; ++e)
        owners[static_cast<size_t>(e)] = (e < 6) ? 0 : 1;
    setUniformOwnership(cfg, owners);
    MoERebalanceController ctrl(cfg);

    // Skewed routing → heavy imbalance between sockets
    fillWindowSkewed(*ctrl.histogram(), 16, 2, 2);
    ASSERT_TRUE(ctrl.shouldRebalance());

    auto result = ctrl.rebalance();

    // With heavy imbalance, the rebalancer returns exact changed entries.
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.size() % 2, 0u);
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
    std::vector<int> owners(8);
    for (int e = 0; e < 8; ++e)
        owners[static_cast<size_t>(e)] = (e < 6) ? 0 : 1;
    setUniformOwnership(cfg, owners);
    MoERebalanceController ctrl(cfg);

    const auto initial = ctrl.currentOwnership();

    fillWindowSkewed(*ctrl.histogram(), 16, 2, 2);
    auto result = ctrl.rebalance();

    if (!result.empty())
    {
        // Pair swaps move a heavy expert to socket 1 and a light expert to socket 0.
        // At least one expert's socket assignment should differ from initial.
        EXPECT_NE(ctrl.currentOwnership(), initial);
        EXPECT_TRUE(
            ctrl.currentOwnership().hasSameLayerCapacitiesAs(initial));
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
    std::vector<int> owners(8);
    for (int e = 0; e < 8; ++e)
        owners[static_cast<size_t>(e)] = (e < 6) ? 0 : 1;
    setUniformOwnership(cfg, owners);
    MoERebalanceController ctrl(cfg);

    EXPECT_EQ(ctrl.totalRebalances(), 0);
    EXPECT_EQ(ctrl.totalSwapPairs(), 0);

    // First rebalance cycle
    fillWindowSkewed(*ctrl.histogram(), 16, 2, 2);
    auto result1 = ctrl.rebalance();

    if (!result1.empty())
    {
        EXPECT_EQ(ctrl.totalRebalances(), 1);
        EXPECT_GT(ctrl.totalSwapPairs(), 0);
        const int swaps_after_first = ctrl.totalSwapPairs();

        // Second rebalance cycle
        fillWindowSkewed(*ctrl.histogram(), 16, 2, 2);
        auto result2 = ctrl.rebalance();

        if (!result2.empty())
        {
            EXPECT_EQ(ctrl.totalRebalances(), 2);
            EXPECT_GE(ctrl.totalSwapPairs(), swaps_after_first);
        }
    }
}

TEST(Test__MoERebalanceController, ObserveMode_NeverRebalances)
{
    // Experts 0-5 on socket 0, 6-7 on socket 1 for extreme imbalance
    auto cfg = makeConfig(MoERebalanceMode::OBSERVE, 8, 2, 2, 2, 16);
    std::vector<int> owners(8);
    for (int e = 0; e < 8; ++e)
        owners[static_cast<size_t>(e)] = (e < 6) ? 0 : 1;
    setUniformOwnership(cfg, owners);
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
    setUniformOwnership(cfg, {0, 0, 0, 0, 1, 1, 1, 1});

    MoERebalanceController ctrl(cfg);
    EXPECT_EQ(ctrl.placementEpoch(), 0u);

    fillWindowSkewed(*ctrl.histogram(), 16, 2, 2);
    const auto rebalanced = ctrl.rebalance();
    ASSERT_FALSE(rebalanced.empty());
    EXPECT_EQ(ctrl.placementEpoch(), 1u);

    const auto &owners_after_rebalance =
        ctrl.currentOwnership().ownersForLayer(0);
    auto hot_expert = std::find(
        owners_after_rebalance.begin(), owners_after_rebalance.end(), 0);
    ASSERT_NE(hot_expert, owners_after_rebalance.end());
    recordExpertHits(
        *ctrl.histogram(),
        0,
        {{static_cast<int>(std::distance(owners_after_rebalance.begin(), hot_expert)), 16}});
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
    setUniformOwnership(cfg, {0, 0, 0, 0, 1, 1, 1, 1});

    MoERebalanceController ctrl(cfg);
    const auto initial_ownership = ctrl.currentOwnership();

    recordExpertHits(*ctrl.histogram(), 0, {{0, 20}});
    recordExpertHits(*ctrl.histogram(), 1, {{4, 20}});
    ASSERT_TRUE(ctrl.shouldRebalance());

    auto replicas = ctrl.proposeReplicas(/*max_replicas_per_socket=*/1);
    ASSERT_EQ(replicas.num_replicated, 2);
    EXPECT_TRUE(replicas.replicated_in_any_layer[0]);
    EXPECT_TRUE(replicas.replicated_in_any_layer[4]);
    EXPECT_TRUE(replicas.hasLayerReplicaPlacement());
    EXPECT_TRUE(replicas.hasReplicaOnParticipant(0, 0, 1));
    EXPECT_TRUE(replicas.hasReplicaOnParticipant(1, 4, 0));
    EXPECT_FALSE(replicas.hasReplicaOnParticipant(0, 4, 0))
        << "a hot expert id must not implicitly expand to every layer";
    EXPECT_FALSE(replicas.hasReplicaOnParticipant(1, 0, 1))
        << "a hot expert id must not implicitly expand to every layer";
    EXPECT_EQ(ctrl.currentOwnership(), initial_ownership);

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
    EXPECT_EQ(ctrl.currentOwnership(), initial_ownership);
}

TEST(Test__MoERebalanceController, ReplicaProposalSkipsHotRemoteExpertWhenTargetIsAlreadyHeavier)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, /*num_experts=*/4, /*num_sockets=*/2,
                          /*num_layers=*/1, /*top_k=*/1, /*window_size=*/16);
    MoERebalanceController ctrl(cfg);

    recordExpertHits(*ctrl.histogram(), 0, {{0, 100}, {1, 90}});

    auto replicas = ctrl.proposeReplicasForParticipants(/*max_replicas_per_participant=*/1);
    ASSERT_EQ(replicas.participantCount(), 2);
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
    ASSERT_EQ(replicas.participantCount(), 3);
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
    ASSERT_TRUE(first.replicated_in_any_layer[1]);

    ctrl.resetRebalanceWindow();
    recordExpertHits(*ctrl.histogram(), 0, {{1, 8}, {3, 10}});
    auto second = ctrl.proposeReplicas(/*max_replicas_per_socket=*/1);
    EXPECT_EQ(second.num_replicated, 1);
    EXPECT_TRUE(second.replicated_in_any_layer[1])
        << "existing replica remains within 50% of the hottest replacement";
    EXPECT_FALSE(second.replicated_in_any_layer[3]);
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
    ASSERT_TRUE(first.replicated_in_any_layer[1]);

    ctrl.resetRebalanceWindow();
    recordExpertHits(*ctrl.histogram(), 0, {{1, 4}, {3, 10}});
    auto second = ctrl.proposeReplicas(/*max_replicas_per_socket=*/1);
    EXPECT_EQ(second.num_replicated, 1);
    EXPECT_FALSE(second.replicated_in_any_layer[1]);
    EXPECT_TRUE(second.replicated_in_any_layer[3])
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
    ASSERT_TRUE(first.replicated_in_any_layer[1]);
    const auto epoch_after_first = ctrl.placementEpoch();

    ctrl.resetRebalanceWindow();
    auto second = ctrl.proposeReplicas(/*max_replicas_per_socket=*/1);
    EXPECT_TRUE(second.sameReplicaPlacement(first))
        << "an empty post-reset window is no evidence, not a replica eviction signal";
    EXPECT_EQ(ctrl.currentReplicas().num_replicated, first.num_replicated);
    EXPECT_EQ(ctrl.placementEpoch(), epoch_after_first);
}

TEST(Test__MoERebalanceController, ReplicaPrefillMaskKeepsReplicatedComputeOnOwnerOnly)
{
    const auto ownership = MoELayeredExpertOwnership::uniform(
        1, 2, {0, 0, 0, 0, 1, 1, 1, 1});
    const auto replicas = makeReplicaSet(
        ownership,
        {
            {0, 0, 1},
            {0, 4, 0},
        });

    std::vector<bool> socket0_mask = {true, true, true, true, true, false, false, false};
    std::vector<bool> socket1_mask = {true, false, false, false, true, true, true, true};

    auto socket0_replicas = replicas;
    socket0_replicas.buildPrefillMask(0, socket0_mask, 0);
    EXPECT_TRUE(socket0_replicas.prefill_mask[0]);
    EXPECT_FALSE(socket0_replicas.prefill_mask[4]);
    EXPECT_TRUE(socket0_replicas.prefill_mask[1]);
    EXPECT_FALSE(socket0_replicas.prefill_mask[5]);

    auto socket1_replicas = replicas;
    socket1_replicas.buildPrefillMask(1, socket1_mask, 0);
    EXPECT_FALSE(socket1_replicas.prefill_mask[0]);
    EXPECT_TRUE(socket1_replicas.prefill_mask[4]);
    EXPECT_FALSE(socket1_replicas.prefill_mask[1]);
    EXPECT_TRUE(socket1_replicas.prefill_mask[5]);
}

TEST(Test__MoERebalanceController, ReplicaPlacementComparisonIgnoresPrefillMask)
{
    const auto ownership = MoELayeredExpertOwnership::uniform(
        1, 2, {0, 0, 1, 1});
    auto a = makeReplicaSet(
        ownership,
        {
            {0, 0, 1},
            {0, 2, 0},
        });
    a.prefill_mask = {true, true, false, false};

    ExpertReplicaSet b = a;
    b.prefill_mask = {false, false, true, true};
    EXPECT_TRUE(a.sameReplicaPlacement(b));

    b.setReplicaOnParticipant(0, 1, 1);
    b.rebuildAggregateReplicaFlags();
    EXPECT_FALSE(a.sameReplicaPlacement(b));
}

TEST(Test__MoERebalanceController, ReplicaArrivalsSinceReturnsOnlyNewResidentExperts)
{
    const auto previous_ownership = MoELayeredExpertOwnership::uniform(
        1, 2, {0, 0, 1, 1, 0, 1});
    const auto previous = makeReplicaSet(
        previous_ownership,
        {
            {0, 0, 1},
            {0, 2, 0},
        });

    const auto current_ownership = MoELayeredExpertOwnership::uniform(
        1, 2, {0, 0, 0, 1, 0, 1});
    const auto current = makeReplicaSet(
        current_ownership,
        {
            {0, 0, 1},
            {0, 1, 1},
            {0, 2, 1},
            {0, 4, 1},
        });

    auto arrivals = current.arrivalsSince(previous);
    EXPECT_EQ(arrivals.num_replicated, 3);
    EXPECT_FALSE(arrivals.replicated_in_any_layer[0]); // unchanged owner/socket residency
    EXPECT_TRUE(arrivals.replicated_in_any_layer[1]);  // new replica
    EXPECT_TRUE(arrivals.replicated_in_any_layer[2]);  // owner changed, must resend
    EXPECT_FALSE(arrivals.replicated_in_any_layer[3]);
    EXPECT_TRUE(arrivals.replicated_in_any_layer[4]); // new replica
    EXPECT_FALSE(arrivals.replicated_in_any_layer[5]);
    EXPECT_EQ(arrivals.base_ownership, current.base_ownership);
    EXPECT_EQ(arrivals.participantCount(), current.participantCount());
}

TEST(Test__MoERebalanceController, ReplicaArrivalsSinceUsesLayerParticipantResidency)
{
    const auto ownership = MoELayeredExpertOwnership::uniform(
        3, 3, {0, 1, 2, 0});
    const auto previous = makeReplicaSet(
        ownership,
        {
            {0, 1, 0},
            {1, 2, 0},
        });
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
    EXPECT_EQ(arrivals.base_ownership, current.base_ownership);
    EXPECT_EQ(arrivals.participantCount(), current.participantCount());
}

TEST(Test__MoERebalanceController, ReplicaDecodeDispatchAssignsEachRoutedExpertToExactlyOneSocket)
{
    const auto replicas = makeReplicaSet(
        MoELayeredExpertOwnership::uniform(1, 2, {0, 1, 0, 1}),
        {
            {0, 0, 1},
            {0, 1, 0},
        });

    const int expert_indices[] = {0, 1, 2, 3};
    const float expert_weights[] = {0.4f, 0.3f, 0.2f, 0.1f};
    const std::vector<bool> socket0_mask = {true, true, true, false};
    const std::vector<bool> socket1_mask = {true, true, false, true};

    bool socket0_compute[4] = {};
    bool socket1_compute[4] = {};
    replicas.assignForToken(
        expert_indices, expert_weights, 4, 0, socket0_mask, socket0_compute, 0);
    replicas.assignForToken(
        expert_indices, expert_weights, 4, 1, socket1_mask, socket1_compute, 0);

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

TEST(Test__MoERebalanceController, ReplicaDecodeDispatchIgnoresZeroWeightPadding)
{
    const auto replicas = makeReplicaSet(
        MoELayeredExpertOwnership::uniform(1, 2, {0, 1, 0, 1}),
        {
            {0, 0, 1},
            {0, 1, 0},
        });

    const int expert_indices[] = {0, -1, 2, 3};
    const float expert_weights[] = {0.4f, 0.0f, 0.2f, 0.1f};
    const std::vector<bool> socket0_mask = {true, true, true, false};
    const std::vector<bool> socket1_mask = {true, true, false, true};

    bool socket0_compute[] = {true, true, true, true};
    bool socket1_compute[] = {true, true, true, true};
    replicas.assignForToken(
        expert_indices,
        expert_weights,
        4,
        0,
        socket0_mask,
        socket0_compute,
        0);
    replicas.assignForToken(
        expert_indices,
        expert_weights,
        4,
        1,
        socket1_mask,
        socket1_compute,
        0);

    EXPECT_FALSE(socket0_compute[1]);
    EXPECT_FALSE(socket1_compute[1]);
    for (const int active_slot : {0, 2, 3})
    {
        EXPECT_NE(socket0_compute[active_slot], socket1_compute[active_slot])
            << "active slot " << active_slot
            << " must still be assigned to exactly one participant";
    }
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

TEST(Test__MoERebalanceController, GpuCacheMasksPreserveOwnershipWithoutMixedDomains)
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

TEST(Test__MoERebalanceController, ComputeExpertMasksCoverEveryOwnerExactlyOnce)
{
    auto cfg = makeConfig(
        MoERebalanceMode::DYNAMIC,
        /*num_experts=*/8,
        /*num_sockets=*/3,
        /*num_layers=*/4,
        /*top_k=*/2,
        /*window_size=*/16);
    MoERebalanceController controller(cfg);

    std::vector<std::vector<std::vector<bool>>> masks;
    for (int participant = 0; participant < controller.participantCount(); ++participant)
        masks.push_back(controller.computeExpertMasks(participant));

    for (int layer = 0; layer < controller.numLayers(); ++layer)
    {
        for (int expert = 0; expert < controller.numExperts(); ++expert)
        {
            int owner_count = 0;
            for (int participant = 0;
                 participant < controller.participantCount();
                 ++participant)
            {
                owner_count += masks[static_cast<size_t>(participant)]
                                    [static_cast<size_t>(layer)]
                                    [static_cast<size_t>(expert)]
                                   ? 1
                                   : 0;
            }
            EXPECT_EQ(owner_count, 1)
                << "layer=" << layer << " expert=" << expert;
        }
    }
}

TEST(Test__MoERebalanceController, DynamicOwnershipKeepsConflictingLayerMoves)
{
    auto cfg = makeConfig(
        MoERebalanceMode::DYNAMIC,
        /*num_experts=*/4,
        /*num_sockets=*/2,
        /*num_layers=*/2,
        /*top_k=*/1,
        /*window_size=*/1);
    cfg.initial_ownership = MoELayeredExpertOwnership(
        2,
        {
            {0, 0, 1, 1},
            {1, 1, 0, 0},
        });
    cfg.rebalance_config.imbalance_threshold = 1.01f;
    cfg.rebalance_config.min_improvement_ratio = 0.0f;
    cfg.rebalance_config.max_swaps_per_layer = 1;
    cfg.rebalance_config.max_total_swaps = 4;
    MoERebalanceController controller(cfg);

    // Each layer has an 80/20 participant split. The opposing owner rows make
    // expert zero's beneficial move point in opposite directions by layer.
    const uint64_t counts[] = {40, 40, 20, 0};
    controller.histogram()->mergeLayerCounts(0, counts, 4, false);
    controller.histogram()->mergeLayerCounts(1, counts, 4, true);
    ASSERT_TRUE(controller.shouldRebalance());

    const auto changes = controller.rebalance();
    ASSERT_EQ(changes.size(), 4u);
    EXPECT_EQ(controller.currentOwnership().owner(0, 0), 1);
    EXPECT_EQ(controller.currentOwnership().owner(1, 0), 0);
    EXPECT_TRUE(
        controller.currentOwnership().hasSameLayerCapacitiesAs(
            cfg.initial_ownership));

    bool saw_layer_zero = false;
    bool saw_layer_one = false;
    for (const auto &change : changes)
    {
        saw_layer_zero |= change.layer_idx == 0 && change.expert_id == 0;
        saw_layer_one |= change.layer_idx == 1 && change.expert_id == 0;
    }
    EXPECT_TRUE(saw_layer_zero);
    EXPECT_TRUE(saw_layer_one);

    ASSERT_TRUE(controller.lastImbalanceBefore().valid);
    ASSERT_TRUE(controller.lastImbalanceAfter().valid);
    EXPECT_LT(
        controller.lastImbalanceAfter().average_spread,
        controller.lastImbalanceBefore().average_spread);
    EXPECT_LE(
        controller.lastImbalanceAfter().worst_spread,
        controller.lastImbalanceBefore().worst_spread);
}

TEST(Test__MoERebalanceController, DynamicMasksReflectLayerSpecificOwners)
{
    auto cfg = makeConfig(
        MoERebalanceMode::DYNAMIC,
        /*num_experts=*/4,
        /*num_sockets=*/2,
        /*num_layers=*/2,
        /*top_k=*/1,
        /*window_size=*/16);
    cfg.initial_ownership = MoELayeredExpertOwnership(
        2,
        {
            {0, 0, 1, 1},
            {1, 1, 0, 0},
        });
    MoERebalanceController controller(cfg);

    const auto participant_zero = controller.computeExpertMasks(0);
    const auto participant_one = controller.computeExpertMasks(1);
    ASSERT_EQ(participant_zero.size(), 2u);
    ASSERT_EQ(participant_one.size(), 2u);

    EXPECT_TRUE(participant_zero[0][0]);
    EXPECT_FALSE(participant_one[0][0]);
    EXPECT_FALSE(participant_zero[1][0]);
    EXPECT_TRUE(participant_one[1][0]);

    for (int layer = 0; layer < 2; ++layer)
    {
        for (int expert = 0; expert < 4; ++expert)
        {
            EXPECT_NE(
                participant_zero[static_cast<size_t>(layer)]
                                [static_cast<size_t>(expert)],
                participant_one[static_cast<size_t>(layer)]
                               [static_cast<size_t>(expert)]);
        }
    }
}
