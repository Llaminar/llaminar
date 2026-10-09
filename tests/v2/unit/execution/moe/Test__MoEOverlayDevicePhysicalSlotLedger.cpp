/**
 * @file Test__MoEOverlayDevicePhysicalSlotLedger.cpp
 * @brief Device-free adversarial tests for physical Dynamic slot lifetimes.
 *
 * Shared inert engine handles prove stage identity, publication overlap and
 * retirement without initializing a GPU or reading weight payloads. Equal
 * transaction numbers on another stage cannot release the retained engines.
 */

#include <gtest/gtest.h>

#include "execution/moe/MoEOverlayDevicePhysicalSlotLedger.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        /** Create a non-dereferenced shared engine identity for lifetime tests. */
        std::shared_ptr<ITensorGemm> engineIdentity()
        {
            auto token = std::make_shared<std::uint64_t>(0xfeedbeefu);
            return std::shared_ptr<ITensorGemm>(
                token,
                reinterpret_cast<ITensorGemm *>(token.get()));
        }

        /** Create three independently retained prepared projection identities. */
        MoEOverlayPreparedExpertPayload payload()
        {
            return {
                engineIdentity(),
                engineIdentity(),
                engineIdentity(),
            };
        }

        /** Resolve a minimal exact GPU owner for one physical test endpoint. */
        MoEExpertOwner owner(
            int participant,
            int layer,
            int expert)
        {
            return {
                .layer_idx = layer,
                .expert_id = expert,
                .tier_idx = 0,
                .owner_participant = participant,
                .device = DeviceId::cuda(participant),
                .resident = true,
                .tier_name = "integer_priority_0",
                .domain_name = "test",
                .domain_participant_index = participant,
                .owner_world_rank = 0,
                .owner_world_rank_known = true,
                .address = GlobalDeviceAddress::cuda(participant),
            };
        }

        /** Build one closed two-edge device batch with exact shadow demand. */
        MoEOverlayDevicePhysicalMovementBatch batch(
            std::uint64_t transaction,
            std::uint64_t base_epoch,
            int first_expert,
            int second_expert,
            int first_model_layer = 0)
        {
            std::vector<MoEOverlayTierMigration> migrations{
                {
                    .layer_idx = first_model_layer,
                    .expert_id = first_expert,
                    .estimated_weight_bytes = 128u,
                    .direction =
                        MoEOverlayTierMigrationDirection::SamePriority,
                    .source = owner(0, first_model_layer, first_expert),
                    .destination = owner(1, first_model_layer, first_expert),
                },
                {
                    .layer_idx = first_model_layer,
                    .expert_id = second_expert,
                    .estimated_weight_bytes = 128u,
                    .direction =
                        MoEOverlayTierMigrationDirection::SamePriority,
                    .source = owner(1, first_model_layer, second_expert),
                    .destination = owner(0, first_model_layer, second_expert),
                },
            };
            return {
                .kind =
                    MoEOverlayDeviceControllerTransactionKind::
                        DynamicPlacement,
                .topology_fingerprint = 0x12345678u,
                .transaction_id = transaction,
                .base_epoch = base_epoch,
                .candidate_epoch = base_epoch + 1u,
                .command_digest = 0x87654321u + transaction,
                .packed_weight_bytes = 256u,
                .command_count = 2u,
                .participant_count = 2u,
                .num_layers = 1u,
                .num_experts = 8u,
                .execution_fingerprint = {
                    .low = 0xa5a50000u + transaction,
                    .high = 0x5a5a0000u + transaction,
                },
                .migrations = std::move(migrations),
                .migration_cycles = {{
                    .layer_idx = first_model_layer,
                    .migration_indices = {0u, 1u},
                }},
                .shadow_requirements = {
                    {
                        .layer_idx = first_model_layer,
                        .tier_idx = 0,
                        .destination_participant = 0,
                        .slot_count = 1u,
                    },
                    {
                        .layer_idx = first_model_layer,
                        .tier_idx = 0,
                        .destination_participant = 1,
                        .slot_count = 1u,
                    },
                },
                .first_model_layer = first_model_layer,
            };
        }

        /** Construct the two bootstrap allocations named by the first wave. */
        MoEOverlayDevicePhysicalSlotLedger ledger(int first_model_layer = 0)
        {
            std::vector<MoEOverlayDeviceInitialPhysicalSlot> initial;
            initial.push_back({
                .key = {.participant_id = 0, .layer_idx = first_model_layer, .expert_id = 0},
                .entered_epoch = 1u,
                .bootstrap_allocation = true,
                .payload = payload(),
            });
            initial.push_back({
                .key = {.participant_id = 1, .layer_idx = first_model_layer, .expert_id = 1},
                .entered_epoch = 1u,
                .bootstrap_allocation = true,
                .payload = payload(),
            });
            return MoEOverlayDevicePhysicalSlotLedger({
                .initial_epoch = 1u,
                .local_participant_ids = {0, 1},
                .initial_slots = std::move(initial),
                .first_model_layer = first_model_layer,
                .num_layers = 1u,
                .num_experts = 8u,
            });
        }
    } // namespace

    /** @brief Gate/up movement uses the same publication/retirement state machine as full experts. */
    TEST(Test__MoEOverlayDevicePhysicalSlotLedger, GateUpLifetimeAndCrossFamilyRejection)
    {
        const auto pair = [] { return MoEOverlayPreparedExpertPayload::gateUp(engineIdentity(), engineIdentity()); };
        MoEOverlayDevicePhysicalSlotLedger slots({
            .initial_epoch = 1u, .local_participant_ids = {0, 1},
            .initial_slots = {
                {.key = {0, 0, 0}, .entered_epoch = 1u, .payload = pair()},
                {.key = {1, 0, 1}, .entered_epoch = 1u, .payload = pair()}},
            .movable_projections = DeviceMoEProjectionSet::GateUp,
            .num_layers = 1u, .num_experts = 8u});
        auto movement = batch(51u, 1u, 0, 1);
        std::string error;
        ASSERT_TRUE(slots.begin(movement, &error)) << error;
        auto source = slots.sourcePayload(movement, {0, 0, 0}, &error);
        ASSERT_TRUE(source);
        EXPECT_FALSE(source->complete());
        EXPECT_EQ(source->down(), nullptr);
        std::weak_ptr<ITensorGemm> old_gate = source->gate();
        source.reset();
        // Complete-but-wrong payloads must not slip through a readiness check.
        EXPECT_FALSE(slots.stage(movement,
            {{{1, 0, 0}, payload()}, {{0, 0, 1}, pair()}}, &error));
        EXPECT_EQ(slots.activeSlotCount(), 2u);
        ASSERT_TRUE(slots.stage(movement,
            {{{1, 0, 0}, pair()}, {{0, 0, 1}, pair()}}, &error)) << error;
        ASSERT_TRUE(slots.publish(movement, &error)) << error;
        EXPECT_EQ(slots.activeSlotCount(), 4u);
        EXPECT_FALSE(old_gate.expired());
        std::vector<MoEOverlayDeviceRetiredPhysicalSlot> retired;
        ASSERT_TRUE(slots.retire(movement, &retired, &error)) << error;
        EXPECT_FALSE(old_gate.expired()); // Retirement receipt still owns its engines.
        retired.clear();
        EXPECT_TRUE(old_gate.expired());
        const auto inventory = slots.snapshot(movement.candidate_epoch, &error);
        ASSERT_TRUE(inventory) << error;
        for (const auto &slot : inventory->slots)
            EXPECT_TRUE(slot.payload.readyFor(DeviceMoEProjectionSet::GateUp));

        EXPECT_THROW((MoEOverlayDevicePhysicalSlotLedger{
            {.initial_epoch = 1u, .local_participant_ids = {0},
             .initial_slots = {{.key = {0, 0, 0}, .entered_epoch = 1u, .payload = pair()}},
             .num_layers = 1u, .num_experts = 8u}}),
            std::invalid_argument);
    }

    /** Every physical lifetime edge binds the same immutable global stage. */
    TEST(Test__MoEOverlayDevicePhysicalSlotLedger, PipelineStageLifetimeRejectsForeignWave)
    {
        for (const int origin : {0, 32, 40, std::numeric_limits<int>::max() - 1})
        {
            SCOPED_TRACE(origin);
            auto slots = ledger(origin);
            const auto movement = batch(61u, 1u, 0, 1, origin);
            // Deliberately preserve every transaction word and execution
            // fingerprint: stage identity must be authenticated independently.
            const auto foreign = batch(61u, 1u, 0, 1, origin == 0 ? 32 : 0);
            ASSERT_TRUE(movement.valid());
            ASSERT_TRUE(foreign.valid());
            std::string error;
            EXPECT_FALSE(slots.begin(foreign, &error));
            EXPECT_FALSE(slots.hasPendingWave());
            ASSERT_TRUE(slots.begin(movement, &error)) << error;
            const MoEOverlayDevicePhysicalSlotKey source{0, origin, 0};
            auto retained = slots.sourcePayload(movement, source, &error);
            ASSERT_TRUE(retained) << error;
            std::weak_ptr<ITensorGemm> old_gate = retained->gate();
            retained.reset();
            EXPECT_FALSE(slots.sourcePayload(foreign, source, &error));
            EXPECT_FALSE(slots.abort(foreign, &error));
            const std::vector<MoEOverlayDeviceStagedPhysicalArrival> arrivals{
                {{1, origin, 0}, payload()}, {{0, origin, 1}, payload()}};
            EXPECT_FALSE(slots.stage(foreign, arrivals, &error));
            ASSERT_TRUE(slots.stage(movement, arrivals, &error)) << error;
            EXPECT_FALSE(slots.publish(foreign, &error));
            EXPECT_EQ(slots.activeSlotCount(), 2u);
            ASSERT_TRUE(slots.publish(movement, &error)) << error;
            EXPECT_EQ(slots.activeSlotCount(), 4u);
            EXPECT_FALSE(old_gate.expired());
            std::vector<MoEOverlayDeviceRetiredPhysicalSlot> retired;
            EXPECT_FALSE(slots.retire(foreign, &retired, &error));
            EXPECT_TRUE(retired.empty());
            EXPECT_EQ(slots.activeSlotCount(), 4u);
            ASSERT_TRUE(slots.retire(movement, &retired, &error)) << error;
            ASSERT_EQ(retired.size(), 2u);
            EXPECT_FALSE(old_gate.expired());
            retired.clear();
            EXPECT_TRUE(old_gate.expired());
            const auto inventory = slots.snapshot(2u, &error);
            ASSERT_TRUE(inventory) << error;
            ASSERT_EQ(inventory->slots.size(), 2u);
            for (const auto &slot : inventory->slots)
                EXPECT_EQ(slot.key.layer_idx, origin);
        }
    }

    /** A rank with no local movement still rejects another stage before opening a wave. */
    TEST(Test__MoEOverlayDevicePhysicalSlotLedger, PipelineStageEmptyRankAndGeometryAreExplicit)
    {
        MoEOverlayDevicePhysicalSlotLedger slots({
            .initial_epoch = 1u, .local_participant_ids = {2},
            .first_model_layer = 32, .num_layers = 1u, .num_experts = 8u});
        auto movement = batch(71u, 1u, 0, 1, 32);
        movement.participant_count = 3u;
        for (int mutation = 0; mutation < 3; ++mutation)
        {
            auto foreign = mutation == 0 ? batch(71u, 1u, 0, 1, 40) : movement;
            foreign.participant_count = 3u;
            if (mutation == 1) ++foreign.num_layers;
            if (mutation == 2) ++foreign.num_experts;
            ASSERT_TRUE(foreign.valid());
            std::string error;
            EXPECT_FALSE(slots.begin(foreign, &error));
            EXPECT_FALSE(error.empty());
            EXPECT_FALSE(slots.hasPendingWave());
            EXPECT_EQ(slots.currentEpoch(), 1u);
        }
        std::string error;
        ASSERT_TRUE(slots.begin(movement, &error)) << error;
        auto changed_geometry = movement;
        ++changed_geometry.num_layers;
        EXPECT_FALSE(slots.stage(changed_geometry, {}, &error));
        ASSERT_TRUE(slots.stage(movement, {}, &error)) << error;
        EXPECT_FALSE(slots.publish(changed_geometry, &error));
        ASSERT_TRUE(slots.publish(movement, &error)) << error;
        std::vector<MoEOverlayDeviceRetiredPhysicalSlot> retired;
        EXPECT_FALSE(slots.retire(changed_geometry, &retired, &error));
        ASSERT_TRUE(slots.retire(movement, &retired, &error)) << error;
        EXPECT_TRUE(retired.empty());
        EXPECT_EQ(slots.activeSlotCount(), 0u);
        EXPECT_EQ(slots.currentEpoch(), 2u);
    }

    /** Initial enrollment rejects foreign keys and overflow before retaining any slot. */
    TEST(Test__MoEOverlayDevicePhysicalSlotLedger, PipelineStageEnrollmentRejectsInvalidGeometry)
    {
        const MoEOverlayDevicePhysicalSlotLedger::Config good{
            .initial_epoch = 1u, .local_participant_ids = {0},
            .initial_slots = {{.key = {0, 32, 0}, .entered_epoch = 1u, .payload = payload()}},
            .first_model_layer = 32, .num_layers = 1u, .num_experts = 8u};
        EXPECT_NO_THROW(MoEOverlayDevicePhysicalSlotLedger{good});
        for (int mutation = 0; mutation < 7; ++mutation)
        {
            auto invalid = good;
            if (mutation == 0) invalid.first_model_layer = -1;
            if (mutation == 1) invalid.num_layers = 0u;
            if (mutation == 2) invalid.first_model_layer = std::numeric_limits<int>::max();
            if (mutation == 3) invalid.num_experts = 0u;
            if (mutation == 4) invalid.initial_slots[0].key.layer_idx = 31;
            if (mutation == 5) invalid.initial_slots[0].key.layer_idx = 33;
            if (mutation == 6) invalid.initial_slots[0].key.expert_id = 8;
            EXPECT_THROW(MoEOverlayDevicePhysicalSlotLedger{invalid}, std::invalid_argument) << mutation;
        }
    }

    TEST(Test__MoEOverlayDevicePhysicalSlotLedger,
         PublicationRetainsOldAndNewSlotsUntilReaderRetirement)
    {
        auto slots = ledger();
        const auto first = batch(11u, 1u, 0, 1);
        ASSERT_TRUE(first.valid());
        std::string error;
        ASSERT_TRUE(slots.begin(first, &error)) << error;
        EXPECT_TRUE(slots.hasPendingWave());
        EXPECT_EQ(slots.currentEpoch(), 1u);
        EXPECT_EQ(slots.activeSlotCount(), 2u);

        EXPECT_TRUE(slots.sourcePayload(
            first,
            {.participant_id = 0, .layer_idx = 0, .expert_id = 0},
            &error));
        EXPECT_TRUE(slots.sourcePayload(
            first,
            {.participant_id = 1, .layer_idx = 0, .expert_id = 1},
            &error));

        ASSERT_TRUE(slots.stage(
            first,
            {
                {
                    .key = {
                        .participant_id = 0,
                        .layer_idx = 0,
                        .expert_id = 1,
                    },
                    .payload = payload(),
                },
                {
                    .key = {
                        .participant_id = 1,
                        .layer_idx = 0,
                        .expert_id = 0,
                    },
                    .payload = payload(),
                },
            },
            &error))
            << error;
        ASSERT_TRUE(slots.publish(first, &error)) << error;
        EXPECT_EQ(slots.currentEpoch(), 2u);
        EXPECT_EQ(slots.activeSlotCount(), 4u)
            << "old readers and the new device bank must overlap";
        EXPECT_TRUE(slots.sourcePayload(
            first,
            {.participant_id = 0, .layer_idx = 0, .expert_id = 0},
            &error));

        std::vector<MoEOverlayDeviceRetiredPhysicalSlot> retired;
        ASSERT_TRUE(slots.retire(first, &retired, &error)) << error;
        ASSERT_EQ(retired.size(), 2u);
        EXPECT_TRUE(retired[0].bootstrap_allocation);
        EXPECT_TRUE(retired[1].bootstrap_allocation);
        EXPECT_EQ(slots.activeSlotCount(), 2u);
        EXPECT_FALSE(slots.hasPendingWave());

        // A later epoch resolves the arrivals retained by the physical ledger,
        // not the setup-time participant registry.
        auto second = batch(12u, 2u, 1, 0);
        ASSERT_TRUE(second.valid());
        ASSERT_TRUE(slots.begin(second, &error)) << error;
        EXPECT_TRUE(slots.sourcePayload(
            second,
            {.participant_id = 0, .layer_idx = 0, .expert_id = 1},
            &error));
        EXPECT_TRUE(slots.sourcePayload(
            second,
            {.participant_id = 1, .layer_idx = 0, .expert_id = 0},
            &error));
        ASSERT_TRUE(slots.abort(second, &error)) << error;
    }

    TEST(Test__MoEOverlayDevicePhysicalSlotLedger,
         IncompleteStageAndStaleWaveCannotMutateDurableInventory)
    {
        auto slots = ledger();
        const auto movement = batch(21u, 1u, 0, 1);
        std::string error;
        ASSERT_TRUE(slots.begin(movement, &error)) << error;
        EXPECT_FALSE(slots.begin(movement, &error));
        EXPECT_FALSE(slots.stage(
            movement,
            {{
                .key = {
                    .participant_id = 1,
                    .layer_idx = 0,
                    .expert_id = 0,
                },
                .payload = payload(),
            }},
            &error));
        EXPECT_EQ(slots.currentEpoch(), 1u);
        EXPECT_EQ(slots.activeSlotCount(), 2u);
        ASSERT_TRUE(slots.abort(movement, &error)) << error;
        EXPECT_FALSE(slots.hasPendingWave());

        auto stale = movement;
        stale.transaction_id += 1u;
        stale.base_epoch = 2u;
        stale.candidate_epoch = 3u;
        stale.execution_fingerprint.low += 1u;
        EXPECT_FALSE(slots.begin(stale, &error));
        EXPECT_EQ(slots.currentEpoch(), 1u);
        EXPECT_EQ(slots.activeSlotCount(), 2u);
    }

    TEST(Test__MoEOverlayDevicePhysicalSlotLedger,
         PublishedWaveCannotBeAbortedOrOverlapped)
    {
        auto slots = ledger();
        const auto movement = batch(31u, 1u, 0, 1);
        std::string error;
        ASSERT_TRUE(slots.begin(movement, &error)) << error;
        ASSERT_TRUE(slots.stage(
            movement,
            {
                {
                    .key = {
                        .participant_id = 0,
                        .layer_idx = 0,
                        .expert_id = 1,
                    },
                    .payload = payload(),
                },
                {
                    .key = {
                        .participant_id = 1,
                        .layer_idx = 0,
                        .expert_id = 0,
                    },
                    .payload = payload(),
                },
            },
            &error))
            << error;
        ASSERT_TRUE(slots.publish(movement, &error)) << error;
        EXPECT_FALSE(slots.abort(movement, &error));
        EXPECT_FALSE(slots.begin(batch(32u, 2u, 1, 0), &error));
        EXPECT_EQ(slots.activeSlotCount(), 4u);
    }
} // namespace llaminar2::test
