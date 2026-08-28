/**
 * @file Test__MoEOverlayDevicePhysicalSlotLedger.cpp
 * @brief Device-free adversarial tests for physical Dynamic slot lifetimes.
 */

#include <gtest/gtest.h>

#include "execution/moe/MoEOverlayDevicePhysicalSlotLedger.h"

#include <cstdint>
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
        MoEOverlayPreparedExpertTriplet triplet()
        {
            return {
                .gate = engineIdentity(),
                .up = engineIdentity(),
                .down = engineIdentity(),
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
            int second_expert)
        {
            std::vector<MoEOverlayTierMigration> migrations{
                {
                    .layer_idx = 0,
                    .expert_id = first_expert,
                    .estimated_weight_bytes = 128u,
                    .direction =
                        MoEOverlayTierMigrationDirection::SamePriority,
                    .source = owner(0, 0, first_expert),
                    .destination = owner(1, 0, first_expert),
                },
                {
                    .layer_idx = 0,
                    .expert_id = second_expert,
                    .estimated_weight_bytes = 128u,
                    .direction =
                        MoEOverlayTierMigrationDirection::SamePriority,
                    .source = owner(1, 0, second_expert),
                    .destination = owner(0, 0, second_expert),
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
                    .layer_idx = 0,
                    .migration_indices = {0u, 1u},
                }},
                .shadow_requirements = {
                    {
                        .layer_idx = 0,
                        .tier_idx = 0,
                        .destination_participant = 0,
                        .slot_count = 1u,
                    },
                    {
                        .layer_idx = 0,
                        .tier_idx = 0,
                        .destination_participant = 1,
                        .slot_count = 1u,
                    },
                },
            };
        }

        /** Construct the two bootstrap allocations named by the first wave. */
        MoEOverlayDevicePhysicalSlotLedger ledger()
        {
            std::vector<MoEOverlayDeviceInitialPhysicalSlot> initial;
            initial.push_back({
                .key = {.participant_id = 0, .layer_idx = 0, .expert_id = 0},
                .entered_epoch = 1u,
                .bootstrap_allocation = true,
                .triplet = triplet(),
            });
            initial.push_back({
                .key = {.participant_id = 1, .layer_idx = 0, .expert_id = 1},
                .entered_epoch = 1u,
                .bootstrap_allocation = true,
                .triplet = triplet(),
            });
            return MoEOverlayDevicePhysicalSlotLedger({
                .initial_epoch = 1u,
                .local_participant_ids = {0, 1},
                .initial_slots = std::move(initial),
            });
        }
    } // namespace

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

        EXPECT_TRUE(slots.sourceTriplet(
            first,
            {.participant_id = 0, .layer_idx = 0, .expert_id = 0},
            &error));
        EXPECT_TRUE(slots.sourceTriplet(
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
                    .triplet = triplet(),
                },
                {
                    .key = {
                        .participant_id = 1,
                        .layer_idx = 0,
                        .expert_id = 0,
                    },
                    .triplet = triplet(),
                },
            },
            &error))
            << error;
        ASSERT_TRUE(slots.publish(first, &error)) << error;
        EXPECT_EQ(slots.currentEpoch(), 2u);
        EXPECT_EQ(slots.activeSlotCount(), 4u)
            << "old readers and the new device bank must overlap";
        EXPECT_TRUE(slots.sourceTriplet(
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
        EXPECT_TRUE(slots.sourceTriplet(
            second,
            {.participant_id = 0, .layer_idx = 0, .expert_id = 1},
            &error));
        EXPECT_TRUE(slots.sourceTriplet(
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
                .triplet = triplet(),
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
                    .triplet = triplet(),
                },
                {
                    .key = {
                        .participant_id = 1,
                        .layer_idx = 0,
                        .expert_id = 0,
                    },
                    .triplet = triplet(),
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
