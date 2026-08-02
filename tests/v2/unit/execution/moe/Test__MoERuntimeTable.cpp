/**
 * @file Test__MoERuntimeTable.cpp
 * @brief Unit tests for graph-facing MoE runtime placement tables.
 */

#include "execution/moe/MoERuntimeTable.h"
#include "execution/moe/DeviceMoERebalanceController.h"
#include "execution/moe/DecodeExpertHistogram.h"

#include <gtest/gtest.h>

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

#include <cstdint>
#include <stdexcept>

namespace llaminar2::test
{
    namespace
    {
        DeviceNativeVNNIMatrixDesc matrixDesc(uintptr_t base, int n, int k)
        {
            DeviceNativeVNNIMatrixDesc desc;
            desc.payload = reinterpret_cast<const uint8_t *>(base);
            desc.scales = reinterpret_cast<const void *>(base + 0x1000u);
            desc.mins = reinterpret_cast<const void *>(base + 0x2000u);
            desc.emins = reinterpret_cast<const void *>(base + 0x3000u);
            desc.n = n;
            desc.k = k;
            desc.blocks_per_row = 4;
            desc.codebook_id = 7;
            return desc;
        }

        DeviceMoEExpertDescriptor expertDescAtBase(
            int expert_id,
            int owner,
            int local_slot,
            uintptr_t base,
            DeviceMoEExpertFlags flags = DeviceMoEExpertFlags::Valid |
                                         DeviceMoEExpertFlags::Resident |
                                         DeviceMoEExpertFlags::LocalCompute)
        {
            DeviceMoEExpertDescriptor desc;
            desc.gate = matrixDesc(base + 0x0100u, 64, 32);
            desc.up = matrixDesc(base + 0x0200u, 64, 32);
            desc.down = matrixDesc(base + 0x0300u, 32, 64);
            desc.logical_expert_id = expert_id;
            desc.owner_participant = owner;
            desc.local_slot = local_slot;
            desc.flags = toMoEExpertFlags(flags);
            return desc;
        }

        DeviceMoEExpertDescriptor expertDesc(int expert_id,
                                             int owner,
                                             int local_slot,
                                             DeviceMoEExpertFlags flags = DeviceMoEExpertFlags::Valid |
                                                                          DeviceMoEExpertFlags::Resident |
                                                                          DeviceMoEExpertFlags::LocalCompute)
        {
            const uintptr_t base = 0x10000000u + static_cast<uintptr_t>(expert_id) * 0x10000u;
            return expertDescAtBase(expert_id, owner, local_slot, base, flags);
        }

        DeviceMoEExpertDescriptor expertOwnerOnlyDesc(int expert_id, int owner)
        {
            DeviceMoEExpertDescriptor desc;
            desc.logical_expert_id = expert_id;
            desc.owner_participant = owner;
            desc.local_slot = -1;
            return desc;
        }

        MoEPlacementUpdate updateForEpoch(uint32_t epoch, int expert_count)
        {
            MoEPlacementUpdate update;
            update.epoch = epoch;
            update.expert_count = static_cast<uint32_t>(expert_count);
            update.experts.reserve(static_cast<size_t>(expert_count));
            update.local_compute_mask.reserve(static_cast<size_t>(expert_count));
            update.replica_role.reserve(static_cast<size_t>(expert_count));

            for (int expert = 0; expert < expert_count; ++expert)
            {
                update.experts.push_back(expertDesc(expert, expert % 2, expert));
                update.local_compute_mask.push_back(1);
                update.replica_role.push_back(static_cast<uint8_t>((expert % 2 == 0)
                                                                       ? DeviceMoEReplicaRole::Primary
                                                                       : DeviceMoEReplicaRole::Replica));
            }
            return update;
        }

        DeviceMoERebalanceConfig rebalanceConfig(
            uint32_t layers,
            uint32_t experts,
            uint32_t top_k,
            uint32_t participant,
            uint32_t participants,
            uint32_t window_tokens,
            uint32_t max_hot_replicas,
            uint32_t root_participant = 0)
        {
            DeviceMoERebalanceConfig config;
            config.num_layers = layers;
            config.num_experts = experts;
            config.top_k = top_k;
            config.participant_id = participant;
            config.participant_count = participants;
            config.root_participant = root_participant;
            config.window_size_tokens = window_tokens;
            config.max_hot_replicas_per_participant = max_hot_replicas;
            return config;
        }

        void addGatheredCount(std::vector<uint64_t> &histograms,
                              const DeviceMoERebalanceConfig &config,
                              uint32_t participant,
                              uint32_t layer,
                              uint32_t expert,
                              uint64_t count)
        {
            const uint64_t participant_stride =
                static_cast<uint64_t>(config.num_layers) *
                static_cast<uint64_t>(config.num_experts);
            const uint64_t idx =
                static_cast<uint64_t>(participant) * participant_stride +
                static_cast<uint64_t>(layer) * static_cast<uint64_t>(config.num_experts) +
                static_cast<uint64_t>(expert);
            histograms[static_cast<size_t>(idx)] += count;
        }
    } // namespace

    TEST(Test__MoERuntimeTable, ConstructionCreatesStableLayerPointers)
    {
        MoERuntimeTable table(DeviceId::cpu(), 3, 4, 2);

        EXPECT_EQ(table.layerCount(), 3);

        auto *layer0 = table.deviceLayerState(0);
        auto *layer1 = table.deviceLayerState(1);
        auto *layer2 = table.deviceLayerState(2);

        EXPECT_NE(layer0, nullptr);
        EXPECT_NE(layer1, nullptr);
        EXPECT_NE(layer2, nullptr);
        EXPECT_NE(layer0, layer1);
        EXPECT_NE(layer1, layer2);
        EXPECT_EQ(layer0, &table.hostLayerState(0));
        EXPECT_EQ(layer0->expert_count, 4u);
        EXPECT_EQ(layer0->top_k, 2u);
        EXPECT_EQ(layer0->active_bank, 0u);
        EXPECT_EQ(layer0->active_epoch, 0u);
        EXPECT_EQ(layer0->participant_id, 0u);
        EXPECT_EQ(layer0->participant_count, 1u);
    }

    TEST(Test__MoERuntimeTable, FullyReplicatedTopologyPublishesEveryParticipantAndCanonicalOwner)
    {
        constexpr int kExperts = 4;
        constexpr int kParticipants = 2;
        const std::vector<int> owners{0, 1, 0, 1};

        for (int local_participant = 0;
             local_participant < kParticipants;
             ++local_participant)
        {
            MoEPlacementUpdate update;
            update.epoch = 1;
            update.expert_count = kExperts;
            for (int expert = 0; expert < kExperts; ++expert)
                update.experts.push_back(expertDesc(expert, -1, expert));

            declareFullyReplicatedPlacementTopology(
                update,
                local_participant,
                kParticipants,
                owners);

            EXPECT_EQ(update.participant_id,
                      static_cast<uint32_t>(local_participant));
            EXPECT_EQ(update.participant_count,
                      static_cast<uint32_t>(kParticipants));
            ASSERT_EQ(update.local_compute_mask.size(), kExperts);
            ASSERT_EQ(update.replica_role.size(), kExperts);
            ASSERT_EQ(update.resident_participant_mask.size(), kExperts);

            for (int expert = 0; expert < kExperts; ++expert)
            {
                const bool is_primary = owners[expert] == local_participant;
                const auto &descriptor = update.experts[expert];
                EXPECT_EQ(update.local_compute_mask[expert], 1u);
                EXPECT_EQ(update.resident_participant_mask[expert], 0b11u);
                EXPECT_EQ(
                    update.replica_role[expert],
                    static_cast<uint8_t>(
                        is_primary
                            ? DeviceMoEReplicaRole::Primary
                            : DeviceMoEReplicaRole::Replica));
                EXPECT_EQ(descriptor.owner_participant, owners[expert]);
                EXPECT_TRUE(hasMoEExpertFlag(
                    descriptor.flags,
                    DeviceMoEExpertFlags::Replicated));
                EXPECT_EQ(
                    hasMoEExpertFlag(
                        descriptor.flags,
                        DeviceMoEExpertFlags::PreferredOwner),
                    is_primary);
            }
        }
    }

    TEST(Test__MoERuntimeTable, FullyReplicatedTopologyRejectsIncompleteDomainMetadata)
    {
        const std::vector<int> incomplete_owners{0};
        const std::vector<int> out_of_domain_owners{0, 2};
        MoEPlacementUpdate update;
        update.epoch = 1;
        update.expert_count = 2;
        update.experts = {
            expertDesc(0, -1, 0),
            expertDesc(1, -1, 1),
        };

        EXPECT_THROW(
            declareFullyReplicatedPlacementTopology(
                update,
                /*local_participant=*/0,
                /*participant_count=*/2,
                incomplete_owners),
            std::invalid_argument);
        EXPECT_THROW(
            declareFullyReplicatedPlacementTopology(
                update,
                /*local_participant=*/0,
                /*participant_count=*/2,
                out_of_domain_owners),
            std::invalid_argument);
    }

    TEST(Test__MoERuntimeTable, DeviceRebalanceConfigValidatesExplicitRootParticipant)
    {
        auto config = rebalanceConfig(/*layers=*/2,
                                      /*experts=*/4,
                                      /*top_k=*/2,
                                      /*participant=*/1,
                                      /*participants=*/3,
                                      /*window_tokens=*/8,
                                      /*max_hot_replicas=*/1,
                                      /*root_participant=*/2);

        EXPECT_TRUE(validateDeviceMoERebalanceConfig(config));
        EXPECT_FALSE(deviceMoEIsRootParticipant(config));
        config.participant_id = 2;
        EXPECT_TRUE(deviceMoEIsRootParticipant(config));

        config.root_participant = 3;
        EXPECT_FALSE(validateDeviceMoERebalanceConfig(config))
            << "root participant must be inside the homogeneous rebalance domain.";
    }

    TEST(Test__MoERuntimeTable, DeviceRebalancePayloadEdgeMaskUsesStableDirectedStride)
    {
        using llaminar2::moe_rebalance_policy::directedParticipantEdgeBit;

        EXPECT_EQ(directedParticipantEdgeBit(0, 1, 8), 1ULL << 1u);
        EXPECT_EQ(directedParticipantEdgeBit(1, 0, 8), 1ULL << 8u);
        EXPECT_EQ(directedParticipantEdgeBit(3, 4, 8), 1ULL << 28u);
        EXPECT_EQ(directedParticipantEdgeBit(7, 7, 8), 1ULL << 63u);
        EXPECT_EQ(directedParticipantEdgeBit(8, 0, 8), 0ULL);
        EXPECT_EQ(directedParticipantEdgeBit(0, 8, 8), 0ULL);
        EXPECT_EQ(directedParticipantEdgeBit(0, 0, 0), 0ULL);
    }

    TEST(Test__MoERuntimeTable, DeviceRebalanceTransferCostEstimateAccountsFixedPayloadCapacity)
    {
        auto config = rebalanceConfig(/*layers=*/4,
                                      /*experts=*/8,
                                      /*top_k=*/2,
                                      /*participant=*/0,
                                      /*participants=*/2,
                                      /*window_tokens=*/8,
                                      /*max_hot_replicas=*/3);

        constexpr uint32_t kLocalTransferSlots = 5;
        constexpr uint64_t kSlotBytes = 1024;
        constexpr uint32_t kCopiedArrivals = 2;

        EXPECT_FALSE(deviceMoERebalanceModeMovesFixedPayloadCapacity(
            DeviceMoERebalanceTransferMode::ResidentOnly));
        EXPECT_FALSE(deviceMoERebalanceModeMovesFixedPayloadCapacity(
            DeviceMoERebalanceTransferMode::CompactTransferSlots));
        EXPECT_TRUE(deviceMoERebalanceModeUsesCollectivePayloadLane(
            DeviceMoERebalanceTransferMode::CompactTransferSlots));
        EXPECT_TRUE(deviceMoERebalanceModeUsesCollectivePayloadLane(
            DeviceMoERebalanceTransferMode::CollectiveSidebandPayload));
        EXPECT_TRUE(deviceMoERebalanceModeMovesFixedPayloadCapacity(
            DeviceMoERebalanceTransferMode::CollectiveSidebandPayload));
        EXPECT_TRUE(deviceMoERebalanceModeMovesFixedPayloadCapacity(
            DeviceMoERebalanceTransferMode::LegacyCollectiveAllGather));
        EXPECT_FALSE(deviceMoERebalanceModeUsesTransferSlots(
            DeviceMoERebalanceTransferMode::ResidentOnly));
        EXPECT_TRUE(deviceMoERebalanceModeUsesTransferSlots(
            DeviceMoERebalanceTransferMode::CompactTransferSlots));
        EXPECT_TRUE(deviceMoERebalanceModePlansMissingArrivals(
            DeviceMoERebalanceTransferMode::CompactTransferSlots));

        const auto estimate =
            estimateDeviceMoERebalanceTransferCost(
                config,
                DeviceMoERebalanceTransferMode::CollectiveSidebandPayload,
                kLocalTransferSlots,
                kSlotBytes,
                kCopiedArrivals);

        EXPECT_EQ(estimate.command_buffer_count, 2u);
        EXPECT_EQ(estimate.transfer_plan_capacity, 12u)
            << "command metadata capacity must be independent from compact payload slots.";
        EXPECT_EQ(estimate.payload_slot_capacity, 5u)
            << "payload capacity is bounded by the rolling transfer slots.";
        EXPECT_EQ(estimate.payload_slot_count, 10u);
        EXPECT_EQ(estimate.slot_payload_bytes, kSlotBytes);
        EXPECT_EQ(estimate.plan_local_bytes,
                  2u * 12u * sizeof(DeviceMoERebalancePlanEntry));
        EXPECT_EQ(estimate.plan_gathered_bytes,
                  2u * estimate.plan_local_bytes);
        EXPECT_EQ(estimate.header_local_bytes,
                  2u * sizeof(DeviceMoERebalanceCommandBufferHeader));
        EXPECT_EQ(estimate.header_gathered_bytes,
                  2u * estimate.header_local_bytes);
        EXPECT_EQ(estimate.source_descriptor_local_bytes, 0u);
        EXPECT_EQ(estimate.source_descriptor_gathered_bytes, 0u);
        EXPECT_EQ(estimate.selected_payload_bucket_slots, 5u);
        EXPECT_EQ(estimate.selected_payload_slot_count, 10u);
        EXPECT_EQ(estimate.selected_payload_local_capacity_bytes, 10u * kSlotBytes);
        EXPECT_EQ(estimate.selected_payload_gathered_capacity_bytes, 20u * kSlotBytes);
        EXPECT_EQ(estimate.payload_local_capacity_bytes, 10u * kSlotBytes);
        EXPECT_EQ(estimate.payload_gathered_capacity_bytes, 20u * kSlotBytes);
        EXPECT_EQ(estimate.captured_payload_slack_bytes, 0u);
        EXPECT_EQ(estimate.useful_payload_bytes, kCopiedArrivals * kSlotBytes);
        EXPECT_EQ(estimate.wasted_payload_capacity_bytes,
                  estimate.payload_gathered_capacity_bytes - estimate.useful_payload_bytes);
        EXPECT_EQ(deviceMoELoadSpread(/*min_load=*/3, /*max_load=*/10), 7u);

        const auto resident_only =
            estimateDeviceMoERebalanceTransferCost(
                config,
                DeviceMoERebalanceTransferMode::ResidentOnly,
                kLocalTransferSlots,
                kSlotBytes,
                kCopiedArrivals);
        EXPECT_EQ(resident_only.transfer_plan_capacity, 0u);
        EXPECT_EQ(resident_only.selected_payload_bucket_slots, 0u);
        EXPECT_EQ(resident_only.selected_payload_gathered_capacity_bytes, 0u);
        EXPECT_EQ(resident_only.payload_gathered_capacity_bytes, 0u);
        EXPECT_EQ(resident_only.useful_payload_bytes, 0u);
        EXPECT_EQ(resident_only.wasted_payload_capacity_bytes, 0u);

        const auto full_window_compact =
            estimateDeviceMoERebalanceTransferCost(
                config,
                DeviceMoERebalanceTransferMode::CompactTransferSlots,
                kLocalTransferSlots,
                kSlotBytes,
                kCopiedArrivals);
        EXPECT_EQ(full_window_compact.transfer_plan_capacity, 48u)
            << "Full-window compact root planning reserves participant^2 command headroom.";

        auto wave_config = config;
        wave_config.layer_wave_count = 1;
        const auto compact =
            estimateDeviceMoERebalanceTransferCost(
                wave_config,
                DeviceMoERebalanceTransferMode::CompactTransferSlots,
                kLocalTransferSlots,
                kSlotBytes,
                kCopiedArrivals,
                /*collective_payload_slot_capacity=*/1);
        EXPECT_EQ(compact.command_buffer_count, 2u);
        EXPECT_EQ(compact.transfer_plan_capacity, 12u)
            << "Production one-layer waves keep compact command metadata bounded.";
        EXPECT_EQ(compact.payload_slot_capacity, 1u);
        EXPECT_EQ(compact.plan_local_bytes,
                  2u * 12u * sizeof(DeviceMoERebalancePlanEntry));
        EXPECT_EQ(compact.header_local_bytes,
                  2u * sizeof(DeviceMoERebalanceCommandBufferHeader));
        EXPECT_EQ(compact.source_descriptor_local_bytes,
                  2u * 2u * 12u * sizeof(DeviceMoEExpertDirectoryEntry));
        EXPECT_EQ(compact.source_descriptor_gathered_bytes, 0u);
        EXPECT_EQ(compact.selected_payload_bucket_slots, 1u);
        EXPECT_EQ(compact.selected_payload_slot_count, 1u);
        EXPECT_EQ(compact.selected_payload_local_capacity_bytes, kSlotBytes);
        EXPECT_EQ(compact.selected_payload_gathered_capacity_bytes, 2u * kSlotBytes);
        EXPECT_EQ(compact.payload_slot_count, 1u);
        EXPECT_EQ(compact.payload_local_capacity_bytes, kSlotBytes);
        EXPECT_EQ(compact.payload_gathered_capacity_bytes, 2u * kSlotBytes);
        EXPECT_EQ(compact.captured_payload_slack_bytes, 0u);
        EXPECT_EQ(compact.useful_payload_bytes, kCopiedArrivals * kSlotBytes);
        EXPECT_EQ(compact.wasted_payload_capacity_bytes,
                  compact.payload_gathered_capacity_bytes - compact.useful_payload_bytes);

        const auto captured_wider_than_selected =
            estimateDeviceMoERebalanceTransferCost(
                wave_config,
                DeviceMoERebalanceTransferMode::CompactTransferSlots,
                kLocalTransferSlots,
                kSlotBytes,
                kCopiedArrivals,
                /*collective_payload_slot_capacity=*/4,
                /*selected_payload_bucket_slots=*/1);
        EXPECT_EQ(captured_wider_than_selected.payload_slot_capacity, 4u);
        EXPECT_EQ(captured_wider_than_selected.payload_slot_count, 4u);
        EXPECT_EQ(captured_wider_than_selected.payload_gathered_capacity_bytes,
                  8u * kSlotBytes);
        EXPECT_EQ(captured_wider_than_selected.selected_payload_bucket_slots, 1u);
        EXPECT_EQ(captured_wider_than_selected.selected_payload_slot_count, 1u);
        EXPECT_EQ(captured_wider_than_selected.selected_payload_gathered_capacity_bytes,
                  2u * kSlotBytes);
        EXPECT_EQ(captured_wider_than_selected.captured_payload_slack_bytes,
                  6u * kSlotBytes)
            << "Perf accounting must expose captured bucket slack separately from useful selected payload.";
        EXPECT_EQ(captured_wider_than_selected.wasted_payload_capacity_bytes, 0u)
            << "Wasted payload capacity is charged to the selected bucket, not the full captured variant.";

        const auto no_payload_bucket =
            estimateDeviceMoERebalanceTransferCost(
                wave_config,
                DeviceMoERebalanceTransferMode::CompactTransferSlots,
                kLocalTransferSlots,
                kSlotBytes,
                /*copied_arrivals=*/0,
                /*collective_payload_slot_capacity=*/4,
                /*selected_payload_bucket_slots=*/0);
        EXPECT_EQ(no_payload_bucket.payload_slot_capacity, 4u);
        EXPECT_EQ(no_payload_bucket.payload_gathered_capacity_bytes,
                  8u * kSlotBytes)
            << "Capacity telemetry still exposes the graph variant size.";
        EXPECT_EQ(no_payload_bucket.selected_payload_bucket_slots, 0u);
        EXPECT_EQ(no_payload_bucket.selected_payload_slot_count, 0u);
        EXPECT_EQ(no_payload_bucket.selected_payload_gathered_capacity_bytes, 0u);
        EXPECT_EQ(no_payload_bucket.captured_payload_slack_bytes, 0u)
            << "A skipped payload graph must not report captured payload slack.";
        EXPECT_EQ(no_payload_bucket.useful_payload_bytes, 0u);
        EXPECT_EQ(no_payload_bucket.wasted_payload_capacity_bytes, 0u)
            << "No-work waves must not look like empty payload-bucket transfers.";
    }

    TEST(Test__MoERuntimeTable, DeviceRebalanceRouterBenefitFloorAllowsBootstrapThenRejectsLowValueWaves)
    {
        using moe_rebalance_policy::transferWaveMeetsRealizedRouterBenefitFloor;

        EXPECT_TRUE(transferWaveMeetsRealizedRouterBenefitFloor(
            /*realized_router_spread_improvement=*/0,
            /*requested_payload_slots=*/2,
            /*min_router_spread_improvement_per_payload_slot=*/64,
            /*existing_hot_cache_active=*/false))
            << "Cold-start arrivals must be allowed before hot-cache routing has had a chance to pay off.";

        EXPECT_TRUE(transferWaveMeetsRealizedRouterBenefitFloor(
            /*realized_router_spread_improvement=*/0,
            /*requested_payload_slots=*/2,
            /*min_router_spread_improvement_per_payload_slot=*/0,
            /*existing_hot_cache_active=*/true))
            << "A zero threshold disables the realized router-benefit gate.";

        EXPECT_FALSE(transferWaveMeetsRealizedRouterBenefitFloor(
            /*realized_router_spread_improvement=*/63,
            /*requested_payload_slots=*/2,
            /*min_router_spread_improvement_per_payload_slot=*/32,
            /*existing_hot_cache_active=*/true));

        EXPECT_TRUE(transferWaveMeetsRealizedRouterBenefitFloor(
            /*realized_router_spread_improvement=*/64,
            /*requested_payload_slots=*/2,
            /*min_router_spread_improvement_per_payload_slot=*/32,
            /*existing_hot_cache_active=*/true));
    }

    TEST(Test__MoERuntimeTable, PrepareInactiveBankCopiesDescriptorsMasksAndReplicaRoles)
    {
        MoERuntimeTable table(DeviceId::cpu(), 1, 4, 2);
        auto update = updateForEpoch(1, 4);
        update.participant_id = 1;
        update.participant_count = 2;

        ASSERT_TRUE(table.prepareInactiveBank(0, update));
        const auto &state = table.hostLayerState(0);
        const auto &bank = state.banks[1];

        EXPECT_EQ(state.active_bank, 0u);
        EXPECT_EQ(state.participant_id, 1u);
        EXPECT_EQ(state.participant_count, 2u);
        EXPECT_EQ(bank.epoch, 1u);
        EXPECT_EQ(bank.expert_count, 4u);
        EXPECT_EQ(bank.experts[2].logical_expert_id, 2);
        EXPECT_EQ(bank.experts[2].owner_participant, 0);
        EXPECT_EQ(bank.experts[2].local_slot, 2);
        EXPECT_TRUE(hasMoEExpertFlag(bank.experts[2].flags, DeviceMoEExpertFlags::Valid));
        EXPECT_EQ(bank.local_compute_mask[2], 1u);
        EXPECT_EQ(bank.replica_role[2], static_cast<uint8_t>(DeviceMoEReplicaRole::Primary));
        EXPECT_EQ(bank.resident_participant_mask[2], 0b11u)
            << "omitted resident masks synthesize owner plus local participant";
        EXPECT_EQ(bank.experts[2].gate.payload, update.experts[2].gate.payload);
        EXPECT_EQ(bank.experts[2].up.scales, update.experts[2].up.scales);
        EXPECT_EQ(bank.experts[2].down.n, update.experts[2].down.n);
    }

    TEST(Test__MoERuntimeTable, ExplicitResidentParticipantMasksRoundTripAndValidate)
    {
        MoERuntimeTable table(DeviceId::cpu(), 1, 4, 2);
        auto update = updateForEpoch(1, 4);
        update.participant_id = 2;
        update.participant_count = 3;
        update.experts[1].owner_participant = 1;
        update.resident_participant_mask = {
            0b101u,
            0b110u,
            0b101u,
            0b110u,
        };

        ASSERT_TRUE(table.prepareInactiveBank(0, update));
        const auto &bank = table.hostLayerState(0).banks[1];
        EXPECT_EQ(bank.resident_participant_mask[0], 0b101u);
        EXPECT_EQ(bank.resident_participant_mask[1], 0b110u);

        auto wrong_size = updateForEpoch(1, 4);
        wrong_size.participant_id = 0;
        wrong_size.participant_count = 2;
        wrong_size.resident_participant_mask = {0b01u, 0b10u};
        EXPECT_THROW(table.prepareInactiveBank(0, wrong_size), std::invalid_argument);

        auto outside_domain = updateForEpoch(1, 4);
        outside_domain.participant_id = 0;
        outside_domain.participant_count = 2;
        outside_domain.resident_participant_mask.assign(4, 0b101u);
        EXPECT_THROW(table.prepareInactiveBank(0, outside_domain), std::invalid_argument);

        auto missing_local = updateForEpoch(1, 4);
        missing_local.participant_id = 1;
        missing_local.participant_count = 2;
        missing_local.resident_participant_mask.assign(4, 0b01u);
        EXPECT_THROW(table.prepareInactiveBank(0, missing_local), std::invalid_argument);

        auto missing_owner = updateForEpoch(1, 4);
        missing_owner.participant_id = 0;
        missing_owner.participant_count = 2;
        missing_owner.experts[1].owner_participant = 1;
        missing_owner.resident_participant_mask.assign(4, 0b01u);
        EXPECT_THROW(table.prepareInactiveBank(0, missing_owner), std::invalid_argument);
    }

    TEST(Test__MoERuntimeTable, PlacementBankTracksMultiResidentExpertCountForHotCacheStats)
    {
        MoERuntimeTable table(DeviceId::cpu(), 1, 4, 2);

        MoEPlacementUpdate static_update;
        static_update.epoch = 1;
        static_update.expert_count = 4;
        static_update.participant_id = 0;
        static_update.participant_count = 2;
        for (int expert = 0; expert < 4; ++expert)
        {
            const int owner = expert / 2;
            static_update.experts.push_back(
                owner == 0 ? expertDesc(expert, owner, expert)
                           : expertOwnerOnlyDesc(expert, owner));
            static_update.local_compute_mask.push_back(owner == 0 ? 1u : 0u);
            static_update.replica_role.push_back(static_cast<uint8_t>(
                owner == 0 ? DeviceMoEReplicaRole::Primary : DeviceMoEReplicaRole::None));
        }

        ASSERT_TRUE(table.prepareInactiveBank(0, static_update));
        EXPECT_EQ(table.hostLayerState(0).banks[1].multi_resident_expert_count, 0u)
            << "Static apportioned placement must not enable hot-cache router-stat work.";
        ASSERT_TRUE(table.flipActiveBank(0, 1, nullptr));
        EXPECT_EQ(table.hostLayerState(0).banks[table.hostLayerState(0).active_bank].multi_resident_expert_count, 0u);

        auto hot_update = static_update;
        hot_update.epoch = 2;
        hot_update.participant_count = 3;
        hot_update.resident_participant_mask = {0b001u, 0b001u, 0b110u, 0b110u};
        hot_update.experts[2].owner_participant = 1;
        hot_update.experts[3].owner_participant = 1;
        hot_update.local_compute_mask = {1u, 1u, 0u, 0u};
        hot_update.replica_role = {
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
            static_cast<uint8_t>(DeviceMoEReplicaRole::Replica),
            static_cast<uint8_t>(DeviceMoEReplicaRole::Replica),
        };

        ASSERT_TRUE(table.prepareInactiveBank(0, hot_update));
        EXPECT_EQ(table.hostLayerState(0).banks[0].multi_resident_expert_count, 2u)
            << "Only multi-resident experts should enable hot-cache router-stat accounting.";
        ASSERT_TRUE(table.flipActiveBank(0, 2, nullptr));
        EXPECT_EQ(table.hostLayerState(0).banks[table.hostLayerState(0).active_bank].multi_resident_expert_count, 2u);
    }

    TEST(Test__MoERuntimeTable, FlipActiveBankAdvancesEpochWithoutChangingLayerPointer)
    {
        MoERuntimeTable table(DeviceId::cpu(), 1, 4, 2);
        auto *stable_ptr = table.deviceLayerState(0);

        ASSERT_TRUE(table.prepareInactiveBank(0, updateForEpoch(1, 4)));
        ASSERT_TRUE(table.flipActiveBank(0, 1, nullptr));

        auto *after_first_flip = table.deviceLayerState(0);
        ASSERT_EQ(stable_ptr, after_first_flip);
        EXPECT_EQ(after_first_flip->active_bank, 1u);
        EXPECT_EQ(after_first_flip->active_epoch, 1u);
        EXPECT_EQ(after_first_flip->banks[1].epoch, 1u);

        auto second_update = updateForEpoch(2, 4);
        second_update.experts[3].owner_participant = 7;
        second_update.experts[3].local_slot = 11;
        second_update.replica_role[3] = static_cast<uint8_t>(DeviceMoEReplicaRole::PreferredReplica);
        ASSERT_TRUE(table.prepareInactiveBank(0, second_update));
        ASSERT_TRUE(table.flipActiveBank(0, 2, nullptr));

        auto *after_second_flip = table.deviceLayerState(0);
        ASSERT_EQ(stable_ptr, after_second_flip);
        EXPECT_EQ(after_second_flip->active_bank, 0u);
        EXPECT_EQ(after_second_flip->active_epoch, 2u);
        EXPECT_EQ(after_second_flip->banks[0].experts[3].owner_participant, 7);
        EXPECT_EQ(after_second_flip->banks[0].experts[3].local_slot, 11);
        EXPECT_EQ(after_second_flip->banks[0].replica_role[3],
                  static_cast<uint8_t>(DeviceMoEReplicaRole::PreferredReplica));
    }

    TEST(Test__MoERuntimeTable, StableLayerPointerObservesActiveBankMaskFlip)
    {
        MoERuntimeTable table(DeviceId::cpu(), 1, 4, 2);
        auto *captured_runtime_ptr = table.deviceLayerState(0);

        auto first_update = updateForEpoch(1, 4);
        first_update.local_compute_mask = {1, 0, 1, 0};
        ASSERT_TRUE(table.prepareInactiveBank(0, first_update));
        ASSERT_TRUE(table.flipActiveBank(0, 1, nullptr));

        auto active_mask = [&]()
        {
            const auto &bank = captured_runtime_ptr->banks[captured_runtime_ptr->active_bank];
            return std::vector<uint8_t>(
                bank.local_compute_mask,
                bank.local_compute_mask + captured_runtime_ptr->expert_count);
        };

        ASSERT_EQ(captured_runtime_ptr, table.deviceLayerState(0));
        EXPECT_EQ(captured_runtime_ptr->active_epoch, 1u);
        EXPECT_EQ(active_mask(), (std::vector<uint8_t>{1, 0, 1, 0}));

        auto second_update = updateForEpoch(2, 4);
        second_update.local_compute_mask = {0, 1, 0, 1};
        ASSERT_TRUE(table.prepareInactiveBank(0, second_update));
        ASSERT_TRUE(table.flipActiveBank(0, 2, nullptr));

        ASSERT_EQ(captured_runtime_ptr, table.deviceLayerState(0));
        EXPECT_EQ(captured_runtime_ptr->active_epoch, 2u);
        EXPECT_EQ(active_mask(), (std::vector<uint8_t>{0, 1, 0, 1}));
    }

    TEST(Test__MoERuntimeTable, ResetDecodeHistogramCountsPreservesPlacementBanks)
    {
        MoERuntimeTable table(DeviceId::cpu(), 1, 4, 2);
        auto *captured_runtime_ptr = table.deviceLayerState(0);

        auto update = updateForEpoch(1, 4);
        update.local_compute_mask = {1, 0, 1, 0};
        update.experts[2].owner_participant = 5;
        update.experts[2].local_slot = 9;
        update.replica_role[2] = static_cast<uint8_t>(DeviceMoEReplicaRole::PreferredReplica);
        ASSERT_TRUE(table.prepareInactiveBank(0, update));
        ASSERT_TRUE(table.flipActiveBank(0, 1, nullptr));

        table.hostLayerState(0).decode_histogram[0] = 7;
        table.hostLayerState(0).decode_histogram[2] = 3;
        table.hostLayerState(0).decode_local_histogram[0] = 5;
        table.hostLayerState(0).decode_local_histogram[2] = 1;
        table.hostLayerState(0).router_hot_cache_eligible_dispatches = 4;
        table.hostLayerState(0).router_hot_cache_used_dispatches = 2;
        table.hostLayerState(0).router_hot_cache_improved_dispatches = 1;
        table.hostLayerState(0).router_hot_cache_default_load_spread_total = 9;
        table.hostLayerState(0).router_hot_cache_actual_load_spread_total = 5;
        table.hostLayerState(0).router_hot_cache_load_spread_improvement_total = 4;
        table.hostLayerState(0).router_hot_cache_active_dispatches = 8;
        table.hostLayerState(0).router_hot_cache_miss_dispatches = 6;
        table.hostLayerState(0).router_hot_cache_selected_expert_slots = 16;
        table.hostLayerState(0).router_hot_cache_replicated_selected_expert_slots = 4;

        /**
         * Request/session boundaries clear runtime counters but must not clear
         * the placement bank observed by graph-captured MoE stages.  Destroying
         * that bank leaves captured verifier graphs pointing at a table whose
         * active descriptors no longer describe the resident expert weights.
         */
        table.resetDecodeHistogramCounts();

        const auto *after_reset = table.deviceLayerState(0);
        ASSERT_EQ(captured_runtime_ptr, after_reset)
            << "Histogram reset must preserve the graph-facing layer pointer.";
        ASSERT_EQ(after_reset->active_bank, 1u);
        ASSERT_EQ(after_reset->active_epoch, 1u);

        const auto &active = after_reset->banks[after_reset->active_bank];
        EXPECT_EQ(active.epoch, 1u);
        EXPECT_EQ(active.experts[2].owner_participant, 5);
        EXPECT_EQ(active.experts[2].local_slot, 9);
        EXPECT_EQ(active.local_compute_mask[0], 1u);
        EXPECT_EQ(active.local_compute_mask[1], 0u);
        EXPECT_EQ(active.local_compute_mask[2], 1u);
        EXPECT_EQ(active.local_compute_mask[3], 0u);
        EXPECT_EQ(active.replica_role[2],
                  static_cast<uint8_t>(DeviceMoEReplicaRole::PreferredReplica));

        for (int expert = 0; expert < 4; ++expert)
        {
            EXPECT_EQ(after_reset->decode_histogram[expert], 0u);
            EXPECT_EQ(after_reset->decode_local_histogram[expert], 0u);
        }
        EXPECT_EQ(after_reset->router_hot_cache_eligible_dispatches, 0u);
        EXPECT_EQ(after_reset->router_hot_cache_used_dispatches, 0u);
        EXPECT_EQ(after_reset->router_hot_cache_improved_dispatches, 0u);
        EXPECT_EQ(after_reset->router_hot_cache_default_load_spread_total, 0u);
        EXPECT_EQ(after_reset->router_hot_cache_actual_load_spread_total, 0u);
        EXPECT_EQ(after_reset->router_hot_cache_load_spread_improvement_total, 0u);
        EXPECT_EQ(after_reset->router_hot_cache_active_dispatches, 0u);
        EXPECT_EQ(after_reset->router_hot_cache_miss_dispatches, 0u);
        EXPECT_EQ(after_reset->router_hot_cache_selected_expert_slots, 0u);
        EXPECT_EQ(after_reset->router_hot_cache_replicated_selected_expert_slots, 0u);
    }

    TEST(Test__MoERuntimeTable, RestoreInitialRuntimeStateRevertsMovedExpertsAndClearsCounters)
    {
        MoERuntimeTable table(DeviceId::cpu(), 1, 4, 2);
        auto *captured_runtime_ptr = table.deviceLayerState(0);

        auto masked_update = [](uint32_t epoch, std::vector<uint8_t> local_mask)
        {
            MoEPlacementUpdate update;
            update.epoch = epoch;
            update.expert_count = 4;
            update.participant_id = 0;
            update.participant_count = 2;
            update.local_compute_mask = std::move(local_mask);
            update.experts.resize(4);
            update.replica_role.assign(4, static_cast<uint8_t>(DeviceMoEReplicaRole::None));
            update.resident_participant_mask.assign(4, 0u);

            for (int expert = 0; expert < 4; ++expert)
            {
                const int owner = expert / 2;
                update.resident_participant_mask[static_cast<size_t>(expert)] =
                    1u << static_cast<uint32_t>(owner);

                if (update.local_compute_mask[static_cast<size_t>(expert)] != 0u)
                {
                    update.experts[static_cast<size_t>(expert)] = expertDesc(expert, owner, expert);
                    update.resident_participant_mask[static_cast<size_t>(expert)] |= 1u;
                    update.replica_role[static_cast<size_t>(expert)] =
                        static_cast<uint8_t>(owner == 0
                                                 ? DeviceMoEReplicaRole::Primary
                                                 : DeviceMoEReplicaRole::Replica);
                }
                else
                {
                    update.experts[static_cast<size_t>(expert)] =
                        expertOwnerOnlyDesc(expert, owner);
                }
            }

            return update;
        };

        EXPECT_FALSE(table.hasInitialRuntimeState());
        ASSERT_TRUE(table.prepareInactiveBank(0, masked_update(1, {1u, 1u, 0u, 0u})));
        ASSERT_TRUE(table.flipActiveBank(0, 1, nullptr));
        EXPECT_TRUE(table.hasInitialRuntimeState());

        ASSERT_EQ(captured_runtime_ptr, table.deviceLayerState(0));
        EXPECT_EQ(captured_runtime_ptr->active_epoch, 1u);
        EXPECT_EQ(captured_runtime_ptr->banks[captured_runtime_ptr->active_bank].local_compute_mask[2], 0u);

        ASSERT_TRUE(table.prepareInactiveBank(0, masked_update(2, {1u, 1u, 1u, 0u})));
        ASSERT_TRUE(table.flipActiveBank(0, 2, nullptr));

        auto &moved_state = table.hostLayerState(0);
        moved_state.decode_histogram[2] = 17;
        moved_state.decode_local_histogram[2] = 9;
        moved_state.router_hot_cache_used_dispatches = 3;
        moved_state.reserved_u64[2] = 11;
        moved_state.reserved_u64[3] = 5;

        ASSERT_EQ(captured_runtime_ptr, table.deviceLayerState(0));
        EXPECT_EQ(captured_runtime_ptr->active_epoch, 2u);
        EXPECT_EQ(captured_runtime_ptr->banks[captured_runtime_ptr->active_bank].local_compute_mask[2], 1u);
        EXPECT_EQ(captured_runtime_ptr->banks[captured_runtime_ptr->active_bank].resident_participant_mask[2],
                  0b11u);

        table.restoreInitialRuntimeState();

        const auto *restored = table.deviceLayerState(0);
        ASSERT_EQ(captured_runtime_ptr, restored)
            << "Placement restore must preserve the graph-facing runtime table pointer.";
        ASSERT_EQ(restored->active_epoch, 1u);
        ASSERT_EQ(restored->participant_id, 0u);
        ASSERT_EQ(restored->participant_count, 2u);

        const auto &bank = restored->banks[restored->active_bank];
        EXPECT_EQ(bank.local_compute_mask[0], 1u);
        EXPECT_EQ(bank.local_compute_mask[1], 1u);
        EXPECT_EQ(bank.local_compute_mask[2], 0u);
        EXPECT_EQ(bank.local_compute_mask[3], 0u);
        EXPECT_EQ(bank.resident_participant_mask[0], 0b01u);
        EXPECT_EQ(bank.resident_participant_mask[1], 0b01u);
        EXPECT_EQ(bank.resident_participant_mask[2], 0b10u);
        EXPECT_EQ(bank.resident_participant_mask[3], 0b10u);
        EXPECT_EQ(restored->decode_histogram[2], 0u);
        EXPECT_EQ(restored->decode_local_histogram[2], 0u);
        EXPECT_EQ(restored->router_hot_cache_used_dispatches, 0u);
        EXPECT_EQ(restored->reserved_u64[2], 0u);
        EXPECT_EQ(restored->reserved_u64[3], 0u);
    }

    TEST(Test__MoERuntimeTable, ResetDecodeRuntimeStatePreservesEveryGraphOwnedScratchBinding)
    {
        MoERuntimeTable table(DeviceId::cpu(), 1, 4, 2);
        ASSERT_TRUE(table.prepareInactiveBank(0, updateForEpoch(1, 4)));
        ASSERT_TRUE(table.flipActiveBank(0, 1, nullptr));

        auto &before = table.hostLayerState(0);
        before.route_expert_ids = reinterpret_cast<int32_t *>(0x1000u);
        before.route_weights = reinterpret_cast<float *>(0x2000u);
        before.route_participant_ids = reinterpret_cast<int32_t *>(0x3000u);
        before.expert_counts = reinterpret_cast<int32_t *>(0x4000u);
        before.expert_offsets = reinterpret_cast<int32_t *>(0x5000u);
        before.grouped_token_ids = reinterpret_cast<int32_t *>(0x6000u);
        before.grouped_route_weights = reinterpret_cast<float *>(0x7000u);
        before.grouped_gate_scratch = reinterpret_cast<float *>(0x8000u);
        before.grouped_up_scratch = reinterpret_cast<float *>(0x9000u);
        before.grouped_output_partials = reinterpret_cast<float *>(0xA000u);
        before.decode_scratch = reinterpret_cast<void *>(0xB000u);
        before.reserved_ptrs[0] = reinterpret_cast<void *>(0xC000u);
        before.reserved_ptrs[1] = reinterpret_cast<void *>(0xD000u);
        before.reserved_ptrs[2] = reinterpret_cast<void *>(0xE000u);
        before.reserved_u64[0] = 96u;
        before.reserved_u64[1] = 96u;
        before.reserved_u64[2] = 17u;
        before.reserved_u64[3] = 19u;
        before.prefill_token_capacity = 48u;
        before.prefill_route_capacity = 96u;
        before.decode_histogram[2] = 23u;
        before.decode_local_histogram[2] = 11u;
        before.router_hot_cache_used_dispatches = 7u;

        table.resetDecodeRuntimeState();

        EXPECT_TRUE(table.decodeRuntimePublicationRequired(0));
        const auto &after = table.hostLayerState(0);
        EXPECT_EQ(after.active_epoch, 0u);
        EXPECT_EQ(after.route_expert_ids, reinterpret_cast<int32_t *>(0x1000u));
        EXPECT_EQ(after.route_weights, reinterpret_cast<float *>(0x2000u));
        EXPECT_EQ(after.route_participant_ids, reinterpret_cast<int32_t *>(0x3000u));
        EXPECT_EQ(after.expert_counts, reinterpret_cast<int32_t *>(0x4000u));
        EXPECT_EQ(after.expert_offsets, reinterpret_cast<int32_t *>(0x5000u));
        EXPECT_EQ(after.grouped_token_ids, reinterpret_cast<int32_t *>(0x6000u));
        EXPECT_EQ(after.grouped_route_weights, reinterpret_cast<float *>(0x7000u));
        EXPECT_EQ(after.grouped_gate_scratch, reinterpret_cast<float *>(0x8000u));
        EXPECT_EQ(after.grouped_up_scratch, reinterpret_cast<float *>(0x9000u));
        EXPECT_EQ(after.grouped_output_partials, reinterpret_cast<float *>(0xA000u));
        EXPECT_EQ(after.decode_scratch, reinterpret_cast<void *>(0xB000u));
        EXPECT_EQ(after.reserved_ptrs[0], reinterpret_cast<void *>(0xC000u));
        EXPECT_EQ(after.reserved_ptrs[1], reinterpret_cast<void *>(0xD000u));
        EXPECT_EQ(after.reserved_ptrs[2], reinterpret_cast<void *>(0xE000u));
        EXPECT_EQ(after.reserved_u64[0], 96u);
        EXPECT_EQ(after.reserved_u64[1], 96u);
        EXPECT_EQ(after.prefill_token_capacity, 48u);
        EXPECT_EQ(after.prefill_route_capacity, 96u);
        EXPECT_EQ(after.reserved_u64[2], 0u);
        EXPECT_EQ(after.reserved_u64[3], 0u);
        EXPECT_EQ(after.decode_histogram[2], 0u);
        EXPECT_EQ(after.decode_local_histogram[2], 0u);
        EXPECT_EQ(after.router_hot_cache_used_dispatches, 0u);

        /*
         * A reset starts a fresh publication generation.  Reusing epoch one is
         * intentional: monotonicity applies within a generation, while the
         * reset has already removed the previous bank from live execution.
         */
        ASSERT_TRUE(table.prepareInactiveBank(0, updateForEpoch(1, 4)));
        ASSERT_TRUE(table.flipActiveBank(0, 1, nullptr));
        EXPECT_FALSE(table.decodeRuntimePublicationRequired(0));
        const auto &republished = table.hostLayerState(0);
        EXPECT_EQ(republished.active_epoch, 1u);
        EXPECT_EQ(republished.route_expert_ids, reinterpret_cast<int32_t *>(0x1000u));
        EXPECT_EQ(republished.reserved_ptrs[2], reinterpret_cast<void *>(0xE000u));
    }

    TEST(Test__MoERuntimeTable, PortableRuntimeStatePreservesTransientLogicalPlacementWithoutPointers)
    {
        MoERuntimeTable table(DeviceId::cpu(), 1, 4, 2);

        auto masked_update = [](uint32_t epoch,
                                std::vector<uint8_t> local_mask,
                                bool transfer_expert_2 = false)
        {
            MoEPlacementUpdate update;
            update.epoch = epoch;
            update.expert_count = 4;
            update.participant_id = 0;
            update.participant_count = 2;
            update.local_compute_mask = std::move(local_mask);
            update.experts.resize(4);
            update.replica_role.assign(4, static_cast<uint8_t>(DeviceMoEReplicaRole::None));
            update.resident_participant_mask.assign(4, 0u);

            for (int expert = 0; expert < 4; ++expert)
            {
                const int owner = expert / 2;
                update.resident_participant_mask[static_cast<size_t>(expert)] =
                    1u << static_cast<uint32_t>(owner);
                if (update.local_compute_mask[static_cast<size_t>(expert)] == 0u)
                {
                    update.experts[static_cast<size_t>(expert)] =
                        expertOwnerOnlyDesc(expert, owner);
                    continue;
                }

                DeviceMoEExpertFlags flags =
                    DeviceMoEExpertFlags::Valid |
                    DeviceMoEExpertFlags::Resident |
                    DeviceMoEExpertFlags::LocalCompute;
                if (owner == 0)
                    flags |= DeviceMoEExpertFlags::PreferredOwner;
                if (transfer_expert_2 && expert == 2)
                    flags |= DeviceMoEExpertFlags::TransferSlot;
                update.experts[static_cast<size_t>(expert)] =
                    expertDesc(expert, owner, expert, flags);
                update.resident_participant_mask[static_cast<size_t>(expert)] |= 0b01u;
                update.replica_role[static_cast<size_t>(expert)] =
                    static_cast<uint8_t>(owner == 0
                                             ? DeviceMoEReplicaRole::Primary
                                             : DeviceMoEReplicaRole::Replica);
            }
            return update;
        };

        ASSERT_TRUE(table.prepareInactiveBank(0, masked_update(1, {1u, 1u, 0u, 0u})));
        ASSERT_TRUE(table.flipActiveBank(0, 1, nullptr));

        ASSERT_TRUE(table.prepareInactiveBank(0, masked_update(2, {1u, 1u, 1u, 0u}, true)));
        ASSERT_TRUE(table.flipActiveBank(0, 2, nullptr));
        table.hostLayerState(0).decode_histogram[2] = 17;
        table.hostLayerState(0).decode_local_histogram[2] = 9;

        std::vector<DeviceMoEPortableLayerRuntimeState> snapshot;
        ASSERT_TRUE(table.capturePortableRuntimeState(snapshot));
        ASSERT_EQ(snapshot.size(), 1u);
        EXPECT_EQ(snapshot[0].requires_device_payload_rehydration, 1u);
        EXPECT_EQ(snapshot[0].experts[2].local_compute, 1u);
        EXPECT_EQ(snapshot[0].experts[2].local_slot, -1)
            << "Portable placement must not serialize rolling transfer-slot identity.";
        EXPECT_EQ(snapshot[0].experts[2].replica_role,
                  static_cast<uint8_t>(DeviceMoEReplicaRole::Replica));
        EXPECT_EQ(snapshot[0].experts[2].resident_participant_mask, 0b11u);
        EXPECT_FALSE(hasMoEExpertFlag(snapshot[0].experts[2].flags,
                                      DeviceMoEExpertFlags::TransferSlot))
            << "Portable placement must never serialize a rolling payload pointer.";
        EXPECT_EQ(snapshot[0].selected_histogram[2], 17u);
        EXPECT_EQ(snapshot[0].local_histogram[2], 9u)
            << "Exact logical placement must retain request-owned routing evidence.";

        EXPECT_FALSE(table.restorePortableRuntimeState(snapshot))
            << "CPU unit tests must not emulate the captured GPU payload-rehydration transaction.";
    }

    /**
     * @brief Canonicalize legal cross-layer reuse of one rolling physical slot.
     *
     * Current-batch LLEP layers execute serially and may reuse the same transfer
     * allocation during non-overlapping intervals. A portable prefix snapshot
     * must preserve both layers' logical replica placement without treating the
     * graph-lifetime subscript as a durable identity. Rehydration can then lease
     * independent live slots for the concurrently restored logical residency.
     */
    TEST(Test__MoERuntimeTable, PortableRuntimeStateCanonicalizesCrossLayerRollingSlotReuse)
    {
        constexpr int kLayerCount = 2;
        constexpr int kExpertCount = 4;
        constexpr int kSharedTransferSlot = 2;
        MoERuntimeTable table(
            DeviceId::cpu(),
            kLayerCount,
            kExpertCount,
            /*top_k=*/2);

        auto placement = [=](uint32_t epoch, bool include_replica)
        {
            MoEPlacementUpdate update;
            update.epoch = epoch;
            update.expert_count = kExpertCount;
            update.participant_id = 0;
            update.participant_count = 2;
            update.experts.resize(kExpertCount);
            update.local_compute_mask.assign(kExpertCount, 0u);
            update.replica_role.assign(
                kExpertCount,
                static_cast<uint8_t>(DeviceMoEReplicaRole::None));
            update.resident_participant_mask = {0b01u, 0b01u, 0b10u, 0b10u};

            for (int expert = 0; expert < kExpertCount; ++expert)
            {
                const int owner = expert < 2 ? 0 : 1;
                if (owner == 0)
                {
                    update.experts[static_cast<size_t>(expert)] =
                        expertDesc(expert, owner, expert);
                    update.local_compute_mask[static_cast<size_t>(expert)] = 1u;
                    update.replica_role[static_cast<size_t>(expert)] =
                        static_cast<uint8_t>(DeviceMoEReplicaRole::Primary);
                }
                else
                {
                    update.experts[static_cast<size_t>(expert)] =
                        expertOwnerOnlyDesc(expert, owner);
                }
            }

            if (include_replica)
            {
                constexpr int kReplicatedExpert = 2;
                update.experts[kReplicatedExpert] = expertDesc(
                    kReplicatedExpert,
                    /*owner=*/1,
                    kSharedTransferSlot,
                    DeviceMoEExpertFlags::Valid |
                        DeviceMoEExpertFlags::Resident |
                        DeviceMoEExpertFlags::LocalCompute |
                        DeviceMoEExpertFlags::TransferSlot);
                update.local_compute_mask[kReplicatedExpert] = 1u;
                update.replica_role[kReplicatedExpert] =
                    static_cast<uint8_t>(DeviceMoEReplicaRole::Replica);
                update.resident_participant_mask[kReplicatedExpert] = 0b11u;
            }
            return update;
        };

        for (int layer = 0; layer < kLayerCount; ++layer)
        {
            ASSERT_TRUE(table.prepareInactiveBank(
                layer,
                placement(/*epoch=*/1, /*include_replica=*/false)));
            ASSERT_TRUE(table.flipActiveBank(layer, /*epoch=*/1, nullptr));
            ASSERT_TRUE(table.prepareInactiveBank(
                layer,
                placement(/*epoch=*/2, /*include_replica=*/true)));
            ASSERT_TRUE(table.flipActiveBank(layer, /*epoch=*/2, nullptr));
        }

        std::vector<DeviceMoEPortableLayerRuntimeState> snapshot;
        ASSERT_TRUE(table.capturePortableRuntimeState(snapshot));
        ASSERT_EQ(snapshot.size(), kLayerCount);
        for (int layer = 0; layer < kLayerCount; ++layer)
        {
            const auto &portable_layer = snapshot[static_cast<size_t>(layer)];
            EXPECT_EQ(portable_layer.requires_device_payload_rehydration, 1u);
            ASSERT_EQ(portable_layer.experts.size(), kExpertCount);
            EXPECT_EQ(portable_layer.experts[2].local_compute, 1u);
            EXPECT_EQ(portable_layer.experts[2].local_slot, -1)
                << "Layer " << layer
                << " retained a graph-lifetime rolling slot in portable state.";
        }
    }

    /**
     * @brief Prove portable placement cannot diverge by participant-local slot ownership.
     *
     * A real LLEP replica is globally visible through the resident participant
     * mask, but its rolling transfer-slot descriptor exists only on the
     * destination participant. Prefix capture must consequently preserve its
     * logical placement from a domain-wide invariant, not from the local
     * TransferSlot flag. This regression models both participants after the
     * same participant-1 to participant-0 replication wave and requires both
     * cache blobs to publish the same exact residency plus the same rehydration
     * requirement.
     */
    TEST(Test__MoERuntimeTable, PortableRuntimeStatePreservesRemoteReplicaOnEveryParticipant)
    {
        auto placement_for_participant =
            [](uint32_t epoch,
               uint32_t participant,
               bool include_transient_replica)
        {
            MoEPlacementUpdate update;
            update.epoch = epoch;
            update.expert_count = 4;
            update.participant_id = participant;
            update.participant_count = 2;
            update.experts.resize(4);
            update.local_compute_mask.assign(4, 0u);
            update.replica_role.assign(
                4,
                static_cast<uint8_t>(DeviceMoEReplicaRole::None));
            update.resident_participant_mask.assign(4, 0u);

            for (int expert = 0; expert < 4; ++expert)
            {
                const int owner = expert / 2;
                const bool owner_local =
                    owner == static_cast<int>(participant);
                update.resident_participant_mask[static_cast<size_t>(expert)] =
                    1u << static_cast<uint32_t>(owner);
                if (!owner_local)
                {
                    update.experts[static_cast<size_t>(expert)] =
                        expertOwnerOnlyDesc(expert, owner);
                    continue;
                }

                update.experts[static_cast<size_t>(expert)] =
                    expertDesc(expert, owner, expert);
                update.local_compute_mask[static_cast<size_t>(expert)] = 1u;
                update.replica_role[static_cast<size_t>(expert)] =
                    static_cast<uint8_t>(DeviceMoEReplicaRole::Primary);
            }

            if (include_transient_replica)
            {
                update.transient_placement_observed = true;
                update.resident_participant_mask[2] = 0b11u;
                if (participant == 0u)
                {
                    update.experts[2] =
                        expertDesc(
                            2,
                            /*owner=*/1,
                            /*local_slot=*/2,
                            DeviceMoEExpertFlags::Valid |
                                DeviceMoEExpertFlags::Resident |
                                DeviceMoEExpertFlags::LocalCompute |
                                DeviceMoEExpertFlags::TransferSlot);
                    update.local_compute_mask[2] = 1u;
                    update.replica_role[2] =
                        static_cast<uint8_t>(DeviceMoEReplicaRole::Replica);
                }
            }
            return update;
        };

        MoERuntimeTable destination(DeviceId::cpu(), 1, 4, 2);
        MoERuntimeTable source(DeviceId::cpu(), 1, 4, 2);

        ASSERT_TRUE(destination.prepareInactiveBank(
            0,
            placement_for_participant(
                /*epoch=*/1,
                /*participant=*/0,
                /*include_transient_replica=*/false)));
        ASSERT_TRUE(destination.flipActiveBank(0, 1, nullptr));
        ASSERT_TRUE(source.prepareInactiveBank(
            0,
            placement_for_participant(
                /*epoch=*/1,
                /*participant=*/1,
                /*include_transient_replica=*/false)));
        ASSERT_TRUE(source.flipActiveBank(0, 1, nullptr));

        ASSERT_TRUE(destination.prepareInactiveBank(
            0,
            placement_for_participant(
                /*epoch=*/2,
                /*participant=*/0,
                /*include_transient_replica=*/true)));
        ASSERT_TRUE(destination.flipActiveBank(0, 2, nullptr));
        ASSERT_TRUE(source.prepareInactiveBank(
            0,
            placement_for_participant(
                /*epoch=*/2,
                /*participant=*/1,
                /*include_transient_replica=*/true)));
        ASSERT_TRUE(source.flipActiveBank(0, 2, nullptr));

        ASSERT_TRUE(deviceMoELayerUsesTransientLocalPayload(
            destination.hostLayerState(0)));
        ASSERT_FALSE(deviceMoELayerUsesTransientLocalPayload(
            source.hostLayerState(0)))
            << "The source sees global replica residency but owns no rolling "
               "destination payload slot.";

        std::vector<DeviceMoEPortableLayerRuntimeState> destination_snapshot;
        std::vector<DeviceMoEPortableLayerRuntimeState> source_snapshot;
        ASSERT_TRUE(destination.capturePortableRuntimeState(destination_snapshot));
        ASSERT_TRUE(source.capturePortableRuntimeState(source_snapshot));
        ASSERT_EQ(destination_snapshot.size(), 1u);
        ASSERT_EQ(source_snapshot.size(), 1u);

        EXPECT_EQ(
            destination_snapshot[0].experts[2].resident_participant_mask,
            0b11u);
        EXPECT_EQ(
            source_snapshot[0].experts[2].resident_participant_mask,
            0b11u);
        EXPECT_EQ(
            destination_snapshot[0].experts[2].resident_participant_mask,
            source_snapshot[0].experts[2].resident_participant_mask)
            << "Every participant must serialize one identical global "
               "residency contract for the restored suffix.";
        EXPECT_EQ(
            destination_snapshot[0].requires_device_payload_rehydration,
            1u);
        EXPECT_EQ(
            source_snapshot[0].requires_device_payload_rehydration,
            1u);
        EXPECT_EQ(destination_snapshot[0].experts[2].local_compute, 1u);
        EXPECT_EQ(source_snapshot[0].experts[2].local_compute, 1u)
            << "The immutable owner and reconstructed replica both retain local compute.";
        EXPECT_FALSE(hasMoEExpertFlag(
            destination_snapshot[0].experts[2].flags,
            DeviceMoEExpertFlags::TransferSlot));
        EXPECT_FALSE(hasMoEExpertFlag(
            source_snapshot[0].experts[2].flags,
            DeviceMoEExpertFlags::TransferSlot));
    }

    TEST(Test__MoERuntimeTable, PortableRuntimeStateRestoreFailsWhenLocalPayloadIsUnavailable)
    {
        MoERuntimeTable source(DeviceId::cpu(), 1, 4, 2);
        auto initial = updateForEpoch(1, 4);
        initial.participant_id = 0;
        initial.participant_count = 2;
        initial.local_compute_mask = {1u, 1u, 0u, 0u};
        initial.experts[1] = expertDesc(1, 0, 1);
        initial.experts[2] = expertOwnerOnlyDesc(2, 1);
        initial.experts[3] = expertOwnerOnlyDesc(3, 1);
        initial.replica_role = {
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
        };
        initial.resident_participant_mask = {0b01u, 0b01u, 0b10u, 0b10u};
        ASSERT_TRUE(source.prepareInactiveBank(0, initial));
        ASSERT_TRUE(source.flipActiveBank(0, 1, nullptr));

        auto moved = initial;
        moved.epoch = 2;
        moved.experts[2] = expertDesc(2, 1, 2);
        moved.local_compute_mask[2] = 1u;
        moved.replica_role[2] = static_cast<uint8_t>(DeviceMoEReplicaRole::Replica);
        moved.resident_participant_mask[2] = 0b11u;
        ASSERT_TRUE(source.prepareInactiveBank(0, moved));
        ASSERT_TRUE(source.flipActiveBank(0, 2, nullptr));

        std::vector<DeviceMoEPortableLayerRuntimeState> snapshot;
        ASSERT_TRUE(source.capturePortableRuntimeState(snapshot));

        MoERuntimeTable destination(DeviceId::cpu(), 1, 4, 2);
        ASSERT_TRUE(destination.prepareInactiveBank(0, initial));
        ASSERT_TRUE(destination.flipActiveBank(0, 1, nullptr));

        EXPECT_FALSE(destination.restorePortableRuntimeState(snapshot))
            << "A portable prefix state that requires local expert payloads must fail fast when the live runner has no descriptor to bind.";
    }

    TEST(Test__MoERuntimeTable, PortableRuntimeStateNeverResolvesTransientTransferSlots)
    {
        MoERuntimeTable source(DeviceId::cpu(), 1, 4, 2);
        auto initial = updateForEpoch(1, 4);
        initial.participant_id = 0;
        initial.participant_count = 2;
        initial.local_compute_mask = {1u, 1u, 0u, 0u};
        initial.experts[1] = expertDesc(1, 0, 1);
        initial.experts[2] = expertOwnerOnlyDesc(2, 1);
        initial.experts[3] = expertOwnerOnlyDesc(3, 1);
        initial.replica_role = {
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
        };
        initial.resident_participant_mask = {0b01u, 0b01u, 0b10u, 0b10u};
        ASSERT_TRUE(source.prepareInactiveBank(0, initial));
        ASSERT_TRUE(source.flipActiveBank(0, 1, nullptr));

        auto moved = initial;
        moved.epoch = 2;
        moved.experts[2] =
            expertDesc(2,
                       1,
                       2,
                       DeviceMoEExpertFlags::Valid |
                           DeviceMoEExpertFlags::Resident |
                           DeviceMoEExpertFlags::LocalCompute |
                           DeviceMoEExpertFlags::TransferSlot);
        moved.local_compute_mask[2] = 1u;
        moved.replica_role[2] = static_cast<uint8_t>(DeviceMoEReplicaRole::Replica);
        moved.resident_participant_mask[2] = 0b11u;
        ASSERT_TRUE(source.prepareInactiveBank(0, moved));
        ASSERT_TRUE(source.flipActiveBank(0, 2, nullptr));

        std::vector<DeviceMoEPortableLayerRuntimeState> snapshot;
        ASSERT_TRUE(source.capturePortableRuntimeState(snapshot));
        ASSERT_EQ(snapshot.size(), 1u);
        ASSERT_EQ(snapshot[0].requires_device_payload_rehydration, 1u);
        ASSERT_EQ(snapshot[0].experts[2].local_compute, 1u);
        ASSERT_EQ(snapshot[0].experts[2].local_slot, -1);

        MoERuntimeTable destination(DeviceId::cpu(), 1, 4, 2);
        ASSERT_TRUE(destination.prepareInactiveBank(0, initial));
        ASSERT_TRUE(destination.flipActiveBank(0, 1, nullptr));

        bool resolver_called = false;
        auto resolver =
            [&](int layer_idx,
                int expert,
                int local_slot,
                DeviceMoEExpertDescriptor &out) -> bool
        {
            resolver_called = true;
            EXPECT_EQ(layer_idx, 0);
            EXPECT_EQ(expert, 2);
            EXPECT_EQ(local_slot, 2);
            out = expertDesc(2,
                             1,
                             2,
                             DeviceMoEExpertFlags::Valid |
                                 DeviceMoEExpertFlags::Resident |
                                 DeviceMoEExpertFlags::TransferSlot);
            return true;
        };

        ASSERT_FALSE(destination.restorePortableRuntimeState(snapshot, nullptr, resolver));
        EXPECT_FALSE(resolver_called)
            << "A portable snapshot must never ask a rolling transfer-slot "
               "directory to resurrect bytes whose lifetime ended at capture.";
    }

    TEST(Test__MoERuntimeTable, RestoreRuntimeSnapshotSkipsUninitializedLayers)
    {
        MoERuntimeTable table(DeviceId::cpu(), 1, 4, 2);
        auto *captured_runtime_ptr = table.deviceLayerState(0);

        auto update = updateForEpoch(1, 4);
        update.participant_id = 1;
        update.participant_count = 2;
        update.resident_participant_mask.assign(4, 0u);
        for (int expert = 0; expert < 4; ++expert)
        {
            const int owner = expert % 2;
            update.experts[static_cast<size_t>(expert)].owner_participant = owner;
            update.resident_participant_mask[static_cast<size_t>(expert)] =
                (1u << static_cast<uint32_t>(owner)) | 0b10u;
            update.replica_role[static_cast<size_t>(expert)] =
                static_cast<uint8_t>(owner == 1
                                         ? DeviceMoEReplicaRole::Primary
                                         : DeviceMoEReplicaRole::Replica);
        }

        ASSERT_TRUE(table.prepareInactiveBank(0, update));
        ASSERT_TRUE(table.flipActiveBank(0, 1, nullptr));
        table.hostLayerState(0).decode_histogram[1] = 5;
        table.hostLayerState(0).router_hot_cache_used_dispatches = 7;

        DeviceMoELayerRuntime uninitialized_snapshot{};
        uninitialized_snapshot.expert_count = 4;
        uninitialized_snapshot.top_k = 2;

        table.restoreRuntimeStateSnapshot(&uninitialized_snapshot, 1, nullptr);

        const auto *restored = table.deviceLayerState(0);
        ASSERT_EQ(captured_runtime_ptr, restored);
        EXPECT_EQ(restored->active_epoch, 1u)
            << "A zero-epoch prefix snapshot means this layer had no cached "
               "runtime override; it must not erase graph-initialized placement.";
        EXPECT_EQ(restored->participant_id, 1u);
        EXPECT_EQ(restored->participant_count, 2u);

        const auto &bank = restored->banks[restored->active_bank];
        EXPECT_EQ(bank.local_compute_mask[0], 1u);
        EXPECT_EQ(bank.experts[0].logical_expert_id, 0);
        EXPECT_EQ(bank.experts[0].local_slot, 0);
        EXPECT_EQ(bank.resident_participant_mask[1], 0b10u);
        EXPECT_EQ(restored->decode_histogram[1], 0u);
        EXPECT_EQ(restored->router_hot_cache_used_dispatches, 0u);
    }

    TEST(Test__MoERuntimeTable, RuntimeLayerDetectsTransientTransferSlotPayloads)
    {
        MoERuntimeTable table(DeviceId::cpu(), 1, 4, 2);

        auto update = updateForEpoch(1, 4);
        update.participant_id = 0;
        update.participant_count = 2;
        update.resident_participant_mask.resize(4);
        for (int expert = 0; expert < 4; ++expert)
        {
            const auto &desc = update.experts[static_cast<size_t>(expert)];
            update.resident_participant_mask[static_cast<size_t>(expert)] =
                0b01u | (1u << static_cast<uint32_t>(desc.owner_participant));
        }
        ASSERT_TRUE(table.prepareInactiveBank(0, update));
        ASSERT_TRUE(table.flipActiveBank(0, 1, nullptr));
        EXPECT_FALSE(deviceMoELayerUsesTransientLocalPayload(table.hostLayerState(0)));
        EXPECT_EQ(
            deviceMoELayerActiveTransferSlotExpertCount(table.hostLayerState(0)),
            0u);

        auto transient = updateForEpoch(2, 4);
        transient.participant_id = 0;
        transient.participant_count = 2;
        transient.resident_participant_mask.resize(4);
        for (int expert = 0; expert < 4; ++expert)
        {
            const auto &desc = transient.experts[static_cast<size_t>(expert)];
            transient.resident_participant_mask[static_cast<size_t>(expert)] =
                0b01u | (1u << static_cast<uint32_t>(desc.owner_participant));
        }
        transient.experts[2].flags |=
            toMoEExpertFlags(DeviceMoEExpertFlags::TransferSlot);
        ASSERT_TRUE(table.prepareInactiveBank(0, transient));
        ASSERT_TRUE(table.flipActiveBank(0, 2, nullptr));
        EXPECT_TRUE(deviceMoELayerUsesTransientLocalPayload(table.hostLayerState(0)))
            << "Prefix-cache runtime snapshots must not replay metadata-only "
               "references to mutable transfer slot payloads.";
        EXPECT_EQ(
            deviceMoELayerActiveTransferSlotExpertCount(table.hostLayerState(0)),
            1u)
            << "Only active, valid, resident, locally executable transfer-slot "
               "experts are applied prefill-movement evidence.";

        const auto config = rebalanceConfig(
            /*layers=*/1,
            /*experts=*/4,
            /*top_k=*/2,
            /*participant=*/0,
            /*participants=*/2,
            /*window_tokens=*/1,
            /*max_hot_replicas=*/0);
        std::vector<uint64_t> gathered(
            static_cast<size_t>(config.participant_count) *
                static_cast<size_t>(config.num_layers) *
                static_cast<size_t>(config.num_experts),
            0u);
        addGatheredCount(gathered, config, 0, 0, 0, 2u);

        DeviceMoERebalanceStatus status;
        ASSERT_TRUE(applyDeviceMoERebalancePolicyHost(
            table.deviceLayerState(0),
            gathered.data(),
            config,
            &status));
        EXPECT_EQ(status.prefill_active_transfer_slot_experts, 1u)
            << "Request-boundary status must preserve final-state evidence from "
               "the preceding prefill publication independently of whether the "
               "new maintenance wave chooses additional movement.";
    }

    TEST(Test__MoERuntimeTable, InvalidLayerBoundsAndUpdatesThrowConsistently)
    {
        MoERuntimeTable table(DeviceId::cpu(), 1, 4, 2);

        EXPECT_THROW(table.deviceLayerState(-1), std::out_of_range);
        EXPECT_THROW(table.hostLayerState(1), std::out_of_range);
        EXPECT_THROW(table.prepareInactiveBank(2, updateForEpoch(1, 4)), std::out_of_range);
        EXPECT_THROW(table.flipActiveBank(0, 1, nullptr), std::runtime_error);

        auto wrong_count = updateForEpoch(1, 3);
        EXPECT_THROW(table.prepareInactiveBank(0, wrong_count), std::invalid_argument);

        auto bad_mask = updateForEpoch(1, 4);
        bad_mask.local_compute_mask.pop_back();
        EXPECT_THROW(table.prepareInactiveBank(0, bad_mask), std::invalid_argument);

        auto bad_participant_count = updateForEpoch(1, 4);
        bad_participant_count.participant_count = 0;
        EXPECT_THROW(table.prepareInactiveBank(0, bad_participant_count), std::invalid_argument);

        bad_participant_count = updateForEpoch(1, 4);
        bad_participant_count.participant_count = kDeviceMoEMaxParticipants + 1;
        EXPECT_THROW(table.prepareInactiveBank(0, bad_participant_count), std::invalid_argument);

        auto bad_participant_id = updateForEpoch(1, 4);
        bad_participant_id.participant_id = 2;
        bad_participant_id.participant_count = 2;
        EXPECT_THROW(table.prepareInactiveBank(0, bad_participant_id), std::invalid_argument);

        auto bad_replica_role = updateForEpoch(1, 4);
        bad_replica_role.replica_role[0] = 99;
        EXPECT_THROW(table.prepareInactiveBank(0, bad_replica_role), std::invalid_argument);

        auto bad_logical = updateForEpoch(1, 4);
        bad_logical.experts[2].logical_expert_id = 3;
        EXPECT_THROW(table.prepareInactiveBank(0, bad_logical), std::invalid_argument);

        auto missing_payload = updateForEpoch(1, 4);
        missing_payload.experts[1].gate.payload = nullptr;
        EXPECT_THROW(table.prepareInactiveBank(0, missing_payload), std::invalid_argument);
    }

    TEST(Test__MoERuntimeTable, EpochsMustBePreparedAndMonotonic)
    {
        MoERuntimeTable table(DeviceId::cpu(), 1, 4, 2);

        ASSERT_TRUE(table.prepareInactiveBank(0, updateForEpoch(2, 4)));
        EXPECT_THROW(table.flipActiveBank(0, 1, nullptr), std::invalid_argument);
        ASSERT_TRUE(table.flipActiveBank(0, 2, nullptr));

        EXPECT_THROW(table.prepareInactiveBank(0, updateForEpoch(2, 4)), std::invalid_argument);
        ASSERT_TRUE(table.prepareInactiveBank(0, updateForEpoch(3, 4)));
        EXPECT_THROW(table.flipActiveBank(0, 2, nullptr), std::invalid_argument);
        ASSERT_TRUE(table.flipActiveBank(0, 3, nullptr));
    }

    TEST(Test__MoERuntimeTable, SyncDecodeHistogramToHostMergesAndResetsCounts)
    {
        MoERuntimeTable table(DeviceId::cpu(), 2, 4, 2);
        table.hostLayerState(0).decode_histogram[0] = 3;
        table.hostLayerState(0).decode_histogram[2] = 1;
        table.hostLayerState(0).decode_local_histogram[0] = 2;
        table.hostLayerState(0).decode_local_histogram[2] = 1;
        table.hostLayerState(1).decode_histogram[1] = 2;
        table.hostLayerState(1).decode_histogram[3] = 2;
        table.hostLayerState(1).decode_local_histogram[1] = 1;
        table.hostLayerState(1).decode_local_histogram[3] = 2;
        table.hostLayerState(0).router_hot_cache_eligible_dispatches = 2;
        table.hostLayerState(0).router_hot_cache_used_dispatches = 1;
        table.hostLayerState(0).router_hot_cache_improved_dispatches = 1;
        table.hostLayerState(0).router_hot_cache_active_dispatches = 4;
        table.hostLayerState(0).router_hot_cache_miss_dispatches = 2;
        table.hostLayerState(0).router_hot_cache_selected_expert_slots = 8;
        table.hostLayerState(0).router_hot_cache_replicated_selected_expert_slots = 3;
        table.hostLayerState(1).router_hot_cache_eligible_dispatches = 3;
        table.hostLayerState(1).router_hot_cache_used_dispatches = 2;
        table.hostLayerState(1).router_hot_cache_improved_dispatches = 1;
        table.hostLayerState(1).router_hot_cache_active_dispatches = 5;
        table.hostLayerState(1).router_hot_cache_miss_dispatches = 2;
        table.hostLayerState(1).router_hot_cache_selected_expert_slots = 10;
        table.hostLayerState(1).router_hot_cache_replicated_selected_expert_slots = 4;

        DecodeExpertHistogramConfig cfg;
        cfg.num_layers = 2;
        cfg.num_experts = 4;
        cfg.top_k = 2;
        cfg.window_size = 8;
        cfg.sockets = {DeviceId(DeviceType::CPU, 0), DeviceId(DeviceType::CPU, 1)};
        cfg.expert_to_socket = {0, 1, 0, 1};
        DecodeExpertHistogram hist(cfg);

        hist.recordTokenBoundary(1);
        ASSERT_TRUE(table.syncDecodeHistogramToHost(hist));

        EXPECT_EQ(hist.activationCount(0, 0), 3u);
        EXPECT_EQ(hist.activationCount(0, 2), 1u);
        EXPECT_EQ(hist.activationCount(1, 1), 2u);
        EXPECT_EQ(hist.activationCount(1, 3), 2u);
        EXPECT_EQ(hist.windowTokenCount(), 1u);

        for (int layer = 0; layer < 2; ++layer)
            for (int expert = 0; expert < 4; ++expert)
            {
                EXPECT_EQ(table.hostLayerState(layer).decode_histogram[expert], 0u);
                EXPECT_EQ(table.hostLayerState(layer).decode_local_histogram[expert], 0u);
            }
        for (int layer = 0; layer < 2; ++layer)
        {
            EXPECT_EQ(table.hostLayerState(layer).router_hot_cache_eligible_dispatches, 0u);
            EXPECT_EQ(table.hostLayerState(layer).router_hot_cache_used_dispatches, 0u);
            EXPECT_EQ(table.hostLayerState(layer).router_hot_cache_improved_dispatches, 0u);
            EXPECT_EQ(table.hostLayerState(layer).router_hot_cache_active_dispatches, 0u);
            EXPECT_EQ(table.hostLayerState(layer).router_hot_cache_miss_dispatches, 0u);
            EXPECT_EQ(table.hostLayerState(layer).router_hot_cache_selected_expert_slots, 0u);
            EXPECT_EQ(table.hostLayerState(layer).router_hot_cache_replicated_selected_expert_slots, 0u);
        }

        ASSERT_TRUE(table.syncDecodeHistogramToHost(hist));
        EXPECT_EQ(hist.activationCount(0, 0), 3u);
        EXPECT_EQ(hist.activationCount(1, 1), 2u);
    }

    TEST(Test__MoERuntimeTable, DeviceRebalancePolicyPublishesHotReplicasFromResidentMasks)
    {
        MoERuntimeTable table(DeviceId::cpu(), 2, 4, 2);

        auto update = updateForEpoch(1, 4);
        update.participant_id = 1;
        update.participant_count = 3;
        update.experts[0].owner_participant = 0;
        update.experts[1].owner_participant = 0;
        update.experts[2].owner_participant = 1;
        update.experts[3].owner_participant = 2;
        update.local_compute_mask = {0, 0, 1, 0};
        update.replica_role = {
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
        };
        update.resident_participant_mask = {
            0b011u, // expert 0 has arrived on participant 1 as a replica slot.
            0b001u,
            0b010u,
            0b110u,
        };
        ASSERT_TRUE(table.prepareInactiveBank(0, update));
        ASSERT_TRUE(table.flipActiveBank(0, 1, nullptr));
        auto layer1_update = update;
        layer1_update.epoch = 1;
        ASSERT_TRUE(table.prepareInactiveBank(1, layer1_update));
        ASSERT_TRUE(table.flipActiveBank(1, 1, nullptr));

        table.hostLayerState(0).decode_histogram[0] = 7;
        table.hostLayerState(0).decode_local_histogram[2] = 3;
        table.hostLayerState(0).router_hot_cache_eligible_dispatches = 2;
        table.hostLayerState(0).router_hot_cache_used_dispatches = 1;
        table.hostLayerState(0).router_hot_cache_improved_dispatches = 1;
        table.hostLayerState(0).router_hot_cache_default_load_spread_total = 6;
        table.hostLayerState(0).router_hot_cache_actual_load_spread_total = 4;
        table.hostLayerState(0).router_hot_cache_load_spread_improvement_total = 2;
        table.hostLayerState(0).router_hot_cache_active_dispatches = 10;
        table.hostLayerState(0).router_hot_cache_miss_dispatches = 6;
        table.hostLayerState(0).router_hot_cache_selected_expert_slots = 20;
        table.hostLayerState(0).router_hot_cache_replicated_selected_expert_slots = 8;
        table.hostLayerState(1).router_hot_cache_eligible_dispatches = 3;
        table.hostLayerState(1).router_hot_cache_used_dispatches = 2;
        table.hostLayerState(1).router_hot_cache_improved_dispatches = 1;
        table.hostLayerState(1).router_hot_cache_default_load_spread_total = 9;
        table.hostLayerState(1).router_hot_cache_actual_load_spread_total = 6;
        table.hostLayerState(1).router_hot_cache_load_spread_improvement_total = 3;
        table.hostLayerState(1).router_hot_cache_active_dispatches = 20;
        table.hostLayerState(1).router_hot_cache_miss_dispatches = 7;
        table.hostLayerState(1).router_hot_cache_selected_expert_slots = 40;
        table.hostLayerState(1).router_hot_cache_replicated_selected_expert_slots = 12;

        const auto config = rebalanceConfig(/*layers=*/2, /*experts=*/4, /*top_k=*/2,
                                            /*participant=*/1, /*participants=*/3,
                                            /*window_tokens=*/2, /*max_hot_replicas=*/1);
        std::vector<uint64_t> gathered(
            static_cast<size_t>(config.participant_count) *
                static_cast<size_t>(config.num_layers) *
                static_cast<size_t>(config.num_experts),
            0);
        addGatheredCount(gathered, config, 0, 0, 0, 30);
        addGatheredCount(gathered, config, 1, 0, 3, 20);
        addGatheredCount(gathered, config, 0, 1, 0, 30);
        addGatheredCount(gathered, config, 2, 1, 3, 20);

        DeviceMoERebalanceStatus status;
        ASSERT_TRUE(applyDeviceMoERebalancePolicyHost(
            table.deviceLayerState(0), gathered.data(), config, &status));

        EXPECT_EQ(status.status_code, static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok));
        EXPECT_EQ(status.windows_applied, 1u);
        EXPECT_EQ(status.changed_layers, 2u);
        EXPECT_EQ(status.selected_replicas, 2u);
        EXPECT_EQ(status.router_hot_cache_eligible_dispatches, 5u);
        EXPECT_EQ(status.router_hot_cache_used_dispatches, 3u);
        EXPECT_EQ(status.router_hot_cache_improved_dispatches, 2u);
        EXPECT_EQ(status.router_hot_cache_default_load_spread_total, 15u);
        EXPECT_EQ(status.router_hot_cache_actual_load_spread_total, 10u);
        EXPECT_EQ(status.router_hot_cache_load_spread_improvement_total, 5u);
        EXPECT_EQ(status.router_hot_cache_active_dispatches, 30u);
        EXPECT_EQ(status.router_hot_cache_miss_dispatches, 13u);
        EXPECT_EQ(status.router_hot_cache_selected_expert_slots, 60u);
        EXPECT_EQ(status.router_hot_cache_replicated_selected_expert_slots, 20u);

        for (int layer = 0; layer < 2; ++layer)
        {
            const auto &state = table.hostLayerState(layer);
            ASSERT_EQ(state.active_epoch, 2u);
            const auto &bank = state.banks[state.active_bank];
            EXPECT_EQ(bank.local_compute_mask[0], 1u) << "hot resident expert should become local";
            EXPECT_TRUE(hasMoEExpertFlag(bank.experts[0].flags, DeviceMoEExpertFlags::Replicated));
            EXPECT_EQ(bank.replica_role[0], static_cast<uint8_t>(DeviceMoEReplicaRole::Replica));
            EXPECT_EQ(bank.local_compute_mask[1], 0u);
            EXPECT_FALSE(hasMoEExpertFlag(bank.experts[1].flags, DeviceMoEExpertFlags::Replicated));
            EXPECT_EQ(bank.local_compute_mask[2], 1u);
            EXPECT_EQ(bank.replica_role[2], static_cast<uint8_t>(DeviceMoEReplicaRole::Primary));
            for (int expert = 0; expert < 4; ++expert)
            {
                EXPECT_EQ(state.decode_histogram[expert], 0u);
                EXPECT_EQ(state.decode_local_histogram[expert], 0u);
            }
            EXPECT_EQ(state.router_hot_cache_eligible_dispatches, 0u);
            EXPECT_EQ(state.router_hot_cache_used_dispatches, 0u);
            EXPECT_EQ(state.router_hot_cache_improved_dispatches, 0u);
            EXPECT_EQ(state.router_hot_cache_default_load_spread_total, 0u);
            EXPECT_EQ(state.router_hot_cache_actual_load_spread_total, 0u);
            EXPECT_EQ(state.router_hot_cache_load_spread_improvement_total, 0u);
            EXPECT_EQ(state.router_hot_cache_active_dispatches, 0u);
            EXPECT_EQ(state.router_hot_cache_miss_dispatches, 0u);
            EXPECT_EQ(state.router_hot_cache_selected_expert_slots, 0u);
            EXPECT_EQ(state.router_hot_cache_replicated_selected_expert_slots, 0u);
        }
    }

    TEST(Test__MoERuntimeTable, DeviceRebalancePolicyHonorsLayerWindow)
    {
        MoERuntimeTable table(DeviceId::cpu(), 3, 4, 2);

        auto update = updateForEpoch(1, 4);
        update.participant_id = 1;
        update.participant_count = 3;
        update.experts[0] = expertOwnerOnlyDesc(0, 0);
        update.experts[1] = expertOwnerOnlyDesc(1, 0);
        update.experts[2].owner_participant = 1;
        update.experts[3] = expertOwnerOnlyDesc(3, 2);
        update.local_compute_mask = {0, 0, 1, 0};
        update.replica_role = {
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
        };
        update.resident_participant_mask = {0b011u, 0b001u, 0b010u, 0b100u};
        for (int layer = 0; layer < 3; ++layer)
        {
            ASSERT_TRUE(table.prepareInactiveBank(layer, update));
            ASSERT_TRUE(table.flipActiveBank(layer, 1, nullptr));
        }

        auto config = rebalanceConfig(/*layers=*/3, /*experts=*/4, /*top_k=*/2,
                                      /*participant=*/1, /*participants=*/3,
                                      /*window_tokens=*/1, /*max_hot_replicas=*/1);
        config.layer_window_start = 1;
        config.layer_window_count = 1;
        std::vector<uint64_t> gathered(
            static_cast<size_t>(config.participant_count) *
                static_cast<size_t>(config.num_layers) *
                static_cast<size_t>(config.num_experts),
            0);
        addGatheredCount(gathered, config, 0, 0, 0, 99);
        addGatheredCount(gathered, config, 0, 1, 0, 99);
        addGatheredCount(gathered, config, 0, 2, 0, 99);

        DeviceMoERebalanceStatus status;
        ASSERT_TRUE(applyDeviceMoERebalancePolicyHost(
            table.deviceLayerState(0), gathered.data(), config, &status));

        EXPECT_EQ(status.status_code, static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok));
        EXPECT_EQ(status.changed_layers, 1u);
        EXPECT_EQ(table.hostLayerState(0).active_epoch, 1u)
            << "rolling rebalance windows must not apply layers before the staged wave";
        EXPECT_EQ(table.hostLayerState(1).active_epoch, 2u);
        EXPECT_EQ(table.hostLayerState(2).active_epoch, 1u)
            << "rolling rebalance windows must not apply layers after the staged wave";

        const auto &layer1_bank =
            table.hostLayerState(1).banks[table.hostLayerState(1).active_bank];
        EXPECT_EQ(layer1_bank.local_compute_mask[0], 1u);
        EXPECT_TRUE(hasMoEExpertFlag(layer1_bank.experts[0].flags,
                                     DeviceMoEExpertFlags::Replicated));
    }

    TEST(Test__MoERuntimeTable, DeviceRebalancePolicyHonorsLayerWaveCount)
    {
        MoERuntimeTable table(DeviceId::cpu(), 4, 4, 2);

        auto update = updateForEpoch(1, 4);
        update.participant_id = 1;
        update.participant_count = 3;
        update.experts[0].owner_participant = 0;
        update.experts[1].owner_participant = 0;
        update.experts[2].owner_participant = 1;
        update.experts[3].owner_participant = 2;
        update.local_compute_mask = {0, 0, 1, 0};
        update.replica_role = {
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
        };
        update.resident_participant_mask = {0b011u, 0b001u, 0b010u, 0b100u};
        for (int layer = 0; layer < 4; ++layer)
        {
            ASSERT_TRUE(table.prepareInactiveBank(layer, update));
            ASSERT_TRUE(table.flipActiveBank(layer, 1, nullptr));
        }

        auto config = rebalanceConfig(/*layers=*/4, /*experts=*/4, /*top_k=*/2,
                                      /*participant=*/1, /*participants=*/3,
                                      /*window_tokens=*/1, /*max_hot_replicas=*/1);
        config.layer_window_start = 1;
        config.layer_window_count = 3;
        config.layer_wave_count = 2;
        std::vector<uint64_t> gathered(
            static_cast<size_t>(config.participant_count) *
                static_cast<size_t>(config.num_layers) *
                static_cast<size_t>(config.num_experts),
            0);
        addGatheredCount(gathered, config, 0, 1, 0, 99);
        addGatheredCount(gathered, config, 0, 2, 0, 99);
        addGatheredCount(gathered, config, 0, 3, 0, 99);

        DeviceMoERebalanceStatus status;
        ASSERT_TRUE(applyDeviceMoERebalancePolicyHost(
            table.deviceLayerState(0), gathered.data(), config, &status));

        EXPECT_EQ(status.status_code, static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok));
        EXPECT_EQ(status.changed_layers, 2u)
            << "Layer wave count must bound one graph-captured maintenance replay.";
        EXPECT_EQ(table.hostLayerState(0).active_epoch, 1u);
        EXPECT_EQ(table.hostLayerState(1).active_epoch, 2u);
        EXPECT_EQ(table.hostLayerState(2).active_epoch, 2u);
        EXPECT_EQ(table.hostLayerState(3).active_epoch, 1u)
            << "The remaining layer should wait for the next rolling maintenance replay.";
    }

    TEST(Test__MoERuntimeTable, DeviceRebalancePolicyWaitsForWindowWithoutMutating)
    {
        MoERuntimeTable table(DeviceId::cpu(), 1, 4, 2);
        auto update = updateForEpoch(1, 4);
        update.participant_id = 1;
        update.participant_count = 3;
        update.experts[0].owner_participant = 0;
        update.experts[1].owner_participant = 0;
        update.experts[2].owner_participant = 1;
        update.experts[3].owner_participant = 2;
        update.local_compute_mask = {0, 0, 1, 0};
        update.replica_role = {
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
        };
        update.resident_participant_mask = {0b011u, 0b001u, 0b010u, 0b100u};
        ASSERT_TRUE(table.prepareInactiveBank(0, update));
        ASSERT_TRUE(table.flipActiveBank(0, 1, nullptr));
        table.hostLayerState(0).decode_histogram[0] = 5;

        const auto config = rebalanceConfig(/*layers=*/1, /*experts=*/4, /*top_k=*/2,
                                            /*participant=*/1, /*participants=*/3,
                                            /*window_tokens=*/64, /*max_hot_replicas=*/1);
        std::vector<uint64_t> gathered(
            static_cast<size_t>(config.participant_count) *
                static_cast<size_t>(config.num_layers) *
                static_cast<size_t>(config.num_experts),
            0);
        addGatheredCount(gathered, config, 0, 0, 0, 4);

        DeviceMoERebalanceStatus status;
        ASSERT_TRUE(applyDeviceMoERebalancePolicyHost(
            table.deviceLayerState(0), gathered.data(), config, &status));

        EXPECT_EQ(status.status_code,
                  static_cast<uint32_t>(DeviceMoERebalanceStatusCode::WindowNotReady));
        EXPECT_EQ(status.skipped_not_ready, 1u);
        EXPECT_EQ(status.skipped_busy_wave, 0u)
            << "host policy WindowNotReady means histogram cadence, not an active wave.";
        EXPECT_EQ(status.window_ready_slots, 4u);
        EXPECT_EQ(status.window_required_slots, 128u);
        EXPECT_EQ(status.windows_applied, 0u);
        EXPECT_EQ(table.hostLayerState(0).active_epoch, 1u);
        EXPECT_EQ(table.hostLayerState(0).decode_histogram[0], 5u)
            << "window wait must not reset runtime counters";
    }

    TEST(Test__MoERuntimeTable, DeviceRebalancePolicyPlansMissingExpertPayloadArrivals)
    {
        MoERuntimeTable table(DeviceId::cpu(), 1, 4, 2);
        auto update = updateForEpoch(1, 4);
        update.participant_id = 1;
        update.participant_count = 3;
        update.experts[0] = expertOwnerOnlyDesc(0, 0);
        update.experts[1] = expertOwnerOnlyDesc(1, 0);
        update.experts[2].owner_participant = 1;
        update.experts[3] = expertOwnerOnlyDesc(3, 2);
        update.local_compute_mask = {0, 0, 1, 0};
        update.replica_role = {
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
        };
        update.resident_participant_mask = {0b001u, 0b001u, 0b010u, 0b100u};
        ASSERT_TRUE(table.prepareInactiveBank(0, update));
        ASSERT_TRUE(table.flipActiveBank(0, 1, nullptr));
        EXPECT_FALSE(table.hostLayerState(0)
                         .banks[table.hostLayerState(0).active_bank]
                         .experts[0]
                         .gate.valid())
            << "Non-local experts should not need local payload descriptors to be arrival candidates.";

        auto config = rebalanceConfig(/*layers=*/1, /*experts=*/4, /*top_k=*/2,
                                      /*participant=*/1, /*participants=*/3,
                                      /*window_tokens=*/1, /*max_hot_replicas=*/1);
        config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::PlanMissingArrivals);
        config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::CollectLoadStats);
        std::vector<uint64_t> gathered(
            static_cast<size_t>(config.participant_count) *
                static_cast<size_t>(config.num_layers) *
                static_cast<size_t>(config.num_experts),
            0);
        addGatheredCount(gathered, config, 0, 0, 0, 6);

        DeviceMoERebalancePlanEntry plan[2];
        uint32_t plan_count = 99;
        DeviceMoERebalanceStatus status;
        ASSERT_TRUE(applyDeviceMoERebalancePolicyHost(
            table.deviceLayerState(0),
            gathered.data(),
            config,
            &status,
            plan,
            &plan_count,
            2));

        EXPECT_EQ(status.status_code, static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok));
        EXPECT_EQ(status.planned_arrivals, 1u);
        EXPECT_EQ(status.plan_overflow, 0u);
        EXPECT_EQ(status.selected_replicas, 0u)
            << "missing arrivals are planned, not exposed as active local replicas";
        EXPECT_EQ(status.pre_policy_load_total, 6u);
        EXPECT_EQ(status.pre_policy_load_min, 0u);
        EXPECT_EQ(status.pre_policy_load_max, 6u);
        EXPECT_EQ(status.post_policy_load_total, 6u);
        EXPECT_EQ(status.post_policy_load_min, 0u);
        EXPECT_EQ(status.post_policy_load_max, 3u)
            << "diagnostic post-policy load must account for planned arrivals without publishing them early";
        EXPECT_EQ(status.pre_policy_participant_load[0], 6u);
        EXPECT_EQ(status.pre_policy_participant_load[1], 0u);
        EXPECT_EQ(status.pre_policy_participant_load[2], 0u);
        EXPECT_EQ(status.post_policy_participant_load[0], 3u);
        EXPECT_EQ(status.post_policy_participant_load[1], 3u);
        EXPECT_EQ(status.post_policy_participant_load[2], 0u);
        EXPECT_GT(status.pre_policy_imbalance_numerator,
                  status.post_policy_imbalance_numerator);
        EXPECT_EQ(status.candidate_arrivals_considered, 1u);
        EXPECT_EQ(status.candidate_arrivals_below_floor, 0u);
        EXPECT_EQ(status.candidate_arrivals_pruned_by_count_bound, 0u);
        EXPECT_EQ(status.candidate_load_spread_improvement_total, 3u);
        EXPECT_EQ(status.candidate_load_spread_improvement_max, 3u);
        EXPECT_EQ(status.accepted_load_spread_improvement_total, 3u);
        EXPECT_EQ(status.accepted_load_spread_improvement_max, 3u);
        ASSERT_EQ(plan_count, 1u);
        EXPECT_EQ(plan[0].op, static_cast<uint32_t>(DeviceMoERebalancePlanOp::ExpertPayloadArrival));
        EXPECT_EQ(plan[0].layer, 0u);
        EXPECT_EQ(plan[0].expert, 0u);
        EXPECT_EQ(plan[0].source_participant, 0u);
        EXPECT_EQ(plan[0].destination_participant, 1u);
        EXPECT_EQ(plan[0].source_resident_mask, 0b001u);
        EXPECT_EQ(plan[0].destination_slot, 0u)
            << "controller assigns graph-consumable transfer slots deterministically";
        EXPECT_EQ(plan[0].payload_slot, 0u)
            << "controller assigns dense outgoing payload slots separately from destination slots";

        const auto &bank = table.hostLayerState(0).banks[table.hostLayerState(0).active_bank];
        EXPECT_EQ(bank.local_compute_mask[0], 0u)
            << "the plan must not expose an expert before graph-side transfer/apply copies it";
        EXPECT_FALSE(hasMoEExpertFlag(bank.experts[0].flags, DeviceMoEExpertFlags::Replicated));
    }

    TEST(Test__MoERuntimeTable, DeviceRebalancePolicyRanksMissingArrivalsBySpreadImprovement)
    {
        MoERuntimeTable table(DeviceId::cpu(), 1, 4, 2);
        auto update = updateForEpoch(1, 4);
        update.participant_id = 1;
        update.participant_count = 3;
        update.experts[0] = expertOwnerOnlyDesc(0, 0);
        update.experts[1] = expertOwnerOnlyDesc(1, 0);
        update.experts[2].owner_participant = 1;
        update.experts[3] = expertOwnerOnlyDesc(3, 2);
        update.local_compute_mask = {0, 0, 1, 0};
        update.replica_role = {
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
        };
        update.resident_participant_mask = {0b001u, 0b001u, 0b010u, 0b100u};
        ASSERT_TRUE(table.prepareInactiveBank(0, update));
        ASSERT_TRUE(table.flipActiveBank(0, 1, nullptr));

        auto config = rebalanceConfig(/*layers=*/1, /*experts=*/4, /*top_k=*/2,
                                      /*participant=*/1, /*participants=*/3,
                                      /*window_tokens=*/1, /*max_hot_replicas=*/1);
        config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::PlanMissingArrivals);
        config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::CollectLoadStats);
        std::vector<uint64_t> gathered(
            static_cast<size_t>(config.participant_count) *
                static_cast<size_t>(config.num_layers) *
                static_cast<size_t>(config.num_experts),
            0);
        addGatheredCount(gathered, config, 0, 0, 0, 50);
        addGatheredCount(gathered, config, 0, 0, 1, 50);
        addGatheredCount(gathered, config, 2, 0, 3, 80);

        DeviceMoERebalancePlanEntry plan[2];
        uint32_t plan_count = 99;
        DeviceMoERebalanceStatus status;
        ASSERT_TRUE(applyDeviceMoERebalancePolicyHost(
            table.deviceLayerState(0),
            gathered.data(),
            config,
            &status,
            plan,
            &plan_count,
            2));

        EXPECT_EQ(status.status_code, static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok));
        EXPECT_EQ(status.planned_arrivals, 1u);
        EXPECT_EQ(status.candidate_arrivals_considered, 3u);
        EXPECT_EQ(status.candidate_arrivals_below_floor, 0u);
        EXPECT_EQ(status.candidate_load_spread_improvement_total, 140u)
            << "Dynamic scores payload movement by the equalizing shift the "
               "least-loaded router can use, not by uniform resident splitting.";
        EXPECT_EQ(status.candidate_load_spread_improvement_max, 50u);
        EXPECT_EQ(status.accepted_load_spread_improvement_total, 50u);
        EXPECT_EQ(status.accepted_load_spread_improvement_max, 50u);
        ASSERT_EQ(plan_count, 1u);
        EXPECT_EQ(plan[0].expert, 0u)
            << "expert 3 is hotter, but moving expert 0 to participant 1 reduces projected "
               "load spread more and should win the transfer slot.";
        EXPECT_EQ(plan[0].source_participant, 0u);
        EXPECT_EQ(plan[0].destination_participant, 1u);
        EXPECT_EQ(status.post_policy_participant_load[0], 75u);
        EXPECT_EQ(status.post_policy_participant_load[1], 25u);
        EXPECT_EQ(status.post_policy_participant_load[2], 80u);
    }

    TEST(Test__MoERuntimeTable, DeviceRebalancePolicySkipsMissingArrivalThatWorsensImbalance)
    {
        MoERuntimeTable table(DeviceId::cpu(), 1, 4, 2);
        auto update = updateForEpoch(1, 4);
        update.participant_id = 1;
        update.participant_count = 3;
        update.experts[0] = expertOwnerOnlyDesc(0, 0);
        update.experts[1] = expertOwnerOnlyDesc(1, 0);
        update.experts[2].owner_participant = 1;
        update.experts[3] = expertOwnerOnlyDesc(3, 2);
        update.local_compute_mask = {0, 0, 1, 0};
        update.replica_role = {
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
        };
        update.resident_participant_mask = {0b001u, 0b001u, 0b010u, 0b100u};
        ASSERT_TRUE(table.prepareInactiveBank(0, update));
        ASSERT_TRUE(table.flipActiveBank(0, 1, nullptr));

        auto config = rebalanceConfig(/*layers=*/1, /*experts=*/4, /*top_k=*/2,
                                      /*participant=*/1, /*participants=*/3,
                                      /*window_tokens=*/1, /*max_hot_replicas=*/1);
        config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::PlanMissingArrivals);
        config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::CollectLoadStats);
        std::vector<uint64_t> gathered(
            static_cast<size_t>(config.participant_count) *
                static_cast<size_t>(config.num_layers) *
                static_cast<size_t>(config.num_experts),
            0);
        addGatheredCount(gathered, config, 0, 0, 0, 6);
        addGatheredCount(gathered, config, 1, 0, 2, 10);

        DeviceMoERebalancePlanEntry plan[2];
        uint32_t plan_count = 99;
        DeviceMoERebalanceStatus status;
        ASSERT_TRUE(applyDeviceMoERebalancePolicyHost(
            table.deviceLayerState(0),
            gathered.data(),
            config,
            &status,
            plan,
            &plan_count,
            2));

        EXPECT_EQ(status.status_code, static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok));
        EXPECT_EQ(status.planned_arrivals, 0u);
        EXPECT_EQ(status.selected_replicas, 0u);
        EXPECT_EQ(status.skipped_no_improvement, 1u)
            << "The controller should reject arrivals that increase projected load spread.";
        EXPECT_EQ(status.pre_policy_load_total, 16u);
        EXPECT_EQ(status.pre_policy_load_min, 0u);
        EXPECT_EQ(status.pre_policy_load_max, 10u);
        EXPECT_EQ(status.post_policy_load_total, 16u);
        EXPECT_EQ(status.post_policy_load_min, 0u);
        EXPECT_EQ(status.post_policy_load_max, 10u);
        EXPECT_EQ(status.pre_policy_imbalance_numerator,
                  status.post_policy_imbalance_numerator);
        EXPECT_EQ(status.candidate_arrivals_considered, 1u);
        EXPECT_EQ(status.candidate_arrivals_below_floor, 1u);
        EXPECT_EQ(status.candidate_arrivals_pruned_by_count_bound, 0u);
        EXPECT_EQ(status.candidate_load_spread_improvement_total, 0u);
        EXPECT_EQ(status.candidate_load_spread_improvement_max, 0u);
        EXPECT_EQ(status.accepted_load_spread_improvement_total, 0u);
        EXPECT_EQ(status.accepted_load_spread_improvement_max, 0u);
        EXPECT_EQ(plan_count, 0u);

        const auto &bank = table.hostLayerState(0).banks[table.hostLayerState(0).active_bank];
        EXPECT_EQ(bank.local_compute_mask[0], 0u);
        EXPECT_FALSE(hasMoEExpertFlag(bank.experts[0].flags, DeviceMoEExpertFlags::Replicated));
    }

    TEST(Test__MoERuntimeTable, DeviceRebalancePolicySkipsTinyLoadSpreadImprovement)
    {
        MoERuntimeTable table(DeviceId::cpu(), 1, 4, 2);
        auto update = updateForEpoch(1, 4);
        update.participant_id = 1;
        update.participant_count = 3;
        update.experts[0] = expertOwnerOnlyDesc(0, 0);
        update.experts[1] = expertOwnerOnlyDesc(1, 0);
        update.experts[2].owner_participant = 1;
        update.experts[3] = expertOwnerOnlyDesc(3, 2);
        update.local_compute_mask = {0, 0, 1, 0};
        update.replica_role = {
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
        };
        update.resident_participant_mask = {0b001u, 0b001u, 0b010u, 0b100u};
        ASSERT_TRUE(table.prepareInactiveBank(0, update));
        ASSERT_TRUE(table.flipActiveBank(0, 1, nullptr));

        auto config = rebalanceConfig(/*layers=*/1, /*experts=*/4, /*top_k=*/2,
                                      /*participant=*/1, /*participants=*/3,
                                      /*window_tokens=*/1, /*max_hot_replicas=*/1);
        config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::PlanMissingArrivals);
        config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::CollectLoadStats);
        config.min_load_spread_improvement = 4;
        std::vector<uint64_t> gathered(
            static_cast<size_t>(config.participant_count) *
                static_cast<size_t>(config.num_layers) *
                static_cast<size_t>(config.num_experts),
            0);
        addGatheredCount(gathered, config, 0, 0, 0, 6);

        DeviceMoERebalancePlanEntry plan[2];
        uint32_t plan_count = 99;
        DeviceMoERebalanceStatus status;
        ASSERT_TRUE(applyDeviceMoERebalancePolicyHost(
            table.deviceLayerState(0),
            gathered.data(),
            config,
            &status,
            plan,
            &plan_count,
            2));

        EXPECT_EQ(status.status_code, static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok));
        EXPECT_EQ(status.planned_arrivals, 0u);
        EXPECT_EQ(status.selected_replicas, 0u);
        EXPECT_EQ(status.skipped_no_improvement, 1u);
        EXPECT_EQ(status.pre_policy_imbalance_numerator, 6u);
        EXPECT_EQ(status.post_policy_imbalance_numerator, 6u);
        EXPECT_EQ(status.pre_policy_participant_load[0], 6u);
        EXPECT_EQ(status.post_policy_participant_load[0], 6u);
        EXPECT_EQ(status.candidate_arrivals_considered, 1u);
        EXPECT_EQ(status.candidate_arrivals_below_floor, 1u);
        EXPECT_EQ(status.candidate_arrivals_pruned_by_count_bound, 0u);
        EXPECT_EQ(status.candidate_load_spread_improvement_total, 3u);
        EXPECT_EQ(status.candidate_load_spread_improvement_max, 3u);
        EXPECT_EQ(status.accepted_load_spread_improvement_total, 0u);
        EXPECT_EQ(status.accepted_load_spread_improvement_max, 0u);
        EXPECT_EQ(plan_count, 0u);
    }

    TEST(Test__MoERuntimeTable, DeviceRebalancePolicyPrunesMissingArrivalBelowCountBound)
    {
        MoERuntimeTable table(DeviceId::cpu(), 1, 4, 2);
        auto update = updateForEpoch(1, 4);
        update.participant_id = 1;
        update.participant_count = 3;
        update.experts[0] = expertOwnerOnlyDesc(0, 0);
        update.experts[1] = expertOwnerOnlyDesc(1, 0);
        update.experts[2].owner_participant = 1;
        update.experts[3] = expertOwnerOnlyDesc(3, 2);
        update.local_compute_mask = {0, 0, 1, 0};
        update.replica_role = {
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
        };
        update.resident_participant_mask = {0b001u, 0b001u, 0b010u, 0b100u};
        ASSERT_TRUE(table.prepareInactiveBank(0, update));
        ASSERT_TRUE(table.flipActiveBank(0, 1, nullptr));

        auto config = rebalanceConfig(/*layers=*/1, /*experts=*/4, /*top_k=*/2,
                                      /*participant=*/1, /*participants=*/3,
                                      /*window_tokens=*/1, /*max_hot_replicas=*/1);
        config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::PlanMissingArrivals);
        config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::CollectLoadStats);
        config.min_load_spread_improvement = 8;
        std::vector<uint64_t> gathered(
            static_cast<size_t>(config.participant_count) *
                static_cast<size_t>(config.num_layers) *
                static_cast<size_t>(config.num_experts),
            0);
        addGatheredCount(gathered, config, 0, 0, 0, 6);

        DeviceMoERebalancePlanEntry plan[2];
        uint32_t plan_count = 99;
        DeviceMoERebalanceStatus status;
        ASSERT_TRUE(applyDeviceMoERebalancePolicyHost(
            table.deviceLayerState(0),
            gathered.data(),
            config,
            &status,
            plan,
            &plan_count,
            2));

        EXPECT_EQ(status.status_code, static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok));
        EXPECT_EQ(status.planned_arrivals, 0u);
        EXPECT_EQ(status.selected_replicas, 0u);
        EXPECT_EQ(status.skipped_no_improvement, 1u);
        EXPECT_EQ(status.pre_policy_load_total, 6u);
        EXPECT_EQ(status.pre_policy_imbalance_numerator,
                  status.post_policy_imbalance_numerator);
        EXPECT_EQ(status.candidate_arrivals_considered, 0u)
            << "count-bound prunes should skip the expensive projected-spread evaluation";
        EXPECT_EQ(status.candidate_arrivals_below_floor, 0u);
        EXPECT_EQ(status.candidate_arrivals_pruned_by_count_bound, 1u);
        EXPECT_EQ(status.candidate_load_spread_improvement_total, 0u);
        EXPECT_EQ(status.candidate_load_spread_improvement_max, 0u);
        EXPECT_EQ(status.accepted_load_spread_improvement_total, 0u);
        EXPECT_EQ(status.accepted_load_spread_improvement_max, 0u);
        EXPECT_EQ(plan_count, 0u);
    }

    TEST(Test__MoERuntimeTable, DeviceRebalancePolicyResidentOnlySkipsMissingReplicaCandidates)
    {
        MoERuntimeTable table(DeviceId::cpu(), 1, 4, 2);
        auto update = updateForEpoch(1, 4);
        update.participant_id = 1;
        update.participant_count = 3;
        update.experts[0].owner_participant = 0;
        update.experts[1].owner_participant = 0;
        update.experts[2].owner_participant = 1;
        update.experts[3].owner_participant = 2;
        update.local_compute_mask = {0, 0, 1, 0};
        update.replica_role = {
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
        };
        update.resident_participant_mask = {0b001u, 0b001u, 0b010u, 0b100u};
        ASSERT_TRUE(table.prepareInactiveBank(0, update));
        ASSERT_TRUE(table.flipActiveBank(0, 1, nullptr));

        const auto config = rebalanceConfig(/*layers=*/1, /*experts=*/4, /*top_k=*/2,
                                            /*participant=*/1, /*participants=*/3,
                                            /*window_tokens=*/1, /*max_hot_replicas=*/2);
        std::vector<uint64_t> gathered(
            static_cast<size_t>(config.participant_count) *
                static_cast<size_t>(config.num_layers) *
                static_cast<size_t>(config.num_experts),
            0);
        addGatheredCount(gathered, config, 0, 0, 0, 8);

        DeviceMoERebalancePlanEntry plan[2];
        uint32_t plan_count = 99;
        DeviceMoERebalanceStatus status;
        ASSERT_TRUE(applyDeviceMoERebalancePolicyHost(
            table.deviceLayerState(0),
            gathered.data(),
            config,
            &status,
            plan,
            &plan_count,
            2));

        EXPECT_EQ(status.status_code, static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok));
        EXPECT_EQ(status.windows_applied, 1u);
        EXPECT_EQ(status.selected_replicas, 0u);
        EXPECT_EQ(status.planned_arrivals, 0u);
        EXPECT_EQ(status.skipped_no_resident, 0u)
            << "Resident-only mode should not rank non-local missing experts as failed arrivals.";
        EXPECT_EQ(plan_count, 0u);

        const auto &bank = table.hostLayerState(0).banks[table.hostLayerState(0).active_bank];
        EXPECT_EQ(bank.local_compute_mask[0], 0u);
        EXPECT_FALSE(hasMoEExpertFlag(bank.experts[0].flags, DeviceMoEExpertFlags::Replicated));
        EXPECT_EQ(bank.local_compute_mask[2], 1u);
        EXPECT_EQ(bank.replica_role[2], static_cast<uint8_t>(DeviceMoEReplicaRole::Primary));
    }

    TEST(Test__MoERuntimeTable, DeviceRebalanceDirectoryPacksOnlyLocalResidentDescriptors)
    {
        MoERuntimeTable table(DeviceId::cpu(), 1, 4, 2);
        auto update = updateForEpoch(1, 4);
        update.participant_id = 1;
        update.participant_count = 3;
        update.experts[0].owner_participant = 0;
        update.experts[1].owner_participant = 0;
        update.experts[2].owner_participant = 1;
        update.experts[3].owner_participant = 2;
        update.local_compute_mask = {0, 0, 1, 0};
        update.replica_role = {
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
        };
        update.resident_participant_mask = {0b001u, 0b001u, 0b010u, 0b100u};
        ASSERT_TRUE(table.prepareInactiveBank(0, update));
        ASSERT_TRUE(table.flipActiveBank(0, 1, nullptr));

        const auto config = rebalanceConfig(/*layers=*/1, /*experts=*/4, /*top_k=*/2,
                                            /*participant=*/1, /*participants=*/3,
                                            /*window_tokens=*/1, /*max_hot_replicas=*/1);
        std::vector<DeviceMoEExpertDirectoryEntry> directory(
            static_cast<size_t>(config.num_layers) * static_cast<size_t>(config.num_experts));

        ASSERT_TRUE(packDeviceMoELocalDirectoryHost(
            table.deviceLayerState(0),
            directory.data(),
            config));

        EXPECT_FALSE(deviceMoEDirectoryEntryReady(directory[0], 1, 0, 0))
            << "non-local owner metadata is not a usable source descriptor";
        EXPECT_FALSE(deviceMoEDirectoryEntryReady(directory[1], 1, 0, 1));
        ASSERT_TRUE(deviceMoEDirectoryEntryReady(directory[2], 1, 0, 2));
        EXPECT_EQ(directory[2].descriptor.logical_expert_id, 2);
        EXPECT_EQ(directory[2].slot_index, 2u);
        EXPECT_TRUE((directory[2].flags &
                     static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::LocalCompute)) != 0u);
        EXPECT_FALSE(deviceMoEDirectoryEntryReady(directory[3], 1, 0, 3));
    }

    TEST(Test__MoERuntimeTable, DeviceRebalanceArrivalApplyFailsClosedUntilPeerCopyCompletes)
    {
        MoERuntimeTable table(DeviceId::cpu(), 1, 4, 2);
        auto update = updateForEpoch(1, 4);
        update.participant_id = 1;
        update.participant_count = 3;
        update.experts[0].owner_participant = 0;
        update.experts[1].owner_participant = 0;
        update.experts[2].owner_participant = 1;
        update.experts[3].owner_participant = 2;
        update.local_compute_mask = {0, 0, 1, 0};
        update.replica_role = {
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
        };
        update.resident_participant_mask = {0b001u, 0b001u, 0b010u, 0b100u};
        ASSERT_TRUE(table.prepareInactiveBank(0, update));
        ASSERT_TRUE(table.flipActiveBank(0, 1, nullptr));

        const auto config = rebalanceConfig(/*layers=*/1, /*experts=*/4, /*top_k=*/2,
                                            /*participant=*/1, /*participants=*/3,
                                            /*window_tokens=*/1, /*max_hot_replicas=*/1);
        DeviceMoERebalancePlanEntry plan;
        plan.op = static_cast<uint32_t>(DeviceMoERebalancePlanOp::ExpertPayloadArrival);
        plan.layer = 0;
        plan.expert = 0;
        plan.source_participant = 0;
        plan.destination_participant = 1;
        plan.source_resident_mask = 0b001u;
        plan.destination_slot = 0;
        plan.payload_slot = 0;

        std::vector<DeviceMoEExpertDirectoryEntry> gathered_directory(
            static_cast<size_t>(config.participant_count) *
            static_cast<size_t>(config.num_layers) *
            static_cast<size_t>(config.num_experts));
        std::vector<DeviceMoEExpertDirectoryEntry> transfer_slots(1);

        DeviceMoERebalanceApplyStatus status;
        ASSERT_TRUE(applyDeviceMoERebalanceArrivalsHost(
            table.deviceLayerState(0),
            &plan,
            1,
            gathered_directory.data(),
            transfer_slots.data(),
            static_cast<uint32_t>(transfer_slots.size()),
            config,
            &status));
        EXPECT_EQ(status.applied_arrivals, 0u);
        EXPECT_EQ(status.missing_source_descriptors, 1u);
        EXPECT_EQ(table.hostLayerState(0).banks[table.hostLayerState(0).active_bank].local_compute_mask[0], 0u);

        auto &source_entry = gathered_directory[deviceMoEDirectoryIndex(config, 0, 0, 0)];
        source_entry.descriptor = expertDesc(0, 0, 0);
        source_entry.layer = 0;
        source_entry.expert = 0;
        source_entry.participant = 0;
        source_entry.resident_mask = 0b001u;
        source_entry.flags =
            static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::Valid) |
            static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::Resident) |
            static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::LocalCompute);
        ASSERT_TRUE(deviceMoEDirectoryCopyReady(source_entry));

        ASSERT_TRUE(applyDeviceMoERebalanceArrivalsHost(
            table.deviceLayerState(0),
            &plan,
            1,
            gathered_directory.data(),
            transfer_slots.data(),
            static_cast<uint32_t>(transfer_slots.size()),
            config,
            &status));
        EXPECT_EQ(status.applied_arrivals, 0u);
        EXPECT_EQ(status.missing_destination_slots, 1u)
            << "a source descriptor alone must not expose peer memory as local compute";

        auto &slot = transfer_slots[0];
        slot.descriptor = expertDesc(0, 0, 17);
        slot.descriptor.logical_expert_id = -1;
        slot.layer = kDeviceMoEInvalidSlot;
        slot.expert = kDeviceMoEInvalidSlot;
        slot.participant = 1;
        slot.resident_mask = 0b010u;
        slot.slot_index = 17;
        slot.flags =
            static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::Valid) |
            static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::Resident) |
            static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::TransferSlot);
        ASSERT_TRUE(deviceMoEDirectoryCopyReady(slot));

        ASSERT_TRUE(applyDeviceMoERebalanceArrivalsHost(
            table.deviceLayerState(0),
            &plan,
            1,
            gathered_directory.data(),
            transfer_slots.data(),
            static_cast<uint32_t>(transfer_slots.size()),
            config,
            &status));
        EXPECT_EQ(status.applied_arrivals, 0u);
        EXPECT_EQ(status.copy_incomplete, 1u)
            << "a staged destination slot must not publish until peer copy completion is visible";
        EXPECT_EQ(table.hostLayerState(0).banks[table.hostLayerState(0).active_bank].local_compute_mask[0], 0u);

        slot.flags |= static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::CopyComplete);

        ASSERT_TRUE(applyDeviceMoERebalanceArrivalsHost(
            table.deviceLayerState(0),
            &plan,
            1,
            gathered_directory.data(),
            transfer_slots.data(),
            static_cast<uint32_t>(transfer_slots.size()),
            config,
            &status));
        EXPECT_EQ(status.applied_arrivals, 0u);
        EXPECT_EQ(status.copy_incomplete, 1u)
            << "copy completion must also prove the generic slot was bound to the planned expert";

        slot.layer = plan.layer;
        slot.expert = plan.expert;
        slot.descriptor.logical_expert_id = static_cast<int32_t>(plan.expert);
        auto &runtime_before_apply = table.hostLayerState(0);
        runtime_before_apply.router_hot_cache_eligible_dispatches = 11;
        runtime_before_apply.router_hot_cache_used_dispatches = 7;
        runtime_before_apply.router_hot_cache_improved_dispatches = 5;
        runtime_before_apply.router_hot_cache_default_load_spread_total = 19;
        runtime_before_apply.router_hot_cache_actual_load_spread_total = 13;
        runtime_before_apply.router_hot_cache_load_spread_improvement_total = 6;
        runtime_before_apply.router_hot_cache_active_dispatches = 23;
        runtime_before_apply.router_hot_cache_miss_dispatches = 17;
        runtime_before_apply.router_hot_cache_selected_expert_slots = 46;
        runtime_before_apply.router_hot_cache_replicated_selected_expert_slots = 12;

        ASSERT_TRUE(applyDeviceMoERebalanceArrivalsHost(
            table.deviceLayerState(0),
            &plan,
            1,
            gathered_directory.data(),
            transfer_slots.data(),
            static_cast<uint32_t>(transfer_slots.size()),
            config,
            &status));
        EXPECT_EQ(status.applied_arrivals, 1u);
        EXPECT_EQ(status.changed_layers, 1u);

        const auto &bank = table.hostLayerState(0).banks[table.hostLayerState(0).active_bank];
        EXPECT_EQ(bank.local_compute_mask[0], 1u);
        EXPECT_EQ(bank.replica_role[0], static_cast<uint8_t>(DeviceMoEReplicaRole::Replica));
        EXPECT_EQ(bank.resident_participant_mask[0], 0b011u);
        EXPECT_EQ(bank.transient_placement_observed, 1u)
            << "The transfer destination must mark the layer as containing "
               "request-lifetime placement.";
        EXPECT_EQ(bank.experts[0].local_slot, 17);
        EXPECT_TRUE(hasMoEExpertFlag(bank.experts[0].flags, DeviceMoEExpertFlags::Replicated));
        EXPECT_TRUE(hasMoEExpertFlag(bank.experts[0].flags, DeviceMoEExpertFlags::LocalCompute));
        EXPECT_EQ(table.hostLayerState(0).router_hot_cache_eligible_dispatches, 11u)
            << "arrival apply starts a new histogram window but must leave router-benefit counters for controller export";
        EXPECT_EQ(table.hostLayerState(0).router_hot_cache_used_dispatches, 7u);
        EXPECT_EQ(table.hostLayerState(0).router_hot_cache_improved_dispatches, 5u);
        EXPECT_EQ(table.hostLayerState(0).router_hot_cache_default_load_spread_total, 19u);
        EXPECT_EQ(table.hostLayerState(0).router_hot_cache_actual_load_spread_total, 13u);
        EXPECT_EQ(table.hostLayerState(0).router_hot_cache_load_spread_improvement_total, 6u);
        EXPECT_EQ(table.hostLayerState(0).router_hot_cache_active_dispatches, 23u);
        EXPECT_EQ(table.hostLayerState(0).router_hot_cache_miss_dispatches, 17u);
        EXPECT_EQ(table.hostLayerState(0).router_hot_cache_selected_expert_slots, 46u);
        EXPECT_EQ(table.hostLayerState(0).router_hot_cache_replicated_selected_expert_slots, 12u);
    }

    TEST(Test__MoERuntimeTable, DeviceRebalanceArrivalApplyPropagatesReplicaMetadataToNonDestination)
    {
        MoERuntimeTable table(DeviceId::cpu(), 1, 4, 2);
        auto update = updateForEpoch(1, 4);
        update.participant_id = 0;
        update.participant_count = 3;
        update.experts[0].owner_participant = 0;
        update.experts[1].owner_participant = 0;
        update.experts[2].owner_participant = 1;
        update.experts[3].owner_participant = 2;
        update.local_compute_mask = {1, 1, 0, 0};
        update.replica_role = {
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
        };
        update.resident_participant_mask = {0b001u, 0b001u, 0b010u, 0b100u};
        ASSERT_TRUE(table.prepareInactiveBank(0, update));
        ASSERT_TRUE(table.flipActiveBank(0, 1, nullptr));

        const auto config = rebalanceConfig(/*layers=*/1, /*experts=*/4, /*top_k=*/2,
                                            /*participant=*/0, /*participants=*/3,
                                            /*window_tokens=*/1, /*max_hot_replicas=*/1);
        DeviceMoERebalancePlanEntry plan;
        plan.op = static_cast<uint32_t>(DeviceMoERebalancePlanOp::ExpertPayloadArrival);
        plan.layer = 0;
        plan.expert = 0;
        plan.source_participant = 0;
        plan.destination_participant = 1;
        plan.source_resident_mask = 0b001u;
        plan.destination_slot = 0;
        plan.payload_slot = 0;

        DeviceMoERebalanceApplyStatus status;
        ASSERT_TRUE(applyDeviceMoERebalanceArrivalsHost(
            table.deviceLayerState(0),
            &plan,
            1,
            nullptr,
            nullptr,
            0,
            config,
            &status));

        EXPECT_EQ(status.applied_arrivals, 0u)
            << "non-destination participants should not count remote arrivals as local slot installs";
        EXPECT_EQ(status.changed_layers, 1u);
        EXPECT_EQ(status.skipped_wrong_destination, 0u)
            << "non-destination participants still need replica metadata for deterministic routing";

        const auto &bank = table.hostLayerState(0).banks[table.hostLayerState(0).active_bank];
        EXPECT_EQ(bank.local_compute_mask[0], 1u)
            << "the owner remains locally computable but can now yield routes to the replica";
        EXPECT_EQ(bank.replica_role[0], static_cast<uint8_t>(DeviceMoEReplicaRole::Primary));
        EXPECT_EQ(bank.resident_participant_mask[0], 0b011u);
        EXPECT_EQ(bank.transient_placement_observed, 1u)
            << "The same gathered transfer plan must publish the transient "
               "placement marker on non-destination participants.";
        EXPECT_TRUE(hasMoEExpertFlag(bank.experts[0].flags, DeviceMoEExpertFlags::Replicated));
        EXPECT_TRUE(hasMoEExpertFlag(bank.experts[0].flags, DeviceMoEExpertFlags::LocalCompute));
    }

    TEST(Test__MoERuntimeTable, DeviceRebalanceOwnershipTransferMovesPrimaryResidency)
    {
        DeviceMoERebalancePlanEntry plan;
        plan.op = static_cast<uint32_t>(DeviceMoERebalancePlanOp::OwnershipTransfer);
        plan.layer = 0;
        plan.expert = 0;
        plan.source_participant = 0;
        plan.destination_participant = 1;
        plan.source_resident_mask = 0b001u;
        plan.destination_slot = 0;
        plan.payload_slot = 0;

        MoERuntimeTable destination_table(DeviceId::cpu(), 1, 4, 2);
        auto destination_update = updateForEpoch(1, 4);
        destination_update.participant_id = 1;
        destination_update.participant_count = 2;
        destination_update.experts[0].owner_participant = 0;
        destination_update.local_compute_mask = {0, 1, 0, 1};
        destination_update.replica_role = {
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
        };
        destination_update.resident_participant_mask = {0b001u, 0b010u, 0b001u, 0b010u};
        ASSERT_TRUE(destination_table.prepareInactiveBank(0, destination_update));
        ASSERT_TRUE(destination_table.flipActiveBank(0, 1, nullptr));

        auto destination_config =
            rebalanceConfig(/*layers=*/1, /*experts=*/4, /*top_k=*/2,
                            /*participant=*/1, /*participants=*/2,
                            /*window_tokens=*/1, /*max_hot_replicas=*/0);
        std::vector<DeviceMoEExpertDirectoryEntry> gathered_directory(
            static_cast<size_t>(destination_config.participant_count) *
            static_cast<size_t>(destination_config.num_layers) *
            static_cast<size_t>(destination_config.num_experts));
        auto &source_entry = gathered_directory[
            deviceMoEDirectoryIndex(destination_config, 0, 0, 0)];
        source_entry.descriptor = expertDesc(0, 0, 0);
        source_entry.layer = 0;
        source_entry.expert = 0;
        source_entry.participant = 0;
        source_entry.resident_mask = 0b001u;
        source_entry.flags =
            static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::Valid) |
            static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::Resident) |
            static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::LocalCompute);
        ASSERT_TRUE(deviceMoEDirectoryCopyReady(source_entry));

        std::vector<DeviceMoEExpertDirectoryEntry> transfer_slots(1);
        auto &slot = transfer_slots[0];
        slot.descriptor = expertDesc(0, 1, 23);
        slot.layer = plan.layer;
        slot.expert = plan.expert;
        slot.participant = 1;
        slot.resident_mask = 0b010u;
        slot.slot_index = 23;
        slot.flags =
            static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::Valid) |
            static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::Resident) |
            static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::TransferSlot) |
            static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::CopyComplete);
        ASSERT_TRUE(deviceMoEDirectoryCopyReady(slot));

        DeviceMoERebalanceApplyStatus destination_status;
        ASSERT_TRUE(applyDeviceMoERebalanceArrivalsHost(
            destination_table.deviceLayerState(0),
            &plan,
            1,
            gathered_directory.data(),
            transfer_slots.data(),
            static_cast<uint32_t>(transfer_slots.size()),
            destination_config,
            &destination_status));

        const auto &destination_bank =
            destination_table.hostLayerState(0).banks[destination_table.hostLayerState(0).active_bank];
        EXPECT_EQ(destination_status.applied_arrivals, 1u);
        EXPECT_EQ(destination_bank.experts[0].owner_participant, 1);
        EXPECT_EQ(destination_bank.experts[0].local_slot, 23);
        EXPECT_EQ(destination_bank.local_compute_mask[0], 1u);
        EXPECT_EQ(destination_bank.replica_role[0],
                  static_cast<uint8_t>(DeviceMoEReplicaRole::Primary));
        EXPECT_EQ(destination_bank.resident_participant_mask[0], 0b010u);
        EXPECT_FALSE(hasMoEExpertFlag(destination_bank.experts[0].flags,
                                      DeviceMoEExpertFlags::Replicated));

        MoERuntimeTable source_table(DeviceId::cpu(), 1, 4, 2);
        auto source_update = updateForEpoch(1, 4);
        source_update.participant_id = 0;
        source_update.participant_count = 2;
        source_update.experts[0].owner_participant = 0;
        source_update.local_compute_mask = {1, 0, 1, 0};
        source_update.replica_role = {
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
        };
        source_update.resident_participant_mask = {0b001u, 0b010u, 0b001u, 0b010u};
        ASSERT_TRUE(source_table.prepareInactiveBank(0, source_update));
        ASSERT_TRUE(source_table.flipActiveBank(0, 1, nullptr));

        auto source_config =
            rebalanceConfig(/*layers=*/1, /*experts=*/4, /*top_k=*/2,
                            /*participant=*/0, /*participants=*/2,
                            /*window_tokens=*/1, /*max_hot_replicas=*/0);
        DeviceMoERebalanceApplyStatus source_status;
        ASSERT_TRUE(applyDeviceMoERebalanceArrivalsHost(
            source_table.deviceLayerState(0),
            &plan,
            1,
            nullptr,
            nullptr,
            0,
            source_config,
            &source_status));

        const auto &source_bank =
            source_table.hostLayerState(0).banks[source_table.hostLayerState(0).active_bank];
        EXPECT_EQ(source_status.applied_arrivals, 0u);
        EXPECT_EQ(source_bank.experts[0].owner_participant, 1);
        EXPECT_EQ(source_bank.local_compute_mask[0], 0u);
        EXPECT_EQ(source_bank.replica_role[0],
                  static_cast<uint8_t>(DeviceMoEReplicaRole::None));
        EXPECT_EQ(source_bank.resident_participant_mask[0], 0b010u);
        EXPECT_FALSE(hasMoEExpertFlag(source_bank.experts[0].flags,
                                      DeviceMoEExpertFlags::Resident));
        EXPECT_FALSE(hasMoEExpertFlag(source_bank.experts[0].flags,
                                      DeviceMoEExpertFlags::LocalCompute));
        EXPECT_FALSE(hasMoEExpertFlag(source_bank.experts[0].flags,
                                      DeviceMoEExpertFlags::Replicated));
    }

    TEST(Test__MoERuntimeTable, DeviceRebalancePolicyReportsTransferPlanOverflow)
    {
        MoERuntimeTable table(DeviceId::cpu(), 1, 4, 2);
        auto update = updateForEpoch(1, 4);
        update.participant_id = 1;
        update.participant_count = 3;
        update.experts[0].owner_participant = 0;
        update.experts[1].owner_participant = 0;
        update.experts[2].owner_participant = 1;
        update.experts[3].owner_participant = 2;
        update.local_compute_mask = {0, 0, 1, 0};
        update.replica_role = {
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
        };
        update.resident_participant_mask = {0b001u, 0b001u, 0b010u, 0b100u};
        ASSERT_TRUE(table.prepareInactiveBank(0, update));
        ASSERT_TRUE(table.flipActiveBank(0, 1, nullptr));

        auto config = rebalanceConfig(/*layers=*/1, /*experts=*/4, /*top_k=*/2,
                                      /*participant=*/1, /*participants=*/3,
                                      /*window_tokens=*/1, /*max_hot_replicas=*/1);
        config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::PlanMissingArrivals);
        std::vector<uint64_t> gathered(
            static_cast<size_t>(config.participant_count) *
                static_cast<size_t>(config.num_layers) *
                static_cast<size_t>(config.num_experts),
            0);
        addGatheredCount(gathered, config, 0, 0, 0, 6);

        DeviceMoERebalancePlanEntry plan[1];
        uint32_t plan_count = 99;
        DeviceMoERebalanceStatus status;
        ASSERT_TRUE(applyDeviceMoERebalancePolicyHost(
            table.deviceLayerState(0),
            gathered.data(),
            config,
            &status,
            plan,
            &plan_count,
            0));

        EXPECT_EQ(status.status_code, static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok));
        EXPECT_EQ(status.planned_arrivals, 0u);
        EXPECT_EQ(status.plan_overflow, 1u);
        EXPECT_EQ(status.skipped_no_resident, 0u)
            << "A full command buffer should stop ranking rather than misclassify capacity as a missing source.";
        EXPECT_EQ(plan_count, 0u);
    }

    TEST(Test__MoERuntimeTable, DeviceRebalancePolicyHandlesFourParticipantDomains)
    {
        MoERuntimeTable table(DeviceId::cpu(), 1, 4, 2);
        auto update = updateForEpoch(1, 4);
        update.participant_id = 3;
        update.participant_count = 4;
        update.experts[0].owner_participant = 0;
        update.experts[1].owner_participant = 1;
        update.experts[2].owner_participant = 2;
        update.experts[3].owner_participant = 3;
        update.local_compute_mask = {0, 0, 0, 1};
        update.replica_role = {
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::None),
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
        };
        update.resident_participant_mask = {0b1001u, 0b0010u, 0b0100u, 0b1000u};
        ASSERT_TRUE(table.prepareInactiveBank(0, update));
        ASSERT_TRUE(table.flipActiveBank(0, 1, nullptr));

        const auto config = rebalanceConfig(/*layers=*/1, /*experts=*/4, /*top_k=*/2,
                                            /*participant=*/3, /*participants=*/4,
                                            /*window_tokens=*/1, /*max_hot_replicas=*/1);
        std::vector<uint64_t> gathered(
            static_cast<size_t>(config.participant_count) *
                static_cast<size_t>(config.num_layers) *
                static_cast<size_t>(config.num_experts),
            0);
        addGatheredCount(gathered, config, 0, 0, 0, 8);

        DeviceMoERebalanceStatus status;
        ASSERT_TRUE(applyDeviceMoERebalancePolicyHost(
            table.deviceLayerState(0), gathered.data(), config, &status));

        const auto &state = table.hostLayerState(0);
        const auto &bank = state.banks[state.active_bank];
        EXPECT_EQ(status.status_code, static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok));
        EXPECT_EQ(bank.local_compute_mask[0], 1u)
            << "policy must not assume a two-participant domain";
        EXPECT_TRUE(hasMoEExpertFlag(bank.experts[0].flags, DeviceMoEExpertFlags::Replicated));
        EXPECT_EQ(bank.replica_role[0], static_cast<uint8_t>(DeviceMoEReplicaRole::Replica));
        EXPECT_EQ(bank.local_compute_mask[3], 1u);
        EXPECT_EQ(bank.replica_role[3], static_cast<uint8_t>(DeviceMoEReplicaRole::Primary));
    }

    TEST(Test__MoERuntimeTable, ConstructorRejectsInvalidBoundsAndCpuMirroring)
    {
        EXPECT_THROW(MoERuntimeTable(DeviceId::invalid(), 1, 4, 2), std::invalid_argument);
        EXPECT_THROW(MoERuntimeTable(DeviceId::cpu(), 0, 4, 2), std::invalid_argument);
        EXPECT_THROW(MoERuntimeTable(DeviceId::cpu(), 1, 0, 2), std::invalid_argument);
        EXPECT_THROW(MoERuntimeTable(DeviceId::cpu(), 1, static_cast<int>(kDeviceMoEMaxExperts) + 1, 2),
                     std::invalid_argument);
        EXPECT_THROW(MoERuntimeTable(DeviceId::cpu(), 1, 4, static_cast<int>(kDeviceMoEMaxTopK) + 1),
                     std::invalid_argument);
        EXPECT_THROW(MoERuntimeTable(DeviceId::cpu(), 1, 4, 2, true), std::runtime_error);
        DeviceMoERuntimeTable::Config prefill_without_mirror;
        prefill_without_mirror.device_id = DeviceId::cpu();
        prefill_without_mirror.num_layers = 1;
        prefill_without_mirror.num_experts = 4;
        prefill_without_mirror.top_k = 2;
        prefill_without_mirror.prefill_token_capacity = 8;
        EXPECT_THROW({ MoERuntimeTable table(prefill_without_mirror); }, std::runtime_error);
        prefill_without_mirror.prefill_token_capacity = -1;
        EXPECT_THROW({ MoERuntimeTable table(prefill_without_mirror); }, std::invalid_argument);
    }

#ifdef HAVE_ROCM
    TEST(Test__MoERuntimeTable, RocmPrefillRouteScratchAllocationTracksCapacity)
    {
        int device_count = 0;
        if (hipGetDeviceCount(&device_count) != hipSuccess || device_count <= 0)
            GTEST_SKIP() << "No ROCm GPU available";

        DeviceMoERuntimeTable::Config config;
        config.device_id = DeviceId::rocm(0);
        config.num_layers = 2;
        config.num_experts = 4;
        config.top_k = 2;
        config.mirror_to_device = true;
        config.prefill_token_capacity = 8;

        MoERuntimeTable table(config);
        EXPECT_TRUE(table.hasPrefillRouteScratchCapacity(0, 8));
        EXPECT_TRUE(table.hasPrefillRouteScratchCapacity(1, 4));
        EXPECT_FALSE(table.hasPrefillRouteScratchCapacity(0, 9));

        const auto &state = table.hostLayerState(0);
        EXPECT_EQ(state.prefill_token_capacity, 8u);
        EXPECT_EQ(state.prefill_route_capacity, 16u);
        EXPECT_NE(state.route_expert_ids, nullptr);
        EXPECT_NE(state.route_weights, nullptr);
        EXPECT_NE(state.expert_counts, nullptr);
        EXPECT_NE(state.expert_offsets, nullptr);
        EXPECT_NE(state.grouped_token_ids, nullptr);
        EXPECT_NE(state.grouped_route_weights, nullptr);
        EXPECT_NE(state.reserved_ptrs[0], nullptr)
            << "prefill LLEP split-table scratch must be a first-class runtime buffer";
        EXPECT_NE(state.reserved_ptrs[1], nullptr)
            << "full current-batch LLEP assignment spans must be first-class runtime scratch";
        EXPECT_NE(state.reserved_ptrs[2], nullptr)
            << "full current-batch LLEP transfer plans must be first-class runtime scratch";
        EXPECT_EQ(state.reserved_u64[0], 4u * kDeviceMoEMaxParticipants);
        EXPECT_EQ(state.reserved_u64[1], 4u * kDeviceMoEMaxParticipants);
        EXPECT_EQ(state.reserved_u64[2], 0u);
        EXPECT_EQ(state.reserved_u64[3], 0u);
        EXPECT_NE(table.deviceLayerState(0), &table.hostLayerState(0));

        void *split_scratch_before_reset = state.reserved_ptrs[0];
        void *span_scratch_before_reset = state.reserved_ptrs[1];
        void *transfer_scratch_before_reset = state.reserved_ptrs[2];

        hipStream_t stream = nullptr;
        ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
        table.resetDecodeRuntimeState(stream);
        EXPECT_EQ(table.hostLayerState(0).reserved_ptrs[0], split_scratch_before_reset)
            << "decode-runtime reset must preserve prefill LLEP scratch bindings";
        EXPECT_EQ(table.hostLayerState(0).reserved_ptrs[1], span_scratch_before_reset)
            << "decode-runtime reset must preserve full current-batch LLEP span scratch";
        EXPECT_EQ(table.hostLayerState(0).reserved_ptrs[2], transfer_scratch_before_reset)
            << "decode-runtime reset must preserve full current-batch LLEP transfer scratch";

        table.ensurePrefillRouteScratchCapacity(12, stream);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
        EXPECT_TRUE(table.hasPrefillRouteScratchCapacity(0, 12));
        EXPECT_EQ(table.hostLayerState(0).prefill_token_capacity, 12u);
        EXPECT_EQ(table.hostLayerState(0).prefill_route_capacity, 24u);
        EXPECT_NE(table.hostLayerState(0).reserved_ptrs[0], nullptr);
        EXPECT_NE(table.hostLayerState(0).reserved_ptrs[1], nullptr);
        EXPECT_NE(table.hostLayerState(0).reserved_ptrs[2], nullptr);
        EXPECT_EQ(table.hostLayerState(0).reserved_u64[0], 4u * kDeviceMoEMaxParticipants);
        EXPECT_EQ(table.hostLayerState(0).reserved_u64[1], 4u * kDeviceMoEMaxParticipants);
    }
#endif

} // namespace llaminar2::test
