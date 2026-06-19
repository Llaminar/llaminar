/**
 * @file Test__MoERuntimeTable.cpp
 * @brief Unit tests for graph-facing MoE runtime placement tables.
 */

#include "execution/moe/MoERuntimeTable.h"
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

        DeviceMoEExpertDescriptor expertDesc(int expert_id,
                                             int owner,
                                             int local_slot,
                                             DeviceMoEExpertFlags flags = DeviceMoEExpertFlags::Valid |
                                                                          DeviceMoEExpertFlags::Resident |
                                                                          DeviceMoEExpertFlags::LocalCompute)
        {
            const uintptr_t base = 0x10000000u + static_cast<uintptr_t>(expert_id) * 0x10000u;
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
    }

    TEST(Test__MoERuntimeTable, PrepareInactiveBankCopiesDescriptorsMasksAndReplicaRoles)
    {
        MoERuntimeTable table(DeviceId::cpu(), 1, 4, 2);
        const auto update = updateForEpoch(1, 4);

        ASSERT_TRUE(table.prepareInactiveBank(0, update));
        const auto &state = table.hostLayerState(0);
        const auto &bank = state.banks[1];

        EXPECT_EQ(state.active_bank, 0u);
        EXPECT_EQ(bank.epoch, 1u);
        EXPECT_EQ(bank.expert_count, 4u);
        EXPECT_EQ(bank.experts[2].logical_expert_id, 2);
        EXPECT_EQ(bank.experts[2].owner_participant, 0);
        EXPECT_EQ(bank.experts[2].local_slot, 2);
        EXPECT_TRUE(hasMoEExpertFlag(bank.experts[2].flags, DeviceMoEExpertFlags::Valid));
        EXPECT_EQ(bank.local_compute_mask[2], 1u);
        EXPECT_EQ(bank.replica_role[2], static_cast<uint8_t>(DeviceMoEReplicaRole::Primary));
        EXPECT_EQ(bank.experts[2].gate.payload, update.experts[2].gate.payload);
        EXPECT_EQ(bank.experts[2].up.scales, update.experts[2].up.scales);
        EXPECT_EQ(bank.experts[2].down.n, update.experts[2].down.n);
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
            EXPECT_EQ(after_reset->decode_histogram[expert], 0u);
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
        table.hostLayerState(1).decode_histogram[1] = 2;
        table.hostLayerState(1).decode_histogram[3] = 2;

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
                EXPECT_EQ(table.hostLayerState(layer).decode_histogram[expert], 0u);

        ASSERT_TRUE(table.syncDecodeHistogramToHost(hist));
        EXPECT_EQ(hist.activationCount(0, 0), 3u);
        EXPECT_EQ(hist.activationCount(1, 1), 2u);
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
        EXPECT_NE(table.deviceLayerState(0), &table.hostLayerState(0));

        table.ensurePrefillRouteScratchCapacity(12);
        EXPECT_TRUE(table.hasPrefillRouteScratchCapacity(0, 12));
        EXPECT_EQ(table.hostLayerState(0).prefill_token_capacity, 12u);
        EXPECT_EQ(table.hostLayerState(0).prefill_route_capacity, 24u);
    }
#endif

} // namespace llaminar2::test
