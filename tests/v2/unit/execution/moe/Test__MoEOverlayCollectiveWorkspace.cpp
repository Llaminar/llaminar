#include "execution/moe/MoEOverlaySparseCollective.h"

#include <gtest/gtest.h>

using namespace llaminar2;

namespace
{
    MoEOverlayCollectiveKey dispatchKey(uint64_t sequence)
    {
        MoEOverlayCollectiveKey key;
        key.generation_id = 1;
        key.step_id = 2;
        key.layer_idx = 3;
        key.tier_idx = 1;
        key.domain_id = 7;
        key.direction = MoEOverlayCollectiveDirection::Dispatch;
        key.sequence = sequence;
        return key;
    }

    MoEOverlayCollectiveKey returnKey(uint64_t sequence)
    {
        auto key = dispatchKey(sequence);
        key.direction = MoEOverlayCollectiveDirection::ReturnReduce;
        return key;
    }

} // namespace

TEST(Test__MoEOverlayCollectiveWorkspace, EnsureCapacityAndResetReuseStoragePointers)
{
    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(16, 32, 8, 2, DeviceId::cpu());

    auto before = workspace.dispatchReceive(5, 1);
    const int32_t *row_ids_ptr = before.row_ids_host;
    const int32_t *entry_offsets_ptr = before.entry_offsets_host;
    const int32_t *expert_ids_ptr = before.expert_ids_host;
    const float *route_weights_ptr = before.route_weights_host;
    const float *hidden_ptr = before.hidden_rows_fp32;
    const size_t row_capacity = before.row_capacity;
    const size_t entry_capacity = before.entry_capacity;

    workspace.resetForStep(4, 9);

    auto after = workspace.dispatchReceive(5, 1);
    EXPECT_EQ(after.live_row_count, 0u);
    EXPECT_EQ(after.live_entry_count, 0u);
    EXPECT_EQ(after.row_capacity, row_capacity);
    EXPECT_EQ(after.entry_capacity, entry_capacity);
    EXPECT_EQ(after.row_ids_host, row_ids_ptr);
    EXPECT_EQ(after.entry_offsets_host, entry_offsets_ptr);
    EXPECT_EQ(after.expert_ids_host, expert_ids_ptr);
    EXPECT_EQ(after.route_weights_host, route_weights_ptr);
    EXPECT_EQ(after.hidden_rows_fp32, hidden_ptr);
    EXPECT_EQ(after.entry_offsets_host[0], 0);
}

TEST(Test__MoEOverlayCollectiveWorkspace, LocalSparseDispatchMovesPayloadAndNoOpCompletesKey)
{
    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(16, 32, 4, 2, DeviceId::cpu());

    MoEOverlayLocalSparseCollectiveContext collective({.participant_count = 2, .slot_count = 8});
    auto key = dispatchKey(33);

    auto outbound0 = workspace.localExpertInput(3, 1);
    outbound0.key = key;
    outbound0.source_participant = 0;
    outbound0.target_participant = 1;
    outbound0.live_row_count = 2;
    outbound0.live_entry_count = 4;
    outbound0.row_ids_host[0] = 10;
    outbound0.row_ids_host[1] = 14;
    outbound0.entry_offsets_host[0] = 0;
    outbound0.entry_offsets_host[1] = 2;
    outbound0.entry_offsets_host[2] = 4;
    outbound0.expert_ids_host[0] = 5;
    outbound0.expert_ids_host[1] = 7;
    outbound0.expert_ids_host[2] = 9;
    outbound0.expert_ids_host[3] = 11;
    outbound0.route_weights_host[0] = 0.5f;
    outbound0.route_weights_host[1] = 0.5f;
    outbound0.route_weights_host[2] = 0.25f;
    outbound0.route_weights_host[3] = 0.75f;
    for (size_t index = 0; index < outbound0.live_row_count * static_cast<size_t>(outbound0.d_model); ++index)
        outbound0.hidden_rows_fp32[index] = static_cast<float>(100 + index);

    auto inbound0 = workspace.dispatchReceive(3, 1);
    auto first = collective.dispatch(key, outbound0, &inbound0, nullptr);
    ASSERT_TRUE(first.ok) << first.error;
    EXPECT_FALSE(first.collective_complete);

    auto no_op = workspace.localExpertInput(3, 1);
    no_op.key = key;
    no_op.source_participant = 1;
    no_op.target_participant = 0;
    no_op.live_row_count = 0;
    no_op.live_entry_count = 0;
    no_op.entry_offsets_host[0] = 0;

    auto inbound1 = workspace.dispatchReceive(3, 1);
    auto second = collective.dispatch(key, no_op, &inbound1, nullptr);
    ASSERT_TRUE(second.ok) << second.error;
    EXPECT_TRUE(second.collective_complete);

    EXPECT_EQ(inbound1.live_row_count, 2u);
    EXPECT_EQ(inbound1.live_entry_count, 4u);
    EXPECT_EQ(inbound1.row_ids_host[0], 10);
    EXPECT_EQ(inbound1.row_ids_host[1], 14);
    EXPECT_EQ(inbound1.entry_offsets_host[0], 0);
    EXPECT_EQ(inbound1.entry_offsets_host[1], 2);
    EXPECT_EQ(inbound1.entry_offsets_host[2], 4);
    EXPECT_EQ(inbound1.expert_ids_host[0], 5);
    EXPECT_EQ(inbound1.expert_ids_host[3], 11);
    EXPECT_FLOAT_EQ(inbound1.route_weights_host[2], 0.25f);
    EXPECT_FLOAT_EQ(inbound1.hidden_rows_fp32[0], 100.0f);
    EXPECT_FLOAT_EQ(inbound1.hidden_rows_fp32[7], 107.0f);

    auto stale = collective.dispatch(key, no_op, &inbound1, nullptr);
    EXPECT_FALSE(stale.ok);
    EXPECT_EQ(stale.error_code, 4);
}

TEST(Test__MoEOverlayCollectiveWorkspace, LocalReturnReduceMovesCompactRowsByKey)
{
    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(16, 32, 4, 2, DeviceId::cpu());

    MoEOverlayLocalSparseCollectiveContext collective({.participant_count = 2, .slot_count = 8});
    auto key = returnKey(34);

    auto outbound0 = workspace.localExpertOutput(3, 1);
    outbound0.key = key;
    outbound0.source_participant = 0;
    outbound0.target_participant = 1;
    outbound0.live_row_count = 2;
    outbound0.row_ids_host[0] = 3;
    outbound0.row_ids_host[1] = 4;
    for (size_t index = 0; index < outbound0.live_row_count * static_cast<size_t>(outbound0.d_model); ++index)
        outbound0.output_rows_fp32[index] = static_cast<float>(200 + index);

    auto inbound0 = workspace.returnReceive(3, 1);
    auto first = collective.returnReduce(key, outbound0, &inbound0, nullptr);
    ASSERT_TRUE(first.ok) << first.error;
    EXPECT_FALSE(first.collective_complete);

    auto no_op = workspace.localExpertOutput(3, 1);
    no_op.key = key;
    no_op.source_participant = 1;
    no_op.target_participant = 0;
    no_op.live_row_count = 0;

    auto inbound1 = workspace.returnReceive(3, 1);
    auto second = collective.returnReduce(key, no_op, &inbound1, nullptr);
    ASSERT_TRUE(second.ok) << second.error;
    EXPECT_TRUE(second.collective_complete);

    EXPECT_EQ(inbound1.live_row_count, 2u);
    EXPECT_EQ(inbound1.row_ids_host[0], 3);
    EXPECT_EQ(inbound1.row_ids_host[1], 4);
    EXPECT_FLOAT_EQ(inbound1.output_rows_fp32[0], 200.0f);
    EXPECT_FLOAT_EQ(inbound1.output_rows_fp32[7], 207.0f);
}
