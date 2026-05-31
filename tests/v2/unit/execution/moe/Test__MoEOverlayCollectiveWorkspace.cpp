#include "execution/moe/MoEOverlaySparseCollective.h"
#include "execution/compute_stages/stages/MoESparseDispatchStage.h"
#include "execution/compute_stages/stages/MoESparseReturnReduceStage.h"
#include "collective/ITPContext.h"
#include "tensors/Tensors.h"
#include "mocks/MockComputeStage.h"

#include <gtest/gtest.h>

#include <set>

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

    class CountingTPContext final : public ITPContext
    {
    public:
        TPScope scope() const override { return TPScope::LOCAL; }
        int degree() const override { return 2; }
        int myIndex() const override { return 0; }
        CollectiveBackendType backend() const override { return CollectiveBackendType::HOST; }
        bool allreduce(TensorBase *tensor) override
        {
            (void)tensor;
            return true;
        }
        bool broadcast(TensorBase *tensor, int source_index = 0) override
        {
            (void)tensor;
            last_source_index = source_index;
            ++broadcast_calls;
            return true;
        }
        bool allgather(const TensorBase *local_shard, TensorBase *global_tensor) override
        {
            (void)local_shard;
            (void)global_tensor;
            return true;
        }

        int broadcast_calls = 0;
        int last_source_index = -1;
    };

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

TEST(Test__MoEOverlayCollectiveWorkspace, MTPCollectiveKeysDoNotAliasMainGraphKeys)
{
    const auto main_key = makeMoEOverlayCollectiveKey(
        5,
        8,
        3,
        1,
        7,
        2,
        MoEOverlayCollectiveDirection::Dispatch);
    const auto mtp_key = makeMTPMoEOverlayCollectiveKey(
        5,
        8,
        0,
        3,
        1,
        7,
        2,
        MoEOverlayCollectiveDirection::Dispatch);

    EXPECT_TRUE(main_key.isValid());
    EXPECT_TRUE(mtp_key.isValid());
    EXPECT_EQ(main_key.key_namespace, MoEOverlayCollectiveNamespace::Main);
    EXPECT_EQ(mtp_key.key_namespace, MoEOverlayCollectiveNamespace::MTP);
    EXPECT_NE(main_key, mtp_key);
    EXPECT_NE(main_key.sequence, mtp_key.sequence);
    EXPECT_NE(main_key.toString(), mtp_key.toString());
    EXPECT_STREQ(toString(main_key.key_namespace), "Main");
    EXPECT_STREQ(toString(mtp_key.key_namespace), "MTP");
}

TEST(Test__MoEOverlayCollectiveWorkspace, MTPCollectiveKeysSeparateDepthParticipantAndDirection)
{
    const auto depth0 = makeMTPMoEOverlayCollectiveKey(
        5,
        8,
        0,
        3,
        1,
        7,
        2,
        MoEOverlayCollectiveDirection::Dispatch);
    const auto depth1 = makeMTPMoEOverlayCollectiveKey(
        5,
        8,
        1,
        3,
        1,
        7,
        2,
        MoEOverlayCollectiveDirection::Dispatch);
    const auto participant3 = makeMTPMoEOverlayCollectiveKey(
        5,
        8,
        0,
        3,
        1,
        7,
        3,
        MoEOverlayCollectiveDirection::Dispatch);
    const auto return_key = makeMTPMoEOverlayCollectiveKey(
        5,
        8,
        0,
        3,
        1,
        7,
        2,
        MoEOverlayCollectiveDirection::ReturnReduce);

    std::set<MoEOverlayCollectiveKey> keys{depth0, depth1, participant3, return_key};
    EXPECT_EQ(keys.size(), 4u);
    EXPECT_TRUE(depth0.isValid());
    EXPECT_TRUE(depth1.isValid());
    EXPECT_TRUE(participant3.isValid());
    EXPECT_TRUE(return_key.isValid());
    EXPECT_NE(depth0.sequence, depth1.sequence);
    EXPECT_NE(depth0.sequence, participant3.sequence);
    EXPECT_NE(depth0.sequence, return_key.sequence);
}

TEST(Test__MoEOverlayCollectiveWorkspace, MTPCollectiveKeyRequiresDepth)
{
    auto key = dispatchKey(77);
    key.key_namespace = MoEOverlayCollectiveNamespace::MTP;
    key.participant_id = 0;
    EXPECT_FALSE(key.isValid());

    key.mtp_depth = 0;
    EXPECT_TRUE(key.isValid());
}

TEST(Test__MoEOverlayCollectiveWorkspace, LocalSparseCollectiveSeparatesMainAndMTPNamespaces)
{
    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(4, 8, 4, 2, DeviceId::cpu());

    MoEOverlayLocalSparseCollectiveContext collective({.participant_count = 1, .slot_count = 1});
    const auto main_key = dispatchKey(33);
    auto mtp_key = main_key;
    mtp_key.key_namespace = MoEOverlayCollectiveNamespace::MTP;
    mtp_key.mtp_depth = 0;
    mtp_key.participant_id = 0;

    auto main_outbound = workspace.localExpertInput(3, 1);
    main_outbound.key = main_key;
    main_outbound.source_participant = 0;
    main_outbound.target_participant = 0;
    main_outbound.live_row_count = 0;
    main_outbound.live_entry_count = 0;
    main_outbound.entry_offsets_host[0] = 0;

    auto main_inbound = workspace.dispatchReceive(3, 1);
    const auto main_result = collective.dispatch(main_key, main_outbound, &main_inbound, nullptr);
    ASSERT_TRUE(main_result.ok) << main_result.error;
    EXPECT_TRUE(main_result.collective_complete);

    auto mtp_outbound = workspace.localExpertInput(3, 1);
    mtp_outbound.key = mtp_key;
    mtp_outbound.source_participant = 0;
    mtp_outbound.target_participant = 0;
    mtp_outbound.live_row_count = 0;
    mtp_outbound.live_entry_count = 0;
    mtp_outbound.entry_offsets_host[0] = 0;

    auto mtp_inbound = workspace.dispatchReceive(3, 1);
    const auto mtp_result = collective.dispatch(mtp_key, mtp_outbound, &mtp_inbound, nullptr);
    EXPECT_TRUE(mtp_result.ok) << mtp_result.error;
    EXPECT_TRUE(mtp_result.collective_complete);
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

TEST(Test__MoEOverlayCollectiveWorkspace, SparseDispatchStageReportsManualBoundaryCompletion)
{
    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(16, 32, 4, 2, DeviceId::cpu());

    MoEOverlayLocalSparseCollectiveContext collective({.participant_count = 2, .slot_count = 8});
    llaminar2::testing::MockDeviceContext ctx(DeviceId::cpu(), ComputeBackendType::CPU);
    auto key = dispatchKey(36);

    auto inbound0 = workspace.dispatchReceive(3, 1);
    MoESparseDispatchStage::Params first_params;
    first_params.device_id = DeviceId::cpu();
    first_params.collective_context = &collective;
    first_params.workspace = &workspace;
    first_params.key = key;
    first_params.source_participant = 0;
    first_params.target_participant = 1;
    first_params.seq_len = 4;
    first_params.top_k = 2;
    first_params.d_model = 4;
    first_params.inbound_rows = &inbound0;

    MoESparseDispatchStage first_stage(std::move(first_params));
    ASSERT_TRUE(first_stage.execute(&ctx));
    EXPECT_TRUE(first_stage.isManualGraphBoundary());
    EXPECT_FALSE(first_stage.manualGraphBoundaryComplete());

    auto inbound1 = workspace.dispatchReceive(3, 1);
    MoESparseDispatchStage::Params second_params;
    second_params.device_id = DeviceId::cpu();
    second_params.collective_context = &collective;
    second_params.workspace = &workspace;
    second_params.key = key;
    second_params.source_participant = 1;
    second_params.target_participant = 0;
    second_params.seq_len = 4;
    second_params.top_k = 2;
    second_params.d_model = 4;
    second_params.inbound_rows = &inbound1;

    MoESparseDispatchStage second_stage(std::move(second_params));
    ASSERT_TRUE(second_stage.execute(&ctx));
    EXPECT_TRUE(second_stage.isManualGraphBoundary());
    EXPECT_TRUE(second_stage.manualGraphBoundaryComplete());
}

TEST(Test__MoEOverlayCollectiveWorkspace, ReturnReduceBroadcastWaitsForCollectiveComplete)
{
    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(16, 32, 4, 2, DeviceId::cpu());

    MoEOverlayLocalSparseCollectiveContext collective({.participant_count = 2, .slot_count = 8});
    CountingTPContext tp_context;
    llaminar2::testing::MockDeviceContext ctx(DeviceId::cpu(), ComputeBackendType::CPU);
    auto key = returnKey(35);

    auto outbound0 = workspace.localExpertOutput(3, 1);
    outbound0.key = key;
    outbound0.source_participant = 0;
    outbound0.target_participant = 1;
    outbound0.live_row_count = 1;
    outbound0.row_ids_host[0] = 2;
    for (int col = 0; col < outbound0.d_model; ++col)
        outbound0.output_rows_fp32[col] = static_cast<float>(10 + col);

    FP32Tensor dense0(std::vector<size_t>{4, 4});
    auto inbound0 = workspace.returnReceive(3, 1);
    MoESparseReturnReduceStage::Params first_params;
    first_params.device_id = DeviceId::cpu();
    first_params.collective_context = &collective;
    first_params.key = key;
    first_params.source_participant = 0;
    first_params.target_participant = 1;
    first_params.outbound_rows = &outbound0;
    first_params.inbound_rows = &inbound0;
    first_params.dense_output = &dense0;
    first_params.seq_len = 4;
    first_params.d_model = 4;
    first_params.broadcast_after_scatter = true;
    first_params.continuation_tp_context = &tp_context;
    first_params.continuation_root_tp_index = 1;

    MoESparseReturnReduceStage first_stage(std::move(first_params));
    ASSERT_TRUE(first_stage.execute(&ctx));
    EXPECT_TRUE(first_stage.isManualGraphBoundary());
    EXPECT_FALSE(first_stage.manualGraphBoundaryComplete());
    EXPECT_EQ(tp_context.broadcast_calls, 0);

    auto outbound1 = workspace.localExpertOutput(3, 1);
    outbound1.key = key;
    outbound1.source_participant = 1;
    outbound1.target_participant = 0;
    outbound1.live_row_count = 0;

    FP32Tensor dense1(std::vector<size_t>{4, 4});
    auto inbound1 = workspace.returnReceive(3, 1);
    MoESparseReturnReduceStage::Params second_params;
    second_params.device_id = DeviceId::cpu();
    second_params.collective_context = &collective;
    second_params.key = key;
    second_params.source_participant = 1;
    second_params.target_participant = 1;
    second_params.outbound_rows = &outbound1;
    second_params.inbound_rows = &inbound1;
    second_params.dense_output = &dense1;
    second_params.seq_len = 4;
    second_params.d_model = 4;
    second_params.broadcast_after_scatter = true;
    second_params.continuation_tp_context = &tp_context;
    second_params.continuation_root_tp_index = 1;

    MoESparseReturnReduceStage second_stage(std::move(second_params));
    ASSERT_TRUE(second_stage.execute(&ctx));
    EXPECT_TRUE(second_stage.isManualGraphBoundary());
    EXPECT_TRUE(second_stage.manualGraphBoundaryComplete());
    EXPECT_EQ(tp_context.broadcast_calls, 1);
    EXPECT_EQ(tp_context.last_source_index, 1);
}
