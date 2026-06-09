/**
 * @file Test__HiddenStateRowSelectStage.cpp
 * @brief Unit tests for bucketed-prefill hidden-state row selection.
 *
 * Verifies that dynamic PrefillReplayParams select the expected hidden-state row
 * on CPU and that LMHeadStage can consume the stable one-row scratch at GEMM
 * activation offset zero while the selected source row changes.
 */

#include <gtest/gtest.h>

#include "execution/compute_stages/stages/HiddenStateRowSelectStage.h"
#include "execution/compute_stages/stages/HiddenStateRowsSelectStage.h"
#include "execution/compute_stages/stages/LMHeadStage.h"
#include "execution/local_execution/device/WorkspaceDescriptor.h"
#include "tensors/Tensors.h"
#include "utils/PreparedWeightTestHarness.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    /// @brief Fill each row with a distinct, easy-to-check affine pattern.
    std::unique_ptr<FP32Tensor> makeHiddenStates(int seq_len, int d_model)
    {
        auto hidden = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model)},
            DeviceId::cpu());
        float *hidden_data = hidden->mutable_data();
        for (int row = 0; row < seq_len; ++row)
        {
            for (int column = 0; column < d_model; ++column)
            {
                hidden_data[static_cast<size_t>(row) * d_model + column] =
                    100.0f * static_cast<float>(row + 1) + static_cast<float>(column);
            }
        }
        return hidden;
    }

    /// @brief Build deterministic FP32 LM-head weights in [vocab_size, d_model] layout.
    std::unique_ptr<FP32Tensor> makeLmHeadWeights(int vocab_size, int d_model)
    {
        auto weights = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(vocab_size), static_cast<size_t>(d_model)},
            DeviceId::cpu());
        float *weight_data = weights->mutable_data();
        for (int vocab_idx = 0; vocab_idx < vocab_size; ++vocab_idx)
        {
            for (int column = 0; column < d_model; ++column)
            {
                weight_data[static_cast<size_t>(vocab_idx) * d_model + column] =
                    0.01f * static_cast<float>((vocab_idx % 7) - 3) +
                    0.001f * static_cast<float>((column % 5) - 2);
            }
        }
        return weights;
    }

    /// @brief Compute one-row LM-head reference from a selected hidden-state row.
    std::vector<float> computeReferenceLogits(
        const FP32Tensor &hidden,
        const FP32Tensor &weights,
        int selected_row,
        int vocab_size,
        int d_model)
    {
        const float *hidden_data = hidden.data();
        const float *weight_data = weights.data();
        std::vector<float> reference(static_cast<size_t>(vocab_size), 0.0f);
        for (int vocab_idx = 0; vocab_idx < vocab_size; ++vocab_idx)
        {
            float dot_product = 0.0f;
            for (int column = 0; column < d_model; ++column)
            {
                dot_product += hidden_data[static_cast<size_t>(selected_row) * d_model + column] *
                               weight_data[static_cast<size_t>(vocab_idx) * d_model + column];
            }
            reference[static_cast<size_t>(vocab_idx)] = dot_product;
        }
        return reference;
    }

    /// @brief Assert that row zero of logits matches a reference vector.
    void expectLogitsNear(const FP32Tensor &logits, const std::vector<float> &reference)
    {
        const float *logit_data = logits.data();
        for (size_t vocab_idx = 0; vocab_idx < reference.size(); ++vocab_idx)
        {
            EXPECT_NEAR(logit_data[vocab_idx], reference[vocab_idx], 1e-4f)
                << "vocab_idx=" << vocab_idx;
        }
    }

    /// @brief Assert that one row of a batched logits tensor matches a reference vector.
    void expectLogitsRowNear(const FP32Tensor &logits, int row, int vocab_size, const std::vector<float> &reference)
    {
        const float *logit_data = logits.data() + static_cast<size_t>(row) * static_cast<size_t>(vocab_size);
        for (size_t vocab_idx = 0; vocab_idx < reference.size(); ++vocab_idx)
        {
            EXPECT_NEAR(logit_data[vocab_idx], reference[vocab_idx], 1e-4f)
                << "row=" << row << " vocab_idx=" << vocab_idx;
        }
    }

    /// @brief Return max absolute difference between current logits and saved logits.
    float maxDifference(const FP32Tensor &logits, const std::vector<float> &saved_logits)
    {
        const float *logit_data = logits.data();
        float max_difference = 0.0f;
        for (size_t vocab_idx = 0; vocab_idx < saved_logits.size(); ++vocab_idx)
        {
            max_difference = std::max(max_difference, std::fabs(logit_data[vocab_idx] - saved_logits[vocab_idx]));
        }
        return max_difference;
    }
}

TEST(Test__HiddenStateRowSelectStage, CPUReplayParamsChangeSelectedRow)
{
    const int bucket_seq_len = 6;
    const int d_model = 8;
    auto hidden = makeHiddenStates(bucket_seq_len, d_model);
    auto scratch = std::make_unique<FP32Tensor>(
        std::vector<size_t>{1, static_cast<size_t>(d_model)},
        DeviceId::cpu());

    HiddenStateRowSelectStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = hidden.get();
    params.output = scratch.get();
    params.seq_len = bucket_seq_len;
    params.d_model = d_model;
    params.selected_row_idx = 1;

    HiddenStateRowSelectStage stage(params);
    ASSERT_TRUE(stage.execute(nullptr));
    for (int column = 0; column < d_model; ++column)
    {
        EXPECT_FLOAT_EQ(scratch->data()[column], hidden->data()[static_cast<size_t>(1) * d_model + column]);
    }

    stage.updatePrefillReplayParams(IComputeStage::PrefillReplayParams{
        /*real_seq_len=*/4,
        /*bucket_seq_len=*/bucket_seq_len,
        /*token_offset=*/128});
    ASSERT_EQ(stage.selectedRowForTesting(), 3);
    ASSERT_TRUE(stage.execute(nullptr));
    for (int column = 0; column < d_model; ++column)
    {
        EXPECT_FLOAT_EQ(scratch->data()[column], hidden->data()[static_cast<size_t>(3) * d_model + column]);
    }
}

TEST(Test__HiddenStateRowSelectStage, GPUWorkspaceRequirementsDeclareSelectedRowScalar)
{
    HiddenStateRowSelectStage::Params params;
    params.device_id = DeviceId::rocm(0);
    params.seq_len = 8;
    params.d_model = 32;
    HiddenStateRowSelectStage stage(params);

    const WorkspaceRequirements reqs = stage.getWorkspaceRequirements(8, 32, 0);
    ASSERT_EQ(reqs.buffers.size(), 1u);
    EXPECT_NE(reqs.buffers[0].name.find(HiddenStateRowSelectStage::WS_SELECTED_ROW_SCALAR), std::string::npos);
    EXPECT_GE(reqs.buffers[0].size_bytes, sizeof(int));
    EXPECT_TRUE(reqs.buffers[0].required);
}

TEST(Test__HiddenStateRowSelectStage, ReplayRowUpdateKeepsDeclaredWorkspaceNameStable)
{
    HiddenStateRowSelectStage::Params params;
    params.device_id = DeviceId::rocm(0);
    params.seq_len = 8;
    params.d_model = 32;
    params.selected_row_idx = 1;
    HiddenStateRowSelectStage stage(params);

    const WorkspaceRequirements before = stage.getWorkspaceRequirements(8, 32, 0);
    ASSERT_EQ(before.buffers.size(), 1u);
    stage.setSelectedRowForReplay(5);
    EXPECT_EQ(stage.selectedRowForTesting(), 5);
    const WorkspaceRequirements after = stage.getWorkspaceRequirements(8, 32, 0);
    ASSERT_EQ(after.buffers.size(), 1u);
    EXPECT_EQ(before.buffers[0].name, after.buffers[0].name);

    stage.setSelectedRowForReplay(99);
    EXPECT_EQ(stage.selectedRowForTesting(), 7);
    const WorkspaceRequirements clamped = stage.getWorkspaceRequirements(8, 32, 0);
    ASSERT_EQ(clamped.buffers.size(), 1u);
    EXPECT_EQ(before.buffers[0].name, clamped.buffers[0].name);
}

TEST(Test__HiddenStateRowSelectStage, UsesExplicitStableWorkspaceBufferName)
{
    HiddenStateRowSelectStage::Params params;
    params.device_id = DeviceId::rocm(0);
    params.seq_len = 8;
    params.d_model = 32;
    params.workspace_buffer_name = "mtp_terminal_hidden_selected_row";
    HiddenStateRowSelectStage stage(params);

    const WorkspaceRequirements reqs = stage.getWorkspaceRequirements(8, 32, 0);
    ASSERT_EQ(reqs.buffers.size(), 1u);
    EXPECT_EQ(reqs.buffers[0].name, "mtp_terminal_hidden_selected_row");

    stage.setSelectedRowForReplay(3);
    const WorkspaceRequirements after = stage.getWorkspaceRequirements(8, 32, 0);
    ASSERT_EQ(after.buffers.size(), 1u);
    EXPECT_EQ(after.buffers[0].name, "mtp_terminal_hidden_selected_row");
}

TEST(Test__HiddenStateRowSelectStage, LMHeadUsesScratchOffsetZeroWhenSelectedRowChanges)
{
    const int bucket_seq_len = 7;
    const int d_model = 16;
    const int vocab_size = 32;
    auto hidden = makeHiddenStates(bucket_seq_len, d_model);
    auto scratch = std::make_unique<FP32Tensor>(
        std::vector<size_t>{1, static_cast<size_t>(d_model)},
        DeviceId::cpu());
    auto weights = makeLmHeadWeights(vocab_size, d_model);
    auto logits = std::make_unique<FP32Tensor>(
        std::vector<size_t>{1, static_cast<size_t>(vocab_size)},
        DeviceId::cpu());
    auto prepared_lm_head = makePreparedGemmFixture(weights.get(), DeviceId::cpu(), "output.weight");

    HiddenStateRowSelectStage::Params row_params;
    row_params.device_id = DeviceId::cpu();
    row_params.input = hidden.get();
    row_params.output = scratch.get();
    row_params.seq_len = bucket_seq_len;
    row_params.d_model = d_model;
    HiddenStateRowSelectStage row_select(row_params);

    LMHeadStage::Params lm_params;
    lm_params.device_id = DeviceId::cpu();
    lm_params.hidden_states = scratch.get();
    lm_params.lm_head_weight = weights.get();
    lm_params.logits = logits.get();
    lm_params.seq_len = 1;
    lm_params.d_model = d_model;
    lm_params.vocab_size = vocab_size;
    lm_params.prepared_ref = prepared_lm_head.ref;
    lm_params.prepared_store = prepared_lm_head.store.get();
    lm_params.use_prefill_replay_row_offset = false;
    LMHeadStage lm_head(lm_params);

    row_select.updatePrefillReplayParams(IComputeStage::PrefillReplayParams{3, bucket_seq_len, 0});
    ASSERT_TRUE(row_select.execute(nullptr));
    ASSERT_EQ(lm_head.activationRowOffsetForLogits(), 0);
    ASSERT_FALSE(lm_head.hasPrefillReplayParams());
    ASSERT_TRUE(lm_head.execute(nullptr));

    const auto first_reference = computeReferenceLogits(*hidden, *weights, 2, vocab_size, d_model);
    expectLogitsNear(*logits, first_reference);
    const std::vector<float> first_logits(logits->data(), logits->data() + vocab_size);

    row_select.updatePrefillReplayParams(IComputeStage::PrefillReplayParams{6, bucket_seq_len, 0});
    ASSERT_TRUE(row_select.execute(nullptr));
    ASSERT_EQ(lm_head.activationRowOffsetForLogits(), 0);
    ASSERT_TRUE(lm_head.execute(nullptr));

    const auto second_reference = computeReferenceLogits(*hidden, *weights, 5, vocab_size, d_model);
    expectLogitsNear(*logits, second_reference);
    EXPECT_GT(maxDifference(*logits, first_logits), 1e-3f);
}

TEST(Test__HiddenStateRowSelectStage, CPUMultiRowSelectPacksRowsInRequestedOrder)
{
    const int seq_len = 7;
    const int d_model = 12;
    const std::vector<int> selected_rows{5, 1, 3};
    auto hidden = makeHiddenStates(seq_len, d_model);
    auto scratch = std::make_unique<FP32Tensor>(
        std::vector<size_t>{selected_rows.size(), static_cast<size_t>(d_model)},
        DeviceId::cpu());

    HiddenStateRowsSelectStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = hidden.get();
    params.output = scratch.get();
    params.seq_len = seq_len;
    params.d_model = d_model;
    params.selected_row_count = static_cast<int>(selected_rows.size());
    params.selected_row_indices = selected_rows;
    HiddenStateRowsSelectStage stage(params);

    ASSERT_TRUE(stage.execute(nullptr));
    for (size_t output_row = 0; output_row < selected_rows.size(); ++output_row)
    {
        const int source_row = selected_rows[output_row];
        for (int column = 0; column < d_model; ++column)
        {
            EXPECT_FLOAT_EQ(
                scratch->data()[output_row * static_cast<size_t>(d_model) + static_cast<size_t>(column)],
                hidden->data()[static_cast<size_t>(source_row) * static_cast<size_t>(d_model) + static_cast<size_t>(column)])
                << "output_row=" << output_row << " column=" << column;
        }
    }
}

TEST(Test__HiddenStateRowSelectStage, BatchedLMHeadFromMultiRowScratchMatchesSerialRows)
{
    const int seq_len = 8;
    const int d_model = 24;
    const int vocab_size = 40;
    const std::vector<int> selected_rows{6, 2, 7};
    auto hidden = makeHiddenStates(seq_len, d_model);
    auto scratch = std::make_unique<FP32Tensor>(
        std::vector<size_t>{selected_rows.size(), static_cast<size_t>(d_model)},
        DeviceId::cpu());
    auto weights = makeLmHeadWeights(vocab_size, d_model);
    auto logits = std::make_unique<FP32Tensor>(
        std::vector<size_t>{selected_rows.size(), static_cast<size_t>(vocab_size)},
        DeviceId::cpu());
    auto prepared_lm_head = makePreparedGemmFixture(weights.get(), DeviceId::cpu(), "output.weight");

    HiddenStateRowsSelectStage::Params row_params;
    row_params.device_id = DeviceId::cpu();
    row_params.input = hidden.get();
    row_params.output = scratch.get();
    row_params.seq_len = seq_len;
    row_params.d_model = d_model;
    row_params.selected_row_count = static_cast<int>(selected_rows.size());
    row_params.selected_row_indices = selected_rows;
    HiddenStateRowsSelectStage row_select(row_params);
    ASSERT_TRUE(row_select.execute(nullptr));

    LMHeadStage::Params lm_params;
    lm_params.device_id = DeviceId::cpu();
    lm_params.hidden_states = scratch.get();
    lm_params.lm_head_weight = weights.get();
    lm_params.logits = logits.get();
    lm_params.seq_len = static_cast<int>(selected_rows.size());
    lm_params.d_model = d_model;
    lm_params.vocab_size = vocab_size;
    lm_params.compute_all_positions = true;
    lm_params.use_prefill_replay_row_offset = false;
    lm_params.prepared_ref = prepared_lm_head.ref;
    lm_params.prepared_store = prepared_lm_head.store.get();
    LMHeadStage lm_head(lm_params);

    ASSERT_EQ(lm_head.activationRowOffsetForLogits(), static_cast<int>(selected_rows.size()) - 1);
    ASSERT_TRUE(lm_head.execute(nullptr));

    for (size_t output_row = 0; output_row < selected_rows.size(); ++output_row)
    {
        const auto reference =
            computeReferenceLogits(*hidden, *weights, selected_rows[output_row], vocab_size, d_model);
        expectLogitsRowNear(*logits, static_cast<int>(output_row), vocab_size, reference);
    }
}

TEST(Test__HiddenStateRowSelectStage, MultiRowReplayUpdateKeepsDeclaredWorkspaceNameStable)
{
    HiddenStateRowsSelectStage::Params params;
    params.device_id = DeviceId::rocm(0);
    params.seq_len = 8;
    params.d_model = 32;
    params.selected_row_count = 3;
    params.selected_row_indices = {1, 2, 3};
    HiddenStateRowsSelectStage stage(params);

    const WorkspaceRequirements before = stage.getWorkspaceRequirements(8, 32, 0);
    ASSERT_EQ(before.buffers.size(), 1u);
    EXPECT_NE(before.buffers[0].name.find(HiddenStateRowsSelectStage::WS_SELECTED_ROWS_ARRAY), std::string::npos);
    EXPECT_GE(before.buffers[0].size_bytes, 3u * sizeof(int));

    ASSERT_TRUE(stage.setSelectedRowsForReplay({5, 0, 7}));
    EXPECT_EQ(stage.selectedRowsForTesting(), std::vector<int>({5, 0, 7}));
    const WorkspaceRequirements after = stage.getWorkspaceRequirements(8, 32, 0);
    ASSERT_EQ(after.buffers.size(), 1u);
    EXPECT_EQ(before.buffers[0].name, after.buffers[0].name);

    EXPECT_FALSE(stage.setSelectedRowsForReplay({1, 2}));
    EXPECT_EQ(stage.selectedRowsForTesting(), std::vector<int>({5, 0, 7}));
}

TEST(Test__HiddenStateRowSelectStage, ExternalRowMetadataDoesNotDeclareOrMutateWorkspace)
{
    HiddenStateRowsSelectStage::Params params;
    params.device_id = DeviceId::cuda(0);
    params.seq_len = 8;
    params.d_model = 32;
    params.selected_row_count = 3;
    params.selected_row_indices = {1, 2, 3};
    params.workspace_buffer_name = "mtp_spec_decode_verifier_rows";
    params.declare_selected_rows_workspace = false;
    params.upload_selected_rows_to_workspace = false;
    HiddenStateRowsSelectStage stage(params);

    EXPECT_TRUE(stage.getWorkspaceRequirements(8, 32, 0).buffers.empty())
        << "External metadata mode must let the metadata owner declare the row-index buffer";
    EXPECT_FALSE(stage.setSelectedRowsForReplay({5, 0, 7}))
        << "External metadata mode must not silently update a stale stage-local row list";
    EXPECT_EQ(stage.selectedRowsForTesting(), std::vector<int>({1, 2, 3}));
}
