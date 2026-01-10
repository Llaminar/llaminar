/**
 * @file Test__StageCoherence.cpp
 * @brief Unit tests for Stage Coherence System
 * @author David Sanftenberg
 * @date January 2026
 *
 * Tests the coherence infrastructure for automatic device synchronization
 * at stage boundaries. Verifies:
 * - CoherencePolicy enum and toString conversion
 * - cohereInputs/markOutputsDirty null handling
 * - Buffer extraction from StageDumpInfo
 * - Stage coherence policy reporting
 */

#include <gtest/gtest.h>
#include "execution/StageCoherence.h"
#include "execution/compute_stages/IComputeStage.h"
#include "execution/compute_stages/stages/GEMMStage.h"
#include "execution/compute_stages/stages/AllreduceStage.h"
#include "execution/compute_stages/stages/AttentionComputeStage.h"
#include "tensors/cpu/CPUTensors.h"
#include "backends/DeviceId.h"

using namespace llaminar2;

// =============================================================================
// CoherencePolicy Tests
// =============================================================================

TEST(Test__StageCoherence, CoherencePolicy_ToString)
{
    // Verify all CoherencePolicy enum values convert to expected strings
    EXPECT_STREQ(toString(CoherencePolicy::NONE), "NONE");
    EXPECT_STREQ(toString(CoherencePolicy::INPUT), "INPUT");
    EXPECT_STREQ(toString(CoherencePolicy::OUTPUT), "OUTPUT");
    EXPECT_STREQ(toString(CoherencePolicy::FULL), "FULL");
}

TEST(Test__StageCoherence, CoherencePolicy_ToString_AllDistinct)
{
    // Verify all policies have distinct string representations
    const char *none_str = toString(CoherencePolicy::NONE);
    const char *input_str = toString(CoherencePolicy::INPUT);
    const char *output_str = toString(CoherencePolicy::OUTPUT);
    const char *full_str = toString(CoherencePolicy::FULL);

    EXPECT_STRNE(none_str, input_str);
    EXPECT_STRNE(none_str, output_str);
    EXPECT_STRNE(none_str, full_str);
    EXPECT_STRNE(input_str, output_str);
    EXPECT_STRNE(input_str, full_str);
    EXPECT_STRNE(output_str, full_str);
}

// =============================================================================
// cohereInputs Tests
// =============================================================================

TEST(Test__StageCoherence, CohereInputs_NullTensor_Skips)
{
    // cohereInputs should handle null tensor pointers gracefully
    std::vector<CoherenceBuffer> inputs;
    inputs.push_back({nullptr, "test_input", nullptr, 0, 0, nullptr, false});

    // Should not crash and should return true (all tensors processed)
    EXPECT_TRUE(cohereInputs(inputs, DeviceId::cpu()));
}

TEST(Test__StageCoherence, CohereInputs_EmptyVector_Succeeds)
{
    // Empty input vector should return true immediately
    std::vector<CoherenceBuffer> inputs;
    EXPECT_TRUE(cohereInputs(inputs, DeviceId::cpu()));
}

TEST(Test__StageCoherence, CohereInputs_MultipleNullTensors_Succeeds)
{
    // Multiple null tensors should all be skipped gracefully
    std::vector<CoherenceBuffer> inputs;
    inputs.push_back({nullptr, "input1", nullptr, 10, 10, "FP32", false});
    inputs.push_back({nullptr, "input2", nullptr, 20, 20, "FP32", false});
    inputs.push_back({nullptr, "input3", nullptr, 30, 30, "FP32", false});

    EXPECT_TRUE(cohereInputs(inputs, DeviceId::cpu()));
}

TEST(Test__StageCoherence, CohereInputs_ValidTensor_Succeeds)
{
    // Create a real FP32 tensor
    auto tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{10, 10}, DeviceId::cpu());

    std::vector<CoherenceBuffer> inputs;
    CoherenceBuffer buf;
    buf.tensor = tensor.get();
    buf.name = "test_tensor";
    buf.rows = 10;
    buf.cols = 10;
    buf.dtype = "FP32";
    buf.is_inout = false;
    inputs.push_back(buf);

    // Should succeed for CPU tensor targeting CPU device
    EXPECT_TRUE(cohereInputs(inputs, DeviceId::cpu()));
}

// =============================================================================
// markOutputsDirty Tests
// =============================================================================

TEST(Test__StageCoherence, MarkOutputsDirty_NullTensor_Skips)
{
    // markOutputsDirty should handle null tensor pointers gracefully
    std::vector<CoherenceBuffer> outputs;
    outputs.push_back({nullptr, "test_output", nullptr, 0, 0, nullptr, false});

    // Should not crash (function returns void)
    EXPECT_NO_THROW(markOutputsDirty(outputs));
}

TEST(Test__StageCoherence, MarkOutputsDirty_EmptyVector_Succeeds)
{
    // Empty output vector should return immediately without crash
    std::vector<CoherenceBuffer> outputs;
    EXPECT_NO_THROW(markOutputsDirty(outputs));
}

TEST(Test__StageCoherence, MarkOutputsDirty_MultipleNullTensors_Succeeds)
{
    // Multiple null tensors should all be skipped gracefully
    std::vector<CoherenceBuffer> outputs;
    outputs.push_back({nullptr, "output1", nullptr, 10, 10, "FP32", false});
    outputs.push_back({nullptr, "output2", nullptr, 20, 20, "FP32", false});

    EXPECT_NO_THROW(markOutputsDirty(outputs));
}

TEST(Test__StageCoherence, MarkOutputsDirty_ValidTensor_Succeeds)
{
    // Create a real FP32 tensor
    auto tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{10, 10}, DeviceId::cpu());

    std::vector<CoherenceBuffer> outputs;
    CoherenceBuffer buf;
    buf.tensor = tensor.get();
    buf.name = "test_tensor";
    buf.rows = 10;
    buf.cols = 10;
    buf.dtype = "FP32";
    buf.is_inout = false;
    outputs.push_back(buf);

    // Should succeed without crashing
    EXPECT_NO_THROW(markOutputsDirty(outputs));
}

// =============================================================================
// Buffer Extraction Tests
// =============================================================================

TEST(Test__StageCoherence, ExtractInputBuffers_FromDumpInfo)
{
    // Create a test tensor
    auto tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{10, 10}, DeviceId::cpu());

    // Build StageDumpInfo with tensor pointer
    StageDumpInfo info;
    auto &input = info.inputs.emplace_back();
    input.name = "test_input";
    input.data = tensor->data();
    input.rows = 10;
    input.cols = 10;
    input.dtype = "FP32";
    input.tensor = tensor.get();

    // Extract and verify
    auto buffers = extractInputBuffers(info);
    ASSERT_EQ(buffers.size(), 1);
    EXPECT_EQ(buffers[0].tensor, tensor.get());
    EXPECT_STREQ(buffers[0].name, "test_input");
    EXPECT_EQ(buffers[0].rows, 10);
    EXPECT_EQ(buffers[0].cols, 10);
    EXPECT_STREQ(buffers[0].dtype, "FP32");
    EXPECT_FALSE(buffers[0].is_inout);
}

TEST(Test__StageCoherence, ExtractOutputBuffers_FromDumpInfo)
{
    // Create a test tensor
    auto tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{20, 30}, DeviceId::cpu());

    // Build StageDumpInfo with output tensor
    StageDumpInfo info;
    auto &output = info.outputs.emplace_back();
    output.name = "test_output";
    output.data = tensor->data();
    output.rows = 20;
    output.cols = 30;
    output.dtype = "FP32";
    output.tensor = tensor.get();

    // Extract and verify
    auto buffers = extractOutputBuffers(info);
    ASSERT_EQ(buffers.size(), 1);
    EXPECT_EQ(buffers[0].tensor, tensor.get());
    EXPECT_STREQ(buffers[0].name, "test_output");
    EXPECT_EQ(buffers[0].rows, 20);
    EXPECT_EQ(buffers[0].cols, 30);
    EXPECT_STREQ(buffers[0].dtype, "FP32");
    EXPECT_FALSE(buffers[0].is_inout);
}

TEST(Test__StageCoherence, ExtractInputBuffers_EmptyDumpInfo)
{
    // Empty StageDumpInfo should yield empty buffer list
    StageDumpInfo info;
    auto buffers = extractInputBuffers(info);
    EXPECT_TRUE(buffers.empty());
}

TEST(Test__StageCoherence, ExtractOutputBuffers_EmptyDumpInfo)
{
    // Empty StageDumpInfo should yield empty buffer list
    StageDumpInfo info;
    auto buffers = extractOutputBuffers(info);
    EXPECT_TRUE(buffers.empty());
}

TEST(Test__StageCoherence, ExtractInputBuffers_MultipleInputs)
{
    // Create test tensors
    auto tensor1 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{10, 10}, DeviceId::cpu());
    auto tensor2 = std::make_unique<FP32Tensor>(
        std::vector<size_t>{20, 20}, DeviceId::cpu());

    // Build StageDumpInfo with multiple inputs
    StageDumpInfo info;
    {
        auto &input = info.inputs.emplace_back();
        input.name = "A";
        input.data = tensor1->data();
        input.rows = 10;
        input.cols = 10;
        input.dtype = "FP32";
        input.tensor = tensor1.get();
    }
    {
        auto &input = info.inputs.emplace_back();
        input.name = "B";
        input.data = tensor2->data();
        input.rows = 20;
        input.cols = 20;
        input.dtype = "FP32";
        input.tensor = tensor2.get();
    }

    // Extract and verify
    auto buffers = extractInputBuffers(info);
    ASSERT_EQ(buffers.size(), 2);
    EXPECT_STREQ(buffers[0].name, "A");
    EXPECT_STREQ(buffers[1].name, "B");
    EXPECT_EQ(buffers[0].tensor, tensor1.get());
    EXPECT_EQ(buffers[1].tensor, tensor2.get());
}

TEST(Test__StageCoherence, ExtractInputBuffers_NullTensorPointer)
{
    // StageDumpInfo may have null tensor pointers (legacy compatibility)
    StageDumpInfo info;
    auto &input = info.inputs.emplace_back();
    input.name = "legacy_input";
    input.data = nullptr;
    input.rows = 10;
    input.cols = 10;
    input.dtype = "FP32";
    input.tensor = nullptr; // No tensor pointer

    // Should still extract the buffer (tensor will be null)
    auto buffers = extractInputBuffers(info);
    ASSERT_EQ(buffers.size(), 1);
    EXPECT_EQ(buffers[0].tensor, nullptr);
    EXPECT_STREQ(buffers[0].name, "legacy_input");
}

// =============================================================================
// Stage Coherence Policy Tests
// =============================================================================

TEST(Test__StageCoherence, GEMMStage_HasFullCoherencePolicy)
{
    // Create minimal tensors for GEMMStage
    auto A = std::make_unique<FP32Tensor>(
        std::vector<size_t>{4, 8}, DeviceId::cpu());
    auto B = std::make_unique<FP32Tensor>(
        std::vector<size_t>{8, 16}, DeviceId::cpu());
    auto C = std::make_unique<FP32Tensor>(
        std::vector<size_t>{4, 16}, DeviceId::cpu());

    // Create GEMM stage
    GEMMStage::Params params;
    params.A = A.get();
    params.B = B.get();
    params.C = C.get();
    params.m = 4;
    params.n = 16;
    params.k = 8;
    params.alpha = 1.0f;
    params.beta = 0.0f;

    GEMMStage stage(params);

    // GEMMStage should use FULL coherence (default for compute stages)
    EXPECT_EQ(stage.coherencePolicy(), CoherencePolicy::FULL);
}

TEST(Test__StageCoherence, AllreduceStage_HasNoneCoherencePolicy)
{
    // AllreduceStage manages its own MPI synchronization
    // and should NOT have automatic coherence

    // Create minimal tensor for AllreduceStage
    auto buffer = std::make_unique<FP32Tensor>(
        std::vector<size_t>{100}, DeviceId::cpu());

    // Create AllreduceStage (mpi_ctx is null for unit test - that's OK)
    AllreduceStage::Params params;
    params.buffer = buffer.get();
    params.mpi_ctx = nullptr; // No MPI context for unit test
    params.count = 100;

    AllreduceStage stage(params);

    // AllreduceStage should return CoherencePolicy::NONE
    EXPECT_EQ(stage.coherencePolicy(), CoherencePolicy::NONE);
}

TEST(Test__StageCoherence, AttentionComputeStage_PreferredDevice)
{
    // Create minimal tensors for AttentionComputeStage
    const int batch = 1;
    const int seq_len = 4;
    const int n_heads = 2;
    const int head_dim = 8;

    auto Q = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(batch * seq_len * n_heads), static_cast<size_t>(head_dim)},
        DeviceId::cpu());
    auto K = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(batch * seq_len * n_heads), static_cast<size_t>(head_dim)},
        DeviceId::cpu());
    auto V = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(batch * seq_len * n_heads), static_cast<size_t>(head_dim)},
        DeviceId::cpu());
    auto output = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(batch * seq_len * n_heads), static_cast<size_t>(head_dim)},
        DeviceId::cpu());

    // Create AttentionComputeStage with explicit device
    AttentionComputeStage::Params params;
    params.Q = Q.get();
    params.K = K.get();
    params.V = V.get();
    params.output = output.get();
    params.batch_size = batch;
    params.seq_len = seq_len;
    params.kv_len = seq_len;
    params.n_heads = n_heads;
    params.n_kv_heads = n_heads;
    params.head_dim = head_dim;
    params.device_id = DeviceId::cpu();

    AttentionComputeStage stage(params);

    // Should report CPU as preferred device
    EXPECT_EQ(stage.preferredDevice(), DeviceId::cpu());
}

TEST(Test__StageCoherence, GEMMStage_PreferredDevice_CPU)
{
    // GEMMStage should report its configured device
    auto A = std::make_unique<FP32Tensor>(
        std::vector<size_t>{4, 8}, DeviceId::cpu());
    auto B = std::make_unique<FP32Tensor>(
        std::vector<size_t>{8, 16}, DeviceId::cpu());
    auto C = std::make_unique<FP32Tensor>(
        std::vector<size_t>{4, 16}, DeviceId::cpu());

    GEMMStage::Params params;
    params.A = A.get();
    params.B = B.get();
    params.C = C.get();
    params.m = 4;
    params.n = 16;
    params.k = 8;
    params.device_id = DeviceId::cpu();

    GEMMStage stage(params);

    EXPECT_EQ(stage.preferredDevice(), DeviceId::cpu());
}

// =============================================================================
// IComputeStage Default Coherence Policy Test
// =============================================================================

TEST(Test__StageCoherence, IComputeStage_DefaultCoherencePolicy_IsFull)
{
    // The base IComputeStage class should default to FULL coherence policy
    // We test this via a concrete stage that doesn't override coherencePolicy()

    auto A = std::make_unique<FP32Tensor>(
        std::vector<size_t>{4, 8}, DeviceId::cpu());
    auto B = std::make_unique<FP32Tensor>(
        std::vector<size_t>{8, 16}, DeviceId::cpu());
    auto C = std::make_unique<FP32Tensor>(
        std::vector<size_t>{4, 16}, DeviceId::cpu());

    GEMMStage::Params params;
    params.A = A.get();
    params.B = B.get();
    params.C = C.get();
    params.m = 4;
    params.n = 16;
    params.k = 8;

    GEMMStage stage(params);

    // GEMMStage doesn't override coherencePolicy(), so it should be FULL
    IComputeStage *base_ptr = &stage;
    EXPECT_EQ(base_ptr->coherencePolicy(), CoherencePolicy::FULL);
}

TEST(Test__StageCoherence, IComputeStage_DefaultPreferredDevice_IsCPU)
{
    // Default preferredDevice() should return CPU

    auto A = std::make_unique<FP32Tensor>(
        std::vector<size_t>{4, 8}, DeviceId::cpu());
    auto B = std::make_unique<FP32Tensor>(
        std::vector<size_t>{8, 16}, DeviceId::cpu());
    auto C = std::make_unique<FP32Tensor>(
        std::vector<size_t>{4, 16}, DeviceId::cpu());

    GEMMStage::Params params;
    params.A = A.get();
    params.B = B.get();
    params.C = C.get();
    params.m = 4;
    params.n = 16;
    params.k = 8;
    // Don't set device_id, use default

    GEMMStage stage(params);

    // Default device_id in Params is cpu()
    EXPECT_EQ(stage.preferredDevice(), DeviceId::cpu());
}
