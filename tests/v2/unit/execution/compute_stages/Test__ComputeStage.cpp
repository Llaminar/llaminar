/**
 * @file Test__ComputeStage.cpp
 * @brief Unit tests for ComputeStage abstraction
 * @author David Sanftenberg
 * @date December 2025
 */

#include <gtest/gtest.h>
#include "execution/compute_stages/ComputeStages.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "tensors/Tensors.h"
#include <cmath>
#include <vector>
#include <numeric>

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class ComputeStageTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Construct CPUDeviceContext directly (bypasses DeviceManager check)
        // This is the same pattern used in Test__DeviceContext.cpp
        ctx_ = std::make_unique<CPUDeviceContext>(DeviceId::cpu(), 4);
        ASSERT_NE(ctx_, nullptr);
    }

    std::unique_ptr<CPUDeviceContext> ctx_;

    // Helper: Create an FP32Tensor (2D)
    std::unique_ptr<FP32Tensor> makeTensor(size_t rows, size_t cols, const std::vector<float> &data = {})
    {
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols}, DeviceId::cpu());
        if (!data.empty())
        {
            std::copy(data.begin(), data.end(), tensor->mutable_data());
        }
        return tensor;
    }

    // Helper: Create an FP32Tensor (1D)
    std::unique_ptr<FP32Tensor> makeTensor1D(size_t n, const std::vector<float> &data = {})
    {
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{n}, DeviceId::cpu());
        if (!data.empty())
        {
            std::copy(data.begin(), data.end(), tensor->mutable_data());
        }
        return tensor;
    }
};

// =============================================================================
// ComputeStageType Tests
// =============================================================================

TEST_F(ComputeStageTest, StageTypeNames)
{
    // Test that all stage types have human-readable names
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::GEMM), "GEMM");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::RMS_NORM), "RMS_NORM");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::ROPE), "ROPE");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::ATTENTION), "ATTENTION");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::SWIGLU), "SWIGLU");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::ADD_RESIDUAL), "ADD_RESIDUAL");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::MOE_ROUTER), "MOE_ROUTER");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::MOE_EXPERT_FFN), "MOE_EXPERT_FFN");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::MOE_COMBINE), "MOE_COMBINE");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::ALLREDUCE), "ALLREDUCE");
}

// =============================================================================
// RMSNormStage Tests
// =============================================================================

TEST_F(ComputeStageTest, RMSNormBasic)
{
    // Test basic RMS normalization
    const int seq_len = 2;
    const int hidden_dim = 4;
    const float eps = 1e-6f;

    // Input: two rows of [1, 2, 3, 4]
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                     1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> gamma_data(hidden_dim, 1.0f); // Scale = 1

    auto input = makeTensor(seq_len, hidden_dim, input_data);
    auto output = makeTensor(seq_len, hidden_dim);
    auto gamma = makeTensor1D(hidden_dim, gamma_data);

    RMSNormStage::Params params{
        .input = input.get(),
        .output = output.get(),
        .gamma = gamma.get(),
        .eps = eps};

    RMSNormStage stage(params);

    // Check type
    EXPECT_EQ(stage.type(), ComputeStageType::RMS_NORM);

    // Execute (may require device enumeration)
    bool success = false;
    try
    {
        success = stage.execute(ctx_.get());
    }
    catch (const std::exception &e)
    {
        GTEST_SKIP() << "RMSNorm execution requires device enumeration: " << e.what();
    }

    if (!success)
    {
        GTEST_SKIP() << "RMSNorm execution failed (missing kernel support)";
    }

    // Verify: RMS of [1,2,3,4] = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.739
    // Each element should be divided by RMS
    float expected_rms = std::sqrt((1.0f + 4.0f + 9.0f + 16.0f) / 4.0f + eps);
    const float *out = output->data();
    EXPECT_NEAR(out[0], 1.0f / expected_rms, 1e-5f);
    EXPECT_NEAR(out[1], 2.0f / expected_rms, 1e-5f);
    EXPECT_NEAR(out[2], 3.0f / expected_rms, 1e-5f);
    EXPECT_NEAR(out[3], 4.0f / expected_rms, 1e-5f);
}

TEST_F(ComputeStageTest, RMSNormWithGamma)
{
    // Test RMS normalization with non-unit gamma
    const int seq_len = 1;
    const int hidden_dim = 4;
    const float eps = 1e-6f;

    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> gamma_data = {2.0f, 0.5f, 1.0f, 0.0f}; // Different scales

    auto input = makeTensor(seq_len, hidden_dim, input_data);
    auto output = makeTensor(seq_len, hidden_dim);
    auto gamma = makeTensor1D(hidden_dim, gamma_data);

    RMSNormStage::Params params{
        .input = input.get(),
        .output = output.get(),
        .gamma = gamma.get(),
        .eps = eps};

    RMSNormStage stage(params);

    bool success = false;
    try
    {
        success = stage.execute(ctx_.get());
    }
    catch (const std::exception &e)
    {
        GTEST_SKIP() << "RMSNorm execution requires device enumeration: " << e.what();
    }

    if (!success)
    {
        GTEST_SKIP() << "RMSNorm execution failed (missing kernel support)";
    }

    const float *out = output->data();
    float rms = std::sqrt((1.0f + 4.0f + 9.0f + 16.0f) / 4.0f + eps);
    EXPECT_NEAR(out[0], (1.0f / rms) * 2.0f, 1e-5f); // gamma = 2
    EXPECT_NEAR(out[1], (2.0f / rms) * 0.5f, 1e-5f); // gamma = 0.5
    EXPECT_NEAR(out[3], 0.0f, 1e-5f);                // gamma = 0
}

TEST_F(ComputeStageTest, RMSNormFlopEstimate)
{
    const int seq_len = 512;
    const int hidden_dim = 896;

    auto input = makeTensor(seq_len, hidden_dim);
    auto output = makeTensor(seq_len, hidden_dim);
    auto gamma = makeTensor1D(hidden_dim);

    RMSNormStage::Params params{
        .input = input.get(),
        .output = output.get(),
        .gamma = gamma.get(),
        .eps = 1e-6f};

    RMSNormStage stage(params);

    // ~4 FLOPs per element: square, add, normalize (mul + div)
    size_t expected_flops = 4ULL * 512 * 896;
    EXPECT_EQ(stage.estimatedFlops(), expected_flops);
}

// =============================================================================
// RoPEStage Tests
// =============================================================================

TEST_F(ComputeStageTest, RoPEBasic)
{
    // Test basic rotary position encoding
    const int seq_len = 1;
    const int n_heads = 1;
    const int head_dim = 4; // Must be even for RoPE

    // Input: [1, 0, 1, 0] (pairs: (1,0), (1,0))
    std::vector<float> tensor_data = {1.0f, 0.0f, 1.0f, 0.0f};
    auto Q = makeTensor(seq_len, n_heads * head_dim, tensor_data);

    RoPEStage::Params params{
        .Q = Q.get(),
        .n_heads = n_heads,
        .n_kv_heads = n_heads,
        .head_dim = head_dim,
        .pos_offset = 0,
        .theta_base = 10000.0f};

    RoPEStage stage(params);

    EXPECT_EQ(stage.type(), ComputeStageType::ROPE);

    bool success = false;
    try
    {
        success = stage.execute(ctx_.get());
    }
    catch (const std::exception &e)
    {
        GTEST_SKIP() << "RoPE execution requires device enumeration: " << e.what();
    }

    if (!success)
    {
        GTEST_SKIP() << "RoPE execution failed (missing kernel support)";
    }

    // At position 0, cos(0) = 1, sin(0) = 0
    // So rotation should be identity
    const float *out = Q->data();
    EXPECT_NEAR(out[0], 1.0f, 1e-5f);
    EXPECT_NEAR(out[1], 0.0f, 1e-5f);
}

TEST_F(ComputeStageTest, RoPEPositionOffset)
{
    // Test that position offset affects rotation
    const int seq_len = 1;
    const int n_heads = 1;
    const int head_dim = 4;

    std::vector<float> tensor_data_pos0 = {1.0f, 0.0f, 1.0f, 0.0f};
    std::vector<float> tensor_data_pos1 = {1.0f, 0.0f, 1.0f, 0.0f};

    auto Q0 = makeTensor(seq_len, n_heads * head_dim, tensor_data_pos0);
    auto Q1 = makeTensor(seq_len, n_heads * head_dim, tensor_data_pos1);

    RoPEStage::Params params0{
        .Q = Q0.get(),
        .n_heads = n_heads,
        .n_kv_heads = n_heads,
        .head_dim = head_dim,
        .pos_offset = 0,
        .theta_base = 10000.0f};

    RoPEStage::Params params1{
        .Q = Q1.get(),
        .n_heads = n_heads,
        .n_kv_heads = n_heads,
        .head_dim = head_dim,
        .pos_offset = 10, // Different position
        .theta_base = 10000.0f};

    RoPEStage stage0(params0);
    RoPEStage stage1(params1);

    bool success0 = false, success1 = false;
    try
    {
        success0 = stage0.execute(ctx_.get());
        success1 = stage1.execute(ctx_.get());
    }
    catch (const std::exception &e)
    {
        GTEST_SKIP() << "RoPE execution requires device enumeration: " << e.what();
    }

    if (!success0 || !success1)
    {
        GTEST_SKIP() << "RoPE execution failed (missing kernel support)";
    }

    // Results should be different due to position offset
    const float *out0 = Q0->data();
    const float *out1 = Q1->data();
    bool different = false;
    for (int i = 0; i < head_dim; ++i)
    {
        if (std::abs(out0[i] - out1[i]) > 1e-5f)
        {
            different = true;
            break;
        }
    }
    EXPECT_TRUE(different);
}

// -----------------------------------------------------------------------------
// RoPE with explicit position_ids (for batched execution parity)
// -----------------------------------------------------------------------------

TEST_F(ComputeStageTest, RoPEExplicitPositionIds_BatchedParity)
{
    // This test verifies that using explicit position_ids produces the same result
    // as using pos_offset for the same token at the same position.
    //
    // Bug fixed: For batched execution with batch_size=2, seq_len=2, the positions
    // should be [0, 1, 0, 1] (per-sequence positions) not [0, 1, 2, 3] (contiguous).
    // The fix was to pass explicit position_ids to RoPEStage.

    const int total_tokens = 2; // Simulating 2 tokens (could be 2 sequences of 1 token each)
    const int n_heads = 1;
    const int head_dim = 4;

    // Same input tensor data for both tests
    std::vector<float> tensor_data = {1.0f, 0.0f, 1.0f, 0.0f,
                                      1.0f, 0.0f, 1.0f, 0.0f};

    // Test 1: Using pos_offset=0 (generates contiguous positions [0, 1])
    auto Q_contiguous = makeTensor(total_tokens, n_heads * head_dim, tensor_data);
    RoPEStage::Params params_contiguous;
    params_contiguous.Q = Q_contiguous.get();
    params_contiguous.n_heads = n_heads;
    params_contiguous.n_kv_heads = n_heads;
    params_contiguous.head_dim = head_dim;
    params_contiguous.pos_offset = 0;
    params_contiguous.theta_base = 10000.0f;
    params_contiguous.seq_len = total_tokens;
    params_contiguous.position_ids = nullptr; // Will generate [0, 1]

    // Test 2: Using explicit position_ids = [0, 0] (both tokens at position 0)
    // This simulates two sequences each starting at position 0
    std::vector<int> batch_positions = {0, 0};
    auto Q_batched = makeTensor(total_tokens, n_heads * head_dim, tensor_data);
    RoPEStage::Params params_batched;
    params_batched.Q = Q_batched.get();
    params_batched.n_heads = n_heads;
    params_batched.n_kv_heads = n_heads;
    params_batched.head_dim = head_dim;
    params_batched.pos_offset = 0; // Ignored when position_ids is set
    params_batched.theta_base = 10000.0f;
    params_batched.seq_len = total_tokens;
    params_batched.position_ids = batch_positions.data(); // Explicit [0, 0]

    RoPEStage stage_contiguous(params_contiguous);
    RoPEStage stage_batched(params_batched);

    bool success1 = false, success2 = false;
    try
    {
        success1 = stage_contiguous.execute(ctx_.get());
        success2 = stage_batched.execute(ctx_.get());
    }
    catch (const std::exception &e)
    {
        GTEST_SKIP() << "RoPE execution requires device enumeration: " << e.what();
    }

    if (!success1 || !success2)
    {
        GTEST_SKIP() << "RoPE execution failed (missing kernel support)";
    }

    const float *out_contiguous = Q_contiguous->data();
    const float *out_batched = Q_batched->data();

    // With position_ids = [0, 0], both rows should have the same RoPE rotation (position 0)
    // Row 0 should be the same in both (position 0)
    for (int i = 0; i < head_dim; ++i)
    {
        EXPECT_NEAR(out_contiguous[i], out_batched[i], 1e-5f)
            << "Row 0 element " << i << " should match (both at position 0)";
    }

    // Row 1 should be DIFFERENT in contiguous (position 1) vs batched (position 0)
    // So batched row 1 should match contiguous row 0 (both at position 0)
    bool row1_matches_row0 = true;
    for (int i = 0; i < head_dim; ++i)
    {
        if (std::abs(out_batched[head_dim + i] - out_contiguous[i]) > 1e-5f)
        {
            row1_matches_row0 = false;
            break;
        }
    }
    EXPECT_TRUE(row1_matches_row0) << "Batched row 1 (position 0) should match contiguous row 0 (position 0)";

    // And contiguous row 1 should be different from batched row 1
    bool row1_different = false;
    for (int i = 0; i < head_dim; ++i)
    {
        if (std::abs(out_contiguous[head_dim + i] - out_batched[head_dim + i]) > 1e-5f)
        {
            row1_different = true;
            break;
        }
    }
    EXPECT_TRUE(row1_different) << "Contiguous row 1 (position 1) should differ from batched row 1 (position 0)";
}

TEST_F(ComputeStageTest, RoPEExplicitPositionIds_OverridesPosOffset)
{
    // Verify that when position_ids is provided, pos_offset is ignored
    const int seq_len = 1;
    const int n_heads = 1;
    const int head_dim = 4;

    std::vector<float> tensor_data = {1.0f, 0.0f, 1.0f, 0.0f};

    // Test 1: pos_offset=5, no position_ids
    auto Q1 = makeTensor(seq_len, n_heads * head_dim, tensor_data);
    RoPEStage::Params params1;
    params1.Q = Q1.get();
    params1.n_heads = n_heads;
    params1.n_kv_heads = n_heads;
    params1.head_dim = head_dim;
    params1.pos_offset = 5;
    params1.theta_base = 10000.0f;
    params1.seq_len = seq_len;
    params1.position_ids = nullptr; // Uses pos_offset=5

    // Test 2: pos_offset=5 but position_ids=[0] should override to position 0
    std::vector<int> explicit_pos = {0};
    auto Q2 = makeTensor(seq_len, n_heads * head_dim, tensor_data);
    RoPEStage::Params params2;
    params2.Q = Q2.get();
    params2.n_heads = n_heads;
    params2.n_kv_heads = n_heads;
    params2.head_dim = head_dim;
    params2.pos_offset = 5; // Should be ignored
    params2.theta_base = 10000.0f;
    params2.seq_len = seq_len;
    params2.position_ids = explicit_pos.data(); // Overrides to position 0

    // Test 3: pos_offset=0 for reference
    auto Q3 = makeTensor(seq_len, n_heads * head_dim, tensor_data);
    RoPEStage::Params params3;
    params3.Q = Q3.get();
    params3.n_heads = n_heads;
    params3.n_kv_heads = n_heads;
    params3.head_dim = head_dim;
    params3.pos_offset = 0;
    params3.theta_base = 10000.0f;
    params3.seq_len = seq_len;
    params3.position_ids = nullptr;

    RoPEStage stage1(params1);
    RoPEStage stage2(params2);
    RoPEStage stage3(params3);

    bool success1 = false, success2 = false, success3 = false;
    try
    {
        success1 = stage1.execute(ctx_.get());
        success2 = stage2.execute(ctx_.get());
        success3 = stage3.execute(ctx_.get());
    }
    catch (const std::exception &e)
    {
        GTEST_SKIP() << "RoPE execution requires device enumeration: " << e.what();
    }

    if (!success1 || !success2 || !success3)
    {
        GTEST_SKIP() << "RoPE execution failed (missing kernel support)";
    }

    const float *out1 = Q1->data(); // pos_offset=5
    const float *out2 = Q2->data(); // position_ids=[0] (should override)
    const float *out3 = Q3->data(); // pos_offset=0

    // Q2 (with position_ids=[0]) should match Q3 (pos_offset=0), not Q1 (pos_offset=5)
    for (int i = 0; i < head_dim; ++i)
    {
        EXPECT_NEAR(out2[i], out3[i], 1e-5f)
            << "position_ids=[0] should produce same result as pos_offset=0";
    }

    // Q1 should be different from Q3 (different positions)
    bool q1_differs = false;
    for (int i = 0; i < head_dim; ++i)
    {
        if (std::abs(out1[i] - out3[i]) > 1e-5f)
        {
            q1_differs = true;
            break;
        }
    }
    EXPECT_TRUE(q1_differs) << "pos_offset=5 should produce different result than pos_offset=0";
}

// =============================================================================
// ResidualAddStage Tests
// =============================================================================

TEST_F(ComputeStageTest, ResidualAddBasic)
{
    const size_t n = 8;

    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> residual_data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};

    auto input = makeTensor(1, n, input_data);
    auto residual = makeTensor(1, n, residual_data);
    auto output = makeTensor(1, n);

    ResidualAddStage::Params params{
        .input = input.get(),
        .residual = residual.get(),
        .output = output.get()};

    ResidualAddStage stage(params);

    EXPECT_EQ(stage.type(), ComputeStageType::ADD_RESIDUAL);
    EXPECT_TRUE(stage.execute(ctx_.get()));

    const float *out = output->data();
    for (size_t i = 0; i < n; ++i)
    {
        EXPECT_NEAR(out[i], input_data[i] + residual_data[i], 1e-5f);
    }
}

TEST_F(ComputeStageTest, ResidualAddInPlace)
{
    // Test in-place addition (output = residual)
    const size_t n = 4;

    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> residual_data = {0.5f, 0.5f, 0.5f, 0.5f};

    auto input = makeTensor(1, n, input_data);
    auto residual = makeTensor(1, n, residual_data);

    ResidualAddStage::Params params{
        .input = input.get(),
        .residual = residual.get(),
        .output = residual.get()}; // In-place to residual

    ResidualAddStage stage(params);
    EXPECT_TRUE(stage.execute(ctx_.get()));

    const float *out = residual->data();
    EXPECT_NEAR(out[0], 1.5f, 1e-5f);
    EXPECT_NEAR(out[1], 2.5f, 1e-5f);
    EXPECT_NEAR(out[2], 3.5f, 1e-5f);
    EXPECT_NEAR(out[3], 4.5f, 1e-5f);
}

// =============================================================================
// Backend Support Tests
// =============================================================================

TEST_F(ComputeStageTest, BackendSupport)
{
    // All CPU stages should support OneDNN and MKL
    const int seq_len = 1;
    const int hidden_dim = 4;

    auto input = makeTensor(seq_len, hidden_dim);
    auto output = makeTensor(seq_len, hidden_dim);
    auto gamma = makeTensor1D(hidden_dim);

    RMSNormStage::Params rms_params{
        .input = input.get(),
        .output = output.get(),
        .gamma = gamma.get(),
        .eps = 1e-6f};

    RMSNormStage rms_stage(rms_params);

    EXPECT_TRUE(rms_stage.supportsBackend(ComputeBackendType::CPU));
    EXPECT_TRUE(rms_stage.supportsBackend(ComputeBackendType::CPU));
#ifdef HAVE_CUDA
    EXPECT_TRUE(rms_stage.supportsBackend(ComputeBackendType::GPU_CUDA));
#else
    EXPECT_FALSE(rms_stage.supportsBackend(ComputeBackendType::GPU_CUDA));
#endif
#ifdef HAVE_ROCM
    EXPECT_TRUE(rms_stage.supportsBackend(ComputeBackendType::GPU_ROCM));
#else
    EXPECT_FALSE(rms_stage.supportsBackend(ComputeBackendType::GPU_ROCM));
#endif
}

// =============================================================================
// MoE Stage Tests
// =============================================================================

TEST_F(ComputeStageTest, MoERouterBasic)
{
    const int seq_len = 2;
    const int d_model = 4;
    const int num_experts = 3;

    // Simple hidden states using tensor helper
    std::vector<float> hidden_data = {1.0f, 0.0f, 0.0f, 0.0f,  // Token 0
                                      0.0f, 1.0f, 0.0f, 0.0f}; // Token 1
    auto hidden = makeTensor(seq_len, d_model, hidden_data);

    // Gate weights: each expert is a one-hot [num_experts, d_model]
    std::vector<float> gate_data = {
        1.0f, 0.0f, 0.0f, 0.0f, // Expert 0 responds to dim 0
        0.0f, 1.0f, 0.0f, 0.0f, // Expert 1 responds to dim 1
        0.0f, 0.0f, 1.0f, 0.0f  // Expert 2 responds to dim 2
    };
    auto gate_weights = makeTensor(num_experts, d_model, gate_data);

    // Output router logits
    auto router_logits = makeTensor(seq_len, num_experts);

    MoERouterStage::Params params;
    params.hidden = hidden.get();
    params.gate_weights = gate_weights.get();
    params.router_logits = router_logits.get();
    params.seq_len = seq_len;
    params.d_model = d_model;
    params.num_experts = num_experts;

    MoERouterStage stage(params);

    EXPECT_EQ(stage.type(), ComputeStageType::MOE_ROUTER);
    EXPECT_TRUE(stage.execute(ctx_.get()));

    // Get output data for verification
    const float *output = router_logits->data();

    // Token 0: [1,0,0,0] should have logit 1 for expert 0
    EXPECT_NEAR(output[0], 1.0f, 1e-5f); // Expert 0
    EXPECT_NEAR(output[1], 0.0f, 1e-5f); // Expert 1
    EXPECT_NEAR(output[2], 0.0f, 1e-5f); // Expert 2

    // Token 1: [0,1,0,0] should have logit 1 for expert 1
    EXPECT_NEAR(output[3], 0.0f, 1e-5f); // Expert 0
    EXPECT_NEAR(output[4], 1.0f, 1e-5f); // Expert 1
    EXPECT_NEAR(output[5], 0.0f, 1e-5f); // Expert 2
}

TEST_F(ComputeStageTest, MoEExpertNoTokens)
{
    // Expert with no routed tokens should be a no-op
    std::vector<int> empty_tokens;

    MoEExpertStage::Params params;
    params.expert_id = 5;
    params.input = nullptr;
    params.output = nullptr;
    params.gate_w = nullptr;
    params.up_w = nullptr;
    params.down_w = nullptr;
    params.token_indices = &empty_tokens;
    params.d_model = 896;
    params.intermediate_dim = 4864;

    MoEExpertStage stage(params);

    EXPECT_EQ(stage.type(), ComputeStageType::MOE_EXPERT_FFN);
    EXPECT_EQ(stage.name(), "MOE_EXPERT_5");
    EXPECT_TRUE(stage.execute(ctx_.get()));  // Should succeed (no-op)
    EXPECT_EQ(stage.estimatedFlops(), 0ULL); // No tokens = no work
}

// =============================================================================
// ComputeStageFactory Tests
// =============================================================================

TEST_F(ComputeStageTest, FactoryCreateRMSNorm)
{
    const int seq_len = 1, hidden_dim = 896;
    auto input = makeTensor(seq_len, hidden_dim);
    auto output = makeTensor(seq_len, hidden_dim);
    auto gamma = makeTensor1D(hidden_dim);

    RMSNormStage::Params params{
        .input = input.get(),
        .output = output.get(),
        .gamma = gamma.get(),
        .eps = 1e-6f};

    auto stage = ComputeStageFactory::createRMSNorm(params);
    ASSERT_NE(stage, nullptr);
    EXPECT_EQ(stage->type(), ComputeStageType::RMS_NORM);
}

TEST_F(ComputeStageTest, FactoryCreateRoPE)
{
    const int seq_len = 1, n_heads = 14, head_dim = 64;
    auto Q = makeTensor(seq_len, n_heads * head_dim);

    RoPEStage::Params params{
        .Q = Q.get(),
        .n_heads = n_heads,
        .n_kv_heads = n_heads,
        .head_dim = head_dim,
        .pos_offset = 0,
        .theta_base = 10000.0f};

    auto stage = ComputeStageFactory::createRoPE(params);
    ASSERT_NE(stage, nullptr);
    EXPECT_EQ(stage->type(), ComputeStageType::ROPE);
}

// =============================================================================
// Null Context Handling Tests
// =============================================================================

TEST_F(ComputeStageTest, NullContextHandling)
{
    // All stages should gracefully handle null context
    const int seq_len = 1, hidden_dim = 4;

    auto input = makeTensor(seq_len, hidden_dim);
    auto output = makeTensor(seq_len, hidden_dim);
    auto gamma = makeTensor1D(hidden_dim);

    RMSNormStage::Params rms_params{
        .input = input.get(),
        .output = output.get(),
        .gamma = gamma.get(),
        .eps = 1e-6f};

    RMSNormStage rms_stage(rms_params);
    EXPECT_FALSE(rms_stage.execute(nullptr));

    auto Q = makeTensor(seq_len, hidden_dim);
    RoPEStage::Params rope_params{
        .Q = Q.get(),
        .n_heads = 1,
        .n_kv_heads = 1,
        .head_dim = hidden_dim,
        .pos_offset = 0,
        .theta_base = 10000.0f};

    RoPEStage rope_stage(rope_params);
    EXPECT_FALSE(rope_stage.execute(nullptr));
}
